"""
ML Parameter Adjuster - Central interface for ML-based parameter optimization.

This is the main module that coordinates ML optimization of strategy parameters.
It provides:
1. Unified interface for all optimization methods
2. Comparison between static human params and ML-adjusted params
3. Condition-aware parameter adjustment
4. Analysis of WHEN ML helps vs WHEN it fails

CRITICAL PRINCIPLE:
- ML adjusts ONLY parameters
- ML NEVER changes strategy logic
- Human defines the trading rules; ML tunes them
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json

from optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    ParameterSpace,
    ParameterType
)
from optimization.bayesian_optimizer import BayesianOptimizer
from optimization.random_search import RandomSearchOptimizer, GridSearchOptimizer
from optimization.evolutionary_optimizer import EvolutionaryOptimizer, DifferentialEvolutionOptimizer
from optimization.multi_objective_optimizer import MultiObjectiveOptimizer
from optimization.simulated_annealing import SimulatedAnnealingOptimizer
from optimization.additional_optimizers import (
    CMAESOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategiesOptimizer,
    HyperbandASHAOptimizer,
)
from optimization.optimizer_registry import (
    OptimizerSpec,
    get_optimizer_registry,
)

logger = logging.getLogger(__name__)

# Explainability
from analysis.explainability import ParameterExplainer, format_report_as_json


class OptimizationMethod(Enum):
    """Available optimization methods."""
    BAYESIAN = "bayesian"
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"
    GRID_SEARCH = "grid_search"


@dataclass
class ParameterAdjustmentResult:
    """
    Result of ML parameter adjustment.
    
    Contains both the optimized parameters and comprehensive
    comparison information to analyze when ML helps.
    """
    # Strategy info
    strategy_name: str
    
    # Human baseline
    human_params: Dict[str, Any]
    human_objective: float
    human_metrics: Dict[str, float] = field(default_factory=dict)
    
    # ML optimized
    ml_params: Dict[str, Any] = field(default_factory=dict)
    ml_objective: float = 0.0
    ml_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Comparison
    improvement_pct: float = 0.0
    improvement_significant: bool = False
    significance_pvalue: Optional[float] = None
    
    # Optimization details
    optimization_method: str = ""
    optimization_time_seconds: float = 0.0
    n_iterations: int = 0
    
    # Which parameters changed most
    parameter_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Condition analysis
    market_condition: str = ""
    ml_helped: bool = False
    failure_reason: Optional[str] = None
    # Explainability report JSON (generated after optimization)
    explainability_report_json: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.human_objective != 0:
            self.improvement_pct = (
                (self.ml_objective - self.human_objective)
                / abs(self.human_objective) * 100
            )
        
        # Calculate parameter changes
        if self.ml_params and self.human_params:
            self.parameter_changes = {}
            for key in self.human_params:
                if key in self.ml_params:
                    human_val = self.human_params[key]
                    ml_val = self.ml_params[key]
                    
                    if isinstance(human_val, (int, float)) and human_val != 0:
                        change_pct = (ml_val - human_val) / abs(human_val) * 100
                    else:
                        change_pct = 0 if human_val == ml_val else 100
                    
                    self.parameter_changes[key] = {
                        'human': human_val,
                        'ml': ml_val,
                        'change_pct': change_pct
                    }
        
        # Determine if ML helped
        self.ml_helped = self.improvement_pct > 1.0  # More than 1% improvement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'human_params': self.human_params,
            'human_objective': self.human_objective,
            'human_metrics': self.human_metrics,
            'ml_params': self.ml_params,
            'ml_objective': self.ml_objective,
            'ml_metrics': self.ml_metrics,
            'improvement_pct': self.improvement_pct,
            'improvement_significant': self.improvement_significant,
            'optimization_method': self.optimization_method,
            'optimization_time_seconds': self.optimization_time_seconds,
            'n_iterations': self.n_iterations,
            'parameter_changes': self.parameter_changes,
            'market_condition': self.market_condition,
            'ml_helped': self.ml_helped,
            'failure_reason': self.failure_reason
            ,
            'explainability_report_json': self.explainability_report_json
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Parameter Adjustment Result: {self.strategy_name} ===",
            f"Optimization Method: {self.optimization_method}",
            f"Market Condition: {self.market_condition}",
            "",
            "COMPARISON:",
            f"  Human baseline: {self.human_objective:.4f}",
            f"  ML optimized:   {self.ml_objective:.4f}",
            f"  Improvement:    {self.improvement_pct:+.2f}%",
            f"  ML Helped:      {'YES' if self.ml_helped else 'NO'}",
            "",
            "PARAMETER CHANGES:",
        ]
        
        # Sort by absolute change percentage (biggest changes first)
        sorted_changes = sorted(
            self.parameter_changes.items(),
            key=lambda x: abs(x[1]['change_pct']),
            reverse=True
        )
        
        for param, change in sorted_changes:
            human_val = change['human']
            ml_val = change['ml']
            change_pct = change['change_pct']
            
            # Generate reasoning for each parameter change
            reason = self._get_change_reason(param, human_val, ml_val, change_pct)
            
            lines.append(
                f"  {param}: {human_val} -> {ml_val} "
                f"({change_pct:+.1f}%)"
            )
            if reason:
                lines.append(f"    └─ REASON: {reason}")
        
        if self.failure_reason:
            lines.append(f"\nFAILURE REASON: {self.failure_reason}")
        
        # Add summary insight
        lines.append("")
        lines.append("KEY INSIGHTS:")
        lines.extend(self._generate_insights())
        
        return "\n".join(lines)
    
    def _get_change_reason(self, param: str, human_val: Any, ml_val: Any, change_pct: float) -> str:
        """Generate reasoning for a specific parameter change."""
        
        # RSI parameters
        if param == 'rsi_lookback':
            if ml_val > human_val:
                return "Longer lookback reduces noise and false signals in volatile market"
            else:
                return "Shorter lookback captures faster momentum shifts"
        
        elif param == 'rsi_buy_threshold':
            if ml_val < human_val:
                return "Lower threshold waits for deeper oversold conditions (more conservative entry)"
            else:
                return "Higher threshold enters earlier to catch more reversals"
        
        elif param == 'rsi_sell_threshold':
            if ml_val > human_val:
                return "Higher threshold waits for stronger overbought signals"
            else:
                return "Lower threshold exits earlier to lock in profits"
        
        # Risk management
        elif param == 'stop_loss_pct':
            if ml_val < human_val:
                return "Tighter stop loss reduces downside risk per trade"
            else:
                return "Wider stop loss avoids premature exits from noise"
        
        elif param == 'take_profit_pct':
            if ml_val > human_val:
                return "Higher take profit targets larger moves (better risk/reward)"
            else:
                return "Lower take profit secures gains faster"
        
        elif param == 'trailing_stop_pct':
            if ml_val < human_val:
                return "Tighter trailing stop locks in profits during reversals"
            else:
                return "Wider trailing stop allows trends to develop"
        
        # Position sizing
        elif param == 'position_size_pct':
            if ml_val > human_val:
                return "Larger position size increases exposure when strategy is confident"
            else:
                return "Smaller position size reduces risk during uncertainty"
        
        # Timing
        elif param == 'entry_confirmation':
            if ml_val > human_val:
                return "More confirmation candles filter out false breakouts"
            else:
                return "Fewer confirmation for faster entry on clear signals"
        
        elif param == 'cooldown_period':
            if ml_val > human_val:
                return "Longer cooldown prevents overtrading in choppy conditions"
            else:
                return "Shorter cooldown captures more opportunities"
        
        elif param == 'max_holding_time':
            if ml_val > human_val:
                return "Longer holding time lets winning trades run further"
            else:
                return "Shorter holding avoids extended drawdowns"
        
        # ADX/Trend filters
        elif param == 'adx_threshold':
            if ml_val > human_val:
                return "Higher ADX threshold ensures stronger trend confirmation"
            else:
                return "Lower ADX threshold allows trades in weaker trends"
        
        elif param == 'min_rsi_slope':
            if ml_val > human_val:
                return "Positive slope requirement filters for momentum direction"
            else:
                return "Negative slope allows contrarian entries on pullbacks"
        
        # EMA parameters
        elif 'ema_fast' in param or 'fast_period' in param:
            if ml_val < human_val:
                return "Shorter fast EMA responds quicker to price changes"
            else:
                return "Longer fast EMA smooths out noise"
        
        elif 'ema_slow' in param or 'slow_period' in param:
            if ml_val > human_val:
                return "Longer slow EMA provides stronger trend baseline"
            else:
                return "Shorter slow EMA adapts faster to trend changes"
        
        # Bollinger Band parameters
        elif 'bb_period' in param:
            if ml_val > human_val:
                return "Longer BB period creates more stable bands"
            else:
                return "Shorter BB period reacts faster to volatility"
        
        elif 'bb_std' in param:
            if ml_val > human_val:
                return "Wider bands reduce false breakout signals"
            else:
                return "Narrower bands identify earlier breakout opportunities"
        
        # Generic fallback
        if abs(change_pct) < 5:
            return "Minor tuning within similar range (marginal optimization)"
        elif abs(change_pct) > 50:
            return "Major adjustment indicates original value was suboptimal for this data"
        else:
            return "Moderate adjustment based on backtest performance"
    
    def _generate_insights(self) -> List[str]:
        """Generate key insights from parameter changes."""
        insights = []
        
        if not self.parameter_changes:
            return ["  No parameter changes to analyze"]
        
        # Find biggest changes
        biggest_change = max(
            self.parameter_changes.items(),
            key=lambda x: abs(x[1]['change_pct'])
        )
        
        insights.append(
            f"  • Largest change: {biggest_change[0]} "
            f"({biggest_change[1]['change_pct']:+.1f}%)"
        )
        
        # Risk management changes
        risk_params = ['stop_loss_pct', 'take_profit_pct', 'position_size_pct']
        risk_changes = [
            (p, self.parameter_changes[p]['change_pct'])
            for p in risk_params if p in self.parameter_changes
        ]
        
        if risk_changes:
            total_risk_change = sum(abs(c[1]) for c in risk_changes)
            if total_risk_change > 30:
                insights.append(
                    "  • ML significantly adjusted risk parameters "
                    "(strategy may have been too aggressive/conservative)"
                )
        
        # Improvement summary
        if self.ml_helped:
            insights.append(
                f"  • ML optimization IMPROVED performance by {self.improvement_pct:.1f}%"
            )
        else:
            insights.append(
                f"  • ML optimization did NOT significantly improve "
                f"(human params already near-optimal)"
            )
        
        return insights


class MLParameterAdjuster:
    """
    Central interface for ML-based parameter adjustment.
    
    This class coordinates the optimization of trading strategy parameters
    using various ML methods, and provides comprehensive analysis of
    when ML optimization helps vs when it fails.
    
    Key responsibilities:
    1. Run optimization with different methods
    2. Compare ML-adjusted params against human baseline
    3. Analyze which market conditions benefit from ML
    4. Track and report optimization history
    
    Usage:
        adjuster = MLParameterAdjuster(strategy_engine, backtest_engine)
        
        # Optimize with comparison to baseline
        result = adjuster.optimize_strategy(
            strategy_name='rsi_mean_reversion',
            train_data=df_train,
            method=OptimizationMethod.BAYESIAN,
            n_iterations=100
        )
        
        # Analyze when ML helps
        analysis = adjuster.analyze_ml_effectiveness()
    """
    
    def __init__(
        self,
        objective_function: Callable[[str, Dict[str, Any], pd.DataFrame], float],
        strategy_bounds: Dict[str, Dict[str, Tuple[float, float]]],
        integer_params: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Initialize the ML Parameter Adjuster.
        
        Args:
            objective_function: Function(strategy_name, params, data) -> objective_value
            strategy_bounds: Dict mapping strategy names to parameter bounds
            integer_params: List of parameter names that should be integers
            verbose: Logging verbosity
        """
        self.objective_function = objective_function
        self.strategy_bounds = strategy_bounds
        self.verbose = verbose

        # Registry of available optimizers (15 declared, some planned)
        self.optimizer_specs: List[OptimizerSpec] = get_optimizer_registry()
        self.optimizer_specs_by_key: Dict[str, OptimizerSpec] = {
            spec.key: spec for spec in self.optimizer_specs
        }
        self.implemented_optimizer_keys: List[str] = [
            spec.key for spec in self.optimizer_specs if spec.status == "implemented"
        ]
        
        # Default integer params
        self.integer_params = integer_params or [
            'rsi_lookback', 'ema_fast_period', 'ema_slow_period',
            'bb_period', 'squeeze_lookback', 'min_squeeze_candles',
            'adx_period', 'atr_period', 'macd_fast', 'macd_slow', 'macd_signal'
        ]
        
        # History of adjustments
        self.adjustment_history: List[ParameterAdjustmentResult] = []
        
        # Method comparison data
        self.method_comparison: Dict[str, List[OptimizationResult]] = {
            key: [] for key in self.implemented_optimizer_keys
        }

    def _resolve_method(self, method: Union[OptimizationMethod, str]) -> Tuple[str, OptimizerSpec]:
        """Resolve method input (enum or string) to registry key and spec."""
        # Legacy enum mapping
        legacy_map = {
            OptimizationMethod.BAYESIAN.value: "bayesian_gp",
            OptimizationMethod.RANDOM_SEARCH.value: "random_search",
            OptimizationMethod.EVOLUTIONARY.value: "genetic_algorithm",
            OptimizationMethod.GRID_SEARCH.value: "grid_search",
        }

        if isinstance(method, OptimizationMethod):
            method = method.value

        method_str = str(method).lower()
        method_key = legacy_map.get(method_str, method_str)

        if method_key not in self.optimizer_specs_by_key:
            raise ValueError(
                f"Unknown optimization method '{method}'. "
                f"Available: {list(self.optimizer_specs_by_key.keys())}"
            )

        spec = self.optimizer_specs_by_key[method_key]
        if spec.status != "implemented":
            raise ValueError(
                f"Optimization method '{spec.name}' ({method_key}) is planned but not implemented yet."
            )
        if spec.cls is None:
            raise ValueError(
                f"No optimizer class wired for '{spec.name}' ({method_key})."
            )
        return method_key, spec
    
    def optimize_strategy(
        self,
        strategy_name: str,
        train_data: pd.DataFrame,
        method: Union[OptimizationMethod, str] = OptimizationMethod.BAYESIAN,
        human_params: Optional[Dict[str, Any]] = None,
        n_iterations: int = 100,
        random_state: Optional[int] = None,
        market_condition: str = "unknown",
        **optimizer_kwargs
    ) -> ParameterAdjustmentResult:
        """
        Optimize parameters for a strategy and compare to human baseline.
        
        Args:
            strategy_name: Name of the strategy to optimize
            train_data: Training data for optimization
            method: Optimization method to use
            human_params: Optional human baseline parameters
            n_iterations: Number of optimization iterations
            random_state: Random seed for reproducibility
            market_condition: Label for current market condition
            **optimizer_kwargs: Additional arguments for the optimizer
            
        Returns:
            ParameterAdjustmentResult with comparison data
        """
        method_key, method_spec = self._resolve_method(method)

        logger.info(
            f"Starting parameter optimization for {strategy_name} "
            f"using {method_spec.name} ({method_key})"
        )
        
        # Get parameter bounds for this strategy
        if strategy_name not in self.strategy_bounds:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        bounds = self.strategy_bounds[strategy_name]
        
        # Create parameter space
        space = ParameterSpace.from_strategy_bounds(bounds, self.integer_params)
        
        # Create objective function wrapper
        def objective_wrapper(params: Dict[str, Any]) -> float:
            return self.objective_function(strategy_name, params, train_data)
        
        # Evaluate human baseline if provided
        if human_params is not None:
            human_objective = objective_wrapper(human_params)
        else:
            human_params = space.get_defaults()
            human_objective = objective_wrapper(human_params) if human_params else 0.0
        
        # Create optimizer
        optimizer = self._create_optimizer(
            method_key=method_key,
            method_spec=method_spec,
            parameter_space=space,
            objective_function=objective_wrapper,
            n_iterations=n_iterations,
            random_state=random_state,
            **optimizer_kwargs
        )
        
        # Run optimization
        opt_result = optimizer.optimize(baseline_objective=human_objective)
        
        # Store method comparison data
        if method_key in self.method_comparison:
            self.method_comparison[method_key].append(opt_result)
        else:
            self.method_comparison[method_key] = [opt_result]
        
        # Create adjustment result
        result = ParameterAdjustmentResult(
            strategy_name=strategy_name,
            human_params=human_params,
            human_objective=human_objective,
            ml_params=opt_result.best_parameters,
            ml_objective=opt_result.best_objective,
            optimization_method=method_key,
            optimization_time_seconds=opt_result.total_time_seconds,
            n_iterations=opt_result.n_iterations,
            market_condition=market_condition
        )
        
        # Store in history
        self.adjustment_history.append(result)
        
        if self.verbose:
            logger.info(result.summary())

        # Generate explainability report for this optimization and attach JSON
        try:
            explainer = ParameterExplainer()
            explainer.log_optimization_result(
                human_params=human_params,
                ml_params=opt_result.best_parameters,
                human_objective=human_objective,
                ml_objective=opt_result.best_objective,
                optimization_method=method_key,
                market_condition=market_condition
            )
            report = explainer.generate_report(strategy_name)
            report_json = format_report_as_json(report)
            result.explainability_report_json = report_json
            logger.info(f"Explainability report generated for {strategy_name}")
        except Exception as e:
            logger.warning(f"Failed to generate explainability report: {e}")
        
        return result
    
    def compare_methods(
        self,
        strategy_name: str,
        train_data: pd.DataFrame,
        methods: Optional[List[Union[OptimizationMethod, str]]] = None,
        n_iterations: int = 50,
        n_repeats: int = 3,
        random_state: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different optimization methods on the same problem.
        
        This is crucial for understanding WHEN different ML methods help.
        
        Args:
            strategy_name: Strategy to optimize
            train_data: Training data
            methods: Methods to compare (default: all)
            n_iterations: Iterations per method
            n_repeats: Number of repeats for statistics
            random_state: Base random seed
            
        Returns:
            Dictionary with comparison statistics
        """
        if methods is None:
            methods = list(self.implemented_optimizer_keys)
        
        comparison = {}
        
        for method in methods:
            method_key, method_spec = self._resolve_method(method)
            method_results = []
            
            for repeat in range(n_repeats):
                seed = random_state + repeat if random_state else None
                
                result = self.optimize_strategy(
                    strategy_name=strategy_name,
                    train_data=train_data,
                    method=method_key,
                    n_iterations=n_iterations,
                    random_state=seed,
                    market_condition=f"comparison_repeat_{repeat}"
                )
                
                method_results.append(result)
            
            # Compute statistics
            objectives = [r.ml_objective for r in method_results]
            improvements = [r.improvement_pct for r in method_results]
            times = [r.optimization_time_seconds for r in method_results]
            
            comparison[method_key] = {
                'mean_objective': np.mean(objectives),
                'std_objective': np.std(objectives),
                'mean_improvement_pct': np.mean(improvements),
                'std_improvement_pct': np.std(improvements),
                'mean_time_seconds': np.mean(times),
                'ml_helped_count': sum(r.ml_helped for r in method_results),
                'ml_helped_rate': sum(r.ml_helped for r in method_results) / n_repeats
            }
        
        # Log comparison
        logger.info("=== Method Comparison Results ===")
        for method_name, stats in comparison.items():
            logger.info(
                f"{method_name}: "
                f"objective={stats['mean_objective']:.4f}±{stats['std_objective']:.4f}, "
                f"improvement={stats['mean_improvement_pct']:.2f}%, "
                f"ML helped {stats['ml_helped_rate']*100:.0f}% of time"
            )
        
        return comparison
    
    def _create_optimizer(
        self,
        method_key: str,
        method_spec: OptimizerSpec,
        parameter_space: ParameterSpace,
        objective_function: Callable,
        n_iterations: int,
        random_state: Optional[int],
        **kwargs
    ) -> BaseOptimizer:
        """Create optimizer instance based on registry spec."""
        common_args = {
            'parameter_space': parameter_space,
            'objective_function': objective_function,
            'maximize': True,  # Assume we're maximizing (e.g., Sharpe ratio)
            'n_iterations': n_iterations,
            'random_state': random_state,
            'verbose': self.verbose
        }

        # Map registry keys to concrete classes/backends
        if method_key in {"bayesian_gp", "bayesian_tpe"}:
            backend = 'optuna' if method_key == "bayesian_tpe" else kwargs.pop('backend', 'skopt')
            return BayesianOptimizer(**common_args, backend=backend, **kwargs)

        if method_key in {"random_search", "latin_hypercube", "sobol"}:
            return RandomSearchOptimizer(**common_args, **kwargs)

        if method_key == "grid_search":
            grid_args = dict(common_args)
            grid_args.pop('n_iterations', None)
            grid_args.pop('random_state', None)
            return GridSearchOptimizer(**grid_args, **kwargs)

        if method_key in {"genetic_algorithm"}:
            return EvolutionaryOptimizer(**common_args, **kwargs)

        if method_key == "differential_evolution":
            return DifferentialEvolutionOptimizer(**common_args, **kwargs)

        if method_key == "simulated_annealing":
            return SimulatedAnnealingOptimizer(**common_args, **kwargs)

        if method_key == "particle_swarm":
            return ParticleSwarmOptimizer(**common_args, **kwargs)

        if method_key == "evolution_strategies":
            return EvolutionStrategiesOptimizer(**common_args, **kwargs)

        if method_key == "cma_es":
            return CMAESOptimizer(**common_args, **kwargs)

        if method_key == "hyperband_asha":
            # Hyperband/ASHA expects objective_function(params, resource).
            # Our base objective_function wrapper is objective_function(params) -> float.
            base_objective = common_args['objective_function']

            def objective_with_resource(params: Dict[str, Any], _resource: int) -> float:
                return base_objective(params)

            hb_args = dict(common_args)
            hb_args['objective_function'] = objective_with_resource
            return HyperbandASHAOptimizer(**hb_args, **kwargs)

        if method_key in {"nsga_ii", "nsga_iii"}:
            return MultiObjectiveOptimizer(**common_args, **kwargs)

        # Planned but not yet implemented (guard should prevent reaching here)
        raise ValueError(
            f"Optimizer '{method_spec.name}' ({method_key}) is not wired yet."
        )
    
    def analyze_ml_effectiveness(
        self,
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze when ML optimization helps vs fails.
        
        This is the core research question: WHEN does ML help,
        HOW does it help, and WHEN does it fail?
        
        Args:
            min_samples: Minimum samples needed for analysis
            
        Returns:
            Analysis dictionary with insights
        """
        if len(self.adjustment_history) < min_samples:
            logger.warning(
                f"Not enough samples ({len(self.adjustment_history)}) "
                f"for meaningful analysis. Need at least {min_samples}."
            )
            return {}
        
        # Overall statistics
        improvements = [r.improvement_pct for r in self.adjustment_history]
        helped_count = sum(r.ml_helped for r in self.adjustment_history)
        
        analysis = {
            'overall': {
                'total_adjustments': len(self.adjustment_history),
                'ml_helped_count': helped_count,
                'ml_helped_rate': helped_count / len(self.adjustment_history),
                'mean_improvement_pct': np.mean(improvements),
                'median_improvement_pct': np.median(improvements),
                'std_improvement_pct': np.std(improvements),
                'max_improvement_pct': np.max(improvements),
                'min_improvement_pct': np.min(improvements)
            }
        }
        
        # By strategy
        by_strategy = {}
        for result in self.adjustment_history:
            if result.strategy_name not in by_strategy:
                by_strategy[result.strategy_name] = []
            by_strategy[result.strategy_name].append(result)
        
        analysis['by_strategy'] = {}
        for strategy, results in by_strategy.items():
            improvements = [r.improvement_pct for r in results]
            helped = sum(r.ml_helped for r in results)
            
            analysis['by_strategy'][strategy] = {
                'n_samples': len(results),
                'ml_helped_rate': helped / len(results),
                'mean_improvement_pct': np.mean(improvements)
            }
        
        # By market condition
        by_condition = {}
        for result in self.adjustment_history:
            if result.market_condition not in by_condition:
                by_condition[result.market_condition] = []
            by_condition[result.market_condition].append(result)
        
        analysis['by_market_condition'] = {}
        for condition, results in by_condition.items():
            improvements = [r.improvement_pct for r in results]
            helped = sum(r.ml_helped for r in results)
            
            analysis['by_market_condition'][condition] = {
                'n_samples': len(results),
                'ml_helped_rate': helped / len(results),
                'mean_improvement_pct': np.mean(improvements)
            }
        
        # By optimization method
        analysis['by_method'] = {}
        by_method = {}
        for result in self.adjustment_history:
            if result.optimization_method not in by_method:
                by_method[result.optimization_method] = []
            by_method[result.optimization_method].append(result)
        
        for method, results in by_method.items():
            improvements = [r.improvement_pct for r in results]
            helped = sum(r.ml_helped for r in results)
            times = [r.optimization_time_seconds for r in results]
            
            analysis['by_method'][method] = {
                'n_samples': len(results),
                'ml_helped_rate': helped / len(results),
                'mean_improvement_pct': np.mean(improvements),
                'mean_time_seconds': np.mean(times),
                'efficiency': np.mean(improvements) / np.mean(times) if np.mean(times) > 0 else 0
            }
        
        # Parameter sensitivity analysis
        analysis['parameter_sensitivity'] = self._analyze_parameter_sensitivity()
        
        return analysis
    
    def _analyze_parameter_sensitivity(self) -> Dict[str, float]:
        """
        Analyze which parameters changed most during optimization.
        
        Returns sensitivity scores for each parameter.
        """
        param_changes = {}
        
        for result in self.adjustment_history:
            for param, change in result.parameter_changes.items():
                if param not in param_changes:
                    param_changes[param] = []
                param_changes[param].append(abs(change['change_pct']))
        
        sensitivity = {}
        for param, changes in param_changes.items():
            sensitivity[param] = {
                'mean_change_pct': np.mean(changes),
                'std_change_pct': np.std(changes),
                'max_change_pct': np.max(changes)
            }
        
        return sensitivity
    
    def get_recommendations(self) -> List[str]:
        """
        Generate recommendations based on optimization history.
        
        Returns list of actionable insights.
        """
        analysis = self.analyze_ml_effectiveness()
        
        if not analysis:
            return ["Insufficient data for recommendations. Run more optimizations."]
        
        recommendations = []
        
        # Overall effectiveness
        overall = analysis.get('overall', {})
        if overall.get('ml_helped_rate', 0) < 0.3:
            recommendations.append(
                "ML optimization helped in less than 30% of cases. "
                "Consider: (1) checking if human params are already near-optimal, "
                "(2) increasing optimization iterations, or "
                "(3) the objective function may be noisy."
            )
        
        # Method recommendations
        by_method = analysis.get('by_method', {})
        if by_method:
            best_method = max(
                by_method.items(),
                key=lambda x: x[1].get('mean_improvement_pct', 0)
            )
            recommendations.append(
                f"Best performing method: {best_method[0]} with "
                f"{best_method[1]['mean_improvement_pct']:.2f}% mean improvement."
            )
        
        # Strategy recommendations
        by_strategy = analysis.get('by_strategy', {})
        for strategy, stats in by_strategy.items():
            if stats.get('ml_helped_rate', 0) < 0.2:
                recommendations.append(
                    f"Strategy '{strategy}' rarely benefits from ML optimization. "
                    "Human parameters may already be well-tuned for this strategy."
                )
        
        # Condition recommendations
        by_condition = analysis.get('by_market_condition', {})
        best_conditions = [
            (cond, stats) for cond, stats in by_condition.items()
            if stats.get('ml_helped_rate', 0) > 0.5
        ]
        if best_conditions:
            recommendations.append(
                f"ML optimization works best in: "
                f"{', '.join(c[0] for c in best_conditions)}"
            )
        
        return recommendations
    
    def export_history(self, filepath: str) -> None:
        """Export adjustment history to JSON."""
        data = {
            'adjustments': [r.to_dict() for r in self.adjustment_history],
            'analysis': self.analyze_ml_effectiveness(),
            'recommendations': self.get_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported optimization history to {filepath}")
    
    def reset_history(self) -> None:
        """Reset optimization history."""
        self.adjustment_history = []
        self.method_comparison = {
            method.value: [] for method in OptimizationMethod
        }
        logger.info("Reset optimization history")
