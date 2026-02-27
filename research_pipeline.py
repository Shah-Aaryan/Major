"""
Research Pipeline.

Orchestrates the complete ML parameter optimization research workflow:
1. Load and preprocess data
2. Generate features
3. Run backtests with human parameters
4. Optimize parameters with different ML methods
5. Run backtests with ML parameters
6. Analyze conditions and failures
7. Generate comparison reports
8. Audit logging throughout

This is the main research engine that answers:
- WHEN does ML help?
- HOW does ML help?
- WHEN does ML fail?
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import pandas as pd
import numpy as np

# Data modules
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.resampler import DataResampler

# Feature modules
from features.feature_engine import FeatureEngine

# Strategy modules
from strategies.strategy_engine import StrategyEngine
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.bollinger_breakout import BollingerBreakoutStrategy

# Optimization modules
from optimization.ml_parameter_adjuster import MLParameterAdjuster
from optimization.bayesian_optimizer import BayesianOptimizer
from optimization.random_search import RandomSearchOptimizer
from optimization.evolutionary_optimizer import EvolutionaryOptimizer

# Backtesting modules
from backtesting.backtest_engine import BacktestEngine
from backtesting.walk_forward import WalkForwardValidator

# Analysis modules
from analysis.condition_analyzer import ConditionAnalyzer
from analysis.failure_detector import FailureDetector
from analysis.comparison_report import ComparisonReport, generate_full_report

# Audit modules
from audit.audit_logger import AuditLogger, AuditEventType, OptimizationAudit

# Config
from config.settings import (
    DataConfig, FeatureConfig, StrategyConfig, OptimizationConfig,
    BacktestConfig, ResearchConfig, AuditConfig
)

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Results from a single research experiment."""
    strategy_name: str
    symbol: str
    timeframe: str
    
    # Performance metrics
    human_metrics: Dict[str, float]
    ml_metrics: Dict[str, float]
    improvement: Dict[str, float]
    
    # Parameters
    human_params: Dict[str, Any]
    ml_params: Dict[str, Any]
    
    # Analysis
    best_optimizer: str
    failures_detected: int
    ml_recommended: bool
    confidence: float
    
    # Timing
    optimization_time: float
    total_time: float
    
    # Reports
    report_path: Optional[str] = None


@dataclass
class ResearchSession:
    """Complete research session results."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Results per strategy
    results: List[ResearchResult] = field(default_factory=list)
    
    # Aggregated findings
    overall_findings: Dict[str, Any] = field(default_factory=dict)
    
    # Paths
    output_dir: str = ""
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        if not self.results:
            return {'status': 'no results'}
        
        ml_wins = sum(1 for r in self.results if r.ml_recommended)
        avg_improvement = np.mean([
            r.improvement.get('sharpe_ratio', 0) for r in self.results
        ])
        
        return {
            'session_id': self.session_id,
            'n_experiments': len(self.results),
            'ml_recommended_count': ml_wins,
            'ml_recommended_pct': ml_wins / len(self.results),
            'avg_sharpe_improvement': avg_improvement,
            'strategies_tested': list(set(r.strategy_name for r in self.results)),
            'total_optimization_time': sum(r.optimization_time for r in self.results)
        }


class ResearchPipeline:
    """
    Main research pipeline for ML parameter optimization studies.
    
    Orchestrates all modules to run complete experiments comparing
    human-defined parameters vs ML-optimized parameters.
    """
    
    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        strategy_config: Optional[StrategyConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
        research_config: Optional[ResearchConfig] = None,
        audit_config: Optional[AuditConfig] = None,
        output_dir: str = "./research_output"
    ):
        """
        Initialize research pipeline.
        
        Args:
            data_config: Data loading configuration
            feature_config: Feature engineering configuration
            strategy_config: Strategy configuration
            optimization_config: Optimization configuration
            backtest_config: Backtesting configuration
            research_config: Research experiment configuration
            audit_config: Audit logging configuration
            output_dir: Directory for output files
        """
        # Configurations
        self.data_config = data_config or DataConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.strategy_config = strategy_config or StrategyConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        self.research_config = research_config or ResearchConfig()
        self.audit_config = audit_config or AuditConfig()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modules
        self.data_loader = DataLoader(self.data_config.data_dir)
        self.preprocessor = DataPreprocessor()
        self.resampler = DataResampler()
        self.feature_engine = FeatureEngine(self.feature_config)
        
        # Audit logger
        self.audit_logger = AuditLogger(
            output_dir=str(self.output_dir / "audit"),
            log_to_console=self.audit_config.log_to_console
        )
        
        # Session tracking
        self.current_session: Optional[ResearchSession] = None
        
        logger.info("Research pipeline initialized")
    
    def run_full_research(
        self,
        data_paths: List[str],
        symbols: Optional[List[str]] = None,
        timeframes: List[str] = ['1m', '5m', '15m'],
        strategies: Optional[List[str]] = None
    ) -> ResearchSession:
        """
        Run complete research pipeline.
        
        Args:
            data_paths: Paths to OHLCV data files
            symbols: Symbols to analyze (auto-detect if None)
            timeframes: Timeframes to test
            strategies: Strategy names to test (all if None)
            
        Returns:
            ResearchSession with all results
        """
        start_time = datetime.now()
        
        # Create session
        self.current_session = ResearchSession(
            session_id=self.audit_logger.session_id,
            start_time=start_time,
            output_dir=str(self.output_dir)
        )
        
        self.audit_logger.log_event(
            AuditEventType.SESSION_START,
            {
                'data_paths': data_paths,
                'timeframes': timeframes,
                'strategies': strategies or ['all']
            },
            explanation="Starting full research pipeline"
        )
        
        # Default strategies
        if not strategies:
            strategies = ['rsi_mean_reversion', 'ema_crossover', 'bollinger_breakout']
        
        # Load data
        logger.info("Loading data...")
        data = self._load_and_prepare_data(data_paths, symbols)
        
        # Run experiments for each combination
        for symbol, ohlcv in data.items():
            for timeframe in timeframes:
                # Resample if needed
                if timeframe != '1m':
                    resampled = self.resampler.resample(ohlcv, timeframe)
                else:
                    resampled = ohlcv
                
                for strategy_name in strategies:
                    try:
                        result = self._run_single_experiment(
                            symbol=symbol,
                            timeframe=timeframe,
                            ohlcv=resampled,
                            strategy_name=strategy_name
                        )
                        self.current_session.results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Experiment failed: {symbol}/{timeframe}/{strategy_name}: {e}")
                        self.audit_logger.log_error(e, {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'strategy': strategy_name
                        })
        
        # Generate overall findings
        self.current_session.overall_findings = self._generate_overall_findings()
        self.current_session.end_time = datetime.now()
        
        # Save session
        self._save_session()
        
        # Close audit logger
        self.audit_logger.close()
        
        logger.info(f"Research completed. Results in {self.output_dir}")
        
        return self.current_session
    
    def _load_and_prepare_data(
        self,
        data_paths: List[str],
        symbols: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load and preprocess all data files."""
        data = {}
        
        for path in data_paths:
            try:
                # Load OHLCV data
                df = self.data_loader.load_csv(path)
                
                # Extract symbol from filename or path
                symbol = Path(path).stem.split('_')[0].upper()
                
                # Preprocess
                df, quality_report = self.preprocessor.preprocess(df, symbol=symbol)
                
                if symbols is None or symbol in symbols:
                    data[symbol] = df
                    logger.info(f"Loaded {len(df)} rows for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        
        return data
    
    def _run_single_experiment(
        self,
        symbol: str,
        timeframe: str,
        ohlcv: pd.DataFrame,
        strategy_name: str
    ) -> ResearchResult:
        """Run a single research experiment."""
        logger.info(f"Running experiment: {symbol}/{timeframe}/{strategy_name}")
        start_time = time.time()
        
        # Create strategy
        strategy = self._create_strategy(strategy_name)
        
        # Generate features (don't drop NaN rows - let backtest handle missing data)
        # Note: generate_features already includes original OHLCV columns, so no concat needed
        data_with_features = self.feature_engine.generate_features(ohlcv, drop_na=False)
        
        # Forward fill remaining NaN values and drop initial rows with NaN
        data_with_features = data_with_features.ffill().bfill()
        # Drop rows that still have critical NaN (first few rows might have)
        data_with_features = data_with_features.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        logger.info(f"Data after feature generation: {len(data_with_features)} rows")
        
        if len(data_with_features) < 100:
            raise ValueError(f"Insufficient data after feature generation: {len(data_with_features)} rows")
        
        # Get human baseline parameters
        human_params = strategy.parameters.to_dict()
        
        # Run backtest with human parameters
        logger.info("Running backtest with human parameters...")
        human_results = self._run_backtest(
            strategy, data_with_features, human_params, "human"
        )
        
        # Run ML optimization
        logger.info("Running ML optimization...")
        opt_start = time.time()
        ml_params, best_optimizer = self._run_optimization(
            strategy, data_with_features, human_params
        )
        opt_time = time.time() - opt_start
        
        # Run backtest with ML parameters
        logger.info("Running backtest with ML parameters...")
        ml_results = self._run_backtest(
            strategy, data_with_features, ml_params, "ml"
        )
        
        # Analyze conditions
        condition_analyzer = self._analyze_conditions(
            data_with_features, human_results, ml_results
        )
        
        # Detect failures
        failure_detector = self._detect_failures(
            human_results, ml_results, human_params, ml_params
        )
        
        # Generate report
        report = generate_full_report(
            strategy_name=strategy_name,
            human_results=human_results,
            ml_results=ml_results,
            condition_analyzer=condition_analyzer,
            failure_detector=failure_detector,
            human_params=human_params,
            ml_params=ml_params,
            data_period=f"{ohlcv.index[0]} to {ohlcv.index[-1]}"
        )
        
        # Save report
        report_path = self._save_report(report, symbol, timeframe, strategy_name)
        
        total_time = time.time() - start_time
        
        # Calculate improvements
        improvements = {}
        for metric in human_results.get('metrics', {}):
            h_val = human_results['metrics'].get(metric, 0)
            m_val = ml_results['metrics'].get(metric, 0)
            if isinstance(h_val, (int, float)) and isinstance(m_val, (int, float)) and h_val != 0:
                improvements[metric] = (m_val - h_val) / abs(h_val)
        
        return ResearchResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            human_metrics=human_results.get('metrics', {}),
            ml_metrics=ml_results.get('metrics', {}),
            improvement=improvements,
            human_params=human_params,
            ml_params=ml_params,
            best_optimizer=best_optimizer,
            failures_detected=len(failure_detector.failures) if failure_detector else 0,
            ml_recommended=report.should_use_ml,
            confidence=report.confidence,
            optimization_time=opt_time,
            total_time=total_time,
            report_path=report_path
        )
    
    def _create_strategy(self, strategy_name: str):
        """Create strategy instance by name."""
        strategies = {
            'rsi_mean_reversion': RSIMeanReversionStrategy,
            'ema_crossover': EMACrossoverStrategy,
            'bollinger_breakout': BollingerBreakoutStrategy
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return strategies[strategy_name]()
    
    def _run_backtest(
        self,
        strategy,
        data: pd.DataFrame,
        params: Dict[str, Any],
        param_source: str
    ) -> Dict[str, Any]:
        """Run backtest with given parameters."""
        # Update strategy parameters
        strategy.update_parameters(params)
        
        # Create backtest config
        from backtesting.backtest_engine import BacktestConfig as BtConfig
        bt_config = BtConfig(
            initial_capital=self.strategy_config.initial_capital,
            commission_pct=self.strategy_config.trading_fee_pct,
            slippage_pct=self.backtest_config.slippage_pct
        )
        
        # Create backtest engine
        backtest = BacktestEngine(config=bt_config)
        
        # Run backtest
        result = backtest.run(strategy, data)
        
        # Convert BacktestResult to dict for compatibility
        results = {
            'metrics': result.metrics.to_dict() if result.metrics else {},
            'trades': [t.to_dict() for t in result.trades] if result.trades else [],
            'equity_curve': result.equity_curve,
            'total_trades': len(result.trades) if result.trades else 0
        }
        
        # Log audit
        self.audit_logger.log_event(
            AuditEventType.PARAMETER_OPTIMIZATION_END if param_source == "ml" else AuditEventType.SESSION_START,
            {
                'strategy': strategy.__class__.__name__,
                'params': params,
                'param_source': param_source,
                'sharpe_ratio': results['metrics'].get('sharpe_ratio', 0),
                'total_return': results['metrics'].get('total_return', 0)
            },
            strategy_name=strategy.__class__.__name__
        )
        
        return results
    
    def _run_optimization(
        self,
        strategy,
        data: pd.DataFrame,
        human_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Run ML optimization and return best parameters."""
        # Get parameter bounds from strategy
        param_bounds = strategy.get_parameter_bounds()
        
        # Create objective function for MLParameterAdjuster
        def objective_func(strategy_name: str, params: Dict[str, Any], train_data: pd.DataFrame) -> float:
            """Evaluate parameters using backtesting."""
            # Update strategy params
            strategy.update_parameters(params)
            
            # Create backtest config
            from backtesting.backtest_engine import BacktestConfig as BtConfig
            bt_config = BtConfig(
                initial_capital=self.strategy_config.initial_capital,
                commission_pct=self.strategy_config.trading_fee_pct,
                slippage_pct=self.backtest_config.slippage_pct
            )
            
            # Run backtest
            backtest = BacktestEngine(config=bt_config)
            result = backtest.run(strategy, train_data)
            
            # Return sharpe ratio as objective
            return result.metrics.sharpe_ratio if result.metrics else 0.0
        
        # Get strategy bounds
        strategy_bounds = {
            strategy.__class__.__name__: param_bounds
        }
        
        # Create ML parameter adjuster
        adjuster = MLParameterAdjuster(
            objective_function=objective_func,
            strategy_bounds=strategy_bounds,
            verbose=True
        )
        
        # Run optimization using Bayesian method
        from optimization.ml_parameter_adjuster import OptimizationMethod
        result = adjuster.optimize_strategy(
            strategy_name=strategy.__class__.__name__,
            train_data=data,
            method=OptimizationMethod.BAYESIAN,
            human_params=human_params,
            n_iterations=self.optimization_config.bayesian_n_calls
        )
        
        # Log optimization audit
        self.audit_logger.log_optimization(OptimizationAudit(
            timestamp=datetime.now(),
            strategy_name=strategy.__class__.__name__,
            optimizer_type=result.optimization_method,
            n_trials=result.n_iterations,
            best_objective=result.ml_objective,
            elapsed_time=result.optimization_time_seconds,
            human_params=human_params,
            ml_params=result.ml_params,
            improvement=result.improvement_pct / 100 if result.improvement_pct else 0
        ))
        
        return result.ml_params or human_params, result.optimization_method or "bayesian"
    
    def _analyze_conditions(
        self,
        data: pd.DataFrame,
        human_results: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Optional[ConditionAnalyzer]:
        """Analyze market conditions and performance."""
        try:
            analyzer = ConditionAnalyzer()
            
            # First identify conditions in the data
            conditions = analyzer.identify_conditions(data)
            
            # Build condition results for analysis
            if conditions:
                # Create condition result for whole period
                condition_results = [(
                    conditions[0] if conditions else None,
                    {
                        'human_metrics': human_results.get('metrics', {}),
                        'ml_metrics': ml_results.get('metrics', {}),
                        'ml_params': {}  # Will be filled by optimization
                    }
                )]
                
                # Analyze ML effectiveness 
                analyzer.analyze_ml_effectiveness(condition_results)
            
            return analyzer
        except Exception as e:
            logger.warning(f"Condition analysis failed: {e}")
            return None
    
    def _detect_failures(
        self,
        human_results: Dict[str, Any],
        ml_results: Dict[str, Any],
        human_params: Dict[str, Any],
        ml_params: Dict[str, Any]
    ) -> FailureDetector:
        """Detect optimization failures."""
        detector = FailureDetector()
        
        # Extract metrics from results
        human_metrics = human_results.get('metrics', {})
        ml_metrics = ml_results.get('metrics', {})
        
        # Call detect_failures with the proper arguments
        detector.detect_failures(
            train_metrics=ml_metrics,  # ML optimized as train
            test_metrics=ml_metrics,   # Same for now (could use walk-forward)
            baseline_metrics=human_metrics,
            ml_params=ml_params
        )
        
        return detector
    
    def _save_report(
        self,
        report: ComparisonReport,
        symbol: str,
        timeframe: str,
        strategy_name: str
    ) -> str:
        """Save comparison report."""
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        filename = f"{symbol}_{timeframe}_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = reports_dir / filename
        
        report.save(str(filepath), format='markdown')
        
        return str(filepath)
    
    def _generate_overall_findings(self) -> Dict[str, Any]:
        """Generate overall research findings."""
        if not self.current_session or not self.current_session.results:
            return {}
        
        results = self.current_session.results
        
        # Aggregate statistics
        total_experiments = len(results)
        ml_recommended = sum(1 for r in results if r.ml_recommended)
        
        avg_sharpe_improvement = np.mean([
            r.improvement.get('sharpe_ratio', 0) for r in results
        ])
        
        # Best/worst scenarios
        best_result = max(results, key=lambda r: r.improvement.get('sharpe_ratio', 0))
        worst_result = min(results, key=lambda r: r.improvement.get('sharpe_ratio', 0))
        
        # By strategy
        by_strategy = {}
        for strategy_name in set(r.strategy_name for r in results):
            strategy_results = [r for r in results if r.strategy_name == strategy_name]
            by_strategy[strategy_name] = {
                'n_experiments': len(strategy_results),
                'ml_recommended_pct': sum(1 for r in strategy_results if r.ml_recommended) / len(strategy_results),
                'avg_sharpe_improvement': np.mean([r.improvement.get('sharpe_ratio', 0) for r in strategy_results])
            }
        
        # By timeframe
        by_timeframe = {}
        for timeframe in set(r.timeframe for r in results):
            tf_results = [r for r in results if r.timeframe == timeframe]
            by_timeframe[timeframe] = {
                'n_experiments': len(tf_results),
                'ml_recommended_pct': sum(1 for r in tf_results if r.ml_recommended) / len(tf_results),
                'avg_sharpe_improvement': np.mean([r.improvement.get('sharpe_ratio', 0) for r in tf_results])
            }
        
        findings = {
            'total_experiments': total_experiments,
            'ml_recommended_count': ml_recommended,
            'ml_recommended_pct': ml_recommended / total_experiments if total_experiments > 0 else 0,
            'avg_sharpe_improvement': avg_sharpe_improvement,
            'best_scenario': {
                'strategy': best_result.strategy_name,
                'symbol': best_result.symbol,
                'timeframe': best_result.timeframe,
                'improvement': best_result.improvement.get('sharpe_ratio', 0)
            },
            'worst_scenario': {
                'strategy': worst_result.strategy_name,
                'symbol': worst_result.symbol,
                'timeframe': worst_result.timeframe,
                'improvement': worst_result.improvement.get('sharpe_ratio', 0)
            },
            'by_strategy': by_strategy,
            'by_timeframe': by_timeframe,
            'key_insights': self._generate_insights(results)
        }
        
        return findings
    
    def _generate_insights(self, results: List[ResearchResult]) -> List[str]:
        """Generate key insights from results."""
        insights = []
        
        ml_pct = sum(1 for r in results if r.ml_recommended) / len(results) if results else 0
        
        if ml_pct > 0.7:
            insights.append("ML optimization consistently improved strategy performance")
        elif ml_pct < 0.3:
            insights.append("ML optimization rarely improved over human parameters")
        else:
            insights.append("ML optimization shows mixed results - context matters")
        
        # Check for strategy-specific patterns
        by_strategy = {}
        for r in results:
            if r.strategy_name not in by_strategy:
                by_strategy[r.strategy_name] = []
            by_strategy[r.strategy_name].append(r.ml_recommended)
        
        for strategy, recommendations in by_strategy.items():
            pct = sum(recommendations) / len(recommendations)
            if pct > 0.8:
                insights.append(f"{strategy} benefits most from ML optimization")
            elif pct < 0.2:
                insights.append(f"{strategy} human parameters are already near-optimal")
        
        # Check for timeframe patterns
        by_tf = {}
        for r in results:
            if r.timeframe not in by_tf:
                by_tf[r.timeframe] = []
            by_tf[r.timeframe].append(r.improvement.get('sharpe_ratio', 0))
        
        for tf, improvements in by_tf.items():
            avg = np.mean(improvements)
            if avg > 0.1:
                insights.append(f"ML helps more on {tf} timeframe")
            elif avg < -0.1:
                insights.append(f"ML hurts performance on {tf} timeframe")
        
        return insights
    
    def _save_session(self) -> None:
        """Save session results."""
        if not self.current_session:
            return
        
        # Save JSON summary
        summary_path = self.output_dir / f"session_{self.current_session.session_id}.json"
        
        session_data = {
            'session_id': self.current_session.session_id,
            'start_time': self.current_session.start_time.isoformat(),
            'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
            'summary': self.current_session.get_summary(),
            'overall_findings': self.current_session.overall_findings,
            'results': [
                {
                    'strategy': r.strategy_name,
                    'symbol': r.symbol,
                    'timeframe': r.timeframe,
                    'ml_recommended': r.ml_recommended,
                    'confidence': r.confidence,
                    'sharpe_improvement': r.improvement.get('sharpe_ratio', 0),
                    'best_optimizer': r.best_optimizer,
                    'report_path': r.report_path
                }
                for r in self.current_session.results
            ]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session saved to {summary_path}")


def run_quick_experiment(
    data_path: str,
    strategy_name: str = 'rsi_mean_reversion',
    timeframe: str = '5m',
    n_trials: int = 50
) -> ResearchResult:
    """
    Quick helper to run a single experiment.
    
    Args:
        data_path: Path to OHLCV data
        strategy_name: Strategy to test
        timeframe: Timeframe to use
        n_trials: Number of optimization trials
        
    Returns:
        ResearchResult
    """
    opt_config = OptimizationConfig(
        bayesian_n_calls=n_trials,
        random_search_n_iter=n_trials
    )
    
    pipeline = ResearchPipeline(
        optimization_config=opt_config,
        output_dir="./quick_experiment"
    )
    
    session = pipeline.run_full_research(
        data_paths=[data_path],
        timeframes=[timeframe],
        strategies=[strategy_name]
    )
    
    return session.results[0] if session.results else None
