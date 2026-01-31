"""
Bayesian Optimization for parameter tuning.

Uses Gaussian Process-based Bayesian Optimization (via scikit-optimize or Optuna)
to efficiently search the parameter space.

IMPORTANT: This optimizer only adjusts PARAMETERS, not strategy logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import time

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationTrial,
    ParameterSpace,
    ParameterType
)

logger = logging.getLogger(__name__)


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using Gaussian Processes.
    
    This optimizer uses a probabilistic model (Gaussian Process) to model
    the objective function and intelligently selects the next parameters
    to evaluate based on expected improvement.
    
    Key features:
    - Efficient exploration of parameter space
    - Handles noisy objective functions
    - Provides uncertainty estimates
    - Works well with expensive objective evaluations
    
    Backend options:
    - 'skopt': scikit-optimize (Gaussian Process)
    - 'optuna': Optuna with TPE sampler
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 100,
        n_initial_points: int = 10,
        backend: str = 'optuna',
        random_state: Optional[int] = None,
        verbose: bool = True,
        acq_func: str = 'EI'  # Acquisition function: EI, PI, LCB
    ):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            parameter_space: Parameter space to optimize over
            objective_function: Function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            n_iterations: Number of optimization iterations
            n_initial_points: Number of random initial samples
            backend: 'skopt' or 'optuna'
            random_state: Random seed
            verbose: Logging verbosity
            acq_func: Acquisition function (EI=Expected Improvement)
        """
        super().__init__(
            parameter_space=parameter_space,
            objective_function=objective_function,
            maximize=maximize,
            n_iterations=n_iterations,
            random_state=random_state,
            verbose=verbose
        )
        
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        
        # Select backend
        if backend == 'optuna' and OPTUNA_AVAILABLE:
            self.backend = 'optuna'
        elif backend == 'skopt' and SKOPT_AVAILABLE:
            self.backend = 'skopt'
        elif OPTUNA_AVAILABLE:
            self.backend = 'optuna'
            logger.warning(f"Backend '{backend}' not available, using optuna")
        elif SKOPT_AVAILABLE:
            self.backend = 'skopt'
            logger.warning(f"Backend '{backend}' not available, using skopt")
        else:
            raise ImportError(
                "Neither optuna nor scikit-optimize is available. "
                "Install with: pip install optuna scikit-optimize"
            )
        
        logger.info(f"Using {self.backend} backend for Bayesian optimization")
        
        self.name = f"BayesianOptimizer({self.backend})"
    
    def optimize(
        self,
        baseline_objective: Optional[float] = None
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            baseline_objective: Optional baseline (human params) for comparison
            
        Returns:
            OptimizationResult with best parameters
        """
        start_time = time.time()
        
        if self.backend == 'optuna':
            result = self._optimize_optuna()
        else:
            result = self._optimize_skopt()
        
        total_time = time.time() - start_time
        
        return self._create_result(
            total_time=total_time,
            baseline_objective=baseline_objective
        )
    
    def _optimize_optuna(self) -> None:
        """Run optimization using Optuna."""
        # Create Optuna study
        direction = 'maximize' if self.maximize else 'minimize'
        
        sampler = TPESampler(
            seed=self.random_state,
            n_startup_trials=self.n_initial_points
        )
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler
        )
        
        # Suppress Optuna logging if not verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective wrapper."""
            params = self._suggest_optuna(trial)
            
            # Evaluate
            opt_trial = self.evaluate(params, trial.number)
            
            return opt_trial.objective_value
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_iterations,
            show_progress_bar=self.verbose
        )
        
        # Store best trial info
        self.optuna_study = study
    
    def _suggest_optuna(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters using Optuna trial."""
        params = {}
        
        for name, spec in self.parameter_space.parameters.items():
            if spec.param_type == ParameterType.INTEGER:
                params[name] = trial.suggest_int(
                    name, int(spec.bounds[0]), int(spec.bounds[1])
                )
            elif spec.param_type == ParameterType.FLOAT:
                params[name] = trial.suggest_float(
                    name, spec.bounds[0], spec.bounds[1]
                )
            elif spec.param_type == ParameterType.CATEGORICAL:
                params[name] = trial.suggest_categorical(name, spec.categories)
            elif spec.param_type == ParameterType.BOOLEAN:
                params[name] = trial.suggest_categorical(name, [True, False])
        
        return params
    
    def _optimize_skopt(self) -> None:
        """Run optimization using scikit-optimize."""
        # Build search space
        dimensions = []
        param_names = []
        
        for name, spec in self.parameter_space.parameters.items():
            param_names.append(name)
            
            if spec.param_type == ParameterType.INTEGER:
                dimensions.append(
                    Integer(int(spec.bounds[0]), int(spec.bounds[1]), name=name)
                )
            elif spec.param_type == ParameterType.FLOAT:
                dimensions.append(
                    Real(spec.bounds[0], spec.bounds[1], name=name)
                )
            elif spec.param_type == ParameterType.CATEGORICAL:
                dimensions.append(
                    Categorical(spec.categories, name=name)
                )
            elif spec.param_type == ParameterType.BOOLEAN:
                dimensions.append(
                    Categorical([True, False], name=name)
                )
        
        # Objective wrapper for skopt
        @use_named_args(dimensions)
        def objective(**kwargs):
            trial = self.evaluate(kwargs)
            # skopt minimizes, so negate if maximizing
            return -trial.objective_value if self.maximize else trial.objective_value
        
        # Run optimization
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=self.n_iterations,
            n_initial_points=self.n_initial_points,
            acq_func=self.acq_func,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        self.skopt_result = result
    
    def suggest_next(self) -> Dict[str, Any]:
        """
        Suggest next parameters (for manual step-by-step optimization).
        
        This method is mainly useful for the random search fallback.
        The main optimization loop uses the backend's internal suggestion.
        """
        # Simple fallback: random sample
        return self.parameter_space.sample_random(self.rng)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get parameter importance from the optimization.
        
        Only available after optimization is complete.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if hasattr(self, 'optuna_study'):
            try:
                importance = optuna.importance.get_param_importances(
                    self.optuna_study
                )
                return dict(importance)
            except Exception as e:
                logger.warning(f"Could not compute parameter importance: {e}")
                return None
        
        # For skopt, we can't easily get importance
        return None
    
    def get_optimization_history_plot_data(self) -> Dict[str, Any]:
        """
        Get data for plotting optimization history.
        
        Returns:
            Dictionary with plot data
        """
        return {
            'trial_numbers': list(range(len(self.trials))),
            'objective_values': [t.objective_value for t in self.trials],
            'best_so_far': self.convergence_history,
            'parameters': [t.parameters for t in self.trials]
        }


class BayesianOptimizerWithPriors(BayesianOptimizer):
    """
    Bayesian Optimizer that can incorporate prior knowledge.
    
    This variant allows specifying prior beliefs about good parameter
    regions, which can speed up convergence when human expertise is available.
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        prior_params: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs
    ):
        """
        Initialize with prior knowledge.
        
        Args:
            parameter_space: Parameter space
            objective_function: Objective to optimize
            prior_params: Dict mapping param names to (mean, std) of prior belief
            **kwargs: Additional arguments for BayesianOptimizer
        """
        super().__init__(parameter_space, objective_function, **kwargs)
        
        self.prior_params = prior_params or {}
        
        # If priors provided, seed initial points near prior means
        if prior_params:
            self._initial_points_from_priors = self._generate_prior_samples()
    
    def _generate_prior_samples(self) -> List[Dict[str, Any]]:
        """Generate initial samples from prior distributions."""
        samples = []
        n_prior_samples = min(self.n_initial_points // 2, 5)
        
        for _ in range(n_prior_samples):
            sample = {}
            for name, spec in self.parameter_space.parameters.items():
                if name in self.prior_params:
                    mean, std = self.prior_params[name]
                    # Sample from truncated normal around prior
                    value = self.rng.normal(mean, std)
                    value = spec.clip(value)
                else:
                    # Random sample
                    value = self.parameter_space.sample_random(self.rng)[name]
                sample[name] = value
            samples.append(sample)
        
        return samples


def create_bayesian_optimizer_for_strategy(
    strategy_bounds: Dict[str, Tuple[float, float]],
    objective_function: Callable[[Dict[str, Any]], float],
    integer_params: Optional[List[str]] = None,
    n_iterations: int = 100,
    **kwargs
) -> BayesianOptimizer:
    """
    Factory function to create a Bayesian optimizer for a trading strategy.
    
    Args:
        strategy_bounds: Parameter bounds from strategy
        objective_function: Function that evaluates parameters
        integer_params: Names of integer parameters
        n_iterations: Number of iterations
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured BayesianOptimizer
    """
    # Common integer parameters in trading strategies
    default_integer_params = [
        'rsi_lookback', 'ema_fast_period', 'ema_slow_period',
        'bb_period', 'squeeze_lookback', 'min_squeeze_candles',
        'adx_period', 'atr_period', 'macd_fast', 'macd_slow', 'macd_signal'
    ]
    
    if integer_params is None:
        integer_params = default_integer_params
    
    space = ParameterSpace.from_strategy_bounds(strategy_bounds, integer_params)
    
    return BayesianOptimizer(
        parameter_space=space,
        objective_function=objective_function,
        n_iterations=n_iterations,
        **kwargs
    )
