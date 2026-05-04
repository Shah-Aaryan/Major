"""
Random Search Optimizer - Baseline for comparison.

Random search serves as a baseline to compare against more sophisticated
optimization methods (Bayesian, Evolutionary). It provides important
context for understanding WHEN ML helps.

Key insight: Random search is surprisingly effective in many cases.
If sophisticated ML barely beats random search, we learn something important
about the optimization landscape.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
import logging
import time

from optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationTrial,
    ParameterSpace,
    ParameterType
)

logger = logging.getLogger(__name__)


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random Search for parameter optimization.
    
    This optimizer randomly samples from the parameter space.
    It serves as a critical baseline for evaluating whether more
    sophisticated methods actually provide value.
    
    Key features:
    - Simple and fast
    - No assumptions about objective function
    - Highly parallelizable
    - Unbiased exploration
    
    When Random Search is competitive:
    - High-dimensional spaces with many local optima
    - Objective functions with low effective dimensionality
    - When limited budget doesn't allow enough iterations for
      model-based methods to learn
    
    RESEARCH VALUE: Comparing ML methods against random search reveals
    whether the ML optimization is actually adding value.
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 100,
        sampling_strategy: str = 'uniform',
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize Random Search Optimizer.
        
        Args:
            parameter_space: Parameter space to search
            objective_function: Function to optimize
            maximize: Whether to maximize objective
            n_iterations: Number of random samples
            sampling_strategy: 'uniform', 'latin_hypercube', or 'sobol'
            random_state: Random seed for reproducibility
            verbose: Logging verbosity
        """
        super().__init__(
            parameter_space=parameter_space,
            objective_function=objective_function,
            maximize=maximize,
            n_iterations=n_iterations,
            random_state=random_state,
            verbose=verbose
        )
        
        self.sampling_strategy = sampling_strategy
        self.name = f"RandomSearch({sampling_strategy})"
        
        # Pre-generate samples if using structured sampling
        if sampling_strategy == 'latin_hypercube':
            self._samples = self._generate_lhs_samples()
        elif sampling_strategy == 'sobol':
            self._samples = self._generate_sobol_samples()
        else:
            self._samples = None
    
    def optimize(
        self,
        baseline_objective: Optional[float] = None
    ) -> OptimizationResult:
        """
        Run random search optimization.
        
        Args:
            baseline_objective: Optional baseline for comparison
            
        Returns:
            OptimizationResult with best parameters found
        """
        start_time = time.time()
        
        logger.info(f"Starting {self.name} with {self.n_iterations} iterations")
        
        for i in range(self.n_iterations):
            # Get next sample
            params = self.suggest_next()
            
            # Evaluate
            self.evaluate(params, trial_id=i)
            
            if self.verbose and (i + 1) % 10 == 0:
                logger.info(
                    f"Iteration {i+1}/{self.n_iterations}: "
                    f"best={self.best_trial.objective_value:.4f}"
                )
        
        total_time = time.time() - start_time
        
        logger.info(
            f"{self.name} complete: best={self.best_trial.objective_value:.4f}, "
            f"time={total_time:.2f}s"
        )
        
        return self._create_result(
            total_time=total_time,
            baseline_objective=baseline_objective
        )
    
    def suggest_next(self) -> Dict[str, Any]:
        """
        Suggest next parameter set to evaluate.
        
        Returns:
            Dictionary of parameter values
        """
        if self._samples is not None and len(self.trials) < len(self._samples):
            # Use pre-generated sample
            return self._samples[len(self.trials)]
        else:
            # Random uniform sample
            return self.parameter_space.sample_random(self.rng)
    
    def _generate_lhs_samples(self) -> List[Dict[str, Any]]:
        """
        Generate Latin Hypercube samples.
        
        Latin Hypercube Sampling ensures better coverage of the
        parameter space than pure random sampling.
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            logger.warning("scipy.stats.qmc not available, using uniform sampling")
            return None
        
        n_dims = len(self.parameter_space.parameters)
        
        # Generate LHS samples in [0, 1]^d
        sampler = qmc.LatinHypercube(d=n_dims, seed=self.random_state)
        unit_samples = sampler.random(n=self.n_iterations)
        
        # Transform to parameter space
        samples = []
        param_names = list(self.parameter_space.parameters.keys())
        
        for unit_sample in unit_samples:
            params = {}
            for j, name in enumerate(param_names):
                spec = self.parameter_space.parameters[name]
                
                if spec.param_type == ParameterType.INTEGER:
                    low, high = int(spec.bounds[0]), int(spec.bounds[1])
                    params[name] = int(low + unit_sample[j] * (high - low + 1))
                    params[name] = min(params[name], high)
                elif spec.param_type == ParameterType.FLOAT:
                    low, high = spec.bounds
                    params[name] = low + unit_sample[j] * (high - low)
                elif spec.param_type == ParameterType.CATEGORICAL:
                    idx = int(unit_sample[j] * len(spec.categories))
                    idx = min(idx, len(spec.categories) - 1)
                    params[name] = spec.categories[idx]
                elif spec.param_type == ParameterType.BOOLEAN:
                    params[name] = unit_sample[j] > 0.5
            
            samples.append(params)
        
        logger.info(f"Generated {len(samples)} Latin Hypercube samples")
        return samples
    
    def _generate_sobol_samples(self) -> List[Dict[str, Any]]:
        """
        Generate Sobol sequence samples.
        
        Sobol sequences are quasi-random low-discrepancy sequences
        that provide even better space coverage than LHS.
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            logger.warning("scipy.stats.qmc not available, using uniform sampling")
            return None
        
        n_dims = len(self.parameter_space.parameters)
        
        # Generate Sobol samples
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=self.random_state)
        unit_samples = sampler.random(n=self.n_iterations)
        
        # Transform to parameter space
        samples = []
        param_names = list(self.parameter_space.parameters.keys())
        
        for unit_sample in unit_samples:
            params = {}
            for j, name in enumerate(param_names):
                spec = self.parameter_space.parameters[name]
                
                if spec.param_type == ParameterType.INTEGER:
                    low, high = int(spec.bounds[0]), int(spec.bounds[1])
                    params[name] = int(low + unit_sample[j] * (high - low + 1))
                    params[name] = min(params[name], high)
                elif spec.param_type == ParameterType.FLOAT:
                    low, high = spec.bounds
                    params[name] = low + unit_sample[j] * (high - low)
                elif spec.param_type == ParameterType.CATEGORICAL:
                    idx = int(unit_sample[j] * len(spec.categories))
                    idx = min(idx, len(spec.categories) - 1)
                    params[name] = spec.categories[idx]
                elif spec.param_type == ParameterType.BOOLEAN:
                    params[name] = unit_sample[j] > 0.5
            
            samples.append(params)
        
        logger.info(f"Generated {len(samples)} Sobol sequence samples")
        return samples


class GridSearchOptimizer(BaseOptimizer):
    """
    Grid Search for parameter optimization.
    
    Exhaustively searches over a grid of parameter values.
    Useful for small parameter spaces or when you want to
    understand the full response surface.
    
    WARNING: Computation scales exponentially with the number
    of parameters. Only use with few parameters.
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        grid_resolution: int = 5,
        maximize: bool = True,
        verbose: bool = True
    ):
        """
        Initialize Grid Search.
        
        Args:
            parameter_space: Parameter space
            objective_function: Function to optimize
            grid_resolution: Number of points per dimension
            maximize: Whether to maximize objective
            verbose: Logging verbosity
        """
        # Calculate total iterations
        n_params = len(parameter_space.parameters)
        n_iterations = grid_resolution ** n_params
        
        super().__init__(
            parameter_space=parameter_space,
            objective_function=objective_function,
            maximize=maximize,
            n_iterations=n_iterations,
            verbose=verbose
        )
        
        self.grid_resolution = grid_resolution
        self.name = f"GridSearch(res={grid_resolution})"
        
        # Generate grid
        self._grid = self._generate_grid()
        
        if len(self._grid) > 10000:
            logger.warning(
                f"Grid search has {len(self._grid)} points. "
                "Consider using random search for large spaces."
            )
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate the parameter grid."""
        import itertools
        
        param_values = {}
        
        for name, spec in self.parameter_space.parameters.items():
            if spec.param_type == ParameterType.INTEGER:
                low, high = int(spec.bounds[0]), int(spec.bounds[1])
                step = max(1, (high - low) // (self.grid_resolution - 1))
                param_values[name] = list(range(low, high + 1, step))[:self.grid_resolution]
            elif spec.param_type == ParameterType.FLOAT:
                param_values[name] = np.linspace(
                    spec.bounds[0], spec.bounds[1], self.grid_resolution
                ).tolist()
            elif spec.param_type == ParameterType.CATEGORICAL:
                param_values[name] = spec.categories
            elif spec.param_type == ParameterType.BOOLEAN:
                param_values[name] = [True, False]
        
        # Generate all combinations
        param_names = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[n] for n in param_names]))
        
        grid = [
            dict(zip(param_names, combo))
            for combo in combinations
        ]
        
        logger.info(f"Generated grid with {len(grid)} parameter combinations")
        return grid
    
    def optimize(
        self,
        baseline_objective: Optional[float] = None
    ) -> OptimizationResult:
        """Run grid search."""
        start_time = time.time()
        
        logger.info(f"Starting Grid Search with {len(self._grid)} combinations")
        
        for i, params in enumerate(self._grid):
            self.evaluate(params, trial_id=i)
            
            if self.verbose and (i + 1) % 100 == 0:
                logger.info(
                    f"Evaluated {i+1}/{len(self._grid)}: "
                    f"best={self.best_trial.objective_value:.4f}"
                )
        
        total_time = time.time() - start_time
        
        return self._create_result(
            total_time=total_time,
            baseline_objective=baseline_objective
        )
    
    def suggest_next(self) -> Dict[str, Any]:
        """Get next grid point."""
        if len(self.trials) < len(self._grid):
            return self._grid[len(self.trials)]
        return self.parameter_space.sample_random()
    
    def get_response_surface(self, param1: str, param2: str) -> pd.DataFrame:
        """
        Get 2D response surface for visualization.
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            
        Returns:
            DataFrame with param1, param2, and objective values
        """
        data = []
        
        for trial in self.trials:
            data.append({
                param1: trial.parameters.get(param1),
                param2: trial.parameters.get(param2),
                'objective': trial.objective_value
            })
        
        df = pd.DataFrame(data)
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            values='objective',
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        
        return pivot


def create_random_search_for_strategy(
    strategy_bounds: Dict[str, Tuple[float, float]],
    objective_function: Callable[[Dict[str, Any]], float],
    integer_params: Optional[List[str]] = None,
    n_iterations: int = 100,
    sampling_strategy: str = 'latin_hypercube',
    **kwargs
) -> RandomSearchOptimizer:
    """
    Factory function to create a Random Search optimizer for a strategy.
    
    Args:
        strategy_bounds: Parameter bounds from strategy
        objective_function: Function to evaluate parameters
        integer_params: Names of integer parameters
        n_iterations: Number of iterations
        sampling_strategy: Sampling method
        **kwargs: Additional arguments
        
    Returns:
        Configured RandomSearchOptimizer
    """
    default_integer_params = [
        'rsi_lookback', 'ema_fast_period', 'ema_slow_period',
        'bb_period', 'squeeze_lookback', 'min_squeeze_candles',
        'adx_period', 'atr_period'
    ]
    
    if integer_params is None:
        integer_params = default_integer_params
    
    space = ParameterSpace.from_strategy_bounds(strategy_bounds, integer_params)
    
    return RandomSearchOptimizer(
        parameter_space=space,
        objective_function=objective_function,
        n_iterations=n_iterations,
        sampling_strategy=sampling_strategy,
        **kwargs
    )
