"""
Simulated Annealing optimizer (dual_annealing backend).

Provides a trajectory-based optimizer as part of the BeyondAlgo registry.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationTrial,
    ParameterSpace,
    ParameterType,
)

logger = logging.getLogger(__name__)


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Simulated Annealing via scipy.optimize.dual_annealing."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 200,
        initial_temp: float = 5230.0,
        restart_temp_ratio: float = 5e-4,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__(
            parameter_space=parameter_space,
            objective_function=objective_function,
            maximize=maximize,
            n_iterations=n_iterations,
            random_state=random_state,
            verbose=verbose,
        )
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio
        self.name = "SimulatedAnnealing"

    def optimize(
        self,
        baseline_objective: Optional[float] = None,
    ) -> OptimizationResult:
        try:
            from scipy.optimize import dual_annealing
        except ImportError as exc:
            raise ImportError(
                "scipy is required for SimulatedAnnealingOptimizer"
            ) from exc

        start_time = time.time()
        param_names = list(self.parameter_space.parameters.keys())
        bounds = [self.parameter_space.parameters[name].bounds for name in param_names]

        def objective_wrapper(x):
            params = {name: x[i] for i, name in enumerate(param_names)}

            # Handle integer/categorical/boolean casts
            for name in param_names:
                spec = self.parameter_space.parameters[name]
                if spec.param_type == ParameterType.INTEGER:
                    params[name] = int(round(params[name]))
                    params[name] = min(max(params[name], spec.bounds[0]), spec.bounds[1])
                elif spec.param_type == ParameterType.CATEGORICAL:
                    idx = int(round(params[name])) % len(spec.categories)
                    params[name] = spec.categories[idx]
                elif spec.param_type == ParameterType.BOOLEAN:
                    params[name] = bool(round(params[name]))

            trial = self.evaluate(params)
            return -trial.objective_value if self.maximize else trial.objective_value

        result = dual_annealing(
            objective_wrapper,
            bounds,
            maxiter=self.n_iterations,
            initial_temp=self.initial_temp,
            restart_temp_ratio=self.restart_temp_ratio,
            seed=self.random_state,
            no_local_search=True,
        )

        total_time = time.time() - start_time

        best_params = {name: result.x[i] for i, name in enumerate(param_names)}
        for name in param_names:
            spec = self.parameter_space.parameters[name]
            if spec.param_type == ParameterType.INTEGER:
                best_params[name] = int(round(best_params[name]))
            elif spec.param_type == ParameterType.CATEGORICAL:
                idx = int(round(best_params[name])) % len(spec.categories)
                best_params[name] = spec.categories[idx]
            elif spec.param_type == ParameterType.BOOLEAN:
                best_params[name] = bool(round(best_params[name]))

        best_objective = -result.fun if self.maximize else result.fun

        self.best_trial = OptimizationTrial(
            trial_id=len(self.trials),
            parameters=best_params,
            objective_value=best_objective,
        )

        return self._create_result(
            total_time=total_time,
            baseline_objective=baseline_objective,
        )

    def suggest_next(self) -> Dict[str, Any]:
        return self.parameter_space.sample_random(self.rng)
