"""
Additional optimizers: CMA-ES, Particle Swarm, Evolution Strategies, Hyperband/ASHA.

These are lightweight wrappers to extend the optimizer registry with functional
implementations using existing libraries (Optuna or built-in loops).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationTrial,
    ParameterSpace,
    ParameterType,
)

logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class CMAESOptimizer(BaseOptimizer):
    """CMA-ES optimizer.

    Prefers Optuna's `CmaEsSampler` when available. If the installed Optuna
    version does not include a CMA-ES sampler, falls back to the standalone
    `cmaes` package.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 100,
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
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for CMAESOptimizer")
        self.name = "CMA-ES"

    def optimize(self, baseline_objective: Optional[float] = None) -> OptimizationResult:
        start = time.time()

        # --- Path A: Optuna CMA-ES sampler (if supported) ---
        if OPTUNA_AVAILABLE:
            sampler_cls = (
                getattr(optuna.samplers, "CmaEsSampler", None)
                or getattr(optuna.samplers, "CMAESampler", None)
            )
            if sampler_cls is not None:
                directions = "maximize" if self.maximize else "minimize"
                try:
                    sampler = sampler_cls(seed=self.random_state)
                except ModuleNotFoundError as e:
                    # Optuna's CMA-ES sampler depends on the external `cmaes` package.
                    raise ImportError(
                        "CMA-ES requires the 'cmaes' package. Install with: pip install cmaes"
                    ) from e

                study = optuna.create_study(direction=directions, sampler=sampler)

                def objective(trial: optuna.Trial) -> float:
                    params: Dict[str, Any] = {}
                    for name, spec in self.parameter_space.parameters.items():
                        if spec.param_type == ParameterType.INTEGER:
                            params[name] = trial.suggest_int(name, int(spec.bounds[0]), int(spec.bounds[1]))
                        elif spec.param_type == ParameterType.FLOAT:
                            params[name] = trial.suggest_float(name, spec.bounds[0], spec.bounds[1])
                        elif spec.param_type == ParameterType.CATEGORICAL:
                            params[name] = trial.suggest_categorical(name, spec.categories)
                        elif spec.param_type == ParameterType.BOOLEAN:
                            params[name] = trial.suggest_categorical(name, [True, False])
                    opt_trial = self.evaluate(params)
                    return opt_trial.objective_value

                study.optimize(objective, n_trials=self.n_iterations, show_progress_bar=False)
                total_time = time.time() - start

                best = study.best_trial
                self.best_trial = OptimizationTrial(
                    trial_id=best.number,
                    parameters=best.params,
                    objective_value=best.value,
                )
                return self._create_result(total_time=total_time, baseline_objective=baseline_objective)

        # --- Path B: Standalone `cmaes` fallback ---
        try:
            from cmaes import CMA  # type: ignore
        except Exception as e:
            raise ImportError(
                "CMA-ES fallback requires the 'cmaes' package. Install with: pip install cmaes"
            ) from e

        # Encode parameters into a continuous vector in [0, 1] and decode back.
        names: List[str] = list(self.parameter_space.parameters.keys())
        specs: List[Any] = [self.parameter_space.parameters[n] for n in names]

        def decode(x01: np.ndarray) -> Dict[str, Any]:
            decoded: Dict[str, Any] = {}
            for i, (name, spec) in enumerate(zip(names, specs)):
                u = float(np.clip(x01[i], 0.0, 1.0))
                lo, hi = float(spec.bounds[0]), float(spec.bounds[1])
                raw = lo + u * (hi - lo)
                if spec.param_type == ParameterType.FLOAT:
                    decoded[name] = float(raw)
                elif spec.param_type == ParameterType.INTEGER:
                    decoded[name] = int(round(raw))
                elif spec.param_type == ParameterType.BOOLEAN:
                    decoded[name] = bool(int(round(raw)) > 0)
                elif spec.param_type == ParameterType.CATEGORICAL:
                    if not spec.categories:
                        decoded[name] = None
                    else:
                        idx = int(round(raw))
                        idx = max(0, min(len(spec.categories) - 1, idx))
                        decoded[name] = spec.categories[idx]
                else:
                    decoded[name] = raw
            return decoded

        dim = max(1, len(names))
        popsize = max(8, min(32, 4 * dim))
        optimizer = CMA(mean=np.full(dim, 0.5, dtype=float), sigma=0.2, population_size=popsize)

        eval_count = 0
        while eval_count < int(self.n_iterations):
            solutions: List[np.ndarray] = []
            fitnesses: List[float] = []
            for _ in range(popsize):
                if eval_count >= int(self.n_iterations):
                    break
                x = optimizer.ask()
                x = np.asarray(x, dtype=float)
                params = decode(x)
                trial = self.evaluate(params)
                eval_count += 1

                # cmaes minimizes; convert if we are maximizing.
                fit = -float(trial.objective_value) if self.maximize else float(trial.objective_value)
                solutions.append(x)
                fitnesses.append(fit)

            if solutions:
                optimizer.tell(solutions, fitnesses)

        total_time = time.time() - start
        return self._create_result(total_time=total_time, baseline_objective=baseline_objective)


class ParticleSwarmOptimizer(BaseOptimizer):
    """Lightweight Particle Swarm Optimization (PSO)."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 100,
        swarm_size: int = 30,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5,
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
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.name = "ParticleSwarm"
        self.rng = np.random.default_rng(random_state)

    def optimize(self, baseline_objective: Optional[float] = None) -> OptimizationResult:
        params = list(self.parameter_space.parameters.items())
        n_dims = len(params)

        # Initialize particles uniformly
        positions = []
        velocities = self.rng.normal(scale=0.1, size=(self.swarm_size, n_dims))
        for _ in range(self.swarm_size):
            sample = self.parameter_space.sample_random(self.rng)
            positions.append([sample[name] for name, _ in params])
        positions = np.array(positions, dtype=float)

        personal_best_pos = positions.copy()
        personal_best_val = np.full(self.swarm_size, -np.inf if self.maximize else np.inf)

        global_best_pos = None
        global_best_val = -np.inf if self.maximize else np.inf

        start = time.time()
        for it in range(self.n_iterations):
            for i in range(self.swarm_size):
                param_dict = {}
                for j, (name, spec) in enumerate(params):
                    val = positions[i, j]
                    if spec.param_type == ParameterType.INTEGER:
                        val = int(round(np.clip(val, spec.bounds[0], spec.bounds[1])))
                    elif spec.param_type == ParameterType.FLOAT:
                        val = float(np.clip(val, spec.bounds[0], spec.bounds[1]))
                    elif spec.param_type == ParameterType.CATEGORICAL:
                        idx = int(abs(val)) % len(spec.categories)
                        val = spec.categories[idx]
                    elif spec.param_type == ParameterType.BOOLEAN:
                        val = bool(round(val))
                    param_dict[name] = val

                trial = self.evaluate(param_dict)
                value = trial.objective_value

                better = value > personal_best_val[i] if self.maximize else value < personal_best_val[i]
                if better:
                    personal_best_val[i] = value
                    personal_best_pos[i] = positions[i].copy()

                if global_best_pos is None:
                    global_best_pos = positions[i].copy()
                    global_best_val = value
                else:
                    better_global = value > global_best_val if self.maximize else value < global_best_val
                    if better_global:
                        global_best_val = value
                        global_best_pos = positions[i].copy()

            # Update velocities and positions
            r1, r2 = self.rng.random(size=(2, self.swarm_size, n_dims))
            velocities = (
                self.inertia * velocities
                + self.cognitive * r1 * (personal_best_pos - positions)
                + self.social * r2 * (global_best_pos - positions)
            )
            positions = positions + velocities

        total_time = time.time() - start

        # Final best trial
        best_params = {}
        for j, (name, spec) in enumerate(params):
            val = global_best_pos[j]
            if spec.param_type == ParameterType.INTEGER:
                val = int(round(np.clip(val, spec.bounds[0], spec.bounds[1])))
            elif spec.param_type == ParameterType.FLOAT:
                val = float(np.clip(val, spec.bounds[0], spec.bounds[1]))
            elif spec.param_type == ParameterType.CATEGORICAL:
                idx = int(abs(val)) % len(spec.categories)
                val = spec.categories[idx]
            elif spec.param_type == ParameterType.BOOLEAN:
                val = bool(round(val))
            best_params[name] = val

        self.best_trial = OptimizationTrial(
            trial_id=len(self.trials),
            parameters=best_params,
            objective_value=global_best_val,
        )

        return self._create_result(total_time=total_time, baseline_objective=baseline_objective)


class EvolutionStrategiesOptimizer(BaseOptimizer):
    """Simple (μ, λ)-ES with Gaussian mutations."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 50,
        mu: int = 10,
        lambd: int = 40,
        sigma: float = 0.1,
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
        self.mu = mu
        self.lambd = lambd
        self.sigma = sigma
        self.name = "EvolutionStrategies"
        self.rng = np.random.default_rng(random_state)

    def optimize(self, baseline_objective: Optional[float] = None) -> OptimizationResult:
        params = list(self.parameter_space.parameters.items())
        n_dims = len(params)

        # Initialize parents
        parents = [self.parameter_space.sample_random(self.rng) for _ in range(self.mu)]
        start = time.time()

        for _ in range(self.n_iterations):
            # Generate offspring
            offspring = []
            for _ in range(self.lambd):
                base = parents[self.rng.integers(0, self.mu)]
                child = {}
                for name, spec in params:
                    val = base[name]
                    if spec.param_type in {ParameterType.INTEGER, ParameterType.FLOAT}:
                        noise = self.rng.normal(scale=self.sigma)
                        val = float(val) + noise
                        val = np.clip(val, spec.bounds[0], spec.bounds[1])
                        if spec.param_type == ParameterType.INTEGER:
                            val = int(round(val))
                    elif spec.param_type == ParameterType.CATEGORICAL:
                        if self.rng.random() < 0.2:
                            val = random.choice(spec.categories)
                    elif spec.param_type == ParameterType.BOOLEAN:
                        if self.rng.random() < 0.1:
                            val = not val
                    child[name] = val
                offspring.append(child)

            # Evaluate offspring
            scored = []
            for child in offspring:
                trial = self.evaluate(child)
                scored.append((trial.objective_value, child))

            # Select next parents (μ best)
            scored.sort(key=lambda item: item[0], reverse=self.maximize)
            parents = [child for _, child in scored[: self.mu]]

        total_time = time.time() - start

        # Best among parents
        best_val = -np.inf if self.maximize else np.inf
        best_params = parents[0]
        for p in parents:
            trial = self.evaluate(p)
            val = trial.objective_value
            better = val > best_val if self.maximize else val < best_val
            if better:
                best_val = val
                best_params = p

        self.best_trial = OptimizationTrial(
            trial_id=len(self.trials),
            parameters=best_params,
            objective_value=best_val,
        )
        return self._create_result(total_time=total_time, baseline_objective=baseline_objective)


class HyperbandASHAOptimizer(BaseOptimizer):
    """Hyperband/ASHA using Optuna pruners."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any], int], float],
        maximize: bool = True,
        n_iterations: int = 50,
        reduction_factor: int = 3,
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
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Hyperband/ASHA optimizer")
        self.reduction_factor = reduction_factor
        self.name = "HyperbandASHA"

    def optimize(self, baseline_objective: Optional[float] = None) -> OptimizationResult:
        direction = "maximize" if self.maximize else "minimize"
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=self.reduction_factor)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, spec in self.parameter_space.parameters.items():
                if spec.param_type == ParameterType.INTEGER:
                    params[name] = trial.suggest_int(name, int(spec.bounds[0]), int(spec.bounds[1]))
                elif spec.param_type == ParameterType.FLOAT:
                    params[name] = trial.suggest_float(name, spec.bounds[0], spec.bounds[1])
                elif spec.param_type == ParameterType.CATEGORICAL:
                    params[name] = trial.suggest_categorical(name, spec.categories)
                elif spec.param_type == ParameterType.BOOLEAN:
                    params[name] = trial.suggest_categorical(name, [True, False])

            # Resource/epoch is trial.number+1; objective_function must accept resource arg
            resource = trial.number + 1
            trial.report(resource, step=resource)
            return self.objective_function(params, resource)

        start = time.time()
        study.optimize(objective, n_trials=self.n_iterations, show_progress_bar=False)
        total_time = time.time() - start

        best = study.best_trial
        self.best_trial = OptimizationTrial(
            trial_id=best.number,
            parameters=best.params,
            objective_value=best.value,
        )
        return self._create_result(total_time=total_time, baseline_objective=baseline_objective)