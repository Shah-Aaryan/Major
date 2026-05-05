"""
Base Optimizer - Abstract base class for all parameter optimizers.

All ML optimization methods inherit from this base class to provide
a consistent interface for parameter optimization.

CRITICAL: The optimizer only adjusts PARAMETERS, never strategy LOGIC.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Callable
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of parameters."""
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    param_type: ParameterType
    bounds: Tuple[float, float]  # (min, max) for numeric
    categories: Optional[List[Any]] = None  # For categorical
    default: Optional[Any] = None
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """Check if a value is valid for this parameter."""
        if self.param_type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        elif self.param_type == ParameterType.CATEGORICAL:
            return value in (self.categories or [])
        elif self.param_type == ParameterType.INTEGER:
            return self.bounds[0] <= value <= self.bounds[1]
        elif self.param_type == ParameterType.FLOAT:
            return self.bounds[0] <= value <= self.bounds[1]
        return False
    
    def clip(self, value: Any) -> Any:
        """Clip value to valid range."""
        if self.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
            clipped = max(self.bounds[0], min(self.bounds[1], value))
            if self.param_type == ParameterType.INTEGER:
                return int(round(clipped))
            return clipped
        return value


@dataclass
class ParameterSpace:
    """
    Defines the parameter space for optimization.
    
    This class encapsulates all parameters that can be optimized,
    their types, bounds, and constraints.
    """
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    
    def add_parameter(
        self,
        name: str,
        param_type: ParameterType,
        bounds: Tuple[float, float],
        categories: Optional[List[Any]] = None,
        default: Optional[Any] = None,
        description: str = ""
    ) -> None:
        """Add a parameter to the space."""
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type=param_type,
            bounds=bounds,
            categories=categories,
            default=default,
            description=description
        )
    
    def add_integer(
        self,
        name: str,
        low: int,
        high: int,
        default: Optional[int] = None,
        description: str = ""
    ) -> None:
        """Add an integer parameter."""
        self.add_parameter(
            name=name,
            param_type=ParameterType.INTEGER,
            bounds=(low, high),
            default=default,
            description=description
        )
    
    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        default: Optional[float] = None,
        description: str = ""
    ) -> None:
        """Add a float parameter."""
        self.add_parameter(
            name=name,
            param_type=ParameterType.FLOAT,
            bounds=(low, high),
            default=default,
            description=description
        )
    
    def add_categorical(
        self,
        name: str,
        categories: List[Any],
        default: Optional[Any] = None,
        description: str = ""
    ) -> None:
        """Add a categorical parameter."""
        self.add_parameter(
            name=name,
            param_type=ParameterType.CATEGORICAL,
            bounds=(0, len(categories) - 1),
            categories=categories,
            default=default,
            description=description
        )
    
    def add_boolean(
        self,
        name: str,
        default: Optional[bool] = None,
        description: str = ""
    ) -> None:
        """Add a boolean parameter."""
        self.add_parameter(
            name=name,
            param_type=ParameterType.BOOLEAN,
            bounds=(0, 1),
            default=default,
            description=description
        )
    
    def sample_random(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Sample a random point from the parameter space."""
        if rng is None:
            rng = np.random.default_rng()
        
        sample = {}
        for name, spec in self.parameters.items():
            if spec.param_type == ParameterType.INTEGER:
                sample[name] = rng.integers(int(spec.bounds[0]), int(spec.bounds[1]) + 1)
            elif spec.param_type == ParameterType.FLOAT:
                sample[name] = rng.uniform(spec.bounds[0], spec.bounds[1])
            elif spec.param_type == ParameterType.CATEGORICAL:
                sample[name] = rng.choice(spec.categories)
            elif spec.param_type == ParameterType.BOOLEAN:
                sample[name] = rng.choice([True, False])
        
        return sample
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return {
            name: spec.default
            for name, spec in self.parameters.items()
            if spec.default is not None
        }
    
    def validate(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a parameter set.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for name, spec in self.parameters.items():
            if name not in params:
                errors.append(f"Missing parameter: {name}")
            elif not spec.validate(params[name]):
                errors.append(
                    f"Invalid value for {name}: {params[name]} "
                    f"(bounds: {spec.bounds})"
                )
        
        return len(errors) == 0, errors
    
    def clip_all(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clip all parameters to valid ranges."""
        return {
            name: self.parameters[name].clip(value)
            for name, value in params.items()
            if name in self.parameters
        }
    
    @classmethod
    def from_strategy_bounds(
        cls,
        bounds: Dict[str, Tuple[float, float]],
        integer_params: Optional[List[str]] = None
    ) -> 'ParameterSpace':
        """
        Create a ParameterSpace from strategy bounds dictionary.
        
        Args:
            bounds: Dictionary mapping parameter names to (min, max) bounds
            integer_params: List of parameter names that should be integers
            
        Returns:
            ParameterSpace instance
        """
        space = cls()
        integer_params = integer_params or []
        
        for name, (low, high) in bounds.items():
            if name in integer_params:
                space.add_integer(name, int(low), int(high))
            else:
                space.add_float(name, low, high)
        
        return space


@dataclass
class OptimizationTrial:
    """Record of a single optimization trial."""
    trial_id: int
    parameters: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trial_id': self.trial_id,
            'parameters': self.parameters,
            'objective_value': self.objective_value,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class OptimizationResult:
    """
    Complete result of an optimization run.
    
    Contains the best parameters found, optimization history,
    and diagnostic information.
    """
    best_parameters: Dict[str, Any]
    best_objective: float
    trials: List[OptimizationTrial]
    
    # Metadata
    strategy_name: str = ""
    optimizer_name: str = ""
    n_iterations: int = 0
    total_time_seconds: float = 0.0
    convergence_history: List[float] = field(default_factory=list)
    
    # Comparison info
    baseline_objective: Optional[float] = None
    improvement_pct: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.baseline_objective is not None and self.baseline_objective != 0:
            self.improvement_pct = (
                (self.best_objective - self.baseline_objective)
                / abs(self.baseline_objective) * 100
            )
    
    def get_top_trials(self, n: int = 5, maximize: bool = True) -> List[OptimizationTrial]:
        """Get the top N trials by objective value."""
        sorted_trials = sorted(
            self.trials,
            key=lambda t: t.objective_value,
            reverse=maximize
        )
        return sorted_trials[:n]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trials to a DataFrame."""
        rows = []
        for trial in self.trials:
            row = {'trial_id': trial.trial_id, 'objective': trial.objective_value}
            row.update(trial.parameters)
            row.update(trial.metrics)
            rows.append(row)
        return pd.DataFrame(rows)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_parameters': self.best_parameters,
            'best_objective': self.best_objective,
            'strategy_name': self.strategy_name,
            'optimizer_name': self.optimizer_name,
            'n_iterations': self.n_iterations,
            'total_time_seconds': self.total_time_seconds,
            'convergence_history': self.convergence_history,
            'baseline_objective': self.baseline_objective,
            'improvement_pct': self.improvement_pct,
            'trials': [t.to_dict() for t in self.trials]
        }
    
    def save(self, filepath: str) -> None:
        """Save result to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved optimization result to {filepath}")


class BaseOptimizer(ABC):
    """
    Abstract base class for parameter optimizers.
    
    All optimization methods (Bayesian, Random, Evolutionary) inherit
    from this class to provide a consistent interface.
    
    IMPORTANT: The optimizer only tunes parameters, never changes strategy logic.
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 100,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the optimizer.
        
        Args:
            parameter_space: Definition of parameters to optimize
            objective_function: Function that takes params and returns objective value
            maximize: If True, maximize the objective; if False, minimize
            n_iterations: Number of optimization iterations
            random_state: Random seed for reproducibility
            verbose: Whether to log progress
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.maximize = maximize
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize random generator
        self.rng = np.random.default_rng(random_state)
        
        # Optimization state
        self.trials: List[OptimizationTrial] = []
        self.best_trial: Optional[OptimizationTrial] = None
        self.convergence_history: List[float] = []
        
        self.name = self.__class__.__name__
    
    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """
        Run the optimization process.
        
        Returns:
            OptimizationResult containing best parameters and history
        """
        pass
    
    def suggest_next(self) -> Dict[str, Any]:
        """Suggest the next parameter set to evaluate.

        Some optimizers in this project override :meth:`optimize` directly and
        do not use an iterative suggest/evaluate loop. For those, a default
        random suggestion keeps the base class instantiable.

        Returns:
            Dictionary of parameter values
        """
        return self.parameter_space.sample_random(self.rng)
    
    def evaluate(
        self,
        parameters: Dict[str, Any],
        trial_id: Optional[int] = None
    ) -> OptimizationTrial:
        """
        Evaluate a parameter set.
        
        Args:
            parameters: Parameter values to evaluate
            trial_id: Optional trial identifier
            
        Returns:
            OptimizationTrial with results
        """
        if trial_id is None:
            trial_id = len(self.trials)
        
        # Clip parameters to valid ranges
        clipped_params = self.parameter_space.clip_all(parameters)
        
        # Evaluate objective
        try:
            objective_value = self.objective_function(clipped_params)
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            objective_value = float('-inf') if self.maximize else float('inf')
        
        # Create trial record
        trial = OptimizationTrial(
            trial_id=trial_id,
            parameters=clipped_params,
            objective_value=objective_value
        )
        
        self.trials.append(trial)
        
        # Update best
        self._update_best(trial)
        
        # Update convergence history
        if self.best_trial:
            self.convergence_history.append(self.best_trial.objective_value)
        
        if self.verbose:
            logger.info(
                f"Trial {trial_id}: objective={objective_value:.4f}, "
                f"best={self.best_trial.objective_value:.4f}"
            )
        
        return trial
    
    def _update_best(self, trial: OptimizationTrial) -> None:
        """Update best trial if this one is better."""
        if self.best_trial is None:
            self.best_trial = trial
        elif self.maximize:
            if trial.objective_value > self.best_trial.objective_value:
                self.best_trial = trial
        else:
            if trial.objective_value < self.best_trial.objective_value:
                self.best_trial = trial
    
    def _create_result(
        self,
        total_time: float,
        baseline_objective: Optional[float] = None
    ) -> OptimizationResult:
        """Create the final optimization result."""
        if self.best_trial is None:
            raise RuntimeError("No trials completed")
        
        return OptimizationResult(
            best_parameters=self.best_trial.parameters,
            best_objective=self.best_trial.objective_value,
            trials=self.trials.copy(),
            optimizer_name=self.name,
            n_iterations=len(self.trials),
            total_time_seconds=total_time,
            convergence_history=self.convergence_history.copy(),
            baseline_objective=baseline_objective
        )
    
    def reset(self) -> None:
        """Reset the optimizer state."""
        self.trials = []
        self.best_trial = None
        self.convergence_history = []
        self.rng = np.random.default_rng(self.random_state)
