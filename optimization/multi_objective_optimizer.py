"""
Multi-Objective Optimization (A5 - P1)

Supports Pareto optimization for trading strategies with multiple objectives:
- Sharpe Ratio (maximize)
- Max Drawdown (minimize) 
- Sortino Ratio (maximize)
- Calmar Ratio (maximize)
- Win Rate (maximize)

Users can select objective preference or get full Pareto front.

Backward compatible with single-objective optimizers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
#from deap import base, creator, tools, algorithms

try:
    import optuna
    from optuna.samplers import NSGAIISampler, NSGAIIISampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

from optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationTrial,
    ParameterSpace,
    ParameterType
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectiveType(Enum):
    """Available optimization objectives."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    TOTAL_RETURN = "total_return"
    PROFIT_FACTOR = "profit_factor"
    
    @property
    def direction(self) -> str:
        """Whether to maximize or minimize this objective."""
        minimize_objectives = {
            ObjectiveType.MAX_DRAWDOWN
        }
        return "minimize" if self in minimize_objectives else "maximize"


@dataclass
class ObjectiveConfig:
    """Configuration for a single objective."""
    objective_type: ObjectiveType
    weight: float = 1.0  # For scalarization
    target: Optional[float] = None  # Constraint target
    priority: int = 1  # 1 = primary, 2 = secondary, etc.
    
    @property
    def direction(self) -> str:
        return self.objective_type.direction


@dataclass
class ParetoSolution:
    """A single solution on the Pareto front."""
    parameters: Dict[str, Any]
    objectives: Dict[str, float]  # objective_name -> value
    rank: int = 0  # Pareto rank (0 = front)
    crowding_distance: float = 0.0
    
    def dominates(self, other: 'ParetoSolution', directions: Dict[str, str]) -> bool:
        """Check if this solution dominates another."""
        dominated = False
        strictly_better_in_one = False
        
        for obj_name, direction in directions.items():
            self_val = self.objectives.get(obj_name, float('nan'))
            other_val = other.objectives.get(obj_name, float('nan'))
            
            if np.isnan(self_val) or np.isnan(other_val):
                continue
            
            if direction == "maximize":
                if self_val < other_val:
                    dominated = True
                    break
                elif self_val > other_val:
                    strictly_better_in_one = True
            else:  # minimize
                if self_val > other_val:
                    dominated = True
                    break
                elif self_val < other_val:
                    strictly_better_in_one = True
        
        return not dominated and strictly_better_in_one
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameters,
            "objectives": self.objectives,
            "rank": self.rank,
            "crowding_distance": self.crowding_distance
        }


@dataclass
class MultiObjectiveResult:
    """
    Result of multi-objective optimization.
    
    Contains the full Pareto front and methods to select solutions
    based on user preferences.
    """
    pareto_front: List[ParetoSolution]
    all_solutions: List[ParetoSolution]
    objectives: List[ObjectiveConfig]
    
    # Metadata
    optimizer_name: str = ""
    n_iterations: int = 0
    total_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_best_for_objective(self, objective: ObjectiveType) -> ParetoSolution:
        """Get the best solution for a specific objective."""
        obj_name = objective.value
        direction = objective.direction
        
        sorted_solutions = sorted(
            self.pareto_front,
            key=lambda s: s.objectives.get(obj_name, float('-inf') if direction == "maximize" else float('inf')),
            reverse=(direction == "maximize")
        )
        
        return sorted_solutions[0] if sorted_solutions else None
    
    def get_balanced_solution(self) -> ParetoSolution:
        """Get the solution with highest crowding distance (most balanced)."""
        if not self.pareto_front:
            return None
        
        return max(self.pareto_front, key=lambda s: s.crowding_distance)
    
    def get_weighted_best(
        self,
        weights: Dict[str, float]
    ) -> ParetoSolution:
        """
        Get best solution using weighted scalarization.
        
        Args:
            weights: Objective weights (positive for maximize, negative for minimize)
            
        Returns:
            Best solution according to weighted sum
        """
        def score(sol: ParetoSolution) -> float:
            total = 0.0
            for obj_name, weight in weights.items():
                val = sol.objectives.get(obj_name, 0)
                # Normalize by direction
                obj_type = ObjectiveType(obj_name)
                if obj_type.direction == "minimize":
                    val = -val  # Convert to maximization
                total += weight * val
            return total
        
        if not self.pareto_front:
            return None
        
        return max(self.pareto_front, key=score)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert Pareto front to DataFrame."""
        rows = []
        for sol in self.pareto_front:
            row = {"rank": sol.rank, "crowding_distance": sol.crowding_distance}
            row.update({f"obj_{k}": v for k, v in sol.objectives.items()})
            row.update({f"param_{k}": v for k, v in sol.parameters.items()})
            rows.append(row)
        return pd.DataFrame(rows)
    
    def to_single_objective_result(
        self,
        preference: ObjectiveType
    ) -> OptimizationResult:
        """
        Convert to single-objective result for backward compatibility.
        
        Args:
            preference: Which objective to use as primary
            
        Returns:
            OptimizationResult compatible with existing code
        """
        best = self.get_best_for_objective(preference)
        
        if best is None:
            raise ValueError("No solutions found")
        
        trials = [
            OptimizationTrial(
                trial_id=i,
                parameters=sol.parameters,
                objective_value=sol.objectives.get(preference.value, 0),
                metrics=sol.objectives
            )
            for i, sol in enumerate(self.all_solutions)
        ]
        
        return OptimizationResult(
            best_parameters=best.parameters,
            best_objective=best.objectives.get(preference.value, 0),
            trials=trials,
            optimizer_name=self.optimizer_name,
            n_iterations=self.n_iterations,
            total_time_seconds=self.total_time_seconds
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pareto_front": [s.to_dict() for s in self.pareto_front],
            "n_solutions": len(self.pareto_front),
            "n_total_evaluated": len(self.all_solutions),
            "objectives": [o.objective_type.value for o in self.objectives],
            "optimizer_name": self.optimizer_name,
            "n_iterations": self.n_iterations,
            "total_time_seconds": self.total_time_seconds,
            "timestamp": self.timestamp.isoformat()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-OBJECTIVE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer supporting Pareto optimization.
    
    Typical use cases:
    - Maximize Sharpe Ratio while minimizing Max Drawdown
    - Balance between multiple risk-adjusted return metrics
    - Find trade-offs between return and risk
    
    Usage:
        optimizer = MultiObjectiveOptimizer(
            parameter_space=space,
            objectives=[
                ObjectiveConfig(ObjectiveType.SHARPE_RATIO, weight=1.0),
                ObjectiveConfig(ObjectiveType.MAX_DRAWDOWN, weight=1.0),
            ],
            objective_function=evaluate_strategy,
            n_iterations=100
        )
        
        result = optimizer.optimize()
        
        # Get balanced solution
        best = result.get_balanced_solution()
        
        # Or get best for specific objective
        best_sharpe = result.get_best_for_objective(ObjectiveType.SHARPE_RATIO)
        
        # Convert to single-objective for backward compatibility
        single_result = result.to_single_objective_result(ObjectiveType.SHARPE_RATIO)
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objectives: List[ObjectiveConfig],
        objective_function: Callable[[Dict[str, Any]], Dict[str, float]],
        n_iterations: int = 100,
        population_size: int = 50,
        backend: str = 'optuna',
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            parameter_space: Parameter space to optimize
            objectives: List of objectives to optimize
            objective_function: Function that returns dict of objective values
            n_iterations: Number of optimization iterations
            population_size: Population size for evolutionary algorithms
            backend: 'optuna' or 'deap'
            random_state: Random seed
            verbose: Logging verbosity
        """
        self.parameter_space = parameter_space
        self.objectives = objectives
        self.objective_function = objective_function
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.backend = backend
        self.random_state = random_state
        self.verbose = verbose
        
        # Validate backend
        if backend == 'optuna' and not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to DEAP")
            self.backend = 'deap'
        if self.backend == 'deap' and not DEAP_AVAILABLE:
            raise ImportError("No multi-objective backend available. Install optuna or deap.")
        
        # All evaluated solutions
        self.all_solutions: List[ParetoSolution] = []
    
    def optimize(self) -> MultiObjectiveResult:
        """Run multi-objective optimization."""
        import time
        start_time = time.time()
        
        if self.backend == 'optuna':
            result = self._optimize_optuna()
        else:
            result = self._optimize_deap()
        
        result.total_time_seconds = time.time() - start_time
        
        if self.verbose:
            logger.info(f"Multi-objective optimization complete. "
                       f"Pareto front size: {len(result.pareto_front)}")
        
        return result
    
    def _optimize_optuna(self) -> MultiObjectiveResult:
        """Optimize using Optuna NSGA-II."""
        
        # Create study with multiple objectives
        directions = [
            "maximize" if obj.objective_type.direction == "maximize" else "minimize"
            for obj in self.objectives
        ]
        
        sampler = NSGAIISampler(
            seed=self.random_state,
            population_size=self.population_size
        )
        
        study = optuna.create_study(
            directions=directions,
            sampler=sampler
        )
        
        # Define objective
        def objective(trial: optuna.Trial) -> Tuple:
            params = self._sample_optuna(trial)
            
            try:
                obj_values = self.objective_function(params)
            except Exception as e:
                logger.warning(f"Objective function failed: {e}")
                obj_values = {
                    obj.objective_type.value: float('nan')
                    for obj in self.objectives
                }
            
            # Store solution
            sol = ParetoSolution(
                parameters=params,
                objectives=obj_values
            )
            self.all_solutions.append(sol)
            
            return tuple(
                obj_values.get(obj.objective_type.value, float('nan'))
                for obj in self.objectives
            )
        
        # Optimize
        if self.verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=self.n_iterations)
        
        # Extract Pareto front
        pareto_front = []
        for trial in study.best_trials:
            params = trial.params
            obj_values = {
                self.objectives[i].objective_type.value: trial.values[i]
                for i in range(len(self.objectives))
            }
            pareto_front.append(ParetoSolution(
                parameters=params,
                objectives=obj_values,
                rank=0
            ))
        
        # Compute crowding distances
        self._compute_crowding_distances(pareto_front)
        
        return MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=self.all_solutions,
            objectives=self.objectives,
            optimizer_name="Optuna-NSGA-II",
            n_iterations=self.n_iterations
        )
    
    def _optimize_deap(self) -> MultiObjectiveResult:
        """Optimize using DEAP NSGA-II."""
        
        # Setup DEAP
        n_obj = len(self.objectives)
        weights = tuple(
            1.0 if obj.objective_type.direction == "maximize" else -1.0
            for obj in self.objectives
        )
        
        # Create fitness class dynamically
        if hasattr(creator, "MOFitness"):
            del creator.MOFitness
        if hasattr(creator, "MOIndividual"):
            del creator.MOIndividual
        
        creator.create("MOFitness", base.Fitness, weights=weights)
        creator.create("MOIndividual", list, fitness=creator.MOFitness)
        
        toolbox = base.Toolbox()
        
        # Parameter representations
        param_names = list(self.parameter_space.parameters.keys())
        n_params = len(param_names)
        
        def create_individual():
            ind = []
            for name in param_names:
                spec = self.parameter_space.parameters[name]
                if spec.param_type == ParameterType.INTEGER:
                    val = np.random.randint(int(spec.bounds[0]), int(spec.bounds[1]) + 1)
                elif spec.param_type == ParameterType.FLOAT:
                    val = np.random.uniform(spec.bounds[0], spec.bounds[1])
                elif spec.param_type == ParameterType.CATEGORICAL:
                    val = np.random.choice(len(spec.categories))
                else:
                    val = np.random.choice([0, 1])
                ind.append(val)
            return creator.MOIndividual(ind)
        
        def decode_individual(ind) -> Dict[str, Any]:
            params = {}
            for i, name in enumerate(param_names):
                spec = self.parameter_space.parameters[name]
                if spec.param_type == ParameterType.INTEGER:
                    params[name] = int(round(ind[i]))
                elif spec.param_type == ParameterType.CATEGORICAL:
                    idx = int(round(ind[i])) % len(spec.categories)
                    params[name] = spec.categories[idx]
                else:
                    params[name] = ind[i]
            return params
        
        def evaluate(ind):
            params = decode_individual(ind)
            try:
                obj_values = self.objective_function(params)
            except Exception:
                return tuple(float('nan') for _ in self.objectives)
            
            # Store solution
            sol = ParetoSolution(parameters=params, objectives=obj_values)
            self.all_solutions.append(sol)
            
            return tuple(
                obj_values.get(obj.objective_type.value, float('nan'))
                for obj in self.objectives
            )
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                        eta=20.0, low=0.0, up=1.0)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                        eta=20.0, low=0.0, up=1.0, indpb=1.0/n_params)
        toolbox.register("select", tools.selNSGA2)
        
        # Initialize population
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        population = toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # Run NSGA-II
        n_generations = self.n_iterations // self.population_size
        
        for gen in range(n_generations):
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            for i in range(0, len(offspring), 2):
                if i + 1 < len(offspring):
                    if np.random.random() < 0.9:
                        toolbox.mate(offspring[i], offspring[i+1])
                        del offspring[i].fitness.values
                        del offspring[i+1].fitness.values
            
            for ind in offspring:
                if np.random.random() < 0.2:
                    toolbox.mutate(ind)
                    del ind.fitness.values
            
            # Evaluate offspring
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)
            
            population = toolbox.select(population + offspring, self.population_size)
            
            if self.verbose and gen % 10 == 0:
                logger.info(f"Generation {gen}/{n_generations}")
        
        # Extract Pareto front
        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        pareto_front = []
        for ind in front:
            params = decode_individual(ind)
            obj_values = {
                self.objectives[i].objective_type.value: ind.fitness.values[i]
                for i in range(len(self.objectives))
            }
            pareto_front.append(ParetoSolution(
                parameters=params,
                objectives=obj_values,
                rank=0
            ))
        
        self._compute_crowding_distances(pareto_front)
        
        return MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=self.all_solutions,
            objectives=self.objectives,
            optimizer_name="DEAP-NSGA-II",
            n_iterations=self.n_iterations
        )
    
    def _sample_optuna(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters using Optuna trial."""
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
        
        return params
    
    def _compute_crowding_distances(
        self,
        solutions: List[ParetoSolution]
    ) -> None:
        """Compute crowding distances for solutions."""
        if len(solutions) <= 2:
            for sol in solutions:
                sol.crowding_distance = float('inf')
            return
        
        n = len(solutions)
        obj_names = [obj.objective_type.value for obj in self.objectives]
        
        # Initialize distances
        for sol in solutions:
            sol.crowding_distance = 0.0
        
        for obj_name in obj_names:
            # Sort by objective
            sorted_sols = sorted(
                solutions,
                key=lambda s: s.objectives.get(obj_name, float('inf'))
            )
            
            # Boundary solutions get infinite distance
            sorted_sols[0].crowding_distance = float('inf')
            sorted_sols[-1].crowding_distance = float('inf')
            
            # Get range
            obj_min = sorted_sols[0].objectives.get(obj_name, 0)
            obj_max = sorted_sols[-1].objectives.get(obj_name, 1)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Compute distances for intermediate solutions
            for i in range(1, n - 1):
                prev_val = sorted_sols[i-1].objectives.get(obj_name, 0)
                next_val = sorted_sols[i+1].objectives.get(obj_name, 0)
                sorted_sols[i].crowding_distance += (next_val - prev_val) / obj_range


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_sharpe_vs_drawdown(
    parameter_space: ParameterSpace,
    backtest_function: Callable[[Dict[str, Any]], Dict[str, float]],
    n_iterations: int = 100,
    preference: Optional[str] = None
) -> Union[MultiObjectiveResult, OptimizationResult]:
    """
    Convenience function for common Sharpe vs Drawdown optimization.
    
    Args:
        parameter_space: Parameter space
        backtest_function: Function returning dict with 'sharpe_ratio' and 'max_drawdown'
        n_iterations: Iterations
        preference: If specified ('sharpe' or 'drawdown'), returns single-objective result
        
    Returns:
        MultiObjectiveResult or OptimizationResult if preference specified
    """
    objectives = [
        ObjectiveConfig(ObjectiveType.SHARPE_RATIO, weight=1.0, priority=1),
        ObjectiveConfig(ObjectiveType.MAX_DRAWDOWN, weight=1.0, priority=2),
    ]
    
    optimizer = MultiObjectiveOptimizer(
        parameter_space=parameter_space,
        objectives=objectives,
        objective_function=backtest_function,
        n_iterations=n_iterations
    )
    
    result = optimizer.optimize()
    
    if preference == 'sharpe':
        return result.to_single_objective_result(ObjectiveType.SHARPE_RATIO)
    elif preference == 'drawdown':
        return result.to_single_objective_result(ObjectiveType.MAX_DRAWDOWN)
    else:
        return result


def create_objective_function(
    backtest_engine,
    strategy,
    data: pd.DataFrame,
    objectives: List[ObjectiveType]
) -> Callable[[Dict[str, Any]], Dict[str, float]]:
    """
    Create an objective function from backtest engine.
    
    Args:
        backtest_engine: BacktestEngine instance
        strategy: Strategy instance
        data: Market data
        objectives: List of objectives to compute
        
    Returns:
        Function that maps parameters to objective values
    """
    def objective_function(params: Dict[str, Any]) -> Dict[str, float]:
        # Run backtest
        result = backtest_engine.run(strategy, data, params)
        metrics = result.metrics
        
        # Extract requested objectives
        obj_values = {}
        for obj in objectives:
            if obj == ObjectiveType.SHARPE_RATIO:
                obj_values[obj.value] = metrics.get('sharpe_ratio', 0)
            elif obj == ObjectiveType.MAX_DRAWDOWN:
                obj_values[obj.value] = metrics.get('max_drawdown', 0)
            elif obj == ObjectiveType.SORTINO_RATIO:
                obj_values[obj.value] = metrics.get('sortino_ratio', 0)
            elif obj == ObjectiveType.CALMAR_RATIO:
                obj_values[obj.value] = metrics.get('calmar_ratio', 0)
            elif obj == ObjectiveType.WIN_RATE:
                obj_values[obj.value] = metrics.get('win_rate', 0)
            elif obj == ObjectiveType.TOTAL_RETURN:
                obj_values[obj.value] = metrics.get('total_return', 0)
            elif obj == ObjectiveType.PROFIT_FACTOR:
                obj_values[obj.value] = metrics.get('profit_factor', 0)
        
        return obj_values
    
    return objective_function


if __name__ == "__main__":
    # Example usage
    print("Multi-Objective Optimizer Example")
    print("=" * 50)
    
    # Create simple parameter space
    space = ParameterSpace()
    space.add_integer("rsi_period", 5, 30)
    space.add_float("rsi_oversold", 20.0, 40.0)
    space.add_float("rsi_overbought", 60.0, 80.0)
    
    # Mock objective function
    def mock_objective(params: Dict[str, Any]) -> Dict[str, float]:
        rsi_period = params['rsi_period']
        oversold = params['rsi_oversold']
        
        # Simulated objectives (with trade-off)
        sharpe = np.random.uniform(0.5, 2.0) + (rsi_period - 15) / 50
        drawdown = np.random.uniform(0.05, 0.20) - (oversold - 30) / 500
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max(0.01, drawdown)
        }
    
    # Define objectives
    objectives = [
        ObjectiveConfig(ObjectiveType.SHARPE_RATIO),
        ObjectiveConfig(ObjectiveType.MAX_DRAWDOWN),
    ]
    
    # Run optimization
    optimizer = MultiObjectiveOptimizer(
        parameter_space=space,
        objectives=objectives,
        objective_function=mock_objective,
        n_iterations=50,
        verbose=True
    )
    
    result = optimizer.optimize()
    
    print(f"\nPareto front size: {len(result.pareto_front)}")
    print("\nTop solutions:")
    
    for i, sol in enumerate(result.pareto_front[:5]):
        print(f"  {i+1}. Sharpe: {sol.objectives.get('sharpe_ratio', 0):.3f}, "
              f"Drawdown: {sol.objectives.get('max_drawdown', 0):.3f}")
    
    # Get best for each objective
    best_sharpe = result.get_best_for_objective(ObjectiveType.SHARPE_RATIO)
    best_dd = result.get_best_for_objective(ObjectiveType.MAX_DRAWDOWN)
    balanced = result.get_balanced_solution()
    
    print(f"\nBest for Sharpe: {best_sharpe.parameters}")
    print(f"Best for Drawdown: {best_dd.parameters}")
    print(f"Most balanced: {balanced.parameters}")
    
    # Convert to single-objective for backward compat
    single_result = result.to_single_objective_result(ObjectiveType.SHARPE_RATIO)
    print(f"\nSingle-objective result: best Sharpe = {single_result.best_objective:.3f}")
