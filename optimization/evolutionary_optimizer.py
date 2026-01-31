"""
Evolutionary Optimization using DEAP.

Implements genetic algorithms and evolutionary strategies for
parameter optimization. These methods are particularly useful for:
- Multi-objective optimization
- Highly non-convex landscapes
- When gradient information is unavailable

IMPORTANT: Evolution only changes PARAMETERS, not strategy logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
import logging
import time
import random

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


class EvolutionaryOptimizer(BaseOptimizer):
    """
    Evolutionary Algorithm for parameter optimization.
    
    Uses genetic algorithm concepts (selection, crossover, mutation)
    to evolve a population of parameter sets toward the optimum.
    
    Key features:
    - Population-based search (parallel exploration)
    - Natural handling of constraints
    - Can escape local optima via mutation
    - Intuitive parameter interpretation
    
    Algorithm components:
    - Selection: Tournament selection
    - Crossover: Blend crossover (BLX-α) for continuous params
    - Mutation: Gaussian mutation with adaptive step sizes
    
    When Evolutionary works well:
    - Highly multi-modal landscapes
    - When function evaluations are cheap
    - Non-differentiable objectives
    - Discrete/mixed parameter spaces
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 100,
        population_size: int = 50,
        mutation_prob: float = 0.2,
        crossover_prob: float = 0.8,
        tournament_size: int = 3,
        elitism: int = 2,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize Evolutionary Optimizer.
        
        Args:
            parameter_space: Parameter space to optimize
            objective_function: Function to optimize
            maximize: Whether to maximize objective
            n_iterations: Number of generations
            population_size: Size of population
            mutation_prob: Probability of mutation per individual
            crossover_prob: Probability of crossover
            tournament_size: Tournament selection size
            elitism: Number of best individuals to preserve
            random_state: Random seed
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
        
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.elitism = elitism
        
        self.name = f"Evolutionary(pop={population_size})"
        
        if not DEAP_AVAILABLE:
            raise ImportError(
                "DEAP is not available. Install with: pip install deap"
            )
        
        # Setup DEAP
        self._setup_deap()
        
        # Evolution history
        self.generation_stats: List[Dict[str, float]] = []
    
    def _setup_deap(self) -> None:
        """Setup DEAP framework."""
        # Create fitness class (maximize or minimize)
        weight = 1.0 if self.maximize else -1.0
        
        # Clear any existing classes (for repeated optimization)
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(weight,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Get parameter bounds
        self.param_names = list(self.parameter_space.parameters.keys())
        
        # Register individual creation
        def create_individual():
            sample = self.parameter_space.sample_random(self.rng)
            return creator.Individual([sample[name] for name in self.param_names])
        
        self.toolbox.register("individual", create_individual)
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        # Register evaluation
        self.toolbox.register("evaluate", self._evaluate_individual)
        
        # Register genetic operators
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.tournament_size
        )
        
        self.toolbox.register("mate", self._blend_crossover, alpha=0.5)
        self.toolbox.register("mutate", self._gaussian_mutate)
    
    def _evaluate_individual(self, individual: List[Any]) -> Tuple[float]:
        """
        Evaluate an individual (DEAP format).
        
        Args:
            individual: List of parameter values
            
        Returns:
            Tuple containing fitness value
        """
        # Convert to parameter dict
        params = self._individual_to_params(individual)
        
        # Evaluate using base method (records trial)
        trial = self.evaluate(params)
        
        return (trial.objective_value,)
    
    def _individual_to_params(self, individual: List[Any]) -> Dict[str, Any]:
        """Convert DEAP individual to parameter dictionary."""
        params = {}
        for i, name in enumerate(self.param_names):
            spec = self.parameter_space.parameters[name]
            value = individual[i]
            
            # Clip to bounds
            if spec.param_type == ParameterType.INTEGER:
                value = int(round(max(spec.bounds[0], min(spec.bounds[1], value))))
            elif spec.param_type == ParameterType.FLOAT:
                value = max(spec.bounds[0], min(spec.bounds[1], value))
            
            params[name] = value
        
        return params
    
    def _params_to_individual(self, params: Dict[str, Any]) -> List[Any]:
        """Convert parameter dictionary to DEAP individual."""
        return [params[name] for name in self.param_names]
    
    def _blend_crossover(
        self,
        ind1: List[Any],
        ind2: List[Any],
        alpha: float = 0.5
    ) -> Tuple[List[Any], List[Any]]:
        """
        Blend crossover (BLX-α) for continuous parameters.
        
        Creates offspring in the range [min - α*d, max + α*d]
        where d is the distance between parents.
        """
        for i in range(len(ind1)):
            spec = self.parameter_space.parameters[self.param_names[i]]
            
            if spec.param_type in (ParameterType.FLOAT, ParameterType.INTEGER):
                x1, x2 = ind1[i], ind2[i]
                gamma = (1 + 2 * alpha) * random.random() - alpha
                ind1[i] = (1 - gamma) * x1 + gamma * x2
                ind2[i] = gamma * x1 + (1 - gamma) * x2
                
                # Clip to bounds
                ind1[i] = max(spec.bounds[0], min(spec.bounds[1], ind1[i]))
                ind2[i] = max(spec.bounds[0], min(spec.bounds[1], ind2[i]))
                
                if spec.param_type == ParameterType.INTEGER:
                    ind1[i] = int(round(ind1[i]))
                    ind2[i] = int(round(ind2[i]))
            else:
                # Uniform crossover for categorical/boolean
                if random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
        
        return ind1, ind2
    
    def _gaussian_mutate(
        self,
        individual: List[Any],
        mu: float = 0.0,
        sigma_frac: float = 0.1
    ) -> Tuple[List[Any]]:
        """
        Gaussian mutation with adaptive step size.
        
        Args:
            individual: Individual to mutate
            mu: Mean of Gaussian (typically 0)
            sigma_frac: Fraction of range to use as std dev
            
        Returns:
            Tuple containing mutated individual
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_prob:
                spec = self.parameter_space.parameters[self.param_names[i]]
                
                if spec.param_type in (ParameterType.FLOAT, ParameterType.INTEGER):
                    # Gaussian mutation
                    range_size = spec.bounds[1] - spec.bounds[0]
                    sigma = sigma_frac * range_size
                    individual[i] += random.gauss(mu, sigma)
                    
                    # Clip to bounds
                    individual[i] = max(spec.bounds[0], min(spec.bounds[1], individual[i]))
                    
                    if spec.param_type == ParameterType.INTEGER:
                        individual[i] = int(round(individual[i]))
                
                elif spec.param_type == ParameterType.CATEGORICAL:
                    # Random category
                    individual[i] = random.choice(spec.categories)
                
                elif spec.param_type == ParameterType.BOOLEAN:
                    # Flip
                    individual[i] = not individual[i]
        
        return (individual,)
    
    def optimize(
        self,
        baseline_objective: Optional[float] = None
    ) -> OptimizationResult:
        """
        Run evolutionary optimization.
        
        Args:
            baseline_objective: Optional baseline for comparison
            
        Returns:
            OptimizationResult with best parameters
        """
        start_time = time.time()
        
        # Set random seed
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        logger.info("Evaluating initial population...")
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Hall of fame for elitism
        hof = tools.HallOfFame(self.elitism)
        hof.update(population)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logger.info(f"Starting evolution for {self.n_iterations} generations")
        
        # Evolution loop
        for gen in range(self.n_iterations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population) - self.elitism)
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Add elites
            offspring.extend(hof.items)
            
            # Update population
            population[:] = offspring
            
            # Update hall of fame
            hof.update(population)
            
            # Record statistics
            record = stats.compile(population)
            self.generation_stats.append({
                'generation': gen,
                'avg': record['avg'],
                'std': record['std'],
                'min': record['min'],
                'max': record['max']
            })
            
            if self.verbose and (gen + 1) % 10 == 0:
                logger.info(
                    f"Gen {gen+1}: avg={record['avg']:.4f}, "
                    f"best={record['max'] if self.maximize else record['min']:.4f}"
                )
        
        total_time = time.time() - start_time
        
        # Get best individual
        best_ind = hof[0]
        best_params = self._individual_to_params(best_ind)
        
        # Update best trial to ensure consistency
        self.best_trial = OptimizationTrial(
            trial_id=len(self.trials),
            parameters=best_params,
            objective_value=best_ind.fitness.values[0],
            metadata={'source': 'hall_of_fame'}
        )
        
        logger.info(
            f"Evolution complete: best={best_ind.fitness.values[0]:.4f}, "
            f"time={total_time:.2f}s"
        )
        
        return self._create_result(
            total_time=total_time,
            baseline_objective=baseline_objective
        )
    
    def suggest_next(self) -> Dict[str, Any]:
        """Suggest next parameters (random sample)."""
        return self.parameter_space.sample_random(self.rng)
    
    def get_population_diversity(self) -> float:
        """
        Calculate population diversity.
        
        Returns average pairwise distance between individuals.
        """
        if not hasattr(self, 'population') or len(self.population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                d = np.sqrt(sum(
                    (self.population[i][k] - self.population[j][k]) ** 2
                    for k in range(len(self.population[i]))
                ))
                distances.append(d)
        
        return np.mean(distances)
    
    def get_evolution_history(self) -> pd.DataFrame:
        """Get evolution history as DataFrame."""
        return pd.DataFrame(self.generation_stats)


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """
    Differential Evolution optimizer.
    
    A simpler evolutionary algorithm that uses differences between
    individuals for mutation. Often works well for continuous optimization.
    
    Uses scipy.optimize.differential_evolution internally.
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_iterations: int = 100,
        population_size: int = 15,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize Differential Evolution.
        
        Args:
            parameter_space: Parameter space
            objective_function: Function to optimize
            maximize: Whether to maximize
            n_iterations: Maximum iterations
            population_size: Population size multiplier (actual = popsize * ndim)
            mutation: Mutation constant range (min, max)
            recombination: Crossover probability
            random_state: Random seed
            verbose: Verbosity
        """
        super().__init__(
            parameter_space=parameter_space,
            objective_function=objective_function,
            maximize=maximize,
            n_iterations=n_iterations,
            random_state=random_state,
            verbose=verbose
        )
        
        self.population_size = population_size
        self.mutation = mutation
        self.recombination = recombination
        
        self.name = "DifferentialEvolution"
    
    def optimize(
        self,
        baseline_objective: Optional[float] = None
    ) -> OptimizationResult:
        """Run differential evolution."""
        from scipy.optimize import differential_evolution
        
        start_time = time.time()
        
        # Get bounds
        bounds = []
        param_names = list(self.parameter_space.parameters.keys())
        
        for name in param_names:
            spec = self.parameter_space.parameters[name]
            bounds.append(spec.bounds)
        
        # Objective wrapper
        def objective_wrapper(x):
            params = {
                name: x[i] for i, name in enumerate(param_names)
            }
            
            # Handle integer params
            for name in param_names:
                spec = self.parameter_space.parameters[name]
                if spec.param_type == ParameterType.INTEGER:
                    params[name] = int(round(params[name]))
            
            trial = self.evaluate(params)
            
            # scipy minimizes, so negate if maximizing
            return -trial.objective_value if self.maximize else trial.objective_value
        
        # Run optimization
        result = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=self.n_iterations,
            popsize=self.population_size,
            mutation=self.mutation,
            recombination=self.recombination,
            seed=self.random_state,
            disp=self.verbose,
            polish=False  # Skip polishing with L-BFGS-B
        )
        
        total_time = time.time() - start_time
        
        # Convert result
        best_params = {
            name: result.x[i] for i, name in enumerate(param_names)
        }
        
        # Handle integers
        for name in param_names:
            spec = self.parameter_space.parameters[name]
            if spec.param_type == ParameterType.INTEGER:
                best_params[name] = int(round(best_params[name]))
        
        best_objective = -result.fun if self.maximize else result.fun
        
        self.best_trial = OptimizationTrial(
            trial_id=len(self.trials),
            parameters=best_params,
            objective_value=best_objective
        )
        
        return self._create_result(
            total_time=total_time,
            baseline_objective=baseline_objective
        )
    
    def suggest_next(self) -> Dict[str, Any]:
        """Suggest next parameters."""
        return self.parameter_space.sample_random(self.rng)


def create_evolutionary_optimizer_for_strategy(
    strategy_bounds: Dict[str, Tuple[float, float]],
    objective_function: Callable[[Dict[str, Any]], float],
    integer_params: Optional[List[str]] = None,
    n_iterations: int = 50,
    population_size: int = 50,
    use_differential_evolution: bool = False,
    **kwargs
) -> BaseOptimizer:
    """
    Factory function to create an evolutionary optimizer for a strategy.
    
    Args:
        strategy_bounds: Parameter bounds from strategy
        objective_function: Function to evaluate parameters
        integer_params: Names of integer parameters
        n_iterations: Number of generations
        population_size: Population size
        use_differential_evolution: Use DE instead of DEAP
        **kwargs: Additional arguments
        
    Returns:
        Configured optimizer
    """
    default_integer_params = [
        'rsi_lookback', 'ema_fast_period', 'ema_slow_period',
        'bb_period', 'squeeze_lookback', 'min_squeeze_candles',
        'adx_period', 'atr_period'
    ]
    
    if integer_params is None:
        integer_params = default_integer_params
    
    space = ParameterSpace.from_strategy_bounds(strategy_bounds, integer_params)
    
    if use_differential_evolution:
        return DifferentialEvolutionOptimizer(
            parameter_space=space,
            objective_function=objective_function,
            n_iterations=n_iterations,
            population_size=population_size,
            **kwargs
        )
    else:
        return EvolutionaryOptimizer(
            parameter_space=space,
            objective_function=objective_function,
            n_iterations=n_iterations,
            population_size=population_size,
            **kwargs
        )
