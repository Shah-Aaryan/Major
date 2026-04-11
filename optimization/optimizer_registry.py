"""
Optimizer registry skeleton for BeyondAlgo.

Phase 1: Declare all 15 planned optimization techniques and mark which are
implemented today. This keeps integration safe while making the roadmap
explicit for upcoming methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

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


@dataclass(frozen=True)
class OptimizerSpec:
    key: str  # machine-friendly identifier
    name: str
    category: str
    status: str  # implemented | planned
    cls: Optional[Type] = None
    supports_multiobjective: bool = False
    notes: str = ""


# NOTE: Keep this list at exactly 15 entries.
_OPTIMIZERS = [
    OptimizerSpec("random_search", "Random Search", "stochastic", "implemented", RandomSearchOptimizer),
    OptimizerSpec("grid_search", "Grid Search", "exhaustive", "implemented", GridSearchOptimizer),
    OptimizerSpec("latin_hypercube", "Latin Hypercube Search", "space-filling", "implemented", RandomSearchOptimizer),
    OptimizerSpec("sobol", "Sobol Sequence Search", "space-filling", "implemented", RandomSearchOptimizer),
    OptimizerSpec("bayesian_gp", "Bayesian Optimization (GP)", "model-based", "implemented", BayesianOptimizer, False, "GP surrogate (EI/PI/LCB)"),
    OptimizerSpec("bayesian_tpe", "Bayesian Optimization (TPE)", "model-based", "implemented", BayesianOptimizer, False, "TPE via Optuna"),
    OptimizerSpec("cma_es", "CMA-ES", "evolutionary", "implemented", CMAESOptimizer),
    OptimizerSpec("differential_evolution", "Differential Evolution", "evolutionary", "implemented", DifferentialEvolutionOptimizer),
    OptimizerSpec("particle_swarm", "Particle Swarm Optimization", "swarm", "implemented", ParticleSwarmOptimizer),
    OptimizerSpec("simulated_annealing", "Simulated Annealing", "trajectory", "implemented", SimulatedAnnealingOptimizer),
    OptimizerSpec("genetic_algorithm", "Genetic Algorithm", "evolutionary", "implemented", EvolutionaryOptimizer, False, "GA via DEAP backend"),
    OptimizerSpec("evolution_strategies", "Evolution Strategies ((mu, lambda)-ES)", "evolutionary", "implemented", EvolutionStrategiesOptimizer),
    OptimizerSpec("nsga_ii", "NSGA-II", "multi-objective", "implemented", MultiObjectiveOptimizer, True),
    OptimizerSpec("nsga_iii", "NSGA-III", "many-objective", "implemented", MultiObjectiveOptimizer, True),
    OptimizerSpec("hyperband_asha", "Hyperband + ASHA", "multi-fidelity", "implemented", HyperbandASHAOptimizer),
]

assert len(_OPTIMIZERS) == 15, "Optimizer registry must contain exactly 15 entries"


def get_optimizer_registry(status: Optional[str] = None):
    """Return optimizer registry, optionally filtered by status."""
    if status is None:
        return list(_OPTIMIZERS)
    return [spec for spec in _OPTIMIZERS if spec.status == status]
