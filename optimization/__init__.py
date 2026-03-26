"""
Optimization module for ML-assisted parameter tuning.

This module provides ML optimization methods to adjust strategy parameters.
IMPORTANT: ML does NOT invent strategies. It ONLY adjusts parameters
of human-defined trading rules.

Available optimizers:
- Bayesian Optimization (scikit-optimize / Optuna)
- Random Search (baseline comparison)
- Evolutionary Strategy (DEAP-based)
"""

from optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    ParameterSpace
)
from optimization.bayesian_optimizer import BayesianOptimizer
from optimization.random_search import RandomSearchOptimizer
from optimization.evolutionary_optimizer import EvolutionaryOptimizer
from optimization.ml_parameter_adjuster import MLParameterAdjuster
from optimization.optimizer_registry import (
    OptimizerSpec,
    get_optimizer_registry,
)

__all__ = [
    'BaseOptimizer',
    'OptimizationResult',
    'ParameterSpace',
    'BayesianOptimizer',
    'RandomSearchOptimizer',
    'EvolutionaryOptimizer',
    'MLParameterAdjuster',
    'OptimizerSpec',
    'get_optimizer_registry'
]
