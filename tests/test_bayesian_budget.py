"""
Unit tests for Bayesian optimizer budget handling.
"""

import pytest
import numpy as np
from optimization.base_optimizer import ParameterSpace, ParameterType
from optimization.bayesian_optimizer import BayesianOptimizer


def dummy_obj(params: dict) -> float:
    return float(params.get("x", 0.0) ** 2)


def test_bayesian_budget_scaling():
    space = ParameterSpace()
    space.add_float("x", -5.0, 5.0)

    # Test small iteration budget
    opt_small = BayesianOptimizer(
        parameter_space=space,
        objective_function=dummy_obj,
        n_iterations=5,
        n_initial_points=10,
        verbose=False
    )
    assert opt_small.n_initial_points < 5
    assert opt_small.n_initial_points >= 1

    result_small = opt_small.optimize()
    assert len(opt_small.trials) == 5

    # Test medium iteration budget
    opt_med = BayesianOptimizer(
        parameter_space=space,
        objective_function=dummy_obj,
        n_iterations=20,
        n_initial_points=10,
        verbose=False
    )
    assert opt_med.n_initial_points <= 10
    result_med = opt_med.optimize()
    assert len(opt_med.trials) == 20
