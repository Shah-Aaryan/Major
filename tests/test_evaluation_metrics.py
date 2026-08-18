"""Unit tests for evaluation.research_metrics."""

import numpy as np
import pandas as pd
import pytest

from evaluation.research_metrics import (
    build_optimizer_comparison_table,
    calculate_optimizer_convergence,
    calculate_parameter_stability,
    calculate_rolling_accuracy,
)


class TestParameterStability:
    def test_empty_history_returns_empty_dict(self):
        assert calculate_parameter_stability([]) == {}

    def test_constant_parameter_scores_perfect_stability(self):
        history = [{"rsi_period": 14} for _ in range(5)]
        result = calculate_parameter_stability(history)
        assert result["rsi_period"] == pytest.approx(1.0)
        assert result["overall"] == pytest.approx(1.0)

    def test_volatile_parameter_scores_lower_than_stable_one(self):
        history = [
            {"stable": 10.0, "volatile": 5.0},
            {"stable": 10.0, "volatile": 50.0},
            {"stable": 10.0, "volatile": 2.0},
            {"stable": 10.0, "volatile": 80.0},
        ]
        result = calculate_parameter_stability(history)
        assert result["stable"] > result["volatile"]

    def test_zero_mean_nonzero_std_scores_zero(self):
        history = [{"p": 1.0}, {"p": -1.0}]
        result = calculate_parameter_stability(history)
        assert result["p"] == 0.0


class TestOptimizerConvergence:
    def test_empty_trials(self):
        result = calculate_optimizer_convergence([])
        assert result["converged"] is False
        assert result["total_trials"] == 0

    def test_flat_objective_converges_quickly(self):
        objectives = [1.0] * 30
        result = calculate_optimizer_convergence(objectives, tolerance=1e-6, patience=5)
        assert result["converged"] is True
        assert result["convergence_trial"] == 0

    def test_strictly_improving_objective_may_not_converge(self):
        objectives = [float(i) for i in range(30)]
        result = calculate_optimizer_convergence(objectives, tolerance=1e-6, patience=5)
        assert result["converged"] is False
        assert result["convergence_speed"] == 1.0

    def test_running_best_is_monotonically_non_decreasing(self):
        objectives = [5.0, 3.0, 8.0, 2.0, 9.0, 1.0]
        result = calculate_optimizer_convergence(objectives, patience=2)
        curve = result["improvement_curve"]
        assert all(curve[i] <= curve[i + 1] for i in range(len(curve) - 1))
        assert curve[-1] == 9.0


class TestRollingAccuracy:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            calculate_rolling_accuracy(["a", "b"], ["a"])

    def test_all_correct_gives_accuracy_one(self):
        preds = ["ml"] * 10
        actuals = ["ml"] * 10
        result = calculate_rolling_accuracy(preds, actuals, window=5)
        assert (result == 1.0).all()

    def test_all_wrong_gives_accuracy_zero(self):
        preds = ["ml"] * 10
        actuals = ["human"] * 10
        result = calculate_rolling_accuracy(preds, actuals, window=5)
        assert (result == 0.0).all()

    def test_mixed_predictions_within_bounds(self):
        preds = ["ml", "human"] * 10
        actuals = ["ml", "ml"] * 10
        result = calculate_rolling_accuracy(preds, actuals, window=4)
        assert result.between(0.0, 1.0).all()


class TestOptimizerComparisonTable:
    def test_empty_results(self):
        table = build_optimizer_comparison_table([])
        assert table.empty

    def test_aggregates_mean_std_count_per_optimizer_and_regime(self):
        results = [
            {"optimizer": "bayesian", "regime": "trending", "objective_value": 1.0},
            {"optimizer": "bayesian", "regime": "trending", "objective_value": 3.0},
            {"optimizer": "random", "regime": "sideways", "objective_value": 0.5},
        ]
        table = build_optimizer_comparison_table(results)

        bayesian_row = table[(table["optimizer"] == "bayesian") & (table["regime"] == "trending")].iloc[0]
        assert bayesian_row["mean"] == pytest.approx(2.0)
        assert bayesian_row["count"] == 2
        assert bayesian_row["best"] == pytest.approx(3.0)

    def test_missing_regime_defaults_to_all(self):
        results = [{"optimizer": "pso", "objective_value": 1.5}]
        table = build_optimizer_comparison_table(results)
        assert table.iloc[0]["regime"] == "all"

    def test_sorted_by_mean_descending(self):
        results = [
            {"optimizer": "a", "regime": "r", "objective_value": 1.0},
            {"optimizer": "b", "regime": "r", "objective_value": 5.0},
        ]
        table = build_optimizer_comparison_table(results)
        assert list(table["mean"]) == sorted(table["mean"], reverse=True)
