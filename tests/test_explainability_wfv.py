"""
Unit tests for Phase 3: Strategy Reliability & Explainability.

Tests:
- ExplainabilityEngine: explain(), regime_breakdown(), history_df(), edge cases
- NestedWalkForwardValidator: run() with a synthetic dataset
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from analysis.explainability_engine import ExplainabilityEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def human_params():
    return {"rsi_buy_threshold": 30.0, "rsi_sell_threshold": 70.0, "rsi_lookback": 14}


@pytest.fixture
def ml_params_better():
    return {"rsi_buy_threshold": 22.0, "rsi_sell_threshold": 75.0, "rsi_lookback": 20}


@pytest.fixture
def ml_params_worse():
    return {"rsi_buy_threshold": 35.0, "rsi_sell_threshold": 65.0, "rsi_lookback": 10}


# ===========================================================================
# ExplainabilityEngine
# ===========================================================================


class TestExplainabilityEngine:
    def test_explain_returns_report(self, human_params, ml_params_better):
        engine = ExplainabilityEngine()
        report = engine.explain(
            human_params=human_params,
            ml_params=ml_params_better,
            human_sharpe=0.8,
            ml_sharpe=1.4,
            regime="volatile",
        )
        assert report.ml_helped is True
        assert abs(report.sharpe_delta - 0.6) < 1e-6
        assert len(report.parameter_shifts) > 0
        assert report.narrative != ""

    def test_explain_ml_hurts(self, human_params, ml_params_worse):
        engine = ExplainabilityEngine()
        report = engine.explain(
            human_params=human_params,
            ml_params=ml_params_worse,
            human_sharpe=1.2,
            ml_sharpe=0.5,
            regime="ranging",
        )
        assert report.ml_helped is False
        assert report.sharpe_delta < 0

    def test_explain_with_feature_context(self, human_params, ml_params_better):
        engine = ExplainabilityEngine()
        report = engine.explain(
            human_params=human_params,
            ml_params=ml_params_better,
            human_sharpe=0.9,
            ml_sharpe=1.3,
            regime="trending_bullish",
            feature_context={"rsi_14": 25.0, "atr_14": 3.5, "bb_width": 0.04},
        )
        assert report.feature_context["rsi_14"] == 25.0
        # At least one reasoning string should mention context values
        has_context = any("25.0" in s.reasoning or "3.5" in s.reasoning
                          for s in report.parameter_shifts)
        assert has_context or len(report.parameter_shifts) == 0  # lenient

    def test_report_to_dict(self, human_params, ml_params_better):
        engine = ExplainabilityEngine()
        report = engine.explain(human_params, ml_params_better, 0.8, 1.4, "volatile")
        d = report.to_dict()
        assert "regime" in d
        assert "ml_helped" in d
        assert "parameter_shifts" in d
        assert "narrative" in d

    def test_regime_breakdown_accumulates_history(self, human_params, ml_params_better, ml_params_worse):
        engine = ExplainabilityEngine()
        # Record 3 cycles in 2 regimes
        engine.explain(human_params, ml_params_better, 0.8, 1.4, "volatile")
        engine.explain(human_params, ml_params_worse, 1.2, 0.5, "ranging")
        engine.explain(human_params, ml_params_better, 0.9, 1.1, "volatile")

        breakdown = engine.regime_breakdown()
        regimes = {rb.regime for rb in breakdown}
        assert "volatile" in regimes
        assert "ranging" in regimes

        volatile_rb = next(rb for rb in breakdown if rb.regime == "volatile")
        assert volatile_rb.n_cycles == 2
        assert volatile_rb.n_helped == 2
        assert volatile_rb.help_rate == 1.0

    def test_regime_breakdown_df(self, human_params, ml_params_better):
        engine = ExplainabilityEngine()
        engine.explain(human_params, ml_params_better, 0.8, 1.4, "volatile")
        df = engine.regime_breakdown_df()
        assert isinstance(df, pd.DataFrame)
        assert "regime" in df.columns
        assert "help_rate" in df.columns

    def test_history_df(self, human_params, ml_params_better):
        engine = ExplainabilityEngine()
        engine.explain(human_params, ml_params_better, 0.8, 1.4, "ranging")
        engine.explain(human_params, ml_params_better, 0.9, 1.0, "volatile")
        df = engine.history_df()
        assert len(df) == 2

    def test_no_change_produces_empty_shifts(self, human_params):
        engine = ExplainabilityEngine()
        # Same params → no shifts
        report = engine.explain(human_params, human_params, 1.0, 1.0, "ranging")
        assert report.parameter_shifts == []

    def test_custom_param_descriptions(self, human_params, ml_params_better):
        engine = ExplainabilityEngine(
            param_descriptions={"rsi_buy_threshold": "Custom RSI buy label"}
        )
        report = engine.explain(human_params, ml_params_better, 0.8, 1.4, "volatile")
        rsi_buy_shift = next(
            (s for s in report.parameter_shifts if s.param_name == "rsi_buy_threshold"), None
        )
        assert rsi_buy_shift is not None
        assert "Custom RSI buy label" in rsi_buy_shift.reasoning


# ===========================================================================
# NestedWalkForwardValidator (lightweight smoke-test only)
# ===========================================================================


def _make_synthetic_features(n: int = 600) -> pd.DataFrame:
    """Simple OHLCV + a single feature column for fast smoke-tests."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    ret = rng.normal(0.0001, 0.002, n)
    close = 50_000 * np.cumprod(1 + ret)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
            "rsi_14": rng.uniform(20, 80, n),
        },
        index=idx,
    )


def test_nested_walk_forward_smoke():
    """Smoke-test: NestedWalkForwardValidator runs without exception."""
    from backtesting.backtest_engine import BacktestConfig
    from backtesting.nested_walk_forward import NestedWalkForwardValidator
    from optimization.base_optimizer import ParameterSpace
    from optimization.random_search import RandomSearchOptimizer as RandomSearch
    from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

    features = _make_synthetic_features(600)
    strategy = RSIMeanReversionStrategy()
    bounds = strategy.get_parameter_bounds()
    space = ParameterSpace.from_strategy_bounds(bounds)

    def optimizer_factory(obj_fn, s):
        return RandomSearch(parameter_space=s, objective_function=obj_fn, n_iterations=5)

    validator = NestedWalkForwardValidator(
        strategy=strategy,
        inner_optimizer_factory=optimizer_factory,
        parameter_space=space,
        backtest_config=BacktestConfig(initial_capital=10_000),
    )

    result = validator.run(features, n_outer=2, n_inner=2, min_fold_rows=30)
    assert len(result.outer_folds) >= 1
    assert result.ml_consistency >= 0.0
    assert result.deflated_sharpe_ratio is not None
    assert result.pbo_estimate >= 0.0
    assert result.summary() != ""
    assert isinstance(result.to_dict(), dict)
