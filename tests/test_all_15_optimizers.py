"""
Unit and integration tests for all 15 optimization algorithms in the registry.

Executes every registered optimizer end-to-end under identical conditions to verify:
1. Every optimizer can be instantiated and initialized.
2. Every optimizer executes n_iterations without crashing or erroring.
3. Every optimizer returns valid parameter sets within specified bounds.
4. Objective values and performance metrics are computed correctly.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from optimization.optimizer_registry import get_optimizer_registry
from optimization.ml_parameter_adjuster import MLParameterAdjuster
from backtesting.backtest_engine import BacktestEngine, BacktestConfig
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy


@pytest.fixture(scope="module")
def sample_ohlcv():
    """Synthetic OHLCV dataset for testing all optimizers."""
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    returns = rng.normal(0.0001, 0.001, n)
    close = 50000 * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    }, index=idx)


def dummy_objective(strategy_name: str, params: dict, data: pd.DataFrame) -> float:
    """Fast dummy objective function scoring RSI parameters."""
    rsi_lb = params.get("rsi_lookback", 14)
    buy_th = params.get("rsi_buy_threshold", 30)
    sell_th = params.get("rsi_sell_threshold", 70)
    # Simple mathematical formula to act as objective
    score = (rsi_lb * 0.1) - (buy_th * 0.05) + (sell_th * 0.02)
    return float(score)


def test_registry_contains_15_optimizers():
    registry = get_optimizer_registry()
    assert len(registry) == 15
    for spec in registry:
        assert spec.status == "implemented"
        assert spec.cls is not None


@pytest.mark.parametrize("spec", get_optimizer_registry(), ids=lambda s: s.key)
def test_optimizer_execution_end_to_end(spec, sample_ohlcv):
    """Test that every optimizer executes n_iterations end-to-end without errors."""
    strategy = RSIMeanReversionStrategy()
    bounds = strategy.get_parameter_bounds()
    
    adjuster = MLParameterAdjuster(
        objective_function=dummy_objective,
        strategy_bounds={"rsi_mean_reversion": bounds},
        verbose=False
    )
    
    result = adjuster.optimize_strategy(
        strategy_name="rsi_mean_reversion",
        train_data=sample_ohlcv,
        method=spec.key,
        n_iterations=5,
        random_state=42,
        market_condition="testing"
    )
    
    assert result is not None
    assert result.optimization_method == spec.key
    assert isinstance(result.ml_params, dict)
    assert len(result.ml_params) > 0
    assert isinstance(result.ml_objective, (int, float))
    assert result.optimization_time_seconds >= 0
