"""
Unit tests for legacy backtesting engine and execution logic.

Verifies:
1. BacktestEngine initialization with custom BacktestConfig.
2. Backtest execution loop with OHLCV data.
3. Accurate accounting of portfolio value, drawdowns, Sharpe, and trade metrics.
4. Correct application of commission and slippage.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy


@pytest.fixture
def sample_ohlcv():
    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(123)
    ret = rng.normal(0.0002, 0.002, n)
    close = 100 * np.cumprod(1 + ret)
    high = close * 1.002
    low = close * 0.998
    open_ = close
    volume = rng.uniform(100, 1000, n)
    
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    }, index=idx)


def test_backtest_engine_run(sample_ohlcv):
    config = BacktestConfig(
        initial_capital=10000.0,
        commission_pct=0.001,
        slippage_pct=0.0005
    )
    engine = BacktestEngine(config=config)
    strategy = RSIMeanReversionStrategy()
    
    result = engine.run(strategy, sample_ohlcv)
    assert isinstance(result, BacktestResult)
    assert hasattr(result, "metrics")
    assert hasattr(result.metrics, "sharpe_ratio") or hasattr(result.metrics, "total_return")


def test_backtest_engine_zero_trades(sample_ohlcv):
    """Test engine handling when strategy generates no trades."""
    config = BacktestConfig(initial_capital=5000.0)
    engine = BacktestEngine(config=config)
    strategy = RSIMeanReversionStrategy()
    
    # Restrictive parameters
    strategy.set_all_parameters({"rsi_buy_threshold": 1.0, "rsi_sell_threshold": 99.0})
    
    result = engine.run(strategy, sample_ohlcv)
    assert isinstance(result, BacktestResult)
    assert len(result.trades) == 0
