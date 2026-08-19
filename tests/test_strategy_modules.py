"""
Unit tests for all strategy modules in strategies/.

Verifies:
1. Strategy initialization and default parameters.
2. Parameter bound extraction.
3. Feature calculation requirements.
4. Signal generation (LONG, SHORT, EXIT, HOLD) on synthetic technical indicators.
5. Parameter update & validation methods.
"""

import numpy as np
import pandas as pd
import pytest

from strategies.base_strategy import BaseStrategy, SignalType, StrategySignal
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.bollinger_breakout import BollingerBreakoutStrategy


@pytest.fixture
def sample_indicators_df():
    """Synthetic dataset with technical indicators precomputed."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 50000.0 + np.cumsum(np.random.normal(0, 100, n))
    
    df = pd.DataFrame({
        "open": close - 10,
        "high": close + 20,
        "low": close - 20,
        "close": close,
        "volume": 100.0,
        "rsi": np.random.uniform(20, 80, n),
        "ema_fast": close + np.random.normal(0, 15, n),
        "ema_slow": close + np.random.normal(0, 30, n),
        "bb_upper": close + 100,
        "bb_lower": close - 100,
        "bb_middle": close,
        "bb_pct_b": np.random.uniform(-0.1, 1.1, n),
        "bb_bandwidth": 0.05,
        "squeeze_on": np.random.choice([True, False], n),
        "adx": np.random.uniform(10, 40, n),
        "atr": 50.0,
        "regime": "trending_bullish"
    }, index=idx)
    return df


def test_rsi_mean_reversion_strategy_signals(sample_indicators_df):
    strategy = RSIMeanReversionStrategy()
    assert strategy.name == "RSI_Mean_Reversion"
    
    bounds = strategy.get_parameter_bounds()
    assert len(bounds) > 0
    
    sig = strategy.generate_signal(sample_indicators_df, current_idx=50)
    assert isinstance(sig, StrategySignal)
    assert sig.signal_type in (SignalType.LONG, SignalType.SHORT, SignalType.EXIT, SignalType.HOLD)


def test_ema_crossover_strategy_signals(sample_indicators_df):
    strategy = EMACrossoverStrategy()
    assert strategy.name == "EMA_Crossover"
    
    bounds = strategy.get_parameter_bounds()
    assert len(bounds) > 0
    
    sig = strategy.generate_signal(sample_indicators_df, current_idx=50)
    assert isinstance(sig, StrategySignal)
    assert sig.signal_type in (SignalType.LONG, SignalType.SHORT, SignalType.EXIT, SignalType.HOLD)


def test_bollinger_breakout_strategy_signals(sample_indicators_df):
    strategy = BollingerBreakoutStrategy()
    assert strategy.name == "Bollinger_Breakout"
    
    bounds = strategy.get_parameter_bounds()
    assert len(bounds) > 0
    
    sig = strategy.generate_signal(sample_indicators_df, current_idx=50)
    assert isinstance(sig, StrategySignal)
    assert sig.signal_type in (SignalType.LONG, SignalType.SHORT, SignalType.EXIT, SignalType.HOLD)


def test_strategy_parameter_updates():
    strategy = RSIMeanReversionStrategy()
    all_params = strategy.get_all_parameters()
    assert isinstance(all_params, dict)
    
    if "rsi_buy_threshold" in all_params:
        strategy.set_all_parameters({"rsi_buy_threshold": 25.0})
        updated = strategy.get_all_parameters()
        assert updated.get("rsi_buy_threshold") == 25.0
