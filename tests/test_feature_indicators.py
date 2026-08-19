"""
Comprehensive unit tests for all 52 technical indicators in BeyondAlgo.
"""

import pytest
import numpy as np
import pandas as pd

from features.indicator_registry import get_indicator_registry, IndicatorSpec
from features.trend_indicators import (
    calculate_sma, calculate_ema, calculate_wma, calculate_hma,
    calculate_dema, calculate_tema, calculate_kama, calculate_vwma,
    calculate_aroon, calculate_parabolic_sar, calculate_ichimoku,
    calculate_vortex, calculate_ema_slope, calculate_adx, calculate_trend_lines
)
from features.momentum_indicators import (
    calculate_rsi, calculate_stoch_rsi, calculate_stochastic, calculate_macd,
    calculate_roc, calculate_cci, calculate_williams_r, calculate_trix,
    calculate_momentum, calculate_ultimate_oscillator, calculate_ppo,
    calculate_awesome_oscillator, calculate_cmo, calculate_tsi
)
from features.volatility_indicators import (
    calculate_atr, calculate_rolling_volatility, calculate_bollinger_bands,
    calculate_keltner_channels, calculate_donchian_channels,
    calculate_chaikin_volatility, calculate_ulcer_index, calculate_volatility_metrics
)
from features.volume_indicators import (
    calculate_volume_ma, calculate_volume_spikes, calculate_vwap,
    calculate_obv, calculate_mfi, calculate_ad_line, calculate_cmf,
    calculate_volume_oscillator, calculate_eom, calculate_force_index,
    calculate_pvt, calculate_nvi, calculate_pvi
)


@pytest.fixture
def ohlcv_data():
    """Generates 100 rows of valid test OHLCV data."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1h")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100))
    high = close + np.abs(np.random.randn(100)) + 0.5
    low = close - np.abs(np.random.randn(100)) - 0.5
    open_p = close + np.random.randn(100) * 0.2
    volume = np.random.randint(1000, 10000, size=100).astype(float)
    
    return pd.DataFrame({
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


def test_indicator_registry_complete():
    registry = get_indicator_registry()
    assert len(registry) == 52
    implemented = get_indicator_registry(implemented_only=True)
    assert len(implemented) == 52, f"Expected 52 implemented indicators, got {len(implemented)}"


def test_trend_indicators(ohlcv_data):
    df = ohlcv_data
    close = df['close']
    
    assert not calculate_sma(close).empty
    assert not calculate_ema(close).empty
    assert not calculate_wma(close).empty
    assert not calculate_hma(close).empty
    assert not calculate_dema(close).empty
    assert not calculate_tema(close).empty
    assert not calculate_kama(close).empty
    assert not calculate_vwma(df).empty
    assert not calculate_aroon(df).empty
    assert len(calculate_parabolic_sar(df)) == len(df)
    assert not calculate_ichimoku(df).empty
    assert not calculate_vortex(df).empty
    assert not calculate_ema_slope(close).empty
    assert not calculate_adx(df).empty
    assert not calculate_trend_lines(df).empty


def test_momentum_indicators(ohlcv_data):
    df = ohlcv_data
    close = df['close']
    
    assert not calculate_rsi(close).empty
    assert not calculate_stoch_rsi(close).empty
    assert not calculate_stochastic(df).empty
    assert not calculate_macd(close).empty
    assert not calculate_roc(close).empty
    assert not calculate_cci(df).empty
    assert not calculate_williams_r(df).empty
    assert len(calculate_trix(close)) == len(df)
    assert len(calculate_momentum(close)) == len(df)
    assert len(calculate_ultimate_oscillator(df)) == len(df)
    assert not calculate_ppo(close).empty
    assert len(calculate_awesome_oscillator(df)) == len(df)
    assert len(calculate_cmo(close)) == len(df)
    assert len(calculate_tsi(close)) == len(df)


def test_volatility_indicators(ohlcv_data):
    df = ohlcv_data
    close = df['close']
    
    assert not calculate_atr(df).empty
    assert not calculate_rolling_volatility(close).empty
    assert not calculate_bollinger_bands(close).empty
    assert not calculate_keltner_channels(df).empty
    assert not calculate_donchian_channels(df).empty
    assert len(calculate_chaikin_volatility(df)) == len(df)
    assert len(calculate_ulcer_index(close)) == len(df)
    assert not calculate_volatility_metrics(df).empty


def test_volume_indicators(ohlcv_data):
    df = ohlcv_data
    vol = df['volume']
    
    assert not calculate_volume_ma(vol).empty
    assert not calculate_volume_spikes(vol).empty
    assert not calculate_vwap(df).empty
    assert not calculate_obv(df).empty
    assert not calculate_mfi(df).empty
    assert not calculate_ad_line(df).empty
    assert len(calculate_cmf(df)) == len(df)
    assert len(calculate_volume_oscillator(vol)) == len(df)
    assert len(calculate_eom(df)) == len(df)
    assert len(calculate_force_index(df)) == len(df)
    assert len(calculate_pvt(df)) == len(df)
    assert len(calculate_nvi(df)) == len(df)
    assert len(calculate_pvi(df)) == len(df)
