"""
Feature Engineering module for ML Trading Research Project.
Generates technical indicators and market regime features.
"""

from features.price_features import (
    calculate_log_returns,
    calculate_rolling_returns,
    calculate_price_momentum,
    calculate_gap_returns,
    PriceFeatureGenerator
)
from features.trend_indicators import (
    calculate_sma,
    calculate_ema,
    calculate_ema_slope,
    calculate_adx,
    TrendIndicatorGenerator
)
from features.momentum_indicators import (
    calculate_rsi,
    calculate_stoch_rsi,
    calculate_macd,
    MomentumIndicatorGenerator
)
from features.volatility_indicators import (
    calculate_atr,
    calculate_rolling_volatility,
    calculate_bollinger_bands,
    VolatilityIndicatorGenerator
)
from features.volume_indicators import (
    calculate_volume_ma,
    calculate_volume_spikes,
    calculate_vwap,
    VolumeIndicatorGenerator
)
from features.regime_features import (
    classify_market_regime,
    detect_volatility_regime,
    RegimeFeatureGenerator
)
from features.indicator_registry import (
    IndicatorSpec,
    get_indicator_registry,
)
from features.feature_engine import FeatureEngine

__all__ = [
    # Price features
    'calculate_log_returns',
    'calculate_rolling_returns',
    'calculate_price_momentum',
    'calculate_gap_returns',
    'PriceFeatureGenerator',
    # Trend indicators
    'calculate_sma',
    'calculate_ema',
    'calculate_ema_slope',
    'calculate_adx',
    'TrendIndicatorGenerator',
    # Momentum indicators
    'calculate_rsi',
    'calculate_stoch_rsi',
    'calculate_macd',
    'MomentumIndicatorGenerator',
    # Volatility indicators
    'calculate_atr',
    'calculate_rolling_volatility',
    'calculate_bollinger_bands',
    'VolatilityIndicatorGenerator',
    # Volume indicators
    'calculate_volume_ma',
    'calculate_volume_spikes',
    'calculate_vwap',
    'VolumeIndicatorGenerator',
    # Regime features
    'classify_market_regime',
    'detect_volatility_regime',
    'RegimeFeatureGenerator',
    'IndicatorSpec',
    'get_indicator_registry',
    # Main engine
    'FeatureEngine'
]
