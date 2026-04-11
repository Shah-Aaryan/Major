"""
Central registry for technical indicators used by BeyondAlgo.

Phase 1 goal: declare the full 50-indicator surface area while
flagging which ones are currently implemented. This keeps the engine
safe (only runs implemented functions) and testable while providing a
clear roadmap for the remaining indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from features.trend_indicators import (
    calculate_sma,
    calculate_ema,
    calculate_ema_slope,
    calculate_adx,
    calculate_trend_lines,
)
from features.momentum_indicators import (
    calculate_rsi,
    calculate_stoch_rsi,
    calculate_macd,
    calculate_roc,
    calculate_cci,
    calculate_williams_r,
)
from features.volatility_indicators import (
    calculate_atr,
    calculate_rolling_volatility,
    calculate_bollinger_bands,
    calculate_keltner_channels,
    calculate_donchian_channels,
    calculate_volatility_metrics,
)
from features.volume_indicators import (
    calculate_volume_ma,
    calculate_volume_spikes,
    calculate_vwap,
    calculate_obv,
    calculate_mfi,
    calculate_ad_line,
)


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    category: str
    implemented: bool
    func: Optional[Callable] = None
    notes: str = ""
    tags: Optional[List[str]] = None


# NOTE: Keep this list exactly at 52 entries.
_INDICATORS: List[IndicatorSpec] = [
    IndicatorSpec("Simple Moving Average (SMA)", "trend", True, calculate_sma),
    IndicatorSpec("Exponential Moving Average (EMA)", "trend", True, calculate_ema),
    IndicatorSpec("Weighted Moving Average (WMA)", "trend", False),
    IndicatorSpec("Hull Moving Average (HMA)", "trend", False),
    IndicatorSpec("Double Exponential Moving Average (DEMA)", "trend", False),
    IndicatorSpec("Triple Exponential Moving Average (TEMA)", "trend", False),
    IndicatorSpec("Kaufman Adaptive Moving Average (KAMA)", "trend", False),
    IndicatorSpec("Volume Weighted Moving Average (VWMA)", "trend", False),
    IndicatorSpec("Moving Average Convergence Divergence (MACD)", "momentum", True, calculate_macd),
    IndicatorSpec("Average Directional Index (ADX)", "trend", True, calculate_adx),
    IndicatorSpec("Aroon", "trend", False),
    IndicatorSpec("Parabolic SAR", "trend", False),
    IndicatorSpec("Ichimoku Cloud", "trend", False),
    IndicatorSpec("Vortex Indicator", "trend", False),
    IndicatorSpec("TRIX", "momentum", False),
    IndicatorSpec("Relative Strength Index (RSI)", "momentum", True, calculate_rsi),
    IndicatorSpec("Stochastic Oscillator", "momentum", False),
    IndicatorSpec("Stochastic RSI", "momentum", True, calculate_stoch_rsi),
    IndicatorSpec("Williams %R", "momentum", True, calculate_williams_r),
    IndicatorSpec("Commodity Channel Index (CCI)", "momentum", True, calculate_cci),
    IndicatorSpec("Rate of Change (ROC)", "momentum", True, calculate_roc),
    IndicatorSpec("Momentum (MOM)", "momentum", False),
    IndicatorSpec("Ultimate Oscillator", "momentum", False),
    IndicatorSpec("Percentage Price Oscillator (PPO)", "momentum", False),
    IndicatorSpec("Awesome Oscillator (AO)", "momentum", False),
    IndicatorSpec("Chande Momentum Oscillator (CMO)", "momentum", False),
    IndicatorSpec("True Strength Index (TSI)", "momentum", False),
    IndicatorSpec("Average True Range (ATR)", "volatility", True, calculate_atr),
    IndicatorSpec("Bollinger Bands", "volatility", True, calculate_bollinger_bands),
    IndicatorSpec("Keltner Channels", "volatility", True, calculate_keltner_channels),
    IndicatorSpec("Donchian Channels", "volatility", True, calculate_donchian_channels),
    IndicatorSpec("Rolling Standard Deviation", "volatility", True, calculate_rolling_volatility),
    IndicatorSpec("Historical Volatility (Close-to-Close)", "volatility", True, calculate_rolling_volatility),
    IndicatorSpec("Chaikin Volatility", "volatility", False),
    IndicatorSpec("Ulcer Index", "volatility", False),
    IndicatorSpec("Parkinson Volatility", "volatility", True, calculate_volatility_metrics),
    IndicatorSpec("Garman-Klass Volatility", "volatility", True, calculate_volatility_metrics),
    IndicatorSpec("Yang-Zhang Volatility", "volatility", True, calculate_volatility_metrics),
    IndicatorSpec("On-Balance Volume (OBV)", "volume", True, calculate_obv),
    IndicatorSpec("Volume Weighted Average Price (VWAP)", "volume", True, calculate_vwap),
    IndicatorSpec("Money Flow Index (MFI)", "volume", True, calculate_mfi),
    IndicatorSpec("Chaikin Money Flow (CMF)", "volume", False),
    IndicatorSpec("Accumulation/Distribution Line (ADL)", "volume", True, calculate_ad_line),
    IndicatorSpec("Volume Oscillator", "volume", False),
    IndicatorSpec("Ease of Movement (EOM)", "volume", False),
    IndicatorSpec("Force Index", "volume", False),
    IndicatorSpec("Price Volume Trend (PVT)", "volume", False),
    IndicatorSpec("Negative Volume Index (NVI)", "volume", False),
    IndicatorSpec("Positive Volume Index (PVI)", "volume", False),
    IndicatorSpec("Volume Profile (POC/VAH/VAL)", "volume", False),
    IndicatorSpec("Trend Lines (Linear Regression)", "trend", True, calculate_trend_lines),
    IndicatorSpec("EMA Slope", "trend", True, calculate_ema_slope),
]

assert len(_INDICATORS) == 52, "Indicator registry must contain exactly 52 entries"


def get_indicator_registry(implemented_only: bool = False) -> List[IndicatorSpec]:
    """Return the indicator registry, optionally filtering to implemented ones."""
    if implemented_only:
        return [spec for spec in _INDICATORS if spec.implemented]
    return list(_INDICATORS)
    