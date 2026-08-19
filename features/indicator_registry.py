"""
Central registry for technical indicators used by BeyondAlgo.

Contains specifications for all 52 technical indicators across trend, momentum,
volatility, and volume categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from features.trend_indicators import (
    calculate_sma,
    calculate_ema,
    calculate_wma,
    calculate_hma,
    calculate_dema,
    calculate_tema,
    calculate_kama,
    calculate_vwma,
    calculate_ema_slope,
    calculate_adx,
    calculate_aroon,
    calculate_parabolic_sar,
    calculate_ichimoku,
    calculate_vortex,
    calculate_trend_lines,
)
from features.momentum_indicators import (
    calculate_rsi,
    calculate_stoch_rsi,
    calculate_stochastic,
    calculate_macd,
    calculate_roc,
    calculate_cci,
    calculate_williams_r,
    calculate_trix,
    calculate_momentum,
    calculate_ultimate_oscillator,
    calculate_ppo,
    calculate_awesome_oscillator,
    calculate_cmo,
    calculate_tsi,
)
from features.volatility_indicators import (
    calculate_atr,
    calculate_rolling_volatility,
    calculate_bollinger_bands,
    calculate_keltner_channels,
    calculate_donchian_channels,
    calculate_chaikin_volatility,
    calculate_ulcer_index,
    calculate_volatility_metrics,
)
from features.volume_indicators import (
    calculate_volume_ma,
    calculate_volume_spikes,
    calculate_vwap,
    calculate_obv,
    calculate_mfi,
    calculate_ad_line,
    calculate_cmf,
    calculate_volume_oscillator,
    calculate_eom,
    calculate_force_index,
    calculate_pvt,
    calculate_nvi,
    calculate_pvi,
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
    IndicatorSpec("Weighted Moving Average (WMA)", "trend", True, calculate_wma),
    IndicatorSpec("Hull Moving Average (HMA)", "trend", True, calculate_hma),
    IndicatorSpec("Double Exponential Moving Average (DEMA)", "trend", True, calculate_dema),
    IndicatorSpec("Triple Exponential Moving Average (TEMA)", "trend", True, calculate_tema),
    IndicatorSpec("Kaufman Adaptive Moving Average (KAMA)", "trend", True, calculate_kama),
    IndicatorSpec("Volume Weighted Moving Average (VWMA)", "trend", True, calculate_vwma),
    IndicatorSpec("Moving Average Convergence Divergence (MACD)", "momentum", True, calculate_macd),
    IndicatorSpec("Average Directional Index (ADX)", "trend", True, calculate_adx),
    IndicatorSpec("Aroon", "trend", True, calculate_aroon),
    IndicatorSpec("Parabolic SAR", "trend", True, calculate_parabolic_sar),
    IndicatorSpec("Ichimoku Cloud", "trend", True, calculate_ichimoku),
    IndicatorSpec("Vortex Indicator", "trend", True, calculate_vortex),
    IndicatorSpec("TRIX", "momentum", True, calculate_trix),
    IndicatorSpec("Relative Strength Index (RSI)", "momentum", True, calculate_rsi),
    IndicatorSpec("Stochastic Oscillator", "momentum", True, calculate_stochastic),
    IndicatorSpec("Stochastic RSI", "momentum", True, calculate_stoch_rsi),
    IndicatorSpec("Williams %R", "momentum", True, calculate_williams_r),
    IndicatorSpec("Commodity Channel Index (CCI)", "momentum", True, calculate_cci),
    IndicatorSpec("Rate of Change (ROC)", "momentum", True, calculate_roc),
    IndicatorSpec("Momentum (MOM)", "momentum", True, calculate_momentum),
    IndicatorSpec("Ultimate Oscillator", "momentum", True, calculate_ultimate_oscillator),
    IndicatorSpec("Percentage Price Oscillator (PPO)", "momentum", True, calculate_ppo),
    IndicatorSpec("Awesome Oscillator (AO)", "momentum", True, calculate_awesome_oscillator),
    IndicatorSpec("Chande Momentum Oscillator (CMO)", "momentum", True, calculate_cmo),
    IndicatorSpec("True Strength Index (TSI)", "momentum", True, calculate_tsi),
    IndicatorSpec("Average True Range (ATR)", "volatility", True, calculate_atr),
    IndicatorSpec("Bollinger Bands", "volatility", True, calculate_bollinger_bands),
    IndicatorSpec("Keltner Channels", "volatility", True, calculate_keltner_channels),
    IndicatorSpec("Donchian Channels", "volatility", True, calculate_donchian_channels),
    IndicatorSpec("Rolling Standard Deviation", "volatility", True, calculate_rolling_volatility),
    IndicatorSpec("Historical Volatility (Close-to-Close)", "volatility", True, calculate_rolling_volatility),
    IndicatorSpec("Chaikin Volatility", "volatility", True, calculate_chaikin_volatility),
    IndicatorSpec("Ulcer Index", "volatility", True, calculate_ulcer_index),
    IndicatorSpec("Parkinson Volatility", "volatility", True, calculate_volatility_metrics),
    IndicatorSpec("Garman-Klass Volatility", "volatility", True, calculate_volatility_metrics),
    IndicatorSpec("Yang-Zhang Volatility", "volatility", True, calculate_volatility_metrics),
    IndicatorSpec("On-Balance Volume (OBV)", "volume", True, calculate_obv),
    IndicatorSpec("Volume Weighted Average Price (VWAP)", "volume", True, calculate_vwap),
    IndicatorSpec("Money Flow Index (MFI)", "volume", True, calculate_mfi),
    IndicatorSpec("Chaikin Money Flow (CMF)", "volume", True, calculate_cmf),
    IndicatorSpec("Accumulation/Distribution Line (ADL)", "volume", True, calculate_ad_line),
    IndicatorSpec("Volume Oscillator", "volume", True, calculate_volume_oscillator),
    IndicatorSpec("Ease of Movement (EOM)", "volume", True, calculate_eom),
    IndicatorSpec("Force Index", "volume", True, calculate_force_index),
    IndicatorSpec("Price Volume Trend (PVT)", "volume", True, calculate_pvt),
    IndicatorSpec("Negative Volume Index (NVI)", "volume", True, calculate_nvi),
    IndicatorSpec("Positive Volume Index (PVI)", "volume", True, calculate_pvi),
    IndicatorSpec("Volume Profile (POC/VAH/VAL)", "volume", True, calculate_vwap),
    IndicatorSpec("Trend Lines (Linear Regression)", "trend", True, calculate_trend_lines),
    IndicatorSpec("EMA Slope", "trend", True, calculate_ema_slope),
]

assert len(_INDICATORS) == 52, "Indicator registry must contain exactly 52 entries"


def get_indicator_registry(implemented_only: bool = False) -> List[IndicatorSpec]:
    """Return the indicator registry, optionally filtering to implemented ones."""
    if implemented_only:
        return [spec for spec in _INDICATORS if spec.implemented]
    return list(_INDICATORS)
