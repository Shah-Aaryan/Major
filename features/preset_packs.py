"""
Experiment Preset Packs.

Provides pre-configured ``(FeatureConfig, optimization_kwargs)`` bundles so
researchers can run controlled experiments with a single keyword argument
instead of laboriously specifying every config knob.

Available presets
-----------------
``"fast"``
    10 indicators, 15 optimizer trials.  Designed for rapid CI/CD runs and
    quick sanity checks.  Completes in < 30 seconds on a typical laptop.

``"balanced"``
    25 indicators, 50 optimizer trials.  A good default for day-to-day
    research that balances speed and thoroughness.

``"research"``
    All 50 indicators, 200 optimizer trials with nested walk-forward splits.
    Intended for final benchmark runs that produce publication-quality results.

Usage
-----
>>> from features.preset_packs import get_preset
>>> feature_cfg, optim_kwargs = get_preset("balanced")
>>> pipeline = AutonomousPipeline(..., feature_config=feature_cfg, **optim_kwargs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from features.feature_engine import FeatureConfig

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------


def _fast_feature_config() -> FeatureConfig:
    return FeatureConfig(
        log_return_periods=[1, 5],
        rolling_return_windows=[5, 20],
        momentum_periods=[10],
        sma_windows=[20, 50],
        ema_windows=[9, 21],
        adx_period=14,
        rsi_periods=[14],
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        atr_periods=[14],
        bollinger_window=20,
        bollinger_std=2.0,
        volume_ma_windows=[20],
        regime_lookback=50,
        trend_threshold=25.0,
        adx_trend_threshold=25.0,
    )


def _balanced_feature_config() -> FeatureConfig:
    return FeatureConfig(
        log_return_periods=[1, 5, 15],
        rolling_return_windows=[5, 15, 60],
        momentum_periods=[10, 20],
        sma_windows=[10, 20, 50, 100],
        ema_windows=[9, 21, 50, 100],
        adx_period=14,
        rsi_periods=[6, 14],
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        atr_periods=[7, 14],
        bollinger_window=20,
        bollinger_std=2.0,
        volume_ma_windows=[10, 20],
        regime_lookback=100,
        trend_threshold=25.0,
        adx_trend_threshold=25.0,
    )


def _research_feature_config() -> FeatureConfig:
    """Full 50-indicator surface."""
    return FeatureConfig(
        log_return_periods=[1, 5, 15, 60],
        rolling_return_windows=[5, 15, 60, 240],
        momentum_periods=[10, 20, 50],
        sma_windows=[5, 10, 20, 50, 100, 200],
        ema_windows=[5, 10, 20, 50, 100, 200],
        adx_period=14,
        rsi_periods=[6, 14, 21],
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        atr_periods=[7, 14, 21],
        bollinger_window=20,
        bollinger_std=2.0,
        volume_ma_windows=[10, 20, 50],
        regime_lookback=100,
        trend_threshold=25.0,
        adx_trend_threshold=25.0,
    )


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, Tuple[FeatureConfig, Dict[str, Any]]] = {
    "fast": (
        _fast_feature_config(),
        {
            "optimization_method": "random_search",
            "n_optimization_iterations": 15,
            "optimization_interval": 30,
            "min_optimization_rows": 60,
        },
    ),
    "balanced": (
        _balanced_feature_config(),
        {
            "optimization_method": "bayesian_tpe",
            "n_optimization_iterations": 50,
            "optimization_interval": 50,
            "min_optimization_rows": 100,
        },
    ),
    "research": (
        _research_feature_config(),
        {
            "optimization_method": "bayesian_tpe",
            "n_optimization_iterations": 200,
            "optimization_interval": 100,
            "min_optimization_rows": 200,
        },
    ),
}


def get_preset(name: str) -> Tuple[FeatureConfig, Dict[str, Any]]:
    """Return ``(FeatureConfig, optimization_kwargs)`` for the named preset.

    Parameters
    ----------
    name:
        One of ``"fast"``, ``"balanced"``, ``"research"``.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised preset.
    """
    name = name.lower().strip()
    if name not in _PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {sorted(_PRESETS.keys())}"
        )
    feature_cfg, optim_kwargs = _PRESETS[name]
    return feature_cfg, dict(optim_kwargs)  # return copies


def list_presets() -> Dict[str, str]:
    """Return a brief description of each available preset."""
    return {
        "fast": "10 indicators, 15 trials — rapid CI/CD and sanity-check runs.",
        "balanced": "25 indicators, 50 trials — good daily research default.",
        "research": "50 indicators, 200 trials — full benchmark / publication quality.",
    }
