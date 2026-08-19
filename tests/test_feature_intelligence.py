"""
Unit tests for Phase 1: Feature Intelligence & Quality Control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.correlation_filter import CorrelationFilter
from features.feature_engine import FeatureConfig, FeatureEngine
from features.feature_selector import FeatureSelector
from features.leakage_detector import FeatureLeakageDetector
from features.preset_packs import get_preset, list_presets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_features(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """
    Build a minimal feature set with enough rows to survive drop_na.
    Uses only a few fast indicators (RSI, SMA, volume_ma) so the warm-up
    is ≤ 200 bars, leaving plenty of clean rows.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    ret = rng.normal(0.0002, 0.003, n)
    close_arr = 50_000 * np.cumprod(1 + ret)
    close = pd.Series(close_arr, index=idx)
    df = pd.DataFrame(
        {
            "open": close_arr * (1 + rng.uniform(-0.001, 0.001, n)),
            "high": close_arr * (1 + np.abs(rng.uniform(0, 0.002, n))),
            "low": close_arr * (1 - np.abs(rng.uniform(0, 0.002, n))),
            "close": close_arr,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )

    # Compute a small, fast feature set manually so we control warm-up length
    df["sma_20"] = close.rolling(20).mean().values
    df["rsi_14"] = _compute_rsi(close, 14).values
    df["vol_20"] = close.rolling(20).std().values
    df["vol_ratio"] = df["volume"] / pd.Series(df["volume"].values, index=idx).rolling(10).mean().values
    df["log_ret"] = np.log(close_arr / np.roll(close_arr, 1))
    df.loc[df.index[0], "log_ret"] = np.nan
    # A synthetic "regime" label
    df["regime_combined"] = np.where(
        pd.Series(df["sma_20"].values, index=idx) > close.shift(1),
        "trending_bullish", "ranging"
    )
    return df


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@pytest.fixture(scope="module")
def small_ohlcv() -> pd.DataFrame:
    """Plain OHLCV for tests that only need raw price data."""
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    ret = rng.normal(0.0002, 0.003, n)
    close = 50_000 * np.cumprod(1 + ret)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.uniform(-0.001, 0.001, n)),
            "high": close * (1 + np.abs(rng.uniform(0, 0.002, n))),
            "low": close * (1 - np.abs(rng.uniform(0, 0.002, n))),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    """Feature DataFrame with NaN rows removed and sufficient clean rows."""
    df = _make_simple_features(300)
    return df.dropna()


@pytest.fixture(scope="module")
def features_with_nan() -> pd.DataFrame:
    """Feature DataFrame with NaN rows preserved."""
    return _make_simple_features(300)


# ===========================================================================
# FeatureLeakageDetector
# ===========================================================================


class TestFeatureLeakageDetector:
    def test_scan_returns_report(self, features_with_nan):
        detector = FeatureLeakageDetector(future_corr_threshold=0.99, warmup_rows=20)
        report = detector.scan(features_with_nan)
        assert isinstance(report.passed, bool)
        assert report.summary() != ""

    def test_no_close_column_skips_future_corr(self, features_with_nan):
        df_no_close = features_with_nan.drop(columns=["close"])
        detector = FeatureLeakageDetector()
        report = detector.scan(df_no_close)
        assert report.high_future_corr_columns == []

    def test_nan_after_warmup_detected(self, small_ohlcv):
        df = small_ohlcv.copy()
        df["always_nan"] = np.nan
        detector = FeatureLeakageDetector(warmup_rows=10)
        report = detector.scan(df)
        assert "always_nan" in report.nan_after_warmup_columns

    def test_report_to_dict_has_required_keys(self, features_with_nan):
        detector = FeatureLeakageDetector()
        report = detector.scan(features_with_nan)
        d = report.to_dict()
        assert "passed" in d
        assert "high_future_corr" in d
        assert "nan_after_warmup" in d


# ===========================================================================
# CorrelationFilter
# ===========================================================================


class TestCorrelationFilter:
    def test_basic_fit_transform_reduces_or_preserves_columns(self, features_df):
        cf = CorrelationFilter(threshold=0.90)
        filtered = cf.fit_transform(features_df)
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered.columns) <= len(features_df.columns)

    def test_ohlcv_always_kept(self, features_df):
        cf = CorrelationFilter(threshold=0.50)
        filtered = cf.fit_transform(features_df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in filtered.columns

    def test_transform_before_fit_raises(self, features_df):
        cf = CorrelationFilter()
        with pytest.raises(RuntimeError):
            cf.transform(features_df)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            CorrelationFilter(threshold=1.5)

    def test_summary_after_fit(self, features_df):
        cf = CorrelationFilter(threshold=0.85)
        cf.fit(features_df)
        summary = cf.summary()
        assert "Dropped" in summary
        assert "Kept" in summary

    def test_fit_transform_idempotent(self, features_df):
        cf = CorrelationFilter(threshold=0.90)
        first = cf.fit_transform(features_df)
        second = cf.transform(features_df)
        pd.testing.assert_frame_equal(first, second)


# ===========================================================================
# FeatureSelector
# ===========================================================================


class TestFeatureSelector:
    def test_rf_importance(self, features_df):
        selector = FeatureSelector(n_top=5, method="rf", regime_column=None)
        top = selector.fit_transform(features_df)
        assert isinstance(top, pd.DataFrame)
        assert len(top.columns) > 0
        assert selector.importance_df_ is not None
        assert len(selector.importance_df_) > 0

    def test_mi_importance(self, features_df):
        selector = FeatureSelector(n_top=5, method="mi", regime_column=None)
        top = selector.fit_transform(features_df)
        assert isinstance(top, pd.DataFrame)
        assert selector.selected_features_ is not None

    def test_transform_before_fit_raises(self, features_df):
        selector = FeatureSelector(regime_column=None)
        with pytest.raises(RuntimeError):
            selector.transform(features_df)

    def test_regime_importance_computed(self, features_df):
        selector = FeatureSelector(n_top=5, regime_column="regime_combined")
        selector.fit(features_df)
        assert isinstance(selector.regime_importance_, dict)


# ===========================================================================
# Preset Packs
# ===========================================================================


class TestPresetPacks:
    @pytest.mark.parametrize("preset_name", ["fast", "balanced", "research"])
    def test_get_preset_returns_config_and_kwargs(self, preset_name):
        feature_cfg, optim_kwargs = get_preset(preset_name)
        assert isinstance(feature_cfg, FeatureConfig)
        assert isinstance(optim_kwargs, dict)
        assert "n_optimization_iterations" in optim_kwargs
        assert "optimization_method" in optim_kwargs

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent_preset")

    def test_list_presets(self):
        presets = list_presets()
        assert "fast" in presets
        assert "balanced" in presets
        assert "research" in presets

    def test_fast_fewer_trials_than_research(self):
        _, fast_kwargs = get_preset("fast")
        _, research_kwargs = get_preset("research")
        assert fast_kwargs["n_optimization_iterations"] < research_kwargs["n_optimization_iterations"]

    def test_get_preset_returns_copy(self):
        _, kwargs1 = get_preset("fast")
        kwargs1["extra_key"] = "mutated"
        _, kwargs2 = get_preset("fast")
        assert "extra_key" not in kwargs2
