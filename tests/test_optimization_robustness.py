"""
Unit tests for Phase 2: Optimization Robustness.

Tests:
- EarlyStoppingCriterion: patience, convergence, reset, min_delta
- MarketNoiseGenerator: Gaussian, fat-tail, regime-shift, volume-spikes, stress_test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.market_noise_generator import MarketNoiseGenerator
from optimization.early_stopping import EarlyStoppingCriterion


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_ohlcv() -> pd.DataFrame:
    """150-bar OHLCV for noise injection tests."""
    rng = np.random.default_rng(7)
    n = 150
    idx = pd.date_range("2024-06-01", periods=n, freq="1h")
    ret = rng.normal(0, 0.002, n)
    close = 40_000 * np.cumprod(1 + ret)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.uniform(-0.001, 0.001, n)),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(500, 5000, n),
        },
        index=idx,
    )


# ===========================================================================
# EarlyStoppingCriterion
# ===========================================================================


class TestEarlyStoppingCriterion:
    def test_converges_after_patience(self):
        criterion = EarlyStoppingCriterion(patience=5, min_delta=1e-4, maximize=True)
        # First trial always "improves"
        assert not criterion.update(1.0)
        # Next 5 with no improvement should trigger stop
        for _ in range(5):
            stopped = criterion.update(1.0)
        assert stopped

    def test_reset_clears_state(self):
        criterion = EarlyStoppingCriterion(patience=3, min_delta=0.0)
        for _ in range(4):
            criterion.update(0.5)
        criterion.reset()
        assert criterion.no_improve_count == 0
        assert criterion.best is None
        assert criterion.history == []

    def test_improvement_resets_counter(self):
        criterion = EarlyStoppingCriterion(patience=3, min_delta=1e-4)
        criterion.update(1.0)
        criterion.update(1.0)
        criterion.update(1.0)
        # Should not have stopped yet (only 2 non-improving after first)
        assert not criterion.update(1.5)  # improvement resets counter
        assert criterion.no_improve_count == 0

    def test_minimize_mode(self):
        criterion = EarlyStoppingCriterion(patience=3, min_delta=1e-4, maximize=False)
        criterion.update(10.0)  # first = improved
        criterion.update(9.0)   # improved (lower)
        assert not criterion.update(9.0)   # no improve
        assert not criterion.update(9.0)   # no improve
        assert criterion.update(9.0)       # patience=3 → stopped

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError):
            EarlyStoppingCriterion(patience=0)

    def test_reason_set_on_stop(self):
        criterion = EarlyStoppingCriterion(patience=2, min_delta=1e-4)
        criterion.update(1.0)
        criterion.update(1.0)
        criterion.update(1.0)
        assert criterion.reason != ""

    def test_history_length_matches_updates(self):
        criterion = EarlyStoppingCriterion(patience=10)
        for i in range(7):
            criterion.update(float(i))
        assert len(criterion.history) == 7


# ===========================================================================
# MarketNoiseGenerator
# ===========================================================================


class TestMarketNoiseGenerator:
    def test_gaussian_noise_preserves_shape(self, clean_ohlcv):
        gen = MarketNoiseGenerator(seed=1)
        noisy = gen.add_gaussian_noise(clean_ohlcv, sigma_fraction=0.001)
        assert noisy.shape == clean_ohlcv.shape

    def test_gaussian_noise_changes_prices(self, clean_ohlcv):
        gen = MarketNoiseGenerator(seed=2)
        noisy = gen.add_gaussian_noise(clean_ohlcv, sigma_fraction=0.01)
        # At least some close prices should differ
        assert not (noisy["close"] == clean_ohlcv["close"]).all()

    def test_prices_remain_positive_after_noise(self, clean_ohlcv):
        gen = MarketNoiseGenerator(seed=3)
        noisy = gen.add_gaussian_noise(clean_ohlcv, sigma_fraction=0.50)  # extreme noise
        assert (noisy["close"] > 0).all()
        assert (noisy["high"] > 0).all()
        assert (noisy["low"] > 0).all()

    def test_fat_tail_shocks(self, clean_ohlcv):
        gen = MarketNoiseGenerator(seed=4)
        noisy = gen.add_fat_tail_shocks(clean_ohlcv, shock_probability=0.5)
        assert noisy.shape == clean_ohlcv.shape
        assert (noisy["close"] > 0).all()

    def test_regime_shift_applied(self, clean_ohlcv):
        gen = MarketNoiseGenerator(seed=5)
        noisy = gen.add_regime_shift(clean_ohlcv, shift_at_fraction=0.5, vol_multiplier=3.0)
        assert noisy.shape == clean_ohlcv.shape
        # Second half should differ from the original
        n = len(noisy)
        orig_second = clean_ohlcv["close"].iloc[n // 2 :]
        noisy_second = noisy["close"].iloc[n // 2 :]
        assert not (orig_second.values == noisy_second.values).all()

    def test_volume_spikes(self, clean_ohlcv):
        gen = MarketNoiseGenerator(seed=6)
        noisy = gen.add_volume_spikes(clean_ohlcv, spike_probability=1.0, spike_multiplier_range=(2, 5))
        assert (noisy["volume"] >= clean_ohlcv["volume"]).all()

    def test_high_low_ordering_preserved(self, clean_ohlcv):
        gen = MarketNoiseGenerator(seed=7)
        noisy = gen.add_gaussian_noise(clean_ohlcv, sigma_fraction=0.005)
        assert (noisy["high"] >= noisy["low"]).all()

    def test_stress_test_composite(self, clean_ohlcv):
        gen = MarketNoiseGenerator(seed=8)
        noisy = gen.stress_test(clean_ohlcv)
        assert noisy.shape == clean_ohlcv.shape
        assert (noisy["close"] > 0).all()

    def test_volume_spikes_without_volume_col(self, clean_ohlcv):
        """Should return unchanged df if volume column is missing."""
        df_no_vol = clean_ohlcv.drop(columns=["volume"])
        gen = MarketNoiseGenerator(seed=9)
        result = gen.add_volume_spikes(df_no_vol)
        pd.testing.assert_frame_equal(result, df_no_vol)
