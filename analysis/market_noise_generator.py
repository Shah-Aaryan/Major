"""
Market Noise Generator — synthetic noise injection for robustness testing.

Creates deliberately degraded copies of an OHLCV DataFrame so that strategy
parameters optimised on clean data can be stress-tested against realistic
market imperfections:

* **Gaussian price noise** — additive white noise scaled to a fraction of
  the bar's ATR.
* **Student-t fat-tail shocks** — occasional large price spikes drawn from a
  heavy-tailed distribution (mimics flash crashes or news events).
* **Volatility regime injection** — multiplies close prices by a stochastic
  volatility process so a calm regime suddenly becomes turbulent.
* **Volume spikes** — random volume multipliers to simulate liquidity events.

Usage
-----
>>> from analysis.market_noise_generator import MarketNoiseGenerator
>>> gen = MarketNoiseGenerator(seed=42)
>>> noisy_df = gen.add_gaussian_noise(clean_df, sigma_fraction=0.002)
>>> stressed_df = gen.add_regime_shift(clean_df, shift_at_fraction=0.5, vol_multiplier=3.0)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketNoiseGenerator:
    """Inject synthetic noise and market stress into OHLCV data.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public noise methods
    # ------------------------------------------------------------------

    def add_gaussian_noise(
        self,
        df: pd.DataFrame,
        sigma_fraction: float = 0.001,
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """Add zero-mean Gaussian noise to price columns.

        Parameters
        ----------
        df:
            Clean OHLCV DataFrame.
        sigma_fraction:
            Noise standard deviation expressed as a fraction of the price
            level at each bar.  E.g. 0.001 = 0.1% of price.
        columns:
            Price columns to perturb.  Defaults to ``["open", "high", "low", "close"]``.

        Returns
        -------
        Noisy copy of ``df``.
        """
        df = df.copy()
        cols = columns or ["open", "high", "low", "close"]
        for col in cols:
            if col not in df.columns:
                continue
            prices = df[col].values.astype(float)
            noise = self.rng.normal(0, sigma_fraction * prices)
            df[col] = np.maximum(prices + noise, 1e-6)  # keep prices positive

        df = self._fix_ohlcv_ordering(df)
        return df

    def add_fat_tail_shocks(
        self,
        df: pd.DataFrame,
        shock_probability: float = 0.01,
        tail_df: float = 3.0,
        shock_scale: float = 0.03,
    ) -> pd.DataFrame:
        """Inject rare but large price shocks drawn from a Student-t distribution.

        Parameters
        ----------
        shock_probability:
            Per-bar probability of a shock event occurring.
        tail_df:
            Degrees of freedom for the Student-t distribution (lower = fatter tails).
        shock_scale:
            Scale of the shock relative to the closing price.
        """
        df = df.copy()
        n = len(df)
        shock_mask = self.rng.random(n) < shock_probability
        shocks = self.rng.standard_t(tail_df, n) * shock_scale

        for col in ["close", "high", "low"]:
            if col not in df.columns:
                continue
            prices = df[col].values.astype(float)
            prices[shock_mask] *= 1 + shocks[shock_mask]
            df[col] = np.maximum(prices, 1e-6)

        df = self._fix_ohlcv_ordering(df)
        logger.info("Fat-tail shocks injected: %d events (p=%.3f)", shock_mask.sum(), shock_probability)
        return df

    def add_regime_shift(
        self,
        df: pd.DataFrame,
        shift_at_fraction: float = 0.5,
        vol_multiplier: float = 3.0,
        drift_multiplier: float = -1.0,
    ) -> pd.DataFrame:
        """Simulate a sudden market regime shift mid-series.

        After ``shift_at_fraction`` of the data the volatility is scaled by
        ``vol_multiplier`` and the drift direction is reversed.

        Parameters
        ----------
        shift_at_fraction:
            Point in the dataset (0–1) where the regime changes.
        vol_multiplier:
            How much more volatile the second regime is.
        drift_multiplier:
            Direction multiplier applied to log-returns in the stressed regime.
        """
        df = df.copy()
        n = len(df)
        shift_idx = max(1, int(n * shift_at_fraction))

        if "close" not in df.columns:
            return df

        close = df["close"].values.astype(float).copy()
        log_returns = np.diff(np.log(close), prepend=np.log(close[0]))

        # Modify the tail regime
        tail = log_returns[shift_idx:]
        stressed = drift_multiplier * tail + self.rng.normal(0, (vol_multiplier - 1) * tail.std(), len(tail))
        log_returns[shift_idx:] = stressed

        # Reconstruct prices from modified returns
        new_close = close[0] * np.exp(np.cumsum(log_returns))
        scale_factor = new_close / close

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = (df[col].values.astype(float) * scale_factor).clip(1e-6)

        df = self._fix_ohlcv_ordering(df)
        logger.info("Regime shift injected at row %d / %d (vol_mult=%.1f)", shift_idx, n, vol_multiplier)
        return df

    def add_volume_spikes(
        self,
        df: pd.DataFrame,
        spike_probability: float = 0.02,
        spike_multiplier_range: tuple = (3.0, 10.0),
    ) -> pd.DataFrame:
        """Randomly multiply bar volume by large factors to simulate liquidity events."""
        df = df.copy()
        if "volume" not in df.columns:
            return df

        n = len(df)
        spike_mask = self.rng.random(n) < spike_probability
        multipliers = self.rng.uniform(*spike_multiplier_range, n)

        volume = df["volume"].values.astype(float).copy()
        volume[spike_mask] *= multipliers[spike_mask]
        df["volume"] = volume
        logger.info("Volume spikes injected: %d events", spike_mask.sum())
        return df

    # ------------------------------------------------------------------
    # Convenience composite
    # ------------------------------------------------------------------

    def stress_test(
        self,
        df: pd.DataFrame,
        gaussian_sigma: float = 0.001,
        shock_prob: float = 0.005,
        regime_shift: bool = True,
    ) -> pd.DataFrame:
        """Apply all noise types in sequence for a full stress test."""
        df = self.add_gaussian_noise(df, sigma_fraction=gaussian_sigma)
        df = self.add_fat_tail_shocks(df, shock_probability=shock_prob)
        if regime_shift:
            df = self.add_regime_shift(df)
        df = self.add_volume_spikes(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_ohlcv_ordering(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure high >= max(open,close) and low <= min(open,close)."""
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                return df
        df = df.copy()
        true_high = df[["open", "close"]].max(axis=1)
        true_low = df[["open", "close"]].min(axis=1)
        df["high"] = df["high"].combine(true_high, max)
        df["low"] = df["low"].combine(true_low, min)
        return df
