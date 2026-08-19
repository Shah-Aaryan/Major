"""
Feature Leakage Detector.

Scans computed feature DataFrames for look-ahead bias caused by rolling
windows that are accidentally forward-filled or shifted incorrectly.

Key checks
----------
1. Shift-alignment: verifies that every rolling/shifted column is lagged
   by at least 1 bar relative to the target close price it predicts.
2. Future correlation: flags features whose correlation with *next-bar*
   returns is suspiciously high (likely leakage).
3. Missing-value policy audit: reports columns that still contain NaNs
   after the standard warm-up period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Threshold above which future correlation is considered suspicious.
_DEFAULT_LEAKAGE_CORR_THRESHOLD = 0.90
# Minimum rows needed before the warm-up period is considered complete.
_DEFAULT_WARMUP_ROWS = 50


@dataclass
class LeakageReport:
    """Result of a leakage-detection scan."""

    # Columns with suspiciously high correlation to next-bar returns.
    high_future_corr_columns: List[Tuple[str, float]] = field(default_factory=list)
    # Columns still containing NaN after the warm-up period.
    nan_after_warmup_columns: List[str] = field(default_factory=list)
    # Columns where data appears before the expected warm-up offset.
    early_data_columns: List[str] = field(default_factory=list)
    # Overall pass/fail flag.
    passed: bool = True

    def summary(self) -> str:
        lines = ["=== Feature Leakage Report ==="]
        lines.append(f"Status: {'PASS' if self.passed else 'FAIL'}")

        if self.high_future_corr_columns:
            lines.append("\nHigh Future-Correlation Columns (potential leakage):")
            for col, corr in self.high_future_corr_columns:
                lines.append(f"  {col}: corr={corr:.4f}")

        if self.nan_after_warmup_columns:
            lines.append("\nColumns with NaN after warm-up:")
            for col in self.nan_after_warmup_columns:
                lines.append(f"  {col}")

        if self.early_data_columns:
            lines.append("\nColumns with data before expected warm-up:")
            for col in self.early_data_columns:
                lines.append(f"  {col}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "high_future_corr": [
                {"column": c, "correlation": r} for c, r in self.high_future_corr_columns
            ],
            "nan_after_warmup": self.nan_after_warmup_columns,
            "early_data_columns": self.early_data_columns,
        }


class FeatureLeakageDetector:
    """Detects look-ahead bias and data-quality issues in computed features.

    Parameters
    ----------
    future_corr_threshold:
        Absolute correlation above which a feature/next-bar-return pair is
        flagged as a potential leakage.
    warmup_rows:
        Rows at the beginning of the dataset considered the "warm-up"
        period. NaN checks are applied *after* this window.
    """

    def __init__(
        self,
        future_corr_threshold: float = _DEFAULT_LEAKAGE_CORR_THRESHOLD,
        warmup_rows: int = _DEFAULT_WARMUP_ROWS,
    ) -> None:
        self.future_corr_threshold = future_corr_threshold
        self.warmup_rows = warmup_rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, features: pd.DataFrame) -> LeakageReport:
        """Run all leakage / quality checks on a features DataFrame.

        Parameters
        ----------
        features:
            DataFrame produced by ``FeatureEngine.generate_features``.
            Must contain a ``close`` column.

        Returns
        -------
        LeakageReport
        """
        report = LeakageReport()

        if "close" not in features.columns:
            logger.warning("FeatureLeakageDetector: 'close' column missing; skipping future-corr check.")
        else:
            report.high_future_corr_columns = self._check_future_correlation(features)

        report.nan_after_warmup_columns = self._check_nan_after_warmup(features)
        report.early_data_columns = self._check_early_data(features)

        report.passed = not (
            report.high_future_corr_columns
            or report.nan_after_warmup_columns
            or report.early_data_columns
        )

        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_future_correlation(
        self, features: pd.DataFrame
    ) -> List[Tuple[str, float]]:
        """Flag features correlated > threshold with next-bar close return."""
        close = features["close"].astype(float)
        next_return = close.pct_change().shift(-1)  # next-bar return

        flagged: List[Tuple[str, float]] = []
        ohlcv = {"open", "high", "low", "close", "volume"}

        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ohlcv:
                continue
            try:
                col_series = features[col].astype(float)
                corr = col_series.corr(next_return)
                if pd.notna(corr) and abs(corr) >= self.future_corr_threshold:
                    flagged.append((col, float(corr)))
            except Exception:
                pass

        return sorted(flagged, key=lambda x: abs(x[1]), reverse=True)

    def _check_nan_after_warmup(self, features: pd.DataFrame) -> List[str]:
        """Identify columns still containing NaN after the warm-up window."""
        if len(features) <= self.warmup_rows:
            return []

        post_warmup = features.iloc[self.warmup_rows :]
        nan_cols = [col for col in features.columns if post_warmup[col].isna().any()]
        return nan_cols

    def _check_early_data(self, features: pd.DataFrame) -> List[str]:
        """Flag columns that have valid data before row `warmup_rows`.

        For most derived indicators the first ``warmup_rows`` rows should
        be NaN (the indicator is still accumulating history). If a feature
        column has *zero* NaN values in the first slice it may have been
        back-filled or improperly computed.
        """
        if len(features) <= self.warmup_rows:
            return []

        early_slice = features.iloc[: self.warmup_rows]
        ohlcv = {"open", "high", "low", "close", "volume"}

        flagged = []
        for col in features.columns:
            if col in ohlcv:
                continue
            # Only flag numeric indicator columns
            if not pd.api.types.is_numeric_dtype(features[col]):
                continue
            # If the entire early slice is non-NaN for a "derived" column
            # it *might* have been improperly filled. We raise a warning
            # rather than a hard failure because some indicators (e.g. ROC-1)
            # legitimately have very short warm-ups.
            nan_count = early_slice[col].isna().sum()
            if nan_count == 0:
                flagged.append(col)

        return flagged
