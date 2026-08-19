"""
Correlation Filter — removes multicollinear features from a DataFrame.

Algorithm
---------
1. Compute the Pearson correlation matrix of all numeric features.
2. For each pair (i, j) with |corr| >= threshold, remove the column
   that has the lower mean absolute correlation with all *other* columns
   (i.e. keep the more "central" / informative one).
3. OHLCV columns and regime/label columns are always preserved.

Typical use
-----------
>>> from features.correlation_filter import CorrelationFilter
>>> cf = CorrelationFilter(threshold=0.90)
>>> filtered_df = cf.fit_transform(features_df)
>>> print(cf.dropped_columns_)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ALWAYS_KEEP = {"open", "high", "low", "close", "volume"}


class CorrelationFilter:
    """Drop highly-correlated numeric feature columns.

    Parameters
    ----------
    threshold:
        Absolute Pearson correlation above which a pair is considered
        redundant. Default is 0.90.
    preserve_columns:
        Additional column names that must never be dropped.
    """

    def __init__(
        self,
        threshold: float = 0.90,
        preserve_columns: Optional[List[str]] = None,
    ) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold
        self.preserve_columns: Set[str] = _ALWAYS_KEEP | set(preserve_columns or [])

        # Fitted state
        self.dropped_columns_: List[str] = []
        self.kept_columns_: List[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "CorrelationFilter":
        """Determine which columns to drop based on correlation analysis.

        Parameters
        ----------
        df:
            Feature DataFrame (any mix of numeric/non-numeric columns).

        Returns
        -------
        self
        """
        numeric_df = df.select_dtypes(include=[np.number])
        # Remove OHLCV + always-preserve from the candidate set
        candidate_cols = [c for c in numeric_df.columns if c not in self.preserve_columns]

        if len(candidate_cols) < 2:
            self.kept_columns_ = list(df.columns)
            self.dropped_columns_ = []
            self._is_fitted = True
            return self

        corr_matrix = numeric_df[candidate_cols].corr().abs()
        to_drop: Set[str] = set()

        # Upper triangle pairs
        for i in range(len(candidate_cols)):
            if candidate_cols[i] in to_drop:
                continue
            for j in range(i + 1, len(candidate_cols)):
                if candidate_cols[j] in to_drop:
                    continue
                if corr_matrix.iloc[i, j] >= self.threshold:
                    # Drop the column with higher mean correlation to others
                    mean_i = corr_matrix.iloc[i].drop(candidate_cols[i]).mean()
                    mean_j = corr_matrix.iloc[j].drop(candidate_cols[j]).mean()
                    drop_col = candidate_cols[j] if mean_i <= mean_j else candidate_cols[i]
                    to_drop.add(drop_col)
                    logger.debug(
                        "Dropping '%s' (corr with '%s' = %.3f)",
                        drop_col,
                        candidate_cols[i] if drop_col == candidate_cols[j] else candidate_cols[j],
                        corr_matrix.iloc[i, j],
                    )

        self.dropped_columns_ = sorted(to_drop)
        self.kept_columns_ = [c for c in df.columns if c not in to_drop]

        logger.info(
            "CorrelationFilter fitted: keeping %d / %d features (dropped %d)",
            len(self.kept_columns_),
            len(df.columns),
            len(self.dropped_columns_),
        )
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted filter to a DataFrame.

        Columns that were not present during ``fit`` are kept as-is.

        Parameters
        ----------
        df:
            DataFrame to filter.

        Returns
        -------
        Filtered DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        cols_to_drop = [c for c in self.dropped_columns_ if c in df.columns]
        return df.drop(columns=cols_to_drop)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def summary(self) -> str:
        """Return a human-readable summary of the filter state."""
        if not self._is_fitted:
            return "CorrelationFilter: not fitted yet."
        return (
            f"CorrelationFilter(threshold={self.threshold})\n"
            f"  Dropped ({len(self.dropped_columns_)}): {self.dropped_columns_}\n"
            f"  Kept ({len(self.kept_columns_)}): {self.kept_columns_[:10]}{'...' if len(self.kept_columns_) > 10 else ''}"
        )
