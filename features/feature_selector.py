"""
Automated Feature Selector & Importance Ranker.

Provides three complementary methods:
1. **Random-Forest importance** — model-based permutation importance.
2. **Mutual information** — non-parametric dependency measure.
3. **Regime-conditional relevance** — importance computed separately per
   market regime label so analysts can see which indicators matter most in
   trending vs. ranging vs. volatile conditions.

Typical usage
-------------
>>> from features.feature_selector import FeatureSelector
>>> selector = FeatureSelector(n_top=20, method="rf")
>>> selector.fit(features_df, target_col="next_return")
>>> top_df = selector.transform(features_df)
>>> print(selector.importance_df_)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV = {"open", "high", "low", "close", "volume"}


class FeatureSelector:
    """Rank features by predictive importance and select the top-N.

    Parameters
    ----------
    n_top:
        Maximum number of features to keep after selection.
    method:
        Importance estimation method: ``"rf"`` (Random Forest) or
        ``"mi"`` (Mutual Information). Falls back to mutual information
        if scikit-learn's RandomForestRegressor is unavailable.
    regime_column:
        Optional column name containing market regime labels.  When set,
        regime-conditional importance scores are also computed.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_top: int = 20,
        method: str = "rf",
        regime_column: Optional[str] = "regime_combined",
        random_state: int = 42,
    ) -> None:
        self.n_top = n_top
        self.method = method
        self.regime_column = regime_column
        self.random_state = random_state

        # Populated after fit()
        self.importance_df_: Optional[pd.DataFrame] = None
        self.regime_importance_: Dict[str, pd.DataFrame] = {}
        self.selected_features_: List[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        target_col: str = "next_return",
        target_horizon: int = 1,
    ) -> "FeatureSelector":
        """Compute feature importance scores.

        Parameters
        ----------
        features:
            DataFrame with all computed features (including ``close``).
        target_col:
            If this column exists in ``features`` it is used directly as the
            regression target.  Otherwise a *next-bar close return* is
            constructed automatically from the ``close`` column.
        target_horizon:
            Number of bars ahead for the auto-constructed target.

        Returns
        -------
        self
        """
        df = features.copy()

        # Build or locate target
        if target_col not in df.columns:
            if "close" not in df.columns:
                raise ValueError("'close' column required to auto-build regression target.")
            df[target_col] = df["close"].pct_change(target_horizon).shift(-target_horizon)

        # Drop rows where target is NaN
        df = df.dropna(subset=[target_col])
        if len(df) < 30:
            raise ValueError("Not enough rows after dropping NaN targets (<30).")

        y = df[target_col].values
        candidate_cols = self._get_candidate_cols(df, target_col)
        X = df[candidate_cols].fillna(0).values

        # Compute importance
        scores = self._compute_importance(X, y, candidate_cols)
        self.importance_df_ = pd.DataFrame(
            {"feature": candidate_cols, "importance": scores}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        # Select top-N
        self.selected_features_ = list(self.importance_df_["feature"].head(self.n_top))

        # Regime-conditional importance
        if self.regime_column and self.regime_column in df.columns:
            self.regime_importance_ = self._compute_regime_importance(df, target_col, candidate_cols)

        logger.info(
            "FeatureSelector fitted: selected %d / %d features",
            len(self.selected_features_),
            len(candidate_cols),
        )
        self._is_fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Keep only selected features (plus OHLCV) in the DataFrame."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        keep = list(_OHLCV & set(features.columns)) + [
            c for c in self.selected_features_ if c in features.columns
        ]
        # Deduplicate while preserving order
        seen: set = set()
        ordered_keep = [c for c in features.columns if c in keep and not (c in seen or seen.add(c))]
        return features[ordered_keep]

    def fit_transform(
        self,
        features: pd.DataFrame,
        target_col: str = "next_return",
        target_horizon: int = 1,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(features, target_col, target_horizon).transform(features)

    def get_regime_importance(self, regime: str) -> Optional[pd.DataFrame]:
        """Return per-regime importance DataFrame (or None if not computed)."""
        return self.regime_importance_.get(regime)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_candidate_cols(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Return numeric columns excluding OHLCV, regime labels, and target."""
        exclude = _OHLCV | {target_col}
        if self.regime_column:
            exclude.add(self.regime_column)
        return [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    def _compute_importance(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        """Dispatch to the chosen importance method."""
        if self.method == "rf":
            return self._rf_importance(X, y, feature_names)
        return self._mi_importance(X, y)

    def _rf_importance(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        try:
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X, y)
            return rf.feature_importances_
        except ImportError:
            logger.warning("scikit-learn not available; falling back to mutual information.")
            return self._mi_importance(X, y)

    def _mi_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            from sklearn.feature_selection import mutual_info_regression

            scores = mutual_info_regression(X, y, random_state=self.random_state)
            return scores
        except ImportError:
            # Pure-Python fallback: variance-based proxy
            logger.warning("scikit-learn not available; using variance proxy for importance.")
            return np.var(X, axis=0)

    def _compute_regime_importance(
        self,
        df: pd.DataFrame,
        target_col: str,
        candidate_cols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        regime_col = self.regime_column
        result: Dict[str, pd.DataFrame] = {}
        regimes = df[regime_col].dropna().unique()

        for regime in regimes:
            regime_df = df[df[regime_col] == regime]
            if len(regime_df) < 30:
                continue
            y_r = regime_df[target_col].values
            X_r = regime_df[candidate_cols].fillna(0).values
            scores = self._compute_importance(X_r, y_r, candidate_cols)
            result[str(regime)] = (
                pd.DataFrame({"feature": candidate_cols, "importance": scores})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        return result
