"""Service layer for feature engineering operations."""

from __future__ import annotations

import logging
from pathlib import Path

from backend.models.request_models import FeatureRequest
from backend.models.response_models import FeatureListResponse, FeatureSummaryResponse
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.resampler import DataResampler
from features.feature_engine import FeatureEngine
from features.indicator_registry import get_indicator_registry

logger = logging.getLogger(__name__)


class FeatureService:
    """Expose the feature engine and indicator registry via the API."""

    def list_indicators(self, implemented_only: bool = False) -> FeatureListResponse:
        registry = get_indicator_registry(implemented_only=implemented_only)
        indicators = [
            {
                "name": spec.name,
                "category": spec.category,
                "implemented": spec.implemented,
                "notes": spec.notes,
                "tags": spec.tags or [],
            }
            for spec in registry
        ]
        total_all = len(registry)
        total_implemented = sum(1 for s in registry if s.implemented)
        return FeatureListResponse(
            indicators=indicators,
            total_implemented=total_implemented,
            total_planned=total_all - total_implemented,
        )

    def generate_feature_summary(self, payload: FeatureRequest) -> FeatureSummaryResponse:
        try:
            path = Path(payload.data_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            path = path.resolve()

            if not path.exists():
                raise FileNotFoundError(f"Data path '{path}' does not exist.")

            loader = DataLoader()
            preprocessor = DataPreprocessor(normalize_prices=False, normalize_volume=False)
            resampler = DataResampler()

            raw = loader.load_csv(str(path))
            df, _ = preprocessor.preprocess(raw)

            if payload.timeframe != "1m":
                df = resampler.resample(df, target_timeframe=payload.timeframe)

            engine = FeatureEngine()
            features = engine.generate_features(
                df,
                feature_groups=payload.feature_groups,
                drop_na=True,
            )

            # Count features per group
            group_map = {
                "price": ["return_", "log_return_", "momentum_", "rolling_"],
                "trend": ["sma_", "ema_", "adx", "trend_", "di_"],
                "momentum": ["rsi_", "macd", "stoch_", "cci", "roc_", "williams_r"],
                "volatility": ["atr_", "bb_", "kc_", "dc_", "volatility_", "vol_", "true_range"],
                "volume": ["volume_", "obv", "vwap", "mfi", "ad_line"],
                "regime": ["regime_", "vol_regime", "trend_regime", "mr_zscore"],
            }
            group_counts: dict[str, int] = {}
            for group, prefixes in group_map.items():
                count = sum(
                    1
                    for col in features.columns
                    if any(col.startswith(p) or p in col for p in prefixes)
                )
                if count:
                    group_counts[group] = count

            # Sample summary stats for the first few features
            sample_stats: dict = {}
            try:
                summary_df = features.describe().T
                sample_stats = {
                    col: {
                        "mean": float(summary_df.loc[col, "mean"]),
                        "std": float(summary_df.loc[col, "std"]),
                    }
                    for col in list(features.columns)[:10]
                    if col in summary_df.index
                }
            except Exception:
                pass

            return FeatureSummaryResponse(
                status="completed",
                data_path=str(path),
                timeframe=payload.timeframe,
                n_features=len(features.columns),
                n_rows=len(features),
                feature_groups=group_counts,
                sample_stats=sample_stats,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Feature summary failed for %s", payload.data_path)
            return FeatureSummaryResponse(
                status="failed",
                data_path=payload.data_path,
                timeframe=payload.timeframe,
                error=str(exc),
            )


feature_service = FeatureService()

