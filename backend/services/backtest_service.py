"""Service layer for backtesting and walk-forward validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.models.request_models import BacktestRequest, WalkForwardRequest
from backend.models.response_models import BacktestResponse, WalkForwardResponse
from config.settings import BacktestConfig, FeatureConfig, OptimizationConfig
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.resampler import DataResampler
from features.feature_engine import FeatureEngine
from backtesting.backtest_engine import BacktestEngine
from backtesting.walk_forward import WalkForwardValidator
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from optimization.optimizer_registry import get_optimizer_registry

logger = logging.getLogger(__name__)

_STRATEGY_MAP = {
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "ema_crossover": EMACrossoverStrategy,
    "bollinger_breakout": BollingerBreakoutStrategy,
}


def _load_and_prepare(data_path: str, timeframe: str):
    """Load CSV, preprocess, resample and generate features."""
    path = Path(data_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data path '{path}' does not exist.")

    loader = DataLoader()
    preprocessor = DataPreprocessor(normalize_prices=False, normalize_volume=False)
    resampler = DataResampler()

    raw = loader.load_csv(str(path))
    df, quality = preprocessor.preprocess(raw)

    if timeframe != "1m":
        df = resampler.resample(df, target_timeframe=timeframe)

    engine = FeatureEngine()
    features = engine.generate_features(df, drop_na=True)
    return features, quality


class BacktestService:
    """Wraps BacktestEngine for API consumption."""

    def run_backtest(self, payload: BacktestRequest) -> BacktestResponse:
        try:
            features, _ = _load_and_prepare(payload.data_path, payload.timeframe)

            strategy_cls = _STRATEGY_MAP.get(payload.strategy)
            if not strategy_cls:
                raise ValueError(f"Unknown strategy: {payload.strategy}")

            strategy = strategy_cls()
            if payload.params:
                strategy.set_strategy_specific_params(payload.params)

            engine = BacktestEngine(config=BacktestConfig())
            result = engine.run(
                strategy=strategy,
                data=features,
            )

            metrics_dict: dict[str, Any] = {}
            if result.metrics:
                metrics_dict = result.metrics.to_dict()

            return BacktestResponse(
                status="completed",
                strategy=payload.strategy,
                timeframe=payload.timeframe,
                metrics=metrics_dict,
                n_trades=metrics_dict.get("total_trades", 0),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Backtest failed")
            return BacktestResponse(
                status="failed",
                strategy=payload.strategy,
                timeframe=payload.timeframe,
                error=str(exc),
            )

    def run_walk_forward(self, payload: WalkForwardRequest) -> WalkForwardResponse:
        try:
            features, _ = _load_and_prepare(payload.data_path, payload.timeframe)

            strategy_cls = _STRATEGY_MAP.get(payload.strategy)
            if not strategy_cls:
                raise ValueError(f"Unknown strategy: {payload.strategy}")

            strategy = strategy_cls()

            # Find a suitable optimizer from registry
            registry = get_optimizer_registry(status="implemented")
            opt_spec = next((s for s in registry if "bayesian" in s.key), registry[0])
            optimizer_cls = opt_spec.cls

            opt_config = OptimizationConfig(
                bayesian_n_calls=payload.trials,
                random_search_n_iter=payload.trials,
            )

            validator = WalkForwardValidator(
                n_windows=payload.n_windows,
                train_ratio=payload.train_ratio,
                backtest_config=BacktestConfig(),
                optimize_each_window=True,
            )

            wf_result = validator.validate(
                data=features,
                strategy=strategy,
                optimizer_cls=optimizer_cls,
                optimization_config=opt_config,
            )

            window_results = []
            for wr in wf_result.window_results:
                window_results.append({
                    "window": wr.window_idx,
                    "train_sharpe": wr.train_metrics.sharpe_ratio if wr.train_metrics else None,
                    "test_sharpe": wr.test_metrics.sharpe_ratio if wr.test_metrics else None,
                    "ml_params": wr.optimized_params,
                })

            summary = wf_result.get_summary() if hasattr(wf_result, "get_summary") else {}

            return WalkForwardResponse(
                status="completed",
                strategy=payload.strategy,
                timeframe=payload.timeframe,
                n_windows=payload.n_windows,
                results=window_results,
                summary=summary,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Walk-forward validation failed")
            return WalkForwardResponse(
                status="failed",
                strategy=payload.strategy,
                timeframe=payload.timeframe,
                error=str(exc),
            )


backtest_service = BacktestService()

