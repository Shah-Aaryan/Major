"""Service layer for optimization operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.models.request_models import OptimizationRequest
from backend.models.response_models import OptimizationResponse, OptimizersListResponse
from config.settings import BacktestConfig, OptimizationConfig
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.resampler import DataResampler
from features.feature_engine import FeatureEngine
from backtesting.backtest_engine import BacktestEngine
from optimization.optimizer_registry import get_optimizer_registry
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.bollinger_breakout import BollingerBreakoutStrategy

logger = logging.getLogger(__name__)

_STRATEGY_MAP = {
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "ema_crossover": EMACrossoverStrategy,
    "bollinger_breakout": BollingerBreakoutStrategy,
}


def _load_features(data_path: str, timeframe: str):
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
    df, _ = preprocessor.preprocess(raw)

    if timeframe != "1m":
        df = resampler.resample(df, target_timeframe=timeframe)

    engine = FeatureEngine()
    return engine.generate_features(df, drop_na=True)


class OptimizationService:
    """Wraps the optimizer registry and runs standalone optimization."""

    def list_optimizers(self) -> OptimizersListResponse:
        registry = get_optimizer_registry()
        optimizers = [
            {
                "key": spec.key,
                "name": spec.name,
                "category": spec.category,
                "status": spec.status,
                "notes": spec.notes,
                "supports_multiobjective": spec.supports_multiobjective,
            }
            for spec in registry
        ]
        return OptimizersListResponse(optimizers=optimizers)

    def run_optimization(self, payload: OptimizationRequest) -> OptimizationResponse:
        try:
            features = _load_features(payload.data_path, payload.timeframe)

            strategy_cls = _STRATEGY_MAP.get(payload.strategy)
            if not strategy_cls:
                raise ValueError(f"Unknown strategy: {payload.strategy}")

            strategy = strategy_cls()

            # Resolve optimizer class from registry
            registry = get_optimizer_registry(status="implemented")
            spec = next((s for s in registry if s.key == payload.optimizer), None)
            if spec is None:
                raise ValueError(
                    f"Unknown or unimplemented optimizer '{payload.optimizer}'. "
                    f"Available: {[s.key for s in registry]}"
                )
            optimizer_cls = spec.cls

            # Split data: 80% train, 20% test
            train_end = int(len(features) * 0.8)
            train_data = features.iloc[:train_end]
            test_data = features.iloc[train_end:]

            backtest_engine = BacktestEngine(config=BacktestConfig())

            # Build ParameterSpace from strategy bounds
            param_bounds = strategy.get_parameter_bounds()
            from optimization.base_optimizer import ParameterSpace
            param_space = ParameterSpace.from_strategy_bounds(
                param_bounds,
                integer_params=strategy.get_integer_params() if hasattr(strategy, "get_integer_params") else [],
            )

            def objective(params: dict[str, Any]) -> float:
                strategy.set_strategy_specific_params(params)
                result = backtest_engine.run(strategy=strategy, data=train_data)
                if result.metrics:
                    val = result.metrics.sharpe_ratio
                    import math
                    if math.isnan(val) or math.isinf(val):
                        return -999.0
                    return val
                return -999.0

            optimizer = optimizer_cls(
                parameter_space=param_space,
                objective_function=objective,
                maximize=True,
                n_iterations=payload.trials,
            )
            opt_result = optimizer.optimize()

            # Evaluate best params on test data
            best_params = opt_result.best_parameters if opt_result else {}
            test_metrics: dict[str, Any] = {}

            if best_params:
                strategy.set_strategy_specific_params(best_params)
                test_result = backtest_engine.run(strategy=strategy, data=test_data)
                if test_result.metrics:
                    test_metrics = test_result.metrics.to_dict()

            return OptimizationResponse(
                status="completed",
                strategy=payload.strategy,
                optimizer=payload.optimizer,
                best_params=best_params,
                best_score=opt_result.best_objective if opt_result else 0.0,
                n_trials=payload.trials,
                metrics=test_metrics,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Optimization failed")
            return OptimizationResponse(
                status="failed",
                strategy=payload.strategy,
                optimizer=payload.optimizer,
                error=str(exc),
            )



optimization_service = OptimizationService()

