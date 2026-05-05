"""Hybrid human+live window optimization flow.

Implements a workflow:
1) Human provides baseline strategy parameters.
2) Load historical OHLCV from CSV.
3) Split into N windows; replace the last K windows with newest market data from CoinGecko.
4) Generate features.
5) Run all implemented single-objective optimizers, pick best.
6) Backtest best ML params vs human params and generate a report.

This is designed to be invoked from main.py via `--hybrid-live`.
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from analysis.comparison_report import generate_full_report
from backtesting.metrics import PERIODS_PER_YEAR
from backtesting.backtest_engine import BacktestConfig as BtConfig, BacktestEngine
from config.settings import DataConfig, FeatureConfig, OptimizationConfig, StrategyConfig
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.resampler import DataResampler
from features.feature_engine import FeatureEngine
from optimization.ml_parameter_adjuster import MLParameterAdjuster
from optimization.optimizer_registry import get_optimizer_registry
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

logger = logging.getLogger(__name__)


_STRATEGY_FACTORY = {
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "ema_crossover": EMACrossoverStrategy,
    "bollinger_breakout": BollingerBreakoutStrategy,
}


def _timeframe_to_rule(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return f"{int(tf[:-1])}min"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}h"
    if tf.endswith("d"):
        return f"{int(tf[:-1])}d"
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _load_human_params(human_params_json: Optional[str], human_params_file: Optional[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if human_params_file:
        with open(human_params_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--human-params-file must contain a JSON object")
        params.update(loaded)

    if human_params_json:
        loaded = json.loads(human_params_json)
        if not isinstance(loaded, dict):
            raise ValueError("--human-params must be a JSON object")
        params.update(loaded)

    return params




def _to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame's DatetimeIndex to UTC-naive timestamps.

    The historical CSV loader produces tz-aware UTC timestamps; CoinGecko timestamps
    are typically tz-naive. Mixing them breaks concat/sort operations.
    """

    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    else:
        # Assume naive timestamps are UTC.
        df = df.copy()
        df.index = df.index.tz_localize("UTC").tz_localize(None)
    return df


def _replace_last_windows(
    historical_ohlcv: pd.DataFrame,
    live_ohlcv: pd.DataFrame,
    n_windows: int,
    replace_windows: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Replace the last `replace_windows` windows of historical data with live data."""

    historical_ohlcv = _to_utc_naive_index(historical_ohlcv)
    live_ohlcv = _to_utc_naive_index(live_ohlcv)

    meta: Dict[str, Any] = {
        "n_windows": n_windows,
        "replace_windows": replace_windows,
        "window_size_rows": None,
        "replaced_rows": 0,
        "live_rows_available": int(len(live_ohlcv)),
    }

    if historical_ohlcv.empty:
        return historical_ohlcv, meta

    if n_windows <= 0:
        return historical_ohlcv, meta

    window_size = max(1, len(historical_ohlcv) // n_windows)
    meta["window_size_rows"] = int(window_size)

    replace_windows = max(0, min(replace_windows, n_windows))
    replace_rows = replace_windows * window_size

    if replace_rows <= 0:
        return historical_ohlcv, meta

    if live_ohlcv.empty:
        logger.warning("No live data from CoinGecko; keeping historical data unchanged")
        return historical_ohlcv, meta

    live_tail = live_ohlcv.iloc[-min(replace_rows, len(live_ohlcv)) :].copy()
    if len(live_tail) < replace_rows:
        logger.warning(
            "Live data has fewer rows (%s) than needed to fully replace (%s). Partial replace only.",
            len(live_tail),
            replace_rows,
        )
        replace_rows = len(live_tail)

    base = historical_ohlcv.iloc[: max(0, len(historical_ohlcv) - replace_rows)].copy()

    combined = pd.concat([base, live_tail], axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    meta["replaced_rows"] = int(replace_rows)

    return combined, meta


def _feature_cache_path(
    data_path: str,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    sample_rows: int = 0,
) -> Path:
    """Build a stable on-disk cache path for generated feature CSVs.

    Includes `sample_rows` in the cache key so truncated datasets produce separate caches.
    """
    source = Path(data_path)
    raw_dir = source.parent
    cache_key = f"{source.resolve()}|{symbol.upper()}|{timeframe}|{strategy_name}|sample_rows={int(sample_rows)}"
    cache_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    cache_name = f"features_{symbol.upper()}_{timeframe}_{strategy_name}_{cache_hash}.csv"
    return raw_dir / cache_name


def _load_cached_features(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load a cached feature CSV if it exists."""
    if not cache_path.is_file():
        return None

    try:
        cached = pd.read_csv(cache_path, parse_dates=[0], index_col=0)
        if not isinstance(cached.index, pd.DatetimeIndex):
            cached.index = pd.to_datetime(cached.index)
        logger.info("Loaded cached feature CSV: %s", cache_path)
        return cached
    except Exception as e:
        logger.warning("Failed to load cached features from %s: %s", cache_path, e)
        return None


def _save_feature_cache(features: pd.DataFrame, cache_path: Path) -> None:
    """Persist generated features to a CSV file in the raw data folder."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(cache_path)
        logger.info("Saved generated features to: %s", cache_path)
    except Exception as e:
        logger.warning("Failed to save feature CSV to %s: %s", cache_path, e)


def _optimizer_kwargs_for_method(
    method_key: str,
    opt_config: OptimizationConfig,
    *,
    per_window_iters: int,
    n_param_dims: int,
) -> Dict[str, Any]:
    """Best-effort config -> kwargs mapping per optimizer method.

    Note: Some methods have *implicit* evaluation multipliers (population-based).
    For hybrid mode we scale those down when `per_window_iters` is small so the
    full sweep can finish in a reasonable time.
    """

    if method_key == "grid_search":
        return {"grid_resolution": opt_config.grid_resolution}

    if method_key in {"bayesian_gp", "bayesian_tpe"}:
        # Keep Bayesian initial random samples <= budget so small --trials still runs.
        n_initial_points = max(1, min(10, int(per_window_iters)))
        backend = "skopt" if method_key == "bayesian_gp" else "optuna"
        return {
            "backend": backend,
            "n_initial_points": n_initial_points,
        }

    if method_key in {"random_search", "latin_hypercube", "sobol"}:
        strategy = (
            method_key
            if method_key in {"latin_hypercube", "sobol"}
            else opt_config.random_search_strategy
        )
        return {"sampling_strategy": strategy}

    if method_key == "genetic_algorithm":
        # Scale population for small iteration budgets.
        population_size = min(opt_config.es_population_size, max(10, 5 * per_window_iters))
        return {
            "population_size": population_size,
            "mutation_prob": opt_config.es_mutation_rate,
            "crossover_prob": opt_config.es_crossover_rate,
        }

    if method_key == "differential_evolution":
        # SciPy DE uses popsize * ndim evals per generation.
        # Keep popsize small for hybrid sweeps.
        population_size = min(opt_config.de_population_size, max(3, 1 + per_window_iters))
        return {
            "population_size": population_size,
            "mutation": opt_config.de_mutation,
            "recombination": opt_config.de_recombination,
        }

    if method_key == "simulated_annealing":
        return {
            "initial_temp": opt_config.sa_initial_temp,
            "restart_temp_ratio": opt_config.sa_restart_temp_ratio,
        }

    if method_key == "particle_swarm":
        swarm_size = min(30, max(10, 5 * per_window_iters))
        return {"swarm_size": swarm_size}

    if method_key == "evolution_strategies":
        # ES evaluates `lambd` offspring per iteration.
        mu = min(10, max(3, per_window_iters + 2))
        lambd = min(40, max(10, mu * 4))
        return {"mu": mu, "lambd": lambd}

    if method_key == "hyperband_asha":
        # Keep defaults but ensure reduction_factor is sane.
        return {"reduction_factor": 3}

    if method_key in {"nsga_ii", "nsga_iii"}:
        # Population-based; keep it modest for hybrid sweeps.
        population_size = min(60, max(20, 5 * per_window_iters))
        return {"population_size": population_size}

    return {}


def _min_iters_required(method_key: str) -> int:
    """Minimum n_iterations needed for method to function."""

    # Allow small budgets; we clamp Bayesian n_initial_points in kwargs.
    if method_key in {"bayesian_gp", "bayesian_tpe"}:
        return 1
    return 1


def _run_backtest(strategy, data_with_features: pd.DataFrame, params: Dict[str, Any], bt_config: BtConfig) -> Dict[str, Any]:
    strategy.update_parameters(params)
    engine = BacktestEngine(config=bt_config)
    result = engine.run(strategy, data_with_features)
    return {
        "metrics": result.metrics.to_dict() if result.metrics else {},
        "trades": [t.to_dict() for t in result.trades] if result.trades else [],
        "equity_curve": result.equity_curve,
        "total_trades": len(result.trades) if result.trades else 0,
    }


@dataclass
class HybridRunArtifacts:
    result_json_path: str
    report_md_path: str


def run_hybrid_live_optimization(
    *,
    data_path: str,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    output_dir: str,
    n_trials: int,
    wf_windows: int,
    wf_train_ratio: float,
    replace_windows: int,
    coingecko_days: int,
    stream_seconds: int = 20,
    stream_interval_seconds: float = 5.0,
    sample_rows: int = 0,
    human_params_json: Optional[str] = None,
    human_params_file: Optional[str] = None,
) -> HybridRunArtifacts:
    """Run the hybrid workflow for a single (symbol,timeframe,strategy)."""

    if strategy_name not in _STRATEGY_FACTORY:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Config + modules
    data_config = DataConfig()
    loader = DataLoader(data_config.data_dir)
    preprocessor = DataPreprocessor()
    resampler = DataResampler()
    feature_engine = FeatureEngine(FeatureConfig())

    opt_config = OptimizationConfig(
        bayesian_n_calls=n_trials,
        random_search_n_iter=n_trials,
    )

    strategy_config = StrategyConfig()
    bt_config = BtConfig(
        initial_capital=strategy_config.initial_capital,
        commission_pct=strategy_config.trading_fee_pct,
        slippage_pct=0.0005,
        periods_per_year=PERIODS_PER_YEAR.get(timeframe, 252 * 24 * 60)
    )

    # Load + preprocess
    raw = loader.load_csv(data_path)
    data_stem = Path(data_path).stem.upper()
    if symbol.upper() not in data_stem:
        logger.warning(
            "Symbol mismatch: requested symbol '%s' does not appear in data file '%s'. "
            "Regime and compatibility results will reflect the loaded file, not the symbol name.",
            symbol,
            Path(data_path).name,
        )
    raw, _quality = preprocessor.preprocess(raw, symbol=symbol)

    if timeframe != "1m":
        raw = resampler.resample(raw, timeframe)

    # Optionally use a truncated dataset (keep last N rows) to reduce memory/time
    if sample_rows and sample_rows > 0:
        if len(raw) > sample_rows:
            logger.info("Truncating dataset to last %s rows (requested sample_rows=%s)", sample_rows, sample_rows)
            raw = raw.iloc[-int(sample_rows):].copy()

    # CoinGecko integration intentionally disabled for hybrid mode.
    # Hybrid now runs purely on local historical data to avoid API-key/rate-limit issues.
    live = pd.DataFrame()
    hybrid_raw, replace_meta = _replace_last_windows(
        historical_ohlcv=raw,
        live_ohlcv=live,
        n_windows=wf_windows,
        replace_windows=replace_windows,
    )

    replace_meta["live_source"] = "disabled"
    replace_meta["live_disabled_reason"] = "coingecko_removed_for_hybrid_mode"

    logger.info(
        "Hybrid window replace: %s",
        replace_meta,
    )

    # Generate or reuse features from a CSV cache in the raw data folder.
    feature_cache_path = _feature_cache_path(data_path, symbol, timeframe, strategy_name, sample_rows=sample_rows)
    data_with_features = _load_cached_features(feature_cache_path)

    if data_with_features is None:
        data_with_features = feature_engine.generate_features(hybrid_raw, drop_na=False)
        data_with_features = data_with_features.ffill().bfill()
        data_with_features = data_with_features.dropna(subset=["open", "high", "low", "close", "volume"])
        _save_feature_cache(data_with_features, feature_cache_path)
    else:
        # Ensure the cached CSV is usable as-is for backtesting and optimization.
        if not isinstance(data_with_features.index, pd.DatetimeIndex):
            data_with_features.index = pd.to_datetime(data_with_features.index)
        data_with_features = data_with_features.sort_index()
        data_with_features = data_with_features.ffill().bfill()
        data_with_features = data_with_features.dropna(subset=["open", "high", "low", "close", "volume"])

    if len(data_with_features) < 500:
        raise ValueError(
            f"Insufficient data after feature generation: {len(data_with_features)} rows (need 500+)"
        )

    # Strategy + human params
    strategy = _STRATEGY_FACTORY[strategy_name]()
    base_human = strategy.parameters.to_dict()
    overrides = _load_human_params(human_params_json, human_params_file)
    human_params = dict(base_human)
    human_params.update(overrides)
    
    # ⚠️ COMPATIBILITY CHECK: Detect market regime and check strategy suitability
    from analysis.strategy_compatibility import early_warning_check
    available_strategies = list(_STRATEGY_FACTORY.keys())
    is_compatible, compat_message, best_alternative = early_warning_check(
        data_with_features,
        strategy_name,
        available_strategies
    )
    
    logger.warning(compat_message)
    
    if not is_compatible and best_alternative:
        logger.warning(
            "Suggested alternative strategy: '%s' (Score: %.0f%%)",
            best_alternative.strategy_name,
            best_alternative.compatibility_score * 100.0,
        )
        # You could add a config option to auto-switch or abort here
        # For now, we continue with warning

    # Objective function for adjuster
    param_bounds = strategy.get_parameter_bounds()
    n_param_dims = len(param_bounds)

    def objective_func(_strategy_name: str, params: Dict[str, Any], train_data: pd.DataFrame) -> float:
        """Anti-overfit objective: evaluates on an internal validation slice.
        
        Uses an 80/20 internal split of the training window so the ML cannot
        memorize the exact training pattern. Only parameters that generalize
        within the training window are selected.
        """
        # === INTERNAL VALIDATION SPLIT (the key anti-overfit mechanism) ===
        # Use first 75% for internal training, last 25% as internal validation.
        # The ML optimizer must find params that work on the UNSEEN 25%, not the
        # training 75% it is optimizing on. This prevents memorization.
        n = len(train_data)
        if n >= 200:
            internal_train = train_data.iloc[:int(n * 0.75)]
            internal_val = train_data.iloc[int(n * 0.75):]
        else:
            # Too little data — fall back to full window
            internal_train = train_data
            internal_val = train_data

        strategy.update_parameters(params)
        engine = BacktestEngine(config=bt_config)

        # Score is based on VALIDATION performance, not training performance
        val_result = engine.run(strategy, internal_val)
        if val_result is None or val_result.metrics is None:
            return -1.0

        val_trades = len(val_result.trades) if getattr(val_result, "trades", None) else 0
        if val_trades < 2:
            return -0.5

        # Also run on training portion to penalize train/val gap (overfit detection)
        train_result = engine.run(strategy, internal_train)
        train_sharpe = float(getattr(train_result.metrics, "sharpe_ratio", 0.0) or 0.0) if train_result and train_result.metrics else 0.0

        # Extract validation metrics
        val_return = float(getattr(val_result.metrics, "total_return", 0.0) or 0.0)
        val_sharpe = float(getattr(val_result.metrics, "sharpe_ratio", 0.0) or 0.0)
        val_sortino = float(getattr(val_result.metrics, "sortino_ratio", 0.0) or 0.0)
        val_max_dd = abs(float(getattr(val_result.metrics, "max_drawdown", 0.0) or 0.0))
        val_win_rate = float(getattr(val_result.metrics, "win_rate", 0.0) or 0.0)

        # 1. Base Score: validation Sortino (downside-risk adjusted)
        base_score = val_sortino if not np.isinf(val_sortino) and val_sortino != 0 else val_sharpe

        # 2. Overfit Gap Penalty: if train >> val, the params are memorizing
        overfit_gap = max(0.0, train_sharpe - val_sharpe)
        overfit_penalty = overfit_gap * 0.5  # Penalize train/val divergence

        # 3. Over-trading Penalty (max 1 trade per 10 bars)
        candles = len(internal_val)
        trade_density = val_trades / max(candles, 1)
        density_penalty = max(0.0, (trade_density - 0.1)) * 20.0

        # 4. Drawdown Penalty (>10%)
        dd_penalty = max(0.0, val_max_dd - 0.10) * 10.0

        # 5. Win-Rate Penalty (<35% = noise chasing)
        win_penalty = max(0.0, 0.35 - val_win_rate) * 2.0

        final_score = base_score - overfit_penalty - density_penalty - dd_penalty - win_penalty

        # Never reward a parameter set with negative validation return
        if val_return < 0:
            final_score = min(final_score, -0.5)  # fixed penalty instead of amplification

        return final_score

    adjuster = MLParameterAdjuster(
        objective_function=objective_func,
        strategy_bounds={strategy.__class__.__name__: param_bounds},
        verbose=False,
    )

    # Optimizer methods
    # Run *all* implemented techniques in the registry, one after another.
    implemented = [spec for spec in get_optimizer_registry(status="implemented")]
    method_keys = [spec.key for spec in implemented]

    # Record per-method outcomes so the final report/JSON shows what ran.
    # status: success | skipped | failed
    method_results: Dict[str, Dict[str, Any]] = {}

    best_method: Optional[str] = None
    best_score = float("-inf")
    best_params: Dict[str, Any] = human_params
    best_explainability_report: Optional[str] = None

    # Walk-forward per method to choose best out-of-sample
    from backtesting.walk_forward import WalkForwardValidator

    bt_logger = logging.getLogger("backtesting.backtest_engine")
    prev_bt_level = bt_logger.level
    bt_logger.setLevel(logging.WARNING)

    try:
        for method_key in method_keys:
            start = time.time()
            try:
                # Grid search scales exponentially with number of parameters.
                # For this project, strategies typically have many tunable params
                # (base risk mgmt + strategy-specific), so grid search becomes infeasible.
                if method_key == "grid_search":
                    # Minimal meaningful resolution is 2; estimate combos and skip if too large.
                    min_resolution = 2
                    est_points = min_resolution ** max(1, n_param_dims)
                    max_points = 5000
                    if est_points > max_points:
                        logger.warning(
                            "Skipping grid_search: estimated %s combinations (min res=%s, dims=%s) exceeds cap=%s",
                            est_points,
                            min_resolution,
                            n_param_dims,
                            max_points,
                        )
                        method_results[method_key] = {
                            "status": "skipped",
                            "reason": f"grid_search_estimated_combinations_exceeds_cap (est={est_points}, cap={max_points})",
                        }
                        continue
                # Respect overall budget: split trials across windows.
                # Example: n_trials=20, wf_windows=5 -> 4 iterations per window.
                base_per_window_iters = max(1, n_trials // max(1, wf_windows))
                per_window_iters = base_per_window_iters
                min_required = _min_iters_required(method_key)
                if per_window_iters < min_required:
                    logger.info(
                        "Bumping %s per-window iterations from %s to %s to satisfy minimum.",
                        method_key,
                        per_window_iters,
                        min_required,
                    )
                    per_window_iters = min_required
                explainability_report = None

                def optimize_func(train_df: pd.DataFrame) -> Dict[str, Any]:
                    nonlocal explainability_report
                    extra_kwargs = _optimizer_kwargs_for_method(
                        method_key,
                        opt_config,
                        per_window_iters=per_window_iters,
                        n_param_dims=n_param_dims,
                    )

                    # NSGA-II/III are multi-objective in this codebase. For hybrid mode,
                    # we still pick a single parameter set by preferring Sortino while
                    # also minimizing drawdown.
                    if method_key in {"nsga_ii", "nsga_iii"}:
                        try:
                            from optimization.base_optimizer import ParameterSpace
                            from optimization.multi_objective_optimizer import (
                                MultiObjectiveOptimizer,
                                ObjectiveConfig,
                                ObjectiveType,
                            )

                            space = ParameterSpace.from_strategy_bounds(param_bounds, adjuster.integer_params)

                            def mo_objective(params: Dict[str, Any]) -> Dict[str, float]:
                                # Reuse the same internal split logic as objective_func.
                                n = len(train_df)
                                if n >= 200:
                                    internal_train = train_df.iloc[: int(n * 0.75)]
                                    internal_val = train_df.iloc[int(n * 0.75) :]
                                else:
                                    internal_train = train_df
                                    internal_val = train_df

                                strategy.update_parameters(params)
                                engine = BacktestEngine(config=bt_config)

                                val_result = engine.run(strategy, internal_val)
                                if val_result is None or val_result.metrics is None:
                                    return {
                                        ObjectiveType.SORTINO_RATIO.value: -1.0,
                                        ObjectiveType.MAX_DRAWDOWN.value: 1.0,
                                    }

                                val_trades = len(val_result.trades) if getattr(val_result, "trades", None) else 0
                                if val_trades < 2:
                                    return {
                                        ObjectiveType.SORTINO_RATIO.value: -0.5,
                                        ObjectiveType.MAX_DRAWDOWN.value: 1.0,
                                    }

                                train_result = engine.run(strategy, internal_train)
                                train_sharpe = (
                                    float(getattr(train_result.metrics, "sharpe_ratio", 0.0) or 0.0)
                                    if train_result and train_result.metrics
                                    else 0.0
                                )

                                val_return = float(getattr(val_result.metrics, "total_return", 0.0) or 0.0)
                                val_sharpe = float(getattr(val_result.metrics, "sharpe_ratio", 0.0) or 0.0)
                                val_sortino = float(getattr(val_result.metrics, "sortino_ratio", 0.0) or 0.0)
                                val_max_dd = abs(float(getattr(val_result.metrics, "max_drawdown", 0.0) or 0.0))
                                val_win_rate = float(getattr(val_result.metrics, "win_rate", 0.0) or 0.0)

                                # Apply the same penalties as the scalar objective by degrading Sortino.
                                base_score = val_sortino if not np.isinf(val_sortino) and val_sortino != 0 else val_sharpe
                                overfit_gap = max(0.0, train_sharpe - val_sharpe)
                                overfit_penalty = overfit_gap * 0.5

                                candles = len(internal_val)
                                trade_density = val_trades / max(candles, 1)
                                density_penalty = max(0.0, (trade_density - 0.1)) * 20.0

                                dd_penalty = max(0.0, val_max_dd - 0.10) * 10.0
                                win_penalty = max(0.0, 0.35 - val_win_rate) * 2.0

                                penalized_sortino = base_score - overfit_penalty - density_penalty - dd_penalty - win_penalty
                                if val_return < 0:
                                    penalized_sortino = min(penalized_sortino, -0.5)

                                return {
                                    ObjectiveType.SORTINO_RATIO.value: float(penalized_sortino),
                                    ObjectiveType.MAX_DRAWDOWN.value: float(val_max_dd),
                                }

                            objectives = [
                                ObjectiveConfig(ObjectiveType.SORTINO_RATIO, weight=1.0, priority=1),
                                ObjectiveConfig(ObjectiveType.MAX_DRAWDOWN, weight=1.0, priority=2),
                            ]

                            pop_size = int(extra_kwargs.get("population_size", 50))
                            optimizer = MultiObjectiveOptimizer(
                                parameter_space=space,
                                objectives=objectives,
                                objective_function=mo_objective,
                                n_iterations=int(per_window_iters),
                                population_size=pop_size,
                                backend="optuna",
                                random_state=None,
                                verbose=False,
                            )

                            mo_result = optimizer.optimize()
                            single = mo_result.to_single_objective_result(ObjectiveType.SORTINO_RATIO)
                            return single.best_parameters or human_params
                        except Exception as e:
                            logger.warning("NSGA optimization failed (%s): %s", method_key, e)
                            return human_params

                    result = adjuster.optimize_strategy(
                        strategy_name=strategy.__class__.__name__,
                        train_data=train_df,
                        method=method_key,
                        human_params=human_params,
                        n_iterations=per_window_iters,
                        **extra_kwargs,
                    )
                    # Capture explainability report from first window
                    if explainability_report is None and result.explainability_report_json:
                        explainability_report = result.explainability_report_json
                    return result.ml_params or human_params

                validator = WalkForwardValidator(
                    strategy=strategy,
                    optimize_function=optimize_func,
                    backtest_config=bt_config,
                    baseline_params=human_params,
                )

                wf = validator.run(
                    data=data_with_features,
                    n_windows=wf_windows,
                    train_ratio=wf_train_ratio,
                    anchored=True,   # Expanding window: each fold trains on ALL prior data
                    min_train_size=200,
                )

                if wf is None or wf.aggregate_test_metrics is None:
                    raise RuntimeError("walk-forward returned no aggregate metrics")

                elapsed = time.time() - start
                # Use Sortino for scoring as it matches the objective function better
                sortino = wf.aggregate_test_metrics.sortino_ratio
                score = sortino if sortino and not np.isinf(sortino) else wf.aggregate_test_metrics.sharpe_ratio
                final_params = wf.windows[-1].optimized_params if wf.windows else human_params

                method_results[method_key] = {
                    "status": "success",
                    "mean_improvement_pct": float(wf.avg_ml_improvement * 100.0),
                    "ml_helped_rate": float(wf.ml_consistency),
                    "mean_time_seconds": float(elapsed),
                    "score": float(score),
                    "score_metric": "sortino_ratio" if sortino and not np.isinf(sortino) else "sharpe_ratio",
                    # Keep old key for backward compatibility with any downstream consumers.
                    "score_sharpe": float(score),
                    "final_params": final_params,
                }

                if score > best_score:
                    best_score = score
                    best_method = method_key
                    best_params = final_params
                    if explainability_report:
                        best_explainability_report = explainability_report

            except Exception as e:
                elapsed = time.time() - start
                logger.warning("Method %s failed after %.1fs: %s", method_key, elapsed, e)
                method_results[method_key] = {
                    "status": "failed",
                    "mean_time_seconds": float(elapsed),
                    "error": str(e),
                }
    finally:
        bt_logger.setLevel(prev_bt_level)

    if best_method is None:
        raise RuntimeError("All optimization methods failed; cannot produce hybrid result")

    # Console summary so it's obvious that multiple optimizers were attempted.
    try:
        print("\n" + "=" * 60)
        print("OPTIMIZER COMPARISON (HYBRID MODE)")
        print("=" * 60)
        print("Method | Status | Score | ML Helped Rate | Time")
        print("-" * 60)
        for k in method_keys:
            r = method_results.get(k, {"status": "unknown"})
            status = r.get("status", "unknown")
            score_val = r.get("score")
            score_metric = r.get("score_metric")
            score_str = ""
            if isinstance(score_val, (int, float)) and score_metric:
                score_str = f"{score_val:.4f} ({score_metric})"
            elif isinstance(score_val, (int, float)):
                score_str = f"{score_val:.4f}"

            helped_rate = r.get("ml_helped_rate")
            helped_str = f"{helped_rate:.0%}" if isinstance(helped_rate, (int, float)) else "-"
            t = r.get("mean_time_seconds")
            t_str = f"{t:.1f}s" if isinstance(t, (int, float)) else "-"
            print(f"{k} | {status} | {score_str or '-'} | {helped_str} | {t_str}")

        print("-" * 60)
        print(f"Best method: {best_method}")
        print("=" * 60 + "\n")
    except Exception as e:
        logger.warning("Failed to print optimizer comparison summary: %s", e)

    # Full backtests for reporting
    human_results = _run_backtest(strategy, data_with_features, human_params, bt_config)
    ml_results = _run_backtest(strategy, data_with_features, best_params, bt_config)

    # Report
    data_period = f"{data_with_features.index[0]} to {data_with_features.index[-1]}"
    report = generate_full_report(
        strategy_name=strategy_name,
        human_results=human_results,
        ml_results=ml_results,
        method_comparison={
            k: {
                "status": v.get("status", "unknown"),
                "mean_improvement_pct": v.get("mean_improvement_pct", 0.0),
                "ml_helped_rate": v.get("ml_helped_rate", 0.0),
                "mean_time_seconds": v.get("mean_time_seconds", 0.0),
                "score": v.get("score", None),
                "score_metric": v.get("score_metric", None),
                "reason": v.get("reason", None),
                "error": v.get("error", None),
            }
            for k, v in method_results.items()
        },
        human_params=human_params,
        ml_params=best_params,
        data_period=data_period,
        best_method_override=best_method,
        explainability_report_json=best_explainability_report,
        strategy_compatibility_report=compat_message,
    )

    out_dir = Path(output_dir)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = reports_dir / f"HYBRID_{symbol}_{timeframe}_{strategy_name}_{ts}.md"
    report.save(str(report_path), format="markdown")

    result_path = out_dir / f"hybrid_session_{ts}.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "mode": "hybrid_live",
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "data_path": data_path,
        "replace_meta": replace_meta,
        "wf_windows": wf_windows,
        "wf_train_ratio": wf_train_ratio,
        "coingecko_days": coingecko_days,
        "human_params": human_params,
        "best_method": best_method,
        "best_score_sharpe": best_score,
        "ml_params": best_params,
        "method_results": {
            # Keep full per-method output (including final_params) so you can compare
            # methods and reproduce the best parameters per optimizer.
            k: v for k, v in method_results.items()
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    # Also persist a template human params file to edit later
    template_path = out_dir / "human_params_template.json"
    if not template_path.exists():
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(human_params, f, indent=2)

    logger.info("Hybrid result JSON: %s", result_path)
    logger.info("Hybrid report: %s", report_path)

    return HybridRunArtifacts(result_json_path=str(result_path), report_md_path=str(report_path))
