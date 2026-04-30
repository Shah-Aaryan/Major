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

from analysis.comparison_report import generate_full_report
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

    return {}


def _min_iters_required(method_key: str) -> int:
    """Minimum n_iterations needed for method to function."""

    # skopt-based GP requires n_calls >= 10.
    if method_key == "bayesian_gp":
        return 10
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
    )

    # Load + preprocess
    raw = loader.load_csv(data_path)
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

    # Objective function for adjuster
    param_bounds = strategy.get_parameter_bounds()
    n_param_dims = len(param_bounds)

    def objective_func(_strategy_name: str, params: Dict[str, Any], train_data: pd.DataFrame) -> float:
        strategy.update_parameters(params)
        engine = BacktestEngine(config=bt_config)
        result = engine.run(strategy, train_data)
        return result.metrics.sharpe_ratio if result.metrics else 0.0

    adjuster = MLParameterAdjuster(
        objective_function=objective_func,
        strategy_bounds={strategy.__class__.__name__: param_bounds},
        verbose=False,
    )

    # Optimizer methods (skip known-incompatible multi-objective wrappers)
    implemented = [spec for spec in get_optimizer_registry(status="implemented")]
    # Skip multi-objective wrappers for this single-objective hybrid flow.
    # Also skip known-broken optimizer(s) so the sweep can complete.
    method_keys = [
        spec.key
        for spec in implemented
        if spec.key not in {"nsga_ii", "nsga_iii", "cma_es"}
    ]

    method_results: Dict[str, Dict[str, Any]] = {}

    best_method: Optional[str] = None
    best_score = float("-inf")
    best_params: Dict[str, Any] = human_params

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
                        continue
                # Respect overall budget: split trials across windows.
                # Example: n_trials=20, wf_windows=5 -> 4 iterations per window.
                base_per_window_iters = max(1, n_trials // max(1, wf_windows))
                min_required = _min_iters_required(method_key)
                if base_per_window_iters < min_required:
                    logger.warning(
                        "Skipping %s: per-window iterations=%s < required=%s. Increase --trials or reduce --wf-windows.",
                        method_key,
                        base_per_window_iters,
                        min_required,
                    )
                    continue

                per_window_iters = base_per_window_iters

                def optimize_func(train_df: pd.DataFrame) -> Dict[str, Any]:
                    extra_kwargs = _optimizer_kwargs_for_method(
                        method_key,
                        opt_config,
                        per_window_iters=per_window_iters,
                        n_param_dims=n_param_dims,
                    )
                    result = adjuster.optimize_strategy(
                        strategy_name=strategy.__class__.__name__,
                        train_data=train_df,
                        method=method_key,
                        human_params=human_params,
                        n_iterations=per_window_iters,
                        **extra_kwargs,
                    )
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
                    anchored=False,
                    min_train_size=100,
                )

                if wf is None or wf.aggregate_test_metrics is None:
                    raise RuntimeError("walk-forward returned no aggregate metrics")

                elapsed = time.time() - start
                score = wf.aggregate_test_metrics.sharpe_ratio
                final_params = wf.windows[-1].optimized_params if wf.windows else human_params

                method_results[method_key] = {
                    "mean_improvement_pct": float(wf.avg_ml_improvement * 100.0),
                    "ml_helped_rate": float(wf.ml_consistency),
                    "mean_time_seconds": float(elapsed),
                    "score_sharpe": float(score),
                    "final_params": final_params,
                }

                if score > best_score:
                    best_score = score
                    best_method = method_key
                    best_params = final_params

            except Exception as e:
                elapsed = time.time() - start
                logger.warning("Method %s failed after %.1fs: %s", method_key, elapsed, e)
    finally:
        bt_logger.setLevel(prev_bt_level)

    if best_method is None:
        raise RuntimeError("All optimization methods failed; cannot produce hybrid result")

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
                "mean_improvement_pct": v.get("mean_improvement_pct", 0.0),
                "ml_helped_rate": v.get("ml_helped_rate", 0.0),
                "mean_time_seconds": v.get("mean_time_seconds", 0.0),
            }
            for k, v in method_results.items()
        },
        human_params=human_params,
        ml_params=best_params,
        data_period=data_period,
        best_method_override=best_method,
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
            k: {kk: vv for kk, vv in v.items() if kk != "final_params"}
            for k, v in method_results.items()
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
