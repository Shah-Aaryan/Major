"""
CLI Runner — single-command autonomous pipeline execution.

Usage examples
--------------
Run a fast benchmark with RSI strategy using Bayesian TPE optimisation:

    python cli.py run-benchmark --preset fast --strategy rsi_mean_reversion

Run a full research benchmark with all 15 optimisers on a CSV file:

    python cli.py run-benchmark --preset research --strategy ema_crossover \
        --data data/BTCUSDT_1h.csv --output ./research_output

Run fair 15-optimizer benchmark harness:

    python cli.py benchmark-15 --iterations 20 --seeds 3 --output ./benchmark_15_results

Run noise robustness stress-test:

    python cli.py stress-test --strategy rsi_mean_reversion \
        --data data/BTCUSDT_1h.csv --noise gaussian --noise-sigma 0.002

Run nested walk-forward validation:

    python cli.py nested-wfv --strategy rsi_mean_reversion \
        --data data/BTCUSDT_1h.csv --outer 5 --inner 3

All commands print a results summary to stdout and write artefacts
(charts, CSV, JSON, audit logs) to the ``--output`` directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def cmd_run_benchmark(args: argparse.Namespace) -> int:
    """Execute the autonomous pipeline with a named preset."""
    from config.settings import MarketDataConfig
    from features.preset_packs import get_preset
    from pipeline.autonomous_loop import AutonomousPipeline

    logger.info("Loading preset '%s' …", args.preset)
    feature_cfg, optim_kwargs = get_preset(args.preset)

    market_config = MarketDataConfig(
        DATA_SOURCE="dataset",
        CSV_PATH=args.data,
        SYMBOL=args.symbol,
        TIMEFRAME=args.timeframe,
    )

    if args.optimizer:
        optim_kwargs["optimization_method"] = args.optimizer
    if args.iterations:
        optim_kwargs["n_optimization_iterations"] = args.iterations

    pipeline = AutonomousPipeline(
        market_config=market_config,
        strategy_name=args.strategy,
        output_dir=args.output,
        audit_backend=args.audit_backend,
        **optim_kwargs,
    )

    logger.info(
        "Running pipeline: strategy=%s, preset=%s, optimizer=%s, max_cycles=%d",
        args.strategy,
        args.preset,
        optim_kwargs.get("optimization_method"),
        args.cycles,
    )

    results = pipeline.run(max_cycles=args.cycles)
    logger.info("Pipeline complete. %d cycles executed.", len(results))
    print(f"\nResults written to: {args.output}")
    return 0


def cmd_benchmark_15(args: argparse.Namespace) -> int:
    """Execute fair 15-optimizer benchmark harness."""
    from analysis.optimizer_benchmark_harness import OptimizerBenchmarkHarness

    logger.info("Running 15-Optimizer Benchmark Harness: iterations=%d, seeds=%d ...", args.iterations, args.seeds)
    harness = OptimizerBenchmarkHarness(output_dir=args.output)
    df_res = harness.run_benchmark(
        n_iterations=args.iterations,
        n_seeds=args.seeds,
        strategy_name=args.strategy,
        objective_name=args.objective_name
    )

    print("\n" + "=" * 60)
    print(" 15-OPTIMIZER BENCHMARK HARNESS SUMMARY")
    print("=" * 60)
    for _, r in df_res.iterrows():
        print(f"#{r['rank']:<2} | {r['name']:<30} | Score: {r['mean_score']:.4f} (+/- {r['std_score']:.4f}) | Time: {r['mean_duration_sec']:.2f}s")
    print("=" * 60)
    print(f"Detailed CSV/JSON/MD reports saved to: {args.output}\n")
    return 0


def cmd_stress_test(args: argparse.Namespace) -> int:
    """Apply synthetic noise to market data and re-run the pipeline."""
    import pandas as pd

    from analysis.market_noise_generator import MarketNoiseGenerator
    from config.settings import MarketDataConfig
    from pipeline.autonomous_loop import AutonomousPipeline

    logger.info("Loading data from '%s' …", args.data)
    df = pd.read_csv(args.data, parse_dates=["timestamp"], index_col="timestamp")

    gen = MarketNoiseGenerator(seed=args.seed)

    if args.noise == "gaussian":
        noisy_df = gen.add_gaussian_noise(df, sigma_fraction=args.noise_sigma)
    elif args.noise == "fat_tail":
        noisy_df = gen.add_fat_tail_shocks(df, shock_probability=args.shock_prob)
    elif args.noise == "regime_shift":
        noisy_df = gen.add_regime_shift(df)
    elif args.noise == "full":
        noisy_df = gen.stress_test(df, gaussian_sigma=args.noise_sigma)
    else:
        logger.error("Unknown noise type '%s'.", args.noise)
        return 1

    noisy_path = Path(args.output) / "noisy_data.csv"
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    noisy_df.to_csv(noisy_path)
    logger.info("Noisy dataset saved to '%s'.", noisy_path)

    market_config = MarketDataConfig(
        DATA_SOURCE="dataset",
        CSV_PATH=str(noisy_path),
        SYMBOL=args.symbol,
        TIMEFRAME=args.timeframe,
    )

    pipeline = AutonomousPipeline(
        market_config=market_config,
        strategy_name=args.strategy,
        output_dir=args.output,
        audit_backend="local",
    )

    results = pipeline.run(max_cycles=args.cycles)
    logger.info("Stress-test complete. %d cycles executed.", len(results))
    print(f"\nStress-test results written to: {args.output}")
    return 0


def cmd_nested_wfv(args: argparse.Namespace) -> int:
    """Run nested walk-forward cross-validation."""
    import json
    import pandas as pd

    from backtesting.backtest_engine import BacktestConfig
    from backtesting.nested_walk_forward import NestedWalkForwardValidator
    from features.feature_engine import FeatureEngine
    from optimization.base_optimizer import ParameterSpace
    from optimization.random_search import RandomSearch
    from pipeline.autonomous_loop import STRATEGY_FACTORY

    logger.info("Loading data from '%s' …", args.data)
    df = pd.read_csv(args.data, parse_dates=["timestamp"], index_col="timestamp")

    logger.info("Generating features …")
    engine = FeatureEngine()
    features = engine.generate_features(df)

    logger.info("Initialising strategy '%s' …", args.strategy)
    if args.strategy not in STRATEGY_FACTORY:
        logger.error("Unknown strategy '%s'. Available: %s", args.strategy, list(STRATEGY_FACTORY))
        return 1

    strategy = STRATEGY_FACTORY[args.strategy]()
    bounds = strategy.get_parameter_bounds()
    param_space = ParameterSpace.from_strategy_bounds(bounds)

    def optimizer_factory(objective_fn, space):
        return RandomSearch(
            parameter_space=space,
            objective_function=objective_fn,
            n_iterations=args.inner_trials,
            random_state=42,
        )

    validator = NestedWalkForwardValidator(
        strategy=strategy,
        inner_optimizer_factory=optimizer_factory,
        parameter_space=param_space,
        backtest_config=BacktestConfig(),
    )

    logger.info("Running nested walk-forward: outer=%d, inner=%d …", args.outer, args.inner)
    result = validator.run(features, n_outer=args.outer, n_inner=args.inner)

    print(result.summary())

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "nested_wfv_result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    logger.info("Result saved to '%s'.", result_path)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="cli.py",
        description="Autonomous Algorithmic Trading & Optimization Pipeline CLI",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # --- run-benchmark ---
    bench = sub.add_parser("run-benchmark", help="Run the autonomous pipeline with a preset.")
    bench.add_argument("--preset", choices=["fast", "balanced", "research"], default="balanced")
    bench.add_argument("--strategy", default="rsi_mean_reversion",
                       help="Strategy name (rsi_mean_reversion | ema_crossover | bollinger_breakout)")
    bench.add_argument("--data", default="tests/fixtures/sample_ohlcv.csv", help="Path to CSV dataset.")
    bench.add_argument("--symbol", default="BTCUSDT")
    bench.add_argument("--timeframe", default="1h")
    bench.add_argument("--output", default="./research_output")
    bench.add_argument("--cycles", type=int, default=5, help="Max pipeline cycles.")
    bench.add_argument("--optimizer", default=None, help="Override preset optimizer.")
    bench.add_argument("--iterations", type=int, default=None, help="Override preset trial count.")
    bench.add_argument("--audit-backend", choices=["local", "blockchain"], default="local")

    # --- benchmark-15 ---
    b15 = sub.add_parser("benchmark-15", help="Fair 15-optimizer benchmark harness.")
    b15.add_argument("--strategy", default="rsi_mean_reversion")
    b15.add_argument("--objective-name", default="RSIMeanReversion")
    b15.add_argument("--iterations", type=int, default=10, help="Trial budget per optimizer.")
    b15.add_argument("--seeds", type=int, default=3, help="Number of random seeds.")
    b15.add_argument("--output", default="./benchmark_15_results")

    # --- stress-test ---
    stress = sub.add_parser("stress-test", help="Inject noise and run robustness pipeline.")
    stress.add_argument("--strategy", default="rsi_mean_reversion")
    stress.add_argument("--data", required=True, help="Clean CSV dataset.")
    stress.add_argument("--symbol", default="BTCUSDT")
    stress.add_argument("--timeframe", default="1h")
    stress.add_argument("--output", default="./stress_output")
    stress.add_argument("--cycles", type=int, default=3)
    stress.add_argument("--noise", choices=["gaussian", "fat_tail", "regime_shift", "full"],
                        default="gaussian")
    stress.add_argument("--noise-sigma", type=float, default=0.001)
    stress.add_argument("--shock-prob", type=float, default=0.005)
    stress.add_argument("--seed", type=int, default=42)

    # --- nested-wfv ---
    nwfv = sub.add_parser("nested-wfv", help="Nested walk-forward cross-validation.")
    nwfv.add_argument("--strategy", default="rsi_mean_reversion")
    nwfv.add_argument("--data", required=True, help="CSV dataset.")
    nwfv.add_argument("--output", default="./nested_wfv_output")
    nwfv.add_argument("--outer", type=int, default=5, help="Number of outer folds.")
    nwfv.add_argument("--inner", type=int, default=3, help="Number of inner folds.")
    nwfv.add_argument("--inner-trials", type=int, default=20, help="Inner optimizer trials.")

    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "run-benchmark": cmd_run_benchmark,
        "benchmark-15": cmd_benchmark_15,
        "stress-test": cmd_stress_test,
        "nested-wfv": cmd_nested_wfv,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    exit_code = fn(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
