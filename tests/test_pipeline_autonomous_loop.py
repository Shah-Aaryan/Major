"""Integration tests for the autonomous ML pipeline loop.

Exercises the real stack (market_data provider, FeatureEngine,
MLParameterAdjuster, ParallelStrategyExecutor, BacktestEngine, AuditLogger,
ChartGenerator) end-to-end against a synthetic OHLCV fixture, using the
fast dependency-light random_search optimizer and small window/interval
sizes so this stays fast and CI-friendly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import MarketDataConfig
from pipeline.autonomous_loop import STRATEGY_FACTORY, AutonomousPipeline

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LARGE_CSV = FIXTURES_DIR / "synthetic_large_ohlcv.csv"
SMALL_CSV = FIXTURES_DIR / "synthetic_small_ohlcv.csv"


def _write_synthetic_csv(path: Path, n: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    returns = rng.normal(0, 0.0006, n)
    close = 30_000 * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0003, n)))
    volume = rng.uniform(5, 50, n)
    df = pd.DataFrame({
        "timestamp": idx.astype(np.int64) // 10**6,
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def large_csv_path() -> str:
    """A 220-row synthetic 1m OHLCV series — enough for one full optimization cycle."""
    if not LARGE_CSV.exists():
        _write_synthetic_csv(LARGE_CSV, n=220)
    return str(LARGE_CSV)


@pytest.fixture(scope="module")
def small_csv_path() -> str:
    """A 40-row series, deliberately too short to ever trigger optimization."""
    if not SMALL_CSV.exists():
        _write_synthetic_csv(SMALL_CSV, n=40)
    return str(SMALL_CSV)


def _make_pipeline(csv_path: str, output_dir: Path, **overrides) -> AutonomousPipeline:
    config = MarketDataConfig(
        DATA_SOURCE="dataset", CSV_PATH=csv_path, WINDOW_SIZE=80, SYMBOL="BTCUSDT",
    )
    defaults = dict(
        market_config=config,
        strategy_name="rsi_mean_reversion",
        optimization_method="random_search",
        n_optimization_iterations=5,
        optimization_interval=60,
        min_optimization_rows=60,
        output_dir=str(output_dir),
    )
    defaults.update(overrides)
    return AutonomousPipeline(**defaults)


class TestAutonomousPipelineConstruction:
    def test_rejects_unknown_strategy_name(self, large_csv_path, tmp_path):
        config = MarketDataConfig(DATA_SOURCE="dataset", CSV_PATH=large_csv_path)
        with pytest.raises(ValueError):
            AutonomousPipeline(
                market_config=config, strategy_name="not_a_real_strategy", output_dir=str(tmp_path),
            )

    def test_defaults_human_params_from_strategy(self, large_csv_path, tmp_path):
        pipeline = _make_pipeline(large_csv_path, tmp_path)
        assert pipeline.human_params
        assert pipeline.ml_params == pipeline.human_params

    def test_all_strategy_factory_entries_are_constructible(self, large_csv_path, tmp_path):
        for name in STRATEGY_FACTORY:
            pipeline = _make_pipeline(large_csv_path, tmp_path / name, strategy_name=name)
            assert pipeline.strategy_cls is STRATEGY_FACTORY[name]


class TestAutonomousPipelineRunCycle:
    def test_run_cycle_returns_none_while_window_warms_up(self, small_csv_path, tmp_path):
        pipeline = _make_pipeline(
            small_csv_path, tmp_path, min_optimization_rows=60, optimization_interval=60,
        )
        pipeline.initialize()
        try:
            for _ in range(40):  # small_csv_path only has 40 rows, well below min_optimization_rows
                result = pipeline.run_cycle()
                assert result is None
        finally:
            pipeline.shutdown()

    def test_run_cycle_returns_none_when_not_yet_due_for_optimization(self, large_csv_path, tmp_path):
        pipeline = _make_pipeline(
            large_csv_path, tmp_path, min_optimization_rows=60, optimization_interval=60,
        )
        pipeline.initialize()
        try:
            results = [pipeline.run_cycle() for _ in range(65)]
            non_none = [r for r in results if r is not None]
            assert len(non_none) == 1
        finally:
            pipeline.shutdown()


class TestAutonomousPipelineFullRun:
    def test_full_run_produces_one_complete_cycle(self, large_csv_path, tmp_path):
        pipeline = _make_pipeline(large_csv_path, tmp_path)
        results = pipeline.run(max_cycles=1)

        assert len(results) == 1
        result = results[0]

        assert result.regime  # non-empty regime label
        assert isinstance(result.ml_params, dict) and result.ml_params
        assert result.execution.ml_equity_curve is not None
        assert len(result.execution.ml_equity_curve) > 0
        assert result.execution.human_equity_curve is not None
        assert "overall" in result.parameter_stability
        assert 0.0 <= result.parameter_stability["overall"] <= 1.0
        assert result.rolling_accuracy is not None
        assert 0.0 <= result.rolling_accuracy <= 1.0

    def test_full_run_writes_chart_files(self, large_csv_path, tmp_path):
        pipeline = _make_pipeline(large_csv_path, tmp_path)
        results = pipeline.run(max_cycles=1)

        chart_paths = results[0].chart_paths
        assert "equity_curve" in chart_paths
        for path in chart_paths.values():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_full_run_writes_audit_log_files(self, large_csv_path, tmp_path):
        _make_pipeline(large_csv_path, tmp_path).run(max_cycles=1)

        audit_dir = tmp_path / "audit"
        jsonl_files = list(audit_dir.glob("*_events.jsonl"))
        assert len(jsonl_files) == 1
        assert jsonl_files[0].stat().st_size > 0

        content = jsonl_files[0].read_text(encoding="utf-8")
        assert "optimization_end" in content
        assert "regime_change" in content

    def test_second_cycle_extends_parameter_history(self, large_csv_path, tmp_path):
        pipeline = _make_pipeline(
            large_csv_path, tmp_path, optimization_interval=60, min_optimization_rows=60,
        )
        results = pipeline.run(max_cycles=2)

        assert len(results) == 2
        assert len(pipeline.parameter_history) == 2

    def test_provider_is_finished_after_full_run(self, small_csv_path, tmp_path):
        """Uses the 40-row fixture with a min_optimization_rows above its length,
        so the run drains the provider without ever triggering an (expensive)
        optimization cycle — this test only checks the run-loop's draining behavior.
        """
        pipeline = _make_pipeline(
            small_csv_path, tmp_path, min_optimization_rows=1000, optimization_interval=1000,
        )
        pipeline.run()
        assert pipeline.provider.is_finished()
        assert pipeline.run_cycle() is None
        assert pipeline._current_regime(pd.DataFrame()) == "unknown"
