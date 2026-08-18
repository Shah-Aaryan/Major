"""Unit tests for the visualization package."""

from pathlib import Path

import pandas as pd
import pytest

from visualization import charts
from visualization.chart_generator import ChartGenerator


@pytest.fixture
def equity_curve() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=50, freq="min")
    values = [100_000.0 * (1.0002 ** i) for i in range(50)]
    return pd.Series(values, index=idx)


class TestChartFunctions:
    def test_plot_equity_curve_creates_file(self, tmp_path, equity_curve):
        out = charts.plot_equity_curve(equity_curve, tmp_path / "equity.png")
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_plot_equity_curve_with_human_overlay(self, tmp_path, equity_curve):
        human = equity_curve * 0.98
        out = charts.plot_equity_curve(equity_curve, tmp_path / "equity.png", human_equity_curve=human)
        assert Path(out).exists()

    def test_plot_profit_curve_creates_file(self, tmp_path, equity_curve):
        out = charts.plot_profit_curve(equity_curve, tmp_path / "profit.png")
        assert Path(out).exists()

    def test_plot_drawdown_curve_creates_file(self, tmp_path, equity_curve):
        out = charts.plot_drawdown_curve(equity_curve, tmp_path / "dd.png")
        assert Path(out).exists()

    def test_plot_rolling_accuracy_creates_file(self, tmp_path):
        series = pd.Series([0.5, 0.6, 0.7, 0.8])
        out = charts.plot_rolling_accuracy(series, tmp_path / "acc.png")
        assert Path(out).exists()

    def test_plot_sharpe_over_time_creates_file(self, tmp_path):
        series = pd.Series([1.0, 1.2, 0.8, 1.5])
        out = charts.plot_sharpe_over_time(series, tmp_path / "sharpe.png")
        assert Path(out).exists()

    def test_plot_parameter_evolution_creates_file(self, tmp_path):
        history = [{"rsi_period": 14}, {"rsi_period": 16}, {"rsi_period": 12}]
        out = charts.plot_parameter_evolution(history, tmp_path / "params.png")
        assert Path(out).exists()

    def test_plot_optimizer_comparison_creates_file(self, tmp_path):
        table = pd.DataFrame({
            "optimizer": ["bayesian", "random"],
            "regime": ["trending", "trending"],
            "mean": [1.5, 0.8],
        })
        out = charts.plot_optimizer_comparison(table, tmp_path / "opt_cmp.png")
        assert Path(out).exists()

    def test_plot_regime_comparison_creates_file(self, tmp_path):
        table = pd.DataFrame({
            "optimizer": ["bayesian", "bayesian"],
            "regime": ["trending", "sideways"],
            "mean": [1.5, 0.8],
        })
        out = charts.plot_regime_comparison(table, tmp_path / "regime_cmp.png")
        assert Path(out).exists()

    def test_plot_convergence_curves_creates_file(self, tmp_path):
        curves = {"bayesian": [0.1, 0.5, 0.9, 0.9], "random": [0.2, 0.3, 0.4, 0.5]}
        out = charts.plot_convergence_curves(curves, tmp_path / "conv.png")
        assert Path(out).exists()

    def test_creates_missing_parent_directories(self, tmp_path, equity_curve):
        nested = tmp_path / "a" / "b" / "c" / "equity.png"
        out = charts.plot_equity_curve(equity_curve, nested)
        assert Path(out).exists()


class TestChartGenerator:
    def test_generates_only_charts_with_available_data(self, tmp_path, equity_curve):
        generator = ChartGenerator(output_dir=tmp_path)
        result = generator.generate_session_charts({"equity_curve": equity_curve})

        assert "equity_curve" in result
        assert "profit_curve" in result
        assert "drawdown_curve" in result
        assert "rolling_accuracy" not in result
        for path in result.values():
            assert Path(path).exists()

    def test_generates_nothing_for_empty_session_data(self, tmp_path):
        generator = ChartGenerator(output_dir=tmp_path)
        result = generator.generate_session_charts({})
        assert result == {}

    def test_generates_full_bundle_when_all_data_present(self, tmp_path, equity_curve):
        generator = ChartGenerator(output_dir=tmp_path)
        session_data = {
            "equity_curve": equity_curve,
            "human_equity_curve": equity_curve * 0.97,
            "rolling_accuracy": pd.Series([0.5, 0.6, 0.7]),
            "sharpe_over_time": pd.Series([1.0, 1.1, 0.9]),
            "parameter_history": [{"rsi": 14}, {"rsi": 16}],
            "optimizer_comparison_table": pd.DataFrame({
                "optimizer": ["bayesian", "random"],
                "regime": ["trending", "sideways"],
                "mean": [1.2, 0.9],
            }),
            "convergence_curves": {"bayesian": [0.1, 0.5, 0.9]},
        }
        result = generator.generate_session_charts(session_data)

        expected_keys = {
            "equity_curve", "profit_curve", "drawdown_curve", "rolling_accuracy",
            "sharpe_over_time", "parameter_evolution", "optimizer_comparison",
            "regime_comparison", "convergence_curves",
        }
        assert expected_keys <= set(result.keys())
