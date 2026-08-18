"""
Chart generation for research analysis.

Every chart renders with the non-interactive "Agg" matplotlib backend and
is saved straight to a PNG file — nothing is displayed interactively, so
this is safe to call from a headless autonomous pipeline run or CI.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import pandas as pd  # noqa: E402  pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)

FIGSIZE = (10, 6)
DPI = 120
GRID_ALPHA = 0.3

PathLike = Union[str, Path]


def _new_figure(title: str, xlabel: str, ylabel: str):
    """Create a styled figure/axes pair shared by every chart function."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=GRID_ALPHA)
    return fig, ax


def _save(fig, output_path: PathLike) -> str:
    """Write ``fig`` to ``output_path`` (creating parent dirs) and close it."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved chart to %s", path)
    return str(path)


def plot_equity_curve(
    equity_curve: pd.Series,
    output_path: PathLike,
    human_equity_curve: Optional[pd.Series] = None,
    title: str = "Equity Curve",
) -> str:
    """Plot portfolio value over time, optionally overlaying human vs. ML.

    Args:
        equity_curve: Portfolio value series (the ML curve, if both are given).
        output_path: PNG file to write.
        human_equity_curve: Optional human-baseline equity curve to overlay.
        title: Chart title.

    Returns:
        The path the chart was saved to.
    """
    fig, ax = _new_figure(title, "Time", "Portfolio Value")
    label = "ML" if human_equity_curve is not None else "Equity"
    ax.plot(equity_curve.index, equity_curve.values, label=label, linewidth=1.5)
    if human_equity_curve is not None:
        ax.plot(
            human_equity_curve.index, human_equity_curve.values,
            label="Human", linewidth=1.5, linestyle="--",
        )
        ax.legend()
    return _save(fig, output_path)


def plot_profit_curve(
    equity_curve: pd.Series,
    output_path: PathLike,
    initial_capital: Optional[float] = None,
    title: str = "Profit Curve",
) -> str:
    """Plot cumulative profit (equity minus starting capital) over time."""
    base = initial_capital if initial_capital is not None else float(equity_curve.iloc[0])
    profit = equity_curve - base
    fig, ax = _new_figure(title, "Time", "Profit")
    ax.plot(profit.index, profit.values, color="tab:green")
    ax.axhline(0, color="black", linewidth=0.8)
    return _save(fig, output_path)


def plot_drawdown_curve(
    equity_curve: pd.Series,
    output_path: PathLike,
    title: str = "Drawdown Curve",
) -> str:
    """Plot drawdown (% below running peak) over time."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max.replace(0, pd.NA)
    drawdown = drawdown.fillna(0.0)

    fig, ax = _new_figure(title, "Time", "Drawdown")
    ax.fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="tab:red", linewidth=1.0)
    return _save(fig, output_path)


def plot_rolling_accuracy(
    rolling_accuracy: pd.Series,
    output_path: PathLike,
    title: str = "Rolling Accuracy",
) -> str:
    """Plot a rolling accuracy series (values expected in [0, 1])."""
    fig, ax = _new_figure(title, "Step", "Accuracy")
    ax.plot(rolling_accuracy.index, rolling_accuracy.values, color="tab:blue")
    ax.set_ylim(0, 1)
    return _save(fig, output_path)


def plot_sharpe_over_time(
    sharpe_series: pd.Series,
    output_path: PathLike,
    title: str = "Sharpe Ratio Over Time",
) -> str:
    """Plot a rolling/windowed Sharpe ratio series over time."""
    fig, ax = _new_figure(title, "Time", "Sharpe Ratio")
    ax.plot(sharpe_series.index, sharpe_series.values, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8)
    return _save(fig, output_path)


def plot_parameter_evolution(
    parameter_history: Sequence[Dict[str, float]],
    output_path: PathLike,
    title: str = "Parameter Evolution",
) -> str:
    """Plot how each optimized parameter's value changed run-over-run."""
    df = pd.DataFrame(list(parameter_history))
    fig, ax = _new_figure(title, "Optimization Run", "Parameter Value")
    for col in df.columns:
        ax.plot(df.index, df[col], marker="o", markersize=3, label=col)
    if len(df.columns) > 0:
        ax.legend(fontsize=8, ncol=2)
    return _save(fig, output_path)


def plot_optimizer_comparison(
    comparison_table: pd.DataFrame,
    output_path: PathLike,
    value_col: str = "mean",
    title: str = "Optimizer Comparison",
) -> str:
    """Bar chart of mean objective value achieved per optimizer.

    Args:
        comparison_table: Output of
            ``evaluation.research_metrics.build_optimizer_comparison_table``.
        output_path: PNG file to write.
        value_col: Column to aggregate/plot.
        title: Chart title.
    """
    fig, ax = _new_figure(title, "Optimizer", value_col)
    grouped = comparison_table.groupby("optimizer")[value_col].mean().sort_values(ascending=False)
    ax.bar(grouped.index, grouped.values, color="tab:orange")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, output_path)


def plot_regime_comparison(
    comparison_table: pd.DataFrame,
    output_path: PathLike,
    value_col: str = "mean",
    title: str = "Regime Comparison",
) -> str:
    """Bar chart of mean objective value achieved per market regime.

    Args:
        comparison_table: Output of
            ``evaluation.research_metrics.build_optimizer_comparison_table``.
        output_path: PNG file to write.
        value_col: Column to aggregate/plot.
        title: Chart title.
    """
    fig, ax = _new_figure(title, "Regime", value_col)
    grouped = comparison_table.groupby("regime")[value_col].mean().sort_values(ascending=False)
    ax.bar(grouped.index, grouped.values, color="tab:cyan")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, output_path)


def plot_convergence_curves(
    convergence_curves: Dict[str, Sequence[float]],
    output_path: PathLike,
    title: str = "Optimizer Convergence",
) -> str:
    """Overlay running-best-objective curves for one or more optimizers.

    Args:
        convergence_curves: Mapping of optimizer name to its
            ``improvement_curve`` (see
            ``evaluation.research_metrics.calculate_optimizer_convergence``).
        output_path: PNG file to write.
        title: Chart title.
    """
    fig, ax = _new_figure(title, "Trial", "Best Objective So Far")
    for name, curve in convergence_curves.items():
        ax.plot(range(len(curve)), list(curve), label=name)
    if convergence_curves:
        ax.legend(fontsize=8)
    return _save(fig, output_path)
