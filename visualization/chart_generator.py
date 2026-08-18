"""
Session-level chart generation.

``ChartGenerator`` binds an output directory (mirroring
``./research_output/charts/`` from the README) and generates whichever
charts the caller has data for, skipping the rest — the autonomous pipeline
calls this once per session/window without needing to know which charts
are applicable.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from visualization import charts

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generates and saves all research charts under a bound output directory."""

    def __init__(self, output_dir: Union[str, Path] = "./research_output/charts"):
        """
        Args:
            output_dir: Directory PNG charts are written into.
        """
        self.output_dir = Path(output_dir)

    def _path(self, filename: str) -> str:
        return str(self.output_dir / filename)

    def generate_session_charts(self, session_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate every chart there is data for in ``session_data``.

        Args:
            session_data: Optional keys, each triggering the matching chart
                when present:
                ``equity_curve`` (pd.Series), ``human_equity_curve`` (pd.Series),
                ``initial_capital`` (float), ``rolling_accuracy`` (pd.Series),
                ``sharpe_over_time`` (pd.Series),
                ``parameter_history`` (list[dict]),
                ``optimizer_comparison_table`` (pd.DataFrame),
                ``convergence_curves`` (dict[str, list[float]]).

        Returns:
            Dict mapping chart name to the saved file path, for every chart
            that was actually generated.
        """
        generated: Dict[str, str] = {}

        equity_curve: Optional[pd.Series] = session_data.get("equity_curve")
        if equity_curve is not None and len(equity_curve) > 0:
            generated["equity_curve"] = charts.plot_equity_curve(
                equity_curve, self._path("equity_curve.png"),
                human_equity_curve=session_data.get("human_equity_curve"),
            )
            generated["profit_curve"] = charts.plot_profit_curve(
                equity_curve, self._path("profit_curve.png"),
                initial_capital=session_data.get("initial_capital"),
            )
            generated["drawdown_curve"] = charts.plot_drawdown_curve(
                equity_curve, self._path("drawdown_curve.png"),
            )

        rolling_accuracy: Optional[pd.Series] = session_data.get("rolling_accuracy")
        if rolling_accuracy is not None and len(rolling_accuracy) > 0:
            generated["rolling_accuracy"] = charts.plot_rolling_accuracy(
                rolling_accuracy, self._path("rolling_accuracy.png"),
            )

        sharpe_over_time: Optional[pd.Series] = session_data.get("sharpe_over_time")
        if sharpe_over_time is not None and len(sharpe_over_time) > 0:
            generated["sharpe_over_time"] = charts.plot_sharpe_over_time(
                sharpe_over_time, self._path("sharpe_over_time.png"),
            )

        parameter_history = session_data.get("parameter_history")
        if parameter_history:
            generated["parameter_evolution"] = charts.plot_parameter_evolution(
                parameter_history, self._path("parameter_evolution.png"),
            )

        comparison_table: Optional[pd.DataFrame] = session_data.get("optimizer_comparison_table")
        if comparison_table is not None and not comparison_table.empty:
            generated["optimizer_comparison"] = charts.plot_optimizer_comparison(
                comparison_table, self._path("optimizer_comparison.png"),
            )
            if "regime" in comparison_table.columns and comparison_table["regime"].nunique() > 1:
                generated["regime_comparison"] = charts.plot_regime_comparison(
                    comparison_table, self._path("regime_comparison.png"),
                )

        convergence_curves = session_data.get("convergence_curves")
        if convergence_curves:
            generated["convergence_curves"] = charts.plot_convergence_curves(
                convergence_curves, self._path("convergence_curves.png"),
            )

        logger.info("Generated %d charts in %s", len(generated), self.output_dir)
        return generated
