"""
Research-specific performance evaluation.

Complements ``backtesting.metrics`` (Sharpe, Sortino, Calmar, drawdown, win
rate, profit factor, expectancy, CAGR, ...) with the metrics that only make
sense across multiple optimization runs: parameter stability, optimizer
convergence, and rolling accuracy of ML vs. human parameter selection.
"""

from evaluation.research_metrics import (
    calculate_optimizer_convergence,
    calculate_parameter_stability,
    calculate_rolling_accuracy,
    build_optimizer_comparison_table,
)

__all__ = [
    "calculate_parameter_stability",
    "calculate_optimizer_convergence",
    "calculate_rolling_accuracy",
    "build_optimizer_comparison_table",
]
