"""
Research metrics that span multiple optimization runs.

``backtesting.metrics`` scores a single equity curve. These functions score
a *sequence* of optimization results: how stable the chosen parameters are
run-over-run, how quickly/reliably an optimizer converges, how accurate a
rolling stream of ML-vs-human comparisons has been, and how optimizers
stack up against each other across market regimes.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_parameter_stability(
    parameter_history: Sequence[Dict[str, float]]
) -> Dict[str, float]:
    """Score how stable optimized parameters are across successive runs.

    Uses the inverse of the coefficient of variation (std / mean) per
    parameter, squashed into ``(0, 1]`` so a constant parameter scores 1.0
    and increasingly volatile parameters approach 0.0.

    Args:
        parameter_history: One dict of parameter values per optimization
            run, e.g. the ``best_parameters`` from successive rolling-window
            optimizations, in chronological order.

    Returns:
        Dict mapping each parameter name to a stability score in ``(0, 1]``,
        plus an ``"overall"`` key with the mean stability across parameters.
        Empty dict if ``parameter_history`` is empty.
    """
    if not parameter_history:
        return {}

    df = pd.DataFrame(list(parameter_history))
    stability: Dict[str, float] = {}

    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        mean = series.mean()
        std = series.std(ddof=0)

        if std == 0:
            cv = 0.0
        elif mean == 0:
            cv = float("inf")
        else:
            cv = abs(std / mean)

        stability[col] = 1.0 / (1.0 + cv) if np.isfinite(cv) else 0.0

    stability["overall"] = float(np.mean(list(stability.values()))) if stability else 0.0
    return stability


def calculate_optimizer_convergence(
    trial_objectives: Sequence[float],
    tolerance: float = 1e-4,
    patience: int = 10,
) -> Dict[str, Any]:
    """Measure how quickly and reliably an optimizer's best score plateaus.

    Args:
        trial_objectives: Raw (not running-best) objective value observed at
            each trial, in the order trials were evaluated.
        tolerance: Maximum spread within a trailing window of ``patience``
            running-best values to be considered "converged".
        patience: Number of consecutive trials the running-best must stay
            within ``tolerance`` to declare convergence.

    Returns:
        Dict with ``converged`` (bool), ``convergence_trial`` (int or None —
        the trial index at which the plateau began), ``total_trials``,
        ``final_objective``, ``convergence_speed`` (fraction of the budget
        used before converging; 1.0 if it never converged), and
        ``improvement_curve`` (the running-best series, for plotting).
    """
    if not trial_objectives:
        return {
            "converged": False,
            "convergence_trial": None,
            "total_trials": 0,
            "final_objective": 0.0,
            "convergence_speed": 1.0,
            "improvement_curve": [],
        }

    running_best: List[float] = []
    best = -np.inf
    for value in trial_objectives:
        best = max(best, value)
        running_best.append(best)

    convergence_trial: Optional[int] = None
    for i in range(patience, len(running_best)):
        window = running_best[i - patience: i + 1]
        if (max(window) - min(window)) <= tolerance:
            convergence_trial = i - patience
            break

    total_trials = len(trial_objectives)
    return {
        "converged": convergence_trial is not None,
        "convergence_trial": convergence_trial,
        "total_trials": total_trials,
        "final_objective": running_best[-1],
        "convergence_speed": (
            convergence_trial / total_trials if convergence_trial is not None else 1.0
        ),
        "improvement_curve": running_best,
    }


def calculate_rolling_accuracy(
    predictions: Sequence[Any],
    actuals: Sequence[Any],
    window: int = 20,
) -> pd.Series:
    """Rolling accuracy of a categorical prediction stream.

    Typical use: ``predictions`` is "ml" or "human" — whichever the ML
    optimizer *expected* to outperform on each rolling window — and
    ``actuals`` is which one *actually* outperformed once realized.

    Args:
        predictions: Predicted/expected labels, one per evaluation step.
        actuals: Realized/observed labels, same length as ``predictions``.
        window: Trailing window size for the rolling accuracy calculation.

    Returns:
        Series of rolling accuracy in ``[0, 1]``, same length as the inputs.

    Raises:
        ValueError: If ``predictions`` and ``actuals`` differ in length.
    """
    if len(predictions) != len(actuals):
        raise ValueError(
            f"predictions ({len(predictions)}) and actuals ({len(actuals)}) "
            f"must be the same length"
        )

    correct = pd.Series([1 if p == a else 0 for p, a in zip(predictions, actuals)], dtype=float)
    return correct.rolling(window=window, min_periods=1).mean()


def build_optimizer_comparison_table(
    results: Sequence[Dict[str, Any]],
    objective_key: str = "objective_value",
    optimizer_key: str = "optimizer",
    regime_key: str = "regime",
) -> pd.DataFrame:
    """Aggregate optimizer results into a mean/std/count comparison table.

    Args:
        results: Flat list of optimization result records, each containing
            at least ``optimizer_key``, ``objective_key``, and (optionally)
            ``regime_key``.
        objective_key: Key holding the numeric objective value to aggregate.
        optimizer_key: Key holding the optimizer name.
        regime_key: Key holding the market regime label. Records missing
            this key are grouped under ``"all"``.

    Returns:
        DataFrame indexed by ``(optimizer, regime)`` with columns
        ``mean``, ``std``, ``count``, ``best``, sorted by ``mean`` descending.
        Empty DataFrame if ``results`` is empty.
    """
    if not results:
        return pd.DataFrame(columns=["optimizer", "regime", "mean", "std", "count", "best"])

    df = pd.DataFrame(list(results))
    if regime_key not in df.columns:
        df[regime_key] = "all"
    df[regime_key] = df[regime_key].fillna("all")

    grouped = df.groupby([optimizer_key, regime_key])[objective_key].agg(
        mean="mean", std="std", count="count", best="max"
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    table = grouped.reset_index().rename(
        columns={optimizer_key: "optimizer", regime_key: "regime"}
    )
    return table.sort_values("mean", ascending=False).reset_index(drop=True)
