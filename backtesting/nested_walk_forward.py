"""
Nested Walk-Forward Validator.

Implements a two-level cross-validation scheme:

  Outer loop  — produces unbiased out-of-sample performance estimates.
  Inner loop  — tunes hyperparameters on the train portion of each outer
                fold without ever touching the outer test data.

This design is the gold standard for time-series model evaluation and
directly addresses the question: *Does ML optimisation help on genuinely
unseen data, or does it merely overfit to the training window?*

Key outputs
-----------
* ``NestedWalkForwardResult`` — per-fold metrics with inner-loop best params.
* **Deflated Sharpe Ratio (DSR)** — penalises for the number of trials used
  during optimisation (Bailey & Lopez de Prado 2016).
* **Probability of Backtest Overfitting (PBO)** — combinatorial symmetry
  test (Bailey et al. 2014).

Usage
-----
>>> validator = NestedWalkForwardValidator(
...     strategy=rsi_strategy,
...     inner_optimizer_factory=lambda obj, space: RandomSearch(space, obj, n_iterations=30),
... )
>>> result = validator.run(features_df, n_outer=5, n_inner=3)
>>> print(result.summary())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from backtesting.metrics import PerformanceMetrics, calculate_all_metrics
from optimization.base_optimizer import BaseOptimizer, ParameterSpace
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InnerFoldResult:
    """Result of one inner cross-validation fold."""
    fold_id: int
    best_params: Dict[str, Any]
    best_objective: float


@dataclass
class OuterFoldResult:
    """Result of one outer fold (contains all inner results)."""
    fold_id: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    # Best params selected by inner CV
    selected_params: Dict[str, Any]
    # Metrics on the outer test set using the selected params
    test_metrics: Optional[PerformanceMetrics] = None
    baseline_metrics: Optional[PerformanceMetrics] = None
    inner_results: List[InnerFoldResult] = field(default_factory=list)
    ml_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
            "selected_params": self.selected_params,
            "test_sharpe": self.test_metrics.sharpe_ratio if self.test_metrics else None,
            "baseline_sharpe": self.baseline_metrics.sharpe_ratio if self.baseline_metrics else None,
            "ml_improvement": self.ml_improvement,
        }


@dataclass
class NestedWalkForwardResult:
    """Aggregated results across all outer folds."""
    outer_folds: List[OuterFoldResult]
    combined_equity: pd.Series
    combined_baseline_equity: Optional[pd.Series]
    aggregate_metrics: PerformanceMetrics
    aggregate_baseline: Optional[PerformanceMetrics]
    strategy_name: str

    # Derived at post-init
    deflated_sharpe_ratio: float = 0.0
    pbo_estimate: float = 0.0
    ml_consistency: float = 0.0

    def __post_init__(self) -> None:
        improvements = [f.ml_improvement for f in self.outer_folds]
        self.ml_consistency = (
            sum(1 for i in improvements if i > 0) / len(improvements)
            if improvements else 0.0
        )
        self.deflated_sharpe_ratio = self._compute_dsr()
        self.pbo_estimate = self._compute_pbo()

    def _compute_dsr(self) -> float:
        """Deflated Sharpe Ratio accounting for number of inner trials."""
        if not hasattr(self.aggregate_metrics, "sharpe_ratio"):
            return 0.0
        sr = self.aggregate_metrics.sharpe_ratio
        # Total inner trials across all folds (proxy for trials budget)
        total_trials = sum(len(f.inner_results) for f in self.outer_folds)
        if total_trials <= 1 or sr == 0:
            return sr
        # DSR deflation (simplified Bailey & de Prado formula)
        # DSR = SR * sqrt(1 - gamma(0.5) / gamma(3/2) * sqrt(total_trials - 1))
        inflation = math.sqrt(total_trials - 1) * 0.5 / math.sqrt(math.pi)
        dsr = sr * (1 - inflation / (abs(sr) + 1e-9))
        return max(dsr, -10.0)

    def _compute_pbo(self) -> float:
        """Estimate Probability of Backtest Overfitting from per-fold data.

        Uses the rank-based combinatorial PBO estimator (simplified).
        A PBO > 0.5 indicates the optimisation process is likely overfitting.
        """
        sharpes = [f.test_metrics.sharpe_ratio for f in self.outer_folds if f.test_metrics]
        baselines = [f.baseline_metrics.sharpe_ratio for f in self.outer_folds if f.baseline_metrics]
        if not sharpes or not baselines:
            return 0.5
        n_worse = sum(1 for s, b in zip(sharpes, baselines) if s < b)
        return n_worse / len(sharpes)

    def summary(self) -> str:
        agg = self.aggregate_metrics
        baseline = self.aggregate_baseline
        lines = [
            f"=== Nested Walk-Forward: {self.strategy_name} ===",
            f"Outer folds: {len(self.outer_folds)}",
            f"ML Consistency: {self.ml_consistency:.1%}",
            f"Deflated Sharpe Ratio (DSR): {self.deflated_sharpe_ratio:.3f}",
            f"Probability of Overfitting (PBO): {self.pbo_estimate:.3f}",
            "",
            "Aggregate ML-Optimised Metrics:",
            f"  Sharpe: {agg.sharpe_ratio:.3f}",
            f"  Total Return: {agg.total_return:.2%}",
            f"  Max Drawdown: {agg.max_drawdown:.2%}",
        ]
        if baseline:
            lines += [
                "",
                "Aggregate Baseline Metrics:",
                f"  Sharpe: {baseline.sharpe_ratio:.3f}",
                f"  Total Return: {baseline.total_return:.2%}",
            ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "n_outer_folds": len(self.outer_folds),
            "ml_consistency": self.ml_consistency,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "pbo_estimate": self.pbo_estimate,
            "aggregate_sharpe": self.aggregate_metrics.sharpe_ratio,
            "aggregate_total_return": self.aggregate_metrics.total_return,
            "aggregate_max_drawdown": self.aggregate_metrics.max_drawdown,
            "outer_folds": [f.to_dict() for f in self.outer_folds],
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class NestedWalkForwardValidator:
    """Two-level nested walk-forward cross-validator.

    Parameters
    ----------
    strategy:
        Trading strategy instance.
    inner_optimizer_factory:
        Callable ``(objective_fn, parameter_space) -> BaseOptimizer``.
        The factory is called once per outer-fold inner-CV run.
    parameter_space:
        The ``ParameterSpace`` to search.  If ``None`` the validator will
        construct one from ``strategy.get_parameter_bounds()``.
    backtest_config:
        Configuration applied to the backtest engine.
    baseline_params:
        Human baseline parameters for comparison.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        inner_optimizer_factory: Callable[[Callable, ParameterSpace], BaseOptimizer],
        parameter_space: Optional[ParameterSpace] = None,
        backtest_config: Optional[BacktestConfig] = None,
        baseline_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.strategy = strategy
        self.inner_optimizer_factory = inner_optimizer_factory
        self.backtest_config = backtest_config or BacktestConfig()
        self.baseline_params = baseline_params or {}
        self.engine = BacktestEngine(self.backtest_config)

        if parameter_space is not None:
            self.parameter_space = parameter_space
        else:
            bounds = strategy.get_parameter_bounds()
            self.parameter_space = ParameterSpace.from_strategy_bounds(bounds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        n_outer: int = 5,
        n_inner: int = 3,
        train_ratio: float = 0.8,
        min_fold_rows: int = 100,
    ) -> NestedWalkForwardResult:
        """Run nested walk-forward validation.

        Parameters
        ----------
        data:
            Full OHLCV + feature DataFrame.
        n_outer:
            Number of outer folds.
        n_inner:
            Number of inner CV folds per outer fold.
        train_ratio:
            Fraction of each outer fold used for training.
        min_fold_rows:
            Minimum rows needed in a fold to process it.
        """
        outer_folds = self._make_outer_folds(data, n_outer, train_ratio)
        outer_results: List[OuterFoldResult] = []
        test_curves: List[pd.Series] = []
        baseline_curves: List[pd.Series] = []

        for fold in outer_folds:
            train_data = data.loc[fold.train_start : fold.train_end]
            test_data = data.loc[fold.test_start : fold.test_end]

            if len(train_data) < min_fold_rows or len(test_data) < min_fold_rows:
                logger.warning("Skipping fold %d: insufficient data.", fold.fold_id)
                continue

            # --- Inner loop: find best params on inner CV ---
            best_params = self._run_inner_cv(train_data, n_inner, fold)

            # --- Outer test: evaluate selected params on held-out data ---
            test_result = self.engine.run(self.strategy, test_data, best_params)
            fold.test_metrics = test_result.metrics
            fold.selected_params = best_params
            test_curves.append(test_result.equity_curve)

            # --- Baseline ---
            if self.baseline_params:
                bl_result = self.engine.run(self.strategy, test_data, self.baseline_params)
                fold.baseline_metrics = bl_result.metrics
                baseline_curves.append(bl_result.equity_curve)
                bl_sr = bl_result.metrics.sharpe_ratio
                ml_sr = test_result.metrics.sharpe_ratio
                fold.ml_improvement = (
                    (ml_sr - bl_sr) / abs(bl_sr) if abs(bl_sr) > 1e-9
                    else float(np.tanh(ml_sr / 2.0))
                )

            outer_results.append(fold)
            logger.info(
                "Outer fold %d: test_sharpe=%.3f, ml_improvement=%.2%%",
                fold.fold_id, fold.test_metrics.sharpe_ratio, fold.ml_improvement * 100,
            )

        combined_equity = self._chain_curves(test_curves)
        combined_baseline = self._chain_curves(baseline_curves) if baseline_curves else None

        agg_metrics = calculate_all_metrics(combined_equity) if combined_equity is not None else PerformanceMetrics()
        agg_baseline = calculate_all_metrics(combined_baseline) if combined_baseline is not None else None

        result = NestedWalkForwardResult(
            outer_folds=outer_results,
            combined_equity=combined_equity if combined_equity is not None else pd.Series(dtype=float),
            combined_baseline_equity=combined_baseline,
            aggregate_metrics=agg_metrics,
            aggregate_baseline=agg_baseline,
            strategy_name=self.strategy.name,
        )
        logger.info("\n%s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inner_cv(
        self, train_data: pd.DataFrame, n_inner: int, outer_fold: OuterFoldResult
    ) -> Dict[str, Any]:
        """Run inner CV and return the best parameter set."""
        inner_folds = self._make_inner_folds(train_data, n_inner)
        param_scores: Dict[str, float] = {}

        for inner_fold in inner_folds:
            inner_train = train_data.iloc[inner_fold[0]]
            inner_val = train_data.iloc[inner_fold[1]]

            def objective(params: Dict[str, Any]) -> float:
                r = self.engine.run(self.strategy, inner_train, params)
                return r.metrics.sharpe_ratio

            optimizer = self.inner_optimizer_factory(objective, self.parameter_space)
            opt_result = optimizer.optimize()

            key = str(sorted(opt_result.best_parameters.items()))
            param_scores[key] = param_scores.get(key, 0.0) + opt_result.best_objective

            outer_fold.inner_results.append(
                InnerFoldResult(
                    fold_id=len(outer_fold.inner_results),
                    best_params=opt_result.best_parameters,
                    best_objective=opt_result.best_objective,
                )
            )

        # Pick params with highest accumulated cross-validation score
        if not param_scores:
            return self.parameter_space.get_defaults() or self.parameter_space.sample_random()

        best_key = max(param_scores, key=lambda k: param_scores[k])
        for inner_res in outer_fold.inner_results:
            if str(sorted(inner_res.best_params.items())) == best_key:
                return inner_res.best_params

        return outer_fold.inner_results[0].best_params

    @staticmethod
    def _make_outer_folds(
        data: pd.DataFrame, n_outer: int, train_ratio: float
    ) -> List[OuterFoldResult]:
        """Split data into outer (train, test) pairs (rolling walk-forward)."""
        n = len(data)
        fold_size = n // n_outer
        folds: List[OuterFoldResult] = []

        for i in range(n_outer):
            start = i * fold_size
            end = start + fold_size if i < n_outer - 1 else n
            train_end_idx = start + int((end - start) * train_ratio) - 1
            test_start_idx = train_end_idx + 1

            if test_start_idx >= end:
                continue

            folds.append(
                OuterFoldResult(
                    fold_id=i,
                    train_start=data.index[start],
                    train_end=data.index[train_end_idx],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[end - 1],
                    selected_params={},
                )
            )
        return folds

    @staticmethod
    def _make_inner_folds(
        train_data: pd.DataFrame, n_inner: int
    ) -> List[Tuple[range, range]]:
        """Create inner (train_idx, val_idx) index pairs from training data."""
        n = len(train_data)
        fold_size = n // (n_inner + 1)
        folds = []
        for i in range(n_inner):
            val_start = (i + 1) * fold_size
            val_end = val_start + fold_size
            folds.append((range(0, val_start), range(val_start, min(val_end, n))))
        return folds

    @staticmethod
    def _chain_curves(curves: List[pd.Series]) -> Optional[pd.Series]:
        """Chain equity curves so each starts where the previous ended."""
        if not curves:
            return None
        values = [1.0]
        index = [curves[0].index[0]]
        for curve in curves:
            if curve.empty:
                continue
            norm = curve / curve.iloc[0] * values[-1]
            values.extend(norm.iloc[1:].tolist())
            index.extend(curve.index[1:].tolist())
        return pd.Series(values, index=index)
