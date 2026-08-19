"""
Early Stopping Criterion for iterative optimization algorithms.

Halts an optimizer when the best objective value has not improved by more
than ``min_delta`` over the last ``patience`` consecutive evaluations.

Designed to wrap any ``BaseOptimizer`` subclass that calls this criterion
after each trial.

Usage
-----
>>> from optimization.early_stopping import EarlyStoppingCriterion
>>> criterion = EarlyStoppingCriterion(patience=15, min_delta=1e-4)
>>> for trial_idx in range(n_iterations):
...     params = optimizer.suggest_next()
...     score = objective(params)
...     if criterion.update(score):
...         break  # converged
>>> print(criterion.reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceRecord:
    """Internal record kept per stopping check."""
    trial: int
    best_so_far: float
    delta: float
    stopped: bool


class EarlyStoppingCriterion:
    """Budget-aware early stopping for iterative optimizers.

    Parameters
    ----------
    patience:
        Number of consecutive trials with improvement < ``min_delta``
        before stopping is triggered.
    min_delta:
        Minimum absolute improvement in objective value that counts as
        meaningful progress.
    maximize:
        Set to ``True`` if a higher objective value is better (e.g. Sharpe
        Ratio), ``False`` if lower is better (e.g. loss).
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        maximize: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize

        self._best: Optional[float] = None
        self._no_improve_count: int = 0
        self._trial: int = 0
        self.reason: str = ""
        self.history: List[ConvergenceRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, objective_value: float) -> bool:
        """Register the latest objective value and check for convergence.

        Parameters
        ----------
        objective_value:
            The objective value from the most recent trial.

        Returns
        -------
        bool
            ``True`` if stopping should be triggered, ``False`` otherwise.
        """
        self._trial += 1
        improved = self._check_improvement(objective_value)

        if improved:
            self._best = objective_value
            self._no_improve_count = 0
            delta = abs(objective_value - (self._best or objective_value))
        else:
            self._no_improve_count += 1
            delta = 0.0

        stopped = self._no_improve_count >= self.patience

        record = ConvergenceRecord(
            trial=self._trial,
            best_so_far=self._best if self._best is not None else objective_value,
            delta=delta,
            stopped=stopped,
        )
        self.history.append(record)

        if stopped:
            self.reason = (
                f"No improvement > {self.min_delta} over {self.patience} trials "
                f"(best={self._best:.6f}, trial={self._trial})"
            )
            logger.info("EarlyStopping: %s", self.reason)

        return stopped

    def reset(self) -> None:
        """Reset the criterion state (useful between optimizer restarts)."""
        self._best = None
        self._no_improve_count = 0
        self._trial = 0
        self.reason = ""
        self.history = []

    @property
    def best(self) -> Optional[float]:
        """Best objective value seen so far."""
        return self._best

    @property
    def no_improve_count(self) -> int:
        """Consecutive trials without meaningful improvement."""
        return self._no_improve_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_improvement(self, value: float) -> bool:
        """Return True if ``value`` is a meaningful improvement over ``_best``."""
        if self._best is None:
            # First observation always counts as an improvement
            return True
        if self.maximize:
            return value > self._best + self.min_delta
        return value < self._best - self.min_delta
