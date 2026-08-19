"""
Explainability Engine — human-readable parameter-shift reasoning.

Answers the core research question:

    "When and why did ML optimisation change a parameter, and did it help?"

For every optimisation cycle the engine:
1. Compares the ML-recommended parameters to the human baseline.
2. Generates plain-English explanations for each shift (e.g. "Tightened
   RSI buy threshold from 30 → 22 in a high-volatility regime").
3. Computes a per-parameter improvement attribution based on the
   performance delta between human and ML parameters.
4. Produces a ``"when ML helped vs hurts"`` regime breakdown.

Usage
-----
>>> from analysis.explainability_engine import ExplainabilityEngine
>>> engine = ExplainabilityEngine()
>>> report = engine.explain(
...     human_params={"rsi_buy_threshold": 30, "rsi_lookback": 14},
...     ml_params={"rsi_buy_threshold": 22, "rsi_lookback": 20},
...     human_sharpe=0.8,
...     ml_sharpe=1.4,
...     regime="volatile",
...     feature_context={"rsi_14": 25.0, "atr_14": 3.5},
... )
>>> print(report.narrative)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ParameterShiftExplanation:
    """Explanation for a single parameter change."""
    param_name: str
    human_value: Any
    ml_value: Any
    change_pct: float
    direction: str          # "tightened", "relaxed", "increased", "decreased"
    reasoning: str          # Human-readable sentence
    attributed_impact: float  # Fraction of total performance delta attributable


@dataclass
class ExplainabilityReport:
    """Full explainability output for one optimisation cycle."""
    human_params: Dict[str, Any]
    ml_params: Dict[str, Any]
    human_sharpe: float
    ml_sharpe: float
    regime: str
    sharpe_delta: float
    ml_helped: bool

    # Per-parameter explanations
    parameter_shifts: List[ParameterShiftExplanation] = field(default_factory=list)
    # High-level narrative paragraph
    narrative: str = ""
    # Feature context at the time of optimisation
    feature_context: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime,
            "ml_helped": self.ml_helped,
            "human_sharpe": self.human_sharpe,
            "ml_sharpe": self.ml_sharpe,
            "sharpe_delta": self.sharpe_delta,
            "narrative": self.narrative,
            "parameter_shifts": [
                {
                    "param": s.param_name,
                    "human": s.human_value,
                    "ml": s.ml_value,
                    "change_pct": s.change_pct,
                    "direction": s.direction,
                    "reasoning": s.reasoning,
                }
                for s in self.parameter_shifts
            ],
        }


@dataclass
class RegimeBreakdown:
    """Aggregated 'ML helped vs hurts' breakdown per regime."""
    regime: str
    n_cycles: int
    n_helped: int
    avg_sharpe_delta: float

    @property
    def help_rate(self) -> float:
        return self.n_helped / self.n_cycles if self.n_cycles > 0 else 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ExplainabilityEngine:
    """Generates human-readable explanations for ML parameter changes.

    Parameters
    ----------
    param_descriptions:
        Optional mapping of parameter name → human-readable description
        (e.g. ``{"rsi_buy_threshold": "RSI oversold threshold"}``).
    """

    _DEFAULT_DESCRIPTIONS: Dict[str, str] = {
        "rsi_buy_threshold": "RSI oversold threshold",
        "rsi_sell_threshold": "RSI overbought threshold",
        "rsi_lookback": "RSI lookback period",
        "fast_ema": "Fast EMA period",
        "slow_ema": "Slow EMA period",
        "stop_loss_pct": "Stop-loss percentage",
        "take_profit_pct": "Take-profit percentage",
        "bb_std_multiplier": "Bollinger Band standard-deviation multiplier",
        "bb_lookback": "Bollinger Band lookback period",
        "atr_multiplier": "ATR-based stop multiplier",
        "position_size": "Position size fraction",
    }

    def __init__(
        self,
        param_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        self.param_descriptions = dict(self._DEFAULT_DESCRIPTIONS)
        if param_descriptions:
            self.param_descriptions.update(param_descriptions)

        # History for regime breakdown analysis
        self._history: List[ExplainabilityReport] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        human_params: Dict[str, Any],
        ml_params: Dict[str, Any],
        human_sharpe: float,
        ml_sharpe: float,
        regime: str = "unknown",
        feature_context: Optional[Dict[str, float]] = None,
    ) -> ExplainabilityReport:
        """Generate an explainability report for one optimisation cycle.

        Parameters
        ----------
        human_params:
            Baseline parameter set (human-defined).
        ml_params:
            ML-recommended parameter set.
        human_sharpe:
            Backtest Sharpe Ratio using ``human_params``.
        ml_sharpe:
            Backtest Sharpe Ratio using ``ml_params``.
        regime:
            Current market regime label.
        feature_context:
            Snapshot of key feature values at the time of optimisation.

        Returns
        -------
        ExplainabilityReport
        """
        delta = ml_sharpe - human_sharpe
        helped = delta > 0

        shifts = self._build_parameter_shifts(
            human_params, ml_params, delta, regime, feature_context or {}
        )
        narrative = self._build_narrative(
            regime, human_sharpe, ml_sharpe, delta, shifts, helped
        )

        report = ExplainabilityReport(
            human_params=human_params,
            ml_params=ml_params,
            human_sharpe=human_sharpe,
            ml_sharpe=ml_sharpe,
            regime=regime,
            sharpe_delta=delta,
            ml_helped=helped,
            parameter_shifts=shifts,
            narrative=narrative,
            feature_context=feature_context or {},
        )
        self._history.append(report)
        return report

    def regime_breakdown(self) -> List[RegimeBreakdown]:
        """Aggregate 'ML helped vs hurts' statistics per market regime."""
        from collections import defaultdict

        buckets: Dict[str, List[ExplainabilityReport]] = defaultdict(list)
        for r in self._history:
            buckets[r.regime].append(r)

        result = []
        for regime, reports in sorted(buckets.items()):
            n_helped = sum(1 for r in reports if r.ml_helped)
            avg_delta = float(np.mean([r.sharpe_delta for r in reports]))
            result.append(
                RegimeBreakdown(
                    regime=regime,
                    n_cycles=len(reports),
                    n_helped=n_helped,
                    avg_sharpe_delta=avg_delta,
                )
            )
        return result

    def regime_breakdown_df(self) -> pd.DataFrame:
        """Return regime breakdown as a DataFrame."""
        rows = [
            {
                "regime": rb.regime,
                "n_cycles": rb.n_cycles,
                "n_helped": rb.n_helped,
                "help_rate": rb.help_rate,
                "avg_sharpe_delta": rb.avg_sharpe_delta,
            }
            for rb in self.regime_breakdown()
        ]
        return pd.DataFrame(rows)

    def history_df(self) -> pd.DataFrame:
        """Return all recorded reports as a DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self._history])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_parameter_shifts(
        self,
        human: Dict[str, Any],
        ml: Dict[str, Any],
        delta: float,
        regime: str,
        context: Dict[str, float],
    ) -> List[ParameterShiftExplanation]:
        shifts = []
        all_params = set(human) | set(ml)

        for param in sorted(all_params):
            hval = human.get(param)
            mval = ml.get(param)
            if hval is None or mval is None:
                continue
            if not isinstance(hval, (int, float)) or not isinstance(mval, (int, float)):
                continue
            if abs(float(hval) - float(mval)) < 1e-9:
                continue

            change_pct = ((float(mval) - float(hval)) / abs(float(hval))) * 100 if float(hval) != 0 else 0.0
            direction = self._direction_label(param, float(hval), float(mval))
            reasoning = self._build_param_reasoning(param, hval, mval, direction, regime, context)

            # Simple attribution: uniform split across changed params (improved later)
            n_changed = sum(
                1 for p in all_params
                if human.get(p) is not None and ml.get(p) is not None
                and isinstance(human.get(p), (int, float))
                and abs(float(human.get(p, 0)) - float(ml.get(p, 0))) > 1e-9
            )
            attributed = delta / max(n_changed, 1)

            shifts.append(ParameterShiftExplanation(
                param_name=param,
                human_value=hval,
                ml_value=mval,
                change_pct=change_pct,
                direction=direction,
                reasoning=reasoning,
                attributed_impact=attributed,
            ))
        return shifts

    def _build_param_reasoning(
        self,
        param: str,
        human_val: float,
        ml_val: float,
        direction: str,
        regime: str,
        context: Dict[str, float],
    ) -> str:
        label = self.param_descriptions.get(param, param.replace("_", " "))
        regime_phrase = {
            "trending_bullish": "in a bullish trending regime",
            "trending_bearish": "in a bearish trending regime",
            "ranging": "in a ranging low-volatility regime",
            "volatile": "amid elevated market volatility",
        }.get(regime, f"in the '{regime}' regime")

        context_hints = []
        if "rsi" in param and "rsi_14" in context:
            context_hints.append(f"RSI was at {context['rsi_14']:.1f}")
        if "atr" in param and "atr_14" in context:
            context_hints.append(f"ATR at {context['atr_14']:.4f}")
        if "bb" in param and "bb_width" in context:
            context_hints.append(f"BB width at {context['bb_width']:.4f}")

        hint_str = (" (" + ", ".join(context_hints) + ")") if context_hints else ""

        return (
            f"ML {direction} {label} from {human_val:.4g} to {ml_val:.4g} "
            f"{regime_phrase}{hint_str}."
        )

    @staticmethod
    def _direction_label(param: str, old: float, new: float) -> str:
        """Determine a human-readable direction word."""
        diff = new - old
        # Threshold-style params: lower threshold = "tightened" (stricter entry)
        if "threshold" in param and "buy" in param:
            return "tightened" if diff < 0 else "relaxed"
        if "threshold" in param and "sell" in param:
            return "tightened" if diff > 0 else "relaxed"
        return "increased" if diff > 0 else "decreased"

    @staticmethod
    def _build_narrative(
        regime: str,
        human_sharpe: float,
        ml_sharpe: float,
        delta: float,
        shifts: List[ParameterShiftExplanation],
        helped: bool,
    ) -> str:
        verdict = "improved" if helped else "degraded"
        n_shifts = len(shifts)
        top_shift = (
            f"The largest change was to «{shifts[0].param_name}» "
            f"({shifts[0].direction} from {shifts[0].human_value:.4g} "
            f"→ {shifts[0].ml_value:.4g})."
            if shifts else ""
        )
        return (
            f"In the '{regime}' regime, ML optimisation {verdict} the Sharpe Ratio "
            f"by {abs(delta):.3f} ({human_sharpe:.3f} → {ml_sharpe:.3f}). "
            f"{n_shifts} parameter{'s were' if n_shifts != 1 else ' was'} changed. "
            f"{top_shift}"
        )
