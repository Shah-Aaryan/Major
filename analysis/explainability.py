"""
Explainability Module (A3 - P1)

Explains WHY ML changed parameters, focusing on parameter-level explainability.
NOT using SHAP/LIME on prices - instead tracks parameter sensitivity and evolution.

Key Metrics:
- Parameter evolution over time
- Performance delta after each update
- Parameter sensitivity (ΔSharpe / Δparameter)
- Stability score (penalize oscillating parameters)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParameterUpdate:
    """Record of a single parameter update."""
    timestamp: datetime
    parameter_name: str
    old_value: Any
    new_value: Any
    change_pct: float
    objective_before: float
    objective_after: float
    objective_delta: float
    optimization_method: str = ""
    market_condition: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "parameter": self.parameter_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change_pct": self.change_pct,
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "objective_delta": self.objective_delta,
            "optimization_method": self.optimization_method,
            "market_condition": self.market_condition
        }


@dataclass
class ParameterSensitivity:
    """Sensitivity analysis for a single parameter."""
    parameter_name: str
    sensitivity: float  # ΔSharpe / Δparameter (normalized)
    directional_impact: str  # "positive", "negative", "neutral"
    stability: float  # 0-1, higher = more stable
    importance_rank: int = 0
    
    # Statistical measures
    mean_value: float = 0.0
    std_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    
    # Oscillation analysis
    direction_changes: int = 0
    oscillation_score: float = 0.0  # 0 = stable, 1 = highly oscillating
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter_name,
            "sensitivity": self.sensitivity,
            "directional_impact": self.directional_impact,
            "stability": self.stability,
            "importance_rank": self.importance_rank,
            "statistics": {
                "mean": self.mean_value,
                "std": self.std_value,
                "min": self.min_value,
                "max": self.max_value
            },
            "oscillation": {
                "direction_changes": self.direction_changes,
                "oscillation_score": self.oscillation_score
            }
        }


@dataclass
class ExplainabilityReport:
    """Complete explainability report for an optimization run."""
    # Metadata
    strategy_name: str
    report_timestamp: datetime
    optimization_count: int
    
    # Parameter summaries
    parameter_sensitivities: Dict[str, ParameterSensitivity]
    parameter_updates: List[ParameterUpdate]
    
    # Overall insights
    most_impactful_parameter: str
    most_stable_parameter: str
    most_volatile_parameter: str
    
    # Performance attribution
    total_improvement: float
    improvement_by_parameter: Dict[str, float]
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "strategy": self.strategy_name,
                "timestamp": self.report_timestamp.isoformat(),
                "optimization_count": self.optimization_count
            },
            "sensitivities": {
                name: sens.to_dict() 
                for name, sens in self.parameter_sensitivities.items()
            },
            "updates": [u.to_dict() for u in self.parameter_updates],
            "insights": {
                "most_impactful": self.most_impactful_parameter,
                "most_stable": self.most_stable_parameter,
                "most_volatile": self.most_volatile_parameter
            },
            "attribution": {
                "total_improvement": self.total_improvement,
                "by_parameter": self.improvement_by_parameter
            },
            "warnings": self.warnings
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"EXPLAINABILITY REPORT: {self.strategy_name}",
            "=" * 60,
            f"Generated: {self.report_timestamp}",
            f"Optimizations analyzed: {self.optimization_count}",
            "",
            "KEY INSIGHTS:",
            f"  Most impactful parameter: {self.most_impactful_parameter}",
            f"  Most stable parameter: {self.most_stable_parameter}",
            f"  Most volatile parameter: {self.most_volatile_parameter}",
            "",
            "PARAMETER SENSITIVITIES:",
        ]
        
        for name, sens in sorted(
            self.parameter_sensitivities.items(),
            key=lambda x: abs(x[1].sensitivity),
            reverse=True
        ):
            lines.append(
                f"  {name}: sensitivity={sens.sensitivity:.4f}, "
                f"stability={sens.stability:.2f}, impact={sens.directional_impact}"
            )
        
        lines.append("")
        lines.append("PERFORMANCE ATTRIBUTION:")
        lines.append(f"  Total improvement: {self.total_improvement:.2%}")
        
        for param, contrib in sorted(
            self.improvement_by_parameter.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            lines.append(f"  {param}: {contrib:.2%}")
        
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ParameterExplainer:
    """
    Tracks and explains ML parameter changes.
    
    This class provides parameter-level explainability:
    - Tracks parameter evolution over time
    - Computes sensitivity (ΔObjective/ΔParameter)
    - Identifies stable vs volatile parameters
    - Attributes performance to parameter changes
    
    Usage:
        explainer = ParameterExplainer()
        
        # Log updates during optimization
        explainer.log_update(
            parameter="rsi_threshold",
            old_value=30, new_value=25,
            objective_before=0.5, objective_after=0.6
        )
        
        # Generate report
        report = explainer.generate_report("my_strategy")
    """
    
    def __init__(self, stability_window: int = 10):
        """
        Initialize the explainer.
        
        Args:
            stability_window: Number of updates to consider for stability
        """
        self.stability_window = stability_window
        self.updates: List[ParameterUpdate] = []
        self.parameter_history: Dict[str, List[Tuple[datetime, Any]]] = {}
        self.objective_history: List[Tuple[datetime, float]] = []
    
    def log_update(
        self,
        parameter: str,
        old_value: Any,
        new_value: Any,
        objective_before: float,
        objective_after: float,
        optimization_method: str = "",
        market_condition: str = "",
        timestamp: Optional[datetime] = None
    ) -> ParameterUpdate:
        """
        Log a parameter update.
        
        Args:
            parameter: Parameter name
            old_value: Value before update
            new_value: Value after update
            objective_before: Objective (e.g., Sharpe) before
            objective_after: Objective after
            optimization_method: Method used (bayesian, evolutionary, etc.)
            market_condition: Market regime at time of update
            timestamp: Update timestamp (defaults to now)
            
        Returns:
            Created ParameterUpdate record
        """
        ts = timestamp or datetime.now()
        
        # Calculate change percentage
        if isinstance(old_value, (int, float)) and old_value != 0:
            change_pct = (new_value - old_value) / abs(old_value) * 100
        else:
            change_pct = 100 if old_value != new_value else 0
        
        update = ParameterUpdate(
            timestamp=ts,
            parameter_name=parameter,
            old_value=old_value,
            new_value=new_value,
            change_pct=change_pct,
            objective_before=objective_before,
            objective_after=objective_after,
            objective_delta=objective_after - objective_before,
            optimization_method=optimization_method,
            market_condition=market_condition
        )
        
        self.updates.append(update)
        
        # Track in history
        if parameter not in self.parameter_history:
            self.parameter_history[parameter] = []
        self.parameter_history[parameter].append((ts, new_value))
        
        self.objective_history.append((ts, objective_after))
        
        logger.debug(f"Logged update: {parameter} {old_value} → {new_value}, Δobjective={update.objective_delta:.4f}")
        
        return update
    
    def log_optimization_result(
        self,
        human_params: Dict[str, Any],
        ml_params: Dict[str, Any],
        human_objective: float,
        ml_objective: float,
        optimization_method: str = "",
        market_condition: str = ""
    ) -> List[ParameterUpdate]:
        """
        Log updates from a full optimization comparison.
        
        Args:
            human_params: Human baseline parameters
            ml_params: ML-optimized parameters
            human_objective: Objective with human params
            ml_objective: Objective with ML params
            optimization_method: Method used
            market_condition: Market regime
            
        Returns:
            List of ParameterUpdate records
        """
        updates = []
        
        for param_name in set(human_params.keys()) & set(ml_params.keys()):
            old_val = human_params[param_name]
            new_val = ml_params[param_name]
            
            if old_val != new_val:
                update = self.log_update(
                    parameter=param_name,
                    old_value=old_val,
                    new_value=new_val,
                    objective_before=human_objective,
                    objective_after=ml_objective,
                    optimization_method=optimization_method,
                    market_condition=market_condition
                )
                updates.append(update)
        
        return updates
    
    def compute_sensitivity(self, parameter: str) -> float:
        """
        Compute parameter sensitivity.
        
        Sensitivity = mean(ΔObjective / ΔParameter) for all updates
        where ΔParameter is normalized to [0, 1] range.
        
        Args:
            parameter: Parameter name
            
        Returns:
            Sensitivity score (higher = more impactful)
        """
        param_updates = [u for u in self.updates if u.parameter_name == parameter]
        
        if not param_updates:
            return 0.0
        
        sensitivities = []
        for update in param_updates:
            if update.change_pct != 0:
                # Normalize change to percentage scale
                normalized_change = abs(update.change_pct) / 100
                sensitivity = update.objective_delta / max(normalized_change, 0.01)
                sensitivities.append(sensitivity)
        
        return np.mean(sensitivities) if sensitivities else 0.0
    
    def compute_stability(self, parameter: str) -> float:
        """
        Compute parameter stability score.
        
        Stability = 1 - (direction_changes / total_updates)
        Penalizes parameters that oscillate frequently.
        
        Args:
            parameter: Parameter name
            
        Returns:
            Stability score (0-1, higher = more stable)
        """
        history = self.parameter_history.get(parameter, [])
        
        if len(history) < 2:
            return 1.0
        
        # Count direction changes
        values = [v for _, v in history]
        if not all(isinstance(v, (int, float)) for v in values):
            return 1.0
        
        direction_changes = 0
        for i in range(2, len(values)):
            prev_delta = values[i-1] - values[i-2]
            curr_delta = values[i] - values[i-1]
            
            # Direction change if sign flips
            if (prev_delta > 0 and curr_delta < 0) or (prev_delta < 0 and curr_delta > 0):
                direction_changes += 1
        
        # More changes = less stable
        max_changes = len(values) - 2
        if max_changes > 0:
            stability = 1 - (direction_changes / max_changes)
        else:
            stability = 1.0
        
        return stability
    
    def compute_oscillation_score(self, parameter: str) -> Tuple[int, float]:
        """
        Compute oscillation metrics.
        
        Returns:
            Tuple of (direction_changes, oscillation_score)
        """
        history = self.parameter_history.get(parameter, [])
        
        if len(history) < 3:
            return 0, 0.0
        
        values = [v for _, v in history]
        if not all(isinstance(v, (int, float)) for v in values):
            return 0, 0.0
        
        # Count direction changes
        direction_changes = 0
        for i in range(2, len(values)):
            prev_delta = values[i-1] - values[i-2]
            curr_delta = values[i] - values[i-1]
            
            if (prev_delta > 0 and curr_delta < 0) or (prev_delta < 0 and curr_delta > 0):
                direction_changes += 1
        
        # Oscillation score
        max_changes = len(values) - 2
        oscillation_score = direction_changes / max_changes if max_changes > 0 else 0.0
        
        return direction_changes, oscillation_score
    
    def compute_parameter_sensitivity(self, parameter: str) -> ParameterSensitivity:
        """
        Compute full sensitivity analysis for a parameter.
        
        Args:
            parameter: Parameter name
            
        Returns:
            ParameterSensitivity object
        """
        sensitivity = self.compute_sensitivity(parameter)
        stability = self.compute_stability(parameter)
        direction_changes, oscillation_score = self.compute_oscillation_score(parameter)
        
        # Determine directional impact
        param_updates = [u for u in self.updates if u.parameter_name == parameter]
        positive_impacts = sum(1 for u in param_updates if u.objective_delta > 0)
        negative_impacts = sum(1 for u in param_updates if u.objective_delta < 0)
        
        if positive_impacts > negative_impacts:
            directional_impact = "positive"
        elif negative_impacts > positive_impacts:
            directional_impact = "negative"
        else:
            directional_impact = "neutral"
        
        # Statistics
        history = self.parameter_history.get(parameter, [])
        values = [v for _, v in history if isinstance(v, (int, float))]
        
        return ParameterSensitivity(
            parameter_name=parameter,
            sensitivity=sensitivity,
            directional_impact=directional_impact,
            stability=stability,
            mean_value=np.mean(values) if values else 0.0,
            std_value=np.std(values) if values else 0.0,
            min_value=min(values) if values else 0.0,
            max_value=max(values) if values else 0.0,
            direction_changes=direction_changes,
            oscillation_score=oscillation_score
        )
    
    def attribute_performance(self) -> Dict[str, float]:
        """
        Attribute performance improvement to each parameter.
        
        Uses a simple attribution model based on update contributions.
        
        Returns:
            Dict mapping parameter name to contribution percentage
        """
        if not self.updates:
            return {}
        
        # Sum objective deltas per parameter
        param_deltas: Dict[str, float] = {}
        for update in self.updates:
            param = update.parameter_name
            if param not in param_deltas:
                param_deltas[param] = 0.0
            param_deltas[param] += update.objective_delta
        
        # Total improvement
        total = sum(abs(d) for d in param_deltas.values())
        
        if total == 0:
            return {p: 0.0 for p in param_deltas}
        
        # Attribution as percentage of total
        attribution = {}
        for param, delta in param_deltas.items():
            attribution[param] = delta / total
        
        return attribution
    
    def generate_report(self, strategy_name: str) -> ExplainabilityReport:
        """
        Generate a complete explainability report.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            ExplainabilityReport with full analysis
        """
        # Compute sensitivities for all parameters
        sensitivities = {}
        for param in self.parameter_history.keys():
            sensitivities[param] = self.compute_parameter_sensitivity(param)
        
        # Rank by importance
        ranked = sorted(
            sensitivities.items(),
            key=lambda x: abs(x[1].sensitivity),
            reverse=True
        )
        for rank, (name, sens) in enumerate(ranked, 1):
            sens.importance_rank = rank
        
        # Find key parameters
        most_impactful = ranked[0][0] if ranked else "none"
        most_stable = max(
            sensitivities.items(),
            key=lambda x: x[1].stability,
            default=(("none", None))
        )[0]
        most_volatile = min(
            sensitivities.items(),
            key=lambda x: x[1].stability,
            default=(("none", None))
        )[0]
        
        # Attribution
        attribution = self.attribute_performance()
        total_improvement = sum(u.objective_delta for u in self.updates)
        
        # Warnings
        warnings = []
        for name, sens in sensitivities.items():
            if sens.oscillation_score > 0.5:
                warnings.append(f"Parameter '{name}' is highly oscillating (score: {sens.oscillation_score:.2f})")
            if sens.stability < 0.3:
                warnings.append(f"Parameter '{name}' has low stability (score: {sens.stability:.2f})")
        
        return ExplainabilityReport(
            strategy_name=strategy_name,
            report_timestamp=datetime.now(),
            optimization_count=len(self.updates),
            parameter_sensitivities=sensitivities,
            parameter_updates=self.updates,
            most_impactful_parameter=most_impactful,
            most_stable_parameter=most_stable,
            most_volatile_parameter=most_volatile,
            total_improvement=total_improvement,
            improvement_by_parameter=attribution,
            warnings=warnings
        )
    
    def get_parameter_evolution(self, parameter: str) -> pd.DataFrame:
        """
        Get parameter evolution as a DataFrame.
        
        Args:
            parameter: Parameter name
            
        Returns:
            DataFrame with timestamp and value columns
        """
        history = self.parameter_history.get(parameter, [])
        
        return pd.DataFrame([
            {"timestamp": ts, "value": val}
            for ts, val in history
        ])
    
    def clear(self):
        """Clear all tracked data."""
        self.updates = []
        self.parameter_history = {}
        self.objective_history = []


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_report_as_json(report: ExplainabilityReport) -> str:
    """Format report as JSON string."""
    return json.dumps(report.to_dict(), indent=2, default=str)


def format_sensitivity_table(
    sensitivities: Dict[str, ParameterSensitivity]
) -> pd.DataFrame:
    """Format sensitivities as a DataFrame table."""
    rows = []
    for name, sens in sensitivities.items():
        rows.append({
            "Parameter": name,
            "Sensitivity": sens.sensitivity,
            "Stability": sens.stability,
            "Impact": sens.directional_impact,
            "Mean": sens.mean_value,
            "Std": sens.std_value,
            "Oscillation": sens.oscillation_score,
            "Rank": sens.importance_rank
        })
    
    return pd.DataFrame(rows).sort_values("Rank")


if __name__ == "__main__":
    # Example usage
    explainer = ParameterExplainer()
    
    # Simulate some updates
    explainer.log_update("rsi_threshold", 30, 25, 0.5, 0.6, "bayesian", "ranging")
    explainer.log_update("rsi_threshold", 25, 28, 0.6, 0.65, "bayesian", "ranging")
    explainer.log_update("rsi_period", 14, 12, 0.5, 0.55, "bayesian", "ranging")
    explainer.log_update("stop_loss_pct", 2.0, 1.5, 0.5, 0.52, "bayesian", "trending")
    
    # Generate report
    report = explainer.generate_report("RSI_Mean_Reversion")
    print(report.summary())
    print()
    print("JSON Output:")
    print(format_report_as_json(report))
