"""
Failure Detector - Identify when and why ML optimization fails.

This module is critical for understanding the LIMITATIONS of ML
parameter optimization. It identifies:
- Failure patterns (when ML hurts performance)
- Root causes of failures
- Warning signs that ML may not help

RESEARCH VALUE: Understanding failures is just as important as
understanding successes. This module helps answer "WHEN does ML fail?"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of ML optimization failures."""
    OVERFITTING = "overfitting"  # Good train, bad test
    REGIME_CHANGE = "regime_change"  # Params optimized for different regime
    NOISE_FITTING = "noise_fitting"  # Fitting to random noise
    BOUNDARY_PARAMS = "boundary_params"  # Params at extreme bounds
    INSTABILITY = "instability"  # Highly variable results
    DEGRADATION = "degradation"  # ML worse than baseline
    NO_IMPROVEMENT = "no_improvement"  # ML same as baseline


@dataclass
class FailurePattern:
    """
    Represents a detected failure pattern.
    
    Contains the type of failure, evidence, and potential remedies.
    """
    failure_type: FailureType
    severity: float  # 0 to 1, higher = more severe
    
    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    strategy_name: str = ""
    market_condition: str = ""
    time_period: str = ""
    
    # Analysis
    root_cause: str = ""
    warning_signs: List[str] = field(default_factory=list)
    remedies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'failure_type': self.failure_type.value,
            'severity': self.severity,
            'evidence': self.evidence,
            'strategy_name': self.strategy_name,
            'market_condition': self.market_condition,
            'root_cause': self.root_cause,
            'warning_signs': self.warning_signs,
            'remedies': self.remedies
        }
    
    def summary(self) -> str:
        return (
            f"FAILURE: {self.failure_type.value} (severity: {self.severity:.2f})\n"
            f"Strategy: {self.strategy_name}\n"
            f"Condition: {self.market_condition}\n"
            f"Root cause: {self.root_cause}\n"
            f"Remedies: {', '.join(self.remedies)}"
        )


class FailureDetector:
    """
    Detects and analyzes ML optimization failures.
    
    This class identifies patterns where ML optimization doesn't help
    or actively hurts performance, providing diagnostic information.
    
    Key failure patterns detected:
    1. Overfitting: Train performance >> Test performance
    2. Regime change: Optimization done in different market regime
    3. Noise fitting: Optimization found noise, not signal
    4. Boundary params: Parameters pushed to extreme bounds
    5. Instability: Results vary wildly across runs
    
    Usage:
        detector = FailureDetector()
        
        failures = detector.detect_failures(
            train_metrics=train_results.metrics,
            test_metrics=test_results.metrics,
            ml_params=optimized_params,
            param_bounds=strategy.get_parameter_bounds()
        )
        
        for failure in failures:
            print(failure.summary())
    """
    
    def __init__(
        self,
        overfitting_threshold: float = 0.3,  # Train-test gap
        degradation_threshold: float = 0.05,  # 5% worse than baseline
        boundary_tolerance: float = 0.05,  # 5% from boundary
        instability_cv_threshold: float = 0.5  # Coefficient of variation
    ):
        """
        Initialize the Failure Detector.
        
        Args:
            overfitting_threshold: Max acceptable train-test performance gap
            degradation_threshold: Min degradation to flag as failure
            boundary_tolerance: Distance from bounds to flag
            instability_cv_threshold: Max CV for stable results
        """
        self.overfitting_threshold = overfitting_threshold
        self.degradation_threshold = degradation_threshold
        self.boundary_tolerance = boundary_tolerance
        self.instability_cv_threshold = instability_cv_threshold
        
        # Store detected failures
        self.failures: List[FailurePattern] = []
    
    def detect_failures(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
        ml_params: Optional[Dict[str, Any]] = None,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        multiple_runs: Optional[List[Dict[str, float]]] = None,
        strategy_name: str = "",
        market_condition: str = ""
    ) -> List[FailurePattern]:
        """
        Detect all failure patterns.
        
        Args:
            train_metrics: In-sample performance metrics
            test_metrics: Out-of-sample performance metrics
            baseline_metrics: Human baseline metrics (optional)
            ml_params: Optimized parameters
            param_bounds: Parameter bounds
            multiple_runs: Results from multiple optimization runs (for stability)
            strategy_name: Name of the strategy
            market_condition: Market condition label
            
        Returns:
            List of detected FailurePattern objects
        """
        failures = []
        
        # 1. Check for overfitting
        overfitting = self._detect_overfitting(
            train_metrics, test_metrics, strategy_name, market_condition
        )
        if overfitting:
            failures.append(overfitting)
        
        # 2. Check for degradation vs baseline
        if baseline_metrics:
            degradation = self._detect_degradation(
                test_metrics, baseline_metrics, strategy_name, market_condition
            )
            if degradation:
                failures.append(degradation)
        
        # 3. Check for boundary parameters
        if ml_params and param_bounds:
            boundary = self._detect_boundary_params(
                ml_params, param_bounds, strategy_name, market_condition
            )
            if boundary:
                failures.append(boundary)
        
        # 4. Check for instability
        if multiple_runs and len(multiple_runs) > 1:
            instability = self._detect_instability(
                multiple_runs, strategy_name, market_condition
            )
            if instability:
                failures.append(instability)
        
        # 5. Check for no improvement
        if baseline_metrics:
            no_improvement = self._detect_no_improvement(
                test_metrics, baseline_metrics, strategy_name, market_condition
            )
            if no_improvement:
                failures.append(no_improvement)
        
        self.failures.extend(failures)
        
        return failures
    
    def _detect_overfitting(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        strategy_name: str,
        market_condition: str
    ) -> Optional[FailurePattern]:
        """Detect overfitting (train >> test)."""
        train_sharpe = train_metrics.get('sharpe_ratio', 0)
        test_sharpe = test_metrics.get('sharpe_ratio', 0)
        
        if train_sharpe <= 0:
            return None
        
        gap = (train_sharpe - test_sharpe) / abs(train_sharpe)
        
        if gap > self.overfitting_threshold:
            # Calculate severity
            severity = min(gap / self.overfitting_threshold, 1.0)
            
            return FailurePattern(
                failure_type=FailureType.OVERFITTING,
                severity=severity,
                evidence={
                    'train_sharpe': train_sharpe,
                    'test_sharpe': test_sharpe,
                    'gap_pct': gap * 100
                },
                strategy_name=strategy_name,
                market_condition=market_condition,
                root_cause=(
                    "Optimization found parameters that work well on training data "
                    "but don't generalize to new data. This typically happens when "
                    "the optimizer exploits noise or idiosyncratic patterns."
                ),
                warning_signs=[
                    "Large gap between train and test performance",
                    "Parameters at extreme values",
                    "Optimization converged very quickly"
                ],
                remedies=[
                    "Increase training data size",
                    "Use cross-validation or walk-forward",
                    "Add regularization/constraints to optimization",
                    "Reduce number of parameters being optimized",
                    "Use simpler optimization (e.g., random search)"
                ]
            )
        
        return None
    
    def _detect_degradation(
        self,
        test_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        strategy_name: str,
        market_condition: str
    ) -> Optional[FailurePattern]:
        """Detect when ML is worse than baseline."""
        test_sharpe = test_metrics.get('sharpe_ratio', 0)
        baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
        
        if baseline_sharpe == 0:
            return None
        
        degradation = (baseline_sharpe - test_sharpe) / abs(baseline_sharpe)
        
        if degradation > self.degradation_threshold:
            severity = min(degradation / 0.5, 1.0)  # 50% degradation = max severity
            
            return FailurePattern(
                failure_type=FailureType.DEGRADATION,
                severity=severity,
                evidence={
                    'baseline_sharpe': baseline_sharpe,
                    'ml_sharpe': test_sharpe,
                    'degradation_pct': degradation * 100
                },
                strategy_name=strategy_name,
                market_condition=market_condition,
                root_cause=(
                    "ML optimization produced parameters that perform worse than "
                    "the human baseline. The optimizer may have found a local optimum "
                    "or the objective function may not align with actual trading goals."
                ),
                warning_signs=[
                    "Optimization objective differs from evaluation metric",
                    "Baseline parameters are already well-tuned",
                    "Market conditions changed between train and test"
                ],
                remedies=[
                    "Use the baseline parameters instead",
                    "Verify objective function matches trading goals",
                    "Check for regime changes in test period",
                    "Consider ensemble of human + ML params"
                ]
            )
        
        return None
    
    def _detect_boundary_params(
        self,
        params: Dict[str, Any],
        bounds: Dict[str, Tuple[float, float]],
        strategy_name: str,
        market_condition: str
    ) -> Optional[FailurePattern]:
        """Detect parameters at boundary values."""
        boundary_params = []
        
        for param_name, value in params.items():
            if param_name not in bounds:
                continue
            
            if not isinstance(value, (int, float)):
                continue
            
            low, high = bounds[param_name]
            range_size = high - low
            
            if range_size <= 0:
                continue
            
            # Check if at boundary
            dist_to_low = (value - low) / range_size
            dist_to_high = (high - value) / range_size
            
            if dist_to_low < self.boundary_tolerance:
                boundary_params.append({
                    'param': param_name,
                    'value': value,
                    'boundary': 'lower',
                    'bound': low
                })
            elif dist_to_high < self.boundary_tolerance:
                boundary_params.append({
                    'param': param_name,
                    'value': value,
                    'boundary': 'upper',
                    'bound': high
                })
        
        if boundary_params:
            severity = min(len(boundary_params) / 3, 1.0)  # 3+ at boundary = max
            
            return FailurePattern(
                failure_type=FailureType.BOUNDARY_PARAMS,
                severity=severity,
                evidence={
                    'boundary_params': boundary_params,
                    'n_at_boundary': len(boundary_params),
                    'total_params': len(params)
                },
                strategy_name=strategy_name,
                market_condition=market_condition,
                root_cause=(
                    "Some optimized parameters are at their boundary values. "
                    "This suggests the search space may be too restrictive, "
                    "or the optimizer found an extreme solution that may not generalize."
                ),
                warning_signs=[
                    "Parameters consistently hitting same boundary",
                    "Unexpected extreme parameter values",
                    "Large changes from baseline parameters"
                ],
                remedies=[
                    "Expand parameter bounds if reasonable",
                    "Add penalty for extreme parameters",
                    "Verify bounds reflect realistic parameter ranges",
                    "Consider if baseline params should be used instead"
                ]
            )
        
        return None
    
    def _detect_instability(
        self,
        multiple_runs: List[Dict[str, float]],
        strategy_name: str,
        market_condition: str
    ) -> Optional[FailurePattern]:
        """Detect instability across multiple runs."""
        sharpes = [run.get('sharpe_ratio', 0) for run in multiple_runs]
        
        if len(sharpes) < 2:
            return None
        
        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        
        if mean_sharpe == 0:
            cv = float('inf') if std_sharpe > 0 else 0
        else:
            cv = std_sharpe / abs(mean_sharpe)
        
        if cv > self.instability_cv_threshold:
            severity = min(cv / (2 * self.instability_cv_threshold), 1.0)
            
            return FailurePattern(
                failure_type=FailureType.INSTABILITY,
                severity=severity,
                evidence={
                    'n_runs': len(sharpes),
                    'mean_sharpe': mean_sharpe,
                    'std_sharpe': std_sharpe,
                    'cv': cv,
                    'min_sharpe': min(sharpes),
                    'max_sharpe': max(sharpes)
                },
                strategy_name=strategy_name,
                market_condition=market_condition,
                root_cause=(
                    "Optimization results vary significantly across runs. "
                    "The optimization landscape may have many local optima, "
                    "or the objective function may be noisy."
                ),
                warning_signs=[
                    "Different random seeds give very different results",
                    "Optimal parameters vary widely",
                    "Convergence to different solutions"
                ],
                remedies=[
                    "Increase optimization iterations",
                    "Use more robust optimizer (e.g., ensemble)",
                    "Run multiple times and take median",
                    "Simplify by optimizing fewer parameters",
                    "Consider if the problem is inherently unstable"
                ]
            )
        
        return None
    
    def _detect_no_improvement(
        self,
        test_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        strategy_name: str,
        market_condition: str
    ) -> Optional[FailurePattern]:
        """Detect when ML provides no meaningful improvement."""
        test_sharpe = test_metrics.get('sharpe_ratio', 0)
        baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
        
        if baseline_sharpe == 0:
            improvement = 0 if test_sharpe == 0 else float('inf')
        else:
            improvement = (test_sharpe - baseline_sharpe) / abs(baseline_sharpe)
        
        # No improvement if within 5% of baseline
        if -0.05 <= improvement <= 0.05:
            return FailurePattern(
                failure_type=FailureType.NO_IMPROVEMENT,
                severity=0.3,  # Low severity - not harmful, just not helpful
                evidence={
                    'baseline_sharpe': baseline_sharpe,
                    'ml_sharpe': test_sharpe,
                    'improvement_pct': improvement * 100
                },
                strategy_name=strategy_name,
                market_condition=market_condition,
                root_cause=(
                    "ML optimization provided no meaningful improvement over baseline. "
                    "The human parameters may already be near-optimal, or the "
                    "optimization problem may be too difficult for the method used."
                ),
                warning_signs=[
                    "Baseline parameters are well-tuned",
                    "Small parameter space",
                    "Low optimization budget"
                ],
                remedies=[
                    "Keep using baseline parameters (simpler is better)",
                    "Try different optimization method",
                    "Increase optimization budget",
                    "Focus optimization effort elsewhere"
                ]
            )
        
        return None
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of all detected failures."""
        if not self.failures:
            return {'n_failures': 0, 'message': 'No failures detected'}
        
        by_type = {}
        for f in self.failures:
            t = f.failure_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(f)
        
        summary = {
            'n_failures': len(self.failures),
            'by_type': {
                t: {
                    'count': len(failures),
                    'avg_severity': np.mean([f.severity for f in failures]),
                    'strategies_affected': list(set(f.strategy_name for f in failures))
                }
                for t, failures in by_type.items()
            },
            'most_severe': max(self.failures, key=lambda f: f.severity).to_dict()
        }
        
        return summary
    
    def get_all_remedies(self) -> List[str]:
        """Get deduplicated list of all remedies across failures."""
        remedies = set()
        for f in self.failures:
            remedies.update(f.remedies)
        return list(remedies)
    
    def should_use_baseline(self) -> Tuple[bool, str]:
        """
        Determine if baseline parameters should be used instead of ML.
        
        Returns:
            Tuple of (should_use_baseline, reason)
        """
        if not self.failures:
            return False, "No failures detected, ML optimization appears successful"
        
        # Check for degradation
        degradation_failures = [
            f for f in self.failures if f.failure_type == FailureType.DEGRADATION
        ]
        if degradation_failures:
            avg_severity = np.mean([f.severity for f in degradation_failures])
            if avg_severity > 0.3:
                return True, (
                    f"ML optimization degrades performance "
                    f"(avg severity: {avg_severity:.2f})"
                )
        
        # Check for severe overfitting
        overfitting_failures = [
            f for f in self.failures if f.failure_type == FailureType.OVERFITTING
        ]
        if overfitting_failures:
            avg_severity = np.mean([f.severity for f in overfitting_failures])
            if avg_severity > 0.5:
                return True, (
                    f"Severe overfitting detected "
                    f"(avg severity: {avg_severity:.2f})"
                )
        
        # Check for instability
        instability_failures = [
            f for f in self.failures if f.failure_type == FailureType.INSTABILITY
        ]
        if instability_failures:
            avg_severity = np.mean([f.severity for f in instability_failures])
            if avg_severity > 0.7:
                return True, (
                    f"Optimization results are unstable "
                    f"(avg severity: {avg_severity:.2f})"
                )
        
        return False, "Failures detected but not severe enough to reject ML optimization"
