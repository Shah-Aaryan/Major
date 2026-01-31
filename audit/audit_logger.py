"""
Audit Logger.

Comprehensive logging system for research transparency.

Logs all decisions, parameter changes, and explanations
to enable:
1. Full reproducibility of research
2. Understanding why ML made certain adjustments
3. Tracking performance attribution
4. Debugging and analysis
"""

import logging
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
import hashlib

import pandas as pd

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # Parameter events
    PARAMETER_CHANGE = "parameter_change"
    PARAMETER_OPTIMIZATION_START = "optimization_start"
    PARAMETER_OPTIMIZATION_END = "optimization_end"
    PARAMETER_RESET = "parameter_reset"
    
    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_EXECUTED = "signal_executed"
    SIGNAL_REJECTED = "signal_rejected"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"
    
    # Analysis events
    CONDITION_DETECTED = "condition_detected"
    FAILURE_DETECTED = "failure_detected"
    REGIME_CHANGE = "regime_change"
    
    # System events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class AuditEvent:
    """
    Base audit event.
    
    Every logged event contains:
    - Timestamp
    - Event type
    - Relevant data
    - Optional explanation
    """
    timestamp: datetime
    event_type: AuditEventType
    data: Dict[str, Any]
    explanation: str = ""
    session_id: str = ""
    strategy_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'data': self.data,
            'explanation': self.explanation,
            'session_id': self.session_id,
            'strategy_name': self.strategy_name
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SignalAudit:
    """
    Detailed audit for trading signals.
    
    Captures why a signal was generated and
    all the factors that contributed to it.
    """
    timestamp: datetime
    strategy_name: str
    signal_type: str  # buy, sell, hold
    confidence: float
    price: float
    
    # Explanation
    primary_reason: str
    secondary_reasons: List[str] = field(default_factory=list)
    
    # Feature values that triggered signal
    triggering_features: Dict[str, float] = field(default_factory=dict)
    
    # Thresholds and comparisons
    thresholds: Dict[str, Any] = field(default_factory=dict)
    
    # Market context
    market_condition: str = ""
    trend_direction: str = ""
    volatility_level: str = ""
    
    # Parameters used
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_source: str = ""  # "human" or "ml"
    
    def generate_explanation(self) -> str:
        """Generate human-readable explanation."""
        explanation = [
            f"Signal: {self.signal_type.upper()} at {self.price:.2f}",
            f"Confidence: {self.confidence:.2%}",
            f"Primary Reason: {self.primary_reason}"
        ]
        
        if self.secondary_reasons:
            explanation.append("Supporting Factors:")
            for reason in self.secondary_reasons[:3]:
                explanation.append(f"  - {reason}")
        
        if self.market_condition:
            explanation.append(f"Market Condition: {self.market_condition}")
        
        if self.parameter_source:
            explanation.append(f"Parameters: {self.parameter_source}")
        
        return "\n".join(explanation)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'price': self.price,
            'primary_reason': self.primary_reason,
            'secondary_reasons': self.secondary_reasons,
            'triggering_features': self.triggering_features,
            'thresholds': self.thresholds,
            'market_condition': self.market_condition,
            'trend_direction': self.trend_direction,
            'volatility_level': self.volatility_level,
            'parameters': self.parameters,
            'parameter_source': self.parameter_source,
            'explanation': self.generate_explanation()
        }


@dataclass
class OptimizationAudit:
    """
    Audit for optimization events.
    
    Tracks what the optimizer tried and why
    certain parameters were selected.
    """
    timestamp: datetime
    strategy_name: str
    optimizer_type: str  # bayesian, random, evolutionary
    
    # Optimization details
    n_trials: int
    best_objective: float
    elapsed_time: float
    
    # Parameter changes
    human_params: Dict[str, Any]
    ml_params: Dict[str, Any]
    parameter_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance comparison
    human_performance: Dict[str, float] = field(default_factory=dict)
    ml_performance: Dict[str, float] = field(default_factory=dict)
    improvement: float = 0.0
    
    # Trials history (sample)
    sample_trials: List[Dict[str, Any]] = field(default_factory=list)
    
    # Convergence
    converged: bool = False
    convergence_reason: str = ""
    
    def generate_explanation(self) -> str:
        """Generate human-readable explanation."""
        explanation = [
            f"Optimization Summary ({self.optimizer_type})",
            f"Trials: {self.n_trials} | Time: {self.elapsed_time:.1f}s",
            f"Best Objective: {self.best_objective:.4f}",
            ""
        ]
        
        # Parameter changes
        explanation.append("Parameter Changes:")
        for param, change in self.parameter_changes.items():
            old = change.get('human', 'N/A')
            new = change.get('ml', 'N/A')
            explanation.append(f"  {param}: {old} → {new}")
        
        explanation.append("")
        
        # Performance
        if self.improvement != 0:
            direction = "improved" if self.improvement > 0 else "degraded"
            explanation.append(f"Performance {direction} by {abs(self.improvement):.2%}")
        
        if self.converged:
            explanation.append(f"Converged: {self.convergence_reason}")
        
        return "\n".join(explanation)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'optimizer_type': self.optimizer_type,
            'n_trials': self.n_trials,
            'best_objective': self.best_objective,
            'elapsed_time': self.elapsed_time,
            'human_params': self.human_params,
            'ml_params': self.ml_params,
            'parameter_changes': self.parameter_changes,
            'human_performance': self.human_performance,
            'ml_performance': self.ml_performance,
            'improvement': self.improvement,
            'sample_trials': self.sample_trials,
            'converged': self.converged,
            'convergence_reason': self.convergence_reason,
            'explanation': self.generate_explanation()
        }


@dataclass
class ParameterChangeAudit:
    """
    Audit for individual parameter changes.
    
    Tracks each parameter adjustment with
    full context and justification.
    """
    timestamp: datetime
    strategy_name: str
    parameter_name: str
    
    old_value: Any
    new_value: Any
    change_pct: float
    
    source: str  # "human", "ml_bayesian", "ml_random", etc.
    reason: str
    
    # Impact analysis
    expected_impact: Dict[str, float] = field(default_factory=dict)
    
    # Context
    market_condition: str = ""
    optimization_trial: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'parameter_name': self.parameter_name,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'change_pct': self.change_pct,
            'source': self.source,
            'reason': self.reason,
            'expected_impact': self.expected_impact,
            'market_condition': self.market_condition,
            'optimization_trial': self.optimization_trial
        }


class AuditLogger:
    """
    Central audit logging system.
    
    Features:
    - Logs all research events with full context
    - Provides query and filtering capabilities
    - Exports to various formats (JSON, CSV, Markdown)
    - Thread-safe for real-time logging
    """
    
    def __init__(
        self,
        output_dir: str = "./audit_logs",
        session_id: Optional[str] = None,
        log_to_file: bool = True,
        log_to_console: bool = False,
        max_events_memory: int = 10000
    ):
        """
        Initialize audit logger.
        
        Args:
            output_dir: Directory for log files
            session_id: Unique session identifier
            log_to_file: Write logs to file
            log_to_console: Print logs to console
            max_events_memory: Max events to keep in memory
        """
        self.output_dir = Path(output_dir)
        self.session_id = session_id or self._generate_session_id()
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.max_events_memory = max_events_memory
        
        # Event storage
        self._events: List[AuditEvent] = []
        self._signal_audits: List[SignalAudit] = []
        self._optimization_audits: List[OptimizationAudit] = []
        self._parameter_audits: List[ParameterChangeAudit] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Create output directory
        if self.log_to_file:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log session start
        self.log_event(
            AuditEventType.SESSION_START,
            {'session_id': self.session_id},
            explanation="Audit logging session started"
        )
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"session_{timestamp}_{hash_suffix}"
    
    def log_event(
        self,
        event_type: AuditEventType,
        data: Dict[str, Any],
        explanation: str = "",
        strategy_name: str = ""
    ) -> AuditEvent:
        """
        Log a generic audit event.
        
        Args:
            event_type: Type of event
            data: Event data
            explanation: Human-readable explanation
            strategy_name: Associated strategy
            
        Returns:
            Created AuditEvent
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            data=data,
            explanation=explanation,
            session_id=self.session_id,
            strategy_name=strategy_name
        )
        
        with self._lock:
            self._events.append(event)
            
            # Trim if too many events
            if len(self._events) > self.max_events_memory:
                self._events = self._events[-self.max_events_memory:]
        
        if self.log_to_console:
            self._print_event(event)
        
        if self.log_to_file:
            self._write_event_to_file(event)
        
        return event
    
    def log_signal(self, signal_audit: SignalAudit) -> None:
        """Log a trading signal with full audit trail."""
        with self._lock:
            self._signal_audits.append(signal_audit)
        
        # Also log as generic event
        self.log_event(
            AuditEventType.SIGNAL_GENERATED,
            signal_audit.to_dict(),
            explanation=signal_audit.generate_explanation(),
            strategy_name=signal_audit.strategy_name
        )
    
    def log_optimization(self, optimization_audit: OptimizationAudit) -> None:
        """Log an optimization result with full audit trail."""
        with self._lock:
            self._optimization_audits.append(optimization_audit)
        
        # Also log as generic event
        self.log_event(
            AuditEventType.PARAMETER_OPTIMIZATION_END,
            optimization_audit.to_dict(),
            explanation=optimization_audit.generate_explanation(),
            strategy_name=optimization_audit.strategy_name
        )
    
    def log_parameter_change(self, param_audit: ParameterChangeAudit) -> None:
        """Log a parameter change with full audit trail."""
        with self._lock:
            self._parameter_audits.append(param_audit)
        
        # Also log as generic event
        self.log_event(
            AuditEventType.PARAMETER_CHANGE,
            param_audit.to_dict(),
            explanation=(
                f"Parameter '{param_audit.parameter_name}' changed "
                f"from {param_audit.old_value} to {param_audit.new_value} "
                f"({param_audit.source})"
            ),
            strategy_name=param_audit.strategy_name
        )
    
    def log_failure(
        self,
        failure_type: str,
        severity: float,
        evidence: Dict[str, Any],
        recommendation: str,
        strategy_name: str = ""
    ) -> None:
        """Log a detected failure."""
        self.log_event(
            AuditEventType.FAILURE_DETECTED,
            {
                'failure_type': failure_type,
                'severity': severity,
                'evidence': evidence,
                'recommendation': recommendation
            },
            explanation=f"Failure detected: {failure_type} (severity: {severity:.2f})",
            strategy_name=strategy_name
        )
    
    def log_condition(
        self,
        condition_type: str,
        condition_value: str,
        metrics: Dict[str, float],
        strategy_name: str = ""
    ) -> None:
        """Log a detected market condition."""
        self.log_event(
            AuditEventType.CONDITION_DETECTED,
            {
                'condition_type': condition_type,
                'condition_value': condition_value,
                'metrics': metrics
            },
            explanation=f"Market condition: {condition_type} = {condition_value}",
            strategy_name=strategy_name
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log an error."""
        self.log_event(
            AuditEventType.ERROR,
            {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            },
            explanation=f"Error: {type(error).__name__}: {str(error)}"
        )
    
    def _print_event(self, event: AuditEvent) -> None:
        """Print event to console."""
        timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[AUDIT] {timestamp} | {event.event_type.value} | {event.explanation[:100]}")
    
    def _write_event_to_file(self, event: AuditEvent) -> None:
        """Write event to log file."""
        filepath = self.output_dir / f"{self.session_id}_events.jsonl"
        
        with open(filepath, 'a') as f:
            f.write(event.to_json() + '\n')
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        strategy_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """
        Query events with filtering.
        
        Args:
            event_type: Filter by event type
            strategy_name: Filter by strategy
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            Filtered list of events
        """
        events = self._events.copy()
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if strategy_name:
            events = [e for e in events if e.strategy_name == strategy_name]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events
    
    def get_signal_audits(self, strategy_name: Optional[str] = None) -> List[SignalAudit]:
        """Get signal audits, optionally filtered by strategy."""
        if strategy_name:
            return [s for s in self._signal_audits if s.strategy_name == strategy_name]
        return self._signal_audits.copy()
    
    def get_optimization_audits(self, strategy_name: Optional[str] = None) -> List[OptimizationAudit]:
        """Get optimization audits, optionally filtered by strategy."""
        if strategy_name:
            return [o for o in self._optimization_audits if o.strategy_name == strategy_name]
        return self._optimization_audits.copy()
    
    def get_parameter_audits(
        self,
        strategy_name: Optional[str] = None,
        parameter_name: Optional[str] = None
    ) -> List[ParameterChangeAudit]:
        """Get parameter change audits with optional filtering."""
        audits = self._parameter_audits.copy()
        
        if strategy_name:
            audits = [p for p in audits if p.strategy_name == strategy_name]
        
        if parameter_name:
            audits = [p for p in audits if p.parameter_name == parameter_name]
        
        return audits
    
    def export_to_csv(self, filepath: str = None) -> pd.DataFrame:
        """Export events to CSV."""
        filepath = filepath or str(self.output_dir / f"{self.session_id}_events.csv")
        
        df = pd.DataFrame([e.to_dict() for e in self._events])
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(df)} events to {filepath}")
        return df
    
    def export_signals_to_csv(self, filepath: str = None) -> pd.DataFrame:
        """Export signal audits to CSV."""
        filepath = filepath or str(self.output_dir / f"{self.session_id}_signals.csv")
        
        df = pd.DataFrame([s.to_dict() for s in self._signal_audits])
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(df)} signals to {filepath}")
        return df
    
    def export_to_markdown(self, filepath: str = None) -> str:
        """Export audit summary to Markdown."""
        filepath = filepath or str(self.output_dir / f"{self.session_id}_audit.md")
        
        md = []
        md.append(f"# Audit Log: {self.session_id}")
        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("")
        
        # Summary
        md.append("## Summary")
        md.append(f"- Total Events: {len(self._events)}")
        md.append(f"- Signals: {len(self._signal_audits)}")
        md.append(f"- Optimizations: {len(self._optimization_audits)}")
        md.append(f"- Parameter Changes: {len(self._parameter_audits)}")
        md.append("")
        
        # Event breakdown
        md.append("## Event Breakdown")
        event_counts = {}
        for event in self._events:
            event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
        
        for event_type, count in sorted(event_counts.items()):
            md.append(f"- {event_type}: {count}")
        md.append("")
        
        # Recent optimizations
        if self._optimization_audits:
            md.append("## Recent Optimizations")
            for opt in self._optimization_audits[-5:]:
                md.append(f"### {opt.strategy_name} ({opt.optimizer_type})")
                md.append(f"- Trials: {opt.n_trials}")
                md.append(f"- Improvement: {opt.improvement:.2%}")
                md.append(f"- Time: {opt.elapsed_time:.1f}s")
                md.append("")
        
        # Parameter changes
        if self._parameter_audits:
            md.append("## Parameter Changes")
            md.append("| Time | Strategy | Parameter | Old | New | Source |")
            md.append("|------|----------|-----------|-----|-----|--------|")
            
            for param in self._parameter_audits[-20:]:
                time = param.timestamp.strftime("%H:%M:%S")
                md.append(
                    f"| {time} | {param.strategy_name} | {param.parameter_name} | "
                    f"{param.old_value} | {param.new_value} | {param.source} |"
                )
            md.append("")
        
        # Write to file
        content = "\n".join(md)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Exported audit summary to {filepath}")
        return content
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        return {
            'session_id': self.session_id,
            'total_events': len(self._events),
            'total_signals': len(self._signal_audits),
            'total_optimizations': len(self._optimization_audits),
            'total_parameter_changes': len(self._parameter_audits),
            'event_types': {
                etype.value: sum(1 for e in self._events if e.event_type == etype)
                for etype in AuditEventType
            },
            'strategies': list(set(e.strategy_name for e in self._events if e.strategy_name))
        }
    
    def close(self) -> None:
        """Close the audit logger and finalize logs."""
        self.log_event(
            AuditEventType.SESSION_END,
            self.get_statistics(),
            explanation="Audit logging session ended"
        )
        
        # Export final summary
        if self.log_to_file:
            self.export_to_markdown()
            
            # Export full JSON
            filepath = self.output_dir / f"{self.session_id}_full.json"
            with open(filepath, 'w') as f:
                json.dump({
                    'session_id': self.session_id,
                    'events': [e.to_dict() for e in self._events],
                    'signals': [s.to_dict() for s in self._signal_audits],
                    'optimizations': [o.to_dict() for o in self._optimization_audits],
                    'parameters': [p.to_dict() for p in self._parameter_audits]
                }, f, indent=2)
        
        logger.info(f"Audit logger closed. Session: {self.session_id}")


def create_signal_audit_from_strategy(
    strategy,
    signal,
    features: Dict[str, float],
    parameter_source: str = "human"
) -> SignalAudit:
    """
    Helper to create SignalAudit from strategy signal.
    
    Args:
        strategy: Strategy instance
        signal: StrategySignal
        features: Current feature values
        parameter_source: "human" or "ml"
        
    Returns:
        SignalAudit instance
    """
    # Get explanation from strategy if available
    explanation = {}
    if hasattr(strategy, 'explain_signal'):
        explanation = strategy.explain_signal(signal, features)
    
    return SignalAudit(
        timestamp=signal.timestamp,
        strategy_name=strategy.__class__.__name__,
        signal_type=signal.signal_type.value,
        confidence=signal.confidence,
        price=signal.price,
        primary_reason=explanation.get('primary_reason', 'Signal generated'),
        secondary_reasons=explanation.get('secondary_reasons', []),
        triggering_features=explanation.get('triggering_features', {}),
        thresholds=explanation.get('thresholds', {}),
        market_condition=explanation.get('market_condition', ''),
        trend_direction=explanation.get('trend_direction', ''),
        volatility_level=explanation.get('volatility_level', ''),
        parameters=strategy.parameters.to_dict() if hasattr(strategy, 'parameters') else {},
        parameter_source=parameter_source
    )
