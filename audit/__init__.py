"""
Audit Logging Module.

Provides comprehensive logging for transparency and
explainability in ML-assisted trading research.

Tracks:
- All parameter changes (human vs ML)
- Trading signals and their explanations
- Optimization decisions
- Performance attribution
"""

from audit.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    SignalAudit,
    OptimizationAudit,
    ParameterChangeAudit
)

__all__ = [
    'AuditLogger',
    'AuditEvent',
    'AuditEventType',
    'SignalAudit',
    'OptimizationAudit',
    'ParameterChangeAudit'
]
