"""Response models for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str


# ─── Research ─────────────────────────────────────────────────────────────────

class ResearchStartResponse(BaseModel):
    """Response returned when a research session is queued."""

    session_id: str
    status: str = "started"


class ResearchResultsResponse(BaseModel):
    """Response returned for research results queries."""

    session_id: str
    status: str
    summary: dict[str, Any] = Field(default_factory=dict)
    overall_findings: dict[str, Any] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class StrategiesResponse(BaseModel):
    """Response containing available strategies."""

    strategies: list[str]


# ─── Sessions ─────────────────────────────────────────────────────────────────

class SessionListResponse(BaseModel):
    """Response listing all known research sessions."""

    sessions: list[dict[str, Any]]


# ─── Paper Trading ────────────────────────────────────────────────────────────

class PaperTradingStatsResponse(BaseModel):
    """Response containing current paper trading statistics."""

    status: str
    session: dict[str, Any] = Field(default_factory=dict)
    replay_progress: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperTradingStartResponse(BaseModel):
    """Response returned when a paper trading session starts."""

    status: str
    message: str
    trading_stats: PaperTradingStatsResponse


class PaperTradingStopResponse(BaseModel):
    """Response returned when a paper trading session stops."""

    status: str
    message: str


# ─── Analysis ─────────────────────────────────────────────────────────────────

class ConditionAnalysisResponse(BaseModel):
    """Result of condition analyzer for a research session."""

    session_id: str
    status: str
    analysis: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class FailureDetectionResponse(BaseModel):
    """Result of failure detection for a research session."""

    session_id: str
    status: str
    failures: list[dict[str, Any]] = Field(default_factory=list)
    failure_summary: dict[str, Any] = Field(default_factory=dict)
    should_use_baseline: bool = False
    reason: str = ""
    error: str | None = None


class ComparisonReportResponse(BaseModel):
    """Full comparison report (human vs ML) for a research session."""

    session_id: str
    status: str
    report: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ExplainabilityResponse(BaseModel):
    """Explainability report for ML parameter changes."""

    session_id: str
    strategy: str
    status: str
    report: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class LeakageCheckResponse(BaseModel):
    """Leakage check result for a data file."""

    data_path: str
    status: str
    passed: bool = False
    critical_issues: int = 0
    warnings_count: int = 0
    report: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ─── Backtesting ──────────────────────────────────────────────────────────────

class BacktestResponse(BaseModel):
    """Result of a standalone backtest run."""

    status: str
    strategy: str
    timeframe: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    n_trades: int = 0
    error: str | None = None


class WalkForwardResponse(BaseModel):
    """Result of walk-forward validation."""

    status: str
    strategy: str
    timeframe: str
    n_windows: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ─── Optimization ─────────────────────────────────────────────────────────────

class OptimizationResponse(BaseModel):
    """Result of a standalone optimization run."""

    status: str
    strategy: str
    optimizer: str
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_score: float = 0.0
    n_trials: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class OptimizersListResponse(BaseModel):
    """List of available optimizers."""

    optimizers: list[dict[str, Any]]


# ─── Features ─────────────────────────────────────────────────────────────────

class FeatureListResponse(BaseModel):
    """List of available features/indicators."""

    indicators: list[dict[str, Any]]
    total_implemented: int = 0
    total_planned: int = 0


class FeatureSummaryResponse(BaseModel):
    """Feature summary for a generated feature set."""

    status: str
    data_path: str
    timeframe: str
    n_features: int = 0
    n_rows: int = 0
    feature_groups: dict[str, int] = Field(default_factory=dict)
    sample_stats: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ─── Data ─────────────────────────────────────────────────────────────────────

class DataFileInfo(BaseModel):
    """Info about a single data file."""

    filename: str
    path: str
    size_mb: float
    n_rows: int
    columns: list[str]
    start_date: str | None = None
    end_date: str | None = None
    timeframe: str = "unknown"


class DataListResponse(BaseModel):
    """List of available data files."""

    files: list[DataFileInfo]
    total: int = 0


class DataQualityResponse(BaseModel):
    """Data quality report for a CSV file."""

    status: str
    data_path: str
    quality_score: float = 0.0
    total_rows: int = 0
    missing_values: dict[str, int] = Field(default_factory=dict)
    duplicates: int = 0
    gaps: int = 0
    invalid_candles: int = 0
    error: str | None = None


# ─── Audit ────────────────────────────────────────────────────────────────────

class AuditEventResponse(BaseModel):
    """A single audit event."""

    event_type: str
    timestamp: str
    data: dict[str, Any] = Field(default_factory=dict)


class AuditTrailResponse(BaseModel):
    """Full audit trail for a session."""

    session_id: str
    status: str
    events: list[dict[str, Any]] = Field(default_factory=list)
    n_events: int = 0
    error: str | None = None


class AuditVerifyResponse(BaseModel):
    """Result of audit trail integrity verification."""

    session_id: str
    status: str
    verified: bool = False
    integrity_score: float = 0.0
    issues: list[str] = Field(default_factory=list)
    error: str | None = None


# ─── Natural Language Parsing ──────────────────────────

class AlgoConvertResponse(BaseModel):
    """Response containing the structured variable format of a strategy."""

    original_input: str
    normalized_logic: str
    detected_symbols: list[str] = Field(default_factory=list)
    valid: bool = True
    variable_format: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

