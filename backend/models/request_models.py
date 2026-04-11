"""Request models for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Strategy names shared across multiple requests ──────────────────────────

_SUPPORTED_STRATEGIES = {"rsi_mean_reversion", "ema_crossover", "bollinger_breakout"}


def _validate_strategy(value: str) -> str:
    if value not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"strategy must be one of: {', '.join(sorted(_SUPPORTED_STRATEGIES))}")
    return value


# ─── Research ────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """Payload for starting a research run."""

    data_path: str = Field(..., examples=["data/raw/btcusdt_1m.csv"])
    strategy: str = Field(..., examples=["rsi_mean_reversion"])
    timeframes: list[str] = Field(default_factory=lambda: ["5m"], min_length=1)
    trials: int = Field(default=50, ge=1)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        return _validate_strategy(value)


# ─── Paper Trading ────────────────────────────────────────────────────────────

class PaperTradingStartRequest(BaseModel):
    """Payload for starting a paper trading session."""

    data_path: str = Field(..., examples=["data/raw/btcusdt_1m.csv"])
    symbol: str = Field(default="BTCUSDT")
    strategy: str = Field(default="rsi_mean_reversion")
    replay_speed: float = Field(default=60.0, gt=0)
    initial_capital: float = Field(default=100000.0, gt=0)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        return _validate_strategy(value)


# ─── Analysis ────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Run condition / failure / comparison analysis on a past research session."""

    session_id: str = Field(..., description="Session ID returned by /run-research")


class ExplainabilityRequest(BaseModel):
    """Generate an explainability report for ML parameter changes."""

    session_id: str
    strategy: str = Field(..., examples=["rsi_mean_reversion"])

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        return _validate_strategy(value)


class LeakageCheckRequest(BaseModel):
    """Check features in a data file for leakage."""

    data_path: str = Field(..., examples=["data/raw/btcusdt_1m.csv"])
    train_ratio: float = Field(default=0.8, gt=0.1, lt=1.0)


# ─── Backtesting ─────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    """Run a standalone backtest."""

    data_path: str = Field(..., examples=["data/raw/btcusdt_1m.csv"])
    strategy: str = Field(..., examples=["rsi_mean_reversion"])
    timeframe: str = Field(default="5m")
    params: dict[str, Any] = Field(default_factory=dict, description="Override strategy parameters")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        return _validate_strategy(value)


class WalkForwardRequest(BaseModel):
    """Run walk-forward validation."""

    data_path: str = Field(..., examples=["data/raw/btcusdt_1m.csv"])
    strategy: str = Field(..., examples=["rsi_mean_reversion"])
    timeframe: str = Field(default="5m")
    n_windows: int = Field(default=5, ge=2)
    train_ratio: float = Field(default=0.8, gt=0.1, lt=1.0)
    trials: int = Field(default=20, ge=1)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        return _validate_strategy(value)


# ─── Optimization ────────────────────────────────────────────────────────────

class OptimizationRequest(BaseModel):
    """Run a standalone optimization."""

    data_path: str = Field(..., examples=["data/raw/btcusdt_1m.csv"])
    strategy: str = Field(..., examples=["rsi_mean_reversion"])
    optimizer: str = Field(default="bayesian_gp", examples=["bayesian_gp", "random_search", "cma_es"])
    trials: int = Field(default=50, ge=1)
    timeframe: str = Field(default="5m")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, value: str) -> str:
        return _validate_strategy(value)


# ─── Features ────────────────────────────────────────────────────────────────

class FeatureRequest(BaseModel):
    """Generate and return feature summary for a data file."""

    data_path: str = Field(..., examples=["data/raw/btcusdt_1m.csv"])
    timeframe: str = Field(default="5m")
    feature_groups: Optional[list[str]] = Field(
        default=None,
        description="Subset of groups: price, trend, momentum, volatility, volume, regime",
    )


# ─── Data ────────────────────────────────────────────────────────────────────

class DataInfoRequest(BaseModel):
    """Return metadata/quality report for a CSV data file."""

    data_path: str = Field(..., examples=["data/raw/btcusdt_1m.csv"])
    timeframe: str = Field(default="1m")


# ─── Audit ───────────────────────────────────────────────────────────────────

class AuditVerifyRequest(BaseModel):
    """Verify the integrity of an audit trail."""

    session_id: str


# ─── Natural Language Parsing ────────────────────────────────────────────────

class AlgoConvertRequest(BaseModel):
    """Payload for converting natural language strategy into variable format."""

    algo_text: str = Field(..., examples=["buy btc if rsi < 30 and adx > 25, sell btc if rsi > 70"])
