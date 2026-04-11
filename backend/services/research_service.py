"""Service layer for research operations."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks

from backend.models.request_models import ResearchRequest
from backend.models.response_models import ResearchResultsResponse, ResearchStartResponse
from backend.utils.session_files import get_research_output_dir, load_session_file
from config.settings import BacktestConfig, OptimizationConfig
from research_pipeline import ResearchPipeline

logger = logging.getLogger(__name__)


class ResearchService:
    """Wraps the existing research pipeline for API consumption."""

    STRATEGIES = [
        "rsi_mean_reversion",
        "ema_crossover",
        "bollinger_breakout",
    ]

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or get_research_output_dir()
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def list_strategies(self) -> list[str]:
        """Return supported strategy identifiers."""
        return self.STRATEGIES.copy()

    def start_research(
        self,
        payload: ResearchRequest,
        background_tasks: BackgroundTasks,
    ) -> ResearchStartResponse:
        """Create a pipeline instance and schedule the run in the background."""
        data_paths = self._resolve_data_paths(payload.data_path)

        opt_config = OptimizationConfig(
            bayesian_n_calls=payload.trials,
            random_search_n_iter=payload.trials,
        )
        pipeline = ResearchPipeline(
            optimization_config=opt_config,
            backtest_config=BacktestConfig(),
            output_dir=str(self.output_dir),
        )
        session_id = pipeline.audit_logger.session_id

        with self._lock:
            self._sessions[session_id] = {
                "status": "started",
                "started_at": datetime.utcnow().isoformat(),
                "request": payload.model_dump(),
            }

        background_tasks.add_task(
            self._run_pipeline,
            pipeline,
            session_id,
            data_paths,
            payload.timeframes,
            [payload.strategy],
        )
        logger.info("Queued research session %s", session_id)
        return ResearchStartResponse(session_id=session_id, status="started")

    def get_results(self, session_id: str) -> ResearchResultsResponse:
        """Return session status and persisted results when available."""
        session_data = load_session_file(session_id, self.output_dir)
        if session_data is not None:
            persisted_status = "failed" if session_data.get("error") else "completed"
            return ResearchResultsResponse(
                session_id=session_id,
                status=persisted_status,
                summary=session_data.get("summary", {}),
                overall_findings=session_data.get("overall_findings", {}),
                results=session_data.get("results", []),
                error=session_data.get("error"),
            )

        with self._lock:
            session_state = self._sessions.get(session_id)

        if session_state is None:
            raise FileNotFoundError(f"Session '{session_id}' was not found.")

        return ResearchResultsResponse(
            session_id=session_id,
            status=session_state["status"],
            summary={"message": "Results are not available yet."},
            overall_findings={},
            results=[],
            error=session_state.get("error"),
        )

    def _run_pipeline(
        self,
        pipeline: ResearchPipeline,
        session_id: str,
        data_paths: list[str],
        timeframes: list[str],
        strategies: list[str],
    ) -> None:
        """Execute the long-running research pipeline."""
        try:
            pipeline.run_full_research(
                data_paths=data_paths,
                timeframes=timeframes,
                strategies=strategies,
            )
            with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id]["status"] = "completed"
                    self._sessions[session_id]["completed_at"] = datetime.utcnow().isoformat()
            logger.info("Completed research session %s", session_id)
        except Exception as exc:
            logger.exception("Research session %s failed", session_id)
            self._write_failure_file(session_id, str(exc))
            with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id]["status"] = "failed"
                    self._sessions[session_id]["error"] = str(exc)

    def _resolve_data_paths(self, data_path: str) -> list[str]:
        """Resolve a file or directory of CSVs into pipeline input paths."""
        target = Path(data_path)
        if not target.is_absolute():
            target = Path.cwd() / target
        target = target.resolve()

        if not target.exists():
            raise FileNotFoundError(f"Data path '{target}' does not exist.")

        if target.is_file():
            if target.suffix.lower() != ".csv":
                raise ValueError("Research data_path must point to a CSV file or a directory of CSV files.")
            return [str(target)]

        data_files = sorted(str(path.resolve()) for path in target.glob("*.csv"))
        if not data_files:
            raise FileNotFoundError(f"No CSV files were found in '{target}'.")
        return data_files

    def _write_failure_file(self, session_id: str, message: str) -> None:
        """Persist an error payload so the frontend can inspect failed sessions."""
        failure_path = self.output_dir / f"session_{session_id}.json"
        payload = {
            "session_id": session_id,
            "start_time": None,
            "end_time": datetime.utcnow().isoformat(),
            "summary": {"message": "Research session failed."},
            "overall_findings": {},
            "results": [],
            "error": message,
        }
        with failure_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


research_service = ResearchService()

