"""Service layer for audit trail operations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.models.request_models import AuditVerifyRequest
from backend.models.response_models import AuditTrailResponse, AuditVerifyResponse
from backend.utils.session_files import get_research_output_dir

logger = logging.getLogger(__name__)


class AuditService:
    """Expose audit trail reading and verification via the API."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or get_research_output_dir()

    def get_audit_trail(self, session_id: str) -> AuditTrailResponse:
        """Read the JSONL audit events for a session."""
        try:
            audit_dir = self.output_dir / "audit"
            # Try events JSONL first
            events_file = audit_dir / f"session_{session_id}_events.jsonl"
            full_file = audit_dir / f"session_{session_id}_full.json"

            events: list[dict[str, Any]] = []

            if events_file.exists():
                with events_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            elif full_file.exists():
                with full_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                events = data.get("events", [])
            else:
                raise FileNotFoundError(
                    f"No audit trail found for session '{session_id}'."
                )

            return AuditTrailResponse(
                session_id=session_id,
                status="found",
                events=events,
                n_events=len(events),
            )
        except FileNotFoundError as exc:
            raise
        except Exception as exc:
            logger.exception("Failed to read audit trail for session %s", session_id)
            return AuditTrailResponse(
                session_id=session_id,
                status="failed",
                error=str(exc),
            )

    def verify_audit(self, payload: AuditVerifyRequest) -> AuditVerifyResponse:
        """Perform basic integrity checks on an audit trail."""
        try:
            trail = self.get_audit_trail(payload.session_id)
            if trail.status == "failed":
                raise RuntimeError(trail.error or "Could not load audit trail.")

            events = trail.events
            issues: list[str] = []

            # Check 1: Events exist
            if not events:
                issues.append("No audit events found.")

            # Check 2: All events have required fields
            required_fields = {"event_type", "timestamp"}
            for i, ev in enumerate(events):
                missing = required_fields - set(ev.keys())
                if missing:
                    issues.append(f"Event {i} missing fields: {missing}")

            # Check 3: Timestamps in order
            timestamps = []
            for ev in events:
                ts = ev.get("timestamp")
                if ts:
                    timestamps.append(str(ts))
            if timestamps != sorted(timestamps):
                issues.append("Audit events are not in chronological order.")

            verified = len(issues) == 0
            integrity_score = 1.0 - (min(len(issues), 10) / 10)

            return AuditVerifyResponse(
                session_id=payload.session_id,
                status="completed",
                verified=verified,
                integrity_score=round(integrity_score, 2),
                issues=issues,
            )
        except FileNotFoundError as exc:
            raise
        except Exception as exc:
            logger.exception("Audit verification failed for session %s", payload.session_id)
            return AuditVerifyResponse(
                session_id=payload.session_id,
                status="failed",
                error=str(exc),
            )

    def list_audit_sessions(self) -> list[dict[str, Any]]:
        """Return all sessions that have audit trails."""
        audit_dir = self.output_dir / "audit"
        if not audit_dir.exists():
            return []

        seen: set[str] = set()
        sessions: list[dict[str, Any]] = []

        for f in sorted(audit_dir.iterdir(), reverse=True):
            if not f.is_file():
                continue
            name = f.name
            # Extract session id from filename pattern: session_<id>_events.jsonl or _full.json
            for suffix in ("_events.jsonl", "_full.json"):
                if name.endswith(suffix) and name.startswith("session_"):
                    session_id = name[len("session_"):-len(suffix)]
                    if session_id not in seen:
                        seen.add(session_id)
                        sessions.append({
                            "session_id": session_id,
                            "file": name,
                            "type": "events" if suffix.endswith(".jsonl") else "full",
                        })

        return sessions


audit_service = AuditService()

