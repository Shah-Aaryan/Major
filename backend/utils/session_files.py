"""Helpers for session file discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_research_output_dir() -> Path:
    """Resolve the project research output directory."""
    return (Path.cwd() / "research_output").resolve()


def load_session_file(session_id: str, output_dir: Path | None = None) -> dict[str, Any] | None:
    """Load a persisted research session JSON file if it exists."""
    base_dir = output_dir or get_research_output_dir()
    session_path = base_dir / f"session_{session_id}.json"
    if not session_path.exists():
        return None

    with session_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
