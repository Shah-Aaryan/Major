"""Service layer for data file operations, quality reports, and session listing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.models.request_models import DataInfoRequest
from backend.models.response_models import DataListResponse, DataFileInfo, DataQualityResponse, SessionListResponse
from backend.utils.session_files import get_research_output_dir, load_session_file
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.resampler import DataResampler

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data/raw")


class DataService:
    """Expose data loading, quality reporting, and session listing via the API."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or get_research_output_dir()

    def list_data_files(self) -> DataListResponse:
        """Return all CSV files found in the project data directories."""
        search_dirs = [_DATA_DIR, Path("data")]
        if not _DATA_DIR.is_absolute():
            search_dirs = [Path.cwd() / d for d in search_dirs]

        files: list[DataFileInfo] = []
        seen: set[str] = set()

        for data_dir in search_dirs:
            if not data_dir.exists():
                continue
            for csv_path in sorted(data_dir.rglob("*.csv")):
                key = str(csv_path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    size_mb = csv_path.stat().st_size / (1024 * 1024)
                    # Quick peek at the file
                    import pandas as pd
                    df = pd.read_csv(csv_path, nrows=5)
                    n_rows_approx = sum(1 for _ in csv_path.open()) - 1  # excluded header
                    start_date = None
                    end_date = None
                    if "timestamp" in df.columns:
                        full_df = pd.read_csv(csv_path, usecols=["timestamp"])
                        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
                        start_date = str(full_df["timestamp"].min())
                        end_date = str(full_df["timestamp"].max())

                    files.append(DataFileInfo(
                        filename=csv_path.name,
                        path=str(csv_path.resolve()),
                        size_mb=round(size_mb, 2),
                        n_rows=n_rows_approx,
                        columns=list(df.columns),
                        start_date=start_date,
                        end_date=end_date,
                    ))
                except Exception as exc:
                    logger.warning("Could not read %s: %s", csv_path, exc)

        return DataListResponse(files=files, total=len(files))

    def get_quality_report(self, payload: DataInfoRequest) -> DataQualityResponse:
        try:
            path = Path(payload.data_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            path = path.resolve()

            if not path.exists():
                raise FileNotFoundError(f"Data path '{path}' does not exist.")

            loader = DataLoader()
            preprocessor = DataPreprocessor(normalize_prices=False, normalize_volume=False)

            raw = loader.load_csv(str(path))
            # analyze_quality doesn't modify raw; it returns DataQualityReport
            report_obj = preprocessor.analyze_quality(raw, symbol=path.stem.upper())

            return DataQualityResponse(
                status="completed",
                data_path=str(path),
                quality_score=report_obj.quality_score,
                total_rows=report_obj.total_rows,
                missing_values={k: int(v) for k, v in report_obj.missing_values.items()},
                duplicates=report_obj.duplicates,
                gaps=report_obj.gaps,
                invalid_candles=report_obj.invalid_candles,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Data quality report failed for %s", payload.data_path)
            return DataQualityResponse(
                status="failed",
                data_path=payload.data_path,
                error=str(exc),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Sessions
    # ──────────────────────────────────────────────────────────────────────────

    def list_sessions(self) -> SessionListResponse:
        """Return all research sessions found in the output directory."""
        sessions: list[dict[str, Any]] = []

        if not self.output_dir.exists():
            return SessionListResponse(sessions=[])

        for session_file in sorted(
            self.output_dir.glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                with session_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                sessions.append({
                    "session_id": data.get("session_id", session_file.stem),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "status": "failed" if data.get("error") else "completed",
                    "strategies": data.get("summary", {}).get("strategies_tested", []),
                    "n_experiments": data.get("summary", {}).get("n_experiments", 0),
                    "file": session_file.name,
                })
            except Exception as exc:
                logger.debug("Could not parse session file %s: %s", session_file, exc)

        return SessionListResponse(sessions=sessions)


data_service = DataService()

