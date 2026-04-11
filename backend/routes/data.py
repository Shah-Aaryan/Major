"""Data management API routes — list files, quality reports, session listing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.models.request_models import DataInfoRequest
from backend.models.response_models import (
    DataListResponse,
    DataQualityResponse,
    SessionListResponse,
)
from backend.services.data_service import data_service

router = APIRouter(prefix="/data", tags=["data"])


@router.get(
    "/files",
    response_model=DataListResponse,
    summary="List all available CSV data files in the project",
)
def list_data_files() -> DataListResponse:
    try:
        return data_service.list_data_files()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list data files.",
        ) from exc


@router.post(
    "/quality",
    response_model=DataQualityResponse,
    summary="Run a data quality report on a CSV file",
)
def data_quality(payload: DataInfoRequest) -> DataQualityResponse:
    try:
        return data_service.get_quality_report(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data quality report failed.",
        ) from exc


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List all completed research sessions",
)
def list_sessions() -> SessionListResponse:
    try:
        return data_service.list_sessions()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions.",
        ) from exc
