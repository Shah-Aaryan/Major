"""Audit trail API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend.models.request_models import AuditVerifyRequest
from backend.models.response_models import AuditTrailResponse, AuditVerifyResponse
from backend.services.audit_service import audit_service

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get(
    "/sessions",
    response_model=list[dict[str, Any]],
    summary="List all sessions that have audit trails",
)
def list_audit_sessions() -> list[dict[str, Any]]:
    try:
        return audit_service.list_audit_sessions()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list audit sessions.",
        ) from exc


@router.get(
    "/trail/{session_id}",
    response_model=AuditTrailResponse,
    summary="Get the full audit trail for a research session",
)
def get_audit_trail(session_id: str) -> AuditTrailResponse:
    try:
        return audit_service.get_audit_trail(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load audit trail.",
        ) from exc


@router.post(
    "/verify",
    response_model=AuditVerifyResponse,
    summary="Verify the integrity of an audit trail",
)
def verify_audit(payload: AuditVerifyRequest) -> AuditVerifyResponse:
    try:
        return audit_service.verify_audit(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit verification failed.",
        ) from exc
