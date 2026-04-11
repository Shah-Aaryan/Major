"""Optimization API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.models.request_models import OptimizationRequest
from backend.models.response_models import OptimizationResponse, OptimizersListResponse
from backend.services.optimization_service import optimization_service

router = APIRouter(prefix="/optimize", tags=["optimization"])


@router.get(
    "/list",
    response_model=OptimizersListResponse,
    summary="List all available optimizers (implemented and planned)",
)
def list_optimizers() -> OptimizersListResponse:
    return optimization_service.list_optimizers()


@router.post(
    "/run",
    response_model=OptimizationResponse,
    summary="Run a standalone optimization for a strategy",
)
def run_optimization(payload: OptimizationRequest) -> OptimizationResponse:
    try:
        return optimization_service.run_optimization(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Optimization failed.",
        ) from exc
