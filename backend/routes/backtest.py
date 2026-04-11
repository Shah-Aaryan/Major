"""Backtesting and walk-forward validation API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.models.request_models import BacktestRequest, WalkForwardRequest
from backend.models.response_models import BacktestResponse, WalkForwardResponse
from backend.services.backtest_service import backtest_service

router = APIRouter(prefix="/backtest", tags=["backtesting"])


@router.post(
    "/run",
    response_model=BacktestResponse,
    summary="Run a standalone backtest for a strategy on a dataset",
)
def run_backtest(payload: BacktestRequest) -> BacktestResponse:
    try:
        return backtest_service.run_backtest(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backtest failed.",
        ) from exc


@router.post(
    "/walk-forward",
    response_model=WalkForwardResponse,
    summary="Run walk-forward validation (re-optimise parameters on each rolling window)",
)
def walk_forward(payload: WalkForwardRequest) -> WalkForwardResponse:
    try:
        return backtest_service.run_walk_forward(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Walk-forward validation failed.",
        ) from exc
