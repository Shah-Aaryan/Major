"""Paper trading API routes."""

from fastapi import APIRouter, HTTPException, status

from backend.models.request_models import PaperTradingStartRequest
from backend.models.response_models import (
    PaperTradingStartResponse,
    PaperTradingStatsResponse,
    PaperTradingStopResponse,
)
from backend.services.trading_service import trading_service

router = APIRouter(tags=["trading"])


@router.post(
    "/start-paper-trading",
    response_model=PaperTradingStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_paper_trading(payload: PaperTradingStartRequest) -> PaperTradingStartResponse:
    """Start a paper trading session."""
    try:
        return trading_service.start(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start paper trading session.",
        ) from exc


@router.post("/stop-paper-trading", response_model=PaperTradingStopResponse)
def stop_paper_trading() -> PaperTradingStopResponse:
    """Stop the active paper trading session."""
    try:
        return trading_service.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop paper trading session.",
        ) from exc


@router.get("/trading-stats", response_model=PaperTradingStatsResponse)
def get_trading_stats() -> PaperTradingStatsResponse:
    """Return the current paper trading statistics."""
    try:
        return trading_service.get_stats()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load paper trading stats.",
        ) from exc
