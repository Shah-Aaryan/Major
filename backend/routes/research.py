"""Research API routes."""

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from backend.models.request_models import ResearchRequest
from backend.models.response_models import (
    ResearchResultsResponse,
    ResearchStartResponse,
    StrategiesResponse,
)
from backend.services.research_service import research_service

router = APIRouter(tags=["research"])


@router.post(
    "/run-research",
    response_model=ResearchStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_research(
    payload: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchStartResponse:
    """Queue a research run in the background."""
    try:
        return research_service.start_research(payload, background_tasks)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start research session.",
        ) from exc


@router.get("/results/{session_id}", response_model=ResearchResultsResponse)
def get_results(session_id: str) -> ResearchResultsResponse:
    """Fetch research results for a session."""
    try:
        return research_service.get_results(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch research results.",
        ) from exc


@router.get("/strategies", response_model=StrategiesResponse)
def list_strategies() -> StrategiesResponse:
    """Return available strategy keys."""
    return StrategiesResponse(strategies=research_service.list_strategies())
