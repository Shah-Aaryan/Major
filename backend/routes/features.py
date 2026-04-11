"""Feature engineering API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from backend.models.request_models import FeatureRequest
from backend.models.response_models import FeatureListResponse, FeatureSummaryResponse
from backend.services.feature_service import feature_service

router = APIRouter(prefix="/features", tags=["features"])


@router.get(
    "/indicators",
    response_model=FeatureListResponse,
    summary="List all technical indicators (implemented + planned)",
)
def list_indicators(
    implemented_only: bool = Query(False, description="Return only implemented indicators"),
) -> FeatureListResponse:
    return feature_service.list_indicators(implemented_only=implemented_only)


@router.post(
    "/summary",
    response_model=FeatureSummaryResponse,
    summary="Generate and summarise features for a data file",
)
def feature_summary(payload: FeatureRequest) -> FeatureSummaryResponse:
    try:
        return feature_service.generate_feature_summary(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature summary generation failed.",
        ) from exc
