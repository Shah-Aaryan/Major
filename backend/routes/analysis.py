"""Analysis API routes — condition analysis, failure detection, comparison report,
explainability, and leakage checking."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Query

from backend.models.request_models import (
    AnalysisRequest,
    ExplainabilityRequest,
    LeakageCheckRequest,
)
from backend.models.response_models import (
    ComparisonReportResponse,
    ConditionAnalysisResponse,
    ExplainabilityResponse,
    FailureDetectionResponse,
    LeakageCheckResponse,
)
from backend.services.analysis_service import analysis_service

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post(
    "/conditions",
    response_model=ConditionAnalysisResponse,
    summary="Analyse ML effectiveness across market conditions for a session",
)
def analyse_conditions(payload: AnalysisRequest) -> ConditionAnalysisResponse:
    try:
        return analysis_service.run_condition_analysis(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Condition analysis failed.",
        ) from exc


@router.post(
    "/failures",
    response_model=FailureDetectionResponse,
    summary="Detect ML optimisation failure patterns for a session",
)
def detect_failures(payload: AnalysisRequest) -> FailureDetectionResponse:
    try:
        return analysis_service.run_failure_detection(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failure detection failed.",
        ) from exc


@router.post(
    "/comparison-report",
    response_model=ComparisonReportResponse,
    summary="Generate a full human vs ML comparison report for a session",
)
def comparison_report(payload: AnalysisRequest) -> ComparisonReportResponse:
    try:
        return analysis_service.generate_comparison_report(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Comparison report generation failed.",
        ) from exc


@router.post(
    "/explainability",
    response_model=ExplainabilityResponse,
    summary="Generate an explainability report for ML parameter changes in a session",
)
def explainability_report(payload: ExplainabilityRequest) -> ExplainabilityResponse:
    try:
        return analysis_service.generate_explainability_report(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explainability report generation failed.",
        ) from exc


@router.post(
    "/leakage-check",
    response_model=LeakageCheckResponse,
    summary="Check a data file for feature leakage (lookahead bias, train/test overlap)",
)
def leakage_check(payload: LeakageCheckRequest) -> LeakageCheckResponse:
    try:
        return analysis_service.check_leakage(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Leakage check failed.",
        ) from exc
