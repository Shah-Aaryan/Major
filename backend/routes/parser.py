"""Natural language strategy parsing API routes."""

from fastapi import APIRouter, HTTPException, status

from backend.models.request_models import AlgoConvertRequest
from backend.models.response_models import AlgoConvertResponse
from strategies.algo_converter import AlgoConverter

router = APIRouter(prefix="/parser", tags=["parser"])
converter = AlgoConverter()


@router.post(
    "/convert-algo",
    response_model=AlgoConvertResponse,
    summary="Convert natural language strategy into fixed variable format",
)
def convert_algo(payload: AlgoConvertRequest) -> AlgoConvertResponse:
    """
    Take a natural language algorithm description and convert it into
    a structured JSON format with variables and rules.
    """
    try:
        result = converter.convert_to_variables(payload.algo_text)
        return AlgoConvertResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to convert algorithm: {str(exc)}",
        ) from exc
