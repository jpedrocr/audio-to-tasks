"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from audio_to_tasks import __version__
from audio_to_tasks.api.dependencies import get_task_extractor
from audio_to_tasks.api.schemas import HealthResponse
from audio_to_tasks.core.task_extractor import TaskExtractor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    extractor: TaskExtractor = Depends(get_task_extractor),
) -> HealthResponse:
    """Check application health status."""
    try:
        ollama_connected = extractor.check_connection()
    except Exception:
        ollama_connected = False

    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        whisper_loaded=False,
        ollama_connected=ollama_connected,
        version=__version__,
    )
