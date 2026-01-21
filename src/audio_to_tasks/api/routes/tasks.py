"""Task extraction API routes."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from audio_to_tasks.api.dependencies import get_task_extractor, get_transcriber
from audio_to_tasks.api.schemas import (
    ExtractTasksRequest,
    ExtractTasksResponse,
    ProcessResponse,
)
from audio_to_tasks.core.exceptions import TaskExtractionError, TranscriptionError
from audio_to_tasks.core.task_extractor import TaskExtractor
from audio_to_tasks.core.transcriber import Transcriber

router = APIRouter()


@router.post("/tasks/extract", response_model=ExtractTasksResponse)
async def extract_tasks_from_text(
    request: ExtractTasksRequest,
    extractor: TaskExtractor = Depends(get_task_extractor),
) -> ExtractTasksResponse:
    """Extract tasks from transcription text.

    Args:
        request: Request containing transcription text.
        extractor: Injected task extractor service.

    Returns:
        ExtractTasksResponse with extracted tasks.
    """
    try:
        task_list = await extractor.extract_tasks_async(request.text)
        return ExtractTasksResponse(success=True, data=task_list)
    except TaskExtractionError as e:
        raise HTTPException(500, f"Task extraction failed: {e.message}") from e


@router.post("/process", response_model=ProcessResponse)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to process"),
    language: str | None = None,
    transcriber: Transcriber = Depends(get_transcriber),
    extractor: TaskExtractor = Depends(get_task_extractor),
) -> ProcessResponse:
    """Process audio file: transcribe and extract tasks.

    Args:
        file: Audio file to process.
        language: Optional language code.
        transcriber: Injected transcriber service.
        extractor: Injected task extractor service.

    Returns:
        ProcessResponse with transcription and tasks.
    """
    start_time = time.time()
    tmp_path: Path | None = None

    try:
        suffix = Path(file.filename or ".wav").suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        transcription = await transcriber.transcribe_async(tmp_path, language=language)

        task_list = await extractor.extract_tasks_async(transcription)

        processing_time = time.time() - start_time

        return ProcessResponse(
            success=True,
            transcription=transcription,
            tasks=task_list,
            processing_time_seconds=processing_time,
        )

    except (TranscriptionError, TaskExtractionError) as e:
        raise HTTPException(500, str(e)) from e
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
