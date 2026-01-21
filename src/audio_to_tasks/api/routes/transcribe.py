"""Transcription API routes."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from audio_to_tasks.api.dependencies import get_transcriber
from audio_to_tasks.api.schemas import TranscribeResponse
from audio_to_tasks.core.exceptions import (
    TranscriptionError,
    UnsupportedAudioFormatError,
)
from audio_to_tasks.core.transcriber import Transcriber

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str | None = None,
    transcriber: Transcriber = Depends(get_transcriber),
) -> TranscribeResponse:
    """Transcribe an uploaded audio file.

    Args:
        file: Audio file (mp3, wav, m4a, flac, ogg, webm).
        language: Optional language code for transcription.
        transcriber: Injected transcriber service.

    Returns:
        TranscribeResponse with transcription result.
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix.lstrip(".") not in Transcriber.SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported format: {suffix}. "
            f"Supported: {', '.join(sorted(Transcriber.SUPPORTED_FORMATS))}",
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        result = await transcriber.transcribe_async(tmp_path, language=language)

        return TranscribeResponse(success=True, data=result)

    except UnsupportedAudioFormatError as e:
        raise HTTPException(400, str(e)) from e
    except TranscriptionError as e:
        raise HTTPException(500, f"Transcription failed: {e.message}") from e
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
