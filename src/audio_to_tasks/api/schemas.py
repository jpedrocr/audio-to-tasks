"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from audio_to_tasks.core.models import TaskList, TranscriptionResult


class TranscribeRequest(BaseModel):
    """Request for transcription from URL or text."""

    language: str | None = Field(
        default=None,
        description="Language code (e.g., 'en', 'es'). Auto-detect if not provided.",
    )


class TranscribeResponse(BaseModel):
    """Response containing transcription result."""

    success: bool
    data: TranscriptionResult | None = None
    error: str | None = None


class ExtractTasksRequest(BaseModel):
    """Request for task extraction from text."""

    text: str = Field(
        min_length=1, description="Transcription text to extract tasks from"
    )


class ExtractTasksResponse(BaseModel):
    """Response containing extracted tasks."""

    success: bool
    data: TaskList | None = None
    error: str | None = None


class ProcessResponse(BaseModel):
    """Response from full processing pipeline."""

    success: bool
    transcription: TranscriptionResult | None = None
    tasks: TaskList | None = None
    processing_time_seconds: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    whisper_loaded: bool
    ollama_connected: bool
    version: str
