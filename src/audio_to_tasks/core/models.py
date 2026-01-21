"""Core Pydantic models for AudioToTasks."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(str, Enum):
    """Task status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """A single task extracted from transcription."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
    )

    title: Annotated[str, Field(min_length=1, max_length=200)]
    description: str | None = None
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    due_date: datetime | None = None
    assignee: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_segment: str | None = Field(
        default=None,
        description="Original transcription segment this task was extracted from",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: list[str] | None) -> list[str]:
        """Normalize tags to lowercase."""
        if v is None:
            return []
        if isinstance(v, list):
            return [
                tag.lower().strip() for tag in v if isinstance(tag, str) and tag.strip()
            ]
        return []


class TaskList(BaseModel):
    """Collection of tasks with metadata."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[Task] = Field(default_factory=list)
    source_audio: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_duration_seconds: float | None = None
    language: str | None = None

    @property
    def task_count(self) -> int:
        """Return total number of tasks."""
        return len(self.tasks)

    @property
    def pending_count(self) -> int:
        """Return count of pending tasks."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio."""

    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    text: str

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end - self.start


class TranscriptionResult(BaseModel):
    """Result of audio transcription."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Full transcription text")
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    language: str = Field(description="Detected language code")
    language_probability: float = Field(
        ge=0, le=1, description="Confidence of language detection"
    )
    duration_seconds: float = Field(ge=0, description="Total audio duration")
    audio_path: str | None = None


class AudioFile(BaseModel):
    """Representation of an audio file for processing."""

    path: Path
    format: str | None = None
    size_bytes: int | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate that path exists and is a file."""
        if not v.exists():
            raise ValueError(f"Audio file not found: {v}")
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v

    @model_validator(mode="after")
    def infer_format_from_path(self) -> AudioFile:
        """Infer format from path if not provided."""
        if self.format is None:
            self.format = self.path.suffix.lstrip(".")
        return self


class ProcessingResult(BaseModel):
    """Combined result of transcription and task extraction."""

    transcription: TranscriptionResult
    task_list: TaskList
    processing_time_seconds: float
