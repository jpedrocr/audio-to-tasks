"""Core business logic for AudioToTasks."""

from audio_to_tasks.core.config import AppConfig, get_config
from audio_to_tasks.core.exceptions import (
    AudioToTasksError,
    ModelLoadError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    TaskExtractionError,
    TranscriptionError,
    UnsupportedAudioFormatError,
)
from audio_to_tasks.core.models import (
    AudioFile,
    ProcessingResult,
    Task,
    TaskList,
    TaskPriority,
    TaskStatus,
    TranscriptionResult,
    TranscriptionSegment,
)
from audio_to_tasks.core.task_extractor import TaskExtractor
from audio_to_tasks.core.transcriber import Transcriber

__all__ = [
    "AppConfig",
    "AudioFile",
    "AudioToTasksError",
    "ModelLoadError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "ProcessingResult",
    "Task",
    "TaskExtractor",
    "TaskExtractionError",
    "TaskList",
    "TaskPriority",
    "TaskStatus",
    "Transcriber",
    "TranscriptionError",
    "TranscriptionResult",
    "TranscriptionSegment",
    "UnsupportedAudioFormatError",
    "get_config",
]
