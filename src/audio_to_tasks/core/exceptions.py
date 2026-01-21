"""Custom exceptions for AudioToTasks."""

from __future__ import annotations


class AudioToTasksError(Exception):
    """Base exception for AudioToTasks."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TranscriptionError(AudioToTasksError):
    """Error during audio transcription."""

    pass


class UnsupportedAudioFormatError(TranscriptionError):
    """Audio format is not supported."""

    SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "flac", "ogg", "webm"}

    def __init__(self, format: str) -> None:
        super().__init__(
            f"Unsupported audio format: {format}. "
            f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
        )
        self.format = format


class ModelLoadError(AudioToTasksError):
    """Error loading ML model."""

    pass


class TaskExtractionError(AudioToTasksError):
    """Error during task extraction from transcription."""

    pass


class OllamaConnectionError(TaskExtractionError):
    """Cannot connect to Ollama server."""

    pass


class OllamaModelNotFoundError(TaskExtractionError):
    """Requested Ollama model is not available."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"Ollama model '{model}' not found. Please run: ollama pull {model}"
        )
        self.model = model


class ConfigurationError(AudioToTasksError):
    """Configuration error."""

    pass
