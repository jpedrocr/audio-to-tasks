"""Audio transcription service using faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from audio_to_tasks.core.config import WhisperConfig, get_config
from audio_to_tasks.core.exceptions import (
    ModelLoadError,
    TranscriptionError,
    UnsupportedAudioFormatError,
)
from audio_to_tasks.core.models import (
    AudioFile,
    TranscriptionResult,
    TranscriptionSegment,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    """Service for transcribing audio files to text using Whisper."""

    SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "flac", "ogg", "webm", "wma", "opus"}

    def __init__(self, config: WhisperConfig | None = None) -> None:
        """Initialize transcriber with configuration.

        Args:
            config: Whisper configuration. If None, uses default from env.
        """
        self._config = config or get_config().whisper
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        """Lazy-load and return the Whisper model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> WhisperModel:
        """Load the Whisper model.

        Returns:
            Loaded WhisperModel instance.

        Raises:
            ModelLoadError: If model cannot be loaded.
        """
        try:
            from faster_whisper import WhisperModel

            device = self._config.device
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            logger.info(
                f"Loading Whisper model '{self._config.model_size}' "
                f"on device '{device}'"
            )

            return WhisperModel(
                self._config.model_size,
                device=device,
                compute_type=self._config.compute_type,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Whisper model: {e}",
                details={"model_size": self._config.model_size},
            ) from e

    def validate_audio_file(self, path: Path) -> AudioFile:
        """Validate audio file exists and has supported format.

        Args:
            path: Path to audio file.

        Returns:
            Validated AudioFile instance.

        Raises:
            UnsupportedAudioFormatError: If format not supported.
            ValueError: If file doesn't exist.
        """
        audio_file = AudioFile(path=path)

        format_lower = (audio_file.format or "").lower()
        if format_lower not in self.SUPPORTED_FORMATS:
            raise UnsupportedAudioFormatError(format_lower)

        return audio_file

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file.
            language: Optional language code (e.g., 'en', 'es').
                     If None, auto-detects.

        Returns:
            TranscriptionResult with full text and segments.

        Raises:
            TranscriptionError: If transcription fails.
            UnsupportedAudioFormatError: If audio format not supported.
        """
        path = Path(audio_path)
        self.validate_audio_file(path)

        try:
            logger.info(f"Transcribing: {path}")

            segments_gen, info = self.model.transcribe(
                str(path),
                beam_size=self._config.beam_size,
                vad_filter=self._config.vad_filter,
                language=language or self._config.language,
            )

            segments = []
            full_text_parts = []

            for segment in segments_gen:
                segments.append(
                    TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text.strip(),
                    )
                )
                full_text_parts.append(segment.text.strip())

            full_text = " ".join(full_text_parts)
            duration = segments[-1].end if segments else 0.0

            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"{duration:.1f}s duration, language={info.language}"
            )

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                language=info.language,
                language_probability=info.language_probability,
                duration_seconds=duration,
                audio_path=str(path),
            )

        except (UnsupportedAudioFormatError, ValueError):
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Transcription failed: {e}",
                details={"audio_path": str(path)},
            ) from e

    async def transcribe_async(
        self,
        audio_path: Path | str,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Async wrapper for transcription (runs in thread pool).

        Args:
            audio_path: Path to audio file.
            language: Optional language code.

        Returns:
            TranscriptionResult with full text and segments.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.transcribe(audio_path, language=language),
        )
