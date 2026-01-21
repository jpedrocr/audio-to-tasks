"""Tests for Transcriber service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audio_to_tasks.core.config import WhisperConfig
from audio_to_tasks.core.exceptions import (
    UnsupportedAudioFormatError,
)
from audio_to_tasks.core.transcriber import Transcriber


class TestTranscriber:
    """Tests for Transcriber class."""

    def test_supported_formats(self) -> None:
        """Test supported formats are defined."""
        assert "mp3" in Transcriber.SUPPORTED_FORMATS
        assert "wav" in Transcriber.SUPPORTED_FORMATS
        assert "m4a" in Transcriber.SUPPORTED_FORMATS
        assert "flac" in Transcriber.SUPPORTED_FORMATS
        assert "ogg" in Transcriber.SUPPORTED_FORMATS
        assert "webm" in Transcriber.SUPPORTED_FORMATS

    def test_init_default_config(self) -> None:
        """Test transcriber initialization with default config."""
        transcriber = Transcriber()
        assert transcriber._config is not None
        assert transcriber._model is None

    def test_init_custom_config(self, test_whisper_config: WhisperConfig) -> None:
        """Test transcriber initialization with custom config."""
        transcriber = Transcriber(test_whisper_config)
        assert transcriber._config == test_whisper_config

    def test_validate_supported_format(self, temp_audio_file: Path) -> None:
        """Test validation of supported audio format."""
        transcriber = Transcriber()
        result = transcriber.validate_audio_file(temp_audio_file)

        assert result.path == temp_audio_file
        assert result.format == "mp3"

    def test_validate_unsupported_format(self, temp_dir: Path) -> None:
        """Test rejection of unsupported format."""
        audio_file = temp_dir / "test.xyz"
        audio_file.write_bytes(b"fake data")

        transcriber = Transcriber()

        with pytest.raises(UnsupportedAudioFormatError) as exc_info:
            transcriber.validate_audio_file(audio_file)

        assert "xyz" in str(exc_info.value)
        assert "Supported" in str(exc_info.value)

    def test_validate_file_not_found(self) -> None:
        """Test validation error when file doesn't exist."""
        transcriber = Transcriber()

        with pytest.raises(ValueError, match="not found"):
            transcriber.validate_audio_file(Path("/nonexistent/file.mp3"))

    def test_transcribe_with_mock(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test transcription with mocked Whisper."""
        config = WhisperConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        result = transcriber.transcribe(temp_audio_file)

        assert result.text == "This is a test transcription."
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert len(result.segments) == 1
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 5.0
        assert result.audio_path == str(temp_audio_file)

    def test_transcribe_with_language(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test transcription with specified language."""
        config = WhisperConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        transcriber.transcribe(temp_audio_file, language="es")

        mock_whisper_model.return_value.transcribe.assert_called_once()
        call_kwargs = mock_whisper_model.return_value.transcribe.call_args[1]
        assert call_kwargs["language"] == "es"

    def test_transcribe_unsupported_format(self, temp_dir: Path) -> None:
        """Test transcription error for unsupported format."""
        audio_file = temp_dir / "test.xyz"
        audio_file.write_bytes(b"fake data")

        transcriber = Transcriber()

        with pytest.raises(UnsupportedAudioFormatError):
            transcriber.transcribe(audio_file)

    def test_lazy_model_loading(self, mock_whisper_model: MagicMock) -> None:
        """Test that model is loaded lazily."""
        config = WhisperConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        mock_whisper_model.assert_not_called()

        _ = transcriber.model

        mock_whisper_model.assert_called_once()

    def test_model_reused(self, mock_whisper_model: MagicMock) -> None:
        """Test that model is reused across calls."""
        config = WhisperConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        _ = transcriber.model
        _ = transcriber.model

        mock_whisper_model.assert_called_once()


class TestTranscriberAsync:
    """Tests for async transcription methods."""

    @pytest.mark.asyncio
    async def test_transcribe_async(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test async transcription."""
        config = WhisperConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        result = await transcriber.transcribe_async(temp_audio_file)

        assert result.text == "This is a test transcription."
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_async_with_language(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test async transcription with language."""
        config = WhisperConfig(model_size="tiny", device="cpu")
        transcriber = Transcriber(config)

        result = await transcriber.transcribe_async(temp_audio_file, language="fr")

        assert result.text == "This is a test transcription."
