"""Integration tests for MCP Server functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audio_to_tasks.core.config import get_config
from audio_to_tasks.core.task_extractor import TaskExtractor
from audio_to_tasks.core.transcriber import Transcriber


class TestMCPTranscribeFunction:
    """Tests for transcription functionality used by MCP."""

    def test_transcribe_audio_success(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test transcribe_audio functionality."""
        transcriber = Transcriber()
        result = transcriber.transcribe(temp_audio_file)

        assert result.text == "This is a test transcription."
        assert result.language == "en"

    def test_transcribe_audio_file_not_found(self) -> None:
        """Test transcribe_audio with nonexistent file."""
        path = Path("/nonexistent/file.mp3")

        with pytest.raises(ValueError, match="not found"):
            from audio_to_tasks.core.models import AudioFile

            AudioFile(path=path)

    def test_transcribe_audio_with_language(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test transcribe_audio with language parameter."""
        transcriber = Transcriber()
        result = transcriber.transcribe(temp_audio_file, language="es")

        assert result.text == "This is a test transcription."


class TestMCPExtractFunction:
    """Tests for task extraction functionality used by MCP."""

    def test_extract_tasks_success(self, mock_ollama_client: MagicMock) -> None:
        """Test extract_tasks functionality."""
        extractor = TaskExtractor()
        result = extractor.extract_tasks("We need to finish the report by Friday.")

        assert len(result.tasks) >= 1

    def test_extract_tasks_empty_text(self, mock_ollama_client: MagicMock) -> None:
        """Test extract_tasks with empty text."""
        extractor = TaskExtractor()
        result = extractor.extract_tasks("")

        assert len(result.tasks) == 0


class TestMCPProcessFunction:
    """Tests for process functionality used by MCP."""

    def test_process_audio_success(
        self,
        mock_whisper_model: MagicMock,
        mock_ollama_client: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test process_audio functionality."""
        transcriber = Transcriber()
        extractor = TaskExtractor()

        transcription = transcriber.transcribe(temp_audio_file)
        task_list = extractor.extract_tasks(transcription)

        assert transcription.text == "This is a test transcription."
        assert task_list.tasks is not None


class TestMCPHealthFunction:
    """Tests for health check functionality used by MCP."""

    def test_check_health_success(self, mock_ollama_client: MagicMock) -> None:
        """Test check_health functionality."""
        extractor = TaskExtractor()
        connected = extractor.check_connection()

        assert connected is True

    def test_check_health_ollama_error(self, mock_ollama_client: MagicMock) -> None:
        """Test check_health when Ollama is unavailable."""
        mock_ollama_client.return_value.list.side_effect = Exception(
            "Connection refused"
        )

        extractor = TaskExtractor()

        from audio_to_tasks.core.exceptions import OllamaConnectionError

        with pytest.raises(OllamaConnectionError):
            extractor.check_connection()


class TestMCPConfigResources:
    """Tests for configuration resources used by MCP."""

    def test_get_settings(self) -> None:
        """Test get_settings functionality."""
        config = get_config()

        assert config.whisper.model_size is not None
        assert config.ollama.host is not None
        assert config.ollama.model is not None

    def test_get_supported_formats(self) -> None:
        """Test get_supported_formats functionality."""
        formats = Transcriber.SUPPORTED_FORMATS

        assert "mp3" in formats
        assert "wav" in formats
        assert "m4a" in formats
