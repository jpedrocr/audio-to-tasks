"""Tests for TaskExtractor service."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from audio_to_tasks.core.config import OllamaConfig
from audio_to_tasks.core.exceptions import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from audio_to_tasks.core.models import TaskPriority, TranscriptionResult
from audio_to_tasks.core.task_extractor import TaskExtractor


class TestTaskExtractor:
    """Tests for TaskExtractor class."""

    def test_init_default_config(self) -> None:
        """Test task extractor initialization with default config."""
        extractor = TaskExtractor()
        assert extractor._config is not None
        assert extractor._client is None

    def test_init_custom_config(self, test_ollama_config: OllamaConfig) -> None:
        """Test task extractor initialization with custom config."""
        extractor = TaskExtractor(test_ollama_config)
        assert extractor._config == test_ollama_config

    def test_extract_tasks_from_text(self, mock_ollama_client: MagicMock) -> None:
        """Test basic task extraction from text."""
        extractor = TaskExtractor()

        result = extractor.extract_tasks("We need to finish the report by Friday.")

        assert result.task_count >= 1
        assert result.tasks[0].title == "Test task"
        assert result.tasks[0].priority == TaskPriority.MEDIUM

    def test_extract_tasks_from_transcription(
        self,
        mock_ollama_client: MagicMock,
        sample_transcription: TranscriptionResult,
    ) -> None:
        """Test task extraction from TranscriptionResult."""
        extractor = TaskExtractor()

        result = extractor.extract_tasks(sample_transcription)

        assert result.source_audio == sample_transcription.audio_path
        assert result.language == "en"
        assert result.total_duration_seconds == sample_transcription.duration_seconds

    def test_extract_tasks_empty_text(self, mock_ollama_client: MagicMock) -> None:
        """Test extraction from empty text."""
        extractor = TaskExtractor()

        result = extractor.extract_tasks("")

        assert result.task_count == 0
        assert result.tasks == []

    def test_extract_tasks_whitespace_only(self, mock_ollama_client: MagicMock) -> None:
        """Test extraction from whitespace-only text."""
        extractor = TaskExtractor()

        result = extractor.extract_tasks("   \n\t  ")

        assert result.task_count == 0

    def test_parse_json_with_markdown(self, mock_ollama_client: MagicMock) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        mock_ollama_client.return_value.chat.return_value = {
            "message": {
                "content": '```json\n{"tasks": [{"title": "Markdown task", "priority": "high", "tags": []}]}\n```'
            }
        }

        extractor = TaskExtractor()
        result = extractor.extract_tasks("Test text")

        assert result.tasks[0].title == "Markdown task"
        assert result.tasks[0].priority == TaskPriority.HIGH

    def test_parse_json_plain(self, mock_ollama_client: MagicMock) -> None:
        """Test parsing plain JSON response."""
        mock_ollama_client.return_value.chat.return_value = {
            "message": {
                "content": '{"tasks": [{"title": "Plain task", "priority": "low", "tags": ["test"]}]}'
            }
        }

        extractor = TaskExtractor()
        result = extractor.extract_tasks("Test text")

        assert result.tasks[0].title == "Plain task"
        assert result.tasks[0].priority == TaskPriority.LOW
        assert "test" in result.tasks[0].tags

    def test_parse_invalid_priority_defaults_to_medium(
        self, mock_ollama_client: MagicMock
    ) -> None:
        """Test that invalid priority defaults to medium."""
        mock_ollama_client.return_value.chat.return_value = {
            "message": {
                "content": '{"tasks": [{"title": "Task", "priority": "invalid", "tags": []}]}'
            }
        }

        extractor = TaskExtractor()
        result = extractor.extract_tasks("Test text")

        assert result.tasks[0].priority == TaskPriority.MEDIUM

    def test_skip_invalid_tasks(self, mock_ollama_client: MagicMock) -> None:
        """Test that invalid tasks are skipped."""
        mock_ollama_client.return_value.chat.return_value = {
            "message": {
                "content": '{"tasks": [{"title": ""}, {"title": "Valid task", "priority": "medium", "tags": []}]}'
            }
        }

        extractor = TaskExtractor()
        result = extractor.extract_tasks("Test text")

        assert result.task_count == 1
        assert result.tasks[0].title == "Valid task"

    def test_check_connection_success(self, mock_ollama_client: MagicMock) -> None:
        """Test successful connection check."""
        extractor = TaskExtractor()

        assert extractor.check_connection() is True

    def test_check_connection_model_not_found(
        self, mock_ollama_client: MagicMock
    ) -> None:
        """Test connection check when model is missing."""
        mock_ollama_client.return_value.list.return_value = {
            "models": [{"name": "other-model"}]
        }

        extractor = TaskExtractor()

        with pytest.raises(OllamaModelNotFoundError) as exc_info:
            extractor.check_connection()

        assert "gemma3" in str(exc_info.value)

    def test_check_connection_server_error(self, mock_ollama_client: MagicMock) -> None:
        """Test connection check when server is unreachable."""
        mock_ollama_client.return_value.list.side_effect = Exception(
            "Connection refused"
        )

        extractor = TaskExtractor()

        with pytest.raises(OllamaConnectionError) as exc_info:
            extractor.check_connection()

        assert "Cannot connect" in str(exc_info.value)

    def test_lazy_client_loading(self, mock_ollama_client: MagicMock) -> None:
        """Test that client is loaded lazily."""
        extractor = TaskExtractor()

        mock_ollama_client.assert_not_called()

        _ = extractor.client

        mock_ollama_client.assert_called_once()


class TestTaskExtractorAsync:
    """Tests for async task extraction methods."""

    @pytest.mark.asyncio
    async def test_extract_tasks_async(self, mock_ollama_client: MagicMock) -> None:
        """Test async task extraction."""
        extractor = TaskExtractor()

        result = await extractor.extract_tasks_async("Test transcription text")

        assert result.task_count >= 1

    @pytest.mark.asyncio
    async def test_extract_tasks_async_from_transcription(
        self,
        mock_ollama_client: MagicMock,
        sample_transcription: TranscriptionResult,
    ) -> None:
        """Test async task extraction from TranscriptionResult."""
        extractor = TaskExtractor()

        result = await extractor.extract_tasks_async(sample_transcription)

        assert result.source_audio == sample_transcription.audio_path
