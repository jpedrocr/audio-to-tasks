"""Shared test fixtures and configuration."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from audio_to_tasks.core.config import AppConfig, OllamaConfig, WhisperConfig
from audio_to_tasks.core.models import (
    Task,
    TaskList,
    TaskPriority,
    TranscriptionResult,
    TranscriptionSegment,
)


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_audio_path(fixtures_dir: Path) -> Path:
    """Return path to sample audio file."""
    return fixtures_dir / "audio" / "sample_short.mp3"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_audio_file(temp_dir: Path) -> Path:
    """Create a temporary audio file for testing."""
    audio_file = temp_dir / "test.mp3"
    audio_file.write_bytes(b"fake audio content for testing")
    return audio_file


@pytest.fixture
def test_whisper_config() -> WhisperConfig:
    """Create test Whisper configuration."""
    return WhisperConfig(model_size="tiny", device="cpu")


@pytest.fixture
def test_ollama_config() -> OllamaConfig:
    """Create test Ollama configuration."""
    return OllamaConfig(model="gemma3:4b")


@pytest.fixture
def test_config() -> AppConfig:
    """Create test configuration."""
    return AppConfig(
        debug=True,
        whisper=WhisperConfig(model_size="tiny", device="cpu"),
        ollama=OllamaConfig(model="gemma3:4b"),
    )


@pytest.fixture
def sample_transcription() -> TranscriptionResult:
    """Create sample transcription result."""
    return TranscriptionResult(
        text="We need to finish the quarterly report by Friday. "
        "John will handle the graphics. "
        "Also, schedule a meeting with the marketing team next week.",
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=3.5,
                text="We need to finish the quarterly report by Friday.",
            ),
            TranscriptionSegment(
                start=3.5, end=6.0, text="John will handle the graphics."
            ),
            TranscriptionSegment(
                start=6.0,
                end=10.0,
                text="Also, schedule a meeting with the marketing team next week.",
            ),
        ],
        language="en",
        language_probability=0.98,
        duration_seconds=10.0,
        audio_path="/tmp/test.mp3",
    )


@pytest.fixture
def sample_task_list() -> TaskList:
    """Create sample task list."""
    return TaskList(
        tasks=[
            Task(
                title="Finish quarterly report",
                priority=TaskPriority.HIGH,
                due_date=None,
                tags=["report", "quarterly"],
            ),
            Task(
                title="Handle graphics for report",
                priority=TaskPriority.MEDIUM,
                assignee="John",
                tags=["graphics"],
            ),
            Task(
                title="Schedule meeting with marketing team",
                priority=TaskPriority.MEDIUM,
                tags=["meeting", "marketing"],
            ),
        ],
        source_audio="/tmp/test.mp3",
        language="en",
    )


@pytest.fixture
def mock_whisper_model() -> Generator[MagicMock, None, None]:
    """Mock Whisper model for testing without GPU."""
    with patch("faster_whisper.WhisperModel") as mock:
        model_instance = MagicMock()
        mock.return_value = model_instance

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = "This is a test transcription."

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        model_instance.transcribe.return_value = (iter([mock_segment]), mock_info)

        yield mock


@pytest.fixture
def mock_ollama_client() -> Generator[MagicMock, None, None]:
    """Mock Ollama client for testing without server."""
    with patch("ollama.Client") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance

        client_instance.chat.return_value = {
            "message": {
                "content": '{"tasks": [{"title": "Test task", "priority": "medium", "tags": []}]}'
            }
        }

        client_instance.list.return_value = {"models": [{"name": "gemma3:4b"}]}

        yield mock


@pytest.fixture
def api_client(
    mock_whisper_model: MagicMock, mock_ollama_client: MagicMock
) -> TestClient:
    """Create test client for API."""
    from audio_to_tasks.api.app import create_app

    app = create_app()
    return TestClient(app)
