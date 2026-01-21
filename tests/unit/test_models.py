"""Tests for Pydantic models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

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


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_values(self) -> None:
        """Test all priority values exist."""
        assert TaskPriority.LOW.value == "low"
        assert TaskPriority.MEDIUM.value == "medium"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.URGENT.value == "urgent"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self) -> None:
        """Test all status values exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTask:
    """Tests for Task model."""

    def test_task_creation_minimal(self) -> None:
        """Test creating task with minimal fields."""
        task = Task(title="Test task")

        assert task.title == "Test task"
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.description is None
        assert task.tags == []
        assert task.assignee is None
        assert task.due_date is None

    def test_task_creation_full(self) -> None:
        """Test creating task with all fields."""
        due = datetime(2024, 12, 31)
        task = Task(
            title="Complete project",
            description="Finish all remaining work",
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            due_date=due,
            assignee="John",
            tags=["project", "urgent"],
        )

        assert task.title == "Complete project"
        assert task.description == "Finish all remaining work"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.due_date == due
        assert task.assignee == "John"
        assert "project" in task.tags
        assert "urgent" in task.tags

    def test_task_title_empty_rejected(self) -> None:
        """Test that empty title is rejected."""
        with pytest.raises(ValidationError):
            Task(title="")

    def test_task_title_whitespace_stripped(self) -> None:
        """Test that title whitespace is stripped."""
        task = Task(title="  Test task  ")
        assert task.title == "Test task"

    def test_task_title_max_length(self) -> None:
        """Test that title exceeding max length is rejected."""
        with pytest.raises(ValidationError):
            Task(title="x" * 201)

    def test_task_tags_normalization(self) -> None:
        """Test that tags are normalized to lowercase."""
        task = Task(title="Test", tags=["TAG1", "  Tag2  ", "TAG3"])
        assert task.tags == ["tag1", "tag2", "tag3"]

    def test_task_tags_empty_strings_removed(self) -> None:
        """Test that empty tag strings are removed."""
        task = Task(title="Test", tags=["tag1", "", "  ", "tag2"])
        assert task.tags == ["tag1", "tag2"]

    def test_task_serialization(self) -> None:
        """Test task JSON serialization."""
        task = Task(title="Test task", priority=TaskPriority.HIGH)
        data = task.model_dump()

        assert data["title"] == "Test task"
        assert data["priority"] == TaskPriority.HIGH

    def test_task_json_round_trip(self) -> None:
        """Test task JSON serialization and deserialization."""
        task = Task(
            title="Test task",
            priority=TaskPriority.HIGH,
            tags=["test"],
        )
        json_str = task.model_dump_json()
        restored = Task.model_validate_json(json_str)

        assert restored.title == task.title
        assert restored.priority == task.priority
        assert restored.tags == task.tags


class TestTaskList:
    """Tests for TaskList model."""

    def test_task_list_empty(self) -> None:
        """Test empty task list."""
        task_list = TaskList()

        assert task_list.task_count == 0
        assert task_list.pending_count == 0
        assert task_list.tasks == []

    def test_task_list_with_tasks(self, sample_task_list: TaskList) -> None:
        """Test task list with tasks."""
        assert sample_task_list.task_count == 3
        assert sample_task_list.pending_count == 3

    def test_task_list_pending_count(self) -> None:
        """Test pending count with mixed statuses."""
        task_list = TaskList(
            tasks=[
                Task(title="Task 1", status=TaskStatus.PENDING),
                Task(title="Task 2", status=TaskStatus.COMPLETED),
                Task(title="Task 3", status=TaskStatus.PENDING),
                Task(title="Task 4", status=TaskStatus.IN_PROGRESS),
            ]
        )

        assert task_list.task_count == 4
        assert task_list.pending_count == 2

    def test_task_list_metadata(self) -> None:
        """Test task list metadata fields."""
        task_list = TaskList(
            tasks=[Task(title="Task 1")],
            source_audio="/path/to/audio.mp3",
            language="en",
            total_duration_seconds=120.5,
        )

        assert task_list.source_audio == "/path/to/audio.mp3"
        assert task_list.language == "en"
        assert task_list.total_duration_seconds == 120.5
        assert task_list.created_at is not None


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment model."""

    def test_segment_creation(self) -> None:
        """Test creating transcription segment."""
        segment = TranscriptionSegment(start=1.0, end=5.5, text="Hello world")

        assert segment.start == 1.0
        assert segment.end == 5.5
        assert segment.text == "Hello world"

    def test_segment_duration_property(self) -> None:
        """Test segment duration property."""
        segment = TranscriptionSegment(start=1.0, end=5.5, text="Test")
        assert segment.duration == 4.5

    def test_segment_negative_start_rejected(self) -> None:
        """Test that negative start time is rejected."""
        with pytest.raises(ValidationError):
            TranscriptionSegment(start=-1.0, end=5.0, text="Test")

    def test_segment_negative_end_rejected(self) -> None:
        """Test that negative end time is rejected."""
        with pytest.raises(ValidationError):
            TranscriptionSegment(start=0.0, end=-1.0, text="Test")


class TestTranscriptionResult:
    """Tests for TranscriptionResult model."""

    def test_transcription_result_creation(self) -> None:
        """Test creating transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            language_probability=0.95,
            duration_seconds=5.0,
        )

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert result.duration_seconds == 5.0
        assert len(result.segments) == 0
        assert result.audio_path is None

    def test_transcription_result_with_segments(self) -> None:
        """Test transcription result with segments."""
        segments = [
            TranscriptionSegment(start=0.0, end=2.5, text="Hello"),
            TranscriptionSegment(start=2.5, end=5.0, text="world"),
        ]
        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            language_probability=0.95,
            duration_seconds=5.0,
        )

        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello"
        assert result.segments[1].text == "world"

    def test_transcription_result_invalid_probability(self) -> None:
        """Test that invalid probability is rejected."""
        with pytest.raises(ValidationError):
            TranscriptionResult(
                text="Test",
                language="en",
                language_probability=1.5,
                duration_seconds=5.0,
            )


class TestAudioFile:
    """Tests for AudioFile model."""

    def test_audio_file_valid(self, temp_audio_file: Path) -> None:
        """Test creating audio file with valid path."""
        audio_file = AudioFile(path=temp_audio_file)

        assert audio_file.path == temp_audio_file
        assert audio_file.format == "mp3"

    def test_audio_file_format_inference(self, temp_dir: Path) -> None:
        """Test format inference from file extension."""
        wav_file = temp_dir / "test.wav"
        wav_file.write_bytes(b"fake wav data")

        audio_file = AudioFile(path=wav_file)
        assert audio_file.format == "wav"

    def test_audio_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(ValidationError, match="not found"):
            AudioFile(path=Path("/nonexistent/file.mp3"))

    def test_audio_file_is_directory(self, temp_dir: Path) -> None:
        """Test error when path is a directory."""
        with pytest.raises(ValidationError, match="not a file"):
            AudioFile(path=temp_dir)


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_processing_result_creation(
        self,
        sample_transcription: TranscriptionResult,
        sample_task_list: TaskList,
    ) -> None:
        """Test creating processing result."""
        result = ProcessingResult(
            transcription=sample_transcription,
            task_list=sample_task_list,
            processing_time_seconds=2.5,
        )

        assert result.transcription == sample_transcription
        assert result.task_list == sample_task_list
        assert result.processing_time_seconds == 2.5
