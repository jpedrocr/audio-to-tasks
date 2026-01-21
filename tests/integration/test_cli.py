"""Integration tests for CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from audio_to_tasks.cli.app import app

runner = CliRunner()


class TestVersionCommand:
    """Tests for version command."""

    def test_version_command(self) -> None:
        """Test version command."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "audio-to-tasks version" in result.output
        assert "0.1.0" in result.output


class TestHealthCommand:
    """Tests for health command."""

    def test_health_command_success(self, mock_ollama_client: MagicMock) -> None:
        """Test health command with working Ollama."""
        result = runner.invoke(app, ["health"])

        assert result.exit_code == 0
        assert "Connected" in result.output

    def test_health_command_failure(self, mock_ollama_client: MagicMock) -> None:
        """Test health command when Ollama is unavailable."""
        mock_ollama_client.return_value.list.side_effect = Exception(
            "Connection refused"
        )

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
        assert "Failed" in result.output


class TestTranscribeCommand:
    """Tests for transcribe command."""

    def test_transcribe_file_not_found(self) -> None:
        """Test transcribe with nonexistent file."""
        result = runner.invoke(
            app,
            ["transcribe", "file", "/nonexistent/file.mp3"],
        )

        assert result.exit_code != 0

    def test_transcribe_file_success(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test transcribe with valid file."""
        result = runner.invoke(
            app,
            ["transcribe", "file", str(temp_audio_file)],
        )

        assert result.exit_code == 0
        assert "This is a test transcription" in result.output

    def test_transcribe_file_json_output(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test transcribe with JSON output."""
        result = runner.invoke(
            app,
            ["transcribe", "file", "--json", str(temp_audio_file)],
        )

        assert result.exit_code == 0
        assert '"text"' in result.output
        assert '"language"' in result.output

    def test_transcribe_file_with_segments(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test transcribe with segment output."""
        result = runner.invoke(
            app,
            ["transcribe", "file", "--segments", str(temp_audio_file)],
        )

        assert result.exit_code == 0
        assert "[00:00" in result.output

    def test_transcribe_file_save_output(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test transcribe with output file."""
        output_file = temp_dir / "output.txt"

        result = runner.invoke(
            app,
            ["transcribe", "file", "--output", str(output_file), str(temp_audio_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "This is a test transcription" in output_file.read_text()


class TestExtractCommand:
    """Tests for extract command."""

    def test_extract_from_text(self, mock_ollama_client: MagicMock) -> None:
        """Test extract command with inline text."""
        result = runner.invoke(
            app,
            ["extract", "text", "Finish the report by Friday"],
        )

        assert result.exit_code == 0
        assert "Test task" in result.output or "EXTRACTED TASKS" in result.output

    def test_extract_from_file(
        self, mock_ollama_client: MagicMock, temp_dir: Path
    ) -> None:
        """Test extract command with file input."""
        text_file = temp_dir / "transcript.txt"
        text_file.write_text("We need to complete the project.")

        result = runner.invoke(
            app,
            ["extract", "text", "--file", str(text_file)],
        )

        assert result.exit_code == 0

    def test_extract_missing_input(self, mock_ollama_client: MagicMock) -> None:
        """Test extract command without input."""
        result = runner.invoke(app, ["extract", "text"])

        assert result.exit_code == 1
        assert "Provide text" in result.output or "ERROR" in result.output

    def test_extract_file_not_found(self, mock_ollama_client: MagicMock) -> None:
        """Test extract command with nonexistent file."""
        result = runner.invoke(
            app,
            ["extract", "text", "--file", "/nonexistent/file.txt"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output or "ERROR" in result.output

    def test_extract_json_output(self, mock_ollama_client: MagicMock) -> None:
        """Test extract command with JSON output."""
        result = runner.invoke(
            app,
            ["extract", "text", "--json", "Complete the task"],
        )

        assert result.exit_code == 0
        assert '"tasks"' in result.output


class TestProcessCommand:
    """Tests for process command."""

    def test_process_file_success(
        self,
        mock_whisper_model: MagicMock,
        mock_ollama_client: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test process command with valid file."""
        result = runner.invoke(
            app,
            ["process", "file", str(temp_audio_file)],
        )

        assert result.exit_code == 0
        assert "EXTRACTED TASKS" in result.output or "processing time" in result.output

    def test_process_file_with_transcript(
        self,
        mock_whisper_model: MagicMock,
        mock_ollama_client: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test process command showing transcript."""
        result = runner.invoke(
            app,
            ["process", "file", "--transcript", str(temp_audio_file)],
        )

        assert result.exit_code == 0
        assert "Transcript" in result.output or "This is a test" in result.output

    def test_process_file_json_output(
        self,
        mock_whisper_model: MagicMock,
        mock_ollama_client: MagicMock,
        temp_audio_file: Path,
    ) -> None:
        """Test process command with JSON output."""
        result = runner.invoke(
            app,
            ["process", "file", "--json", str(temp_audio_file)],
        )

        assert result.exit_code == 0
        assert '"transcription"' in result.output
        assert '"task_list"' in result.output

    def test_process_file_not_found(self) -> None:
        """Test process command with nonexistent file."""
        result = runner.invoke(
            app,
            ["process", "file", "/nonexistent/file.mp3"],
        )

        assert result.exit_code != 0


class TestHelpCommands:
    """Tests for help commands."""

    def test_main_help(self) -> None:
        """Test main help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "transcribe" in result.output
        assert "extract" in result.output
        assert "process" in result.output

    def test_transcribe_help(self) -> None:
        """Test transcribe help."""
        result = runner.invoke(app, ["transcribe", "--help"])

        assert result.exit_code == 0
        assert "file" in result.output

    def test_extract_help(self) -> None:
        """Test extract help."""
        result = runner.invoke(app, ["extract", "--help"])

        assert result.exit_code == 0
        assert "text" in result.output

    def test_process_help(self) -> None:
        """Test process help."""
        result = runner.invoke(app, ["process", "--help"])

        assert result.exit_code == 0
        assert "file" in result.output
