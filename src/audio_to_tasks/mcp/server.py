"""MCP Server for AudioToTasks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from audio_to_tasks.core.config import get_config
from audio_to_tasks.core.task_extractor import TaskExtractor
from audio_to_tasks.core.transcriber import Transcriber

mcp = FastMCP(
    name="AudioToTasks",
    instructions="Convert audio recordings to actionable task lists using Whisper for transcription and Ollama for task extraction.",
)

_transcriber: Transcriber | None = None
_extractor: TaskExtractor | None = None


def get_transcriber() -> Transcriber:
    """Get or create transcriber instance."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def get_extractor() -> TaskExtractor:
    """Get or create task extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = TaskExtractor()
    return _extractor


@mcp.tool()
def transcribe_audio(
    audio_path: Annotated[str, Field(description="Path to audio file to transcribe")],
    language: Annotated[
        str | None, Field(description="Language code (e.g., 'en')")
    ] = None,
) -> dict[str, Any]:
    """Transcribe an audio file to text using Whisper.

    Supports formats: mp3, wav, m4a, flac, ogg, webm.
    Returns full transcription text with segments and timing information.
    """
    transcriber = get_transcriber()
    path = Path(audio_path)

    if not path.exists():
        return {"error": f"File not found: {audio_path}"}

    try:
        result = transcriber.transcribe(path, language=language)
        return result.model_dump()
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def extract_tasks(
    text: Annotated[str, Field(description="Transcription text to extract tasks from")],
) -> dict[str, Any]:
    """Extract actionable tasks from transcription text using Ollama.

    Analyzes the text and identifies action items, deadlines, assignees, and priorities.
    Returns a structured list of tasks.
    """
    extractor = get_extractor()

    try:
        result = extractor.extract_tasks(text)
        return result.model_dump()
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def process_audio(
    audio_path: Annotated[str, Field(description="Path to audio file")],
    language: Annotated[str | None, Field(description="Language code")] = None,
) -> dict[str, Any]:
    """Process audio file: transcribe and extract tasks in one step.

    Combines transcription and task extraction for convenience.
    Returns both the transcription and extracted tasks.
    """
    transcriber = get_transcriber()
    extractor = get_extractor()
    path = Path(audio_path)

    if not path.exists():
        return {"error": f"File not found: {audio_path}"}

    try:
        transcription = transcriber.transcribe(path, language=language)
        task_list = extractor.extract_tasks(transcription)

        return {
            "transcription": transcription.model_dump(),
            "tasks": task_list.model_dump(),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def check_health() -> dict[str, Any]:
    """Check the health status of AudioToTasks services.

    Verifies Ollama connection and model availability.
    Returns status information for each component.
    """
    extractor = get_extractor()
    config = get_config()

    status: dict[str, Any] = {
        "ollama_connected": False,
        "ollama_host": config.ollama.host,
        "ollama_model": config.ollama.model,
        "whisper_model": config.whisper.model_size,
    }

    try:
        extractor.check_connection()
        status["ollama_connected"] = True
    except Exception as e:
        status["ollama_error"] = str(e)

    return status


@mcp.resource("config://settings")
def get_settings() -> dict[str, Any]:
    """Get current AudioToTasks configuration settings."""
    config = get_config()
    return {
        "whisper": {
            "model_size": config.whisper.model_size,
            "device": config.whisper.device,
            "vad_filter": config.whisper.vad_filter,
        },
        "ollama": {
            "host": config.ollama.host,
            "model": config.ollama.model,
            "temperature": config.ollama.temperature,
        },
    }


@mcp.resource("formats://supported")
def get_supported_formats() -> dict[str, Any]:
    """Get list of supported audio formats."""
    return {
        "formats": sorted(Transcriber.SUPPORTED_FORMATS),
        "description": "Audio formats supported for transcription",
    }


def run_server() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run_server()
