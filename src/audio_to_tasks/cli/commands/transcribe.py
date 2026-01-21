"""Transcribe CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from audio_to_tasks.cli.utils import (
    console,
    format_duration,
    print_error,
    print_success,
)

app = typer.Typer(help="Transcribe audio files to text.")


@app.command("file")
def transcribe_file(
    audio_path: Annotated[
        Path,
        typer.Argument(
            help="Path to audio file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path (default: stdout)"),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Language code (e.g., 'en', 'es')"),
    ] = None,
    model_size: Annotated[
        str,
        typer.Option("--model", "-m", help="Whisper model size"),
    ] = "base",
    show_segments: Annotated[
        bool,
        typer.Option(
            "--segments", "-s", help="Show individual segments with timestamps"
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
) -> None:
    """Transcribe an audio file to text.

    Examples:
        audio-to-tasks transcribe file meeting.mp3
        audio-to-tasks transcribe file --language en --output transcript.txt audio.wav
        audio-to-tasks transcribe file --json --segments recording.m4a
    """
    from audio_to_tasks.core.config import WhisperConfig
    from audio_to_tasks.core.exceptions import TranscriptionError
    from audio_to_tasks.core.transcriber import Transcriber

    config = WhisperConfig(model_size=model_size)  # type: ignore[arg-type]
    transcriber = Transcriber(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        try:
            progress.add_task(description="Transcribing...", total=None)
            result = transcriber.transcribe(audio_path, language=language)
        except TranscriptionError as e:
            print_error(f"Transcription failed: {e.message}")
            raise typer.Exit(1) from None

    if json_output:
        output_text = result.model_dump_json(indent=2)
    elif show_segments:
        lines = []
        for seg in result.segments:
            timestamp = f"[{format_duration(seg.start)} -> {format_duration(seg.end)}]"
            lines.append(f"{timestamp} {seg.text}")
        output_text = "\n".join(lines)
    else:
        output_text = result.text

    if output:
        output.write_text(output_text)
        print_success(f"Transcription saved to: {output}")
    else:
        console.print(output_text)

    console.print()
    console.print(
        f"[dim]Language: {result.language} "
        f"(confidence: {result.language_probability:.1%})[/dim]"
    )
    console.print(f"[dim]Duration: {format_duration(result.duration_seconds)}[/dim]")
