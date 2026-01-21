"""Process (transcribe + extract) CLI command."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from audio_to_tasks.cli.utils import (
    console,
    format_duration,
    print_error,
    print_success,
    print_task_list,
)

app = typer.Typer(help="Process audio files (transcribe + extract tasks).")


@app.command("file")
def process_file(
    audio_path: Annotated[
        Path,
        typer.Argument(
            help="Path to audio file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON file path"),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Language code"),
    ] = None,
    model_size: Annotated[
        str,
        typer.Option("--model", "-m", help="Whisper model size"),
    ] = "base",
    show_transcript: Annotated[
        bool,
        typer.Option("--transcript", "-t", help="Show full transcript"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
) -> None:
    """Process audio file: transcribe and extract tasks.

    Examples:
        audio-to-tasks process file meeting.mp3
        audio-to-tasks process file --transcript --output results.json audio.wav
    """
    from audio_to_tasks.core.config import WhisperConfig
    from audio_to_tasks.core.exceptions import TaskExtractionError, TranscriptionError
    from audio_to_tasks.core.models import ProcessingResult
    from audio_to_tasks.core.task_extractor import TaskExtractor
    from audio_to_tasks.core.transcriber import Transcriber

    start_time = time.time()

    whisper_config = WhisperConfig(model_size=model_size)  # type: ignore[arg-type]
    transcriber = Transcriber(whisper_config)
    extractor = TaskExtractor()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            description="Step 1/2: Transcribing audio...", total=None
        )
        try:
            transcription = transcriber.transcribe(audio_path, language=language)
        except TranscriptionError as e:
            print_error(f"Transcription failed: {e.message}")
            raise typer.Exit(1) from None

        console.print(
            f"  [dim]Detected language: {transcription.language} "
            f"({transcription.language_probability:.1%})[/dim]"
        )
        console.print(
            f"  [dim]Duration: {format_duration(transcription.duration_seconds)}[/dim]"
        )

        progress.update(task, description="Step 2/2: Extracting tasks...")
        try:
            task_list = extractor.extract_tasks(transcription)
        except TaskExtractionError as e:
            print_error(f"Task extraction failed: {e.message}")
            raise typer.Exit(1) from None

    processing_time = time.time() - start_time

    result = ProcessingResult(
        transcription=transcription,
        task_list=task_list,
        processing_time_seconds=processing_time,
    )

    if json_output:
        output_text = result.model_dump_json(indent=2)
        if output:
            output.write_text(output_text)
            print_success(f"Results saved to: {output}")
        else:
            console.print(output_text)
    else:
        if show_transcript:
            console.print()
            console.rule("[bold]Transcript[/bold]")
            console.print(transcription.text)
            console.rule()

        print_task_list(task_list)

        if output:
            output.write_text(result.model_dump_json(indent=2))
            print_success(f"Results saved to: {output}")

    console.print(f"[dim]Total processing time: {processing_time:.1f}s[/dim]")
