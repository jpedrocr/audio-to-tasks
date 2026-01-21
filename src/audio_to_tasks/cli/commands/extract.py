"""Extract tasks CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from audio_to_tasks.cli.utils import (
    console,
    print_error,
    print_success,
    print_task_list,
)

app = typer.Typer(help="Extract tasks from text.")


@app.command("text")
def extract_from_text(
    text: Annotated[
        str | None,
        typer.Argument(help="Text to extract tasks from (or use --file)"),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="File containing text to process"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
) -> None:
    """Extract tasks from text or file.

    Examples:
        audio-to-tasks extract text "We need to finish the report by Friday"
        audio-to-tasks extract text --file transcript.txt
        audio-to-tasks extract text --file transcript.txt --json --output tasks.json
    """
    from audio_to_tasks.core.exceptions import TaskExtractionError
    from audio_to_tasks.core.task_extractor import TaskExtractor

    if file:
        if not file.exists():
            print_error(f"File not found: {file}")
            raise typer.Exit(1)
        input_text = file.read_text()
    elif text:
        input_text = text
    else:
        print_error("Provide text argument or --file option")
        raise typer.Exit(1)

    extractor = TaskExtractor()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        try:
            progress.add_task(description="Extracting tasks...", total=None)
            task_list = extractor.extract_tasks(input_text)
        except TaskExtractionError as e:
            print_error(f"Extraction failed: {e.message}")
            raise typer.Exit(1) from None

    if json_output:
        output_text = task_list.model_dump_json(indent=2)
        if output:
            output.write_text(output_text)
            print_success(f"Tasks saved to: {output}")
        else:
            console.print(output_text)
    else:
        print_task_list(task_list)
        if output:
            output.write_text(task_list.model_dump_json(indent=2))
            print_success(f"Tasks saved to: {output}")
