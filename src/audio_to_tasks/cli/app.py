"""AudioToTasks CLI application."""

from __future__ import annotations

import typer
from rich.console import Console

from audio_to_tasks.cli.commands import extract, process, transcribe

console = Console()

app = typer.Typer(
    name="audio-to-tasks",
    help="Convert audio recordings to actionable task lists.",
    no_args_is_help=True,
)

app.add_typer(transcribe.app, name="transcribe")
app.add_typer(extract.app, name="extract")
app.add_typer(process.app, name="process")


@app.command()
def version() -> None:
    """Show version information."""
    from audio_to_tasks import __version__

    console.print(f"audio-to-tasks version {__version__}")


@app.command()
def health() -> None:
    """Check system health (Ollama connection, etc.)."""
    from audio_to_tasks.core.task_extractor import TaskExtractor

    console.print("Checking system health...")

    extractor = TaskExtractor()
    try:
        extractor.check_connection()
        console.print("[bold green]Ollama: Connected[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Ollama: Failed - {e}[/bold red]")
        raise typer.Exit(1) from None

    console.print("[bold green]All systems operational[/bold green]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the API server."""
    import uvicorn

    console.print(f"Starting API server at http://{host}:{port}")
    console.print("API docs available at /api/docs")

    uvicorn.run(
        "audio_to_tasks.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
