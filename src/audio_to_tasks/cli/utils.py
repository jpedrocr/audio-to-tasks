"""CLI utility functions."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from audio_to_tasks.core.models import TaskList, TaskPriority

console = Console()
error_console = Console(stderr=True)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS or HH:MM:SS."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_success(message: str) -> None:
    """Print success message in green."""
    console.print(f"[bold green][SUCCESS][/bold green] {message}")


def print_error(message: str) -> None:
    """Print error message in red."""
    error_console.print(f"[bold red][ERROR][/bold red] {message}")


def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    console.print(f"[bold yellow][WARNING][/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[bold blue][INFO][/bold blue] {message}")


PRIORITY_COLORS = {
    TaskPriority.LOW: "blue",
    TaskPriority.MEDIUM: "white",
    TaskPriority.HIGH: "yellow",
    TaskPriority.URGENT: "red",
}


def print_task_list(task_list: TaskList) -> None:
    """Pretty-print a task list to the console."""
    console.print()
    console.rule(f"[bold]EXTRACTED TASKS ({task_list.task_count} total)[/bold]")
    console.print()

    if not task_list.tasks:
        console.print("  [dim]No tasks found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", min_width=30)
    table.add_column("Priority", width=10)
    table.add_column("Assignee", width=15)
    table.add_column("Tags", width=20)

    for i, task in enumerate(task_list.tasks, 1):
        color = PRIORITY_COLORS.get(task.priority, "white")
        priority_str = f"[{color}]{task.priority.value.upper()}[/{color}]"
        assignee = task.assignee or "-"
        tags = ", ".join(task.tags) if task.tags else "-"

        table.add_row(
            str(i),
            task.title,
            priority_str,
            assignee,
            tags,
        )

    console.print(table)
    console.print()
