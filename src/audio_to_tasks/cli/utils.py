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

    for i, task in enumerate(task_list.tasks, 1):
        color = PRIORITY_COLORS.get(task.priority, "white")
        priority_str = f"[{color}]{task.priority.value.upper()}[/{color}]"
        assignee = f" | Assignee: {task.assignee}" if task.assignee else ""
        tags = f" | Tags: {', '.join(task.tags)}" if task.tags else ""

        console.print(f"[bold]{i}. {task.title}[/bold]")
        console.print(f"   Priority: {priority_str}{assignee}{tags}")
        if task.description:
            console.print(f"   [dim]{task.description}[/dim]")
        console.print()
