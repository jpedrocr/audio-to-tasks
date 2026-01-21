"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache

from audio_to_tasks.core.task_extractor import TaskExtractor
from audio_to_tasks.core.transcriber import Transcriber


@lru_cache
def get_transcriber() -> Transcriber:
    """Get cached transcriber instance."""
    return Transcriber()


@lru_cache
def get_task_extractor() -> TaskExtractor:
    """Get cached task extractor instance."""
    return TaskExtractor()
