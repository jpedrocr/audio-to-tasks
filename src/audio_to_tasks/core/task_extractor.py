"""Task extraction service using Ollama."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

from audio_to_tasks.core.config import OllamaConfig, get_config
from audio_to_tasks.core.exceptions import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    TaskExtractionError,
)
from audio_to_tasks.core.models import (
    Task,
    TaskList,
    TaskPriority,
    TranscriptionResult,
)

if TYPE_CHECKING:
    from ollama import Client

logger = logging.getLogger(__name__)


TASK_EXTRACTION_PROMPT = """You are a task extraction assistant. Analyze the following transcription from a meeting or voice note and extract actionable tasks.

For each task, you MUST provide:
- title: A clear, concise task title (action verb + object)
- description: REQUIRED - A brief summary explaining what needs to be done and any relevant context from the transcription. Always include this field with meaningful content.
- priority: "low", "medium", "high", or "urgent"
- assignee: Person responsible (if mentioned, otherwise null)
- due_date: Deadline in ISO format (if mentioned, otherwise null)
- tags: Relevant categories

Return ONLY valid JSON in this exact format:
{{
    "tasks": [
        {{
            "title": "Task title here",
            "description": "Brief summary of the task with context from the transcription",
            "priority": "medium",
            "assignee": null,
            "due_date": null,
            "tags": ["tag1", "tag2"]
        }}
    ]
}}

TRANSCRIPTION:
---
{transcription}
---

Extract all tasks from the transcription above. Each task MUST have a description summarizing what needs to be done. If no clear tasks are found, return {{"tasks": []}}.
"""


class TaskExtractor:
    """Service for extracting tasks from transcriptions using Ollama."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        """Initialize task extractor with configuration.

        Args:
            config: Ollama configuration. If None, uses default from env.
        """
        self._config = config or get_config().ollama
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        """Lazy-load and return Ollama client."""
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self._config.host)
        return self._client

    def check_connection(self) -> bool:
        """Check if Ollama server is reachable and model is available.

        Returns:
            True if connection successful.

        Raises:
            OllamaConnectionError: If cannot connect to server.
            OllamaModelNotFoundError: If model not available.
        """
        try:
            models_response = self.client.list()
            models: list[dict[str, Any]] = models_response.get("models", [])
            model_names = [m.get("name", "") for m in models]

            model_base = self._config.model.split(":")[0]
            if not any(model_base in name for name in model_names):
                raise OllamaModelNotFoundError(self._config.model)

            return True

        except OllamaModelNotFoundError:
            raise
        except Exception as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._config.host}: {e}"
            ) from e

    def _parse_llm_response(self, response_text: str) -> list[dict[str, Any]]:
        """Parse LLM response into task dictionaries.

        Args:
            response_text: Raw response from LLM.

        Returns:
            List of task dictionaries.

        Raises:
            TaskExtractionError: If response cannot be parsed.
        """
        try:
            data = json.loads(response_text)
            return cast(list[dict[str, Any]], data.get("tasks", []))
        except json.JSONDecodeError:
            pass

        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            response_text,
            re.DOTALL,
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return cast(list[dict[str, Any]], data.get("tasks", []))
            except json.JSONDecodeError:
                pass

        json_match = re.search(r"\{.*\"tasks\".*\}", response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return cast(list[dict[str, Any]], data.get("tasks", []))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse LLM response: {response_text[:200]}")
        raise TaskExtractionError(
            "Failed to parse task extraction response",
            details={"response_preview": response_text[:500]},
        )

    def extract_tasks(
        self,
        transcription: TranscriptionResult | str,
    ) -> TaskList:
        """Extract tasks from transcription text.

        Args:
            transcription: TranscriptionResult or raw text.

        Returns:
            TaskList containing extracted tasks.

        Raises:
            TaskExtractionError: If extraction fails.
            OllamaConnectionError: If cannot connect to Ollama.
        """
        if isinstance(transcription, TranscriptionResult):
            text = transcription.text
            source_audio = transcription.audio_path
            duration = transcription.duration_seconds
            language = transcription.language
        else:
            text = transcription
            source_audio = None
            duration = None
            language = None

        if not text.strip():
            logger.info("Empty transcription, returning empty task list")
            return TaskList(
                tasks=[],
                source_audio=source_audio,
                total_duration_seconds=duration,
                language=language,
            )

        try:
            logger.info(f"Extracting tasks using model: {self._config.model}")

            prompt = TASK_EXTRACTION_PROMPT.format(transcription=text)

            response = self.client.chat(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self._config.temperature,
                    "num_predict": self._config.max_tokens,
                },
            )

            response_text = response["message"]["content"]
            task_dicts = self._parse_llm_response(response_text)

            tasks = []
            for task_dict in task_dicts:
                try:
                    if "priority" in task_dict and task_dict["priority"]:
                        priority_str = str(task_dict["priority"]).lower()
                        try:
                            task_dict["priority"] = TaskPriority(priority_str)
                        except ValueError:
                            task_dict["priority"] = TaskPriority.MEDIUM

                    task = Task(**task_dict)
                    tasks.append(task)
                except Exception as e:
                    logger.warning(f"Skipping invalid task: {task_dict} - {e}")

            logger.info(f"Extracted {len(tasks)} tasks")

            return TaskList(
                tasks=tasks,
                source_audio=source_audio,
                total_duration_seconds=duration,
                language=language,
            )

        except (OllamaConnectionError, TaskExtractionError):
            raise
        except Exception as e:
            raise TaskExtractionError(
                f"Task extraction failed: {e}",
                details={"model": self._config.model},
            ) from e

    async def extract_tasks_async(
        self,
        transcription: TranscriptionResult | str,
    ) -> TaskList:
        """Async wrapper for task extraction (runs in thread pool).

        Args:
            transcription: TranscriptionResult or raw text.

        Returns:
            TaskList containing extracted tasks.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.extract_tasks(transcription),
        )
