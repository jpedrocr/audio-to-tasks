"""Configuration management for AudioToTasks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WhisperConfig(BaseSettings):
    """Whisper transcription configuration."""

    model_config = SettingsConfigDict(env_prefix="WHISPER_")

    model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "base"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    compute_type: Literal["float16", "float32", "int8", "int8_float16"] = "float32"
    beam_size: int = Field(default=5, ge=1, le=10)
    vad_filter: bool = True
    language: str | None = None


class OllamaConfig(BaseSettings):
    """Ollama LLM configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    host: str = "http://localhost:11434"
    model: str = "gemma3:4b"
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=100, le=8192)
    timeout: float = Field(default=120.0, ge=10)


class APIConfig(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)
    reload: bool = False
    workers: int = Field(default=1, ge=1, le=8)
    max_upload_size_mb: int = Field(default=100, ge=1, le=500)

    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_upload_size_mb * 1024 * 1024


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AudioToTasks"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    temp_dir: Path = Field(default=Path("/tmp/audio_to_tasks"))

    @field_validator("temp_dir", mode="after")
    @classmethod
    def ensure_temp_dir(cls, v: Path) -> Path:
        """Ensure temp directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache
def get_config() -> AppConfig:
    """Get cached application configuration."""
    return AppConfig()
