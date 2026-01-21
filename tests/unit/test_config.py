"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_to_tasks.core.config import (
    APIConfig,
    AppConfig,
    OllamaConfig,
    WhisperConfig,
    get_config,
)


class TestWhisperConfig:
    """Tests for WhisperConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = WhisperConfig()

        assert config.model_size == "base"
        assert config.device == "auto"
        assert config.compute_type == "float32"
        assert config.beam_size == 5
        assert config.vad_filter is True
        assert config.language is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = WhisperConfig(
            model_size="large-v3",
            device="cuda",
            compute_type="float16",
            beam_size=3,
            vad_filter=False,
            language="en",
        )

        assert config.model_size == "large-v3"
        assert config.device == "cuda"
        assert config.compute_type == "float16"
        assert config.beam_size == 3
        assert config.vad_filter is False
        assert config.language == "en"

    def test_beam_size_validation(self) -> None:
        """Test beam size validation."""
        with pytest.raises(ValueError):
            WhisperConfig(beam_size=0)
        with pytest.raises(ValueError):
            WhisperConfig(beam_size=11)


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = OllamaConfig()

        assert config.host == "http://localhost:11434"
        assert config.model == "gemma3:4b"
        assert config.temperature == 0.3
        assert config.max_tokens == 2048
        assert config.timeout == 120.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = OllamaConfig(
            host="http://custom:11434",
            model="llama2:7b",
            temperature=0.7,
            max_tokens=4096,
            timeout=60.0,
        )

        assert config.host == "http://custom:11434"
        assert config.model == "llama2:7b"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 60.0

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValueError):
            OllamaConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            OllamaConfig(temperature=2.1)


class TestAPIConfig:
    """Tests for APIConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = APIConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.reload is False
        assert config.workers == 1
        assert config.max_upload_size_mb == 100

    def test_max_upload_size_bytes(self) -> None:
        """Test max upload size conversion."""
        config = APIConfig(max_upload_size_mb=50)
        assert config.max_upload_size_bytes == 50 * 1024 * 1024

    def test_port_validation(self) -> None:
        """Test port validation."""
        with pytest.raises(ValueError):
            APIConfig(port=80)
        with pytest.raises(ValueError):
            APIConfig(port=70000)


class TestAppConfig:
    """Tests for AppConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AppConfig()

        assert config.app_name == "AudioToTasks"
        assert config.debug is False
        assert config.log_level == "INFO"
        assert isinstance(config.whisper, WhisperConfig)
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.api, APIConfig)

    def test_temp_dir_creation(self, temp_dir: Path) -> None:
        """Test temp directory is created if it doesn't exist."""
        new_temp = temp_dir / "new_temp_dir"
        config = AppConfig(temp_dir=new_temp)

        assert config.temp_dir.exists()
        assert config.temp_dir.is_dir()

    def test_nested_config(self) -> None:
        """Test nested configuration objects."""
        config = AppConfig(
            whisper=WhisperConfig(model_size="tiny"),
            ollama=OllamaConfig(model="custom:model"),
        )

        assert config.whisper.model_size == "tiny"
        assert config.ollama.model == "custom:model"


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_cached(self) -> None:
        """Test that get_config returns cached instance."""
        get_config.cache_clear()
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_config_returns_app_config(self) -> None:
        """Test that get_config returns AppConfig instance."""
        get_config.cache_clear()
        config = get_config()

        assert isinstance(config, AppConfig)
