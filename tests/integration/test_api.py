"""Integration tests for REST API."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, api_client: TestClient) -> None:
        """Test health endpoint returns status."""
        response = api_client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "ollama_connected" in data
        assert "whisper_loaded" in data

    def test_health_check_version(self, api_client: TestClient) -> None:
        """Test health endpoint returns correct version."""
        response = api_client.get("/api/health")

        data = response.json()
        assert data["version"] == "0.1.0"


class TestTranscribeEndpoint:
    """Tests for transcription endpoint."""

    def test_transcribe_audio_file(self, api_client: TestClient) -> None:
        """Test transcribing uploaded audio file."""
        audio_content = b"fake audio content"

        response = api_client.post(
            "/api/transcribe",
            files={"file": ("test.mp3", io.BytesIO(audio_content), "audio/mpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "text" in data["data"]
        assert "language" in data["data"]

    def test_transcribe_with_language(self, api_client: TestClient) -> None:
        """Test transcription with language parameter."""
        audio_content = b"fake audio content"

        response = api_client.post(
            "/api/transcribe?language=en",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_transcribe_unsupported_format(self, api_client: TestClient) -> None:
        """Test rejection of unsupported format."""
        response = api_client.post(
            "/api/transcribe",
            files={
                "file": ("test.xyz", io.BytesIO(b"data"), "application/octet-stream")
            },
        )

        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_transcribe_no_file(self, api_client: TestClient) -> None:
        """Test error when no file provided."""
        response = api_client.post("/api/transcribe")

        assert response.status_code == 422


class TestTasksEndpoint:
    """Tests for task extraction endpoint."""

    def test_extract_tasks_from_text(self, api_client: TestClient) -> None:
        """Test task extraction from text."""
        response = api_client.post(
            "/api/tasks/extract",
            json={"text": "We need to finish the report by Friday."},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "tasks" in data["data"]

    def test_extract_tasks_empty_text(self, api_client: TestClient) -> None:
        """Test error with empty text."""
        response = api_client.post(
            "/api/tasks/extract",
            json={"text": ""},
        )

        assert response.status_code == 422

    def test_extract_tasks_missing_text(self, api_client: TestClient) -> None:
        """Test error with missing text field."""
        response = api_client.post(
            "/api/tasks/extract",
            json={},
        )

        assert response.status_code == 422


class TestProcessEndpoint:
    """Tests for full processing endpoint."""

    def test_process_audio(self, api_client: TestClient) -> None:
        """Test full audio processing pipeline."""
        audio_content = b"fake audio content"

        response = api_client.post(
            "/api/process",
            files={"file": ("meeting.wav", io.BytesIO(audio_content), "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "transcription" in data
        assert "tasks" in data
        assert "processing_time_seconds" in data

    def test_process_audio_with_language(self, api_client: TestClient) -> None:
        """Test processing with language parameter."""
        audio_content = b"fake audio content"

        response = api_client.post(
            "/api/process?language=es",
            files={"file": ("meeting.mp3", io.BytesIO(audio_content), "audio/mpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_process_transcription_content(self, api_client: TestClient) -> None:
        """Test that transcription contains expected fields."""
        audio_content = b"fake audio content"

        response = api_client.post(
            "/api/process",
            files={"file": ("test.mp3", io.BytesIO(audio_content), "audio/mpeg")},
        )

        data = response.json()
        transcription = data["transcription"]

        assert "text" in transcription
        assert "language" in transcription
        assert "language_probability" in transcription
        assert "duration_seconds" in transcription
        assert "segments" in transcription

    def test_process_tasks_content(self, api_client: TestClient) -> None:
        """Test that tasks contains expected fields."""
        audio_content = b"fake audio content"

        response = api_client.post(
            "/api/process",
            files={"file": ("test.mp3", io.BytesIO(audio_content), "audio/mpeg")},
        )

        data = response.json()
        tasks = data["tasks"]

        assert "tasks" in tasks
        assert "created_at" in tasks


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_swagger_docs_available(self, api_client: TestClient) -> None:
        """Test Swagger UI is available."""
        response = api_client.get("/api/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_available(self, api_client: TestClient) -> None:
        """Test ReDoc is available."""
        response = api_client.get("/api/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_schema(self, api_client: TestClient) -> None:
        """Test OpenAPI schema is available."""
        response = api_client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/api/health" in data["paths"]
        assert "/api/transcribe" in data["paths"]
        assert "/api/process" in data["paths"]
