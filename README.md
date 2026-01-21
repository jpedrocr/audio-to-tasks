# AudioToTasks

Convert audio recordings to actionable task lists using Whisper for transcription and Ollama for task extraction.

## Features

- **Audio Transcription**: Transcribe audio files using faster-whisper (local, efficient)
- **Task Extraction**: Extract actionable tasks from transcriptions using Ollama with Gemma3:4B
- **Multiple Interfaces**:
  - REST API (FastAPI)
  - Web UI
  - CLI (Typer)
  - MCP Server (FastMCP)

## Requirements

- Python 3.12+
- FFmpeg (for audio processing)
- Ollama with Gemma3:4B model

## Installation

```bash
# Clone the repository
git clone https://github.com/jpedrocr/audio-to-tasks.git
cd audio-to-tasks

# Install the package
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

### Prerequisites

1. **Install FFmpeg**:
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Windows
   choco install ffmpeg
   ```

2. **Install and run Ollama**:
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull the Gemma3:4B model
   ollama pull gemma3:4b
   ```

## Usage

### CLI

```bash
# Check system health
audio-to-tasks health

# Transcribe an audio file
audio-to-tasks transcribe file meeting.mp3

# Transcribe with specific language
audio-to-tasks transcribe file --language en meeting.mp3

# Extract tasks from text
audio-to-tasks extract text "We need to finish the report by Friday"

# Extract tasks from file
audio-to-tasks extract text --file transcript.txt

# Process audio (transcribe + extract tasks)
audio-to-tasks process file meeting.mp3

# Process with transcript output
audio-to-tasks process file --transcript meeting.mp3

# Output as JSON
audio-to-tasks process file --json meeting.mp3

# Save results to file
audio-to-tasks process file --output results.json meeting.mp3
```

### API Server

```bash
# Start the API server
audio-to-tasks serve

# Or with custom host/port
audio-to-tasks serve --host 0.0.0.0 --port 8080

# With auto-reload for development
audio-to-tasks serve --reload
```

API endpoints:
- `GET /api/health` - Health check
- `POST /api/transcribe` - Transcribe audio file
- `POST /api/tasks/extract` - Extract tasks from text
- `POST /api/process` - Full pipeline (transcribe + extract)
- `GET /api/docs` - Swagger UI
- `GET /api/redoc` - ReDoc

### Web UI

Access the web interface at `http://localhost:8000` after starting the server.

### MCP Server

```bash
# Run the MCP server
python -m audio_to_tasks.mcp.server
```

Available tools:
- `transcribe_audio` - Transcribe audio file
- `extract_tasks` - Extract tasks from text
- `process_audio` - Full pipeline
- `check_health` - Health check

Available resources:
- `config://settings` - Current configuration
- `formats://supported` - Supported audio formats

## Configuration

Configuration can be set via environment variables or `.env` file:

```bash
# Application
DEBUG=false
LOG_LEVEL=INFO

# Whisper
WHISPER_MODEL_SIZE=base  # tiny, base, small, medium, large-v3
WHISPER_DEVICE=auto      # cpu, cuda, auto
WHISPER_VAD_FILTER=true

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TEMPERATURE=0.3

# API
API_HOST=127.0.0.1
API_PORT=8000
```

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC
- OGG
- WebM
- WMA

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov

# Run linting
ruff check .

# Run formatting
black .

# Run type checking
mypy src
```

## Project Structure

```
audio-to-tasks/
├── src/audio_to_tasks/
│   ├── core/           # Core business logic
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   ├── models.py
│   │   ├── transcriber.py
│   │   └── task_extractor.py
│   ├── api/            # REST API (FastAPI)
│   ├── cli/            # CLI (Typer)
│   ├── mcp/            # MCP Server (FastMCP)
│   └── web/            # Web UI
├── tests/
│   ├── unit/
│   └── integration/
└── pyproject.toml
```

## License

MIT
