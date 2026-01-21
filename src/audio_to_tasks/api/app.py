"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from audio_to_tasks.api.routes import health, tasks, transcribe
from audio_to_tasks.core.config import get_config

WEB_DIR = Path(__file__).parent.parent / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    config = get_config()
    if not config.debug:
        pass
    yield


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = get_config()

    app = FastAPI(
        title=config.app_name,
        description="Convert audio recordings to actionable task lists",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.debug else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(transcribe.router, prefix="/api", tags=["Transcription"])
    app.include_router(tasks.router, prefix="/api", tags=["Tasks"])

    # Static files and templates for web UI
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    if TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=TEMPLATES_DIR)

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index(request: Request) -> HTMLResponse:
            """Render main page."""
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "title": "AudioToTasks"},
            )

    return app


app = create_app()
