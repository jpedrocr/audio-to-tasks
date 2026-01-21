"""Web UI for AudioToTasks."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

web_app = FastAPI(title="AudioToTasks Web")

web_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@web_app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render main page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "AudioToTasks"},
    )
