"""FastAPI entry point for the CrowdPerception operator dashboard.

Run:
    uv run uvicorn server.main:app --reload --port 8000

Then open http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.routes import calibration, catalog, session, stream, telemetry_ws

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"

app = FastAPI(title="CrowdPerception Operator Dashboard")

app.include_router(stream.router)
app.include_router(telemetry_ws.router)
app.include_router(session.router)
app.include_router(catalog.router)
app.include_router(calibration.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
