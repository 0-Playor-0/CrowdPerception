from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from server.pipeline_runner import DEFAULT_RANGE_PRESET, RANGE_PRESETS, start_session, stop_session
from server.state import session

router = APIRouter()

DATA_DIR = Path("data")
CALIBRATION_DIR = Path("calibration")
UPLOAD_DIR = Path("data/uploads")

TILING_DOWNSCALE_WARNING = (
    "downscale is on with tiling on -- measured 69% recall loss on dense frames "
    "with this combination; consider --no-downscale or --no-tile instead"
)


@router.post("/api/session/start")
async def api_session_start(
    video_filename: str | None = Form(default=None),
    calibration_filename: str | None = Form(default=None),
    range_preset: str = Form(default=DEFAULT_RANGE_PRESET),
    tile: bool = Form(default=True),
    downscale: bool = Form(default=False),
    video_file: UploadFile | None = None,
):
    if range_preset not in RANGE_PRESETS:
        raise HTTPException(
            400, f"range_preset must be one of {sorted(RANGE_PRESETS)}, got {range_preset!r}"
        )

    if video_file is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / video_file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(video_file.file, f)
        video_path = dest
    elif video_filename:
        video_path = DATA_DIR / video_filename
    else:
        raise HTTPException(400, "provide either video_filename or video_file")

    if not video_path.exists():
        raise HTTPException(404, f"video not found: {video_path}")

    if not calibration_filename:
        raise HTTPException(400, "calibration_filename is required (select existing or calibrate first)")
    calibration_path = CALIBRATION_DIR / calibration_filename
    if not calibration_path.exists():
        raise HTTPException(404, f"calibration not found: {calibration_path}")

    warning = TILING_DOWNSCALE_WARNING if (tile and downscale) else None

    await run_in_threadpool(start_session, video_path, calibration_path, range_preset, tile, downscale)

    return {
        "started": True,
        "video": video_path.name,
        "calibration": calibration_path.name,
        "range_preset": range_preset,
        "confidence_threshold": RANGE_PRESETS[range_preset],
        "tile": tile,
        "downscale": downscale,
        "warning": warning,
    }


@router.post("/api/session/stop")
async def api_session_stop():
    await run_in_threadpool(stop_session)
    return {"stopped": True}


@router.get("/api/session/status")
async def api_session_status():
    return {
        "status": session.status.value,
        "error": session.error_message,
        "telemetry": session.snapshot_telemetry(),
    }
