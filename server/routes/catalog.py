from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

DATA_DIR = Path("data")
CALIBRATION_DIR = Path("calibration")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@router.get("/api/videos")
async def api_videos():
    if not DATA_DIR.exists():
        return {"videos": []}
    videos = []
    for p in sorted(DATA_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append({"filename": p.name, "size_bytes": p.stat().st_size})
    return {"videos": videos}


@router.get("/api/calibrations")
async def api_calibrations():
    if not CALIBRATION_DIR.exists():
        return {"calibrations": []}
    calibrations = []
    for p in sorted(CALIBRATION_DIR.glob("*.json")):
        try:
            with p.open("r") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        calibrations.append({
            "filename": p.name,
            "video": record.get("video"),
            "source": record.get("source"),
            "world_width_m": record.get("world_width_m"),
            "world_height_m": record.get("world_height_m"),
        })
    return {"calibrations": calibrations}
