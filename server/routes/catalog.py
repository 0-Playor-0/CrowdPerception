from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

router = APIRouter()

DATA_DIR = Path("data")
CALIBRATION_DIR = Path("calibration")
UPLOAD_DIR = Path("data/uploads")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@router.get("/api/videos")
async def api_videos():
    videos = []
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.glob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append({"filename": p.name, "size_bytes": p.stat().st_size})
    if UPLOAD_DIR.exists():
        for p in sorted(UPLOAD_DIR.glob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append({"filename": f"uploads/{p.name}", "size_bytes": p.stat().st_size})
    return {"videos": videos}


@router.post("/api/videos/upload")
async def api_videos_upload(video_file: UploadFile):
    """Saves an uploaded video under data/uploads/ and returns the filename
    (relative to DATA_DIR, e.g. "uploads/foo.mp4") that /api/frame/first,
    /api/calibration/points, and /api/session/start's video_filename field
    all already resolve correctly via DATA_DIR / filename -- calibrating a
    freshly uploaded video is just a normal video path once this returns.

    Called as soon as a file is picked in the FAB modal (not deferred to
    session-start) specifically so calibration -- which needs the file on
    disk to grab a frame from -- can run on it before a session exists."""
    if not video_file.filename:
        raise HTTPException(400, "no filename provided")

    safe_name = Path(video_file.filename).name  # strips any directory components
    ext = Path(safe_name).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(400, f"unsupported video extension {ext!r}, expected one of {sorted(VIDEO_EXTENSIONS)}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / safe_name
    if dest.exists():
        # Never silently overwrite a previous upload that might be backing
        # an existing calibration -- same reasoning as the calibration-file
        # overwrite guard in server/routes/calibration.py.
        stem = Path(safe_name).stem
        safe_name = f"{stem}_{int(time.time() * 1000)}{ext}"
        dest = UPLOAD_DIR / safe_name

    with dest.open("wb") as f:
        shutil.copyfileobj(video_file.file, f)

    return {"filename": f"uploads/{safe_name}", "size_bytes": dest.stat().st_size}


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
