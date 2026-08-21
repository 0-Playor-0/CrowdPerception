from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from scripts.calibrate_video import (
    DEFAULT_ESTIMATED_HEIGHT_M,
    DEFAULT_ESTIMATED_WIDTH_M,
    ESTIMATED_NOTE,
    QuadValidationError,
    build_calibration_record,
    grab_frame,
)
from server.schemas import CalibrationPointsRequest

router = APIRouter()

DATA_DIR = Path("data")
CALIBRATION_DIR = Path("calibration")


@router.get("/api/frame/first")
async def api_frame_first(video: str = Query(...), frame_index: int = Query(default=0)):
    video_path = DATA_DIR / video
    if not video_path.exists():
        raise HTTPException(404, f"video not found: {video_path}")
    try:
        frame = grab_frame(str(video_path), frame_index)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))

    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise HTTPException(500, "failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/png")


def _resolve_output_path(video_stem: str, requested_filename: str | None) -> Path:
    """Turns the client's requested output filename (or the CLI tool's own
    <video_stem>.json default, if none was given) into a path confined to
    CALIBRATION_DIR -- strips any directory components from a client-
    supplied name so this can never be tricked into writing outside
    calibration/ (e.g. "../../etc/passwd.json")."""
    raw_name = requested_filename if requested_filename else f"{video_stem}.json"
    safe_name = Path(raw_name).name  # drops any leading path components
    if not safe_name.endswith(".json"):
        raise HTTPException(400, f"output_filename must end with .json, got {raw_name!r}")
    if not safe_name or safe_name in (".json",):
        raise HTTPException(400, f"invalid output_filename: {raw_name!r}")
    return CALIBRATION_DIR / safe_name


@router.post("/api/calibration/points")
async def api_calibration_points(body: CalibrationPointsRequest):
    video_path = DATA_DIR / body.video
    if not video_path.exists():
        raise HTTPException(404, f"video not found: {video_path}")

    out_path = _resolve_output_path(video_path.stem, body.output_filename)

    # Calibration files are load-bearing: every world-coordinate and
    # density figure in the project's docs traces back to one of these.
    # Refuse to clobber an existing file unless the caller explicitly opts
    # in -- silently overwriting one (as happened once during dashboard
    # testing, recovered only because git still had the original) would
    # invalidate published numbers with no warning.
    if out_path.exists() and not body.overwrite:
        raise HTTPException(
            409,
            f"{out_path.name} already exists in calibration/ -- refusing to overwrite. "
            "Pass overwrite=true to replace it, or choose a different output filename.",
        )

    if body.world_width_m is None and body.world_height_m is None:
        width_m, height_m = DEFAULT_ESTIMATED_WIDTH_M, DEFAULT_ESTIMATED_HEIGHT_M
        source, note = "ESTIMATED", ESTIMATED_NOTE
    elif body.world_width_m is not None and body.world_height_m is not None:
        width_m, height_m = body.world_width_m, body.world_height_m
        source, note = "USER_MEASURED", "User-entered measurement via the browser calibration clicker."
    else:
        raise HTTPException(400, "provide both world_width_m and world_height_m, or neither (for ESTIMATED)")

    image_points = np.array(body.points, dtype=np.float64)

    try:
        frame = grab_frame(str(video_path), body.frame_index)
        height_px, width_px = frame.shape[:2]
        record = build_calibration_record(
            image_points=image_points,
            width_m=width_m,
            height_m=height_m,
            source=source,
            note=note,
            video_path=str(video_path),
            frame_index=body.frame_index,
            video_resolution=(width_px, height_px),
        )
    except QuadValidationError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(record, f, indent=2)

    return {"written": out_path.name, "source": source, "record": record}
