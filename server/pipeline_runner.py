"""Background-thread wrapper around the existing perception pipeline.

This module does NOT reimplement detection, tracking, projection, or
heatmap rendering. It imports scripts.live_perception.LivePerceptionRunner
(the reference implementation) unmodified, calls process_frame() once per
frame -- the single pipeline pass -- and calls the SAME build_composite()
the CLI tool uses to get the camera+heatmap composite image. The only new
code here is: running that loop in a background thread instead of a
cv2.imshow loop, JPEG-encoding the composite, and splitting it into two
images.

That split reads runner.last_camera_pane_width_px -- metadata
build_composite() itself sets to the pane width it actually drew (see
scripts/live_perception.py), not a value recomputed here from
DISPLAY_MAX_HEIGHT_PX/frame shape. An earlier version of this file did
recompute it independently; if build_composite's internal layout ever
changed, that recomputation would silently drift out of sync and produce
a wrong crop with no error. _split_composite() below also sanity-checks
the boundary against the composite's actual width and raises rather than
streaming a bad crop if it's ever out of range.
"""

from __future__ import annotations

import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.live_perception import LivePerceptionRunner, load_calibration
from server.state import SessionStatus, session

JPEG_QUALITY = 80
DENSITY_ALARM_ONSET_PERSONS_PER_M2 = 2.5

# Confidence-threshold presets by camera range, picked by the operator in
# the FAB modal (not auto-detected -- the pipeline has no way to know how
# far the camera is from its subjects). Long-range/elevated crowd shots
# (e.g. the Myeongdong demo clip) render people as small, blurry boxes, so
# detection needs to run sensitive to catch them at all -- and in a dense
# crowd a stray false positive is visually lost among real detections
# anyway. Short-range shots have few, large, clearly-resolved subjects,
# where the SAME sensitivity produces false positives that stand out
# individually, so a stricter threshold reads as more correct there.
# These three numbers are the operator's own picks, not independently
# re-measured per range the way 0.15 was for the Myeongdong clip
# specifically (scripts/diagnose_tiled_nms.py) -- treat them as reasonable
# starting points per range, not validated optima.
RANGE_PRESETS: dict[str, float] = {
    "long": 0.05,
    "mid": 0.10,
    "short": 0.30,
}
DEFAULT_RANGE_PRESET = "mid"


def _split_composite(runner: LivePerceptionRunner, composite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits build_composite()'s [camera_pane | heatmap_panel] composite
    back into its two panes, using the boundary build_composite() itself
    reported rather than recomputing it -- see module docstring."""
    boundary = runner.last_camera_pane_width_px
    composite_width = composite.shape[1]
    if boundary is None or not (0 < boundary < composite_width):
        raise RuntimeError(
            f"composite split boundary is invalid (last_camera_pane_width_px={boundary!r}, "
            f"composite width={composite_width}) -- refusing to stream a possibly-wrong crop. "
            "This means build_composite()'s layout changed in a way this split can't follow; "
            "fix the split logic before re-enabling streaming."
        )
    return composite[:, :boundary], composite[:, boundary:]


def _build_args(
    video_path: Path,
    confidence_threshold: float,
    tile: bool,
    downscale: bool,
    outputs_dir: Path,
    trajectory_dir: Path,
) -> argparse.Namespace:
    """Mirrors scripts/live_perception.py's parse_args() defaults -- only
    the fields the FAB modal actually exposes (range preset -> confidence
    threshold, tile, downscale) vary."""
    return argparse.Namespace(
        video=str(video_path),
        model="models/yolo11s.pt",
        device="auto",
        confidence_threshold=confidence_threshold,
        iou_threshold=0.7,
        tile=tile,
        tile_size=[1280, 1280],
        tile_overlap=0.1,
        tile_conf_threshold=None,
        nms_iou_threshold=0.7,
        imgsz=None,
        downscale_long_edge=1920 if downscale else 0,
        heatmap_mode="density",
        cell_size=0.5,
        heatmap_window_seconds=2.0,
        heatmap_sigma=1.0,
        heatmap_color_variant="hard",
        heatmap_vmax=3.0,
        frame_heatmap=True,
        frame_heatmap_cell_size_px=60,
        frame_heatmap_vmax=0.5,
        outputs_dir=str(outputs_dir),
        trajectory_dir=str(trajectory_dir),
    )


def _build_telemetry(runner: LivePerceptionRunner, hud: dict, calibration: dict, video_path: Path,
                      range_preset: str, confidence_threshold: float) -> dict[str, Any]:
    grid = runner.heatmap.grid("density")
    peak = float(grid.max()) if grid.size else 0.0
    mean = float(grid.mean()) if grid.size else 0.0
    cells_above = int(np.sum(grid > DENSITY_ALARM_ONSET_PERSONS_PER_M2)) if grid.size else 0

    return {
        "frame_idx": hud["frame_idx"],
        "total_frames": hud["total_frames"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": {"detected": hud["detections"], "in_quad": hud["in_quad"], "tracked": hud["tracked"]},
        "density": {
            "peak_persons_per_m2": peak,
            "mean_persons_per_m2": mean,
            "cells_above_2_5": cells_above,
        },
        "performance": {"fps": hud["fps"], "detect_ms": hud["detect_ms"], "track_ms": hud["track_ms"]},
        "calibration": {
            "source": calibration["source"],
            "note": calibration.get("note", ""),
            "world_width_m": calibration.get("world_width_m"),
            "world_height_m": calibration.get("world_height_m"),
        },
        "detection": {
            "range_preset": range_preset,
            "confidence_threshold": confidence_threshold,
        },
        "status": SessionStatus.RUNNING.value,
        "video": Path(video_path).name,
        "total_ids_seen": hud["total_ids"],
        "error": None,
    }


def run_pipeline(
    video_path: Path,
    calibration_path: Path,
    range_preset: str,
    confidence_threshold: float,
    tile: bool,
    downscale: bool,
    outputs_dir: Path,
    trajectory_dir: Path,
) -> None:
    session.set_status(SessionStatus.CALIBRATING)

    try:
        calibration = load_calibration(calibration_path)
    except SystemExit as exc:
        session.set_status(SessionStatus.ERROR, str(exc))
        return

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        session.set_status(SessionStatus.ERROR, f"could not open video: {video_path}")
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args = _build_args(video_path, confidence_threshold, tile, downscale, outputs_dir, trajectory_dir)

    try:
        runner = LivePerceptionRunner(args, calibration, fps, total_frames, frame_width, frame_height)
    except Exception as exc:
        capture.release()
        session.set_status(SessionStatus.ERROR, f"pipeline init failed: {exc}")
        return

    session.set_status(SessionStatus.RUNNING)

    frame_idx = 0
    frames_processed = 0
    try:
        while not session.stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                break

            render_data, hud = runner.process_frame(frame, frame_idx)
            composite = runner.build_composite(render_data, hud)
            camera_img, heatmap_img = _split_composite(runner, composite)

            ok_c, camera_buf = cv2.imencode(".jpg", camera_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            ok_h, heatmap_buf = cv2.imencode(".jpg", heatmap_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            if ok_c and ok_h:
                telemetry = _build_telemetry(runner, hud, calibration, video_path,
                                              range_preset, confidence_threshold)
                session.publish_frame(camera_buf.tobytes(), heatmap_buf.tobytes(), telemetry)

            frame_idx += 1
            frames_processed += 1
    except Exception as exc:
        session.set_status(SessionStatus.ERROR, f"pipeline error at frame {frame_idx}: {exc}")
        capture.release()
        runner.close(frames_processed)
        return

    capture.release()
    runner.close(frames_processed)
    if session.status != SessionStatus.ERROR:
        session.reset_to_idle()


def start_session(
    video_path: Path,
    calibration_path: Path,
    range_preset: str,
    tile: bool,
    downscale: bool,
    outputs_dir: Path = Path("outputs"),
    trajectory_dir: Path = Path("outputs/trajectories"),
) -> None:
    confidence_threshold = RANGE_PRESETS[range_preset]
    stop_session(timeout=5.0)
    session.stop_event.clear()
    thread = threading.Thread(
        target=run_pipeline,
        args=(video_path, calibration_path, range_preset, confidence_threshold,
              tile, downscale, outputs_dir, trajectory_dir),
        daemon=True,
    )
    session.thread = thread
    thread.start()


def stop_session(timeout: float = 5.0) -> None:
    if session.thread is not None and session.thread.is_alive():
        session.stop_event.set()
        session.thread.join(timeout=timeout)
    session.thread = None
    session.reset_to_idle()
