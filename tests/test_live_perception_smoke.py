"""Headless end-to-end smoke test for scripts/live_perception.py: runs the
real pipeline (real detector, real tracker, real ground projection) against
100 frames of the real demo video with --no-window, and checks it exits
cleanly and produces every output artifact it promises (trajectory CSV,
track-stats sidecar, a snapshot PNG). Slow (real YOLO inference on real
frames) -- skipped automatically if the video/model/calibration aren't
present in this checkout."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from scripts.live_perception import LivePerceptionRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = REPO_ROOT / "data" / "127690-739144743.mp4"
MODEL_PATH = REPO_ROOT / "models" / "yolo11s.pt"
CALIBRATION_PATH = REPO_ROOT / "calibration" / "127690-739144743.json"

pytestmark = pytest.mark.skipif(
    not (VIDEO_PATH.exists() and MODEL_PATH.exists() and CALIBRATION_PATH.exists()),
    reason="needs the real demo video, model, and calibration file",
)


def test_headless_100_frames(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    trajectory_dir = tmp_path / "outputs" / "trajectories"
    snapshot_frame = 50

    result = subprocess.run(
        [
            sys.executable, "scripts/live_perception.py",
            "--video", str(VIDEO_PATH),
            "--calibration", str(CALIBRATION_PATH),
            "--no-window",
            "--max-frames", "100",
            "--snapshot-frames", str(snapshot_frame),
            "--outputs-dir", str(outputs_dir),
            "--trajectory-dir", str(trajectory_dir),
            "--no-tile",   # untiled: keeps this smoke test fast, tiling is exercised elsewhere
            "--downscale", "1920",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"

    video_stem = VIDEO_PATH.stem
    trajectory_csv = trajectory_dir / f"{video_stem}_trajectory.csv"
    stats_json = outputs_dir / f"{video_stem}_track_stats.json"
    snapshot_png = outputs_dir / "snapshots" / f"{video_stem}_frame{snapshot_frame}.png"

    assert trajectory_csv.exists(), result.stdout
    assert stats_json.exists(), result.stdout
    assert snapshot_png.exists(), result.stdout

    with trajectory_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    # Not every frame need have a confirmed track, but over 100 real frames
    # of a crowd video we expect at least some rows.
    assert len(rows) > 0
    frame_indices = {int(row["frame_idx"]) for row in rows}
    assert max(frame_indices) < 100

    with stats_json.open("r") as f:
        stats = json.load(f)
    assert stats["total_frames_processed"] == 100
    assert stats["n_rows_written"] == len(rows)


def test_funnel_in_quad_count_matches_heatmap_fed_count(tmp_path: Path) -> None:
    """Permanent regression test for the single-source-of-truth in-quad
    fix (perception/ground.py's GroundProjector, wired up in
    LivePerceptionRunner.process_frame): the number of detections counted
    as 'in quad' by the funnel must exactly equal the number of points
    handed to the metric heatmap's update(), every frame. Before that fix,
    a second, independent in-quad test (sv.PolygonZone, on a rounded-to-
    integer-pixels copy of the calibrated quad) filtered the tracker's
    input while GroundProjector's own full-float test fed the heatmap, and
    the two silently disagreed at the polygon boundary -- this is exactly
    the check that would have caught that bug, promoted here from a
    one-off manual verification run to a permanent test so the bug class
    can't silently come back."""
    args = argparse.Namespace(
        video=str(VIDEO_PATH), calibration=str(CALIBRATION_PATH),
        model=str(MODEL_PATH), device="auto",
        confidence_threshold=0.02, iou_threshold=0.7,
        tile=False, tile_size=[1280, 1280], tile_overlap=0.1,
        tile_conf_threshold=None, nms_iou_threshold=0.7, imgsz=None,
        downscale_long_edge=1920,
        heatmap_mode="count", cell_size=0.5, heatmap_window_seconds=2.0,
        heatmap_sigma=1.0, heatmap_color_variant="hard", heatmap_vmax=3.0,
        frame_heatmap=False, frame_heatmap_cell_size_px=60, frame_heatmap_vmax=0.5,
        outputs_dir=str(tmp_path / "outputs"),
        trajectory_dir=str(tmp_path / "outputs" / "trajectories"),
    )
    with CALIBRATION_PATH.open("r") as f:
        calibration = json.load(f)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = 30
    runner = LivePerceptionRunner(args, calibration, fps, total_frames=n_frames,
                                   frame_width=frame_width, frame_height=frame_height)

    fed_counts: list[int] = []
    original_update = runner.heatmap.update

    def spy_update(t_sec: float, world_xy_in_quad) -> None:
        fed_counts.append(len(world_xy_in_quad))
        original_update(t_sec, world_xy_in_quad)

    runner.heatmap.update = spy_update

    frames_read = 0
    for frame_idx in range(n_frames):
        ok, frame = capture.read()
        if not ok:
            break
        runner.process_frame(frame, frame_idx)
        frames_read += 1
    capture.release()
    runner.close(frames_read)

    assert frames_read > 0, "could not read any frames from the demo video"
    assert sum(fed_counts) > 0, "expected at least one in-quad detection over these frames"
    assert runner.funnel_totals["in_quad"] == sum(fed_counts), (
        f"funnel in_quad total ({runner.funnel_totals['in_quad']}) must equal the total "
        f"points fed to the heatmap ({sum(fed_counts)}) -- a mismatch means something is "
        f"once again computing 'in quad' two different ways"
    )
