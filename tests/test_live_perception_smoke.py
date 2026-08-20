"""Headless end-to-end smoke test for scripts/live_perception.py: runs the
real pipeline (real detector, real tracker, real ground projection) against
100 frames of the real demo video with --no-window, and checks it exits
cleanly and produces every output artifact it promises (trajectory CSV,
track-stats sidecar, a snapshot PNG). Slow (real YOLO inference on real
frames) -- skipped automatically if the video/model/calibration aren't
present in this checkout."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

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
