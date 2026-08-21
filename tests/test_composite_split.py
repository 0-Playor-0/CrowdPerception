"""Tests for server/pipeline_runner.py's composite split -- the boundary
that separates LivePerceptionRunner.build_composite()'s single
[camera_pane | heatmap_panel] image back into the two MJPEG streams.

An earlier version of _split_composite() recomputed that boundary from
DISPLAY_MAX_HEIGHT_PX and the raw frame's shape instead of reading it back
from the runner. If build_composite()'s internal layout ever changed, that
recomputation would silently drift and produce a wrong crop with no
error -- these tests cover both the fast always-on guard (test 1-3, no
model/video needed) and the real end-to-end invariant against the actual
pipeline (test 4, skipped if the real demo assets aren't present, same
convention as tests/test_live_perception_smoke.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from perception.heatmap import render_heatmap_panel
from scripts.live_perception import LivePerceptionRunner
from server.pipeline_runner import _split_composite

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = REPO_ROOT / "data" / "127690-739144743.mp4"
MODEL_PATH = REPO_ROOT / "models" / "yolo11s.pt"
CALIBRATION_PATH = REPO_ROOT / "calibration" / "127690-739144743.json"


def _fake_composite(width: int = 300, height: int = 100) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_splits_at_the_reported_boundary() -> None:
    composite = _fake_composite(width=300)
    fake_runner = SimpleNamespace(last_camera_pane_width_px=180)

    camera_img, heatmap_img = _split_composite(fake_runner, composite)

    assert camera_img.shape[1] == 180
    assert heatmap_img.shape[1] == 120
    assert camera_img.shape[1] + heatmap_img.shape[1] == composite.shape[1]


def test_raises_when_boundary_is_missing() -> None:
    composite = _fake_composite(width=300)
    fake_runner = SimpleNamespace(last_camera_pane_width_px=None)

    with pytest.raises(RuntimeError, match="invalid"):
        _split_composite(fake_runner, composite)


@pytest.mark.parametrize("bad_boundary", [0, -5, 300, 301, 10_000])
def test_raises_when_boundary_is_out_of_range(bad_boundary: int) -> None:
    # A layout change that made build_composite() report a boundary at or
    # past the composite's actual width (or non-positive) must fail loudly
    # here rather than silently produce an empty or nonsensical crop.
    composite = _fake_composite(width=300)
    fake_runner = SimpleNamespace(last_camera_pane_width_px=bad_boundary)

    with pytest.raises(RuntimeError, match="invalid"):
        _split_composite(fake_runner, composite)


pytestmark_real_pipeline = pytest.mark.skipif(
    not (VIDEO_PATH.exists() and MODEL_PATH.exists() and CALIBRATION_PATH.exists()),
    reason="needs the real demo video, model, and calibration file",
)


@pytestmark_real_pipeline
def test_reported_boundary_matches_the_real_composite_layout(tmp_path: Path) -> None:
    """End-to-end: runs one real frame through the actual pipeline and
    checks last_camera_pane_width_px against an INDEPENDENTLY re-rendered
    heatmap panel (same render_heatmap_panel call build_composite makes
    internally) rather than against a second copy of the same arithmetic
    -- this is what would actually catch a future layout change, e.g. a
    margin added between panes that a duplicated formula would miss."""
    with CALIBRATION_PATH.open("r") as f:
        calibration = json.load(f)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args = argparse.Namespace(
        video=str(VIDEO_PATH), calibration=str(CALIBRATION_PATH),
        model=str(MODEL_PATH), device="auto",
        confidence_threshold=0.02, iou_threshold=0.7,
        tile=False, tile_size=[1280, 1280], tile_overlap=0.1,
        tile_conf_threshold=None, nms_iou_threshold=0.7, imgsz=None,
        downscale_long_edge=1920,
        heatmap_mode="density", cell_size=0.5, heatmap_window_seconds=2.0,
        heatmap_sigma=1.0, heatmap_color_variant="hard", heatmap_vmax=3.0,
        frame_heatmap=False, frame_heatmap_cell_size_px=60, frame_heatmap_vmax=0.5,
        outputs_dir=str(tmp_path / "outputs"),
        trajectory_dir=str(tmp_path / "outputs" / "trajectories"),
    )
    runner = LivePerceptionRunner(args, calibration, fps, total_frames=1,
                                   frame_width=frame_width, frame_height=frame_height)

    ok, frame = capture.read()
    capture.release()
    assert ok, "could not read a frame from the demo video"

    render_data, hud = runner.process_frame(frame, frame_idx=0)
    composite = runner.build_composite(render_data, hud)
    runner.close(frames_processed=1)

    assert runner.last_camera_pane_width_px is not None
    assert 0 < runner.last_camera_pane_width_px < composite.shape[1]

    camera_img, heatmap_img = _split_composite(runner, composite)

    independent_heatmap_panel = render_heatmap_panel(
        runner.heatmap, runner.quad_m, runner.heatmap_mode,
        panel_height_px=camera_img.shape[0], color_variant=args.heatmap_color_variant,
    )
    assert heatmap_img.shape[1] == independent_heatmap_panel.shape[1], (
        "the split boundary no longer matches build_composite()'s actual heatmap panel "
        "width -- the composite layout changed without last_camera_pane_width_px following it"
    )
