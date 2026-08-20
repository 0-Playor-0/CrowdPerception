"""Tests for perception/ground.py.

Covers: foot-point vs centroid disagreement (quantified on a real detected
box from the project's real footage, not just a synthetic example -- the
whole point is to show the size of the bug a centroid anchor would have
been), known-corner projection against the calibration file's own stated
world coordinate, and in/out-of-quad exclusion bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.detector import Detector
from perception.ground import GroundProjector, centroid_points, foot_points, to_ground

CALIBRATION_PATH = Path("calibration/127690-739144743.json")
VIDEO_PATH = Path("data/127690-739144743.mp4")
MODEL_PATH = Path("models/yolo11s.pt")


def _load_calibration() -> dict:
    with CALIBRATION_PATH.open("r") as f:
        return json.load(f)


def test_foot_points_basic() -> None:
    xyxy = np.array([[10.0, 20.0, 30.0, 60.0]])
    np.testing.assert_allclose(foot_points(xyxy), [[20.0, 60.0]])


def test_centroid_points_basic() -> None:
    xyxy = np.array([[10.0, 20.0, 30.0, 60.0]])
    np.testing.assert_allclose(centroid_points(xyxy), [[20.0, 40.0]])


def test_foot_and_centroid_are_the_same_x_but_different_y() -> None:
    # By construction, foot and centroid always share x (both are box
    # horizontal midpoint) and differ only in y -- foot uses y2 (box
    # bottom), centroid uses (y1+y2)/2. Confirms the two anchors are not
    # accidentally identical for an ordinary (non-degenerate) box.
    xyxy = np.array([[100.0, 200.0, 140.0, 300.0]])
    foot = foot_points(xyxy)[0]
    centroid = centroid_points(xyxy)[0]
    assert foot[0] == pytest.approx(centroid[0])
    assert foot[1] != pytest.approx(centroid[1])
    assert foot[1] > centroid[1]   # foot (y2) is always below centroid on a real box


@pytest.mark.skipif(not CALIBRATION_PATH.exists(), reason="no calibration file for the demo video")
def test_known_corner_projects_to_its_stated_world_point() -> None:
    calibration = _load_calibration()
    H = np.array(calibration["H"], dtype=np.float64)
    image_points = np.array(calibration["image_points"], dtype=np.float64)
    world_points = np.array(calibration["world_points"], dtype=np.float64)

    projected = to_ground(image_points, H)
    np.testing.assert_allclose(projected, world_points, atol=1e-3)


@pytest.mark.skipif(not CALIBRATION_PATH.exists(), reason="no calibration file for the demo video")
def test_ground_projector_excludes_points_outside_quad() -> None:
    calibration = _load_calibration()
    H = np.array(calibration["H"], dtype=np.float64)
    quad_px = np.array(calibration["image_points"], dtype=np.float64)
    projector = GroundProjector(H, quad_px)

    quad_center = quad_px.mean(axis=0)
    inside_box = np.array([quad_center[0] - 5, quad_center[1] - 30, quad_center[0] + 5, quad_center[1]])
    # Far outside the quad and the frame's plausible ground area entirely.
    outside_box = np.array([5.0, 5.0, 15.0, 15.0])

    result = projector.project(np.array([inside_box, outside_box]))

    assert result.in_quad[0], "a box centered in the calibrated quad should not be excluded"
    assert not result.in_quad[1], "a box in the top-left corner should fall outside the calibrated quad"
    assert not np.isnan(result.world_xy[0]).any()
    assert np.isnan(result.world_xy[1]).all(), "excluded detections must get NaN world coords, not extrapolated garbage"
    assert result.n_excluded == 1


def test_ground_projector_defaults_to_foot_anchor() -> None:
    H = np.eye(3)
    quad_px = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    projector = GroundProjector(H, quad_px)
    assert projector.anchor == "foot"


def test_ground_projector_centroid_anchor_uses_centroid_not_foot() -> None:
    H = np.eye(3)   # identity: world coords == pixel coords, keeps this test purely about which anchor is picked
    quad_px = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    box = np.array([[10.0, 20.0, 30.0, 60.0]])   # foot=(20,60), centroid=(20,40)

    foot_projector = GroundProjector(H, quad_px, anchor="foot")
    centroid_projector = GroundProjector(H, quad_px, anchor="centroid")

    foot_result = foot_projector.project(box)
    centroid_result = centroid_projector.project(box)

    np.testing.assert_allclose(foot_result.foot_px[0], [20.0, 60.0])
    np.testing.assert_allclose(centroid_result.foot_px[0], [20.0, 40.0])
    assert not np.allclose(foot_result.world_xy[0], centroid_result.world_xy[0]), (
        "foot and centroid anchors must project to different world positions for a non-degenerate box"
    )


def test_ground_projector_rejects_invalid_anchor() -> None:
    H = np.eye(3)
    quad_px = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    with pytest.raises(ValueError):
        GroundProjector(H, quad_px, anchor="head")


@pytest.mark.skipif(
    not (CALIBRATION_PATH.exists() and VIDEO_PATH.exists() and MODEL_PATH.exists()),
    reason="needs the real demo video, model, and calibration file",
)
def test_foot_vs_centroid_disagreement_on_a_real_frame() -> None:
    """Runs the real detector on frame 0 of the real demo video, takes one
    in-quad detection, and prints/asserts how many metres apart the foot
    point and centroid anchors land after projection through the real
    calibration homography -- this is the size of the bug a centroid-based
    ground projection would have introduced, made concrete instead of
    theoretical."""
    calibration = _load_calibration()
    H = np.array(calibration["H"], dtype=np.float64)
    quad_px = np.array(calibration["image_points"], dtype=np.float64)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    ok, frame = capture.read()
    capture.release()
    assert ok, "could not read frame 0 of the demo video"

    detector = Detector(model_path=str(MODEL_PATH), device="auto",
                         confidence_threshold=0.3, iou_threshold=0.7, person_class_id=0)
    detections = detector.detect(frame)
    assert len(detections) > 0, "expected at least one person detection on frame 0"

    feet = foot_points(detections.xyxy)
    in_quad = np.array([
        cv2.pointPolygonTest(quad_px.astype(np.float32), tuple(p), False) >= 0 for p in feet
    ])
    assert in_quad.any(), "expected at least one in-quad detection on frame 0"

    box = detections.xyxy[in_quad][0]
    foot_world = to_ground(foot_points(box.reshape(1, 4)), H)[0]
    centroid_world = to_ground(centroid_points(box.reshape(1, 4)), H)[0]
    disagreement_m = float(np.linalg.norm(foot_world - centroid_world))

    print(f"\n[test_ground] box={box.tolist()}  foot_world={foot_world.tolist()}  "
          f"centroid_world={centroid_world.tolist()}  disagreement={disagreement_m:.3f}m")

    assert disagreement_m > 0.1, (
        "expected the foot-point and centroid anchors to disagree by a "
        "non-trivial margin on this oblique real footage -- if this ever "
        "shrinks to ~0 either the box is degenerate or the camera geometry "
        "changed enough to revisit this assumption"
    )
