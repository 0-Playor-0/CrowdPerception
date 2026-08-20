"""Tests for scripts/calibrate_video.py's pure calibration-building logic
(build_calibration_record / validate_quad) -- no GUI, no stdin involved."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_video import QuadValidationError, build_calibration_record, validate_quad

RECTANGLE_IMAGE_POINTS = np.array([
    [100.0, 500.0],   # near-left
    [500.0, 500.0],   # near-right
    [420.0, 100.0],   # far-right
    [180.0, 100.0],   # far-left
])

CALIBRATION_PATH = Path("calibration/127690-739144743.json")


def test_homography_round_trip_under_1px() -> None:
    record = build_calibration_record(
        image_points=RECTANGLE_IMAGE_POINTS,
        width_m=7.0, height_m=5.0,
        source="ESTIMATED", note="test",
        video_path="x.mp4", frame_index=0, video_resolution=(1920, 1080),
    )
    assert record["round_trip_error_px"] < 1.0


def test_world_points_form_the_requested_rectangle() -> None:
    record = build_calibration_record(
        image_points=RECTANGLE_IMAGE_POINTS,
        width_m=8.0, height_m=4.0,
        source="USER_MEASURED", note="test",
        video_path="x.mp4", frame_index=0, video_resolution=(1920, 1080),
    )
    world = np.array(record["world_points"])
    np.testing.assert_allclose(world, [[-4, 0], [4, 0], [4, 4], [-4, 4]])


def test_near_far_scale_reported_and_far_is_coarser() -> None:
    # Near edge (0-1) is wider in pixels than far edge (2-3) for this
    # perspective quad -- far-edge scale (m/px) should come out larger
    # (fewer pixels covering the same real-world width = each pixel is
    # "worth" more metres), i.e. this reproduces the foreshortening
    # direction documented in docs/REAL_FOOTAGE_FINDINGS.md C2.4.
    record = build_calibration_record(
        image_points=RECTANGLE_IMAGE_POINTS,
        width_m=7.0, height_m=5.0,
        source="ESTIMATED", note="test",
        video_path="x.mp4", frame_index=0, video_resolution=(1920, 1080),
    )
    assert record["far_edge_scale_m_per_px"] > record["near_edge_scale_m_per_px"]


def test_bowtie_quad_is_rejected() -> None:
    # Visit the corners out of convex order (near-left, far-right,
    # near-right, far-left) -- connecting them in sequence crosses the two
    # "diagonal" edges, the classic bowtie/self-intersecting quadrilateral.
    bowtie_points = RECTANGLE_IMAGE_POINTS[[0, 2, 1, 3]]
    with pytest.raises(QuadValidationError):
        validate_quad(bowtie_points)


def test_non_convex_quad_is_rejected() -> None:
    # Push one point (far-right) inward past the quad's interior -- concave.
    concave_points = RECTANGLE_IMAGE_POINTS.copy()
    concave_points[2] = [300.0, 400.0]   # deep inside what should be the quad's interior
    with pytest.raises(QuadValidationError):
        validate_quad(concave_points)


def test_convex_quad_passes_validation() -> None:
    validate_quad(RECTANGLE_IMAGE_POINTS)   # should not raise


def test_rejects_wrong_point_count() -> None:
    with pytest.raises(ValueError):
        build_calibration_record(
            image_points=RECTANGLE_IMAGE_POINTS[:3],
            width_m=7.0, height_m=5.0,
            source="ESTIMATED", note="test",
            video_path="x.mp4", frame_index=0, video_resolution=(1920, 1080),
        )


@pytest.mark.skipif(not CALIBRATION_PATH.exists(), reason="no calibration file for the demo video")
def test_saved_demo_calibration_round_trips_under_1px() -> None:
    with CALIBRATION_PATH.open("r") as f:
        record = json.load(f)
    assert record["round_trip_error_px"] < 1.0
    assert record["source"] in ("ESTIMATED", "USER_MEASURED")
