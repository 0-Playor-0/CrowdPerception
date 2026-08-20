"""Tests for perception/geometry.py.

IMPORTANT on test design: a naive round-trip test (pixel -> world -> pixel,
assert you get the original pixel back) is NOT a meaningful correctness
check here. cv2.getPerspectiveTransform solves for a matrix that satisfies
the 4 corner correspondences *exactly*, and forward-then-inverse of the same
matrix is mathematically guaranteed to cancel out -- that would pass even if
target_points_m in config.yaml were completely wrong (e.g. metres swapped
for feet, or a typo'd digit), because the round trip never checks the
mapping against an INDEPENDENTLY known real-world value.

Instead, test_known_midpoint_distance_is_preserved below checks actual
metric distance in world space against a value computed independently of
ViewTransformer, via a geometric property (see comment in that test).
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.geometry import ViewTransformer, polygon_area, polygon_edge_lengths


def test_known_midpoint_distance_is_preserved() -> None:
    # SOURCE is constructed as a PARALLELOGRAM (BR = TL + width_vec +
    # height_vec exactly) rather than a general perspective quadrilateral.
    # This matters for the test: under a purely affine map (which is what a
    # parallelogram-to-rectangle correspondence produces, no projective
    # skew), edge MIDPOINTS map to edge midpoints exactly. That lets us
    # compute the "expected" world-space midpoint by simple averaging --
    # independently of ViewTransformer -- instead of trusting the code under
    # test to also produce the expected value.
    top_left_px = np.array([100.0, 500.0])
    width_vec_px = np.array([400.0, -20.0])
    height_vec_px = np.array([50.0, 300.0])
    top_right_px = top_left_px + width_vec_px
    bottom_left_px = top_left_px + height_vec_px
    bottom_right_px = top_left_px + width_vec_px + height_vec_px

    source_px = np.array([top_left_px, top_right_px, bottom_right_px, bottom_left_px])

    # TARGET: a real 4m x 6m ground rectangle, corners in the same order as
    # source_px (top_left, top_right, bottom_right, bottom_left).
    target_m = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 6.0], [0.0, 6.0]])

    transformer = ViewTransformer(source_px, target_m)

    # Necessary-but-not-sufficient sanity check: corners must round-trip,
    # since cv2.getPerspectiveTransform is solved to satisfy them exactly.
    # This alone would NOT catch a wrong target_m -- see module docstring.
    corners_world = transformer.transform_points(source_px)
    np.testing.assert_allclose(corners_world, target_m, atol=1e-3)

    # Pixel-space midpoints of the left edge and the top edge -- computed by
    # plain averaging, nothing to do with ViewTransformer.
    left_edge_midpoint_px = (top_left_px + bottom_left_px) / 2.0
    top_edge_midpoint_px = (top_left_px + top_right_px) / 2.0

    # Independently expected world-space midpoints (affine property: edge
    # midpoint in pixel space -> edge midpoint in metre space).
    expected_left_midpoint_m = np.array([0.0, 3.0])   # midpoint of (0,0)-(0,6)
    expected_top_midpoint_m = np.array([2.0, 0.0])     # midpoint of (0,0)-(4,0)
    expected_distance_m = float(
        np.linalg.norm(expected_left_midpoint_m - expected_top_midpoint_m)
    )  # sqrt(2^2 + 3^2) = sqrt(13) ~= 3.6056 m

    world_points = transformer.transform_points(
        np.array([left_edge_midpoint_px, top_edge_midpoint_px])
    )

    np.testing.assert_allclose(world_points[0], expected_left_midpoint_m, atol=1e-2)
    np.testing.assert_allclose(world_points[1], expected_top_midpoint_m, atol=1e-2)

    measured_distance_m = float(np.linalg.norm(world_points[0] - world_points[1]))
    assert measured_distance_m == pytest.approx(expected_distance_m, abs=1e-2)


def test_ill_conditioned_source_warns(capsys: pytest.CaptureFixture[str]) -> None:
    # Four nearly-collinear pixel points (a degenerate, almost-flat quad) --
    # simulates picking calibration points too close to the horizon.
    source_px = np.array([[0.0, 500.0], [100.0, 500.5], [200.0, 501.0], [300.0, 501.5]])
    target_m = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 6.0], [0.0, 6.0]])

    ViewTransformer(source_px, target_m)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "condition number" in captured.out


def test_polygon_edge_lengths_and_area_for_known_rectangle() -> None:
    # A plain 4m x 6m axis-aligned rectangle -- edge lengths and area are
    # trivial to verify by hand, so this just confirms the shoelace/edge math
    # itself isn't buggy (e.g. off-by-one in the wraparound index).
    rectangle_m = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 6.0], [0.0, 6.0]])

    edge_lengths = polygon_edge_lengths(rectangle_m)
    assert edge_lengths == pytest.approx([4.0, 6.0, 4.0, 6.0])

    area = polygon_area(rectangle_m)
    assert area == pytest.approx(24.0)


def test_view_transformer_rejects_wrong_point_count() -> None:
    with pytest.raises(ValueError):
        ViewTransformer(np.zeros((3, 2)), np.zeros((4, 2)))
