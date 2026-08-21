"""Pixel-box -> ground-plane metre projection, anchored at the FOOT POINT.

CRITICAL, and the entire reason this module is separate from
perception/geometry.py's generic ViewTransformer: the anchor for projecting
a bounding box onto the ground plane must be the box's bottom-centre
((x1+x2)/2, y2) -- where the person's feet actually touch the ground --
never the box centroid. The centroid sits at roughly torso height; on an
oblique camera view (any camera that isn't looking straight down), a
torso-height point does not lie on the ground plane at all, so running it
through a ground-plane homography produces a position that is metres off,
not just slightly biased. The homography is only valid for points that
actually lie on the calibrated ground plane. Measured on real footage
(tests/test_ground.py::test_foot_point_is_the_correct_anchor_not_centroid):
0.88m disagreement on a single ordinary detection, not a worst case.

Audit of existing centroid-vs-foot-point usage in this repo (checked before
writing this module, since the task asked for it explicitly):
scripts/world_density.py's main loop already uses
`detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)`,
i.e. the foot point, not the centroid. No centroid-based ground projection
bug was found anywhere in this codebase.

This is a fixed decision, not a configuration option. An earlier version of
this module let the anchor be selected at runtime (GroundProjector's
"anchor" parameter, a --ground-anchor CLI flag, and a
scripts/compare_ground_anchor.py A/B tool) to measure the disagreement
empirically across a full clip. That measurement is done -- centroid is
wrong here, foot is right, there is nothing left to compare -- and the
selectability was reverted in full: no anchor parameter, no CLI flag, no
branch. foot_points() is the only anchor perception/ground.py ever uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perception.geometry import point_in_polygon


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre anchor of each box: ((x1+x2)/2, y2). This is the
    ground-contact point -- see module docstring for why it must be this
    and not the box centroid.

    xyxy: (N, 4) array of [x1, y1, x2, y2] pixel boxes.
    Returns: (N, 2) array of [x, y] pixel points.
    """
    xyxy = np.asarray(xyxy, dtype=np.float64)
    if xyxy.size == 0:
        return xyxy.reshape(0, 2)
    x1, _, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    return np.stack([(x1 + x2) / 2.0, y2], axis=1)


def centroid_points(xyxy: np.ndarray) -> np.ndarray:
    """Box centroid -- (x1+x2)/2, (y1+y2)/2. WRONG anchor for ground-plane
    projection on an oblique view -- see module docstring. Exists ONLY to
    support tests/test_ground.py's regression test that documents why foot
    is the fixed choice (quantifies the real disagreement on real footage).
    Nothing in the live pipeline calls this -- GroundProjector always uses
    foot_points()."""
    xyxy = np.asarray(xyxy, dtype=np.float64)
    if xyxy.size == 0:
        return xyxy.reshape(0, 2)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=1)


def to_ground(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Vectorised pixel -> metre ground-plane projection via homography H.

    points_px: (N, 2) pixel points. H: (3, 3) homography (pixels -> metres).
    Returns: (N, 2) metre points, same frame as whatever world coordinates H
    was fit against.
    """
    points_px = np.asarray(points_px, dtype=np.float64)
    if points_px.size == 0:
        return points_px.reshape(0, 2)
    reshaped = points_px.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(reshaped, np.asarray(H, dtype=np.float32))
    return transformed.reshape(-1, 2).astype(np.float64)


@dataclass
class GroundProjection:
    world_xy: np.ndarray       # (N, 2) metres -- NaN row where in_quad is False (no valid ground position)
    foot_px: np.ndarray        # (N, 2) pixels -- the foot point (bottom-centre) used for this projection
    in_quad: np.ndarray        # (N,) bool -- True if the foot point falls inside the calibrated quad
    n_excluded: int            # convenience: int(np.sum(~in_quad))


class GroundProjector:
    """Combines foot-point extraction, the pixel->metre homography, and
    quad-membership exclusion into one call. A detection whose foot point
    falls outside the calibrated quad has no valid ground position -- rather
    than silently emitting whatever the homography extrapolates to out
    there (garbage: the homography is only fit to be trustworthy inside the
    quad), its world_xy row is set to NaN and in_quad is False. Callers
    (scripts/live_perception.py) are expected to still draw/log these
    detections (dimmed/dashed, in_quad=False) rather than drop them
    entirely -- only their world coordinate is unusable, not the detection
    itself.

    This is also the SINGLE source of truth for "is this detection in the
    calibrated quad" for the whole live pipeline -- callers should call
    project() once per frame on the full detection set and reuse the result
    (in_quad mask, world_xy, foot_px) everywhere, rather than running a
    second, independent in-quad test elsewhere. A previous version of
    scripts/live_perception.py ran a second test (sv.PolygonZone, on a
    rounded-to-integer-pixels copy of the quad) purely to filter the
    tracker's input, and it silently disagreed with this class's own
    full-float test on ~0.15% of real in-quad detections at the boundary --
    fixed by removing that second test, not by reconciling the two.
    """

    def __init__(self, H: np.ndarray, quad_px: np.ndarray) -> None:
        self._H = np.asarray(H, dtype=np.float64)
        self._quad_px = np.asarray(quad_px, dtype=np.float64)
        if self._quad_px.shape != (4, 2):
            raise ValueError(f"quad_px must be shape (4, 2), got {self._quad_px.shape}")

    def project(self, xyxy: np.ndarray) -> GroundProjection:
        xyxy = np.asarray(xyxy, dtype=np.float64)
        n = xyxy.shape[0]
        if n == 0:
            empty2 = np.zeros((0, 2), dtype=np.float64)
            return GroundProjection(empty2, empty2, np.zeros((0,), dtype=bool), 0)

        feet = foot_points(xyxy)
        in_quad = np.array(
            [point_in_polygon(pt, self._quad_px) for pt in feet], dtype=bool
        )

        world_xy = np.full((n, 2), np.nan, dtype=np.float64)
        if in_quad.any():
            world_xy[in_quad] = to_ground(feet[in_quad], self._H)

        return GroundProjection(
            world_xy=world_xy,
            foot_px=feet,
            in_quad=in_quad,
            n_excluded=int(np.sum(~in_quad)),
        )
