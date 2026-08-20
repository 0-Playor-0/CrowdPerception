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
actually lie on the calibrated ground plane.

Audit of existing centroid-vs-foot-point usage in this repo (checked before
writing this module, since the task asked for it explicitly):
scripts/world_density.py's main loop already uses
`detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)`,
i.e. the foot point, not the centroid. No centroid-based ground projection
bug was found anywhere in this codebase. This module exists to give
tiled/tracked pixel boxes the same correct anchor with vectorised NumPy
instead of supervision's per-Detections helper, and to add the
in-quad-or-excluded bookkeeping that call site doesn't do.
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
    """Box centroid -- (x1+x2)/2, (y1+y2)/2. Provided ONLY for the
    foot-vs-centroid disagreement test (tests/test_ground.py) that
    quantifies how far wrong a centroid-based projection would have been on
    this project's real footage; nothing in the live pipeline uses this."""
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
    foot_px: np.ndarray        # (N, 2) pixels -- the anchor actually used
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
