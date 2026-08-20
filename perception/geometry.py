"""Pixel <-> metre ground-plane conversion via a fixed-camera homography.

Everything in this file assumes a FIXED camera looking down at a flat ground
plane. The homography maps a 4-point pixel quadrilateral (SOURCE) onto its
real-world metre-space counterpart (TARGET); any other point inside that
region is mapped by the same fitted transform.
"""

from __future__ import annotations

import cv2
import numpy as np

# Condition number above this suggests SOURCE points are nearly collinear or
# sit very close together in the image -- e.g. picked too near the horizon,
# where the perspective mapping to distant ground truth becomes extreme.
# Above this threshold, small pixel-detection noise turns into large,
# unreliable swings in far-field metre coordinates.
ILL_CONDITIONED_THRESHOLD = 1.0e6


class ViewTransformer:
    """Maps points from image pixel space to ground-plane metre space.

    Built from four point correspondences: SOURCE (pixels, read off a video
    frame) and TARGET (metres, physically measured on the ground). Both
    arrays must describe the SAME four ground-plane locations, in the same
    order -- point i in source_px must be the same physical spot as point i
    in target_m.
    """

    def __init__(self, source_px: np.ndarray, target_m: np.ndarray) -> None:
        source_px = np.asarray(source_px, dtype=np.float64)
        target_m = np.asarray(target_m, dtype=np.float64)

        if source_px.shape != (4, 2) or target_m.shape != (4, 2):
            raise ValueError(
                "ViewTransformer requires exactly 4 (x, y) point pairs for "
                f"both source and target, got source={source_px.shape}, "
                f"target={target_m.shape}"
            )

        # cv2.getPerspectiveTransform solves for the 3x3 homography matrix M
        # such that M @ [x_px, y_px, 1]^T ~ [x_m, y_m, 1]^T (up to scale) for
        # all 4 correspondence pairs. Units: input pixels, output metres.
        self._matrix = cv2.getPerspectiveTransform(
            source_px.astype(np.float32), target_m.astype(np.float32)
        )
        self._warn_if_ill_conditioned()

    def _warn_if_ill_conditioned(self) -> None:
        # Warn at runtime instead of failing silently -- the matrix solve
        # "succeeds" numerically either way, but a high condition number
        # means far-field world coordinates will be unreliable even though
        # no exception is raised.
        condition_number = np.linalg.cond(self._matrix)
        if condition_number > ILL_CONDITIONED_THRESHOLD:
            print(
                f"[geometry] WARNING: homography condition number "
                f"{condition_number:.2e} exceeds {ILL_CONDITIONED_THRESHOLD:.0e}. "
                "SOURCE points may be too close to the horizon or nearly "
                "collinear -- re-pick calibration points closer to the "
                "camera / less extreme in perspective, or expect unstable "
                "world coordinates for points far from the camera."
            )

    def transform_points(self, points_px: np.ndarray) -> np.ndarray:
        """Convert an array of (x, y) points from PIXELS to METRES.

        Input units: pixels, image coordinates (origin top-left).
        Output units: metres, ground-plane coordinates, same frame as the
        target_m points passed to __init__.
        """
        points_px = np.asarray(points_px)
        if points_px.size == 0:
            return points_px.reshape(0, 2).astype(np.float32)

        # cv2.perspectiveTransform expects shape (N, 1, 2) float32.
        reshaped = points_px.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self._matrix)
        return transformed.reshape(-1, 2)


def polygon_edge_lengths(points: np.ndarray) -> list[float]:
    """Euclidean length of each edge of a closed polygon, in the input's own
    units (e.g. pass metre-space points to get edge lengths in metres)."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    return [
        float(np.linalg.norm(points[(i + 1) % n] - points[i])) for i in range(n)
    ]


def polygon_area(points: np.ndarray) -> float:
    """Area of a closed polygon via the shoelace formula. Works for any
    simple (non-self-intersecting) polygon, not just rectangles -- useful
    since a hand-measured ground calibration quad is rarely a perfect
    rectangle."""
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test, in whatever units `point` and
    `polygon` share (world metres or pixels, both work). Single shared
    implementation -- this was independently duplicated in
    scripts/validate_perception.py and scripts/plot_coverage.py before
    being pulled in here; both now import it from this module instead."""
    x, y = point
    n = len(polygon)
    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            else:
                x_intersect = x1
            if x1 == x2 or x <= x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside
