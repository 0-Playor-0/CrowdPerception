"""Synthetic "detector" that reads ground truth instead of running YOLO.

RESTORED FILE -- INTENTIONALLY UNREFERENCED IN THIS CHECKOUT. Do not
delete this as dead code just because nothing in this perception-only fork
imports it. This module is the ground-truth detection path behind the
parent synthetic-calibration project's validated results and the
outstanding Var(V)/pressure ablation there -- it requires simulator/
(specifically simulator/camera.py), which this fork deliberately does not
carry (see the top-level project instructions: no simulator/, no
analytics/, no alarm tiers here). It is kept in this checkout, unused, so
it stays available and importable the moment simulator/ is present again
(e.g. back in the parent project, or a future merge) -- that is exactly
why its simulator import is lazy (see _import_camera below) rather than a
module-level import: a perception-only checkout can still import this
file, and every other part of perception/ and any live script, without
ever touching simulator/ or failing because it's absent. Only actually
instantiating TruthDetector requires simulator/ to exist.

Why this exists (original context, still accurate): simulator/renderer.py
draws agents as flat-colored capsule silhouettes, and empirically YOLO
finds ZERO person detections on that video at any usable confidence
threshold (verified directly -- 0/150+ agents detected across a full clip,
one spurious false positive at 10% confidence). That's a real domain-gap
finding, not a bug in detector.py (which correctly finds 4/4 people on a
real photo). Rather than rebuild the renderer to be YOLO-detectable, this
module generates detections directly from truth_tracks.jsonl +
truth_camera.json, with injected noise, so the tracker/geometry/velocity
stages downstream of detection can still be validated end-to-end against
exact ground truth. Detection recall/precision computed against THIS
source are not measurements of anything -- see the gating in
scripts/validate_perception.py (parent project).

Same detect(frame) -> Detections interface as perception/detector.py, so
a config-driven entry point in the parent project can pick which class to
instantiate based on a detector.source config key. Internally, it tracks
its own frame counter rather than taking frame_id as an argument (to keep
the interface identical to Detector.detect) -- this assumes detect() is
called exactly once per frame, in order, starting at frame 0.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv


def _load_truth_by_frame(truth_dir: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    with (truth_dir / "truth_tracks.jsonl").open("r") as f:
        for line in f:
            record = json.loads(line)
            by_frame[record["frame_id"]].append(record)
    return by_frame


def _load_camera(truth_dir: Path) -> "Camera":  # noqa: F821 -- see _import_camera
    Camera = _import_camera()
    with (truth_dir / "truth_camera.json").open("r") as f:
        camera_raw = json.load(f)["camera"]
    return Camera(
        position_m=np.array(camera_raw["position_m"]),
        look_at_m=np.array(camera_raw["look_at_m"]),
        up_hint=np.array(camera_raw["up_hint"]),
        fov_deg=camera_raw["fov_deg"],
        image_width_px=camera_raw["image_width_px"],
        image_height_px=camera_raw["image_height_px"],
    )


def _import_camera():
    """Deferred import of simulator.camera.Camera -- see module docstring.

    Only called from TruthDetector construction, so a perception-only
    checkout (this fork) can import perception.truth_detector, perception
    (the package), and any live script freely; the ImportError only
    surfaces if you actually try to instantiate TruthDetector, at which
    point a clear message is far more useful than the raw
    ModuleNotFoundError."""
    try:
        from simulator.camera import Camera
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TruthDetector requires simulator/camera.py -- not present in "
            "this checkout (this is the perception-only fork; simulator/ "
            "was intentionally not carried forward). Use the real "
            "perception.detector.Detector instead, or run this against a "
            "checkout that includes simulator/ (the parent "
            "synthetic-calibration project)."
        ) from exc
    return Camera


class TruthDetector:
    """Generates sv.Detections from simulator ground truth instead of
    running a real detector. All noise parameters default to a small
    non-zero value (a "real detector" is never noiseless) and can each be
    set to 0 independently for an exact-input sanity run.

    Requires simulator/camera.py (not present in this perception-only fork)
    -- see the module docstring's RESTORED FILE note. Constructing this
    class raises ModuleNotFoundError with an actionable message if
    simulator/ is absent; every other part of perception/ and any live
    script is unaffected."""

    def __init__(
        self,
        truth_dir: Path,
        human_width_m: float,
        human_height_m: float,
        bbox_jitter_px_std: float,
        dropout_probability: float,
        false_positive_rate: float,
        confidence_mean: float,
        confidence_std: float,
        person_class_id: int,
        seed: int,
    ) -> None:
        self._truth_by_frame = _load_truth_by_frame(truth_dir)
        self._camera = _load_camera(truth_dir)
        self._human_width_m = human_width_m
        self._human_height_m = human_height_m
        self._bbox_jitter_px_std = bbox_jitter_px_std
        self._dropout_probability = dropout_probability
        self._false_positive_rate = false_positive_rate
        self._confidence_mean = confidence_mean
        self._confidence_std = confidence_std
        self._person_class_id = person_class_id
        self._rng = np.random.default_rng(seed)
        self._next_frame_id = 0

        print(
            f"[truth_detector] truth_dir={truth_dir} "
            f"dropout={dropout_probability} false_positive_rate={false_positive_rate} "
            f"bbox_jitter_px_std={bbox_jitter_px_std} seed={seed}"
        )

    def _sample_confidence(self, n: int) -> np.ndarray:
        confidence = self._rng.normal(self._confidence_mean, self._confidence_std, size=n)
        return np.clip(confidence, 0.01, 1.0)

    def _human_bbox_px(self, x_world: float, y_world: float) -> np.ndarray | None:
        """Projects a human-sized box centered on (x_world, y_world) to
        pixel xyxy. Returns None if any of the box's defining 3D points
        project behind the camera (shouldn't happen for agents already
        known to be in-scene, but guarded rather than assumed)."""
        right_xy = self._camera.R[0, :2]
        right_xy = right_xy / (np.linalg.norm(right_xy) + 1e-9)
        half_width = self._human_width_m / 2.0

        points_world = np.array(
            [
                [x_world, y_world, 0.0],                                            # feet
                [x_world, y_world, self._human_height_m],                            # head
                [x_world + half_width * right_xy[0], y_world + half_width * right_xy[1], self._human_height_m / 2.0],
                [x_world - half_width * right_xy[0], y_world - half_width * right_xy[1], self._human_height_m / 2.0],
            ]
        )
        pixel_coords, in_front = self._camera.project(points_world)
        if not in_front.all():
            return None

        feet_px, head_px, side_a_px, side_b_px = pixel_coords
        center_x = (feet_px[0] + head_px[0]) / 2.0
        half_width_px = abs(side_a_px[0] - side_b_px[0]) / 2.0
        y_min = min(head_px[1], feet_px[1])
        y_max = max(head_px[1], feet_px[1])
        return np.array([center_x - half_width_px, y_min, center_x + half_width_px, y_max])

    def _false_positive_bbox_px(self, image_width_px: int, image_height_px: int) -> np.ndarray:
        # No ground-truth position to anchor to -- a plausible-sized box at
        # a uniformly random image location, same size heuristic as a real
        # detection so it isn't trivially distinguishable by shape alone.
        approx_height_px = image_height_px * 0.15
        approx_width_px = approx_height_px * (self._human_width_m / self._human_height_m)
        cx = self._rng.uniform(0, image_width_px)
        cy = self._rng.uniform(0, image_height_px)
        return np.array(
            [cx - approx_width_px / 2, cy - approx_height_px / 2, cx + approx_width_px / 2, cy + approx_height_px / 2]
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        frame_id = self._next_frame_id
        self._next_frame_id += 1

        image_height_px, image_width_px = frame.shape[:2]
        boxes: list[np.ndarray] = []
        # Tags each box with the truth agent_id it came from -- false
        # positives get -1. This is what lets a funnel report (parent
        # project) trace a specific truth agent through every downstream
        # pipeline stage (zone filter, tracker, velocity), instead of only
        # seeing the final output. sv.Detections.data survives
        # boolean/index-based slicing (verified against this supervision
        # version, including inside trackers.OCSORTTracker's own internal
        # reconstruction), so the tag rides along automatically.
        agent_ids: list[int] = []
        for agent in self._truth_by_frame.get(frame_id, []):
            if self._rng.random() < self._dropout_probability:
                continue   # simulates a missed detection
            box = self._human_bbox_px(agent["x_world"], agent["y_world"])
            if box is not None:
                boxes.append(box)
                agent_ids.append(agent["agent_id"])

        false_positive_count = self._rng.poisson(self._false_positive_rate)
        for _ in range(false_positive_count):
            boxes.append(self._false_positive_bbox_px(image_width_px, image_height_px))
            agent_ids.append(-1)

        if not boxes:
            # Keep the truth_agent_id key present even when empty -- an
            # absent key (which bare sv.Detections.empty() would produce)
            # is indistinguishable from "not using TruthDetector at all" to
            # a funnel-instrumentation caller.
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float64),
                class_id=np.zeros((0,), dtype=int),
                data={"truth_agent_id": np.zeros((0,), dtype=int)},
            )

        xyxy = np.array(boxes, dtype=np.float64)
        if self._bbox_jitter_px_std > 0:
            xyxy = xyxy + self._rng.normal(0.0, self._bbox_jitter_px_std, size=xyxy.shape)
        # Keep boxes valid (x_min<x_max, y_min<y_max) after jitter, and
        # inside the frame -- a real detector never emits an inverted or
        # off-canvas box.
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, image_width_px - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], xyxy[:, 0] + 1, image_width_px)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, image_height_px - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], xyxy[:, 1] + 1, image_height_px)

        confidence = self._sample_confidence(len(boxes))
        class_id = np.full(len(boxes), self._person_class_id, dtype=int)

        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=confidence,
            class_id=class_id,
            data={"truth_agent_id": np.array(agent_ids, dtype=int)},
        )
