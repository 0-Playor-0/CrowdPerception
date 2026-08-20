"""Ultralytics YOLO wrapper: frame in, person-only supervision.Detections out.

Device selection happens once at construction, not per frame -- moving a
model between devices is expensive, and this project's hard constraint is
"CPU or Apple MPS, never CUDA" (no NVIDIA GPU on this machine).

Also holds TiledDetector, a SAHI-style tiled-inference wrapper (see class
docstring below) -- carried forward from scripts/detect_track.py and
scripts/world_density.py's ad hoc `build_slicer` helper, refactored into a
reusable class with the same behaviour: same supervision.InferenceSlicer
under the hood, same default tile size / overlap / cross-tile NMS, so
docs/REAL_FOOTAGE_FINDINGS.md's C2.3 measurements still apply unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


def resolve_device(requested: str) -> str:
    """"auto" picks MPS if available, else CPU. An explicit "cuda" request
    is rejected outright rather than silently falling back -- this machine
    has no NVIDIA GPU, so a CUDA request almost certainly means a config
    copied from elsewhere, not an intentional choice."""
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if "cuda" in requested.lower():
        raise ValueError(
            f"detector.device={requested!r} requests CUDA -- this machine has "
            "no NVIDIA GPU. Use 'auto', 'mps', or 'cpu' in config.yaml."
        )
    return requested


class Detector:
    """Runs YOLO on a frame and returns only person-class detections."""

    def __init__(
        self,
        model_path: str,
        device: str,
        confidence_threshold: float,
        iou_threshold: float,
        person_class_id: int,
        imgsz: int | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        # None (the default) means "don't pass imgsz at all" -- Ultralytics
        # then uses its own model default (yolo11s.pt: 640) exactly as
        # before this parameter existed. This is intentional: whatever is
        # fed to detect() (a whole frame OR a single tile from
        # TiledDetector) gets resized to imgsz before the network sees it,
        # so a 1280px tile with imgsz left at 640 is halved on the way in --
        # see docs/TOTEST_FINDINGS.md TEST 2 for why that distinction matters.
        self.imgsz = imgsz

        resolved_device = resolve_device(device)
        self._model = YOLO(model_path)
        self._model.to(resolved_device)
        print(f"[detector] model={model_path} device={resolved_device} "
              f"person_class_id={person_class_id} conf>={confidence_threshold} "
              f"imgsz={imgsz if imgsz is not None else '(ultralytics default)'}")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Runs detection on one BGR frame, returns only person-class boxes."""
        predict_kwargs = {"conf": self.confidence_threshold, "iou": self.iou_threshold, "verbose": False}
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz
        result = self._model(frame, **predict_kwargs)[0]
        detections = sv.Detections.from_ultralytics(result)

        # YOLO detects all COCO classes by default -- this project only
        # cares about pedestrians, so everything else is dropped here,
        # before it ever reaches the tracker or geometry stage.
        person_mask = detections.class_id == self.person_class_id
        return detections[person_mask]


@dataclass
class TileConfig:
    """Tiling knobs, split out from TiledDetector so callers (CLI arg
    parsing in scripts/live_perception.py, tests) can build one without
    also touching detector/model construction.

    Deliberately does NOT include imgsz -- that's a property of the
    wrapped Detector (what resolution each tile gets fed to the network
    at), independent of tile_size (how big a crop each tile is). Construct
    the inner Detector with the imgsz you want before wrapping it in
    TiledDetector."""

    tile_size: tuple[int, int] = (1280, 1280)     # (width, height) px per tile
    overlap_ratio: float = 0.1                     # fraction of tile_size each neighbour overlaps by
    per_tile_conf_threshold: float | None = None    # None = trust the wrapped detector's own threshold
    nms_iou_threshold: float = 0.7                  # cross-tile merge NMS, applied in full-frame coords

    def overlap_wh(self) -> tuple[int, int]:
        width, height = self.tile_size
        return (round(width * self.overlap_ratio), round(height * self.overlap_ratio))


class TiledDetector:
    """SAHI-style tiled inference: slices a frame into overlapping tiles,
    runs the wrapped detector on each tile, and merges results with NMS in
    FULL-FRAME coordinates -- so a person straddling a tile seam (visible,
    in the overlap band, in two adjacent tiles) collapses back to one box,
    not two (see tests/test_detector.py::test_seam_duplicate_collapses_to_one_box).

    Delegates the actual slicing/offsetting/merging to
    supervision.InferenceSlicer (unchanged behaviour from the ad hoc
    `build_slicer` this replaces) -- InferenceSlicer already accumulates
    every tile's detections into full-frame pixel coordinates before
    applying its overlap_filter, which is exactly the "global NMS in
    full-frame coordinates" this class is required to do; there was no
    reason to reimplement that merge logic by hand.

    Set tile_config=None (or pass enabled=False) to bypass tiling entirely
    and fall back to a single whole-frame detector.detect() call -- this is
    what --no-tile wires up to in scripts/live_perception.py.

    Exposes last_n_tiles / last_n_before_nms / last_n_after_nms alongside
    last_latency_ms after each detect() call -- a tiling geometry sweep
    (comparing tile/imgsz combinations) needs exactly these numbers, and
    two throwaway scripts had already hand-rolled this instrumentation
    externally before it was worth folding into the class itself.
    """

    def __init__(
        self,
        detector: Detector,
        tile_config: TileConfig | None = None,
        enabled: bool = True,
    ) -> None:
        self._detector = detector
        self._tile_config = tile_config or TileConfig()
        self._enabled = enabled
        self._last_latency_ms: float = 0.0
        self._last_n_tiles: int = 0
        self._last_n_before_nms: int = 0
        self._last_n_after_nms: int = 0
        self._n_tiles_this_call = 0
        self._n_before_nms_this_call = 0

        if self._enabled:
            self._slicer = sv.InferenceSlicer(
                callback=self._detect_tile,
                slice_wh=self._tile_config.tile_size,
                overlap_wh=self._tile_config.overlap_wh(),
                overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
                iou_threshold=self._tile_config.nms_iou_threshold,
            )
        else:
            self._slicer = None

        print(
            f"[detector] TiledDetector enabled={self._enabled} "
            f"tile_size={self._tile_config.tile_size} "
            f"overlap_wh={self._tile_config.overlap_wh()} "
            f"nms_iou={self._tile_config.nms_iou_threshold} "
            f"per_tile_conf={self._tile_config.per_tile_conf_threshold} "
            f"imgsz={getattr(detector, 'imgsz', None) or '(ultralytics default)'}"
        )

    def _detect_tile(self, image_slice: np.ndarray) -> sv.Detections:
        self._n_tiles_this_call += 1
        detections = self._detector.detect(image_slice)
        threshold = self._tile_config.per_tile_conf_threshold
        if threshold is not None and detections.confidence is not None and len(detections) > 0:
            detections = detections[detections.confidence >= threshold]
        self._n_before_nms_this_call += len(detections)
        return detections

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Same detect(frame) -> Detections interface as Detector, so this
        is a drop-in replacement anywhere a plain Detector is used."""
        t0 = time.perf_counter()
        self._n_tiles_this_call = 0
        self._n_before_nms_this_call = 0
        if self._slicer is not None:
            detections = self._slicer(frame)
        else:
            detections = self._detector.detect(frame)
            self._n_tiles_this_call = 1
            self._n_before_nms_this_call = len(detections)
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._last_n_tiles = self._n_tiles_this_call
        self._last_n_before_nms = self._n_before_nms_this_call
        self._last_n_after_nms = len(detections)
        return detections

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_latency_ms(self) -> float:
        """Wall-clock time of the most recent detect() call, in
        milliseconds -- covers every tile in tiled mode. Exposed so a
        caller's HUD can show real measured per-frame detection cost
        instead of a guess."""
        return self._last_latency_ms

    @property
    def last_n_tiles(self) -> int:
        """Number of tiles processed in the most recent detect() call (1
        when tiling is disabled -- the single whole-frame pass)."""
        return self._last_n_tiles

    @property
    def last_n_before_nms(self) -> int:
        """Sum of per-tile detection counts (after each tile's own
        per_tile_conf_threshold filter, before the cross-tile merge NMS)
        from the most recent detect() call. Equal to last_n_after_nms when
        tiling is disabled -- there is no cross-tile merge to distinguish
        from in that case."""
        return self._last_n_before_nms

    @property
    def last_n_after_nms(self) -> int:
        """len() of the most recent detect() call's return value -- same
        number, exposed alongside last_n_before_nms so both halves of the
        merge are readable from one object without re-deriving len(result)
        yourself."""
        return self._last_n_after_nms
