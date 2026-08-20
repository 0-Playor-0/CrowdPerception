"""Tests for perception/detector.py's TiledDetector.

The headline case: a person straddling a tile seam must yield ONE box, not
two. We can't (and shouldn't) test that with a real YOLO model -- it would
be slow and its output isn't a controllable "known duplicate". Instead a
tiny fake detector finds a solid-colour rectangular blob in whatever image
slice it's given and returns its bounding box in that slice's own local
pixel coordinates -- same detect(frame) -> Detections interface real
detectors use. Placed so the blob sits fully inside the overlap band
between two tiles, both tiles independently "detect" the identical
full-size blob, and TiledDetector's cross-tile NMS must collapse the two
resulting (identical, IoU=1.0) full-frame boxes into one.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

import perception.detector as detector_module
from perception.detector import Detector, TileConfig, TiledDetector

BLOB_COLOR = (200, 200, 200)


class _ColorBlobDetector:
    """Fake detector: finds the single solid-colour blob in `frame`, if
    present, and returns its bbox. Mimics perception/detector.py's
    Detector.detect(frame) -> sv.Detections interface exactly."""

    def __init__(self, color_bgr: tuple[int, int, int] = BLOB_COLOR) -> None:
        self._color = np.array(color_bgr, dtype=np.uint8)
        self.n_calls = 0

    def detect(self, frame: np.ndarray) -> sv.Detections:
        self.n_calls += 1
        mask = np.all(frame == self._color, axis=-1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float64),
                class_id=np.zeros((0,), dtype=int),
            )
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        return sv.Detections(
            xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float64),
            class_id=np.array([0], dtype=int),
        )


def _make_seam_straddling_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """100x50 image; two 60x50 tiles (step=40, overlap=20) will be placed at
    x=[0,60) and x=[40,100). A blob at x=[45,56), y=[10,40) sits entirely
    inside the [40,60) overlap band -- fully visible, identically, in BOTH
    tiles."""
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    x1, y1, x2, y2 = 45, 10, 56, 40
    image[y1:y2, x1:x2] = BLOB_COLOR
    return image, (x1, y1, x2, y2)


def test_seam_duplicate_collapses_to_one_box() -> None:
    image, expected_box = _make_seam_straddling_image()
    fake_detector = _ColorBlobDetector()
    tile_config = TileConfig(tile_size=(60, 50), overlap_ratio=1 / 3, nms_iou_threshold=0.5)
    tiled = TiledDetector(fake_detector, tile_config=tile_config, enabled=True)

    result = tiled.detect(image)

    assert len(result) == 1, (
        f"expected exactly one merged box for a person straddling the tile "
        f"seam, got {len(result)}: {result.xyxy.tolist()}"
    )
    np.testing.assert_allclose(result.xyxy[0], expected_box, atol=1.0)
    # sanity: both tiles really did run (otherwise this test would pass
    # trivially even if tiling were silently disabled)
    assert fake_detector.n_calls >= 2


def test_no_tile_fallback_matches_whole_frame_detect() -> None:
    image, expected_box = _make_seam_straddling_image()
    fake_detector = _ColorBlobDetector()
    tiled = TiledDetector(fake_detector, enabled=False)

    result = tiled.detect(image)

    assert fake_detector.n_calls == 1, "disabled tiling must call the wrapped detector exactly once, whole-frame"
    assert len(result) == 1
    np.testing.assert_allclose(result.xyxy[0], expected_box, atol=1.0)


def test_latency_is_measured_and_nonnegative() -> None:
    image, _ = _make_seam_straddling_image()
    fake_detector = _ColorBlobDetector()
    tiled = TiledDetector(fake_detector, tile_config=TileConfig(tile_size=(60, 50), overlap_ratio=1 / 3))

    assert tiled.last_latency_ms == 0.0   # nothing detected yet
    tiled.detect(image)
    assert tiled.last_latency_ms >= 0.0


def test_per_tile_conf_threshold_filters_low_confidence_tile_detections() -> None:
    image, _ = _make_seam_straddling_image()

    class _LowConfDetector(_ColorBlobDetector):
        def detect(self, frame: np.ndarray) -> sv.Detections:
            detections = super().detect(frame)
            if len(detections) > 0:
                detections.confidence[:] = 0.2
            return detections

    tile_config = TileConfig(tile_size=(60, 50), overlap_ratio=1 / 3, per_tile_conf_threshold=0.5)
    tiled = TiledDetector(_LowConfDetector(), tile_config=tile_config, enabled=True)

    result = tiled.detect(image)
    assert len(result) == 0, "per-tile confidence threshold should have dropped every 0.2-confidence detection"


def test_before_after_nms_and_tile_count_exposed() -> None:
    image, _ = _make_seam_straddling_image()
    fake_detector = _ColorBlobDetector()
    tile_config = TileConfig(tile_size=(60, 50), overlap_ratio=1 / 3, nms_iou_threshold=0.5)
    tiled = TiledDetector(fake_detector, tile_config=tile_config, enabled=True)

    tiled.detect(image)

    assert tiled.last_n_tiles >= 2
    # the blob is detected once per tile it's visible in (both tiles here) --
    # before the cross-tile merge NMS collapses the two identical boxes to one
    assert tiled.last_n_before_nms == tiled.last_n_tiles
    assert tiled.last_n_after_nms == 1


def test_before_after_nms_equal_when_tiling_disabled() -> None:
    image, _ = _make_seam_straddling_image()
    fake_detector = _ColorBlobDetector()
    tiled = TiledDetector(fake_detector, enabled=False)

    tiled.detect(image)

    assert tiled.last_n_tiles == 1
    assert tiled.last_n_before_nms == tiled.last_n_after_nms == 1


class _KwargsCapturingModel:
    """Stands in for ultralytics.YOLO: records the kwargs Detector.detect()
    calls it with, then raises immediately -- avoids needing to fabricate a
    full fake ultralytics Results object just to check what imgsz was (or
    wasn't) passed through."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.calls: list[dict] = []

    def to(self, _device: str) -> "_KwargsCapturingModel":
        return self

    def __call__(self, _frame: np.ndarray, **kwargs: object):
        self.calls.append(kwargs)
        raise _StopAfterCapture()


class _StopAfterCapture(Exception):
    pass


def test_imgsz_forwarded_to_ultralytics_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _KwargsCapturingModel()
    monkeypatch.setattr(detector_module, "YOLO", lambda _path: fake_model)

    det = Detector(
        model_path="unused.pt", device="cpu", confidence_threshold=0.3,
        iou_threshold=0.7, person_class_id=0, imgsz=1280,
    )
    with pytest.raises(_StopAfterCapture):
        det.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert fake_model.calls[-1]["imgsz"] == 1280


def test_imgsz_omitted_by_default_preserving_ultralytics_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _KwargsCapturingModel()
    monkeypatch.setattr(detector_module, "YOLO", lambda _path: fake_model)

    det = Detector(
        model_path="unused.pt", device="cpu", confidence_threshold=0.3,
        iou_threshold=0.7, person_class_id=0,
    )
    assert det.imgsz is None
    with pytest.raises(_StopAfterCapture):
        det.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert "imgsz" not in fake_model.calls[-1], (
        "imgsz must be omitted (not e.g. set to 640) when not explicitly requested -- "
        "this is what keeps every existing caller's behaviour byte-for-byte unchanged"
    )
