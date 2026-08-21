"""Diagnoses false positives in tiled YOLO detection on the Myeongdong demo
clip: classifies every detection surviving the CURRENT cross-tile NMS
(iou=0.7) into DUPLICATE / NESTED / SEAM ARTEFACT / NON-PERSON-candidate /
VALID, reports before/after counts at both NMS stages (per-tile Ultralytics
NMS, cross-tile merge NMS), and sweeps cross-tile IoU thresholds + NMS vs
NMM without re-running the detector (pure array ops on cached per-tile
detections).

Read-only diagnostic. Imports perception.detector.Detector directly and
uses it exactly as scripts/live_perception.py's build_detector() does
(same confidence/iou/tile_size/tile_overlap/imgsz defaults) -- does not
change or reimplement detection/NMS math, only measures it. Does NOT
modify perception/, scripts/live_perception.py, or scripts/detect_track.py.

Usage:
    uv run python scripts/diagnose_tiled_nms.py
    uv run python scripts/diagnose_tiled_nms.py --confidence-threshold 0.07 \\
        --out-dir outputs/nms_diagnosis_conf0.07
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import supervision as sv
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.detection.utils.iou_and_nms import box_iou_batch, box_non_max_suppression

from perception.detector import Detector

VIDEO_PATH = Path("data/127690-739144743.mp4")
MODEL_PATH = "models/yolo11s.pt"
FRAME_INDICES = [0, 135, 300, 512]

# Pipeline defaults -- matches scripts/live_perception.py's
# build_detector()/TileConfig defaults. CONFIDENCE_THRESHOLD tracks that
# script's own default (raised 0.02 -> 0.15 after this script's own
# measurements, see live_perception.py's --confidence-threshold help
# text) -- override via --confidence-threshold to sweep other values.
# Tile size and imgsz are not swept by this script.
CONFIDENCE_THRESHOLD = 0.15
PER_TILE_IOU_THRESHOLD = 0.7          # Ultralytics' own per-tile NMS
CROSS_TILE_IOU_THRESHOLD = 0.7        # current TiledDetector default
TILE_SIZE = (1280, 1280)
TILE_OVERLAP_RATIO = 0.1

SWEEP_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

# Classification thresholds -- diagnostic-only constants, not pipeline config.
DUPLICATE_IOU_MIN = 0.3          # candidate "same person, two boxes" at pre-merge stage
NESTED_CONTAINMENT_MIN = 0.7     # >=70% of the smaller box sits inside the larger one
NESTED_IOU_MAX = 0.5             # ... while IoU stays low enough that NMS wouldn't catch it
ASPECT_RATIO_LOW = 0.7            # h/w below this is a non-person-shape candidate
ASPECT_RATIO_HIGH = 5.5           # h/w above this is a non-person-shape candidate

OUT_DIR = Path("outputs/nms_diagnosis")

# height_px = m*y_bottom + c, fit in docs/REAL_FOOTAGE_FINDINGS.md C2.4.
# R^2=0.464 -- a genuinely weak fit, used here ONLY as a rough seam-artefact
# heuristic (a box far shorter than this predicts, near a tile seam, is a
# candidate worth a human look), never as a precision measurement.
HEIGHT_FIT_M = 0.24215
HEIGHT_FIT_C = -74.852


def grab_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return frame


def tile_offsets(width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    overlap_wh = (round(TILE_SIZE[0] * TILE_OVERLAP_RATIO), round(TILE_SIZE[1] * TILE_OVERLAP_RATIO))
    offsets = InferenceSlicer._generate_offset(
        resolution_wh=(width, height), slice_wh=TILE_SIZE, overlap_wh=overlap_wh
    )
    return offsets, overlap_wh


@dataclass
class TileDetections:
    xyxy: np.ndarray
    confidence: np.ndarray
    tile_index: np.ndarray
    touches_tile_edge: np.ndarray   # True if the box was clipped by ITS OWN tile's crop
    per_tile_raw_count: list[int] = field(default_factory=list)
    per_tile_nms_count: list[int] = field(default_factory=list)


EDGE_TOUCH_TOLERANCE_PX = 3  # sub-pixel/rounding slack for "touches the crop boundary"


def detect_per_tile(detector_normal: Detector, detector_raw: Detector,
                     frame: np.ndarray, offsets: np.ndarray) -> TileDetections:
    """Runs BOTH a normal (iou=0.7, real pipeline config) and a near-NMS-
    disabled (iou=1.0) detector over every tile, so Step 2 can report
    genuine before/after counts for the per-tile Ultralytics NMS stage.
    Only detector_normal's output feeds classification/crops -- detector_raw
    is for the Step 2 count comparison only.

    touches_tile_edge is computed in TILE-LOCAL coordinates, before the
    full-frame offset is added: True means this box's edge sits within
    EDGE_TOUCH_TOLERANCE_PX of that tile's own crop boundary -- i.e. the
    detector's box was literally cut off by the tile crop, the direct
    signature of a seam artefact (vs. classify_seam_candidates' earlier,
    much looser "near some global seam line" heuristic)."""
    all_xyxy, all_conf, all_tile_idx, all_touches_edge = [], [], [], []
    per_tile_raw, per_tile_nms = [], []

    for tile_i, (x1, y1, x2, y2) in enumerate(offsets):
        crop = frame[y1:y2, x1:x2]
        tile_w, tile_h = x2 - x1, y2 - y1

        raw_det = detector_raw.detect(crop)
        per_tile_raw.append(len(raw_det))

        det = detector_normal.detect(crop)
        per_tile_nms.append(len(det))

        if len(det) == 0:
            continue
        local_xyxy = det.xyxy.copy()
        touches = (
            (local_xyxy[:, 0] <= EDGE_TOUCH_TOLERANCE_PX)
            | (local_xyxy[:, 1] <= EDGE_TOUCH_TOLERANCE_PX)
            | (local_xyxy[:, 2] >= tile_w - EDGE_TOUCH_TOLERANCE_PX)
            | (local_xyxy[:, 3] >= tile_h - EDGE_TOUCH_TOLERANCE_PX)
        )
        xyxy = local_xyxy.copy()
        xyxy[:, [0, 2]] += x1
        xyxy[:, [1, 3]] += y1
        all_xyxy.append(xyxy)
        all_conf.append(det.confidence)
        all_tile_idx.extend([tile_i] * len(det))
        all_touches_edge.append(touches)

    if all_xyxy:
        xyxy = np.concatenate(all_xyxy, axis=0)
        confidence = np.concatenate(all_conf, axis=0)
        tile_index = np.array(all_tile_idx, dtype=int)
        touches_tile_edge = np.concatenate(all_touches_edge, axis=0)
    else:
        xyxy = np.zeros((0, 4))
        confidence = np.zeros((0,))
        tile_index = np.zeros((0,), dtype=int)
        touches_tile_edge = np.zeros((0,), dtype=bool)

    return TileDetections(xyxy, confidence, tile_index, touches_tile_edge, per_tile_raw, per_tile_nms)


def iou_matrix(xyxy: np.ndarray) -> np.ndarray:
    if len(xyxy) == 0:
        return np.zeros((0, 0))
    return box_iou_batch(xyxy, xyxy)


def containment_matrix(xyxy: np.ndarray) -> np.ndarray:
    """containment[i, j] = fraction of box i's area that lies inside box j.
    Not symmetric -- containment[i, j] != containment[j, i] in general."""
    n = len(xyxy)
    if n == 0:
        return np.zeros((0, 0))
    x1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
    y1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    x2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    y2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_i = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = inter / area_i[:, None]
    return np.nan_to_num(ratio)


def cross_tile_merge(pre_merge: TileDetections, iou_threshold: float) -> np.ndarray:
    """Returns a boolean keep-mask over pre_merge's detections after the
    SAME NMS the real TiledDetector applies (sv.Detections.with_nms ->
    box_non_max_suppression), at the given IoU threshold. Class-agnostic
    is irrelevant here since every detection is already person-only."""
    if len(pre_merge.xyxy) == 0:
        return np.zeros((0,), dtype=bool)
    predictions = np.concatenate([pre_merge.xyxy, pre_merge.confidence[:, None]], axis=1)
    return box_non_max_suppression(predictions, iou_threshold=iou_threshold)


def expected_height_px(y_bottom: float) -> float:
    return HEIGHT_FIT_M * y_bottom + HEIGHT_FIT_C


def classify_seam_candidates(final_xyxy: np.ndarray, final_touches_edge: np.ndarray,
                              frame_w: int, frame_h: int) -> list[dict]:
    """A seam candidate is a FINAL surviving detection whose box, in its
    OWN source tile's local coordinates, touched that tile's crop boundary
    (see detect_per_tile's touches_tile_edge) -- i.e. the detector's box
    was literally clipped by the tile crop and nothing during cross-tile
    merge replaced it with a complete version. Frame edges are excluded
    (a person genuinely exiting the frame at x=0 isn't a tiling artefact).
    expected_height_px is reported as context only (R^2=0.464 -- not a
    filter), so a reviewer can see whether the box also LOOKS partial."""
    candidates = []
    for i, ((x1, y1, x2, y2), touches) in enumerate(zip(final_xyxy, final_touches_edge)):
        if not touches:
            continue
        at_frame_edge = x1 <= 1 or y1 <= 1 or x2 >= frame_w - 1 or y2 >= frame_h - 1
        if at_frame_edge:
            continue
        expected = expected_height_px(y2)
        height_px = y2 - y1
        candidates.append({
            "index": i,
            "height_px": float(height_px),
            "expected_height_px": float(expected),
            "height_ratio_vs_expected": float(height_px / expected) if expected > 0 else None,
        })
    return candidates


def classify_shape_outliers(xyxy: np.ndarray) -> list[int]:
    candidates = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        w, h = x2 - x1, y2 - y1
        if w <= 0:
            continue
        ratio = h / w
        if ratio < ASPECT_RATIO_LOW or ratio > ASPECT_RATIO_HIGH:
            candidates.append(i)
    return candidates


def save_crop(frame: np.ndarray, boxes: list[tuple[np.ndarray, tuple[int, int, int], str]],
              path: Path, pad: int = 60) -> None:
    all_xyxy = np.array([b[0] for b in boxes])
    x1 = max(0, int(all_xyxy[:, 0].min()) - pad)
    y1 = max(0, int(all_xyxy[:, 1].min()) - pad)
    x2 = min(frame.shape[1], int(all_xyxy[:, 2].max()) + pad)
    y2 = min(frame.shape[0], int(all_xyxy[:, 3].max()) + pad)
    crop = frame[y1:y2, x1:x2].copy()
    for box, color, label in boxes:
        bx1, by1, bx2, by2 = (box - [x1, y1, x1, y1]).astype(int)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)
        cv2.putText(crop, label, (bx1, max(12, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


def save_overview(frame: np.ndarray, xyxy: np.ndarray, path: Path, seam_lines: np.ndarray) -> None:
    img = frame.copy()
    xs = sorted({int(o[0]) for o in seam_lines} | {int(o[2]) for o in seam_lines})
    ys = sorted({int(o[1]) for o in seam_lines} | {int(o[3]) for o in seam_lines})
    for x in xs:
        if 0 < x < img.shape[1]:
            cv2.line(img, (x, 0), (x, img.shape[0]), (255, 220, 0), 1, cv2.LINE_AA)
    for y in ys:
        if 0 < y < img.shape[0]:
            cv2.line(img, (0, y), (img.shape[1], y), (255, 220, 0), 1, cv2.LINE_AA)
    for i, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(img, str(i), (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def analyze_frame(frame_idx: int, detector_normal: Detector, detector_raw: Detector,
                   real_tiled, out_dir: Path) -> dict:
    frame = grab_frame(VIDEO_PATH, frame_idx)
    h, w = frame.shape[:2]
    offsets, overlap_wh = tile_offsets(w, h)

    pre_merge = detect_per_tile(detector_normal, detector_raw, frame, offsets)

    # Sanity check against the REAL TiledDetector class -- confirms this
    # script's manual tiling reproduces the actual pipeline, not a
    # subtly-different reimplementation.
    real_final = real_tiled.detect(frame)

    keep_current = cross_tile_merge(pre_merge, CROSS_TILE_IOU_THRESHOLD)
    final_xyxy = pre_merge.xyxy[keep_current]
    final_conf = pre_merge.confidence[keep_current]
    final_touches_edge = pre_merge.touches_tile_edge[keep_current]
    final_tile_index = pre_merge.tile_index[keep_current]

    parity_ok = len(final_xyxy) == len(real_final)

    frame_dir = out_dir / f"frame{frame_idx:04d}"
    save_overview(frame, final_xyxy, frame_dir / "overview.jpg", offsets)

    # ---- (a) DUPLICATE: pre-merge pairs with IoU >= threshold, report
    # whether both survived the CURRENT cross-tile NMS ----
    pre_iou = iou_matrix(pre_merge.xyxy)
    n_pre = len(pre_merge.xyxy)
    duplicate_pairs = []
    seen = set()
    for i in range(n_pre):
        for j in range(i + 1, n_pre):
            if pre_iou[i, j] >= DUPLICATE_IOU_MIN:
                pair = (i, j)
                if pair in seen:
                    continue
                seen.add(pair)
                both_survived = bool(keep_current[i] and keep_current[j])
                duplicate_pairs.append({
                    "i": int(i), "j": int(j), "iou": float(pre_iou[i, j]),
                    "tile_i": int(pre_merge.tile_index[i]), "tile_j": int(pre_merge.tile_index[j]),
                    "both_survived_current_nms": both_survived,
                })
                if both_survived:
                    save_crop(frame, [
                        (pre_merge.xyxy[i], (0, 220, 0), f"#{i} conf={pre_merge.confidence[i]:.2f}"),
                        (pre_merge.xyxy[j], (0, 140, 255), f"#{j} conf={pre_merge.confidence[j]:.2f}"),
                    ], frame_dir / f"case_a_duplicate_{i}_{j}.jpg")

    # ---- (b) NESTED: among FINAL survivors, high containment + low IoU ----
    final_containment = containment_matrix(final_xyxy)
    final_iou = iou_matrix(final_xyxy)
    n_final = len(final_xyxy)
    nested_pairs = []
    seen_nested = set()
    for i in range(n_final):
        for j in range(n_final):
            if i == j:
                continue
            pair = tuple(sorted((i, j)))
            if pair in seen_nested:
                continue
            c = final_containment[i, j]  # fraction of i inside j
            if c >= NESTED_CONTAINMENT_MIN and final_iou[i, j] <= NESTED_IOU_MAX:
                seen_nested.add(pair)
                nested_pairs.append({
                    "inner": int(i), "outer": int(j),
                    "containment": float(c), "iou": float(final_iou[i, j]),
                })
                save_crop(frame, [
                    (final_xyxy[j], (255, 180, 0), f"#{j} outer"),
                    (final_xyxy[i], (0, 140, 255), f"#{i} inner c={c:.2f}"),
                ], frame_dir / f"case_b_nested_{i}_in_{j}.jpg")

    # ---- (c) SEAM ARTEFACT: final survivors whose box was clipped by their
    # own source tile's crop boundary (touches_tile_edge, tile-local) ----
    seam_candidates = classify_seam_candidates(final_xyxy, final_touches_edge, w, h)
    for cand in seam_candidates:
        i = cand["index"]
        save_crop(frame, [(final_xyxy[i], (0, 60, 255),
                            f"#{i} seam conf={final_conf[i]:.2f} tile={final_tile_index[i]}")],
                   frame_dir / f"case_c_seam_{i}.jpg", pad=100)

    # ---- (d) NON-PERSON candidates: shape outliers + everything not yet
    # explained by (a)/(b)/(c), for visual review ----
    explained = {c["index"] for c in seam_candidates}
    for p in nested_pairs:
        explained.add(p["inner"])
        explained.add(p["outer"])
    shape_outliers = [i for i in classify_shape_outliers(final_xyxy) if i not in explained]
    for i in shape_outliers:
        save_crop(frame, [(final_xyxy[i], (255, 0, 255), f"#{i} shape? conf={final_conf[i]:.2f}")],
                   frame_dir / f"case_d_shape_outlier_{i}.jpg", pad=80)

    return {
        "frame_idx": frame_idx,
        "resolution": [w, h],
        "n_tiles": len(offsets),
        "stage_counts": {
            "before_per_tile_nms_total": int(sum(pre_merge.per_tile_raw_count)),
            "after_per_tile_nms_before_cross_tile": int(sum(pre_merge.per_tile_nms_count)),
            "after_cross_tile_nms_current": int(len(final_xyxy)),
            "per_tile_raw_counts": pre_merge.per_tile_raw_count,
            "per_tile_nms_counts": pre_merge.per_tile_nms_count,
        },
        "real_tiled_detector_parity": {
            "matches_manual_reimplementation_count": parity_ok,
            "real_tiled_detector_final_count": int(len(real_final)),
            "manual_final_count": int(len(final_xyxy)),
        },
        "duplicate_candidates": duplicate_pairs,
        "duplicate_both_survived_count": sum(1 for p in duplicate_pairs if p["both_survived_current_nms"]),
        "duplicate_both_survived_same_tile": sum(
            1 for p in duplicate_pairs if p["both_survived_current_nms"] and p["tile_i"] == p["tile_j"]
        ),
        "duplicate_both_survived_cross_tile": sum(
            1 for p in duplicate_pairs if p["both_survived_current_nms"] and p["tile_i"] != p["tile_j"]
        ),
        "nested_candidates": nested_pairs,
        "seam_candidates": seam_candidates,
        "shape_outlier_indices": shape_outliers,
        "final_confidences": final_conf.tolist(),
        "final_xyxy": final_xyxy.tolist(),
        "pre_merge_xyxy": pre_merge.xyxy.tolist(),
        "pre_merge_confidence": pre_merge.confidence.tolist(),
        "pre_merge_tile_index": pre_merge.tile_index.tolist(),
        "tile_offsets": offsets.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD,
                         help=f"detector confidence threshold (default: {CONFIDENCE_THRESHOLD}, the live pipeline's own default)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                         help=f"output directory for crops/report.json (default: {OUT_DIR})")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_normal = Detector(MODEL_PATH, device="auto", confidence_threshold=args.confidence_threshold,
                                iou_threshold=PER_TILE_IOU_THRESHOLD, person_class_id=0)
    detector_raw = Detector(MODEL_PATH, device="auto", confidence_threshold=args.confidence_threshold,
                             iou_threshold=1.0, person_class_id=0)

    from perception.detector import TiledDetector, TileConfig
    tile_config = TileConfig(tile_size=TILE_SIZE, overlap_ratio=TILE_OVERLAP_RATIO,
                              nms_iou_threshold=CROSS_TILE_IOU_THRESHOLD)
    real_tiled = TiledDetector(detector_normal, tile_config=tile_config, enabled=True)

    report = {"confidence_threshold": args.confidence_threshold, "frames": []}
    for frame_idx in FRAME_INDICES:
        print(f"[diagnose] analyzing frame {frame_idx} (confidence_threshold={args.confidence_threshold}) ...")
        result = analyze_frame(frame_idx, detector_normal, detector_raw, real_tiled, out_dir)
        report["frames"].append(result)
        sc = result["stage_counts"]
        print(f"[diagnose] frame {frame_idx}: before_per_tile_nms={sc['before_per_tile_nms_total']} "
              f"after_per_tile_nms={sc['after_per_tile_nms_before_cross_tile']} "
              f"after_cross_tile_nms={sc['after_cross_tile_nms_current']} "
              f"parity_ok={result['real_tiled_detector_parity']['matches_manual_reimplementation_count']} "
              f"duplicates_both_survived={result['duplicate_both_survived_count']} "
              f"(same_tile={result['duplicate_both_survived_same_tile']} "
              f"cross_tile={result['duplicate_both_survived_cross_tile']}) "
              f"nested={len(result['nested_candidates'])} "
              f"seam_candidates={len(result['seam_candidates'])} "
              f"shape_outliers={len(result['shape_outlier_indices'])}")

    with (out_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[diagnose] wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
