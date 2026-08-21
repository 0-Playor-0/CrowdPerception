"""Step 5b: checks whether specific NON-PERSON candidate detections
(identified by visual review of scripts/diagnose_tiled_nms.py's output)
persist at a fixed image location across the whole clip -- the signature
of a static object (mannequin, poster) vs a moving real person. Read-only,
reports counts only; does NOT implement a persistence filter.

Runs single-tile inference (just the one tile each candidate falls in,
not the full 8-tile pass) at a fixed sampling stride across all 540
frames -- cheap, since it reuses the same Detector instance and only
needs to know "is there still a high-IoU detection near this fixed box"
per sample.

Usage:
    uv run python scripts/check_static_persistence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from supervision.detection.utils.iou_and_nms import box_iou_batch

from perception.detector import Detector

VIDEO_PATH = Path("data/127690-739144743.mp4")
MODEL_PATH = "models/yolo11s.pt"
CONFIDENCE_THRESHOLD = 0.02
PER_TILE_IOU_THRESHOLD = 0.7
SAMPLE_STRIDE = 10  # every 10th frame -> 54 samples over 540 frames
IOU_MATCH_THRESHOLD = 0.4  # "still roughly the same box" at this location

# (tile crop x1,y1,x2,y2), (candidate box in FULL-FRAME coords), label
CANDIDATES = [
    ((0, 0, 1280, 1280), (172.2, 202.8, 273.6, 516.4), "frame0 idx0 poster/backdrop conf=0.54"),
    ((0, 0, 1280, 1280), (91.6, 300.2, 116.4, 369.6), "frame0 idx1 poster (small) conf=0.07"),
]


def main() -> None:
    detector = Detector(MODEL_PATH, device="auto", confidence_threshold=CONFIDENCE_THRESHOLD,
                         iou_threshold=PER_TILE_IOU_THRESHOLD, person_class_id=0)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = list(range(0, total_frames, SAMPLE_STRIDE))

    hit_counts = {label: 0 for _, _, label in CANDIDATES}
    best_iou_per_sample = {label: [] for _, _, label in CANDIDATES}

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        for (tx1, ty1, tx2, ty2), (cx1, cy1, cx2, cy2), label in CANDIDATES:
            crop = frame[ty1:ty2, tx1:tx2]
            det = detector.detect(crop)
            if len(det) == 0:
                best_iou_per_sample[label].append(0.0)
                continue
            local_target = np.array([[cx1 - tx1, cy1 - ty1, cx2 - tx1, cy2 - ty1]])
            ious = box_iou_batch(local_target, det.xyxy)[0]
            best_iou = float(ious.max())
            best_iou_per_sample[label].append(best_iou)
            if best_iou >= IOU_MATCH_THRESHOLD:
                hit_counts[label] += 1

    cap.release()

    n_samples = len(sample_indices)
    print(f"[persistence] {n_samples} samples (every {SAMPLE_STRIDE}th frame of {total_frames})")
    for _, _, label in CANDIDATES:
        hits = hit_counts[label]
        ious = best_iou_per_sample[label]
        print(f"[persistence] {label}: matched in {hits}/{n_samples} samples "
              f"({100 * hits / n_samples:.0f}%)  mean best-IoU={np.mean(ious):.2f}  "
              f"median best-IoU={np.median(ious):.2f}")


if __name__ == "__main__":
    main()
