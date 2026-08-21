"""Step 3 of the false-positive diagnosis: sweeps the cross-tile merge IoU
threshold (0.3/0.5/0.7-current/0.9), and separately tests non-max MERGE in
place of suppression, WITHOUT re-running the detector -- it reuses the
pre-merge per-tile detections cached in outputs/nms_diagnosis/report.json
by scripts/diagnose_tiled_nms.py (pure array ops on already-computed boxes).

For NMS (suppression), "two different people merged" doesn't apply --
suppression only ever deletes a box, never combines geometry. What matters
there is the inverse: did raising suppression aggressiveness (lower IoU
threshold) delete a box that was a genuinely DIFFERENT person standing
close to a higher-confidence detection, not a duplicate of it? This script
flags candidate cases (suppressed box whose confidence is not much lower
than its suppressor's, AND whose center is far enough away to plausibly be
a different person) for visual review -- it cannot make that call by
geometry alone.

For NMM (merge), a group of >=2 boxes merged together where the ORIGINAL
box centers are far apart (relative to typical person-box size) is a
direct, checkable signature of "different people merged into one box".

Usage:
    uv run python scripts/sweep_cross_tile_nms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from supervision.detection.utils.iou_and_nms import box_iou_batch, box_non_max_suppression

REPORT_PATH = Path("outputs/nms_diagnosis/report.json")
OUT_DIR = Path("outputs/nms_diagnosis")
VIDEO_PATH = Path("data/127690-739144743.mp4")

SWEEP_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

# A merged group is a "different people merged" candidate if any two
# member boxes' centers are more than this fraction of the SMALLER box's
# own diagonal apart -- i.e. the merge grew the box well beyond what a
# single person's pose/motion-blur would explain. Diagnostic-only.
MERGE_CENTER_DISTANCE_RATIO = 0.6

# A suppressed-in-sweep box is a "possibly different person" candidate if
# its confidence is within this fraction of its suppressor's confidence
# (a big confidence gap is more consistent with "same person, one blurrier
# duplicate box" than with two distinct, separately-detected people).
SUPPRESSED_CONFIDENCE_RATIO_MIN = 0.5


def grab_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index}")
    return frame


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


def sweep_nms(xyxy: np.ndarray, confidence: np.ndarray, frame: np.ndarray, frame_dir: Path) -> dict:
    predictions = np.concatenate([xyxy, confidence[:, None]], axis=1)
    ious = box_iou_batch(xyxy, xyxy)
    results = {}
    for t in SWEEP_THRESHOLDS:
        keep = box_non_max_suppression(predictions, iou_threshold=t)
        n_kept = int(keep.sum())

        # Duplicates remaining: kept boxes that STILL have IoU >= 0.3 with
        # another kept box (i.e. the sweep didn't fully clean them up).
        kept_idx = np.where(keep)[0]
        remaining_dupes = 0
        for a in range(len(kept_idx)):
            for b in range(a + 1, len(kept_idx)):
                if ious[kept_idx[a], kept_idx[b]] >= 0.3:
                    remaining_dupes += 1

        # Different-person-suppressed candidates: for each suppressed box,
        # find its highest-IoU (>= t, since that's why it was suppressed)
        # surviving neighbour; flag if confidence ratio is high (not an
        # obvious "duplicate, lower-confidence copy" case).
        suppressed_idx = np.where(~keep)[0]
        candidates = []
        for s in suppressed_idx:
            neighbour_ious = ious[s].copy()
            neighbour_ious[~keep] = -1  # only consider surviving neighbours
            best = int(np.argmax(neighbour_ious))
            if neighbour_ious[best] < t:
                continue  # not actually why it was suppressed
            conf_ratio = confidence[s] / confidence[best] if confidence[best] > 0 else 0
            if conf_ratio >= SUPPRESSED_CONFIDENCE_RATIO_MIN:
                candidates.append({
                    "suppressed_index": int(s), "survivor_index": best,
                    "iou": float(neighbour_ious[best]),
                    "suppressed_confidence": float(confidence[s]),
                    "survivor_confidence": float(confidence[best]),
                })

        for c in candidates[:5]:  # cap crops per threshold to keep this reviewable
            save_crop(frame, [
                (xyxy[c["survivor_index"]], (0, 220, 0), f"kept conf={c['survivor_confidence']:.2f}"),
                (xyxy[c["suppressed_index"]], (0, 0, 255), f"suppressed conf={c['suppressed_confidence']:.2f}"),
            ], frame_dir / f"sweep_nms_t{t}_suppressed_{c['suppressed_index']}.jpg")

        results[str(t)] = {
            "total_detections": n_kept,
            "duplicates_remaining": remaining_dupes,
            "possibly_different_person_suppressed_count": len(candidates),
            "possibly_different_person_suppressed": candidates,
        }
    return results


def sweep_nmm(xyxy: np.ndarray, confidence: np.ndarray, frame: np.ndarray, frame_dir: Path) -> dict:
    predictions = np.concatenate([xyxy, confidence[:, None]], axis=1)
    results = {}
    for t in SWEEP_THRESHOLDS:
        groups = sv.box_non_max_merge(predictions, iou_threshold=t)
        n_merged_boxes = len(groups)

        different_person_candidates = []
        for group in groups:
            if len(group) < 2:
                continue
            boxes = xyxy[group]
            centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
            diag = np.sqrt((boxes[:, 2] - boxes[:, 0]) ** 2 + (boxes[:, 3] - boxes[:, 1]) ** 2)
            min_diag = float(diag.min())
            max_center_dist = float(np.max(
                np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1))
            ))
            if min_diag > 0 and max_center_dist > MERGE_CENTER_DISTANCE_RATIO * min_diag:
                union_x1 = float(boxes[:, 0].min())
                union_y1 = float(boxes[:, 1].min())
                union_x2 = float(boxes[:, 2].max())
                union_y2 = float(boxes[:, 3].max())
                different_person_candidates.append({
                    "group_indices": [int(g) for g in group],
                    "max_center_distance_px": max_center_dist,
                    "min_member_diagonal_px": min_diag,
                    "union_box": [union_x1, union_y1, union_x2, union_y2],
                })
                colors = [(0, 220, 0), (0, 140, 255), (255, 0, 255), (0, 220, 220)]
                boxes_to_draw = [(xyxy[g], colors[i % len(colors)], f"#{g}") for i, g in enumerate(group)]
                save_crop(frame, boxes_to_draw,
                           frame_dir / f"sweep_nmm_t{t}_group_{'_'.join(str(g) for g in group)}.jpg")

        results[str(t)] = {
            "total_boxes_after_merge": n_merged_boxes,
            "groups_with_multiple_members": sum(1 for g in groups if len(g) > 1),
            "possibly_different_person_merged_count": len(different_person_candidates),
            "possibly_different_person_merged": different_person_candidates,
        }
    return results


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    sweep_report = {"frames": []}

    for fr in report["frames"]:
        frame_idx = fr["frame_idx"]
        print(f"[sweep] frame {frame_idx} ...")
        xyxy = np.array(fr["pre_merge_xyxy"])
        confidence = np.array(fr["pre_merge_confidence"])
        frame = grab_frame(VIDEO_PATH, frame_idx)
        frame_dir = OUT_DIR / f"frame{frame_idx:04d}"

        nms_results = sweep_nms(xyxy, confidence, frame, frame_dir)
        nmm_results = sweep_nmm(xyxy, confidence, frame, frame_dir)

        for t in SWEEP_THRESHOLDS:
            n = nms_results[str(t)]
            m = nmm_results[str(t)]
            print(f"[sweep]   iou={t}: NMS total={n['total_detections']} "
                  f"dup_remaining={n['duplicates_remaining']} "
                  f"different_person_suppressed={n['possibly_different_person_suppressed_count']}  |  "
                  f"NMM total={m['total_boxes_after_merge']} "
                  f"multi_member_groups={m['groups_with_multiple_members']} "
                  f"different_person_merged={m['possibly_different_person_merged_count']}")

        sweep_report["frames"].append({
            "frame_idx": frame_idx,
            "pre_merge_total": len(xyxy),
            "nms_sweep": nms_results,
            "nmm_sweep": nmm_results,
        })

    with (OUT_DIR / "sweep_report.json").open("w") as f:
        json.dump(sweep_report, f, indent=2)
    print(f"[sweep] wrote {OUT_DIR / 'sweep_report.json'}")


if __name__ == "__main__":
    main()
