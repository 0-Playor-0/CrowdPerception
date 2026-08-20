"""C2.2 -- Detection + tracking on real footage, PIXEL SPACE ONLY, no
homography/calibration. Runs perception/detector.py + perception/tracker.py
directly, independent of scripts/live_perception.py's full pipeline --
kept as-is because docs/REAL_FOOTAGE_FINDINGS.md's C2.2/C2.3 findings cite
this exact script and are meant to stay reproducible from it.

Optional tiled inference (C2.3): supervision's InferenceSlicer, config-
toggled via --tile, default off. Slices the frame into overlapping tiles,
runs the detector on each, merges results (NMS across tile boundaries).

Reports (printed + written to --stats-out):
  - detections per frame (raw YOLO output, before tracker confirmation)
  - confidence distribution
  - measured fps (detect+track, excludes video I/O and annotation)
  - unique CONFIRMED track IDs seen across the whole video
  - per-frame confirmed-track count

Writes an annotated video (boxes + track IDs) to --annotated-out.

Usage:
    uv run python scripts/detect_track.py --video data/127690-739144743.mp4 \\
        --model models/yolo11s.pt --confidence-threshold 0.3 \\
        --annotated-out output/detect_track_annotated.mp4 \\
        --stats-out output/detect_track_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import supervision as sv

from perception.detector import Detector
from perception.tracker import Tracker


def build_slicer(detector: Detector, slice_wh: tuple[int, int], overlap_wh: tuple[int, int], iou_threshold: float) -> sv.InferenceSlicer:
    def callback(image_slice: np.ndarray) -> sv.Detections:
        return detector.detect(image_slice)

    return sv.InferenceSlicer(
        callback=callback,
        slice_wh=slice_wh,
        overlap_wh=overlap_wh,
        iou_threshold=iou_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", default="models/yolo11s.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.02)   # lowered from 0.3 -> 0.2 -> 0.1 -> 0.02, see scripts/live_perception.py's --confidence-threshold help text
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--annotated-out", type=Path, default=None)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--tile", action="store_true", help="C2.3: use InferenceSlicer tiled inference instead of single-pass")
    parser.add_argument("--tile-size", type=int, nargs=2, default=[1280, 1280], metavar=("W", "H"))
    parser.add_argument("--tile-overlap-px", type=int, nargs=2, default=[128, 128], metavar=("W", "H"))
    args = parser.parse_args()

    detector = Detector(
        model_path=args.model, device=args.device, confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold, person_class_id=args.person_class_id,
    )
    slicer = build_slicer(detector, tuple(args.tile_size), tuple(args.tile_overlap_px), args.iou_threshold) if args.tile else None

    video_info = sv.VideoInfo.from_video_path(str(args.video))
    fps = video_info.fps
    print(f"[detect_track] video={args.video}  {video_info.width}x{video_info.height}  fps={fps}  "
          f"total_frames={video_info.total_frames}  tiled={args.tile}")

    tracker = Tracker(frame_rate=fps)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_position=sv.Position.TOP_LEFT)

    video_sink = None
    if args.annotated_out:
        args.annotated_out.parent.mkdir(parents=True, exist_ok=True)
        video_sink = sv.VideoSink(str(args.annotated_out), video_info)
        video_sink.__enter__()

    per_frame_counts: list[int] = []
    per_frame_confirmed_counts: list[int] = []
    all_confidences: list[float] = []
    confirmed_track_ids: set[int] = set()
    frame_times_sec: list[float] = []

    frame_generator = sv.get_video_frames_generator(source_path=str(args.video))
    n = 0
    for frame in frame_generator:
        t0 = time.perf_counter()
        detections = slicer(frame) if slicer is not None else detector.detect(frame)
        raw_count = len(detections)
        if detections.confidence is not None:
            all_confidences.extend(detections.confidence.tolist())

        tracked = tracker.update(detections)
        elapsed = time.perf_counter() - t0
        frame_times_sec.append(elapsed)

        confirmed = tracked[tracked.tracker_id >= 0] if tracked.tracker_id is not None and len(tracked) > 0 else tracked
        for tid in (confirmed.tracker_id if confirmed.tracker_id is not None else []):
            confirmed_track_ids.add(int(tid))

        per_frame_counts.append(raw_count)
        per_frame_confirmed_counts.append(len(confirmed))

        if video_sink is not None:
            labels = [f"#{tid}" for tid in confirmed.tracker_id] if confirmed.tracker_id is not None else []
            annotated = frame.copy()
            annotated = box_annotator.annotate(scene=annotated, detections=confirmed)
            annotated = label_annotator.annotate(scene=annotated, detections=confirmed, labels=labels)
            video_sink.write_frame(annotated)

        n += 1
        if n % 60 == 0:
            print(f"[detect_track] frame {n}/{video_info.total_frames}  raw_det={raw_count}  confirmed={len(confirmed)}")
        if args.max_frames and n >= args.max_frames:
            break

    if video_sink is not None:
        video_sink.__exit__(None, None, None)

    confidences_arr = np.array(all_confidences)
    measured_fps = n / sum(frame_times_sec) if frame_times_sec else float("nan")

    stats = {
        "video": str(args.video), "n_frames_processed": n, "tiled": args.tile,
        "raw_detections_per_frame": per_frame_counts,
        "confirmed_tracks_per_frame": per_frame_confirmed_counts,
        "raw_det_mean": float(np.mean(per_frame_counts)) if per_frame_counts else 0.0,
        "raw_det_min": int(np.min(per_frame_counts)) if per_frame_counts else 0,
        "raw_det_max": int(np.max(per_frame_counts)) if per_frame_counts else 0,
        "confidence_mean": float(confidences_arr.mean()) if confidences_arr.size else None,
        "confidence_p10": float(np.percentile(confidences_arr, 10)) if confidences_arr.size else None,
        "confidence_p50": float(np.percentile(confidences_arr, 50)) if confidences_arr.size else None,
        "confidence_p90": float(np.percentile(confidences_arr, 90)) if confidences_arr.size else None,
        "unique_confirmed_track_ids": len(confirmed_track_ids),
        "measured_fps_detect_and_track_only": measured_fps,
    }

    print("\n[detect_track] ==== summary ====")
    print(f"  raw detections/frame: mean={stats['raw_det_mean']:.2f}  min={stats['raw_det_min']}  max={stats['raw_det_max']}")
    print(f"  confidence: mean={stats['confidence_mean']}  p10={stats['confidence_p10']}  p50={stats['confidence_p50']}  p90={stats['confidence_p90']}")
    print(f"  unique confirmed track IDs across video: {stats['unique_confirmed_track_ids']}")
    print(f"  measured fps (detect+track only, excludes I/O and annotation): {measured_fps:.2f}")

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        with args.stats_out.open("w") as f:
            json.dump(stats, f, indent=2)
        print(f"[detect_track] wrote {args.stats_out}")

    if video_sink is not None:
        print(f"[detect_track] wrote {args.annotated_out}")


if __name__ == "__main__":
    main()
