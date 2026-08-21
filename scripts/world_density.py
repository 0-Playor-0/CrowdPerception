"""C2.5 -- Full record stream (world coordinates) + persons/m^2 over time,
using the height-ESTIMATED calibration from C2.4 (see docs -- NOT measured).

Reuses perception/detector.py, perception/tracker.py, perception/geometry.py,
perception/velocity.py directly, independent of scripts/live_perception.py's
full pipeline.

docs/REAL_FOOTAGE_FINDINGS.md's C2.5 findings were produced by this script
at --confidence-threshold 0.02 (the whole project's default at the time).
That default has since moved to 0.05 to keep exactly 3 confidence values
in use anywhere in the project (see server/pipeline_runner.py's
RANGE_PRESETS) -- pass --confidence-threshold 0.02 explicitly to reproduce
those exact historical figures; the current default will not.

Density is computed directly here (persons in the calibrated TARGET quad /
quad area, and per 1m grid cell for the spatial peak) -- NOT analytics/pressure.py
(not carried into this project; Task 2 is perception-only, no analytics/
thresholds rebuild).

Usage:
    uv run python scripts/world_density.py --video data/127690-739144743.mp4 \\
        --model models/yolo11s.pt --tile --cell-size-m 1.0 \\
        --stats-out output/world_density_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import supervision as sv

from perception.detector import Detector
from perception.geometry import ViewTransformer, polygon_area
from perception.tracker import Tracker
from perception.velocity import VelocityTracker
from scripts.detect_track import build_slicer

# Height-estimated calibration from C2.4 -- see docs/REAL_FOOTAGE_FINDINGS.md.
# Fit: height_px = 0.24215*y + -74.852 (R^2=0.464, n=478, residual std 34% of
# mean height -- SUBSTANTIAL scatter, treat downstream numbers as rough).
SOURCE_PX = np.array([
    [1250.0, 2000.0],   # near-left
    [3100.0, 2000.0],   # near-right
    [2450.0, 1050.0],   # far-right
    [1750.0, 1050.0],   # far-left
])
TARGET_M = np.array([
    [-3.84, 0.0],
    [3.84, 0.0],
    [3.31, 5.79],
    [-3.31, 5.79],
])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", default="models/yolo11s.pt")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)   # "long" range preset -- this script's default subject (data/127690-739144743.mp4) is a long-range elevated crowd shot; see server/pipeline_runner.py's RANGE_PRESETS, the only 3 confidence values used anywhere in this project
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--cell-size-m", type=float, default=1.0)
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--max-speed-mps", type=float, default=10.0)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--records-out", type=Path, default=None)
    args = parser.parse_args()

    detector = Detector(model_path=args.model, device="auto", confidence_threshold=args.confidence_threshold,
                         iou_threshold=0.7, person_class_id=0)
    slicer = build_slicer(detector, (1280, 1280), (128, 128), 0.7) if args.tile else None

    video_info = sv.VideoInfo.from_video_path(str(args.video))
    fps = video_info.fps
    window_frames = max(2, round(fps * args.window_seconds))

    view_transformer = ViewTransformer(SOURCE_PX, TARGET_M)
    quad_area_m2 = polygon_area(TARGET_M)
    tracker = Tracker(frame_rate=fps)
    velocity_tracker = VelocityTracker(window_frames, args.max_speed_mps)

    print(f"[world_density] TARGET quad area: {quad_area_m2:.2f} m^2  (height-ESTIMATED calibration, not measured)")
    print(f"[world_density] video={args.video}  tiled={args.tile}  cell_size_m={args.cell_size_m}")

    records = []
    density_over_time = []
    cell_counts_alltime: dict[tuple[int, int], int] = defaultdict(int)

    frame_generator = sv.get_video_frames_generator(source_path=str(args.video))
    for frame_id, frame in enumerate(frame_generator):
        t_sec = frame_id / fps
        detections = slicer(frame) if slicer is not None else detector.detect(frame)
        tracked = tracker.update(detections)
        if tracked.tracker_id is not None and len(tracked) > 0:
            confirmed = tracked[tracked.tracker_id >= 0]
        else:
            confirmed = tracked

        if len(confirmed) == 0:
            density_over_time.append({"frame_id": frame_id, "t_sec": t_sec, "in_quad": 0, "density_persons_per_m2": 0.0})
            continue

        anchors_px = confirmed.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        world_xy = view_transformer.transform_points(anchors_px)

        in_quad_count = 0
        cell_counts_this_frame: dict[tuple[int, int], int] = defaultdict(int)
        for i, track_id in enumerate(confirmed.tracker_id):
            x_world, y_world = world_xy[i]
            estimate = velocity_tracker.update(int(track_id), t_sec, float(x_world), float(y_world))
            if estimate is None:
                continue
            records.append({
                "frame_id": frame_id, "t_sec": t_sec, "track_id": int(track_id),
                "x_world": estimate.x_world, "y_world": estimate.y_world,
                "v_x": estimate.v_x, "v_y": estimate.v_y, "window_full": estimate.window_full,
            })
            # "in quad" -- inside the calibrated TARGET rectangle's rough x/y bounds
            xmin, xmax = TARGET_M[:, 0].min(), TARGET_M[:, 0].max()
            ymin, ymax = TARGET_M[:, 1].min(), TARGET_M[:, 1].max()
            if xmin <= x_world <= xmax and ymin <= y_world <= ymax:
                in_quad_count += 1
                ci, cj = int(np.floor((x_world - xmin) / args.cell_size_m)), int(np.floor((y_world - ymin) / args.cell_size_m))
                cell_counts_this_frame[(ci, cj)] += 1
                cell_counts_alltime[(ci, cj)] += 1

        peak_cell_density = 0.0
        if cell_counts_this_frame:
            peak_cell_count = max(cell_counts_this_frame.values())
            peak_cell_density = peak_cell_count / (args.cell_size_m ** 2)

        density_over_time.append({
            "frame_id": frame_id, "t_sec": t_sec, "in_quad": in_quad_count,
            "density_persons_per_m2": in_quad_count / quad_area_m2,
            "peak_cell_density_persons_per_m2": peak_cell_density,
        })

        if frame_id % 60 == 0:
            print(f"[world_density] frame {frame_id}/{video_info.total_frames}  in_quad={in_quad_count}  "
                  f"density={in_quad_count/quad_area_m2:.2f} p/m^2")

    densities = np.array([d["density_persons_per_m2"] for d in density_over_time])
    peak_cell_densities = np.array([d.get("peak_cell_density_persons_per_m2", 0.0) for d in density_over_time])
    peak_idx = int(np.argmax(densities))
    peak_cell_idx = int(np.argmax(peak_cell_densities))

    busiest_cell = max(cell_counts_alltime.items(), key=lambda kv: kv[1]) if cell_counts_alltime else None

    summary = {
        "quad_area_m2": quad_area_m2,
        "n_frames": len(density_over_time),
        "n_records": len(records),
        "mean_quad_density_persons_per_m2": float(densities.mean()),
        "peak_quad_density_persons_per_m2": float(densities.max()),
        "peak_quad_density_frame_id": density_over_time[peak_idx]["frame_id"],
        "peak_quad_density_t_sec": density_over_time[peak_idx]["t_sec"],
        "peak_cell_density_persons_per_m2": float(peak_cell_densities.max()),
        "peak_cell_density_frame_id": density_over_time[peak_cell_idx]["frame_id"],
        "busiest_cell_ij_alltime_count": busiest_cell,
        "calibration": "HEIGHT-ESTIMATED, NOT MEASURED -- see docs/REAL_FOOTAGE_FINDINGS.md C2.4",
    }
    print("\n[world_density] ==== summary ====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        with args.stats_out.open("w") as f:
            json.dump({"summary": summary, "density_over_time": density_over_time}, f, indent=2)
        print(f"[world_density] wrote {args.stats_out}")
    if args.records_out:
        args.records_out.parent.mkdir(parents=True, exist_ok=True)
        with args.records_out.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[world_density] wrote {args.records_out}")


if __name__ == "__main__":
    main()
