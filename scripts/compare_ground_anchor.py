"""A/B comparison: foot-point vs centroid ground-plane anchor, on a full clip.

Not part of the live pipeline -- this is the empirical follow-up to
tests/test_ground.py::test_foot_vs_centroid_disagreement_on_a_real_frame,
which measured 0.88m of disagreement on ONE ordinary detection. This
script extends that to every in-quad detection across a whole clip and
reports the aggregate picture: how big the disagreement is, whether it's
uniform or gets worse for people far from the camera (it should -- see
perception/ground.py's module docstring on why an oblique view makes this
worse with distance), and what it does to density figures if you got the
anchor wrong.

Requires a clip with a VALID (fixed-camera) calibration -- this is a
ground-plane comparison, so it inherits the same camera-fixity requirement
as any other world-coordinate figure in this project. Defaults to
Myeongdong (data/127690-739144743.mp4), the only clip with both a fixed
camera and a real (USER_MEASURED) calibration as of this writing.

Usage:
    uv run python scripts/compare_ground_anchor.py \\
        --video data/127690-739144743.mp4 --calibration calibration/127690-739144743.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import cv2
import numpy as np
import supervision as sv

from perception.detector import Detector, TileConfig, TiledDetector
from perception.ground import GroundProjector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="data/127690-739144743.mp4")
    parser.add_argument("--calibration", default="calibration/127690-739144743.json")
    parser.add_argument("--model", default="models/yolo11s.pt")
    parser.add_argument("--confidence-threshold", type=float, default=0.02)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--tile-size", type=int, nargs=2, default=[1280, 1280])
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--no-tile", action="store_true")
    parser.add_argument("--downscale", dest="downscale_long_edge", type=int, default=1920)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    args = parser.parse_args()

    with open(args.calibration) as f:
        calibration = json.load(f)
    if calibration["source"] not in ("USER_MEASURED",):
        print(f"[compare_ground_anchor] WARNING: calibration source is {calibration['source']!r}, "
              "not USER_MEASURED -- proceeding, but treat the metre-space numbers below accordingly.")
    quad_px = np.array(calibration["image_points"], dtype=np.float64)
    H = np.array(calibration["H"], dtype=np.float64)

    base = Detector(model_path=args.model, device="auto", confidence_threshold=args.confidence_threshold,
                     iou_threshold=args.iou_threshold, person_class_id=0, imgsz=args.imgsz)
    tile_config = TileConfig(tile_size=tuple(args.tile_size), overlap_ratio=args.tile_overlap)
    detector = TiledDetector(base, tile_config=tile_config, enabled=not args.no_tile)

    def downscale(frame: np.ndarray) -> tuple[np.ndarray, float]:
        if args.downscale_long_edge <= 0:
            return frame, 1.0
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side <= args.downscale_long_edge:
            return frame, 1.0
        scale = args.downscale_long_edge / long_side
        return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale

    def detect(frame: np.ndarray) -> sv.Detections:
        small, scale = downscale(frame)
        dets = detector.detect(small)
        if scale != 1.0 and len(dets) > 0:
            dets.xyxy = dets.xyxy / scale
        return dets

    polygon_zone = sv.PolygonZone(polygon=np.round(quad_px).astype(np.int64))
    foot_projector = GroundProjector(H, quad_px, anchor="foot")
    centroid_projector = GroundProjector(H, quad_px, anchor="centroid")

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[compare_ground_anchor] video={args.video}  calibration={args.calibration} "
          f"(source={calibration['source']})  total_frames={total_frames}")

    disagreements_m: list[float] = []
    foot_row_px: list[float] = []   # y-pixel of the foot anchor -- larger y = nearer the camera on this footage

    frame_idx = 0
    n_frames_processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw = detect(frame)
        in_zone_mask = polygon_zone.trigger(raw)
        in_quad = raw[in_zone_mask]

        foot_proj = foot_projector.project(in_quad.xyxy)
        centroid_proj = centroid_projector.project(in_quad.xyxy)
        # both projectors test in-quad membership using their OWN anchor, so restrict this
        # comparison to detections BOTH consider in-quad -- an honest apples-to-apples set,
        # not artificially inflating disagreement with points only one anchor puts in-quad.
        both_in_quad = foot_proj.in_quad & centroid_proj.in_quad
        if both_in_quad.any():
            diffs = foot_proj.world_xy[both_in_quad] - centroid_proj.world_xy[both_in_quad]
            dists = np.linalg.norm(diffs, axis=1)
            disagreements_m.extend(dists.tolist())
            foot_row_px.extend(foot_proj.foot_px[both_in_quad][:, 1].tolist())

        frame_idx += 1
        n_frames_processed += 1
        if args.max_frames and n_frames_processed >= args.max_frames:
            break
        if frame_idx % 60 == 0:
            print(f"[compare_ground_anchor] frame {frame_idx}/{total_frames}")
    cap.release()

    arr = np.array(disagreements_m)
    rows = np.array(foot_row_px)
    print(f"\n[compare_ground_anchor] frames processed: {n_frames_processed}")
    print(f"[compare_ground_anchor] detections in-quad under BOTH anchors: {arr.size}")
    if arr.size == 0:
        print("[compare_ground_anchor] no comparable detections -- nothing to report")
        return

    print("\n=== foot vs centroid world-position disagreement (metres) ===")
    print(f"  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
          f"p90={np.percentile(arr,90):.3f}  p99={np.percentile(arr,99):.3f}  max={arr.max():.3f}")

    # near vs far split: rows are pixel y of the foot point; on this footage
    # (camera looking down/along the street) larger y = nearer the camera.
    median_row = np.median(rows)
    near_mask = rows >= median_row
    far_mask = ~near_mask
    print("\n=== disagreement by rough near/far split (foot-anchor pixel row vs median) ===")
    print(f"  near half (larger pixel row, closer to camera): n={near_mask.sum()}  "
          f"mean={arr[near_mask].mean():.3f}m  median={np.median(arr[near_mask]):.3f}m")
    print(f"  far half  (smaller pixel row, further from camera): n={far_mask.sum()}  "
          f"mean={arr[far_mask].mean():.3f}m  median={np.median(arr[far_mask]):.3f}m")
    print("  (if 'far' disagreement is meaningfully bigger than 'near', that matches this "
          "project's documented reasoning: torso-height points diverge further from the ground "
          "plane, in world terms, the more oblique/distant the view -- this is a real geometric "
          "effect, not noise)")


if __name__ == "__main__":
    main()
