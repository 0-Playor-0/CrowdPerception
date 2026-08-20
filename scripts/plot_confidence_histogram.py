"""Diagnostic: histogram of raw detection confidence scores, to visually
find where a confidence-threshold "cliff" (real detections vs noise) sits.

Not part of the live pipeline -- this is a testing/tuning tool. Runs the
detector at a near-zero confidence floor (so nothing gets cut before you
can see it) across a handful of sampled frames, and plots the resulting
score distribution with the currently-considered threshold values marked.

Three panels: linear, log (same wide bins, full 0-1 range), and a third
zoomed panel over [floor, --zoom-max] with much finer bins (default 0.01
wide, i.e. resolving 0.11-0.12, 0.12-0.13, ... individually) and a tick
label at every bin edge -- this is the panel for actually reading off
"how many detections would I gain/lose moving the threshold by 0.01" in
the range that matters for picking a threshold, not just eyeballing a
coarse shape.

Usage:
    uv run python scripts/plot_confidence_histogram.py --video data/REJECTED_ToTest_moving_camera.mp4 \\
        --frames 180 300 450 915 --imgsz 1280 --out outputs/confidence_histogram.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perception.detector import Detector, TileConfig, TiledDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 60, 90],
                         help="frame indices to sample -- default is clip-agnostic (first ~3s at "
                              "typical fps); pass your own to match a specific clip's length/content "
                              "(a prior default here was hardcoded to ToTest.mp4's own frame indices "
                              "including 915, which silently under-samples any shorter clip)")
    parser.add_argument("--model", default="models/yolo11s.pt")
    parser.add_argument("--tile-size", type=int, nargs=2, default=[1280, 1280], metavar=("W", "H"))
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--floor", type=float, default=0.01,
                         help="near-zero conf passed to the detector so nothing is pre-filtered")
    parser.add_argument("--mark", type=float, nargs="+", default=[0.1, 0.2, 0.3],
                         help="threshold values to draw as vertical reference lines")
    parser.add_argument("--bin-width", type=float, default=0.02, help="bin width for the top two (full-range) panels")
    parser.add_argument("--zoom-max", type=float, default=0.3, help="upper x-limit of the fine-grained zoom panel")
    parser.add_argument("--zoom-bin-width", type=float, default=0.01,
                         help="bin width for the zoom panel -- e.g. 0.01 resolves 0.11-0.12 individually")
    parser.add_argument("--no-tile", action="store_true")
    parser.add_argument("--out", default="outputs/confidence_histogram.png")
    args = parser.parse_args()

    base = Detector(
        model_path=args.model, device="auto", confidence_threshold=args.floor,
        iou_threshold=0.7, person_class_id=0, imgsz=args.imgsz,
    )
    tile_config = TileConfig(
        tile_size=tuple(args.tile_size), overlap_ratio=args.tile_overlap,
        nms_iou_threshold=args.nms_iou_threshold,
    )
    detector = TiledDetector(base, tile_config=tile_config, enabled=not args.no_tile)

    cap = cv2.VideoCapture(args.video)
    all_conf: list[float] = []
    per_frame_counts: dict[int, int] = {}
    for idx in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[plot_confidence_histogram] could not read frame {idx}, skipping")
            continue
        dets = detector.detect(frame)
        conf = dets.confidence.tolist() if dets.confidence is not None and len(dets) else []
        all_conf.extend(conf)
        per_frame_counts[idx] = len(conf)
        print(f"[plot_confidence_histogram] frame {idx}: {len(conf)} raw detections at conf>={args.floor}")
    cap.release()

    conf_arr = np.array(all_conf)
    print(f"\n[plot_confidence_histogram] total detections across {len(args.frames)} frames: {len(conf_arr)}")
    for m in args.mark:
        n_at_or_above = int(np.sum(conf_arr >= m))
        print(f"  n(conf >= {m}) = {n_at_or_above}")

    # per-0.01-step counts in the zoom range, printed so the numbers behind
    # the zoom panel are also readable without squinting at the plot
    print(f"\n[plot_confidence_histogram] per-{args.zoom_bin_width:g}-step counts, "
          f"[{args.floor:g}, {args.zoom_max:g}):")
    zoom_edges = np.arange(args.floor, args.zoom_max + args.zoom_bin_width / 2, args.zoom_bin_width)
    for lo, hi in zip(zoom_edges[:-1], zoom_edges[1:]):
        n = int(np.sum((conf_arr >= lo) & (conf_arr < hi)))
        print(f"  [{lo:.2f}, {hi:.2f}): {n}")

    fig = plt.figure(figsize=(11, 11))
    ax_lin = fig.add_subplot(3, 1, 1)
    ax_log = fig.add_subplot(3, 1, 2, sharex=ax_lin)
    ax_zoom = fig.add_subplot(3, 1, 3)

    full_bins = np.arange(args.floor, 1.0 + args.bin_width, args.bin_width)
    mark_colors = ["#DD8452", "#C44E52", "#55A868", "#8172B3", "#937860"]

    for ax, yscale in ((ax_lin, "linear"), (ax_log, "log")):
        ax.hist(conf_arr, bins=full_bins, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.set_yscale(yscale)
        ax.set_ylabel(f"detection count ({yscale})")
        for m, color in zip(args.mark, mark_colors):
            ax.axvline(m, color=color, linestyle="--", linewidth=1.5, label=f"threshold={m}")
        ax.grid(alpha=0.3)
        ax.set_xticks(np.arange(0, 1.01, 0.1))

    ax_lin.set_title(
        f"Detection confidence distribution -- {Path(args.video).name}, "
        f"frames {args.frames}, imgsz={args.imgsz}, n={len(conf_arr)}"
    )
    ax_lin.legend(loc="upper right")
    ax_log.legend(loc="upper right")

    # zoom panel: fine bins, a tick+label at every bin edge, over [floor, zoom_max]
    zoom_mask = conf_arr < args.zoom_max
    ax_zoom.hist(conf_arr[zoom_mask], bins=zoom_edges, color="#4C72B0", edgecolor="white", linewidth=0.5)
    for m, color in zip(args.mark, mark_colors):
        if m <= args.zoom_max:
            ax_zoom.axvline(m, color=color, linestyle="--", linewidth=1.5, label=f"threshold={m}")
    ax_zoom.set_xlim(args.floor, args.zoom_max)
    ax_zoom.set_xticks(zoom_edges)
    ax_zoom.set_xticklabels([f"{e:.2f}" for e in zoom_edges], rotation=90, fontsize=7)
    ax_zoom.set_ylabel(f"detection count (linear, {args.zoom_bin_width:g}-wide bins)")
    ax_zoom.set_xlabel("confidence score")
    ax_zoom.set_title(f"Zoom: [{args.floor:g}, {args.zoom_max:g}), {args.zoom_bin_width:g}-wide bins")
    ax_zoom.grid(alpha=0.3, axis="y")
    ax_zoom.legend(loc="upper right")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\n[plot_confidence_histogram] wrote {out_path}")


if __name__ == "__main__":
    main()
