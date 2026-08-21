"""Live perception demo: video in, two panes out -- annotated camera view
(boxes + track IDs + foot points + total count + fps) side by side with a
top-down ground-plane density heatmap -- while writing per-person
trajectories to disk.

Requires a calibration file from scripts/calibrate_video.py (default:
calibration/<video_stem>.json).

Runs with or without a GUI window (--no-window for headless/CI use --
still writes trajectories, stats, --save-video, and any --snapshot-frames).

Usage:
    uv run python scripts/live_perception.py --video data/xxx.mp4

    # headless smoke test, 100 frames, snapshot at frame 50
    uv run python scripts/live_perception.py --video data/xxx.mp4 \\
        --no-window --max-frames 100 --snapshot-frames 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import supervision as sv

from perception.detector import Detector, TiledDetector, TileConfig
from perception.ground import GroundProjector, foot_points
from perception.heatmap import GroundHeatmap, render_heatmap_overlay, render_heatmap_panel
from perception.tracker import Tracker
from perception.trajectory import TrajectoryLogger

DISPLAY_MAX_HEIGHT_PX = 900
WINDOW_NAME = "live_perception -- camera | heatmap"

COLOR_TRACKED = (0, 220, 0)          # solid green: confirmed track, in quad
COLOR_UNTRACKED_IN_QUAD = (0, 200, 255)  # solid amber: detected in quad, not (yet) a confirmed track
COLOR_OUTSIDE_QUAD = (150, 150, 150)     # dashed grey: outside the calibrated quad


# ---------------------------------------------------------------------------
# Detection wrapper: applies --downscale before handing frames to the
# (possibly tiled) detector, then rescales boxes back to ORIGINAL frame
# pixel coordinates. Everything downstream (tracking, calibration
# homography, drawing) works in original-resolution pixel space regardless
# of what resolution detection actually ran at -- this is a display/perf
# concern local to this script, not something perception/detector.py needs
# to know about.
# ---------------------------------------------------------------------------
class DownscaledDetector:
    def __init__(self, inner: Detector | TiledDetector, long_edge_px: int) -> None:
        self._inner = inner
        self._long_edge_px = long_edge_px

    def _downscale(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        if self._long_edge_px <= 0:
            return frame, 1.0
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side <= self._long_edge_px:
            return frame, 1.0
        scale = self._long_edge_px / long_side
        resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    def detect(self, frame: np.ndarray) -> sv.Detections:
        small, scale = self._downscale(frame)
        detections = self._inner.detect(small)
        if scale != 1.0 and len(detections) > 0:
            detections.xyxy = detections.xyxy / scale
        return detections

    @property
    def last_latency_ms(self) -> float:
        return getattr(self._inner, "last_latency_ms", 0.0)


def build_detector(args: argparse.Namespace) -> DownscaledDetector | TiledDetector:
    base = Detector(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        person_class_id=0,
        imgsz=args.imgsz,
    )
    tile_config = TileConfig(
        tile_size=tuple(args.tile_size),
        overlap_ratio=args.tile_overlap,
        per_tile_conf_threshold=args.tile_conf_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )
    tiled = TiledDetector(base, tile_config=tile_config, enabled=args.tile)
    if args.downscale_long_edge > 0:
        return DownscaledDetector(tiled, args.downscale_long_edge)
    return tiled


def load_calibration(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"[live_perception] calibration file not found: {path}\n"
            "Run scripts/calibrate_video.py first, e.g.:\n"
            f"  uv run python scripts/calibrate_video.py --video <your video> --out {path}"
        )
    with path.open("r") as f:
        return json.load(f)


def _match_mask_by_xyxy(candidates_xyxy: np.ndarray, reference_xyxy: np.ndarray, atol: float = 1e-3) -> np.ndarray:
    """Boolean mask over candidates_xyxy: True where a (near-)identical box
    exists in reference_xyxy.

    Used to recover which in-quad detections did NOT end up as a confirmed
    track, purely from box geometry: OC-SORT's update() passes matched and
    newly-spawned boxes through unchanged (see trackers.core.ocsort.tracker
    -- detection_boxes are re-indexed, never transformed), so a detection
    that became "confirmed" has an exactly-matching xyxy row in the
    tracker's output. Detections dropped by OC-SORT's internal
    high_conf_det_threshold filter, or not yet past minimum_consecutive_frames,
    have no match here -- that's exactly the "in quad, not tracked" set this
    script now draws instead of silently dropping.
    """
    if len(candidates_xyxy) == 0 or len(reference_xyxy) == 0:
        return np.zeros(len(candidates_xyxy), dtype=bool)
    matched = np.zeros(len(candidates_xyxy), dtype=bool)
    for i, box in enumerate(candidates_xyxy):
        if np.any(np.all(np.isclose(reference_xyxy, box, atol=atol), axis=1)):
            matched[i] = True
    return matched


# ---------------------------------------------------------------------------
# Drawing helpers -- camera pane
# ---------------------------------------------------------------------------
def _dashed_line(img: np.ndarray, p1: tuple[float, float], p2: tuple[float, float],
                  color: tuple[int, int, int], thickness: int = 1, dash_len_px: float = 8.0) -> None:
    p1_arr, p2_arr = np.array(p1, dtype=np.float64), np.array(p2, dtype=np.float64)
    dist = float(np.linalg.norm(p2_arr - p1_arr))
    if dist < 1e-6:
        return
    n_dashes = max(1, int(dist // dash_len_px))
    for i in range(n_dashes):
        start = p1_arr + (p2_arr - p1_arr) * (i / n_dashes)
        end = p1_arr + (p2_arr - p1_arr) * ((i + 0.5) / n_dashes)
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness, cv2.LINE_AA)


def _draw_dashed_rect(img: np.ndarray, pt1: tuple[float, float], pt2: tuple[float, float],
                       color: tuple[int, int, int], thickness: int = 1) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    _dashed_line(img, (x1, y1), (x2, y1), color, thickness)
    _dashed_line(img, (x2, y1), (x2, y2), color, thickness)
    _dashed_line(img, (x2, y2), (x1, y2), color, thickness)
    _dashed_line(img, (x1, y2), (x1, y1), color, thickness)


def _put_outlined_text(img: np.ndarray, text: str, org: tuple[int, int],
                        color: tuple[int, int, int], scale: float = 0.55) -> None:
    """For text drawn directly over unpredictable video content (box
    labels, legend swatches). HUD text no longer needs this -- it now sits
    on a solid background band, see draw_hud."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_camera_pane(
    display_frame: np.ndarray,
    quad_px_display: np.ndarray,
    out_of_quad_xyxy_display: np.ndarray,
    untracked_in_quad_xyxy_display: np.ndarray,
    untracked_in_quad_foot_px_display: np.ndarray,
    confirmed_xyxy_display: np.ndarray,
    confirmed_ids: np.ndarray,
    confirmed_foot_px_display: np.ndarray,
) -> np.ndarray:
    """Draws three visually distinct detection states (see module-level
    COLOR_* constants and the legend drawn by draw_legend):
      - tracked + in quad      -> solid green box + ID + foot dot
      - in quad, not tracked   -> solid amber box + foot dot, no ID
      - outside quad           -> dashed grey box
    An in-quad detection that never becomes a confirmed track (dropped by
    the tracker's internal confidence gate, or not yet past its consecutive-
    frame confirmation requirement -- see the Issue 1 diagnosis) is real,
    visible information: drawing nothing for it made the display look like
    it was ignoring obvious people, when in fact those people WERE detected.
    """
    img = display_frame.copy()

    cv2.polylines(img, [quad_px_display.astype(np.int32)], isClosed=True,
                  color=(255, 220, 0), thickness=2, lineType=cv2.LINE_AA)

    for x1, y1, x2, y2 in out_of_quad_xyxy_display:
        _draw_dashed_rect(img, (x1, y1), (x2, y2), COLOR_OUTSIDE_QUAD, 1)

    for i in range(len(untracked_in_quad_xyxy_display)):
        x1, y1, x2, y2 = untracked_in_quad_xyxy_display[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_UNTRACKED_IN_QUAD, 2, cv2.LINE_AA)
        fx, fy = untracked_in_quad_foot_px_display[i]
        cv2.circle(img, (int(fx), int(fy)), 4, (0, 0, 255), -1, cv2.LINE_AA)

    for i in range(len(confirmed_xyxy_display)):
        x1, y1, x2, y2 = confirmed_xyxy_display[i].astype(int)
        track_id = int(confirmed_ids[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_TRACKED, 2, cv2.LINE_AA)
        _put_outlined_text(img, f"#{track_id}", (x1, max(0, y1 - 8)), COLOR_TRACKED, 0.5)
        fx, fy = confirmed_foot_px_display[i]
        cv2.circle(img, (int(fx), int(fy)), 4, (0, 0, 255), -1, cv2.LINE_AA)

    return img


def draw_legend(img: np.ndarray) -> None:
    """Small always-on-a-solid-background legend, top-right corner, well
    away from the bottom-left HUD -- explains the three box states."""
    entries = [
        (COLOR_TRACKED, "tracked, in quad"),
        (COLOR_UNTRACKED_IN_QUAD, "in quad, not tracked"),
        (COLOR_OUTSIDE_QUAD, "outside quad"),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    line_height = 20
    swatch_w = 16

    max_text_width = 0
    for _, label in entries:
        (w, _h), _ = cv2.getTextSize(label, font, font_scale, 1)
        max_text_width = max(max_text_width, w)

    band_width = swatch_w + 10 + max_text_width + 16
    band_height = len(entries) * line_height + 12

    img_h, img_w = img.shape[:2]
    x0 = img_w - band_width - 10
    y0 = 10

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + band_width, y0 + band_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + band_width, y0 + band_height), (90, 90, 90), 1, cv2.LINE_AA)

    for i, (color, label) in enumerate(entries):
        y = y0 + 10 + i * line_height
        cv2.rectangle(img, (x0 + 8, y), (x0 + 8 + swatch_w, y + 10), color, -1)
        cv2.putText(img, label, (x0 + 8 + swatch_w + 8, y + 10), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, hud: dict) -> None:
    """Bottom-left, on a near-solid background band sized to the actual
    text (not a fixed guess), monospace-ish font (FONT_HERSHEY_PLAIN, whose
    glyphs are far closer to fixed-width than SIMPLEX) with generous line
    spacing -- fixes both the wrong-label bug (funnel stages were being
    mislabeled) and the overlap-with-video legibility bug from the previous
    version."""
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.15
    thickness = 1
    line_height = 22

    lines = [
        f"detections:{hud['detections']:>4d}  in quad:{hud['in_quad']:>4d}  "
        f"tracked:{hud['tracked']:>4d}  excluded:{hud['excluded']:>4d}",
        f"total IDs seen:{hud['total_ids']:>5d}    fps:{hud['fps']:5.1f}",
        f"detect:{hud['detect_ms']:6.1f}ms  track:{hud['track_ms']:6.1f}ms",
        f"frame:{hud['frame_idx']:>5d}/{hud['total_frames']:<5d}  tiling:{'ON' if hud['tiling'] else 'OFF'}",
        f"heatmap:{hud['heatmap_mode']:<7s} ('h' toggles)",
    ]
    is_estimated = hud["calibration_source"] == "ESTIMATED"
    cal_line = f"calibration: {hud['calibration_source']}" + ("  -- NOT MEASURED" if is_estimated else "")
    all_lines = lines + [cal_line]

    max_width = 0
    for line in all_lines:
        (w, _h), _ = cv2.getTextSize(line, font, font_scale, thickness + 1)
        max_width = max(max_width, w)
    band_width = max_width + 24
    band_height = len(all_lines) * line_height + 16

    img_h, img_w = img.shape[:2]
    x0 = 10
    y0 = img_h - band_height - 10

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + band_width, y0 + band_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + band_width, y0 + band_height), (110, 110, 110), 1, cv2.LINE_AA)

    text_x = x0 + 12
    y = y0 + line_height
    for line in lines:
        cv2.putText(img, line, (text_x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height

    cal_color = (0, 150, 255) if is_estimated else (60, 220, 60)
    cv2.putText(img, cal_line, (text_x, y), font, font_scale, cal_color, thickness + 1, cv2.LINE_AA)

    if hud["paused"]:
        _put_outlined_text(img, "PAUSED (space to resume)", (x0, y0 - 12), (0, 220, 255), 0.6)


# ---------------------------------------------------------------------------
# Per-frame pipeline
# ---------------------------------------------------------------------------
class LivePerceptionRunner:
    def __init__(
        self, args: argparse.Namespace, calibration: dict, fps: float, total_frames: int,
        frame_width: int, frame_height: int,
    ) -> None:
        self.args = args
        self.calibration = calibration
        self.fps = fps
        self.total_frames = total_frames

        self.detector = build_detector(args)
        self.tracker = Tracker(frame_rate=fps)

        quad_px = np.array(calibration["image_points"], dtype=np.float64)
        quad_m = np.array(calibration["world_points"], dtype=np.float64)
        H = np.array(calibration["H"], dtype=np.float64)
        self.quad_px = quad_px
        self.quad_m = quad_m
        self.ground_projector = GroundProjector(H, quad_px)

        # Metric top-down heatmap -- real persons/m^2, valid ONLY inside the
        # calibrated quad (the homography isn't trustworthy outside it).
        self.heatmap = GroundHeatmap(
            quad_m,
            cell_size_m=args.cell_size,
            window_seconds=args.heatmap_window_seconds,
            sigma=args.heatmap_sigma,
            count_vmax=args.heatmap_vmax,
        )
        self.heatmap_mode = args.heatmap_mode

        # Whole-frame PIXEL-space heatmap, overlaid directly on the camera
        # pane -- covers people outside the quad too (which the metric
        # heatmap above cannot honestly do). "count" mode only: a
        # persons/px^2 "density" would be meaningless. Same GroundHeatmap
        # class, just constructed over the frame's own pixel bounding box
        # instead of the calibrated quad -- see perception/heatmap.py's
        # module docstring for why that's a valid reuse.
        self.frame_heatmap: GroundHeatmap | None = None
        if args.frame_heatmap:
            frame_quad_px = np.array(
                [[0.0, 0.0], [frame_width, 0.0], [frame_width, frame_height], [0.0, frame_height]],
                dtype=np.float64,
            )
            self.frame_heatmap = GroundHeatmap(
                frame_quad_px,
                cell_size_m=args.frame_heatmap_cell_size_px,
                window_seconds=args.heatmap_window_seconds,
                sigma=args.heatmap_sigma,
                count_vmax=args.frame_heatmap_vmax,
            )

        video_stem = Path(args.video).stem
        outputs_dir = Path(args.outputs_dir)
        trajectory_dir = Path(args.trajectory_dir)
        self.stats_path = outputs_dir / f"{video_stem}_track_stats.json"
        parquet_path = None
        try:
            import polars  # noqa: F401
            parquet_path = trajectory_dir / f"{video_stem}_trajectory.parquet"
        except ImportError:
            pass
        self.trajectory_logger = TrajectoryLogger(
            csv_path=trajectory_dir / f"{video_stem}_trajectory.csv",
            parquet_path=parquet_path,
        )

        self.tile_source_label = "tiled" if args.tile else "untiled"

        self.total_unique_ids: set[int] = set()
        self._fps_history: deque[float] = deque(maxlen=30)
        self.funnel_totals = {"total": 0, "in_quad": 0, "tracked": 0, "emitted": 0}
        self._last_funnel_print = time.perf_counter()

        # Metadata only -- set by build_composite() below, read by
        # server/pipeline_runner.py to split the composite it returns back
        # into separate camera/heatmap images for the two MJPEG streams,
        # without recomputing (and risking silently drifting from) the
        # boundary build_composite actually used. Does not affect this
        # class's own rendering in any way -- nothing here reads it back.
        self.last_camera_pane_width_px: int | None = None

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> tuple[np.ndarray, dict]:
        t_sec = frame_idx / self.fps
        t_frame_start = time.perf_counter()

        raw_detections = self.detector.detect(frame)
        detect_ms = getattr(self.detector, "last_latency_ms", 0.0)

        # Single source of truth for "is this detection in the calibrated
        # quad": GroundProjector's own foot-point-anchored point-in-polygon
        # test, computed once here and reused everywhere below. This used
        # to run TWICE with two different polygon representations --
        # sv.PolygonZone (rounded to integer pixels) filtered the tracker's
        # input, then GroundProjector's own full-float test ran AGAIN just
        # for the heatmap -- and they occasionally disagreed right at the
        # boundary (measured on real footage: ~0.15% of in-quad detections
        # were silently dropped before ever reaching the heatmap, purely
        # from int-vs-float polygon rounding). One test, reused, means "in
        # quad" means the same thing everywhere in this frame, and the
        # heatmap genuinely tracks everyone detected in the quad -- not
        # everyone minus a rounding artifact.
        raw_projection = self.ground_projector.project(raw_detections.xyxy)
        in_zone_mask = raw_projection.in_quad
        in_quad_detections = raw_detections[in_zone_mask]
        out_of_quad_detections = raw_detections[~in_zone_mask]
        in_quad_world_points_all = raw_projection.world_xy[in_zone_mask]

        t0 = time.perf_counter()
        tracked = self.tracker.update(in_quad_detections)
        track_ms = (time.perf_counter() - t0) * 1000.0

        if tracked.tracker_id is not None and len(tracked) > 0:
            confirmed = tracked[tracked.tracker_id >= 0]
        else:
            confirmed = tracked

        # Everything OC-SORT was GIVEN (in_quad_detections) that did NOT
        # come back out as a confirmed track -- dropped by the tracker's
        # internal confidence gate or not yet past its consecutive-frame
        # confirmation requirement (see Issue 1's diagnosis). Recovered by
        # box geometry since OC-SORT doesn't expose this mapping directly
        # (see _match_mask_by_xyxy's docstring).
        confirmed_match_mask = _match_mask_by_xyxy(in_quad_detections.xyxy, confirmed.xyxy)
        untracked_in_quad_xyxy = in_quad_detections.xyxy[~confirmed_match_mask]
        untracked_in_quad_foot_px = raw_projection.foot_px[in_zone_mask][~confirmed_match_mask]

        projection = self.ground_projector.project(confirmed.xyxy)

        excluded_this_frame = int(len(out_of_quad_detections)) + projection.n_excluded
        self.funnel_totals["total"] += len(raw_detections)
        self.funnel_totals["in_quad"] += len(in_quad_detections)
        self.funnel_totals["tracked"] += len(confirmed)

        # Metric heatmap: fed from EVERY in-quad detection (raw_projection,
        # computed once above), not just confirmed tracks. Confirmed-only
        # starves the heatmap almost completely -- OC-SORT confirms a small
        # fraction of in-quad detections (see docs/TOTEST_FINDINGS.md /
        # REAL_FOOTAGE_FINDINGS.md C3, ~85-97% loss before confirmation on
        # the two real clips measured) -- while the heatmap's own job
        # (density) doesn't need a stable track ID, only a valid ground
        # position, which every in-quad detection already has.
        self.heatmap.update(t_sec, in_quad_world_points_all)

        # Whole-frame pixel heatmap: fed from every raw detection,
        # in-quad or not -- no homography needed, it's just where boxes
        # landed on screen.
        if self.frame_heatmap is not None:
            self.frame_heatmap.update(t_sec, foot_points(raw_detections.xyxy))

        for i in range(len(confirmed)):
            track_id = int(confirmed.tracker_id[i])
            self.total_unique_ids.add(track_id)
            x1, y1, x2, y2 = confirmed.xyxy[i]
            conf = float(confirmed.confidence[i]) if confirmed.confidence is not None else float("nan")
            fx, fy = projection.foot_px[i]
            is_in_quad = bool(projection.in_quad[i])
            wx, wy = projection.world_xy[i] if is_in_quad else (float("nan"), float("nan"))

            self.trajectory_logger.log_row(
                frame_idx=frame_idx,
                timestamp_s=t_sec,
                track_id=track_id,
                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                conf=conf,
                foot_px=(float(fx), float(fy)),
                world_xy=(float(wx), float(wy)),
                in_quad=is_in_quad,
                tile_source=self.tile_source_label,
            )

        self.funnel_totals["emitted"] += len(confirmed)

        now = time.perf_counter()
        self._fps_history.append(now - t_frame_start)
        measured_fps = 1.0 / (sum(self._fps_history) / len(self._fps_history)) if self._fps_history else 0.0

        if now - self._last_funnel_print >= 2.0:
            print(f"[live_perception] frame {frame_idx}/{self.total_frames}  "
                  f"detections total={len(raw_detections)} in_quad={len(in_quad_detections)} "
                  f"tracked={len(confirmed)} emitted={len(confirmed)}  "
                  f"excluded(this frame)={excluded_this_frame}  fps={measured_fps:.1f}")
            self._last_funnel_print = now

        # Four funnel stages, unambiguous and matching what's printed to
        # stdout and what's actually drawn -- see Issue 2: the previous HUD
        # showed the TRACKED count under an "in quad" label.
        hud = {
            "detections": len(raw_detections),
            "in_quad": len(in_quad_detections),
            "tracked": len(confirmed),
            "excluded": excluded_this_frame,
            "total_ids": len(self.total_unique_ids),
            "fps": measured_fps,
            "detect_ms": detect_ms,
            "track_ms": track_ms,
            "frame_idx": frame_idx,
            "total_frames": self.total_frames,
            "tiling": self.args.tile,
            "heatmap_mode": self.heatmap_mode,
            "calibration_source": self.calibration["source"],
            "paused": False,
        }

        render_data = self._render_frame_data(
            frame, out_of_quad_detections.xyxy,
            untracked_in_quad_xyxy, untracked_in_quad_foot_px,
            confirmed, projection,
        )
        return render_data, hud

    def _render_frame_data(self, frame, out_of_quad_xyxy, untracked_in_quad_xyxy,
                            untracked_in_quad_foot_px, confirmed, projection):
        return {
            "frame": frame,
            "out_of_quad_xyxy": out_of_quad_xyxy,
            "untracked_in_quad_xyxy": untracked_in_quad_xyxy,
            "untracked_in_quad_foot_px": untracked_in_quad_foot_px,
            "confirmed_xyxy": confirmed.xyxy,
            "confirmed_ids": confirmed.tracker_id if confirmed.tracker_id is not None else np.zeros((0,), dtype=int),
            "confirmed_foot_px": projection.foot_px,
        }

    def build_composite(self, render_data: dict, hud: dict) -> np.ndarray:
        frame = render_data["frame"]
        h, w = frame.shape[:2]
        display_scale = min(1.0, DISPLAY_MAX_HEIGHT_PX / h)
        display_frame = cv2.resize(frame, (round(w * display_scale), round(h * display_scale))) \
            if display_scale < 1.0 else frame.copy()

        if self.frame_heatmap is not None:
            # applied BEFORE camera_pane draws boxes/quad/HUD on top, so
            # those stay fully legible over the heatmap tint. Same colour
            # ramp shape as the metric panel, own (uncalibrated) vmax.
            display_frame = render_heatmap_overlay(
                self.frame_heatmap, display_frame, mode="count",
                color_variant=self.args.heatmap_color_variant,
            )

        def scale_xyxy(xyxy: np.ndarray) -> np.ndarray:
            return xyxy * display_scale if len(xyxy) else xyxy.reshape(0, 4)

        def scale_xy(xy: np.ndarray) -> np.ndarray:
            return xy * display_scale if len(xy) else xy.reshape(0, 2)

        camera_pane = draw_camera_pane(
            display_frame,
            self.quad_px * display_scale,
            scale_xyxy(render_data["out_of_quad_xyxy"]),
            scale_xyxy(render_data["untracked_in_quad_xyxy"]),
            scale_xy(render_data["untracked_in_quad_foot_px"]),
            scale_xyxy(render_data["confirmed_xyxy"]),
            render_data["confirmed_ids"],
            scale_xy(render_data["confirmed_foot_px"]),
        )
        draw_legend(camera_pane)
        draw_hud(camera_pane, hud)

        # The real, measured width of the pane just drawn -- not a
        # recomputation from display_scale/w, so a caller reading this
        # afterwards (server/pipeline_runner.py) always gets the boundary
        # this call actually used, even if the scaling/layout above changes
        # later. Pure metadata: nothing below this line depends on it.
        self.last_camera_pane_width_px = camera_pane.shape[1]

        heatmap_panel = render_heatmap_panel(
            self.heatmap, self.quad_m, self.heatmap_mode, panel_height_px=camera_pane.shape[0],
            color_variant=self.args.heatmap_color_variant,
        )
        return np.hstack([camera_pane, heatmap_panel])

    def close(self, frames_processed: int) -> None:
        self.trajectory_logger.close(frames_processed, self.stats_path)
        print(f"[live_perception] funnel totals over the run: {self.funnel_totals}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", default=None, help="default: calibration/<video_stem>.json")
    parser.add_argument("--model", default="models/yolo11s.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.15,
                         help="raised from 0.02 -> 0.15 on the Myeongdong demo clip "
                              "(data/127690-739144743.mp4, tiled, frames 0/135/300/512): measured "
                              "duplicate-detection rate (same person, two boxes surviving cross-tile "
                              "NMS) dropped from ~130%% of final detections at 0.02 to ~40%% at 0.15, "
                              "with final detections/frame falling from ~178 to ~62 -- see "
                              "scripts/diagnose_tiled_nms.py and outputs/nms_diagnosis_conf0.15/. "
                              "No hand count exists for this clip, so the recall cost of raising the "
                              "threshold this far is not quantified -- treat 0.15 as a measured "
                              "false-positive win, not a validated precision/recall optimum. The old "
                              "0.02 default sat inside the single largest noise spike in the measured "
                              "confidence distribution (see outputs/confidence_histogram*.png) -- "
                              "see docs/TOTEST_FINDINGS.md TEST 4 for that original reasoning")
    parser.add_argument("--iou-threshold", type=float, default=0.7)

    parser.add_argument("--tile", action=argparse.BooleanOptionalAction, default=True,
                         help="tiled (SAHI-style) inference for recall; --no-tile for whole-frame speed")
    parser.add_argument("--tile-size", type=int, nargs=2, default=[1280, 1280], metavar=("W", "H"),
                         help="pixel size of each tile crop (default: 1280 1280)")
    parser.add_argument("--tile-overlap", dest="tile_overlap", type=float, default=0.1,
                         help="fraction of tile_size each neighbouring tile overlaps by (default: 0.1)")
    parser.add_argument("--tile-conf-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=None,
                         help="resolution each tile (or whole frame, if --no-tile) is resized to before "
                              "the network sees it. Default: unset, i.e. Ultralytics' own model default "
                              "(640 for yolo11s.pt) -- so a --tile-size larger than this gets downscaled "
                              "on the way in unless you raise --imgsz to match. Unchanged default behaviour.")

    parser.add_argument("--downscale", dest="downscale_long_edge", type=int, default=1920,
                         help="resize longest edge to this many px before detection; 0 disables")

    parser.add_argument("--save-video", action="store_true",
                         help="write the composite to outputs/<video_stem>_annotated.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--heatmap-mode", choices=["count", "density"], default="count")
    parser.add_argument("--cell-size", type=float, default=0.5)
    parser.add_argument("--heatmap-window-seconds", type=float, default=2.0)
    parser.add_argument("--heatmap-sigma", type=float, default=1.0)
    parser.add_argument("--heatmap-color-variant", choices=["hard", "eased"], default="hard",
                         help="density colour ramp transition width around the 2.5 persons/m^2 "
                              "alarm onset -- 'hard': sharp edge, maximally legible, strobes on "
                              "cells oscillating near 2.5. 'eased': gentler blend, less strobing, "
                              "less crisp. See perception/heatmap.py's COLOR_STOPS_HARD/EASED.")
    parser.add_argument("--heatmap-vmax", type=float, default=3.0,
                         help="fixed colour-scale ceiling for the metric top-down panel's COUNT "
                              "mode (persons/cell) -- DENSITY mode has its own fixed ceiling, "
                              "perception.heatmap.DEFAULT_DENSITY_VMAX_PERSONS_PER_M2, "
                              "independent of this flag. Lower this if 'cur. max' in "
                              "the panel legend stays far below it in count mode -- the heatmap "
                              "looks empty when real data is dwarfed by too high a fixed scale.")

    parser.add_argument("--frame-heatmap", action=argparse.BooleanOptionalAction, default=True,
                         help="translucent whole-frame heatmap overlaid on the camera pane, built "
                              "from every detection (in-quad or not) in pixel space -- unlike the "
                              "metric panel, not limited to the calibrated quad. --no-frame-heatmap disables it.")
    parser.add_argument("--frame-heatmap-cell-size-px", type=int, default=60,
                         help="bigger cells = less rolling-window dilution per cell, more visible overlay")
    parser.add_argument("--frame-heatmap-vmax", type=float, default=0.5,
                         help="fixed colour-scale ceiling for the whole-frame overlay, in persons/cell "
                              "(this is a window-averaged count over many small cells, not a raw "
                              "per-frame count -- stays well under 1.0 except at real hotspots, see "
                              "perception/heatmap.py's render_heatmap_overlay docstring)")

    parser.add_argument("--window", action=argparse.BooleanOptionalAction, default=True,
                         help="show a live GUI window; --no-window for headless runs")
    parser.add_argument("--snapshot-frames", type=int, nargs="*", default=[],
                         help="frame indices to save a composite PNG for, regardless of --window")

    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--trajectory-dir", default="outputs/trajectories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    calibration_path = Path(args.calibration) if args.calibration else Path("calibration") / f"{Path(args.video).stem}.json"
    calibration = load_calibration(calibration_path)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"[live_perception] could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[live_perception] video={args.video}  fps={fps:.2f}  total_frames={total_frames}  "
          f"tiling={args.tile}  downscale_long_edge={args.downscale_long_edge}  "
          f"calibration={calibration_path} (source={calibration['source']})")
    if calibration["source"] == "ESTIMATED":
        print(f"[live_perception] WARNING: calibration source is ESTIMATED, not measured -- "
              f"{calibration.get('note', '')}")

    runner = LivePerceptionRunner(args, calibration, fps, total_frames, frame_width, frame_height)

    video_sink = None
    if args.save_video:
        out_path = Path(args.outputs_dir) / f"{Path(args.video).stem}_annotated.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        video_sink = {"path": out_path, "writer": None}

    outputs_dir = Path(args.outputs_dir)
    snapshot_frames = set(args.snapshot_frames)

    if args.window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_idx = 0
    frames_processed = 0
    paused = False
    last_composite = None
    heatmap_mode = args.heatmap_mode

    try:
        while True:
            if args.max_frames and frames_processed >= args.max_frames:
                break

            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break

                runner.heatmap_mode = heatmap_mode
                render_data, hud = runner.process_frame(frame, frame_idx)
                hud["paused"] = False
                composite = runner.build_composite(render_data, hud)
                last_composite = composite
                frames_processed += 1

                if video_sink is not None:
                    if video_sink["writer"] is None:
                        h, w = composite.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_sink["writer"] = cv2.VideoWriter(str(video_sink["path"]), fourcc, fps, (w, h))
                    video_sink["writer"].write(composite)

                if frame_idx in snapshot_frames:
                    snap_path = outputs_dir / "snapshots" / f"{Path(args.video).stem}_frame{frame_idx}.png"
                    snap_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap_path), composite)
                    print(f"[live_perception] wrote snapshot {snap_path}")

                frame_idx += 1
            else:
                composite = last_composite
                if composite is not None:
                    hud_paused = composite.copy()
                    _put_outlined_text(hud_paused, "PAUSED (space to resume)",
                                        (12, hud_paused.shape[0] - 14), (0, 220, 255), 0.6)
                    composite = hud_paused

            if args.window and composite is not None:
                cv2.imshow(WINDOW_NAME, composite)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("h"):
                    heatmap_mode = "density" if heatmap_mode == "count" else "count"
                elif key == ord("s") and last_composite is not None:
                    snap_path = outputs_dir / "snapshots" / f"{Path(args.video).stem}_frame{frame_idx - 1}_manual.png"
                    snap_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap_path), last_composite)
                    print(f"[live_perception] wrote snapshot {snap_path}")
            elif not args.window and paused:
                # headless mode has no key events to unpause with -- pausing
                # should never happen here since nothing sets `paused` True
                # without a window, but guard against an infinite loop anyway.
                paused = False
    except KeyboardInterrupt:
        print(f"\n[live_perception] interrupted at frame {frame_idx} -- flushing")
    finally:
        capture.release()
        if args.window:
            cv2.destroyWindow(WINDOW_NAME)
        if video_sink is not None and video_sink["writer"] is not None:
            video_sink["writer"].release()
            print(f"[live_perception] wrote {video_sink['path']}")
        runner.close(frames_processed)

    print(f"[live_perception] done: {frames_processed} frames processed")


if __name__ == "__main__":
    main()
