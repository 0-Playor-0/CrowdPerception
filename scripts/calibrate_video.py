"""Interactive ground-plane calibration: click 4 points on a video frame,
tell it the real-world size of the quad they trace out, get back a
homography (calibration/<video_stem>.json) that scripts/live_perception.py
loads directly.

Click order matters and is fixed: near-left, near-right, far-right,
far-left -- i.e. walk the calibrated ground rectangle's corners starting at
the bottom-left (closest to camera, camera's left) and going clockwise as
seen in the frame. The world quad is built assuming this exact order forms
a rectangle: near-left=(-w/2, 0), near-right=(w/2, 0), far-right=(w/2, h),
far-left=(-w/2, h), where w/h are the width/height you enter in metres.

Controls while picking:
    left-click   place the next point (up to 4)
    u            undo the last point
    r            reset (clear all points)
    q            abort -- no file is written

After the 4th point, you're prompted on stdin for the quad's real-world
width and height in metres. Press enter on BOTH to accept an ESTIMATED
default (7.0m x 5.25m, chosen to match the ballpark this project already
established from a height-based perspective fit on real footage -- see
docs/REAL_FOOTAGE_FINDINGS.md C2.4, R^2=0.464, NOT a real measurement) --
in that case "source" is written as "ESTIMATED" with a note explaining
where the number came from. Type real numbers for BOTH and "source" is
written as "USER_MEASURED".

Usage:
    uv run python scripts/calibrate_video.py --video data/xxx.mp4 --frame 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ESTIMATED fallback if the user presses enter on both prompts -- see module
# docstring. 7.0m matches the task's stated default; 5.25m = 7.0 * 0.75,
# approximating the depth/width ratio (5.82/7.68 ~= 0.758) this project
# already measured via the height-based perspective fit in
# docs/REAL_FOOTAGE_FINDINGS.md C2.4. Both numbers are placeholders, not
# measurements -- the note field written into the output JSON says so.
DEFAULT_ESTIMATED_WIDTH_M = 7.0
DEFAULT_ESTIMATED_HEIGHT_M = 5.25
ESTIMATED_NOTE = (
    "No stdin measurement given -- width/height default to "
    f"{DEFAULT_ESTIMATED_WIDTH_M}m x {DEFAULT_ESTIMATED_HEIGHT_M}m, chosen to "
    "match the order of magnitude this project already found via a "
    "height-based perspective fit on real footage (height_px = 0.24215*y - "
    "74.852, R^2=0.464, n=478 -- see docs/REAL_FOOTAGE_FINDINGS.md C2.4). "
    "That fit itself was WEAK (R^2=0.464, ~34% residual). This is not a "
    "measurement of THIS quad -- re-run with real numbers for anything "
    "beyond order-of-magnitude/demo use."
)
CORNER_LABELS = ["near-left", "near-right", "far-right", "far-left"]
ROUND_TRIP_MAX_ERROR_PX = 1.0


class QuadValidationError(ValueError):
    pass


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


def _is_convex(points: np.ndarray) -> bool:
    n = len(points)
    signs = []
    for i in range(n):
        a, b, c = points[i], points[(i + 1) % n], points[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def validate_quad(image_points: np.ndarray) -> None:
    """Refuses non-convex or self-intersecting (bowtie) quads -- either
    means the 4 clicks weren't actually the 4 corners of a simple
    quadrilateral in near-left/near-right/far-right/far-left order, and any
    homography fit through them would be meaningless."""
    p0, p1, p2, p3 = image_points
    if _segments_intersect(p0, p1, p2, p3) or _segments_intersect(p1, p2, p3, p0):
        raise QuadValidationError(
            "clicked quad is self-intersecting (bowtie) -- check click order: "
            f"{', '.join(CORNER_LABELS)}"
        )
    if not _is_convex(image_points):
        raise QuadValidationError(
            "clicked quad is non-convex -- check click order/positions: "
            f"{', '.join(CORNER_LABELS)}"
        )


def build_calibration_record(
    image_points: np.ndarray,
    width_m: float,
    height_m: float,
    source: str,
    note: str,
    video_path: str,
    frame_index: int,
    video_resolution: tuple[int, int],
) -> dict:
    """Pure function (no I/O, no stdin, no GUI) so it can be unit-tested and
    reused to script a calibration file without the interactive picker."""
    image_points = np.asarray(image_points, dtype=np.float64)
    if image_points.shape != (4, 2):
        raise ValueError(f"image_points must be shape (4, 2), got {image_points.shape}")
    if width_m <= 0 or height_m <= 0:
        raise ValueError(f"width_m/height_m must be > 0, got {width_m}/{height_m}")

    validate_quad(image_points)

    world_points = np.array(
        [
            [-width_m / 2.0, 0.0],
            [width_m / 2.0, 0.0],
            [width_m / 2.0, height_m],
            [-width_m / 2.0, height_m],
        ],
        dtype=np.float64,
    )

    H = cv2.getPerspectiveTransform(image_points.astype(np.float32), world_points.astype(np.float32))
    H_inv = cv2.getPerspectiveTransform(world_points.astype(np.float32), image_points.astype(np.float32))

    world_check = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)
    back_px = cv2.perspectiveTransform(
        world_check.reshape(-1, 1, 2).astype(np.float32), H_inv
    ).reshape(-1, 2)
    round_trip_error_px = np.linalg.norm(back_px - image_points, axis=1)
    max_error_px = float(round_trip_error_px.max())

    if max_error_px >= ROUND_TRIP_MAX_ERROR_PX:
        raise QuadValidationError(
            f"homography round-trip error {max_error_px:.4f}px exceeds "
            f"{ROUND_TRIP_MAX_ERROR_PX}px -- refusing to write a calibration "
            "this unreliable. This should not normally happen for 4 exact "
            "correspondences; check for duplicate/near-duplicate clicked points."
        )

    near_edge_px = float(np.linalg.norm(image_points[1] - image_points[0]))
    far_edge_px = float(np.linalg.norm(image_points[2] - image_points[3]))
    near_scale_m_per_px = width_m / near_edge_px if near_edge_px > 0 else float("nan")
    far_scale_m_per_px = width_m / far_edge_px if far_edge_px > 0 else float("nan")

    record = {
        "video": str(video_path),
        "frame_index": frame_index,
        "video_resolution": list(video_resolution),
        "image_points": image_points.tolist(),
        "world_points": world_points.tolist(),
        "H": H.tolist(),
        "H_inv": H_inv.tolist(),
        "world_width_m": width_m,
        "world_height_m": height_m,
        "source": source,
        "note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "round_trip_error_px": max_error_px,
        "near_edge_scale_m_per_px": near_scale_m_per_px,
        "far_edge_scale_m_per_px": far_scale_m_per_px,
    }
    return record


class _PointPicker:
    """Owns click state + rendering for the interactive OpenCV window.
    Kept separate from main() so the mouse-callback closure has a clean
    place to live."""

    def __init__(self, base_frame: np.ndarray) -> None:
        self.base_frame = base_frame
        self.points: list[tuple[int, int]] = []
        self.aborted = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def render(self) -> np.ndarray:
        img = self.base_frame.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(img, (x, y), 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(img, f"{i}:{CORNER_LABELS[i]}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=(len(self.points) == 4),
                          color=(0, 200, 255), thickness=2, lineType=cv2.LINE_AA)
        next_label = CORNER_LABELS[len(self.points)] if len(self.points) < 4 else "done"
        cv2.putText(
            img,
            f"click {len(self.points) + 1}/4 ({next_label})   u=undo  r=reset  q=abort",
            (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
        return img


def pick_points_interactive(frame: np.ndarray) -> np.ndarray:
    picker = _PointPicker(frame)
    window_name = "calibrate_video -- click near-left, near-right, far-right, far-left"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, picker.on_mouse)

    try:
        while True:
            cv2.imshow(window_name, picker.render())
            key = cv2.waitKey(20) & 0xFF
            if key == ord("u"):
                if picker.points:
                    picker.points.pop()
            elif key == ord("r"):
                picker.points.clear()
            elif key == ord("q"):
                picker.aborted = True
                break
            elif len(picker.points) == 4:
                cv2.imshow(window_name, picker.render())
                cv2.waitKey(300)
                break
    finally:
        cv2.destroyWindow(window_name)

    if picker.aborted:
        raise SystemExit("[calibrate_video] aborted ('q') -- no file written")
    return np.array(picker.points, dtype=np.float64)


def prompt_world_size() -> tuple[float, float, str, str]:
    width_raw = input(
        f"Real-world WIDTH of the clicked quad, metres [default {DEFAULT_ESTIMATED_WIDTH_M}]: "
    ).strip()
    height_raw = input(
        f"Real-world HEIGHT (near-to-far depth) of the clicked quad, metres "
        f"[default {DEFAULT_ESTIMATED_HEIGHT_M}]: "
    ).strip()

    if not width_raw and not height_raw:
        return DEFAULT_ESTIMATED_WIDTH_M, DEFAULT_ESTIMATED_HEIGHT_M, "ESTIMATED", ESTIMATED_NOTE

    def parse(raw: str, default: float, name: str) -> float:
        if not raw:
            print(f"[calibrate_video] no {name} given but the other field was -- "
                  f"using default {default}m for {name} too (mixed manual/default "
                  "input is unusual; re-run and answer both if that's not intended).")
            return default
        try:
            return float(raw)
        except ValueError:
            raise SystemExit(f"[calibrate_video] could not parse {name!r} as a number: {raw!r}")

    width_m = parse(width_raw, DEFAULT_ESTIMATED_WIDTH_M, "width")
    height_m = parse(height_raw, DEFAULT_ESTIMATED_HEIGHT_M, "height")
    note = "User-entered measurement via stdin at calibration time."
    return width_m, height_m, "USER_MEASURED", note


def grab_frame(video_path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    if frame_index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from: {video_path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default=None, help="default: calibration/<video_stem>.json")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path("calibration") / f"{Path(args.video).stem}.json"

    frame = grab_frame(args.video, args.frame)
    height_px, width_px = frame.shape[:2]
    print(f"[calibrate_video] video={args.video} frame={args.frame} "
          f"resolution={width_px}x{height_px}")

    image_points = pick_points_interactive(frame)
    print(f"[calibrate_video] clicked points (px): {image_points.tolist()}")

    width_m, height_m, source, note = prompt_world_size()

    try:
        record = build_calibration_record(
            image_points=image_points,
            width_m=width_m,
            height_m=height_m,
            source=source,
            note=note,
            video_path=args.video,
            frame_index=args.frame,
            video_resolution=(width_px, height_px),
        )
    except QuadValidationError as exc:
        raise SystemExit(f"[calibrate_video] refusing to write calibration: {exc}")

    print(f"[calibrate_video] homography round-trip error: {record['round_trip_error_px']:.4f}px "
          f"(must be < {ROUND_TRIP_MAX_ERROR_PX}px)")
    print(f"[calibrate_video] near-edge scale: {record['near_edge_scale_m_per_px'] * 1000:.2f} mm/px   "
          f"far-edge scale: {record['far_edge_scale_m_per_px'] * 1000:.2f} mm/px   "
          f"foreshortening ratio (far/near): "
          f"{record['far_edge_scale_m_per_px'] / record['near_edge_scale_m_per_px']:.2f}x")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"[calibrate_video] source={source}  wrote {out_path}")


if __name__ == "__main__":
    main()
