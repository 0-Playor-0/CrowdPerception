"""Top-down ground-plane density heatmap over a rolling temporal window.

Deliberately NOT analytics/pressure.py and NOT calibrated alarm tiers --
this is perception-only visualisation. It answers "roughly where has it
been busy over the last couple of seconds", with a color scale that means
the same thing every frame; it does not classify anything as safe/warning/
danger.

GroundHeatmap itself is coordinate-system-agnostic -- nothing in its
accumulation math assumes metres. scripts/live_perception.py uses it two
ways: once over the calibrated quad in WORLD METRES (real persons/m²,
"density" mode meaningful), and once over the whole camera frame in PIXEL
space for the on-video overlay (render_heatmap_overlay below) where
"density" would be persons/px² -- meaningless, so only "count" mode is
used for that one. The homography that produces world coordinates is only
valid inside the calibrated quad (see perception/ground.py) -- there is no
honest way to extend real persons/m² density past it, which is why the
whole-frame view is pixel-space occupancy, not an extrapolated density.

Two design choices worth calling out:

1. Rolling temporal window instead of instantaneous per-frame counts. A
   single frame's occupancy grid is visually noisy (a person on a cell
   boundary flickers between cells frame to frame; a brief detection
   dropout blanks a cell for one frame). Averaging over the last
   window_seconds smooths that out. The accumulator stores each frame's
   in-quad world points with its timestamp and divides the summed
   per-cell counts by the number of frames actually in the window -- so a
   single person standing still in one cell reads as ~1 occupant, not
   window_frames occupants, regardless of the source video's fps.

2. Fixed colour scale (vmax), not auto-scaled per frame. An auto-scaling
   heatmap remaps its own colour range to whatever the current frame's max
   happens to be, which means "red" means a different density every frame
   -- actively misleading for anything you'd read density off of by eye.
   vmax is fixed for the whole run (a constructor argument with a plain
   default) and rendered as a printed number so the viewer knows what the
   top of the scale represents; the actual observed peak this frame is
   reported separately (grid.max()), not used to move the scale.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Density colour ramp -- control points, not a formula.
#
# 4.0 persons/m^2 (DEFAULT_DENSITY_VMAX_PERSONS_PER_M2) is the fixed colour
# ceiling: density mode's scale always tops out (full red) here. 2.5
# persons/m^2 (ALARM_ONSET_PERSONS_PER_M2) is the point the ramp is tuned to
# already read as "clearly red" -- the visual alarm point. 3.0 persons/m^2
# (CRUSH_BENCHMARK_PERSONS_PER_M2) is the crush-density benchmark this
# project's docs have cited throughout (docs/REAL_FOOTAGE_FINDINGS.md).
#
# All three of these are DISPLAY CHOICES -- where the colour ramp is tuned
# to look alarming -- not measured or calibrated safety thresholds. Do not
# read "2.5" or "4.0" as validated limits; they were picked so the render
# reads correctly to a human, nothing more.
#
# Two ramp variants, same endpoints, different transition width around the
# alarm onset:
#   - COLOR_STOPS_HARD:  a ~0.2 persons/m^2-wide ramp into red right at 2.5.
#     Sharp, maximally legible edge -- but a cell whose measured density
#     oscillates across 2.5 frame to frame will visibly flicker between
#     "orange" and "red".
#   - COLOR_STOPS_EASED: the same endpoints, but the ramp into red starts
#     earlier (~0.9 persons/m^2-wide) so the slope through the alarm point
#     is gentler -- less flicker on borderline cells, at the cost of a
#     less crisp-looking edge exactly at 2.5.
# Both stop lists are defined in "reference" persons/m^2 units (0 to
# DEFAULT_DENSITY_VMAX_PERSONS_PER_M2); _piecewise_colormap rescales
# whatever vmax a given GroundHeatmap actually uses onto this same 0..4.0
# reference range, so the ramp SHAPE is shared by density mode, count mode,
# and the pixel-space camera overlay even though their vmax values differ
# and mean different things (see render_heatmap_overlay).
# ---------------------------------------------------------------------------

DEFAULT_DENSITY_VMAX_PERSONS_PER_M2 = 4.0
ALARM_ONSET_PERSONS_PER_M2 = 2.5
CRUSH_BENCHMARK_PERSONS_PER_M2 = 3.0

_PALE_YELLOW = (200, 255, 255)   # BGR
_AMBER = (0, 170, 255)
_DEEP_ORANGE = (0, 100, 255)
_ALARM_RED = (0, 30, 255)        # target colour AT the alarm onset, both variants
_FULL_RED = (0, 0, 255)

COLOR_STOPS_HARD: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, _PALE_YELLOW),
    (1.0, _AMBER),
    (2.3, _DEEP_ORANGE),                          # still pre-alarm
    (ALARM_ONSET_PERSONS_PER_M2, _ALARM_RED),      # 2.5 -- sharp jump, only 0.2 wide from the previous stop
    (DEFAULT_DENSITY_VMAX_PERSONS_PER_M2, _FULL_RED),
]

COLOR_STOPS_EASED: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, _PALE_YELLOW),
    (1.0, _AMBER),
    (1.6, _DEEP_ORANGE),                          # transition starts earlier
    (ALARM_ONSET_PERSONS_PER_M2, _ALARM_RED),      # 2.5 -- same target colour, ~0.9-wide approach
    (DEFAULT_DENSITY_VMAX_PERSONS_PER_M2, _FULL_RED),
]

COLOR_VARIANTS = {"hard": COLOR_STOPS_HARD, "eased": COLOR_STOPS_EASED}


def _piecewise_colormap(reference_values: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Piecewise-linear interpolation between explicit (value, BGR) control
    points. `reference_values` must already be expressed in the same units
    the stops are defined in (persons/m^2, 0..DEFAULT_DENSITY_VMAX_PERSONS_PER_M2
    by convention here) -- callers with a different vmax rescale first (see
    render_heatmap_panel / render_heatmap_overlay). Values outside the
    stops' range clamp to the nearest endpoint colour."""
    ordered = sorted(stops, key=lambda s: s[0])
    xs = np.array([s[0] for s in ordered], dtype=np.float64)
    colors = np.array([s[1] for s in ordered], dtype=np.float64)   # (n_stops, 3) BGR

    flat = reference_values.astype(np.float64).ravel()
    out = np.empty((flat.size, 3), dtype=np.float64)
    for channel in range(3):
        out[:, channel] = np.interp(flat, xs, colors[:, channel])
    return out.reshape(reference_values.shape + (3,)).astype(np.uint8)


class GroundHeatmap:
    def __init__(
        self,
        quad_m: np.ndarray,
        cell_size_m: float = 0.5,
        window_seconds: float = 2.0,
        sigma: float = 1.0,
        count_vmax: float = 3.0,
        density_vmax: float | None = None,
    ) -> None:
        quad_m = np.asarray(quad_m, dtype=np.float64)
        if quad_m.shape != (4, 2):
            raise ValueError(f"quad_m must be shape (4, 2), got {quad_m.shape}")
        if cell_size_m <= 0:
            raise ValueError(f"cell_size_m must be > 0, got {cell_size_m}")

        x_min, y_min = quad_m.min(axis=0)
        x_max, y_max = quad_m.max(axis=0)
        self._origin = (float(x_min), float(y_min))
        self._cell_size_m = float(cell_size_m)
        self.n_cols = max(1, int(np.ceil((x_max - x_min) / cell_size_m)))
        self.n_rows = max(1, int(np.ceil((y_max - y_min) / cell_size_m)))

        self._window_seconds = window_seconds
        self._sigma = sigma
        self._history: deque[tuple[float, np.ndarray]] = deque()

        self._count_vmax = count_vmax
        # NOT derived from count_vmax/cell_size -- density mode's scale is a
        # fixed, real-world benchmark (see DEFAULT_DENSITY_VMAX_PERSONS_PER_M2),
        # independent of whatever count_vmax happens to be set to.
        self._density_vmax = (
            density_vmax if density_vmax is not None else DEFAULT_DENSITY_VMAX_PERSONS_PER_M2
        )

    @property
    def cell_size_m(self) -> float:
        return self._cell_size_m

    def cell_area_m2(self) -> float:
        return self._cell_size_m ** 2

    @property
    def origin_m(self) -> tuple[float, float]:
        return self._origin

    def update(self, t_sec: float, world_xy_in_quad: np.ndarray) -> None:
        """Feed one frame's worth of in-quad world points. Points outside
        the quad's bounding box are silently dropped by _raw_count_grid
        (out of grid range) -- callers should already have excluded
        out-of-quad detections via perception.ground.GroundProjector before
        calling this, this is just a second line of defense."""
        points = np.asarray(world_xy_in_quad, dtype=np.float64).reshape(-1, 2)
        self._history.append((t_sec, points))
        cutoff = t_sec - self._window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def reset(self) -> None:
        self._history.clear()

    def _raw_count_grid(self) -> np.ndarray:
        """Mean per-frame occupancy count per cell, averaged over every
        frame currently in the rolling window (not just frames with
        people in them -- an empty frame still counts toward the
        denominator, correctly pulling the average down when the window
        spans a mix of busy and empty frames)."""
        grid = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
        n_frames = len(self._history)
        if n_frames == 0:
            return grid

        x_min, y_min = self._origin
        for _, points in self._history:
            if points.size == 0:
                continue
            col = np.floor((points[:, 0] - x_min) / self._cell_size_m).astype(np.int64)
            row = np.floor((points[:, 1] - y_min) / self._cell_size_m).astype(np.int64)
            in_range = (row >= 0) & (row < self.n_rows) & (col >= 0) & (col < self.n_cols)
            for r, c in zip(row[in_range], col[in_range]):
                grid[r, c] += 1.0
        return grid / n_frames

    def grid(self, mode: str) -> np.ndarray:
        """mode: 'count' (persons per cell) or 'density' (persons/m^2).
        Gaussian-smoothed (sigma=0 disables smoothing) in both cases."""
        raw = self._raw_count_grid()
        smoothed = gaussian_filter(raw, sigma=self._sigma) if self._sigma > 0 else raw
        if mode == "count":
            return smoothed
        if mode == "density":
            return smoothed / self.cell_area_m2()
        raise ValueError(f"heatmap mode must be 'count' or 'density', got {mode!r}")

    def vmax(self, mode: str) -> float:
        if mode == "count":
            return self._count_vmax
        if mode == "density":
            return self._density_vmax
        raise ValueError(f"heatmap mode must be 'count' or 'density', got {mode!r}")

    def n_frames_in_window(self) -> int:
        return len(self._history)


# ---------------------------------------------------------------------------
# Rendering: top-down panel, fixed colour scale, legend, quad outline + metre
# grid. Kept in this module (rather than scripts/live_perception.py) because
# it only needs a GroundHeatmap and a quad -- no camera-pane state -- and
# having the render logic next to the grid math it renders makes it possible
# to unit-test panel dimensions/legend text without spinning up the whole
# live script.
# ---------------------------------------------------------------------------

LEGEND_WIDTH_PX = 90
MARGIN_PX = 28


def _grid_to_bgr(grid: np.ndarray, vmax: float, color_variant: str) -> np.ndarray:
    """Rescales grid values (whatever units/vmax this GroundHeatmap uses)
    onto the 0..DEFAULT_DENSITY_VMAX_PERSONS_PER_M2 reference range the
    colour stops are defined in, then applies the piecewise ramp. This is
    what lets density mode (vmax=4.0, native units) and count mode / the
    pixel overlay (arbitrary other vmax) share the same ramp SHAPE."""
    stops = COLOR_VARIANTS[color_variant]
    reference = np.clip(grid, 0.0, vmax) / max(vmax, 1e-9) * DEFAULT_DENSITY_VMAX_PERSONS_PER_M2
    return _piecewise_colormap(reference, stops)


def render_heatmap_panel(
    heatmap: GroundHeatmap,
    quad_m: np.ndarray,
    mode: str,
    panel_height_px: int,
    color_variant: str = "hard",
) -> np.ndarray:
    """Renders the top-down density panel: coloured grid (fixed vmax scale),
    quad outline, 1m grid lines, and a legend strip with the fixed vmax and
    the current frame's observed max printed. Returns a BGR uint8 image
    exactly panel_height_px tall, ready to sit next to the camera pane.

    color_variant: "hard" or "eased" -- see COLOR_STOPS_HARD/COLOR_STOPS_EASED."""
    grid = heatmap.grid(mode)
    vmax = heatmap.vmax(mode)
    observed_max = float(grid.max()) if grid.size else 0.0

    plot_height_px = panel_height_px - 2 * MARGIN_PX
    px_per_m = plot_height_px / max(heatmap.n_rows * heatmap.cell_size_m, 1e-6)
    plot_width_px = max(1, round(heatmap.n_cols * heatmap.cell_size_m * px_per_m))

    grid_color = _grid_to_bgr(grid, vmax, color_variant)
    # nearest-neighbour upscale -- cells should read as flat blocks, not blurred
    grid_img = cv2.resize(grid_color, (plot_width_px, plot_height_px), interpolation=cv2.INTER_NEAREST)
    # flip vertically: world y grows AWAY from the camera (far edge), and the
    # panel should read like a map with "near" (camera-side) at the bottom.
    grid_img = cv2.flip(grid_img, 0)

    panel_width_px = plot_width_px + LEGEND_WIDTH_PX + 3 * MARGIN_PX
    panel = np.full((panel_height_px, panel_width_px, 3), 24, dtype=np.uint8)
    panel[MARGIN_PX:MARGIN_PX + plot_height_px, MARGIN_PX:MARGIN_PX + plot_width_px] = grid_img

    def world_to_panel_xy(x_m: float, y_m: float) -> tuple[int, int]:
        origin_x, origin_y = heatmap.origin_m
        px = MARGIN_PX + (x_m - origin_x) * px_per_m
        py = MARGIN_PX + plot_height_px - (y_m - origin_y) * px_per_m
        return int(round(px)), int(round(py))

    # 1m metre grid, drawn faint over the density image
    origin_x, origin_y = heatmap.origin_m
    x_m = origin_x
    while x_m <= origin_x + heatmap.n_cols * heatmap.cell_size_m + 1e-6:
        p0 = world_to_panel_xy(x_m, origin_y)
        p1 = world_to_panel_xy(x_m, origin_y + heatmap.n_rows * heatmap.cell_size_m)
        cv2.line(panel, p0, p1, (90, 90, 90), 1, cv2.LINE_AA)
        x_m += 1.0
    y_m = origin_y
    while y_m <= origin_y + heatmap.n_rows * heatmap.cell_size_m + 1e-6:
        p0 = world_to_panel_xy(origin_x, y_m)
        p1 = world_to_panel_xy(origin_x + heatmap.n_cols * heatmap.cell_size_m, y_m)
        cv2.line(panel, p0, p1, (90, 90, 90), 1, cv2.LINE_AA)
        y_m += 1.0

    # calibrated quad outline (may be a trapezoid, not the axis-aligned grid bbox)
    quad_panel_pts = np.array([world_to_panel_xy(x, y) for x, y in quad_m], dtype=np.int32)
    cv2.polylines(panel, [quad_panel_pts], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # legend: vertical gradient bar, high value at top, on the SAME ramp/variant as the grid
    legend_x0 = MARGIN_PX + plot_width_px + MARGIN_PX
    legend_x1 = legend_x0 + LEGEND_WIDTH_PX // 3
    legend_values = np.linspace(vmax, 0.0, plot_height_px, dtype=np.float64).reshape(-1, 1)
    gradient_color = _grid_to_bgr(legend_values, vmax, color_variant)
    gradient_color = np.repeat(gradient_color, legend_x1 - legend_x0, axis=1)
    panel[MARGIN_PX:MARGIN_PX + plot_height_px, legend_x0:legend_x1] = gradient_color
    cv2.rectangle(panel, (legend_x0, MARGIN_PX), (legend_x1, MARGIN_PX + plot_height_px), (255, 255, 255), 1)

    unit = "persons" if mode == "count" else "p/m^2"

    def legend_tick_y(value: float) -> int:
        frac = np.clip(value / vmax, 0.0, 1.0) if vmax > 0 else 0.0
        return int(round(MARGIN_PX + (1.0 - frac) * plot_height_px))

    cv2.putText(panel, f"{vmax:.1f} {unit}", (legend_x1 + 4, MARGIN_PX + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"0 {unit}", (legend_x1 + 4, MARGIN_PX + plot_height_px),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Labelled ticks at the alarm onset and crush benchmark -- density mode
    # only, since these are persons/m^2 values with no meaning in count mode.
    if mode == "density":
        for value, label, color in (
            (ALARM_ONSET_PERSONS_PER_M2, "ALARM", (0, 60, 255)),
            (CRUSH_BENCHMARK_PERSONS_PER_M2, "CRUSH", (0, 200, 255)),
        ):
            if 0.0 <= value <= vmax:
                y = legend_tick_y(value)
                cv2.line(panel, (legend_x0 - 4, y), (legend_x1 + 3, y), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, f"{value:.1f} {label}", (legend_x1 + 4, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    cv2.putText(panel, f"scale max: {vmax:.2f} {unit}  ({color_variant})", (MARGIN_PX, panel_height_px - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"cur. max: {observed_max:.2f} {unit}  mode={mode}  win={heatmap.n_frames_in_window()}f",
                (MARGIN_PX, panel_height_px - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return panel


def render_heatmap_overlay(
    heatmap: GroundHeatmap,
    base_img: np.ndarray,
    mode: str = "count",
    max_alpha: float = 0.7,
    gamma: float = 0.5,
    color_variant: str = "hard",
) -> np.ndarray:
    """Blends a heatmap directly onto base_img in that image's own pixel
    coordinates -- for a GroundHeatmap constructed over the FULL FRAME's
    pixel bounding box (see scripts/live_perception.py's frame_heatmap),
    so grid cell (r, c) maps linearly onto a region of base_img with no
    separate coordinate transform needed.

    UNCALIBRATED relative occupancy, not persons/m^2 -- this heatmap is
    built in pixel space with its own vmax (persons/cell in an arbitrary
    pixel-sized cell), unrelated to the metric panel's density_vmax/2.5/4.0.
    It shares the SAME colour ramp shape (see _grid_to_bgr) purely for
    visual consistency across the two panes -- "redder here" still means
    "closer to this pane's own vmax," never a specific persons/m^2 figure.
    Always label this pane as relative/uncalibrated wherever it's shown.

    Per-pixel alpha is proportional to that pixel's normalised value (0 at
    empty cells, up to max_alpha at vmax) -- empty regions stay fully
    see-through (the video shows through untouched) and only occupied
    regions get tinted, rather than a flat translucent layer sitting over
    the whole frame regardless of content.

    gamma < 1 (default 0.5, i.e. sqrt) shapes ALPHA (visibility) only, not
    colour -- it compresses the visual range so lightly-occupied cells are
    still visibly tinted instead of reading as fully transparent. This
    pixel-space grid divides by however many frames are in the rolling
    window, over many more/smaller cells than the metric panel, so raw
    values are routinely a few percent of vmax even with real people in
    frame (verified: a 20-person cluster over 56 frames of window peaks
    around 2.2, but an evenly-scattered crowd across the same frame peaks
    under 0.1 -- gamma keeps that visible without changing the underlying
    numbers, or the colour-to-density mapping, which stays linear-in-the-
    piecewise-stops exactly like the metric panel's."""
    grid = heatmap.grid(mode)
    vmax = heatmap.vmax(mode)
    normalized = np.clip(grid, 0.0, vmax) / max(vmax, 1e-9)

    h, w = base_img.shape[:2]
    norm_resized = cv2.resize(normalized.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    color = _grid_to_bgr(norm_resized * vmax, vmax, color_variant)

    alpha_shape = norm_resized ** gamma if gamma != 1.0 else norm_resized
    alpha = (alpha_shape * max_alpha)[..., None]
    blended = base_img.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return blended.astype(np.uint8)
