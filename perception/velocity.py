"""Per-track world-position history, smoothed velocity, and outlier rejection.

Raw per-frame world positions are noisy (detection jitter, occasional
tracker ID switches). Differencing two consecutive frames amplifies that
noise directly into the velocity estimate, so instead each track keeps a
short sliding window of recent (t_sec, x_world, y_world) samples and fits a
straight line through the window -- velocity is the fit's slope, not a
two-point difference.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityEstimate:
    x_world: float          # metres, current (latest accepted) ground-plane position
    y_world: float          # metres
    v_x: float               # m/s, least-squares fit slope over the window
    v_y: float               # m/s
    window_full: bool        # True only once the window has window_frames samples --
                              # exclude window_full=False rows from downstream
                              # variance/statistics, the fit is under-constrained before then.


class VelocityTracker:
    """Tracks per-track_id position history and produces smoothed velocity.

    window_frames: number of samples kept per track (~= window_seconds * fps,
        computed by the caller since only it knows the video's fps).
    max_speed_mps: reject any incoming sample whose implied instantaneous
        speed from the last ACCEPTED sample exceeds this. This is a filter
        for tracker ID switches (a person's track_id jumping to a different
        person elsewhere in frame implies 15-20+ m/s), not a cap on how fast
        a real person can move -- keep this above realistic sprinting speed
        (~8-9 m/s) or you'll reject the exact "running crowd" events this
        system exists to detect.
    """

    def __init__(self, window_frames: int, max_speed_mps: float) -> None:
        if window_frames < 2:
            raise ValueError("window_frames must be >= 2 to fit a velocity")
        self._window_frames = window_frames
        self._max_speed_mps = max_speed_mps
        self._history: dict[int, deque[tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=window_frames)
        )
        # Rejection counters, kept per-track so you can see whether a
        # specific track_id is flaky (likely a real ID switch) versus
        # scattered single rejections across many tracks (likely detector
        # jitter near the outlier threshold).
        self.rejected_per_track: dict[int, int] = defaultdict(int)
        self.accepted_per_track: dict[int, int] = defaultdict(int)

    def update(
        self, track_id: int, t_sec: float, x_world: float, y_world: float
    ) -> VelocityEstimate | None:
        """Feed one new observation for track_id. Returns None if the
        sample was rejected as an outlier -- caller should skip emitting a
        record for this track this frame, the position is untrustworthy."""
        history = self._history[track_id]

        if history:
            last_t, last_x, last_y = history[-1]
            dt = t_sec - last_t
            if dt > 0:
                implied_speed = float(np.hypot(x_world - last_x, y_world - last_y) / dt)
                if implied_speed > self._max_speed_mps:
                    self.rejected_per_track[track_id] += 1
                    return None
            # dt <= 0 (duplicate/non-monotonic timestamp) skips the check --
            # timestamps are expected to be strictly increasing per frame;
            # that's a caller bug, not something this filter should mask.

        history.append((t_sec, x_world, y_world))
        self.accepted_per_track[track_id] += 1

        if len(history) < 2:
            # Can't fit a line through a single point -- report zero
            # velocity rather than fabricating a number from insufficient data.
            return VelocityEstimate(x_world, y_world, 0.0, 0.0, window_full=False)

        # Least-squares linear fit of x(t) and y(t) over the window; the fit
        # slope is the smoothed velocity. Far less noisy than differencing
        # two consecutive frames, at the cost of some lag (up to
        # window_seconds) before velocity reflects a sudden direction change.
        t_arr = np.array([sample[0] for sample in history], dtype=np.float64)
        x_arr = np.array([sample[1] for sample in history], dtype=np.float64)
        y_arr = np.array([sample[2] for sample in history], dtype=np.float64)

        v_x, _ = np.polyfit(t_arr, x_arr, deg=1)
        v_y, _ = np.polyfit(t_arr, y_arr, deg=1)

        window_full = len(history) == self._window_frames
        return VelocityEstimate(x_world, y_world, float(v_x), float(v_y), window_full)

    @property
    def total_rejected(self) -> int:
        return sum(self.rejected_per_track.values())

    @property
    def total_accepted(self) -> int:
        return sum(self.accepted_per_track.values())

    def log_summary(self) -> None:
        """Prints rejection stats per track and overall -- lets you eyeball
        tracker quality; a track_id with many rejections usually means an
        ID switch happened on that track."""
        total = self.total_accepted + self.total_rejected
        if total == 0:
            print("[velocity] no samples processed")
            return
        pct = 100.0 * self.total_rejected / total
        print(
            f"[velocity] rejected {self.total_rejected}/{total} samples "
            f"({pct:.2f}%) as outliers (> {self._max_speed_mps} m/s implied)"
        )
        flagged = {tid: n for tid, n in self.rejected_per_track.items() if n > 0}
        if flagged:
            print("[velocity] rejections by track_id:")
            for tid, n in sorted(flagged.items(), key=lambda kv: -kv[1]):
                print(f"  track_id={tid}: {n} rejected")
