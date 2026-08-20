"""Append-only per-frame trajectory logging + the track-churn diagnostic.

Writes one CSV row per (track_id, frame) -- position only, no velocity (that
is deliberately deferred to a later fitting pass over the persisted CSV, not
computed here). Flushed to disk periodically so a crash mid-video loses at
most the last partial flush window, not the whole run.

Also computes and writes a <video_stem>_track_stats.json sidecar at close()
time: unique track count, track lifetime distribution, and the fraction of
tracks that survive at least 1.0s / 2.5s. This is the tracker-churn
diagnostic explicitly asked for -- the prior real-footage run
(docs/REAL_FOOTAGE_FINDINGS.md, C2.2) produced 93 confirmed track IDs in an
18s clip with confirmed-count sitting at 0-2 per frame almost the entire
run, i.e. tracks were being confirmed, lost, and reassigned a new ID rather
than persisted. That number is reproduced here as a live, per-run
diagnostic so you can see directly whether tiled detection (much better
recall, per C2.3) makes that better or worse -- it does NOT fix OC-SORT's
confirmation/re-ID behaviour by itself, and this task does not swap the
tracker, so don't expect this number to have magically improved.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "conf",
    "foot_px_x",
    "foot_px_y",
    "world_x_m",
    "world_y_m",
    "in_quad",
    "tile_source",
]


@dataclass
class _TrackLifetime:
    first_t: float
    last_t: float
    n_observations: int


class TrajectoryLogger:
    """One instance per video run. Call log_row() once per (track, frame),
    then close() exactly once at the end (also safe to call from a
    KeyboardInterrupt/'q'-quit handler -- it flushes and writes the stats
    sidecar regardless of how the run ended)."""

    def __init__(
        self,
        csv_path: Path,
        parquet_path: Path | None = None,
        flush_every_n_rows: int = 50,
        flush_every_seconds: float = 2.0,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self._csv_path.open("w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(COLUMNS)
        self._csv_file.flush()

        self._parquet_path = Path(parquet_path) if parquet_path else None
        # Buffered in memory only if parquet output is actually requested --
        # no reason to hold every record in RAM for a CSV-only run.
        self._parquet_rows: list[dict] | None = [] if self._parquet_path else None

        self._flush_every_n_rows = flush_every_n_rows
        self._flush_every_seconds = flush_every_seconds
        self._rows_since_flush = 0
        self._last_flush_time = time.monotonic()

        self._track_lifetimes: dict[int, _TrackLifetime] = {}
        self._per_frame_track_count: dict[int, int] = defaultdict(int)
        self._n_rows_written = 0
        self._closed = False

        print(f"[trajectory] writing {self._csv_path}"
              + (f" (+ {self._parquet_path})" if self._parquet_path else ""))

    def log_row(
        self,
        frame_idx: int,
        timestamp_s: float,
        track_id: int,
        bbox_xyxy: tuple[float, float, float, float],
        conf: float,
        foot_px: tuple[float, float],
        world_xy: tuple[float, float],
        in_quad: bool,
        tile_source: str,
    ) -> None:
        x1, y1, x2, y2 = bbox_xyxy
        fx, fy = foot_px
        wx, wy = world_xy
        row = [
            frame_idx, timestamp_s, track_id,
            x1, y1, x2, y2,
            conf, fx, fy, wx, wy,
            int(bool(in_quad)), tile_source,
        ]
        self._csv_writer.writerow(row)
        self._n_rows_written += 1
        self._rows_since_flush += 1

        if self._parquet_rows is not None:
            self._parquet_rows.append(dict(zip(COLUMNS, row)))

        lifetime = self._track_lifetimes.get(track_id)
        if lifetime is None:
            self._track_lifetimes[track_id] = _TrackLifetime(timestamp_s, timestamp_s, 1)
        else:
            lifetime.last_t = timestamp_s
            lifetime.n_observations += 1

        self._per_frame_track_count[frame_idx] += 1

        self._maybe_flush()

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        due_by_count = self._rows_since_flush >= self._flush_every_n_rows
        due_by_time = (now - self._last_flush_time) >= self._flush_every_seconds
        if due_by_count or due_by_time:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())
            self._rows_since_flush = 0
            self._last_flush_time = now

    @property
    def n_rows_written(self) -> int:
        return self._n_rows_written

    @property
    def n_unique_tracks(self) -> int:
        return len(self._track_lifetimes)

    def compute_stats(self, total_frames_processed: int) -> dict:
        """Computable at any point (not just at close()) so a caller can
        print a live-updating churn readout, but the final sidecar is
        always written from a call inside close()."""
        lifetimes_s = np.array(
            [lt.last_t - lt.first_t for lt in self._track_lifetimes.values()],
            dtype=np.float64,
        )
        n = int(lifetimes_s.size)

        def pct(p: float) -> float:
            return float(np.percentile(lifetimes_s, p)) if n else 0.0

        mean_simultaneous = (
            sum(self._per_frame_track_count.values()) / total_frames_processed
            if total_frames_processed > 0 else 0.0
        )

        return {
            "n_unique_track_ids": n,
            "n_rows_written": self._n_rows_written,
            "total_frames_processed": total_frames_processed,
            "track_lifetime_seconds": {
                "min": float(lifetimes_s.min()) if n else 0.0,
                "median": pct(50),
                "p90": pct(90),
                "max": float(lifetimes_s.max()) if n else 0.0,
            },
            "fraction_surviving_ge_1_0s": float(np.mean(lifetimes_s >= 1.0)) if n else 0.0,
            "fraction_surviving_ge_2_5s": float(np.mean(lifetimes_s >= 2.5)) if n else 0.0,
            "mean_simultaneous_track_count_per_frame": mean_simultaneous,
        }

    def close(self, total_frames_processed: int, stats_path: Path) -> dict:
        if self._closed:
            return self.compute_stats(total_frames_processed)
        self._closed = True

        self._csv_file.flush()
        self._csv_file.close()

        if self._parquet_rows is not None:
            try:
                import polars as pl
            except ImportError:
                print("[trajectory] polars not installed -- skipping parquet output "
                      f"({self._parquet_path} not written); CSV is unaffected.")
            else:
                pl.DataFrame(self._parquet_rows).write_parquet(str(self._parquet_path))
                print(f"[trajectory] wrote {self._parquet_path} "
                      f"({len(self._parquet_rows)} rows)")

        stats = self.compute_stats(total_frames_processed)
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w") as f:
            json.dump(stats, f, indent=2)

        self._print_summary(stats, stats_path)
        return stats

    @staticmethod
    def _print_summary(stats: dict, stats_path: Path) -> None:
        lt = stats["track_lifetime_seconds"]
        print("\n[trajectory] ==== track churn summary ====")
        print(f"  unique track IDs: {stats['n_unique_track_ids']}  "
              f"(over {stats['total_frames_processed']} frames, "
              f"{stats['n_rows_written']} rows)")
        print(f"  lifetime (s): min={lt['min']:.2f}  median={lt['median']:.2f}  "
              f"p90={lt['p90']:.2f}  max={lt['max']:.2f}")
        print(f"  surviving >= 1.0s: {100 * stats['fraction_surviving_ge_1_0s']:.1f}%")
        print(f"  surviving >= 2.5s: {100 * stats['fraction_surviving_ge_2_5s']:.1f}%")
        print(f"  mean simultaneous tracked people/frame: "
              f"{stats['mean_simultaneous_track_count_per_frame']:.2f}")
        print(f"  wrote {stats_path}")
