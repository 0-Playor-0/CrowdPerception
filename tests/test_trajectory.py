"""Tests for perception/trajectory.py: CSV schema, row-count reconciliation
against a synthetic per-frame funnel, and the track-churn stats sidecar."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from perception.trajectory import COLUMNS, TrajectoryLogger


def _log_synthetic_run(logger: TrajectoryLogger) -> dict[int, int]:
    """Logs a small synthetic scenario with a KNOWN funnel (per-frame
    emitted-row count) and known per-track lifetimes:

    - track 1: observed frames 0..29 (30 rows @ 30fps -> lifetime ~0.967s,
      i.e. just under the 1.0s bucket)
    - track 2: observed frames 0..89 (90 rows @ 30fps -> lifetime ~2.967s,
      over both the 1.0s and 2.5s buckets)
    - track 3: observed only frame 5 (1 row -> lifetime 0.0s)

    Returns the expected per-frame emitted-row funnel {frame_idx: n_rows}.
    """
    fps = 30.0
    funnel: dict[int, int] = {}

    def emit(frame_idx: int, track_id: int) -> None:
        t = frame_idx / fps
        logger.log_row(
            frame_idx=frame_idx, timestamp_s=t, track_id=track_id,
            bbox_xyxy=(10.0, 10.0, 20.0, 40.0), conf=0.8,
            foot_px=(15.0, 40.0), world_xy=(1.0, 2.0),
            in_quad=True, tile_source="tiled",
        )
        funnel[frame_idx] = funnel.get(frame_idx, 0) + 1

    for frame_idx in range(30):
        emit(frame_idx, track_id=1)
    for frame_idx in range(90):
        emit(frame_idx, track_id=2)
    emit(5, track_id=3)

    return funnel


def test_csv_schema_matches_declared_columns(tmp_path: Path) -> None:
    logger = TrajectoryLogger(csv_path=tmp_path / "traj.csv")
    logger.log_row(
        frame_idx=0, timestamp_s=0.0, track_id=1,
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0), conf=0.5,
        foot_px=(2.0, 4.0), world_xy=(0.5, 0.5),
        in_quad=True, tile_source="untiled",
    )
    logger.close(total_frames_processed=1, stats_path=tmp_path / "stats.json")

    with (tmp_path / "traj.csv").open("r", newline="") as f:
        header = next(csv.reader(f))
    assert header == COLUMNS


def test_row_count_reconciles_against_per_frame_funnel(tmp_path: Path) -> None:
    logger = TrajectoryLogger(csv_path=tmp_path / "traj.csv")
    expected_funnel = _log_synthetic_run(logger)
    total_frames_processed = 90   # the longest track's span
    stats = logger.close(total_frames_processed, stats_path=tmp_path / "stats.json")

    with (tmp_path / "traj.csv").open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == sum(expected_funnel.values()) == 30 + 90 + 1

    rows_per_frame: dict[int, int] = {}
    for row in rows:
        frame_idx = int(row["frame_idx"])
        rows_per_frame[frame_idx] = rows_per_frame.get(frame_idx, 0) + 1
    assert rows_per_frame == expected_funnel, "CSV row count per frame must exactly match the funnel that produced it"

    assert stats["n_rows_written"] == len(rows)


def test_track_lifetime_stats_and_survival_fractions(tmp_path: Path) -> None:
    logger = TrajectoryLogger(csv_path=tmp_path / "traj.csv")
    _log_synthetic_run(logger)
    stats = logger.close(total_frames_processed=90, stats_path=tmp_path / "stats.json")

    assert stats["n_unique_track_ids"] == 3
    # track 1: 29/30 = 0.9667s (< 1.0s), track 2: 89/30 = 2.9667s (>= 2.5s), track 3: 0.0s
    assert stats["track_lifetime_seconds"]["min"] == pytest.approx(0.0, abs=1e-9)
    assert stats["track_lifetime_seconds"]["max"] == pytest.approx(89 / 30.0, abs=1e-6)
    # exactly 1/3 tracks (track 2) clears each bar
    assert stats["fraction_surviving_ge_1_0s"] == pytest.approx(1 / 3)
    assert stats["fraction_surviving_ge_2_5s"] == pytest.approx(1 / 3)


def test_sidecar_json_written_with_expected_keys(tmp_path: Path) -> None:
    logger = TrajectoryLogger(csv_path=tmp_path / "traj.csv")
    _log_synthetic_run(logger)
    stats_path = tmp_path / "video_track_stats.json"
    logger.close(total_frames_processed=90, stats_path=stats_path)

    assert stats_path.exists()
    with stats_path.open("r") as f:
        on_disk = json.load(f)
    for key in (
        "n_unique_track_ids", "track_lifetime_seconds",
        "fraction_surviving_ge_1_0s", "fraction_surviving_ge_2_5s",
        "mean_simultaneous_track_count_per_frame",
    ):
        assert key in on_disk


def test_mean_simultaneous_track_count(tmp_path: Path) -> None:
    logger = TrajectoryLogger(csv_path=tmp_path / "traj.csv")
    # 2 simultaneous tracks for 10 frames, total_frames_processed=20 (10 empty frames)
    for frame_idx in range(10):
        for track_id in (1, 2):
            logger.log_row(
                frame_idx=frame_idx, timestamp_s=frame_idx / 30.0, track_id=track_id,
                bbox_xyxy=(0.0, 0.0, 1.0, 1.0), conf=0.9,
                foot_px=(0.5, 1.0), world_xy=(0.0, 0.0),
                in_quad=True, tile_source="tiled",
            )
    stats = logger.close(total_frames_processed=20, stats_path=tmp_path / "stats.json")
    # 20 rows total / 20 frames processed = 1.0 mean simultaneous tracks
    assert stats["mean_simultaneous_track_count_per_frame"] == pytest.approx(1.0)
