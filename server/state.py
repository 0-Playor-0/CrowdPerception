"""Shared, thread-safe state between the background pipeline thread and the
FastAPI event loop. The pipeline thread is the only writer; HTTP/WS handlers
are readers plus the occasional start/stop signal -- everything crossing
that boundary goes through `_lock`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    RUNNING = "running"
    ERROR = "error"


def idle_telemetry() -> dict[str, Any]:
    return {
        "frame_idx": 0,
        "total_frames": 0,
        "timestamp": None,
        "counts": {"detected": 0, "in_quad": 0, "tracked": 0},
        "density": {"peak_persons_per_m2": 0.0, "mean_persons_per_m2": 0.0, "cells_above_2_5": 0},
        "performance": {"fps": 0.0, "detect_ms": 0.0, "track_ms": 0.0},
        "calibration": None,
        "detection": None,
        "status": SessionStatus.IDLE.value,
        "video": None,
        "total_ids_seen": 0,
        "error": None,
    }


class PipelineSession:
    """One perception session. A new session replaces the previous one
    entirely (stop-then-start) -- this app never runs two pipelines at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status: SessionStatus = SessionStatus.IDLE
        self.error_message: str | None = None

        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.frame_counter: int = 0
        self.latest_camera_jpeg: bytes | None = None
        self.latest_heatmap_jpeg: bytes | None = None
        self.latest_telemetry: dict[str, Any] = idle_telemetry()

    def snapshot_frames(self) -> tuple[int, bytes | None, bytes | None]:
        with self._lock:
            return self.frame_counter, self.latest_camera_jpeg, self.latest_heatmap_jpeg

    def snapshot_telemetry(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.latest_telemetry)

    def publish_frame(self, camera_jpeg: bytes, heatmap_jpeg: bytes, telemetry: dict[str, Any]) -> None:
        with self._lock:
            self.frame_counter += 1
            self.latest_camera_jpeg = camera_jpeg
            self.latest_heatmap_jpeg = heatmap_jpeg
            self.latest_telemetry = telemetry

    def set_status(self, status: SessionStatus, error_message: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.error_message = error_message
            self.latest_telemetry["status"] = status.value
            self.latest_telemetry["error"] = error_message

    def reset_to_idle(self) -> None:
        with self._lock:
            self.status = SessionStatus.IDLE
            self.error_message = None
            self.latest_telemetry = idle_telemetry()
            self.latest_camera_jpeg = None
            self.latest_heatmap_jpeg = None

    def is_running(self) -> bool:
        with self._lock:
            return self.thread is not None and self.thread.is_alive()


session = PipelineSession()
