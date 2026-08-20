"""Thin wrapper over trackers.OCSORTTracker (roboflow/trackers, Apache-2.0).

Swapping trackers (e.g. to ByteTrackTracker, also in the same package) is a
one-line change: update the import and _TRACKER_CLASS below. Every tracker
in that package shares the same BaseTracker interface --
update(detections) -> Detections, reset() -> None -- so nothing else in
this file or its callers would need to change.
"""

from __future__ import annotations

import supervision as sv
from trackers import OCSORTTracker

_TRACKER_CLASS = OCSORTTracker


class Tracker:
    def __init__(self, frame_rate: float, **tracker_params: object) -> None:
        self._tracker = _TRACKER_CLASS(frame_rate=frame_rate, **tracker_params)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Assigns/updates tracker_id on detections. Tracks younger than
        the tracker's minimum_consecutive_frames (OC-SORT default: 3) are
        given tracker_id == -1 -- not yet confirmed. Callers should filter
        those out rather than reporting a track_id that might still change."""
        return self._tracker.update(detections)

    def reset(self) -> None:
        self._tracker.reset()
