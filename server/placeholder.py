"""Static idle-state JPEG frames served by the MJPEG endpoints when no
session is running -- so /api/stream/camera and /api/stream/heatmap never
serve a blank body, per the "nothing blank, nothing stale" UI rule."""

from __future__ import annotations

import cv2
import numpy as np


def _make_placeholder(width: int, height: int, title: str, subtitle: str) -> bytes:
    img = np.full((height, width, 3), (36, 24, 14), dtype=np.uint8)  # dark navy, BGR
    cv2.putText(img, title, (width // 2 - 160, height // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (width // 2 - 160, height // 2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("failed to encode placeholder JPEG")
    return buf.tobytes()


CAMERA_IDLE_JPEG = _make_placeholder(960, 720, "No video loaded", "Start a session to see live detection")
HEATMAP_IDLE_JPEG = _make_placeholder(480, 720, "No video loaded", "Density heatmap appears once running")
