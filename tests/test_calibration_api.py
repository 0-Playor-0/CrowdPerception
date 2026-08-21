"""Tests for POST /api/calibration/points -- specifically the overwrite
guard added after dashboard testing once overwrote the real demo
calibration file (calibration/127690-739144743.json), recovered only
because git still had the original. Calibration files are load-bearing:
every world-coordinate and density figure in the project's docs traces
back to one, so a silent overwrite must be impossible by default.

Uses a real (tiny, synthetic) video file for the success-path tests since
server/routes/calibration.py's endpoint calls scripts/calibrate_video.py's
grab_frame() -- a garbage file fails to open before the overwrite check
even applies. The refusal test itself never needs a readable video: the
overwrite guard runs before any frame is read.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.routes.calibration as calibration_route
from server.main import app

# A simple convex quad in image-pixel space -- same shape used by
# tests/test_calibration.py, reused here since it's the tuple already known
# to pass validate_quad()/build_calibration_record()'s convexity check.
RECTANGLE_POINTS = [
    [10.0, 50.0],
    [50.0, 50.0],
    [42.0, 10.0],
    [18.0, 10.0],
]


def _make_tiny_video(path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    calibration_dir = tmp_path / "calibration"
    data_dir.mkdir()
    calibration_dir.mkdir()
    _make_tiny_video(data_dir / "fake.mp4")

    monkeypatch.setattr(calibration_route, "DATA_DIR", data_dir)
    monkeypatch.setattr(calibration_route, "CALIBRATION_DIR", calibration_dir)

    return TestClient(app), calibration_dir


def test_refuses_to_overwrite_existing_calibration_by_default(client) -> None:
    test_client, calibration_dir = client
    existing = calibration_dir / "fake_existing.json"
    existing.write_text(json.dumps({"source": "USER_MEASURED", "note": "do not clobber me"}))

    response = test_client.post(
        "/api/calibration/points",
        json={"video": "fake.mp4", "points": RECTANGLE_POINTS, "output_filename": "fake_existing.json"},
    )

    assert response.status_code == 409
    assert "overwrite" in response.json()["detail"].lower()
    # the refused write must leave the original file byte-for-byte untouched
    assert json.loads(existing.read_text())["note"] == "do not clobber me"


def test_overwrite_true_allows_replacing_existing_calibration(client) -> None:
    test_client, calibration_dir = client
    existing = calibration_dir / "fake_existing.json"
    existing.write_text(json.dumps({"source": "USER_MEASURED", "note": "will be replaced"}))

    response = test_client.post(
        "/api/calibration/points",
        json={
            "video": "fake.mp4",
            "points": RECTANGLE_POINTS,
            "output_filename": "fake_existing.json",
            "overwrite": True,
        },
    )

    assert response.status_code == 200
    assert json.loads(existing.read_text())["note"] != "will be replaced"


def test_default_output_filename_is_video_stem_when_none_given(client) -> None:
    test_client, calibration_dir = client
    response = test_client.post(
        "/api/calibration/points",
        json={"video": "fake.mp4", "points": RECTANGLE_POINTS},
    )
    assert response.status_code == 200
    assert response.json()["written"] == "fake.json"
    assert (calibration_dir / "fake.json").exists()


def test_output_filename_cannot_escape_calibration_dir(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/api/calibration/points",
        json={
            "video": "fake.mp4",
            "points": RECTANGLE_POINTS,
            "output_filename": "../../etc/passwd.json",
        },
    )
    # path components are stripped down to a bare filename, so this either
    # succeeds by writing "passwd.json" inside calibration_dir, or is
    # rejected -- it must never land outside CALIBRATION_DIR.
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert response.json()["written"] == "passwd.json"
