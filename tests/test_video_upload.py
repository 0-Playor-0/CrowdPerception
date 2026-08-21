"""Tests for POST /api/videos/upload -- the endpoint that makes calibration
possible on a freshly uploaded video: it saves the file to data/uploads/
immediately (not deferred to session-start) specifically so
/api/frame/first and /api/calibration/points have something on disk to
read before any session exists. See server/routes/catalog.py.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import server.routes.catalog as catalog_route
from server.main import app


def _tiny_mp4_bytes() -> bytes:
    # Content doesn't need to be a real video for these tests -- the
    # endpoint only writes bytes to disk and checks the extension.
    return b"not a real video, only used to exercise the upload endpoint"


def test_upload_saves_under_uploads_and_returns_relative_filename(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(catalog_route, "UPLOAD_DIR", tmp_path / "uploads")
    client = TestClient(app)

    response = client.post(
        "/api/videos/upload",
        files={"video_file": ("my_video.mp4", _tiny_mp4_bytes(), "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "uploads/my_video.mp4"
    assert (tmp_path / "uploads" / "my_video.mp4").exists()


def test_upload_rejects_non_video_extension(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(catalog_route, "UPLOAD_DIR", tmp_path / "uploads")
    client = TestClient(app)

    response = client.post(
        "/api/videos/upload",
        files={"video_file": ("not_a_video.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert not (tmp_path / "uploads").exists() or not list((tmp_path / "uploads").glob("*"))


def test_upload_never_silently_overwrites_a_same_named_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(catalog_route, "UPLOAD_DIR", tmp_path / "uploads")
    client = TestClient(app)

    first = client.post(
        "/api/videos/upload",
        files={"video_file": ("clip.mp4", b"first upload content", "video/mp4")},
    )
    second = client.post(
        "/api/videos/upload",
        files={"video_file": ("clip.mp4", b"second, different upload content", "video/mp4")},
    )

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["filename"] != second.json()["filename"], (
        "two different uploads sharing a filename must not collide -- "
        "the first upload's bytes must not be silently overwritten"
    )
    first_path = tmp_path / "uploads" / "clip.mp4"
    assert first_path.read_bytes() == b"first upload content"


def test_upload_strips_directory_components_from_the_filename(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(catalog_route, "UPLOAD_DIR", tmp_path / "uploads")
    client = TestClient(app)

    response = client.post(
        "/api/videos/upload",
        files={"video_file": ("../../etc/passwd.mp4", _tiny_mp4_bytes(), "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "uploads/passwd.mp4"
    assert (tmp_path / "uploads" / "passwd.mp4").exists()


def test_uploaded_video_is_listed_by_api_videos(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(catalog_route, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(catalog_route, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    client = TestClient(app)

    client.post(
        "/api/videos/upload",
        files={"video_file": ("newly_uploaded.mp4", _tiny_mp4_bytes(), "video/mp4")},
    )

    videos = client.get("/api/videos").json()["videos"]
    filenames = [v["filename"] for v in videos]
    assert "uploads/newly_uploaded.mp4" in filenames
