from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from server.placeholder import CAMERA_IDLE_JPEG, HEATMAP_IDLE_JPEG
from server.state import session

router = APIRouter()

POLL_INTERVAL_S = 0.05
BOUNDARY = b"frame"


async def _mjpeg_generator(pick_jpeg):
    last_counter = -1
    while True:
        counter, camera_jpeg, heatmap_jpeg = session.snapshot_frames()
        jpeg = pick_jpeg(camera_jpeg, heatmap_jpeg)
        if jpeg is None:
            jpeg = pick_jpeg(CAMERA_IDLE_JPEG, HEATMAP_IDLE_JPEG)
            counter = -2  # always (re-)serve the idle frame while idle
        if counter != last_counter:
            last_counter = counter
            yield (
                b"--" + BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
        await asyncio.sleep(POLL_INTERVAL_S)


@router.get("/api/stream/camera")
async def stream_camera():
    return StreamingResponse(
        _mjpeg_generator(lambda cam, heat: cam),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
    )


@router.get("/api/stream/heatmap")
async def stream_heatmap():
    return StreamingResponse(
        _mjpeg_generator(lambda cam, heat: heat),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
    )
