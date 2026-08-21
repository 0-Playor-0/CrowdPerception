from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.state import session

router = APIRouter()

TELEMETRY_INTERVAL_S = 1 / 3  # ~3 Hz per spec


@router.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(session.snapshot_telemetry())
            await asyncio.sleep(TELEMETRY_INTERVAL_S)
    except WebSocketDisconnect:
        pass
