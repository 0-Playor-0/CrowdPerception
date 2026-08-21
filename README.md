# CrowdPerception

Tiled YOLO11 detection, OC-SORT tracking, 4-point homography ground
projection, and a rolling-window density heatmap. See `perception/` for the
pipeline itself and `scripts/live_perception.py` for the reference CLI
(OpenCV dual-pane window: annotated camera feed + top-down heatmap).

## Operator dashboard

A FastAPI + vanilla-JS web dashboard wraps the same pipeline for
browser-based operation: live annotated video, live density heatmap,
detection funnel (detected -> in quad -> tracked), and an in-browser
4-point calibration tool. It imports `perception/` and
`scripts/live_perception.py`/`scripts/calibrate_video.py` directly --
detection, tracking, projection, and heatmap rendering are not
reimplemented anywhere in `server/`.

Run it:

```bash
uv run uvicorn server.main:app --port 8000
```

Then open **http://localhost:8000**.

With nothing loaded, every panel shows an explicit idle state. Click the
floating action button (bottom-right) to pick a video and calibration (or
calibrate a new video by clicking 4 ground-plane points on its first
frame) and start a session.

### Architecture, briefly

- `GET /api/stream/camera` and `GET /api/stream/heatmap` are MJPEG
  (`multipart/x-mixed-replace`) endpoints. Both are produced from the
  *same* pipeline pass per frame -- `server/pipeline_runner.py` calls
  `LivePerceptionRunner.process_frame()` and `.build_composite()` once,
  then splits the composite image into the camera pane and the heatmap
  panel by re-deriving the camera pane's pixel width (same
  `DISPLAY_MAX_HEIGHT_PX` constant the CLI tool itself uses). Inference
  never runs twice for one frame.
- `WS /ws/telemetry` pushes JSON (counts, density stats, fps, calibration
  metadata) at ~3 Hz. Every number the dashboard renders comes from this
  payload -- nothing is computed in JavaScript.
- The pipeline runs in a background `threading.Thread`
  (`server/pipeline_runner.py`), not on the FastAPI event loop, so the
  server stays responsive while a frame is processing (measured ~2.8 fps
  tiled at 4K on an M1 MacBook Air -- not real-time, and the UI says so).
- `server/state.py` holds the one shared `PipelineSession` (status, latest
  encoded frames, latest telemetry) behind a lock; HTTP/WS handlers only
  read it or signal start/stop.

### Known limits (by design, not oversights)

- **Single session at a time.** Starting a new session stops whatever is
  running first.
- **Calibration is estimated unless measured.** The calibration banner at
  the top of the page reads the calibration file's own `source`/`note`
  fields verbatim -- when a video has no real-world measurement, density
  figures carry roughly 2x uncertainty, and the UI says so in amber.
- **No fixity check.** The homography assumes a fixed camera. A moving
  camera invalidates every density figure; the FAB modal warns about this,
  but nothing here automatically detects camera motion.
- **localhost only.** See below.

## Hosting

This is built for `localhost` only. Running it for real users needs, at
minimum:

- A GPU-backed instance (CUDA or otherwise) sized for sustained tiled
  YOLO11 inference per concurrent session.
- Model weights (`models/yolo11s.pt`) packaged into the deployment rather
  than committed to git.
- Video ingestion via upload/object storage instead of reading
  `data/*.mp4` off local disk (or a live RTSP/RTMP path).
- Auth and per-session isolation -- today, one process serves exactly one
  anonymous session with no ownership checks.

All frontend API calls go through one constant
(`static/js/config.js::API_BASE_URL`), so pointing the UI at a hosted
backend is a one-line change -- the API itself is not hosted anywhere
today. `docs/HOSTING_PROMPT.md` has a ready-to-paste prompt for the
follow-up session that would actually build this out.

## Tests

```bash
uv run pytest
```
