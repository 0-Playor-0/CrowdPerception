# Prompt: turn the local dashboard into a hosted API

This is a ready-to-paste prompt for a *separate* AI coding session whose job
is to take the local-only FastAPI dashboard in this repo and turn it into an
online-hosted service. It is deliberately NOT something this session should
do tonight -- see `README.md`'s Hosting section for why (GPU box, weights,
video upload, auth, per-session isolation all become real infrastructure
work, not a code change).

Copy everything between the `---` lines into a fresh session once you're
ready to take that on.

---

I have a local-only FastAPI + vanilla-JS operator dashboard for a
person-detection/tracking/density-estimation pipeline (YOLO11 + OC-SORT +
homography-based ground projection + a rolling-window density heatmap). It
currently:

- Runs a synchronous CV pipeline (~2-3 fps on a laptop GPU/MPS, not
  real-time) in a background thread inside a single FastAPI process.
- Streams two MJPEG feeds over plain HTTP (`multipart/x-mixed-replace`):
  one annotated camera view (boxes, track IDs, calibrated-quad outline),
  one top-down density heatmap. Both come from the SAME per-frame pipeline
  pass -- one inference call fans out to both streams; there is no code
  path that runs inference twice.
- Pushes numeric telemetry (detection/track counts, density stats, fps,
  calibration metadata) over a WebSocket at ~3 Hz.
- Reads video from local disk (`data/*.mp4`) and calibration from local
  JSON files (`calibration/*.json`), both selected through a "start
  session" flow (existing file, uploaded file, or an in-browser 4-point
  calibration clicker).
- Has no auth, no multi-tenancy, and assumes exactly one session runs at a
  time, on localhost.

I want to turn this into an online-hosted API that a remote model/client
can consume as a live stream of annotated images (boxes) and heatmap
images, instead of a developer pointing a browser at localhost. Please help
me design and implement:

1. **Compute**: a GPU-backed hosting target (the pipeline currently
   targets Apple MPS or CPU -- it will need a CUDA-capable inference path,
   or to stay CPU-bound and accept the fps hit) sized for sustained
   ~2-5 fps tiled YOLO11 inference per active session, and how many
   concurrent sessions one instance should support before autoscaling.
2. **Model weights**: how to package/pull `models/yolo11s.pt` (or a
   larger/smaller variant) into the deployed image or a model volume,
   without committing weights to git.
3. **Video ingestion**: replacing local `data/*.mp4` reads with an upload
   endpoint backed by object storage (size limits, allowed formats,
   virus/type scanning, lifecycle/cleanup of old uploads), plus a path for
   ingesting a live RTSP/RTMP camera feed instead of a pre-recorded file if
   that becomes a requirement later.
4. **Streaming to a remote model/client**: keep or replace the MJPEG
   approach for a hosted, possibly cross-origin, possibly authenticated
   consumer -- evaluate MJPEG-over-HTTPS vs. a WebSocket/WebRTC frame feed
   (base64 or binary) vs. periodic snapshot polling, given the target
   consumer is a model or service, not a browser `<img>` tag. Preserve the
   existing behavior of one pipeline pass feeding both the annotated and
   heatmap streams -- don't let a hosted redesign accidentally double the
   inference cost per frame.
5. **Auth & per-session isolation**: API keys or OAuth for whoever starts a
   session; strict session ownership so one caller can't see or stop
   another's stream; resource limits (max concurrent sessions per key, max
   session duration) so one client can't monopolize the GPU.
6. **Calibration handling**: the in-browser 4-point clicker currently
   writes a calibration JSON straight to local disk keyed by video
   filename -- redesign this as a per-session, per-tenant artifact (not a
   shared filename), and keep the existing convexity/round-trip validation
   (already isolated in `scripts/calibrate_video.py::build_calibration_record`)
   rather than reimplementing it.
7. **Observability & cost**: metrics/logs for GPU utilization, per-session
   fps, and session duration, since compute cost scales directly with
   concurrent sessions and video length.

Constraints carried over from the local version, please keep these:
- Never claim or imply real-time performance -- the UI/API must keep
  reporting measured fps and marking it "not real-time."
- Calibration source (`ESTIMATED` vs `USER_MEASURED`) and its uncertainty
  note must remain visible in every response/stream, not just the local
  dashboard banner.
- The three-stage detection funnel (detected -> in quad -> tracked) must
  stay visible/queryable, not collapsed into a single "people count" --
  it's a deliberate honesty feature, not a debug artifact.
- Do not add face recognition or any form of per-person biometric
  identification -- this stays a counting/density system.

Please start by proposing an architecture (component diagram is fine) and
a rough cost/complexity estimate before writing code, since the compute
and storage choices above significantly change the implementation.

---
