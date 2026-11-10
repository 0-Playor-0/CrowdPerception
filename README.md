# CrowdPerception

The perception layer for a crowd-crush early-warning system. Video goes in. Out comes each person's position on the ground in **metres**, a smoothed velocity per track, and a top-down density heatmap. Those are the inputs a pressure or alarm model needs.

This repo covers perception only. The social-force simulator and the pressure analytics live in the parent project, so there are no alarm tiers here.

## Pipeline

```
video frame
  └─ YOLOv11 person detector        perception/detector.py   (optional SAHI-style tiled inference)
      └─ OC-SORT tracker            perception/tracker.py
          └─ foot-point anchor      perception/ground.py     (box bottom-centre, never the centroid)
              └─ homography         perception/geometry.py   (pixel → ground-plane metres, conditioning check)
                  ├─ velocity       perception/velocity.py   (least-squares fit over a sliding window + outlier rejection)
                  ├─ trajectories   perception/trajectory.py (CSV log + track-churn diagnostics)
                  └─ heatmap        perception/heatmap.py    (rolling-window occupancy / persons per m²)
```

`scripts/live_perception.py` runs the whole chain. It shows two panes side by side: the annotated camera view and the top-down density map.

## Design decisions

- **Foot point, not centroid.** On an oblique camera, a box's centroid sits at torso height, which is not on the ground plane. Pushing it through a ground-plane homography puts people metres away from where they really are.
- **Fitted velocity, not frame differences.** Differencing consecutive positions amplifies detection jitter. Each track instead fits a line through a short window of positions and uses the slope as its velocity.
- **Fixed-camera gate.** Before using a clip, its camera is checked for movement by matching ORB keypoints on a static patch. A clip that drifts gets rejected, because a moving camera makes the homography (and every density number) invalid.
- **Honest labelling.** When a calibration was estimated from human height instead of measured, it is marked `ESTIMATED` everywhere it appears.

## Findings on real footage

Tested on an 18 s, 4K night clip of Myeongdong, Seoul, compared against hand counts on 5 frames. The full write-up with commands is in [`docs/REAL_FOOTAGE_FINDINGS.md`](docs/REAL_FOOTAGE_FINDINGS.md).

- **Untiled detection undercounts** dense background crowds: about 12 detections per frame against hand counts of 19 to 28.
- **Tiled inference** (1280 px tiles) recovers the missed clusters, but overshoots and adds false positives such as mannequins in shop windows. It is also about 8 to 10× slower (around 2.5 fps on Apple MPS). Use it offline, where recall matters more than speed.
- **Tracker ID churn.** Motion blur makes detections flicker, which gave 93 confirmed track IDs in 18 s. This is a known open problem.
- Two other candidate clips were **rejected** because the camera moved (see `docs/TESTING_FINDINGS.md` and `docs/TOTEST_FINDINGS.md`).

## Usage

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Runs on CPU or Apple MPS. Videos (`data/`) and YOLO weights (`models/`) are not committed.

```bash
uv sync

# 1. calibrate: click 4 ground points and enter the quad's real size in metres
uv run python scripts/calibrate_video.py --video data/your_clip.mp4

# 2. run the live pipeline
uv run python scripts/live_perception.py --video data/your_clip.mp4

# headless smoke run
uv run python scripts/live_perception.py --video data/your_clip.mp4 --no-window --max-frames 100

# tests
uv run pytest
```

## Stack

Python · Ultralytics YOLO · roboflow `supervision` + `trackers` (OC-SORT) · OpenCV · NumPy · Polars
