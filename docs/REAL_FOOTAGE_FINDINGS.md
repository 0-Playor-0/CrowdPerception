# Real Footage Findings — Myeongdong crowd video (2026-08-20)

Video: `data/127690-739144743.mp4`. All numbers below are from real runs on
this exact file, commands included. Where a number is derived from an
estimated (not measured) calibration, it is labelled ESTIMATED wherever it
appears — never presented as a real measurement.

---

## C2.1 — Inspection

```bash
uv run python -c "import cv2; cap=cv2.VideoCapture('data/127690-739144743.mp4'); ..."
```

- **Resolution**: 3840×2160 (4K)
- **FPS**: 30.0
- **Frame count**: 540
- **Duration**: 18.0s
- **Codec**: h264

**Camera stability — checked quantitatively, not just by eye.** ORB
keypoint matching on a static signage patch (rows 100–500, cols 0–700)
across frames 0/135/270/404/539: median keypoint shift = **0.0px** at
135/270/404, **0.88px** at 539 (366 matches) — consistent with encoding
noise, not motion. **Camera is fixed.** Cleared to proceed per the stop
condition.

- **Scene**: Myeongdong shopping street, Seoul — identifiable by signage. Night, artificial lighting, significant per-frame motion blur on moving pedestrians (long effective exposure).
- **Camera angle**: elevated oblique — looking down and along the street from roughly 3rd/4th-floor height, not overhead. The street runs diagonally away from camera, not straight ahead.
- **Ground plane**: visible for most of the frame (paved walkway), partially occluded by pedestrians and by storefront awnings/signage on the left edge.
- **Density**: varies over the clip. Hand counts (below) range ~18–28 visible people per sampled frame, not evenly distributed — one frame (t=4.5s) shows a distinct dense cluster.

---

## C2.2 — Detection + tracking, pixel space, no calibration

```bash
uv run python scripts/detect_track.py --video data/127690-739144743.mp4 \
    --model models/yolo11s.pt --annotated-out output/detect_track_annotated.mp4 \
    --stats-out output/detect_track_stats.json
```

Full 540-frame run, untiled (single YOLO pass per frame):

- **Raw detections/frame**: mean 12.20, min 4, max 21
- **Confidence**: mean 0.516, p10 0.331, p50 0.495, p90 0.733
- **Measured fps (detect+track only)**: 22.96 (steady-state; a 20-frame smoke test measured 3.28fps, entirely explained by MPS's first-call kernel-compilation cost — amortizes away over the full run, 22.96 is the representative figure)
- **Unique CONFIRMED track IDs across the whole 18s video**: 93

**93 unique confirmed IDs in 18 seconds, with confirmed-count staying at 0–2 per frame almost the entire run** (see `output/detect_track_stats.json`'s `confirmed_tracks_per_frame`) is itself the headline tracking finding: OC-SORT's `minimum_consecutive_frames` confirmation requires 3 consecutive detections on the same identity, and this footage's detection flicker (motion blur / occlusion causing a person to be missed one frame, re-detected the next as a "new" object) means tracks are constantly confirmed, lost, and reassigned a new ID rather than persisted. This is real ID churn on real footage — nothing like this was ever observable on PETS2009 (too sparse to stress the tracker) or on synthetic data (OC-SORT was only ever exercised against clean synthetic detections in the prior project).

**Hand-count vs detector, 5 frames** (same indices used for C2.1's stability check: 0, 135, 270, 404, 539 → t=0.0/4.5/9.0/13.47/17.97s). Hand counts done by visual inspection of the plain frame, independently, before viewing detector output. Annotated PNGs saved for independent verification:

| Frame (t) | Hand count | Detector (untiled) | Missed | Saved as |
|---|---|---|---|---|
| 0 (0.0s) | ~25 (uncertain ±3) | 14 | ~11 (44%) | `output/handcount_frames/frame_0000_detected.png` |
| 135 (4.5s) | ~28 (uncertain ±4, densest frame) | 8 | ~20 (71%) | `output/handcount_frames/frame_0135_detected.png` |
| 270 (9.0s) | ~20 (uncertain ±3) | 13 | ~7 (35%) | `output/handcount_frames/frame_0270_detected.png` |
| 404 (13.47s) | ~19 (uncertain ±3) | 13 | ~6 (32%) | `output/handcount_frames/frame_0404_detected.png` |
| 539 (17.97s) | ~23 (uncertain ±3) | 9 | ~14 (61%) | `output/handcount_frames/frame_0539_detected.png` |

Hand counts are inherently approximate on a dense, motion-blurred real
scene — stated uncertainty ranges reflect that, not false precision. Undercount
is worst on the densest frame (135) and generally worse than PETS2009's
detector characterization (which was never actually run — see
`docs/REALITY_CHECK.md` — this is the project's first real measurement of
this kind).

**Observed failure modes** (visible directly in the annotated PNGs above):
1. **Distant/small people against cluttered signage backgrounds** — the single largest failure mode. In frame 135, an entire scattered group of ~9 people in the middle-background (near the "주차금지" sign and beyond) got **zero** detections in the untiled pass.
2. **Merged silhouettes in dense, motion-blurred clusters** — 2–3 people standing/walking close together with motion blur frequently get one box instead of several, or no box at all.
3. Both failure modes are corrected substantially by tiled inference (C2.3).

---

## C2.3 — Tiled inference (SAHI-style, `supervision.InferenceSlicer`)

Implemented in `scripts/detect_track.py --tile` (config-toggled, default
off) and `scripts/world_density.py --tile`. `slice_wh=(1280,1280)`,
`overlap_wh=(128,128)`, NMS `iou_threshold=0.7` across tile boundaries.

Same 5 frames, tiled vs untiled:

| Frame | Untiled count | Tiled count | Hand count | Tiled fps cost |
|---|---|---|---|---|
| 0 | 14 | 49 | ~25 | 2.74s (first call, MPS warmup) |
| 135 | 8 | 49 | ~28 | 0.38s |
| 270 | 13 | 57 | ~20 | 0.39s |
| 404 | 13 | 44 | ~19 | 0.43s |
| 539 | 9 | 39 | ~23 | 0.37s |

Tiled counts exceed hand counts, not just close the gap — visual
inspection of `output/handcount_frames/frame_0000_tiled.png` shows why:
tiling recovers almost the entire previously-missed background cluster
(a real, substantial improvement), **but also introduces at least one
confirmed false positive** — a mannequin in a reflective shop-window
display (top-left of frame 0, boxed at confidence 0.54) is not a person.
Tiling trades undercounting for a mix of much-improved recall and a small
amount of new false-positive risk from reflections/displays — net
improvement, not a free lunch.

**FPS cost**: steady-state tiled inference runs ~0.37–0.43s/frame
(~2.3–2.7 fps) vs untiled's measured 22.96fps — **roughly an 8–10x
slowdown** for full-frame tiling at this resolution and tile size.

**Recommendation, on evidence**: use tiled inference for anything where
recall matters more than throughput (density/crush estimation, hand-count
comparison, offline analysis) — the recall gain is large and directly
visible in the saved PNGs. Do NOT use it for anything needing real-time
throughput at this resolution/tile size without either a smaller tile
count, a lower-resolution input, or GPU hardware faster than this
machine's MPS backend — 2.3–2.7fps is not real-time on a 30fps source.

---

## C2.4 — Calibration from human height (ESTIMATED, not measured)

**Labelled everywhere it is used: HEIGHT-ESTIMATED CALIBRATION, NOT
MEASURED.** No known scene dimensions exist for this video.

Method: collected 713 person bounding boxes (tiled detector, confidence
≥0.4) across 18 frames sampled every 30th frame; filtered to 478 samples
after dropping boxes touching the frame's bottom edge (cropped, biased
height) and confidence <0.5. Fit `height_px = m*y_bottom + c` (standard
perspective-scale approximation used in crowd-counting literature when no
calibration target is available), assuming every detected person is 1.7m
tall.

```
height_px = 0.24215*y + -74.852   R^2=0.464   n=478
residual std: 81.1px (34.0% of mean height 238.9px)
```

**R²=0.464 is a genuinely weak fit — flagging this explicitly rather than
hiding it behind a plausible-looking downstream number.** ~34% of the
variance in a detected person's apparent height at a given image row is
*not* explained by depth/perspective alone. The most likely cause,
specific to this footage: the camera is not looking straight down the
street (oblique angle, C2.1) — image row does not cleanly determine depth
on its own here, since two people at the same row can be at different
depths depending on their horizontal position on the diagonally-receding
street. Motion blur distorting box heights and genuine height variation
among real people are secondary contributors.

Built a 4-point homography from this fit anyway (the trend is real and
strong even if noisy — height clearly grows 70px→434px, a 6x range, across
the fitted row span) via `perception.geometry.ViewTransformer`, using SOURCE
points tracing the visible walkway in frame 0 and TARGET points computed by
integrating the fitted local scale between near and far rows:

```
near row (y=2000) scale: 4.2 mm/px   far row (y=1050) scale: 9.5 mm/px
```

**Sanity check (edge lengths / area, same check `calibrate.py --check`
does for synthetic scenarios)**:

```
edge 0->1 (near width):  7.68 m
edge 1->2:                5.82 m
edge 2->3 (far width):    6.63 m
edge 3->0:                5.82 m
area:                    41.46 m^2
```

**Plausibility**: 6.6–7.7m for a Myeongdong pedestrian shopping street
walkway is a believable order of magnitude (these streets are commonly
several metres wide). `ViewTransformer`'s own ill-conditioning check
passed with no warning. **Confidence: moderate-low.** The homography is
usable for order-of-magnitude density estimates, not for anything claiming
sub-metre precision — the 34% per-sample height residual propagates
directly into position/density uncertainty of a similar magnitude. This is
declared explicitly rather than producing a false sense of precision: if
you need trustworthy metre-space numbers from this footage, the fix is a
real measured reference (a known object/distance in frame), not more
samples fed into this same regression.

---

## C2.5 — World coordinates and density

```bash
uv run python scripts/world_density.py --video data/127690-739144743.mp4 \
    --tile --cell-size-m 1.0 \
    --stats-out output/world_density_stats.json --records-out output/world_density_records.jsonl
```

Full 540-frame run, tiled detection (chosen over untiled specifically
because C2.2/C2.3 showed untiled substantially undercounts — using it here
would understate density and could misleadingly suggest lower risk than
real). Density computed directly (count / calibrated quad area, and
per-1m-cell count) — no analytics/pressure module was reintroduced,
per the "perception only" rule.

- **Calibrated TARGET quad area**: 41.40 m² (height-ESTIMATED, §C2.4)
- **Records produced**: 4,285 (`output/world_density_records.jsonl`)
- **Mean density over the quad**: 0.081 persons/m²
- **Peak quad-average density**: 0.242 persons/m² (frame 2, t=0.07s)
- **Peak SINGLE-CELL density (1m×1m)**: **3.0 persons/m²** (frame 22, t=0.73s)
- **Busiest cell over the whole clip** (by cumulative observation count, not simultaneous): cell (5,3), 217 total person-frame observations

**This determines whether crush-relevant conditions are present: partially
yes, at least momentarily, at the single-cell level.** 3.0 persons/m² in
one 1m² cell is in the same order of magnitude as the "warning" density
tier (3.0 persons/m²) used for the density-based tiering in the prior
(synthetic) project — genuinely different from PETS2009, which topped out
around 2–3 *people in the entire scene*, never mind per square metre. This
is the most crush-relevant real number produced anywhere in this project
so far.

**Caveat that must travel with this number**: it comes from a
height-ESTIMATED calibration with ~34% per-sample residual noise (§C2.4).
A 1m² cell is small enough that the position error implied by that
residual could plausibly merge two adjacent people's positions into one
cell, or split one real position across a cell boundary — meaning this
specific peak figure should be read as "real density got uncomfortably
high somewhere in this scene, order of magnitude ~3 persons/m²," not as a
precise, trustworthy 3.00. Re-deriving this with a measured (not
estimated) calibration is the single highest-value follow-up if this
number needs to go on a slide with confidence.

---

## C3 — ToTest.mp4 (Westfield mall) cross-footage comparison

> **ToTest.mp4 is REJECTED as a demo clip and has been renamed to
> `data/REJECTED_ToTest_moving_camera.mp4`** (camera not fixed, 92.7px
> median ORB drift) -- kept here as historical record of what was
> measured and why it was rejected, not as a source for any current
> default, threshold, or density figure.

Full per-clip detail in `docs/TOTEST_FINDINGS.md`. This section asks
specifically: which of the findings above (Myeongdong) generalize to a
second, very different real clip, and which were specific to that footage?
ToTest.mp4: indoor shopping mall, elevated balcony view, 2560×1440,
30.518fps, 954 frames, much denser and more multi-level than Myeongdong's
street scene. **No hand counts exist for either clip's detector output** —
everything below is relative counts and funnel arithmetic, not a
correctness claim.

**New, load-bearing difference from Myeongdong: this camera is not
fixed.** ORB drift check (same method as §C2.1): median 92.7px, max
451.1px, growing monotonically across the clip — vs Myeongdong's
confirmed-static ≤0.88px. Myeongdong's calibration/tiling findings all
implicitly assumed (and verified) a fixed camera; ToTest is the first real
clip in this project where that assumption doesn't hold, and it was
treated as informational rather than blocking specifically for this
comparison task (it was a hard stop in the immediately preceding
investigation of this same clip).

### Tiling gain ratio — direction generalizes, magnitude does not

| | Myeongdong (§C2.3) | ToTest (tiling sweep) |
|---|---|---|
| Comparison | tile1280/imgsz640 vs no-tile/imgsz640 | same |
| Per-sample ratios | 3.5x, 6.1x, 4.4x, 3.4x, 4.3x | 7.7x, 31.0x, 3.5x, 4.0x |
| Mean | **~4.35x**, fairly stable | **~11.5x**, highly unstable |
| Why the difference | untiled floor still finds 8–14 people/frame — small but non-degenerate | untiled floor finds 1–11 people/frame — several samples are near-total detection failure, so ratios blow up on a tiny denominator rather than reflecting a genuinely bigger tiling effect |

**Tiling substantially helping is the part that generalizes.** The exact
multiplier does not — it should never be quoted as a portable constant
between scenes; it depends on how badly the untiled floor is already
failing on that specific footage's crowd density/scale distribution.

**New information ToTest adds that Myeongdong's original investigation
never tested**: an imgsz-only ablation (tile geometry held fixed at
1280px, only the per-tile network input resolution changed from the
default 640 to 1280) gained **2.88x** over baseline on ToTest, entirely
independent of any tile-count change. Myeongdong's C2.3 never varied
imgsz — its "tiled" measurements used the same imgsz=640-on-1280px-tiles
default ToTest's config A reproduces, meaning **Myeongdong's own tiled
numbers likely have room to improve the same way**, though this was not
re-measured on that clip in this task (out of scope; flagging it as a
followup, not re-running it).

### Tracker confirmation loss — generalizes, and gets WORSE

| | Myeongdong | ToTest |
|---|---:|---:|
| in_quad detections (full-clip run) | 13,039 | 6,797 |
| confirmed (tracked) | 1,928 | 191 |
| **loss before confirmation** | **85.21%** | **97.19%** |
| unique confirmed track IDs | 275 | 17 |
| median track lifetime | 0.667s | 0.23s |
| surviving ≥1.0s | 43.3% | 17.6% |
| surviving ≥2.5s | 24.4% | 5.9% |

Same OC-SORT configuration on both clips — `minimum_consecutive_frames=3`,
`high_conf_det_threshold=0.6`, `minimum_iou_threshold=0.3`,
`lost_track_buffer=30` frames — nothing retuned between or during either
investigation. **Severe tracker confirmation loss is not a Myeongdong
quirk — it generalizes, and this second clip is meaningfully worse on
every churn metric.**

The standout candidate explanation, and the one most directly supported by
this comparison specifically: **ToTest's camera moves; Myeongdong's does
not.** Camera motion adds a second source of frame-to-frame pixel
displacement (independent of each person's own movement) directly on top
of the mechanism `minimum_iou_threshold`-gated association already
struggles with on Myeongdong's *static*, motion-blurred footage (§C2.2).
A moving reference frame should make IoU-based association strictly
harder, not easier — the direction of the effect matches what was
measured. This is a strong, testable hypothesis, **not a proven causal
result** — isolating it would need stabilizing ToTest's footage (frame
registration against the ORB drift already measured) and re-running the
same funnel, which is future work, not done here.

Secondary, non-exclusive contributors that likely compound the effect,
neither isolated in this task: ToTest's calibrated quad is a much smaller
catchment relative to the visible crowd (mean 7.1 in-quad detections/frame
vs Myeongdong's mean ~24/frame — computed as 13,039/540), and the scene
itself is denser with more multi-level occlusion (escalators, balconies)
than Myeongdong's street-level view.
