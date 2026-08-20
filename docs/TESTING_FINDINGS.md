# Testing.mp4 — Findings (candidate demo clip evaluation, 2026-08-20)

Video: `data/Testing.mp4`. Scene: an indoor event entry lobby (traditional
dress visible, likely a cultural/religious festival), elevated view, dense
crowd inside and an even denser crowd visible through glass entry doors
beyond. Evaluated as a **replacement candidate** for the current demo clip
(Myeongdong, `data/127690-739144743.mp4`). **No hand counts exist.**
Nothing below is a recall or accuracy claim — relative counts and measured
geometry only. Nothing was tuned or fixed as part of this evaluation.

**Verdict up front: REJECT.** Deciding factor: camera fixity fails (STEP
1, hard gate). See STEP 7 for full reasoning.

---

## STEP 0 — properties

**Resolution**: 2320×1080. **FPS**: 30.0. **Frame count**: 336.
**Duration**: 11.2s. **Codec**: h264.

Notably shorter than both existing clips (Myeongdong 18.0s, ToTest 31.3s).

---

## STEP 1 — CAMERA FIXITY (stop condition)

Same method as the other two clips: ORB keypoint matching on a static
patch, median/max pixel drift vs a frame-0 baseline. Patch: rows 0–180,
cols 1850–2320 (entry pillar + doorframe, upper-right — the only
close-to-crowd-free region in this scene; some crowd visible through the
glass within it, tolerated the same way the original Myeongdong patch
tolerated partial pedestrian occlusion, since the median statistic is
robust to a minority of contaminated keypoints).

**Per-second drift trace** (1 sample/second, 11 samples vs the frame-0 baseline):

| t (s) | median drift (px) | max drift (px) |
|---:|---:|---:|
| 1.00 | 6.230 | 385.8 |
| 2.00 | 5.015 | 407.7 |
| 3.00 | 5.310 | 374.5 |
| 4.00 | **25.495** | 392.5 |
| 5.00 | 29.184 | 385.3 |
| 6.00 | 27.256 | 375.4 |
| 7.00 | 38.412 | 384.3 |
| 8.00 | 22.222 | 370.9 |
| 9.00 | 32.888 | 374.7 |
| 10.00 | 38.973 | 417.4 |
| 11.00 | 50.918 | 373.2 |

**Median of per-second medians: 27.3px. P90: 39.0px. Max (of maxes): 417.4px.**

**Not localised, not constant either — it's a step change followed by a
climb.** The first 3 seconds sit at 5–6px (already 2.5–3x over the ~2px
threshold on their own); at t=4s it jumps roughly 5x to ~25px and then
keeps climbing, reaching ~51px by the last full second. This is not
sensor noise or an ORB artifact — I cross-checked visually: overlaying
the same patch crop from frame 0 against frame 330 shows the entry pillar
and door mullions have visibly shifted position (`/tmp/testing_patch_compare.png`
during this session; not committed, regeneratable from the frames listed
above). **STOP CONDITION TRIGGERED.** Homography and every persons/m²
figure derived from this clip's calibration would be invalid for anything
past the first ~3 seconds, and already marginal before that.

Per your instruction: not calibrating, not computing in-quad counts,
density, or the heatmap. STEP 3 (pixel-space detection) still runs below.
**STEPS 4, 5, and 6 are skipped** — the instruction was to run Step 3 only
and go straight to the Step 7 verdict.

---

## STEP 2 — calibration status

`calibration/Testing.json` **exists**. Reporting its stored contents as-is
(not reusing or recomputing anything):

- **Source**: `USER_MEASURED`
- **World dims**: 5.0m × 3.0m
- **Round-trip error**: 0.0px
- **Near/far edge scale**: 0.00561 / 0.00684 m/px → **far/near ratio 1.22×**
- **Grid at 0.5m cells**: 10 cols × 6 rows = 60 cells

**This is, on paper, a genuinely better calibration than Myeongdong's** —
real clicked correspondences with a perfect round-trip, vs. a height-fit
regression with R²=0.464. **It doesn't matter.** A calibration fit to
frame 0's camera position describes frame 0's camera position; by t=4s
the camera has moved and that same pixel quad no longer points at the
same patch of floor. This is the same failure mode that invalidated
ToTest's calibration — good input data, wrong assumption (fixed camera)
underneath it.

**Does the scene have a usable calibration reference, independent of the
fixity problem?** Possibly — the polished stone floor visible in the
lower-center of frame 330 appears to have a regular joint/tile pattern
that could serve as a scale reference if the actual tile dimensions were
knowable (would need someone who knows the venue, or a visible standard
object — e.g. the glass entry doors, if they're a standard commercial
height ~2.1m, could anchor a rough scale check). Moot given STEP 1;
noted only because you asked what I'd measure against if fixity had passed.

---

## STEP 3 — detection sweep (pixel-space only, valid regardless of fixity)

Frames **30, 150, 240, 315** (1.00s / 5.00s / 8.00s / 10.50s), picked from
a config-A raw-count scan (every 15th frame, 23 samples) as a density
proxy. **Density is flat across this clip** — 46 to 65 detections across
the whole scan, no dramatic swing (worth noting ahead of STEP 7 point 5).
Exported to `docs/handcount/`: `Testing_frame<N>_clean.png` +
`Testing_frame<N>_baselineA.png` (see `docs/handcount/Testing_README.md`).

`overlap_ratio=0.1`, cross-tile `nms_iou_threshold=0.7`, per-tile YOLO
`iou=0.7`, `conf=0.02` (current project default per `config.yaml` —
**this is the existing shared default, not something tuned for this
clip**; flagging per your "state every assumption" instruction).

| frame | config | tiles/frame | before NMS | after NMS | fps (detect-only) |
|---|---|---:|---:|---:|---:|
| 30 | A tile1280/imgsz640 (baseline) | 2 | 154 | 146 | 0.54 (cold start) |
| 30 | B tile1280/imgsz1280 | 2 | 313 | 286 | 1.02 (cold start) |
| 30 | C tile960/imgsz960 | 6 | 718 | 366 | 1.19 (cold start) |
| 30 | D tile640/imgsz640 | 8 | 471 | 414 | 1.41 (cold start) |
| 30 | E no-tile/imgsz640 (floor) | 1 | 105 | 105 | 2.76 (cold start) |
| 150 | A | 2 | 145 | 138 | 11.64 |
| 150 | B | 2 | 282 | 263 | 3.89 |
| 150 | C | 6 | 643 | 319 | 1.81 |
| 150 | D | 8 | 407 | 347 | 2.41 |
| 150 | E | 1 | 107 | 107 | 27.78 |
| 240 (proxy peak) | A | 2 | 140 | 133 | 11.67 |
| 240 | B | 2 | 310 | 292 | 3.75 |
| 240 | C | 6 | 687 | 346 | 1.75 |
| 240 | D | 8 | 434 | 380 | 2.49 |
| 240 | E | 1 | 118 | 118 | 29.17 |
| 315 | A | 2 | 145 | 137 | 11.44 |
| 315 | B | 2 | 300 | 281 | 3.64 |
| 315 | C | 6 | 689 | 350 | 1.74 |
| 315 | D | 8 | 422 | 375 | 2.43 |
| 315 | E | 1 | 114 | 114 | 29.39 |

Only 2 tiles/frame for A/B at this resolution (2320×1080 barely exceeds
one 1280px tile per axis) — smaller than either other clip. Steady-state
fps (mean of frames 150/240/315):

| config | steady-state fps |
|---|---:|
| A tile1280/imgsz640 | 11.58 |
| B tile1280/imgsz1280 | 3.76 |
| C tile960/imgsz960 | 1.77 |
| D tile640/imgsz640 | 2.44 |
| E no-tile/imgsz640 | 28.78 |

### Detections-after-NMS as a multiple of baseline A

| frame | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| 30 | 1.00x | 1.96x | 2.51x | 2.84x | 0.72x |
| 150 | 1.00x | 1.91x | 2.31x | 2.51x | 0.78x |
| 240 | 1.00x | 2.20x | 2.60x | 2.86x | 0.89x |
| 315 | 1.00x | 2.05x | 2.55x | 2.74x | 0.83x |
| **mean** | **1.00x** | **2.03x** | **2.49x** | **2.74x** | **0.80x** |

**Notably more stable than ToTest's equivalent table** (which ranged
3.4x–31x on the A-vs-E comparison due to E's near-degenerate counts).
Here E stays at a modest 0.72–0.89x of baseline rather than collapsing —
consistent with this being a lower native resolution (2320×1080 vs
ToTest's 2560×1440 and Myeongdong's 3840×2160), where whole-frame
inference at imgsz=640 doesn't need to downsample as aggressively, so the
untiled floor doesn't fail as badly here. This is genuinely useful,
relative-only information; it is not a recall or accuracy claim and no
hand count exists to check it against.

**STEPS 4, 5, and 6 skipped per the stop condition.**

---

## STEP 7 — REPLACE OR REJECT

### Three-way comparison

| | Myeongdong (incumbent) | ToTest (previously rejected) | Testing (this evaluation) |
|---|---|---|---|
| Resolution | 3840×2160 | 2560×1440 | 2320×1080 |
| Duration | 18.0s | 31.3s | 11.2s |
| ORB drift (median / max) | 0.0–0.88px | 92.7px / 451.1px | **27.3px / 417.4px** |
| Camera fixed? | **Yes** | No | **No** |
| Calibration source | ESTIMATED (R²=0.464) | USER_MEASURED (invalidated) | USER_MEASURED (invalidated) |
| Calibration quality if it counted | weak height-fit, ~34% residual | perfect round-trip, but moot | perfect round-trip (0.0px), but moot |
| Tiling gain ratio (A vs E) | ~4.35x, stable | ~11.5x, unstable (3.4–31x) | ~1.25x (1/0.80), the most stable of the three |
| in_quad detections/frame | ~24.15 | ~7.13 | not computed (fixity failed) |
| Tracker confirmation rate | 14.79% (85.21% loss) | 2.81% (97.19% loss) | not computed (fixity failed) |
| Peak density | ~3.0 persons/m² | 0.759 persons/m² (drift-caveated) | not computed (fixity failed) |
| fps at best available config | ~2.82 (tiled 4K) | ~1.09 (config B) | ~3.76 (config B) / ~2.44 (config D) |

### Scoring, in the requested order

**1. Fixed camera — hard gate.** **FAIL.** 27.3px median is ~14x over the
~2px threshold, with a visually confirmed progressive shift (pillar
position moves between frame 0 and frame 330). This alone forces REJECT
regardless of anything below — the same reasoning that rejected ToTest.

**2. Peak density.** Not measurable — no valid calibration to compute it
against. Cannot claim this clip is denser or thinner than Myeongdong's
~3.0 persons/m² peak in any way that means anything.

**3. Calibration quality.** Ironically the best of the three on paper —
genuine `USER_MEASURED`, perfect 0.0px round-trip, better than both
Myeongdong's ESTIMATED R²=0.464 and ToTest's (also-invalidated)
USER_MEASURED. **Entirely moot**: a perfect calibration for a camera
position the camera doesn't stay at isn't an advantage.

**4. Detection and tracking behaviour vs Myeongdong.** Partial picture
only (STEP 6 skipped). What STEP 3 does show: a *more stable* tiling gain
than ToTest (1.25–2.74x depending on config, vs ToTest's noisy 3.4–31x),
and reasonable fps headroom (3.76fps at config B, the highest of any
tiled config across all three clips at comparable settings) — this
resolution is cheap to run. But none of this compensates for gate 1.

**5. Duration and density variation.** Two separate negatives here, not
just one: **11.2s is by far the shortest of the three clips** (63% of
Myeongdong's length, 36% of ToTest's), and **density is flat** — the
proxy scan ranged only 46–65 detections across the whole clip with no
visible escalation, unlike Myeongdong or a clip with a clear build-up.
Even setting fixity aside entirely, this would be a **weaker demo
narrative** than the incumbent: nothing to show ramping from calm to
crowded.

### Verdict: **REJECT**

**Deciding factor: camera motion (27.3px median ORB drift vs the ~2px
threshold), the same failure mode that rejected ToTest.** Not marginal —
even the best 3-second window (5–6px) is 2.5–3x over threshold before the
worse jump at t=4s. Secondary factors (flat density, short duration) would
have made this a weaker demo pick than Myeongdong even in a world where
fixity had passed, so this isn't a close call being resolved by a hard
gate — it fails on both the gate and the qualitative fit.

### What switching would cost (not incurred — reported per instruction)

Not applicable, since the verdict is REJECT and nothing changes. For the
record, had this clip passed: a fresh interactive `calibrate_video.py`
run (impossible here regardless, given fixity — you'd need a genuinely
fixed shot of this venue first), re-derived `--heatmap-vmax` and
`--frame-heatmap-vmax` (this clip's own detection density, not
Myeongdong's or ToTest's), a fresh tile-config choice from a full STEP
3–6 sweep, and every density/persons-per-m² number currently in
`docs/REAL_FOOTAGE_FINDINGS.md` would need a Testing.mp4-specific
equivalent rather than being comparable at all. None of this is needed
now — Myeongdong remains the incumbent.

---

## Summary (printed to stdout)

```
STEP0: 2320x1080  30.0fps  336 frames  11.2s  h264
STEP1: STOP CONDITION TRIGGERED. Median ORB drift 27.3px (p90 39.0px, max 417.4px) vs ~2px threshold --
  not localised, not constant: 5-6px for the first 3s, then a 5x jump to ~25px at t=4s, climbing to ~51px
  by t=11s. Visually confirmed (pillar position shifts, frame 0 vs frame 330). Camera is NOT fixed.
STEP2: calibration/Testing.json exists -- USER_MEASURED, 5.0x3.0m, round_trip_error=0.0px (better than
  Myeongdong's ESTIMATED R^2=0.464 calibration on paper). Invalidated by STEP1 regardless.
STEP3 (pixel-space only, fixity failed): tiling gain mean ratio A=1.00x B=2.03x C=2.49x D=2.74x E=0.80x --
  the MOST STABLE tiling-gain table of the three clips measured so far. Steady-state fps: A=11.58 B=3.76
  C=1.77 D=2.44 E=28.78.
STEPS 4-6: skipped per the stop condition.
STEP7 VERDICT: REJECT. Deciding factor: camera motion (27.3px median drift, ~14x over threshold), same
  failure mode as ToTest. Secondary: flat density (46-65 raw detections across the whole clip, no
  build-up) and shortest duration of the three (11.2s) would have made this a weaker demo pick even had
  fixity passed. Myeongdong remains the incumbent demo clip.
```

---

## Addendum — audit for defaults/thresholds tuned on ToTest.mp4 numbers

`data/ToTest.mp4` was later rejected outright and renamed to
`data/REJECTED_ToTest_moving_camera.mp4` (92.7px median ORB drift). This
audit checks whether anything in the codebase quietly baked in a
ToTest-specific number as a default.

**Found and fixed**: `scripts/plot_confidence_histogram.py`'s `--frames`
default was literally `[180, 300, 450, 915]` -- ToTest's own
density-scan-selected frame indices. Frame 915 doesn't exist in either
valid clip (Myeongdong: 540 frames, Testing.mp4: 336 frames) -- the script
fails gracefully (prints "could not read frame 915, skipping" and
continues with 3 frames instead of 4) rather than crashing, but it's a
real ToTest-tuned default. Changed to `[0, 30, 60, 90]` (clip-agnostic:
first ~3s at typical fps).

**Checked and NOT ToTest-specific**:
- `--confidence-threshold` default `0.02` (config default, iterated
  `0.3 -> 0.2 -> 0.1 -> 0.02` across several turns): the underlying
  observation (a huge low-confidence noise spike, no clean bimodal border)
  was confirmed on BOTH ToTest.mp4 and Testing.mp4 independently, and
  confidence-score distribution is a pixel-space/detection-quality
  property, not something camera motion invalidates the way it invalidates
  world-position/density figures. Still flagged as untested against a hand
  count on ANY clip, including the two valid ones -- that caveat was never
  ToTest-specific, it's still open.
- `--tile-size`/`--tile-overlap`/`--nms-iou-threshold` defaults
  (1280px/0.1/0.7): predate ToTest entirely, from Myeongdong's original
  C2.3 SAHI investigation.
- `--downscale` default (1920): given directly in this project's original
  task spec, not derived from any clip's measurements.
- `--imgsz` default (unset): deliberately chosen to preserve prior
  behaviour exactly, not tuned to any clip.
- `DEFAULT_DENSITY_VMAX_PERSONS_PER_M2` (4.0), `ALARM_ONSET_PERSONS_PER_M2`
  (2.5), `CRUSH_BENCHMARK_PERSONS_PER_M2` (3.0): given directly by task
  instruction and this project's own Myeongdong-derived crush-density
  finding: cross-checked against Myeongdong's real full-clip percentiles
  in this task's Task 3 (p99=2.14, max=5.98, only 0.08% of populated
  observations exceed 4.0) -- not ToTest-derived at all.

**Checked and genuinely uncertain**: `--frame-heatmap-cell-size-px` (60)
and `--frame-heatmap-vmax` (0.5) for the pixel-space camera overlay were
tuned by eye specifically on ToTest.mp4 footage in the session that added
that overlay. The underlying data (raw pixel-space detection density) isn't
invalidated by ToTest's camera motion the way world-coordinate figures are
-- pixel occupancy doesn't need a valid homography. Re-checked visually
against Myeongdong's real footage in this task's Task 3 (see the snapshot
PNGs) and it still looks reasonable there. But it was never checked against
a resolution/density combination very different from ToTest's, so treat it
as under-verified, not wrong -- `--frame-heatmap-vmax` remains a plain CLI
override if a future clip needs a different value.
