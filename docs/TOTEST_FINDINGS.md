# ToTest.mp4 — Findings (relative comparison sweep, 2026-08-20)

> **REJECTED as a demo clip — camera is not fixed (92.7px median ORB
> drift, see SETUP below).** File renamed to
> `data/REJECTED_ToTest_moving_camera.mp4` so it can't be reached for by
> accident. Its calibration and every density/persons-per-m² figure below
> are invalid. Pixel-space detection counts (TEST 2–5) are still real
> relative comparisons and remain useful for that purpose only. Do not use
> this clip's numbers as a default or threshold anywhere else in this
> project — see docs/TESTING_FINDINGS.md's audit for defaults that need
> checking against this.

Video: `data/REJECTED_ToTest_moving_camera.mp4`. Scene: an indoor shopping mall (John Lewis /
Westfield Square signage), elevated balcony view, dense multi-level crowd.
**No hand counts exist for this clip yet** — see `docs/handcount/README.md`
(table stubbed, marked PENDING). **Nothing below is a recall or accuracy
claim.** Every count is either a raw detector output or a relative
comparison between two detector configurations on the same frames.

---

## SETUP

**Resolution**: 2560×1440. **FPS**: 30.518. **Frame count**: 954.
**Duration**: 31.26s. **Codec**: h264.

**ORB drift check (informational only — not a stop condition this
round)**: same method as the Myeongdong footage — ORB keypoint matching on
a static signage patch (rows 0–260, cols 0–500: "JOHN LEWIS & PARTNERS" +
"H&M" signage, no crowd), 5-point sample at 0/25/50/75/100% of the clip.
**Median drift 92.7px, max drift 451.1px**, growing monotonically from
36.7px (25%) to 148.0px (100%) — this camera visibly pans/zooms over the
clip. Reported as requested; not treated as blocking, and no calibration
or homography work was skipped because of it in this task (unlike the
prior run on this same clip, where it was a stop condition — that
constraint doesn't apply here).

**Calibration**: `calibration/ToTest.json` **exists** — `source:
USER_MEASURED`, world 9.0m × 15.0m, near/far-edge scale 10.70/14.57
mm/px (foreshortening ratio 1.36×), `round_trip_error_px: 0.0`. Since it
exists, TEST 5 (full pipeline, in-quad + world coordinates) runs below.
Given the ORB drift just measured, **the in-quad/density numbers in TEST 5
carry the same caveat the drift implies** — see TEST 5's own caveat
paragraph. I did not regenerate, edit, or reuse any other calibration file
for this clip.

---

## FRAME SELECTION

See `docs/handcount/README.md` for the full stub. Frames **180, 300, 450,
915** (5.90s / 9.83s / 14.75s / 29.98s), selected from a tiled-detector
raw-count scan (every 15th frame, 64 samples, config A: tile=1280px /
imgsz=640, the pipeline's current default) as a **relative density proxy
only**: picked the observed minimum, the observed maximum, one mid-range
point, and a second high point late in the clip for temporal spread.

Exported to `docs/handcount/`:
- `ToTest_frame<N>_clean.png` — clean, full-resolution, zero annotations
- `ToTest_frame<N>_baselineA.png` — same frame with config A's boxes drawn
  (orange = confidence <0.4, green = confidence ≥0.4)

---

## TEST 1 — existing test suite

```
uv run pytest tests/ -q
```
**40 passed, 0 failed** (36 pre-existing + 4 new, added alongside the
`imgsz`/before-NMS/after-NMS instrumentation described below —
`test_imgsz_forwarded_to_ultralytics_when_set`,
`test_imgsz_omitted_by_default_preserving_ultralytics_default`,
`test_before_after_nms_and_tile_count_exposed`,
`test_before_after_nms_equal_when_tiling_disabled`). Nothing regressed.

---

## TEST 2 — tiling geometry sweep

### imgsz, reported before the table

**Confirmed: `imgsz=640` is applied to every tile regardless of tile
size, including 1280px tiles** — `perception/detector.py`'s
`Detector.detect()` previously never passed `imgsz` to the Ultralytics
call at all, so Ultralytics' own model default (`640` for `yolo11s.pt`,
confirmed via `model.overrides['imgsz']`) applied unconditionally. A
1280×1280 tile was being halved on the way into the network. **This is
now configurable** (see "CLI flags" below) but the default is unchanged —
config A below reproduces the exact current shipped behaviour.

### Sweep results, 4 frames × 5 configs

`overlap_ratio=0.1`, cross-tile `nms_iou_threshold=0.7`, `conf=0.3`,
per-tile YOLO `iou=0.7` — held constant; only `tile_size`/`imgsz` vary.
Run through the real production `Detector`/`TiledDetector` classes (not a
standalone script) using the new `imgsz` parameter and the new
`last_n_tiles` / `last_n_before_nms` / `last_n_after_nms` instrumentation
added to `TiledDetector` for exactly this purpose.

| frame | config | tiles/frame | before NMS | after NMS | fps (detect-only) |
|---|---|---:|---:|---:|---:|
| 180 | A tile1280/imgsz640 (baseline) | 6 | 45 | 23 | 0.51 (warmup, see below) |
| 180 | B tile1280/imgsz1280 | 6 | 207 | 92 | 0.89 (warmup) |
| 180 | C tile960/imgsz960 | 6 | 116 | 87 | 1.24 (warmup) |
| 180 | D tile640/imgsz640 | 15 | 163 | 120 | 1.20 (warmup) |
| 180 | E no-tile/imgsz640 (floor) | 1 | 3 | 3 | 2.58 (warmup) |
| 300 | A | 6 | 56 | 31 | 3.50 |
| 300 | B | 6 | 207 | 82 | 1.07 |
| 300 | C | 6 | 114 | 85 | 1.84 |
| 300 | D | 15 | 172 | 123 | 1.42 |
| 300 | E | 1 | 1 | 1 | 14.42 |
| 450 (peak) | A | 6 | 69 | 38 | 3.51 |
| 450 | B | 6 | 204 | 95 | 1.08 |
| 450 | C | 6 | 120 | 84 | 1.80 |
| 450 | D | 15 | 171 | 127 | 1.47 |
| 450 | E | 1 | 11 | 11 | 27.40 |
| 915 | A | 6 | 72 | 36 | 3.54 |
| 915 | B | 6 | 191 | 86 | 1.12 |
| 915 | C | 6 | 97 | 69 | 1.88 |
| 915 | D | 15 | 170 | 121 | 1.38 |
| 915 | E | 1 | 9 | 9 | 33.71 |

Frame 180 was each config's first call and includes one-time MPS
kernel-compile warmup (most visible on A: 0.51 vs its own steady-state
3.5); **steady-state fps** (mean of frames 300/450/915):

| config | steady-state fps |
|---|---:|
| A tile1280/imgsz640 | 3.52 |
| B tile1280/imgsz1280 | 1.09 |
| C tile960/imgsz960 | 1.84 |
| D tile640/imgsz640 | 1.42 |
| E no-tile/imgsz640 | 25.2 (range 14.4–33.7 — noisy, driven by how much post-processing near-zero detection counts skip) |

### Summary: detections-after-NMS as a multiple of baseline A

| frame | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| 180 | 1.00x | 4.00x | 3.78x | 5.22x | 0.13x |
| 300 | 1.00x | 2.65x | 2.74x | 3.97x | 0.03x |
| 450 | 1.00x | 2.50x | 2.21x | 3.34x | 0.29x |
| 915 | 1.00x | 2.39x | 1.92x | 3.36x | 0.25x |
| **mean** | **1.00x** | **2.88x** | **2.66x** | **3.97x** | **0.18x** |

Every tiled config (B/C/D) roughly doubles-to-quadruples detection counts
over the current baseline (A) on every sampled frame; E (no tiling) is far
below A on every frame. D has the highest ratio but also the most tiles
(15, finest granularity) — see TEST 3 before treating that as free.

---

## TEST 3 — fragmentation (configs C and D)

**BBox height vs tile size, all 4 frames:**

| config (tile) | frame | n | height p50 | height p90 | height max | 60%-of-tile | n exceeding |
|---|---|---:|---:|---:|---:|---:|---:|
| C (960px) | 180 | 87 | 187px | 295px | 355px | 576px | **0** |
| C (960px) | 300 | 85 | 194px | 286px | 341px | 576px | **0** |
| C (960px) | 450 | 84 | 187px | 284px | 306px | 576px | **0** |
| C (960px) | 915 | 69 | 192px | 283px | 327px | 576px | **0** |
| D (640px) | 180 | 120 | 165px | 279px | 356px | 384px | **0** |
| D (640px) | 300 | 123 | 168px | 276px | 340px | 384px | **0** |
| D (640px) | 450 | 127 | 149px | 259px | 304px | 384px | **0** |
| D (640px) | 915 | 121 | 141px | 254px | 328px | 384px | **0** |

**Zero detections exceed 60% of their tile's height anywhere in this
sweep.** Tallest box observed: 356px (D, frame 180) vs a 384px threshold.

**Stacked-partial-box heuristic** (horizontal-IoU ≥0.6, small vertical
gap between one box's bottom and the other's top, low full-box IoU):
flagged 17/16/10/3 pairs for C and 25/18/13/12 for D across the 4 frames.
**I visually spot-checked 2 flagged pairs — both were two distinct real
people** standing/walking vertically close together (an adult+child on an
escalator; two people on a stairway), **not fragmentation.** This mall has
multiple levels/escalators/stairs in frame, which produces exactly this
geometry constantly among real, separate people — the heuristic has real
false positives here. **I did not check all ~90 flagged pairs** —
combined with the zero-over-60%-height result I found no confirmed
fragmentation in what I checked, but this is not exhaustive.

Bottom-third annotated crops (config A and D, all 4 frames):
`outputs/tiling_sweep/{A_tile1280_imgsz640,D_tile640_imgsz640}_frame{180,300,450,915}_bottom_third.png`

---

## TEST 4 — false-positive signals

**Confidence 0.3–0.4 detections per config:**

| config | 180 | 300 | 450 | 915 |
|---|---|---|---|---|
| A (baseline) | 12/23 (52%) | 13/31 (42%) | 20/38 (53%) | 19/36 (53%) |
| B | 28/92 (30%) | 18/82 (22%) | 28/95 (29%) | 26/86 (30%) |
| C | 21/87 (24%) | 21/85 (25%) | 25/84 (30%) | 21/69 (30%) |
| D | 27/120 (23%) | 27/123 (22%) | 25/127 (20%) | 30/121 (25%) |
| E (no-tile) | 3/3 (100%) | 0/1 (0%) | 5/11 (45%) | 4/9 (44%) |

**Baseline A has the highest low-confidence share of any tiled config** —
roughly half its detections sit in 0.3–0.4, vs a fifth-to-a-third for
B/C/D. Consistent with (not proof of) the imgsz-640-on-1280-tile
downscaling producing blurrier input than B/C/D's higher-resolution tiles.

**Visual check, densest frame (450)**: inspected the full annotated frame
for A, B, and D (`outputs/tiling_sweep/*_frame450_full.png`) and zoomed
specifically into the storefront/mannequin area (rows 150–450, cols
400–800 — several headless mannequin torsos visible in the John Lewis
window). **No boxes landed on the mannequins in what I checked.** Not an
exhaustive scan of every one of B/C/D's 82–127 boxes per frame.

---

## TEST 5 — full pipeline run

`calibration/ToTest.json` exists, so this ran. **Config used: B
(tile=1280px, imgsz=1280).** Chosen over C and D specifically because it
matches baseline A's tile *geometry* exactly (6 tiles/frame, same seams) —
so it isolates "what does full-resolution-per-tile inference recover"
without also introducing C/D's finer, unconfirmed-but-flagged (TEST 3)
seam-splitting exposure. It has a lower mean gain than D (2.88x vs
3.97x, TEST 2) but D's gain is entangled with 15 tiles/frame instead of
6 — B is the safer "best" pick on this evidence, not the single
highest-numbered one.

```
uv run python scripts/live_perception.py --video data/REJECTED_ToTest_moving_camera.mp4 \
    --calibration calibration/ToTest.json --no-window \
    --tile --tile-size 1280 1280 --tile-overlap 0.1 --imgsz 1280 --downscale 0
```

Full 954-frame run, ~14 minutes wall-clock at ~1.1 fps steady state
(matches TEST 2's config-B measurement).

**Funnel totals**: total **81,521** → in_quad **6,797** → tracked **191**
→ emitted **191**. Mean in-quad detections/frame: 7.125 (this quad is a
small 9m×15m patch of a much larger, denser scene — only ~8.3% of raw
detections fall inside it at all).

**In-quad → tracked loss: 6,606 of 6,797 (97.19%)** — see TEST 6 for how
this compares to Myeongdong's already-documented ~85% loss on the exact
same tracker configuration (nothing retuned here, per your instruction).

**Churn summary**:
- Unique confirmed track IDs: **17** (over 954 frames, 191 rows)
- Lifetime: min 0.00s, **median 0.23s**, p90 1.04s, max 4.72s
- Surviving ≥1.0s: **17.6%** — Surviving ≥2.5s: **5.9%**
- Mean simultaneous tracked people/frame: **0.20**

**Peak heatmap density** (rolling 2.0s window, 0.5m cells, replayed over
all 954 frames from the persisted trajectory CSV — not just the 181
frames that had a confirmed track, which would silently skip most of the
window and distort the rolling average): **0.759 persons/m²** (density
mode) / **0.19 persons/cell** (count mode), both peaking at **frame 181
(t=5.93s)**. Calibration source: **`USER_MEASURED`** (world 9.0×15.0m).

**Caveat this MUST travel with, given the ORB drift measured in SETUP**:
median 92.7px / max 451.1px drift means the camera is a materially
different framing by the time you're 6+ seconds into the clip, while the
calibrated quad polygon is fixed at its frame-0 pixel coordinates for the
*entire* run. Past roughly the first second or two, "in_quad" is
increasingly testing membership against a quad that no longer corresponds
to the same real-world floor patch the camera is actually looking at —
and every world coordinate / density number inherits that same drift.
**Frame 181 (t=5.93s, where the reported peak density falls) already has
an estimated ~28px of drift built up** (linearly interpolating between the
t=0 baseline and the t=7.8s/36.7px SETUP sample — a rough estimate, not a
direct measurement at this exact frame) — treat the peak density figure as
indicative of order-of-magnitude only, not a precise 0.759. This is
exactly the caveat the original stop condition (previous task on this
same clip) existed to prevent silently skipping.

---

## TEST 6 — cross-footage comparison

Full write-up in `docs/REAL_FOOTAGE_FINDINGS.md` (new section appended,
alongside the original Myeongdong findings for direct comparison). Summary:

**Tiling gain ratio**: the *direction* generalizes (tiling substantially
increases detections on both clips, no exceptions) but the *magnitude*
does not generalize as a fixed multiplier. Myeongdong's tile1280/imgsz640-
vs-no-tile ratio was a fairly stable ~4.35x across its 5 sampled frames
(range 3.4–6.1x). The same comparison on ToTest averages ~11.5x but swings
3.4x–31x frame to frame, because the no-tile floor here is degenerate (1–11
raw detections/frame) — small absolute denominators make the ratio noisy,
not because the underlying effect is bigger. **Don't quote a single
tiling-gain multiplier as if it transfers between clips.**

**Tracker confirmation loss**: generalizes in *direction and severity
ranking* but got *worse*, not the same, on ToTest — 97.19% in-quad
detections lost before confirmation here vs Myeongdong's already-severe
85.21%, on the byte-identical tracker configuration (nothing retuned in
either investigation). The standout candidate explanation, directly
supported by this task's own SETUP measurement and not present on
Myeongdong: **this camera moves (median 92.7px drift) while Myeongdong's
was confirmed static (≤0.88px)**. Camera motion adds a second source of
frame-to-frame pixel displacement on top of each person's own movement,
which stresses exactly the IoU-based association `minimum_iou_threshold`
gates on — consistent with, but not proven by, a controlled test in this
task (that would need stabilizing the footage first, out of scope here).
Shorter median track lifetime (0.23s vs 0.667s) and lower survival
fractions at both thresholds point the same direction.

---

## CLI flags added

`scripts/live_perception.py` now exposes `--tile-size W H`,
`--tile-overlap <ratio>` (renamed from the previous `--tile-overlap-ratio`
to match what was asked for here), and `--imgsz <int>`. **Defaults
unchanged** — `--imgsz` defaults to unset (Ultralytics' own default, 640),
reproducing config A's exact current behaviour when no flags are passed.
`perception/detector.py`'s `Detector` gained the underlying `imgsz`
parameter and `TiledDetector` gained `last_n_tiles` /
`last_n_before_nms` / `last_n_after_nms` read-only properties (used to
build TEST 2's table without a throwaway script). 4 new tests cover both;
see TEST 1.

---

## Hand-count table

**PENDING** — see `docs/handcount/README.md`. Nothing above should be
treated as validated until that table is filled in.

---

## Summary (printed to stdout at completion)

```
SETUP: 2560x1440  30.518fps  954 frames  31.26s  h264
  ORB drift (informational, not a stop condition): median 92.7px, max 451.1px -- camera pans/zooms.
  calibration/ToTest.json exists (USER_MEASURED, 9.0x15.0m) -- TEST 5 ran.
TEST1: 40/40 tests pass (36 pre-existing + 4 new for the imgsz/instrumentation additions). No regressions.
TEST2: imgsz=640 confirmed applied to ALL tile sizes incl. 1280px by default (halves each tile on input).
  Mean detections-after-NMS as multiple of baseline A: B(tile1280/imgsz1280)=2.88x C(tile960/imgsz960)=2.66x
  D(tile640/imgsz640)=3.97x E(no-tile)=0.18x. Steady-state fps: A=3.52 B=1.09 C=1.84 D=1.42 E=~25(noisy).
TEST3: 0 detections exceed 60% of tile height in C or D, any frame. Stacked-partial-box heuristic flagged
  ~90 candidate pairs; 2 spot-checked, both real distinct people (multi-level mall, escalators/stairs),
  not fragmentation. Not exhaustively verified.
TEST4: baseline A has the highest low-confidence (0.3-0.4) share (~50%) of any config, ~2x B/C/D's share.
  No mannequin false positives found in the one region spot-checked on frame 450.
TEST5 (config B, full 954-frame run): funnel total=81521 in_quad=6797 tracked=191 emitted=191
  (97.19% in-quad loss before confirmation). 17 unique IDs, median lifetime 0.23s, p90 1.04s,
  17.6%/5.9% surviving >=1.0s/2.5s, mean simultaneous tracked/frame=0.20. Peak density 0.759 persons/m^2
  at frame 181 (t=5.93s) -- CAVEATED by the camera-motion finding above; not precise past the first ~1-2s.
TEST6: tiling-gain DIRECTION generalizes, magnitude does not (Myeongdong ~4.35x stable, ToTest ~11.5x
  but noisy/unstable). Tracker confirmation loss generalizes and WORSENS on ToTest (97.2% vs 85.2%),
  best explained by this clip's camera motion (absent on Myeongdong) stressing IoU-based association.
Added --tile-size/--tile-overlap/--imgsz CLI flags to live_perception.py; defaults unchanged.
```

