# ToTest.mp4 — hand-count frames

> **REJECTED as a demo clip — camera is not fixed.** File renamed to
> `data/REJECTED_ToTest_moving_camera.mp4`. Frames below are still valid
> for pixel-space hand-counting (detection quality only); no world-metre
> or density figure from this clip should be trusted.

Video: `data/REJECTED_ToTest_moving_camera.mp4` (2560x1440, 30.518 fps, 954 frames, 31.26s).
Indoor shopping mall, elevated balcony view.

ORB camera-fixity check (informational only this round, not a stop
condition): median drift 92.7px, max 451.1px across the clip vs a frame-0
baseline — the camera pans/zooms noticeably. See `docs/TOTEST_FINDINGS.md`
SETUP for the full per-sample breakdown. Not treated as blocking here, but
worth knowing before reading too much into any single frame's absolute
scale.

## Frames chosen, and why

Selected from a tiled-detector raw-count scan (every 15th frame, 64
samples across the full clip, `tile=1280px / imgsz=640` — the pipeline's
current shipped default, i.e. config A in TEST 2) as a **relative density
proxy only** — not a hand count, not a correctness claim. Picked the
observed minimum, the observed maximum, one mid-range point, and a second
high point late in the clip for temporal spread rather than clustering two
picks near the same timestamp. Note this proxy config is the one TEST 2
shows undercounts the most relative to the alternatives -- the density
*ranking* across the scanned frames is still a reasonable ordering (all
configs' counts move together directionally on this footage), but don't
read the specific proxy numbers as absolute.

| Frame | Timestamp | Density band |
|---|---|---|
| 180 | 5.90s | low |
| 300 | 9.83s | medium |
| 450 | 14.75s | peak |
| 915 | 29.98s | high, late in the clip (temporal spread from 450) |

Files per frame, in `docs/handcount/`:
- `ToTest_frame<N>_clean.png` — clean, full-resolution, **zero annotations**
- `ToTest_frame<N>_baselineA.png` — same frame with the **baseline config
  (A: tile=1280px, imgsz=640, the pipeline's current shipped default)**
  boxes drawn on it, confidence printed per box. Orange = confidence <0.4,
  green = confidence >=0.4. This is what the detector you'd actually run
  today sees — useful context for judging its counts against your own,
  but don't let it anchor your count before you've made it independently.

## Hand-count table — PENDING

Count from the **clean** image first; use the `_baselineA` copy only
afterward, if at all, to compare. Counts are people, not boxes.

| Frame | Your count | Method | Confidence/notes | Date |
|---|---|---|---|---|
| 180 | PENDING | | | |
| 300 | PENDING | | | |
| 450 | PENDING | | | |
| 915 | PENDING | | | |
