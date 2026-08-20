# Testing.mp4 — hand-count frames

Video: `data/Testing.mp4` (2320x1080, 30.0 fps, 336 frames, 11.2s).
Indoor event lobby / entry hall, elevated view, dense crowd in traditional
dress, glass entry doors with an even denser crowd visible outside.

**Camera fixity FAILED** (see `docs/TESTING_FINDINGS.md` STEP 1): median
ORB drift 27.3px, max 417.4px, visually confirmed (the entry pillar
visibly shifts position between frame 0 and frame 330). Do not read
anything ground-plane/metric into these frames -- pixel-space detection
counts only.

## Frames chosen, and why

Selected from a tiled-detector raw-count scan (every 15th frame, 23
samples across the full clip, config A: tile=1280px/imgsz=640) as a
**relative density proxy only**. Density here is much flatter than either
other clip in this project (46-65 detections across the whole scan, no
dramatic swing) -- picked the observed minimum, the observed maximum, one
mid-range point, and a late point for temporal spread.

| Frame | Timestamp | Raw count (proxy) |
|---|---|---|
| 30 | 1.00s | 46 (observed min) |
| 150 | 5.00s | 56 (mid-range) |
| 240 | 8.00s | 65 (observed max) |
| 315 | 10.50s | 52 (late, temporal spread) |

Files per frame, in `docs/handcount/`:
- `Testing_frame<N>_clean.png` -- clean, full-resolution, zero annotations
- `Testing_frame<N>_baselineA.png` -- config A boxes drawn (orange =
  confidence <0.4, green = confidence >=0.4)

## Hand-count table -- PENDING

Not a priority given the STEP 7 verdict (REJECT) -- fill in only if you
want detection-quality numbers independent of the reject decision.

| Frame | Your count | Method | Confidence/notes | Date |
|---|---|---|---|---|
| 30 | PENDING | | | |
| 150 | PENDING | | | |
| 240 | PENDING | | | |
| 315 | PENDING | | | |
