# Operator dashboard — findings

Silent-failure findings specific to `server/` and `static/` (the FastAPI
dashboard), kept separate from the video-analysis findings docs
(`REAL_FOOTAGE_FINDINGS.md`, `TESTING_FINDINGS.md`, `TOTEST_FINDINGS.md`),
which are about the perception pipeline itself, not the web layer around it.

---

## MJPEG parts need an explicit `Content-Length` header, or Chrome silently never renders them (2026-08-21)

**Symptom**: `GET /api/stream/camera` and `/api/stream/heatmap` both
returned `200 OK` with a correct `multipart/x-mixed-replace; boundary=...`
content type, and `curl` confirmed real JPEG bytes were flowing over the
wire at the expected frame rate. But the `<img>` tags pointed at those
URLs never rendered anything — `naturalWidth`/`naturalHeight` stayed `0`
and `complete` stayed `false` indefinitely. No console error, no failed
network request, nothing in DevTools to point at a cause.

**Cause**: each multipart part was written as

```
--frame\r\nContent-Type: image/jpeg\r\n\r\n<jpeg bytes>\r\n
```

with no `Content-Length` header. This is enough for a naive manual parser
(splitting on the boundary string) to reconstruct the frames correctly,
which is why `curl`-and-inspect testing looked fine. Chrome's own
multipart/x-mixed-replace decoder, at least in the version exercised here,
did not reliably find the end of a part without one — it needs either the
`Content-Length` or to scan for the next boundary occurrence, and in
practice the frames simply never completed decoding, with no error
surfaced anywhere.

**Fix**: add `Content-Length: <byte length of this part>` to each part's
headers (`server/routes/stream.py`). One header, no other change needed —
frames rendered immediately and continuously after.

**Why this belongs here**: this is exactly the "no error, no crash, just a
blank/wrong result" failure class the project's other findings docs track
for the perception pipeline itself — same discipline applies to the web
layer. If you add a third MJPEG-like endpoint later, keep the
`Content-Length` header; do not assume browser MJPEG decoders are as
forgiving as a `curl | split` script.
