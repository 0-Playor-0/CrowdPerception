// WebSocket telemetry client. Every number rendered on this page comes
// straight from the /ws/telemetry payload -- nothing here is computed or
// guessed client-side.

const DONUT_ARC_LENGTH = 157; // path length of the semicircle arcs in index.html

let wsReconnectDelayMs = 300;
const WS_RECONNECT_MAX_MS = 2000;

function fmt(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function applyIdleState() {
  document.getElementById("camera-subtitle").textContent = "No video loaded";
  document.getElementById("heatmap-subtitle").textContent = "No video loaded";

  setText("m-detected", "—");
  setText("m-inquad", "—");
  setText("m-peak-density", "—");
  setText("m-fps", "—");
  setText("m-peak-density-sub", " ");

  setText("f-detected", "—");
  setText("f-inquad", "—");
  setText("f-tracked", "—");
  setText("f-pct-inquad", "—");
  setText("f-pct-tracked", "—");
  document.getElementById("funnel-bar-inquad").style.width = "0%";
  document.getElementById("funnel-bar-tracked").style.width = "0%";

  setText("donut-inquad-value", "—");
  setText("donut-inquad-pct", " ");
  document.getElementById("donut-inquad-arc").style.strokeDashoffset = DONUT_ARC_LENGTH;

  setText("stat-detected", "—");
  setText("stat-ids", "—");
  setText("gauge-fps-value", "—");
  document.getElementById("gauge-fps-arc").style.strokeDashoffset = DONUT_ARC_LENGTH;

  const banner = document.getElementById("calibration-banner");
  banner.classList.remove("estimated");
  document.getElementById("calibration-banner-text").textContent =
    "No session running — start a session to see calibration source.";
}

function applyTelemetry(t) {
  const pill = document.getElementById("session-status-pill");
  pill.textContent = t.status;
  pill.className = "status-pill " + t.status;

  if (t.status === "idle") {
    applyIdleState();
    return;
  }

  if (t.status === "error") {
    document.getElementById("calibration-banner-text").textContent = t.error || "pipeline error";
    return;
  }

  if (t.status === "calibrating" || !t.calibration) {
    // running has been requested but no frame has arrived yet
    return;
  }

  const { counts, density, performance, calibration, detection } = t;

  const rangeLabel = detection
    ? `${detection.range_preset} range, conf ${fmt(detection.confidence_threshold, 2)}`
    : "";

  document.getElementById("camera-subtitle").textContent =
    `frame ${t.frame_idx} / ${t.total_frames} · ${fmt(performance.fps, 1)} fps` +
    (rangeLabel ? ` · ${rangeLabel}` : "");
  document.getElementById("heatmap-subtitle").textContent =
    `frame ${t.frame_idx} / ${t.total_frames}`;

  setText("m-detected", fmt(counts.detected));
  setText("m-inquad", fmt(counts.in_quad));
  setText("m-fps", fmt(performance.fps, 1));

  const peakEl = document.getElementById("m-peak-density");
  setText("m-peak-density", fmt(density.peak_persons_per_m2, 2));
  const peakWarn = density.peak_persons_per_m2 > 2.5;
  peakEl.classList.toggle("warn", peakWarn);
  setText("m-peak-density-sub", peakWarn ? "above alarm onset (2.5)" : " ");

  // Funnel
  setText("f-detected", fmt(counts.detected));
  setText("f-inquad", fmt(counts.in_quad));
  setText("f-tracked", fmt(counts.tracked));
  const pctInQuad = counts.detected > 0 ? (counts.in_quad / counts.detected) * 100 : 0;
  const pctTracked = counts.in_quad > 0 ? (counts.tracked / counts.in_quad) * 100 : 0;
  setText("f-pct-inquad", fmt(pctInQuad, 0) + "%");
  setText("f-pct-tracked", fmt(pctTracked, 0) + "%");
  document.getElementById("funnel-bar-inquad").style.width = pctInQuad.toFixed(1) + "%";
  document.getElementById("funnel-bar-tracked").style.width =
    (counts.detected > 0 ? (counts.tracked / counts.detected) * 100 : 0).toFixed(1) + "%";

  // In-quad donut (retained % of all detections)
  setText("donut-inquad-value", fmt(counts.in_quad));
  setText("donut-inquad-pct", fmt(pctInQuad, 0) + "% of detected");
  const donutOffset = DONUT_ARC_LENGTH * (1 - pctInQuad / 100);
  document.getElementById("donut-inquad-arc").style.strokeDashoffset = donutOffset;

  // People stats + fps gauge
  setText("stat-detected", fmt(counts.detected));
  setText("stat-ids", fmt(t.total_ids_seen));
  setText("gauge-fps-value", fmt(performance.fps, 1));
  const fpsFrac = Math.min(1, performance.fps / 10); // gauge tops out at a generous 10fps reference
  document.getElementById("gauge-fps-arc").style.strokeDashoffset = DONUT_ARC_LENGTH * (1 - fpsFrac);

  // Calibration banner
  const banner = document.getElementById("calibration-banner");
  const isEstimated = calibration.source === "ESTIMATED";
  banner.classList.toggle("estimated", isEstimated);
  document.getElementById("calibration-banner-text").textContent =
    `${calibration.source} — ${calibration.note} (world: ${fmt(calibration.world_width_m, 2)}m x ${fmt(calibration.world_height_m, 2)}m)`;
}

function connectTelemetry() {
  const ws = new WebSocket(`${WS_BASE_URL}/ws/telemetry`);
  const dot = document.getElementById("ws-dot");
  const label = document.getElementById("ws-label");

  ws.onopen = () => {
    dot.className = "ws-dot connected";
    label.textContent = "live";
    wsReconnectDelayMs = 300;
  };

  ws.onmessage = (event) => {
    try {
      applyTelemetry(JSON.parse(event.data));
    } catch (err) {
      console.error("telemetry parse error", err);
    }
  };

  ws.onclose = () => {
    dot.className = "ws-dot disconnected";
    label.textContent = "reconnecting…";
    setTimeout(connectTelemetry, wsReconnectDelayMs);
    wsReconnectDelayMs = Math.min(WS_RECONNECT_MAX_MS, wsReconnectDelayMs * 1.5);
  };

  ws.onerror = () => ws.close();
}

document.addEventListener("DOMContentLoaded", () => {
  applyIdleState();
  document.getElementById("camera-stream").src = `${API_BASE_URL}/api/stream/camera`;
  document.getElementById("heatmap-stream").src = `${API_BASE_URL}/api/stream/heatmap`;
  connectTelemetry();
});
