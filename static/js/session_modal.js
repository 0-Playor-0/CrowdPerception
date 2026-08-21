// FAB + 3-step session-start modal. Talks to /api/videos, /api/videos/upload,
// /api/calibrations, /api/calibration/points, and /api/session/start|stop|status.
// No inline image/base64 streaming here -- this only ever sets up a session;
// the MJPEG <img> tags and the telemetry websocket (telemetry.js) pick it up
// once the pipeline thread starts publishing frames.
//
// An uploaded file is sent to /api/videos/upload as soon as it's picked --
// not deferred to session-start -- specifically so it lands on disk in time
// for "calibrate now" (step 2) to grab a frame from it. From that point on,
// an uploaded video and a catalog video are the same thing: a filename
// relative to data/, carried in wizard.videoFilename.

const wizard = {
  step: 1,
  videoFilename: null,
  calibrationFilename: null,
};

function el(id) { return document.getElementById(id); }

function openModal() {
  el("session-modal").classList.remove("hidden");
  goToStep(1);
  refreshVideoList();
  refreshCalibrationList();
  refreshStopButtonVisibility();
}

function closeModal() {
  el("session-modal").classList.add("hidden");
}

async function refreshStopButtonVisibility() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/session/status`);
    const data = await res.json();
    el("btn-stop-session").style.display = data.status === "running" ? "inline-block" : "none";
  } catch (err) {
    // status endpoint should always be reachable locally; ignore transient errors
  }
}

function goToStep(step) {
  wizard.step = step;
  document.querySelectorAll(".modal-step").forEach((section) => {
    section.classList.toggle("hidden", Number(section.dataset.step) !== step);
  });
  document.querySelectorAll(".step-dot").forEach((dot) => {
    const n = Number(dot.dataset.step);
    dot.classList.toggle("active", n === step);
    dot.classList.toggle("done", n < step);
  });
  el("btn-back").disabled = step === 1;
  el("btn-next").textContent = step === 3 ? "Start session" : "Next";
  updateNextEnabled();
}

function updateNextEnabled() {
  const btn = el("btn-next");
  if (wizard.step === 1) {
    btn.disabled = !wizard.videoFilename;
  } else if (wizard.step === 2) {
    btn.disabled = !wizard.calibrationFilename;
  } else {
    btn.disabled = false;
  }
}

async function refreshVideoList() {
  const container = el("video-list");
  container.innerHTML = "<div class='modal-hint'>loading…</div>";
  const res = await fetch(`${API_BASE_URL}/api/videos`);
  const data = await res.json();
  container.innerHTML = "";
  if (data.videos.length === 0) {
    container.innerHTML = "<div class='modal-hint'>no videos in data/ yet — upload one below</div>";
  }
  data.videos.forEach((v) => {
    const item = document.createElement("div");
    item.className = "option-item";
    item.innerHTML = `<span>${v.filename}</span><span class="option-meta">${(v.size_bytes / 1e6).toFixed(1)} MB</span>`;
    item.addEventListener("click", () => {
      wizard.videoFilename = v.filename;
      el("video-upload").value = "";
      el("upload-status").textContent = "";
      el("upload-status").className = "upload-status";
      document.querySelectorAll("#video-list .option-item").forEach((n) => n.classList.remove("selected"));
      item.classList.add("selected");
      updateNextEnabled();
    });
    container.appendChild(item);
  });
}

async function refreshCalibrationList() {
  const container = el("calibration-list");
  container.innerHTML = "<div class='modal-hint'>loading…</div>";
  const res = await fetch(`${API_BASE_URL}/api/calibrations`);
  const data = await res.json();
  container.innerHTML = "";
  if (data.calibrations.length === 0) {
    container.innerHTML = "<div class='modal-hint'>no calibration files yet — calibrate now below</div>";
  }
  data.calibrations.forEach((c) => {
    const item = document.createElement("div");
    item.className = "option-item";
    const dims = c.world_width_m && c.world_height_m ? `${c.world_width_m}m × ${c.world_height_m}m` : "";
    item.innerHTML = `<span>${c.filename}</span><span class="option-meta">${c.source} ${dims}</span>`;
    item.addEventListener("click", () => {
      wizard.calibrationFilename = c.filename;
      document.querySelectorAll("#calibration-list .option-item").forEach((n) => n.classList.remove("selected"));
      item.classList.add("selected");
      updateNextEnabled();
    });
    container.appendChild(item);
  });
}

function setupCalibrateNow() {
  el("btn-calibrate-now").addEventListener("click", async () => {
    if (!wizard.videoFilename) {
      alert("Choose or upload a video in step 1 first.");
      return;
    }
    el("calibrate-panel").classList.remove("hidden");
    el("calibration-error").classList.add("hidden");
    await CalibrationClicker.loadFrame(wizard.videoFilename);
  });

  el("btn-calib-undo").addEventListener("click", () => CalibrationClicker.undo());
  el("btn-calib-reset").addEventListener("click", () => CalibrationClicker.reset());

  el("btn-calib-submit").addEventListener("click", async () => {
    const points = CalibrationClicker.getPoints();
    const widthRaw = el("calib-width-m").value.trim();
    const heightRaw = el("calib-height-m").value.trim();
    const outputFilename = el("calib-output-filename").value.trim();
    const body = {
      video: CalibrationClicker.getVideo(),
      points,
      world_width_m: widthRaw === "" ? null : Number(widthRaw),
      world_height_m: heightRaw === "" ? null : Number(heightRaw),
      frame_index: 0,
      output_filename: outputFilename || undefined,
      overwrite: el("calib-overwrite").checked,
    };
    const errorEl = el("calibration-error");
    errorEl.classList.add("hidden");
    try {
      const res = await fetch(`${API_BASE_URL}/api/calibration/points`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        errorEl.textContent = data.detail || "calibration failed";
        errorEl.classList.remove("hidden");
        return;
      }
      wizard.calibrationFilename = data.written;
      await refreshCalibrationList();
      updateNextEnabled();
      errorEl.classList.add("hidden");
      el("calibrate-panel").classList.add("hidden");
    } catch (err) {
      errorEl.textContent = "network error writing calibration";
      errorEl.classList.remove("hidden");
    }
  });
}

async function startSession() {
  el("start-progress").classList.remove("hidden");
  el("start-error").classList.add("hidden");
  el("btn-next").disabled = true;

  const form = new FormData();
  form.append("video_filename", wizard.videoFilename);
  form.append("calibration_filename", wizard.calibrationFilename);
  const rangePreset = document.querySelector('input[name="range-preset"]:checked').value;
  form.append("range_preset", rangePreset);
  form.append("tile", el("opt-tile").checked);
  form.append("downscale", el("opt-downscale").checked);

  try {
    const res = await fetch(`${API_BASE_URL}/api/session/start`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "failed to start session");
    }
    await waitForRunning();
    closeModal();
  } catch (err) {
    el("start-progress").classList.add("hidden");
    el("start-error").textContent = err.message;
    el("start-error").classList.remove("hidden");
    el("btn-next").disabled = false;
  }
}

async function waitForRunning(timeoutMs = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const res = await fetch(`${API_BASE_URL}/api/session/status`);
    const data = await res.json();
    if (data.status === "running" && data.telemetry.frame_idx > 0) return;
    if (data.status === "error") throw new Error(data.error || "pipeline failed to start");
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error("timed out waiting for the first frame");
}

document.addEventListener("DOMContentLoaded", () => {
  el("fab").addEventListener("click", openModal);
  el("modal-close").addEventListener("click", closeModal);

  el("video-upload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    wizard.videoFilename = null;
    document.querySelectorAll("#video-list .option-item").forEach((n) => n.classList.remove("selected"));
    updateNextEnabled();

    const statusEl = el("upload-status");
    statusEl.textContent = `uploading ${file.name}…`;
    statusEl.className = "upload-status uploading";
    el("btn-next").disabled = true;

    const form = new FormData();
    form.append("video_file", file);
    try {
      const res = await fetch(`${API_BASE_URL}/api/videos/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "upload failed");

      wizard.videoFilename = data.filename;
      statusEl.textContent = `uploaded: ${data.filename}`;
      statusEl.className = "upload-status success";
      updateNextEnabled();
    } catch (err) {
      statusEl.textContent = err.message;
      statusEl.className = "upload-status error";
      el("video-upload").value = "";
    }
  });

  el("btn-back").addEventListener("click", () => goToStep(Math.max(1, wizard.step - 1)));
  el("btn-next").addEventListener("click", () => {
    if (wizard.step < 3) {
      goToStep(wizard.step + 1);
    } else {
      startSession();
    }
  });

  el("btn-stop-session").addEventListener("click", async () => {
    await fetch(`${API_BASE_URL}/api/session/stop`, { method: "POST" });
    closeModal();
  });

  setupCalibrateNow();
});
