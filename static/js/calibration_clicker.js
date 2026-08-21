// 4-point calibration clicker: canvas + click handlers, no library. Mirrors
// scripts/calibrate_video.py's click order (near-left, near-right,
// far-right, far-left) and lets server/routes/calibration.py do the actual
// homography/validation work (build_calibration_record, reused as-is).

const CORNER_LABELS = ["near-left", "near-right", "far-right", "far-left"];

const CalibrationClicker = (() => {
  let canvas, ctx;
  let points = []; // in ORIGINAL image pixel coordinates
  let imageEl = null;
  let currentVideo = null;

  function init() {
    canvas = document.getElementById("calibration-canvas");
    ctx = canvas.getContext("2d");
    canvas.addEventListener("click", onCanvasClick);
  }

  // <video_stem>_<timestamp>.json -- deliberately NOT <video_stem>.json.
  // Calibration files are load-bearing (every world-coordinate and density
  // figure in the project's docs traces back to one), so a fresh
  // calibration must never land on an existing file by default -- see
  // server/routes/calibration.py's overwrite check, which this default is
  // designed to avoid tripping in normal use.
  function defaultOutputFilename(videoFilename) {
    const stem = videoFilename.replace(/\.[^./]+$/, "");
    return `${stem}_${Date.now()}.json`;
  }

  async function loadFrame(videoFilename) {
    currentVideo = videoFilename;
    points = [];
    const url = `${API_BASE_URL}/api/frame/first?video=${encodeURIComponent(videoFilename)}`;
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });
    imageEl = img;
    const displayWidth = Math.min(560, img.naturalWidth);
    const scale = displayWidth / img.naturalWidth;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width = displayWidth + "px";
    canvas.style.height = Math.round(img.naturalHeight * scale) + "px";

    const filenameInput = document.getElementById("calib-output-filename");
    filenameInput.value = defaultOutputFilename(videoFilename);
    document.getElementById("calib-overwrite").checked = false;

    render();
    updateStatus();
  }

  function onCanvasClick(event) {
    if (points.length >= 4 || !imageEl) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    points.push([x, y]);
    render();
    updateStatus();
  }

  function undo() {
    points.pop();
    render();
    updateStatus();
  }

  function reset() {
    points = [];
    render();
    updateStatus();
  }

  function render() {
    if (!imageEl) return;
    ctx.drawImage(imageEl, 0, 0);
    if (points.length >= 2) {
      ctx.strokeStyle = "#ffb547";
      ctx.lineWidth = Math.max(2, canvas.width / 400);
      ctx.beginPath();
      points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
      if (points.length === 4) ctx.closePath();
      ctx.stroke();
    }
    points.forEach(([x, y], i) => {
      ctx.fillStyle = "#2dd4a7";
      ctx.beginPath();
      ctx.arc(x, y, Math.max(5, canvas.width / 150), 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = `${Math.max(14, canvas.width / 45)}px sans-serif`;
      ctx.fillText(`${i}:${CORNER_LABELS[i]}`, x + 10, y - 10);
    });
  }

  function updateStatus() {
    const statusEl = document.getElementById("calibration-click-status");
    const submitBtn = document.getElementById("btn-calib-submit");
    if (points.length < 4) {
      statusEl.textContent = `${points.length} / 4 points — click ${CORNER_LABELS[points.length]}`;
      submitBtn.disabled = true;
    } else {
      statusEl.textContent = "4 / 4 points placed — enter dimensions (or leave blank for ESTIMATED) and submit";
      submitBtn.disabled = false;
    }
  }

  function getPoints() {
    return points.map(([x, y]) => [x, y]);
  }

  function getVideo() {
    return currentVideo;
  }

  return { init, loadFrame, undo, reset, getPoints, getVideo };
})();

document.addEventListener("DOMContentLoaded", () => CalibrationClicker.init());
