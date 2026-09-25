// Placement Mirror browser side: capture, canvas resize, raw RGB frames over one WebSocket.
// Frame message layout matches app/vision/pipeline.py parse_frame (little endian):
// "PMF1", frame_id u32, capture_ms f64, width u16, height u16, face size u16, pose size u16,
// then RGB bytes of the frame, the face detector input and the pose detector input.
"use strict";

const WIDTH = 640;
const HEIGHT = 480;
const FACE_SIZE = 256;
const POSE_SIZE = 128;
let maxFps = 30; // set by the server's mode event
const HEADER_BYTES = 24;

const $ = (id) => document.getElementById(id);
const video = $("preview");
const frameCanvas = $("frame-canvas");
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });
const faceCtx = $("face-canvas").getContext("2d", { willReadFrequently: true });
const poseCtx = $("pose-canvas").getContext("2d", { willReadFrequently: true });

const state = {
  ws: null,
  stream: null,
  running: false,
  busy: false,
  want: { face: true, pose: true },
  frameId: 0,
  lastSend: 0,
  lastDecoded: -1,
  sentTimes: [],
  dropped: { busy: 0, buffered: 0 },
  lastPose: null,
};

function show(el, text) {
  el.textContent = text;
  el.hidden = !text;
}

function fmt(v, digits = 1) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "none";
}

// Letterbox like app.vision.roi.letterbox_affine: keep aspect, center, zero padding.
function letterbox(ctx, size) {
  const scale = Math.min(size / WIDTH, size / HEIGHT);
  const w = Math.floor(WIDTH * scale);
  const h = Math.floor(HEIGHT * scale);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(frameCanvas, 0, 0, WIDTH, HEIGHT, Math.floor((size - w) / 2), Math.floor((size - h) / 2), w, h);
  return ctx.getImageData(0, 0, size, size).data;
}

function rgbaToRgb(rgba, out, offset) {
  for (let i = 0, j = offset; i < rgba.length; i += 4, j += 3) {
    out[j] = rgba[i];
    out[j + 1] = rgba[i + 1];
    out[j + 2] = rgba[i + 2];
  }
}

function drawFrame() {
  // Center crop the camera image to 4:3 so the 640x480 frame is not stretched.
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const target = WIDTH / HEIGHT;
  let sw = vw;
  let sh = vh;
  if (vw / vh > target) sw = Math.round(vh * target);
  else sh = Math.round(vw / target);
  frameCtx.drawImage(video, (vw - sw) / 2, (vh - sh) / 2, sw, sh, 0, 0, WIDTH, HEIGHT);
}

function sendFrame(now) {
  const ws = state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (state.busy) {
    state.dropped.busy += 1;
    return;
  }
  if (ws.bufferedAmount > 2 * WIDTH * HEIGHT * 3) {
    state.dropped.buffered += 1;
    return;
  }
  drawFrame();
  const faceSize = state.want.face ? FACE_SIZE : 0;
  const poseSize = state.want.pose ? POSE_SIZE : 0;
  const frameBytes = WIDTH * HEIGHT * 3;
  const buf = new ArrayBuffer(HEADER_BYTES + frameBytes + faceSize * faceSize * 3 + poseSize * poseSize * 3);
  const view = new DataView(buf);
  [0x50, 0x4d, 0x46, 0x31].forEach((b, i) => view.setUint8(i, b));
  view.setUint32(4, state.frameId, true);
  view.setFloat64(8, now, true);
  view.setUint16(16, WIDTH, true);
  view.setUint16(18, HEIGHT, true);
  view.setUint16(20, faceSize, true);
  view.setUint16(22, poseSize, true);
  const bytes = new Uint8Array(buf);
  rgbaToRgb(frameCtx.getImageData(0, 0, WIDTH, HEIGHT).data, bytes, HEADER_BYTES);
  let offset = HEADER_BYTES + frameBytes;
  if (faceSize) {
    rgbaToRgb(letterbox(faceCtx, FACE_SIZE), bytes, offset);
    offset += faceSize * faceSize * 3;
  }
  if (poseSize) rgbaToRgb(letterbox(poseCtx, POSE_SIZE), bytes, offset);
  ws.send(buf);
  state.frameId += 1;
  state.sentTimes.push(now);
}

// Runs on every animation frame and sends only when the video has decoded a new frame,
// at most maxFps times a second. requestVideoFrameCallback is not used because headless
// Edge fires it at about 13 per second while the camera delivers 30.
function tick() {
  if (!state.running) return;
  const now = performance.now();
  const decoded = video.getVideoPlaybackQuality().totalVideoFrames;
  if (decoded !== state.lastDecoded && now - state.lastSend >= 1000 / maxFps - 2) {
    state.lastDecoded = decoded;
    state.lastSend = now;
    sendFrame(now);
  }
  state.sentTimes = state.sentTimes.filter((t) => t > now - 1000);
  $("sent-fps").textContent = String(state.sentTimes.length);
  $("dropped").textContent = String(state.dropped.busy + state.dropped.buffered);
  requestAnimationFrame(tick);
}

function renderModels(models, notice) {
  const body = $("models");
  body.replaceChildren();
  for (const [name, s] of Object.entries(models)) {
    const tr = document.createElement("tr");
    for (const text of [name, s.compute_unit, s.path, s.reason]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  show($("notice"), notice || "");
}

function renderIndicators(ev) {
  $("processed-fps").textContent = String(ev.processed_fps);
  $("latency").textContent = `${fmt(ev.server_latency_ms)} ms from arrival to result`;
  const face = ev.face;
  const facing = $("facing");
  if (face.facing === true || face.facing === false) {
    facing.textContent = face.facing ? "Yes" : "No";
    facing.dataset.state = face.facing ? "good" : "bad";
  } else {
    facing.textContent = ev.calibration.face ? `Unknown (${face.state})` : "Not calibrated";
    facing.dataset.state = "unknown";
  }
  $("face-detail").textContent = face.yaw === undefined
    ? face.state
    : `${face.state}, yaw ${fmt(face.yaw)}, pitch ${fmt(face.pitch)}` +
      (face.yaw_change !== undefined ? `, change from calibration yaw ${fmt(face.yaw_change)} pitch ${fmt(face.pitch_change)}` : "");

  const pose = ev.pose;
  const posture = $("posture");
  if (pose && pose.posture) {
    posture.textContent = pose.posture === "ok" ? "Upright" : "Check posture";
    posture.dataset.state = pose.posture === "ok" ? "good" : "bad";
  } else {
    posture.textContent = ev.calibration.pose ? `Unknown (${pose ? pose.state : "no pose yet"})` : "Not calibrated";
    posture.dataset.state = "unknown";
  }
  $("pose-detail").textContent = !pose
    ? "no pose yet"
    : pose.score !== undefined
      ? `${pose.state}, score ${fmt(pose.score, 2)}, shoulder tilt change ${fmt(pose.tilt_change_deg)} deg, head drop ${fmt(pose.head_drop, 3)}`
      : pose.state;
}

function renderCalibration(c) {
  const text = { "not calibrated": "Not calibrated", running: "Calibrating, hold still", done: "Done",
    partial: "Partly done", failed: "Failed" }[c.state] || c.state;
  $("calibration").textContent = c.message ? `${text}. ${c.message}` : text;
}

function renderStats(ev) {
  const b = ev.browser || {};
  $("stats-summary").textContent = `${fmt(ev.duration_s)} s, ${ev.frames_processed} frames processed of ` +
    `${ev.frames_received} received, ${ev.frames_replaced_in_mailbox} replaced on the server, ` +
    `${(b.dropped_busy || 0) + (b.dropped_buffered || 0)} dropped in the browser, face detector ran ${ev.face_detector_runs} times.`;
  const body = $("stats");
  body.replaceChildren();
  for (const [stage, s] of Object.entries(ev.stages)) {
    const tr = document.createElement("tr");
    for (const text of [stage, String(s.count), fmt(s.p50_ms, 2), fmt(s.p95_ms, 2)]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  const a = ev.audio;
  if (a && a.summary) {
    const s = a.summary;
    const p = (m) => (m && m.count ? `p50 ${fmt(m.p50, 0)} ms, p95 ${fmt(m.p95, 0)} ms` : "none");
    $("stats-summary").textContent += ` Audio: ${a.segments.length} segments, VAD latency ${p(s.vad_latency_ms)}, ` +
      `encoder ${p(s.encoder_ms)}, end to end ${p(s.end_to_end_delay_ms)}.`;
  }
  $("stats-section").hidden = false;
}

function renderSpeech(ev) {
  $("wpm").textContent = fmt(ev.wpm, 0);
  const parts = Object.entries(ev.fillers_by_word || {}).map(([w, n]) => `${w} ${n}`);
  $("fillers").textContent = parts.length ? `${ev.filler_count} (${parts.join(", ")})` : String(ev.filler_count);
  $("words").textContent = String(ev.words_total);
}

function renderTranscript(ev) {
  const li = document.createElement("li");
  li.textContent = ev.text.trim() || "(no words)";
  li.title = `${fmt(ev.audio_s)} s of audio, transcript ${fmt(ev.end_to_end_delay_ms, 0)} ms after speech ended`;
  $("transcript").appendChild(li);
  li.scrollIntoView({ block: "nearest" });
}

function renderPause(ev) {
  const pause = $("pause");
  if (ev.state === "started") {
    pause.textContent = "Pausing";
    pause.dataset.state = "bad";
  } else {
    pause.textContent = `Last pause ${fmt(ev.duration_s)} s`;
    pause.dataset.state = "unknown";
  }
}

function onMessage(msg) {
  const ev = JSON.parse(msg.data);
  switch (ev.type) {
    case "status": renderModels(ev.models, ev.notice); break;
    case "detector_inputs": state.want = { face: ev.face, pose: ev.pose }; break;
    case "busy": state.busy = ev.busy; break;
    case "indicators": renderIndicators(ev); renderCalibration(ev.calibration); break;
    case "calibration": renderCalibration(ev); break;
    case "stats": renderStats(ev); break;
    case "mode":
      maxFps = ev.face_max_fps;
      $("mode").textContent = ev.label;
      $("mode").dataset.state = ev.mode === "npu" ? "good" : "bad";
      break;
    case "speech": renderSpeech(ev); break;
    case "transcript": renderTranscript(ev); break;
    case "pause": renderPause(ev); break;
    case "audio": $("audio-source").textContent = `Listening: ${ev.source}`; break;
    case "error": show($("error"), ev.message); break;
    default: break;
  }
}

function connect() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { $("connection").textContent = "Connected"; resolve(ws); };
    ws.onerror = () => reject(new Error("WebSocket connection failed"));
    ws.onclose = () => { $("connection").textContent = "Not connected"; stopCapture(); };
    ws.onmessage = onMessage;
  });
}

async function start() {
  show($("error"), "");
  $("start").disabled = true;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: WIDTH }, height: { ideal: HEIGHT }, frameRate: { ideal: 30, max: 30 } },
      audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    state.ws = await connect();
    Object.assign(state, { running: true, busy: false, frameId: 0, sentTimes: [], dropped: { busy: 0, buffered: 0 } });
    $("calibrate").disabled = false;
    $("stop").disabled = false;
    tick();
  } catch (err) {
    show($("error"), `Could not start: ${err.message}`);
    stopCapture();
  }
}

function stopCapture() {
  state.running = false;
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  $("start").disabled = false;
  $("calibrate").disabled = true;
  $("stop").disabled = true;
}

function stop() {
  const ws = state.ws;
  state.running = false;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop", browser: { dropped_busy: state.dropped.busy, dropped_buffered: state.dropped.buffered, frames_sent: state.frameId } }));
  }
  stopCapture();
}

$("start").addEventListener("click", start);
$("stop").addEventListener("click", stop);
$("calibrate").addEventListener("click", () => {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "calibrate" }));
});

fetch("/api/status").then((r) => r.json()).then((s) => renderModels(s.models, s.notice)).catch(() => {});
