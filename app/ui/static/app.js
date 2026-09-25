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
    posture.textContent = { upright: "Upright", slouching: "Slouching", leaning: "Leaning" }[pose.posture];
    posture.dataset.state = pose.posture === "upright" ? "good" : "bad";
  } else {
    posture.textContent = ev.calibration.pose ? `Unknown (${pose ? pose.state : "no pose yet"})` : "Not calibrated";
    posture.dataset.state = "unknown";
  }
  $("pose-detail").textContent = !pose
    ? "no pose yet"
    : pose.slouching !== undefined
      ? `${pose.state}, slouching ${pose.slouching ? "yes" : "no"} (head drop ${fmt(pose.head_drop, 3)}), ` +
        `leaning ${pose.leaning ? "yes" : "no"} (shoulder tilt change ${fmt(pose.tilt_change_deg)} deg)`
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
  const runs = ev.audio_runs || [];
  const a = runs[runs.length - 1];
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

// ---------------------------------------------------------------- session flow

const session = { config: { replay: null, auto_stop_extra_s: 60 }, bank: null, state: "none", limitS: null,
  answerStart: null, timer: null };

const STATE_TEXT = {
  choose_question: "Choose a question.",
  calibrate: "Press Calibrate, then face the camera and sit straight for 3 seconds.",
  ready: "Calibrated. Press Start answer when you are ready.",
  answering: "Answering. Press Stop answer when you finish.",
  processing: "Processing the answer.",
  report: "Answer saved. Open the report below, or choose another question or calibrate again.",
};

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

function mmss(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function updateButtons() {
  const st = session.state;
  const on = state.running;
  $("select-question").disabled = !on || !["choose_question", "calibrate", "ready", "report"].includes(st);
  $("calibrate").disabled = !on || !["calibrate", "ready", "report"].includes(st);
  $("start-answer").disabled = !on || st !== "ready";
  $("stop-answer").disabled = !on || st !== "answering";
}

function tickTimer() {
  const s = (performance.now() - session.answerStart) / 1000;
  $("answer-timer").textContent = `${mmss(s)} of ${mmss(session.limitS)}. The answer stops by itself at the limit.`;
}

function startTimer() {
  session.answerStart = performance.now();
  tickTimer();
  session.timer = setInterval(tickTimer, 500);
}

function stopTimer() {
  if (session.timer) clearInterval(session.timer);
  session.timer = null;
}

function renderSessionState(ev) {
  const prev = session.state;
  session.state = ev.state;
  session.limitS = ev.limit_s;
  let text = STATE_TEXT[ev.state] || ev.state;
  if (ev.message) text += ` ${ev.message}`;
  if (ev.state === "report" && ev.summary) {
    text += ` Recorded ${ev.summary.frames} video frames and ${ev.summary.segments} speech segments.`;
  }
  $("session-state").textContent = text;
  if (ev.question) showQuestion(ev.question);
  if (ev.state === "answering" && prev !== "answering") startTimer();
  if (ev.state !== "answering") stopTimer();
  if (ev.state === "processing" && session.config.replay) video.pause();
  if (ev.state === "answering") $("report-link").hidden = true;
  if (ev.state === "report" && ev.report_id) {
    $("report-anchor").href = `report.html?id=${encodeURIComponent(ev.report_id)}`;
    $("report-link").hidden = false;
  }
  updateButtons();
  // Move focus to the next action so the whole flow works from the keyboard.
  const next = { calibrate: "calibrate", ready: "start-answer", answering: "stop-answer",
    report: ev.report_id ? "report-anchor" : "category" }[ev.state];
  if (next && prev !== ev.state) $(next).focus();
}

function currentQuestion() {
  return session.bank ? session.bank.questions.find((q) => q.id === $("question").value) : null;
}

function showQuestion(q) {
  if (!q) return;
  $("question-text").textContent = q.text;
  $("question-meta").textContent = `${q.type === "behavioral" ? "Behavioral" : "Technical"} question. ` +
    `Suggested time ${mmss(q.suggested_time_s)}. Stops by itself at ${mmss(q.suggested_time_s + session.config.auto_stop_extra_s)}.`;
}

function fillQuestions() {
  const cat = $("category").value;
  const select = $("question");
  select.replaceChildren();
  for (const q of session.bank.questions.filter((x) => x.category === cat)) {
    select.add(new Option(`${q.id}  ${q.text.length > 70 ? q.text.slice(0, 67) + "..." : q.text}`, q.id));
  }
  showQuestion(currentQuestion());
}

function renderBank(bank) {
  session.bank = bank;
  const cat = $("category");
  cat.replaceChildren();
  for (const [id, label] of Object.entries(bank.categories)) cat.add(new Option(label, id));
  fillQuestions();
  $("bank-status").textContent = `Question bank status: ${bank.status}`;
}

// ---------------------------------------------------------------- connection and sources

function onMessage(msg) {
  const ev = JSON.parse(msg.data);
  switch (ev.type) {
    case "status": renderModels(ev.models, ev.notice); break;
    case "session_state": renderSessionState(ev); break;
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

// Replay mode: the video file is the frame source. It stays paused except during
// calibration (first seconds of the file) and the answer (from the start of the file).
function loadReplayVideo() {
  return new Promise((resolve, reject) => {
    video.autoplay = false;
    video.srcObject = null;
    video.muted = true;
    video.loop = false;
    video.onloadeddata = () => { video.pause(); video.currentTime = 0; resolve(); };
    video.onerror = () => reject(new Error("the replay video could not be loaded"));
    video.src = session.config.replay.video_url;
  });
}

async function start() {
  show($("error"), "");
  $("start").disabled = true;
  try {
    if (session.config.replay) {
      await loadReplayVideo();
    } else {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: WIDTH }, height: { ideal: HEIGHT }, frameRate: { ideal: 30, max: 30 } },
        audio: false,
      });
      video.srcObject = state.stream;
      await video.play();
    }
    state.ws = await connect();
    Object.assign(state, { running: true, busy: false, frameId: 0, sentTimes: [], dropped: { busy: 0, buffered: 0 } });
    $("stop").disabled = false;
    updateButtons();
    tick();
    $("category").focus();
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
  if (session.config.replay) video.pause();
  stopTimer();
  $("start").disabled = false;
  $("stop").disabled = true;
  updateButtons();
}

function stop() {
  send({ type: "stop", browser: { dropped_busy: state.dropped.busy, dropped_buffered: state.dropped.buffered,
    frames_sent: state.frameId } });
  state.running = false;
  stopCapture();
}

$("start").addEventListener("click", start);
$("stop").addEventListener("click", stop);
$("category").addEventListener("change", fillQuestions);
$("question").addEventListener("change", () => showQuestion(currentQuestion()));
$("select-question").addEventListener("click", () => {
  const q = currentQuestion();
  if (q) send({ type: "select_question", id: q.id });
});
$("calibrate").addEventListener("click", () => {
  send({ type: "calibrate" });
  if (session.config.replay) {
    video.currentTime = 0;
    video.play();
    setTimeout(() => {
      if (session.state !== "answering") { video.pause(); video.currentTime = 0; }
    }, 3600);
  }
});
$("start-answer").addEventListener("click", () => {
  send({ type: "start_answer" });
  $("transcript").replaceChildren();
  if (session.config.replay) { video.currentTime = 0; video.play(); }
});
$("stop-answer").addEventListener("click", () => {
  send({ type: "stop_answer" });
  if (session.config.replay) video.pause();
});
video.addEventListener("ended", () => {
  if (session.config.replay && session.state === "answering") send({ type: "stop_answer", reason: "end of replay video" });
});

async function init() {
  try {
    const [config, bank, status] = await Promise.all(
      ["/api/config", "/api/questions", "/api/status"].map((u) => fetch(u).then((r) => r.json())));
    session.config = config;
    renderBank(bank);
    renderModels(status.models, status.notice);
    if (config.replay) {
      $("start").textContent = "Load replay video";
      $("camera-heading").textContent = "Replay video";
      show($("replay-banner"), `Replay mode: ${config.replay.video} and ${config.replay.audio} play from the start ` +
        "when you press Start answer.");
    }
  } catch (err) {
    show($("error"), `Could not load the app settings: ${err.message}`);
  }
  updateButtons();
}

init();
