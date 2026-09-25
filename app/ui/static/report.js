// Report view for one saved answer: /report.html?id=<session id>
"use strict";

const $ = (id) => document.getElementById(id);

const MODE_TEXT = { npu: "NPU mode", cpu_fallback: "CPU fallback mode, reduced frame rate" };

function showError(text) {
  $("error").textContent = text;
  $("error").hidden = false;
}

function addPair(dl, term, value) {
  const div = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = term;
  dd.textContent = value;
  div.append(dt, dd);
  dl.append(div);
}

function windowName(w) {
  return `${mmss(w.start)} to ${mmss(w.end)}`;
}

function guideFor(report, metric) {
  return report.guidelines.find((g) => g.metric === metric) || null;
}

function valueWithUnit(value, unit) {
  return unit.startsWith("%") ? `${fixed(value)}${unit}` : `${fixed(value)} ${unit}`;
}

function renderHeader(r) {
  document.title = `Report ${r.question.id} - Placement Mirror`;
  $("question-text").textContent = r.question.text;
  const meta = $("answer-meta");
  addPair(meta, "Answered", r.answer.started.replace("T", " "));
  addPair(meta, "Question", `${r.question.id}, ${r.question.type}`);
  addPair(meta, "Answer length", `${mmss(r.answer.duration_s)} (limit ${mmss(r.answer.limit_s)})`);
  addPair(meta, "Stopped by", r.answer.stopped_by);
  addPair(meta, "Mode", MODE_TEXT[r.mode] || r.mode);
  if (r.replay) {
    $("replay-note").textContent = `Replay session: video ${r.replay.video} and audio ${r.replay.audio}, ` +
      "not a live camera and microphone.";
    $("replay-note").hidden = false;
  }
}

function renderImprovements(r) {
  const list = $("improvements");
  for (const item of r.feedback.improvements) {
    const li = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = item.label;
    const where = item.outside_guideline ? "Outside the guideline." : "Inside the guideline, near its limit.";
    const guide = document.createElement("span");
    guide.className = "guide-kind";
    guide.textContent = `Author's coaching default: ${item.guideline.text}.`;
    const action = document.createElement("p");
    action.className = "action";
    action.textContent = `Action: ${item.action}`;
    li.append(name, `: your value ${valueWithUnit(item.value, item.unit)}. ${where} `, guide, action);
    list.append(li);
  }
  if (r.feedback.note) {
    $("feedback-note").textContent = r.feedback.note;
    $("feedback-note").hidden = false;
  }
}

function renderSummary(r) {
  const o = r.overall;
  const dl = $("summary");
  addPair(dl, "Facing the camera", `${fixed(o.facing_camera_pct)}% of ${o.frames} video frames`);
  addPair(dl, "Not facing the camera", `${fixed(o.not_facing_pct)}% of video frames`);
  addPair(dl, "Face not visible", `${fixed(o.face_not_visible_pct)}% of video frames`);
  if (o.slouching_pct === null) {
    addPair(dl, "Posture", "not measured (shoulders were not seen at calibration or during the answer)");
  } else {
    addPair(dl, "Slouching", `${fixed(o.slouching_pct)}% of ${o.posture_frames} frames with posture`);
    addPair(dl, "Leaning", `${fixed(o.leaning_pct)}% of ${o.posture_frames} frames with posture`);
  }
  if (r.audio_source === null) {
    addPair(dl, "Speech", "not measured (no audio in this session)");
    return;
  }
  addPair(dl, "Speaking pace", `${fixed(o.wpm, 0)} words per minute (${o.words} words in ${mmss(o.duration_s)})`);
  addPair(dl, "Filler words", `${o.filler_count} (${fixed(o.fillers_per_min)} per minute)`);
  addPair(dl, "Long pauses", o.long_pause_count
    ? `${o.long_pause_count}, longest ${fixed(o.longest_pause_s)} s (${fixed(o.long_pauses_per_min)} per minute)`
    : "none");
}

function windowTicks(r) {
  const ticks = r.windows.map((w) => ({ value: w.start, text: mmss(w.start) }));
  const end = r.windows.length ? r.windows[r.windows.length - 1].end : 0;
  ticks.push({ value: end, text: mmss(end) });
  return { min: 0, max: end || 1, title: "Time in the answer (m:ss)", ticks };
}

function range(values, unit) {
  const v = values.filter((x) => x !== null);
  if (!v.length) return "no values";
  return `from ${fixed(Math.min(...v))}${unit} to ${fixed(Math.max(...v))}${unit}`;
}

function guideSentence(g) {
  return g ? ` Guideline for ${g.label.toLowerCase()}: ${g.text}. ${g.kind}` : "";
}

function renderFacing(r) {
  const values = r.windows.map((w) => w.facing_camera_pct);
  const hidden = r.windows.map((w) => w.face_not_visible_pct);
  const g = guideFor(r, "facing_camera_pct");
  const gHidden = guideFor(r, "face_not_visible_pct");
  const series = [
    { name: "Facing the camera", cls: "series-a", marker: "circle",
      points: r.windows.map((w) => ({ x0: w.start, x1: w.end, y: w.facing_camera_pct })) },
    { name: "Face not visible", cls: "series-c", marker: "square",
      points: r.windows.map((w) => ({ x0: w.start, x1: w.end, y: w.face_not_visible_pct })) },
  ];
  const box = $("facing-chart");
  box.append(drawChart({
    label: `Step chart per ${r.window_s} second window of the share of video frames facing the camera, ` +
      `${range(values, "%")}, and with the face not visible, ${range(hidden, "%")}.${guideSentence(g)}` +
      `${guideSentence(gHidden)} The table below lists every window.`,
    kind: "step", x: windowTicks(r), y: { max: 100, step: 20, title: "% of frames" }, series,
    guide: [g, gHidden].filter(Boolean),
  }));
  box.append(chartLegend(series));
  box.append(dataTable(`Face per ${r.window_s} s window`,
    ["Window", "Facing the camera %", "Not facing %", "Face not visible %", "Frames"],
    r.windows.map((w) => [windowName(w), fixed(w.facing_camera_pct), fixed(w.not_facing_pct),
      fixed(w.face_not_visible_pct), String(w.frames)])));
}

function renderPosture(r) {
  const box = $("posture-chart");
  if (r.overall.slouching_pct === null) {
    const p = document.createElement("p");
    p.textContent = "Posture was not measured in this answer. Calibrate with your shoulders in view to measure it.";
    box.append(p);
    return;
  }
  const g = guideFor(r, "slouching_pct");
  const lean = guideFor(r, "leaning_pct");
  const series = [
    { name: "Slouching", cls: "series-b", marker: "circle",
      points: r.windows.map((w) => ({ x0: w.start, x1: w.end, y: w.slouching_pct })) },
    { name: "Leaning", cls: "series-c", marker: "square",
      points: r.windows.map((w) => ({ x0: w.start, x1: w.end, y: w.leaning_pct })) },
  ];
  const sameGuide = g && lean && g.low === lean.low && g.high === lean.high;
  box.append(drawChart({
    label: `Step chart of slouching and leaning per ${r.window_s} second window as a share of frames with posture. ` +
      `Slouching ${range(r.windows.map((w) => w.slouching_pct), "%")}, leaning ` +
      `${range(r.windows.map((w) => w.leaning_pct), "%")}.${sameGuide ? guideSentence(g) : ""} ` +
      "The table below lists every window.",
    kind: "step", x: windowTicks(r), y: { max: 100, step: 20, title: "% of frames" }, series,
    guide: sameGuide ? { ...g, label: "Slouching and leaning" } : null,
  }));
  box.append(chartLegend(series));
  box.append(dataTable(`Posture flags per ${r.window_s} s window`,
    ["Window", "Slouching %", "Leaning %", "Frames with posture"],
    r.windows.map((w) => [windowName(w), fixed(w.slouching_pct), fixed(w.leaning_pct), String(w.posture_frames)])));
}

function renderPace(r) {
  const box = $("pace-chart");
  if (r.audio_source === null) {
    const p = document.createElement("p");
    p.textContent = "Speech was not measured in this answer.";
    box.append(p);
    return;
  }
  const values = r.windows.map((w) => w.wpm);
  const g = guideFor(r, "wpm");
  const y = yScale(values.concat(g && g.high !== null ? [g.high] : []), 100);
  const series = [{ name: "Words per minute", cls: "series-a", marker: "circle",
    points: r.windows.map((w) => ({ x0: w.start, x1: w.end, y: w.wpm })) }];
  box.append(drawChart({
    label: `Step chart of words per minute per ${r.window_s} second window, ${range(values, "")}.` +
      `${guideSentence(g)} The table below lists every window.`,
    kind: "step", x: windowTicks(r), y: { ...y, title: "Words per minute" }, series, guide: g,
  }));
  box.append(dataTable(`Speaking pace per ${r.window_s} s window`, ["Window", "Words", "Words per minute"],
    r.windows.map((w) => [windowName(w), String(w.words), fixed(w.wpm, 0)])));
}

function renderTranscript(r) {
  $("pause-min").textContent = String(r.pause_min_s);
  const p = $("transcript");
  if (!r.transcript.length) {
    p.textContent = r.audio_source === null ? "No audio in this session." : "No speech was transcribed.";
    return;
  }
  for (const item of r.transcript) {
    if (item.kind === "pause") {
      const span = document.createElement("span");
      span.className = "pause";
      span.textContent = item.at_end
        ? `[pause ${fixed(item.duration_s)} s at the end, not counted]`
        : `[pause ${fixed(item.duration_s)} s]`;
      p.append(" ", span, " ");
      continue;
    }
    for (const part of item.parts) {
      if (part.filler) {
        const mark = document.createElement("mark");
        mark.className = "filler";
        mark.textContent = part.text;
        p.append(mark);
      } else {
        p.append(part.text);
      }
    }
    p.append(" ");
  }
  const events = r.fillers.map((f) => {
    const where = document.createElement("span");
    const mark = document.createElement("mark");
    mark.className = "filler";
    mark.textContent = f.word;
    where.append(f.before, mark, f.after);
    return { t: f.t, row: [`about ${mmss(f.t)}`, `Filler "${f.word}"`, where] };
  }).concat(r.long_pauses.map((x) => ({ t: x.start, row: [mmss(x.start), `Long pause, ${fixed(x.duration_s)} s`,
    x.after_text ? `after "${x.after_text}"` : "before the first words"] })));
  events.sort((a, b) => a.t - b.t);
  if (events.length) {
    $("events-table").append(dataTable("Fillers and long pauses. Filler times are estimated from word position.",
      ["Time", "Event", "Where in the transcript"], events.map((e) => e.row)));
  }
}

async function main() {
  const id = new URLSearchParams(location.search).get("id");
  if (!id) {
    showError("No answer was chosen. Open a report from the History page.");
    return;
  }
  let report;
  try {
    const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/report`);
    if (!res.ok) throw new Error(`the server answered ${res.status}`);
    report = await res.json();
  } catch (err) {
    showError(`Could not load the report: ${err.message}`);
    return;
  }
  renderHeader(report);
  renderImprovements(report);
  renderSummary(report);
  renderFacing(report);
  renderPosture(report);
  renderPace(report);
  renderTranscript(report);
  $("report").hidden = false;
}

main();
