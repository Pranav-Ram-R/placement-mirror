// History view: every saved answer and the trend of four metrics across answers.
"use strict";

const $ = (id) => document.getElementById(id);

function showError(text) {
  $("error").textContent = text;
  $("error").hidden = false;
}

function trendChart(metric, rows, guide) {
  const box = document.createElement("div");
  const h3 = document.createElement("h3");
  h3.textContent = `${metric.label} (${metric.unit})`;
  const values = rows.map((r) => r[metric.metric]);
  const y = metric.unit === "%"
    ? { max: 100, step: 20 }
    : yScale(values.concat(guide && guide.high !== null ? [guide.high] : []), metric.metric === "wpm" ? 100 : 2);
  const n = rows.length;
  const every = Math.max(1, Math.ceil(n / 10));
  const ticks = rows.map((_, i) => ({ value: i + 1, text: `#${i + 1}` })).filter((_, i) => i % every === 0 || i === n - 1);
  const shown = values.filter((v) => v !== null).map((v) => format(metric, v));
  const change = shown.length >= 2
    ? `${shown[0]} in the first measured answer and ${shown[shown.length - 1]} in the latest`
    : shown.length === 1 ? `${shown[0]} in the only measured answer` : "not measured yet";
  const guideNote = guide ? ` Guideline: ${guide.text}. ${guide.kind}` : "";
  box.append(h3, drawChart({
    label: `Line chart of ${metric.label.toLowerCase()} (${metric.unit}) over ${answers(n)}, ${change}.` +
      `${guideNote} The table of saved answers below lists every value.`,
    kind: "line",
    x: { min: n > 1 ? 1 : 0, max: n > 1 ? n : 2, title: "Saved answer number", ticks },
    y: { ...y, title: metric.unit },
    series: [{ name: metric.label, cls: "series-a", marker: "circle",
      points: rows.map((r, i) => ({ x: i + 1, y: r[metric.metric] })) }],
    guide,
  }));
  return box;
}

function answers(n) {
  return `${n} saved answer${n === 1 ? "" : "s"}`;
}

function format(metric, v) {
  if (v === null || v === undefined) return "not measured";
  return metric.unit === "count" ? String(v) : fixed(v);
}

function reportLink(row, i) {
  const a = document.createElement("a");
  a.href = `report.html?id=${encodeURIComponent(row.session_id)}`;
  a.textContent = `Report ${i + 1}`;
  return a;
}

async function main() {
  let data;
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) throw new Error(`the server answered ${res.status}`);
    data = await res.json();
  } catch (err) {
    showError(`Could not load the history: ${err.message}`);
    return;
  }
  const rows = data.sessions.filter((s) => !s.error);
  const broken = data.sessions.filter((s) => s.error);
  $("history-summary").textContent = rows.length
    ? `${answers(rows.length)}, oldest first.${rows.length < 2 ? " Answer one more question to see a trend." : ""}`
    : "No saved answers yet. Answer a question on the Practice page and it shows here.";
  if (broken.length) showError(`${answers(broken.length)} could not be read: ${broken.map((b) => b.session_id).join(", ")}`);
  if (!rows.length) return;

  const guides = Object.fromEntries(data.guidelines.map((g) => [g.metric, g]));
  for (const metric of data.trend_metrics) $("trends").append(trendChart(metric, rows, guides[metric.metric] || null));

  const headers = ["#", "Answered", "Question", "Length"].concat(
    data.trend_metrics.map((m) => `${m.label} (${m.unit})`), ["Report"]);
  $("sessions").append(dataTable("Saved answers and the values in the charts above", headers, rows.map((r, i) => [
    String(i + 1),
    r.started.replace("T", " "),
    `${r.question.id}: ${r.question.text}${r.replay ? " (replay session)" : ""}`,
    mmss(r.duration_s),
    ...data.trend_metrics.map((m) => format(m, r[m.metric])),
    reportLink(r, i),
  ])));
}

main();
