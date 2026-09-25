// Hand-drawn SVG charts for the report and history pages. No chart library, nothing is
// loaded from the network.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg"; // the SVG namespace name, not a request

function svgEl(name, attrs = {}, text = undefined) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

// Axis maximum and tick step for values that should show from zero.
function yScale(values, floor) {
  const top = Math.max(floor, ...values.filter((v) => v !== null && v !== undefined));
  const step = top <= 10 ? 2 : top <= 50 ? 10 : top <= 100 ? 20 : top <= 250 ? 50 : 100;
  return { max: Math.ceil(top / step) * step, step };
}

function guideText(g) {
  if (g.low !== null && g.high !== null) return `${g.low} to ${g.high}`;
  return g.low !== null ? `at least ${g.low}` : `at most ${g.high}`;
}

/**
 * Draws a chart and returns the <svg> element.
 *   label   aria-label for the whole chart
 *   kind    "step": one level per time window, points {x0, x1, y}
 *           "line": one point per item, points {x, y}
 *   x       {min, max, title, ticks: [{value, text}]}
 *   y       {max, step, title}
 *   series  [{name, cls, marker: "circle" | "square", points}]. A null y leaves a gap.
 *   guide   {low, high, label} guideline, or a list of them, drawn as dashed lines
 *           with a band between low and high. With more than one, each is named by label.
 */
function drawChart(opts) {
  const W = 640, H = 250, M = { left: 56, right: 20, top: 22, bottom: 46 };
  const pw = W - M.left - M.right, ph = H - M.top - M.bottom;
  const xs = (x) => M.left + ((x - opts.x.min) / ((opts.x.max - opts.x.min) || 1)) * pw;
  const ys = (y) => M.top + ph - (Math.min(y, opts.y.max) / opts.y.max) * ph;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label, class: "chart" });

  for (let v = 0; v <= opts.y.max + 1e-9; v += opts.y.step) {
    svg.append(svgEl("line", { x1: M.left, x2: M.left + pw, y1: ys(v), y2: ys(v), class: "grid" }));
    svg.append(svgEl("text", { x: M.left - 8, y: ys(v) + 4, class: "tick", "text-anchor": "end" }, String(v)));
  }
  const guides = [].concat(opts.guide || []);
  for (const g of guides) {
    if (g.low !== null && g.high !== null) {
      svg.append(svgEl("rect", { x: M.left, y: ys(g.high), width: pw, height: ys(g.low) - ys(g.high),
        class: "guide-band" }));
    }
    for (const v of [g.low, g.high]) {
      if (v !== null) svg.append(svgEl("line", { x1: M.left, x2: M.left + pw, y1: ys(v), y2: ys(v), class: "guide-line" }));
    }
    const top = g.high !== null ? g.high : g.low;
    const name = guides.length > 1 ? `${g.label}: ` : "";
    svg.append(svgEl("text", { x: M.left + pw - 4, y: ys(top) - 5, class: "guide-text", "text-anchor": "end" },
      `${name}author's default ${guideText(g)}`));
  }
  for (const t of opts.x.ticks) {
    svg.append(svgEl("text", { x: xs(t.value), y: M.top + ph + 18, class: "tick", "text-anchor": "middle" }, t.text));
  }
  svg.append(svgEl("line", { x1: M.left, x2: M.left, y1: M.top, y2: M.top + ph, class: "axis" }));
  svg.append(svgEl("line", { x1: M.left, x2: M.left + pw, y1: M.top + ph, y2: M.top + ph, class: "axis" }));
  svg.append(svgEl("text", { x: M.left + pw / 2, y: H - 6, class: "axis-title", "text-anchor": "middle" }, opts.x.title));
  const mid = M.top + ph / 2;
  svg.append(svgEl("text", { x: 14, y: mid, class: "axis-title", "text-anchor": "middle", transform: `rotate(-90 14 ${mid})` },
    opts.y.title));

  for (const s of opts.series) {
    const group = svgEl("g", { class: `series ${s.cls}` });
    let d = "";
    let joined = false;
    const marks = [];
    for (const p of s.points) {
      if (p.y === null || p.y === undefined) { joined = false; continue; }
      const y = ys(p.y);
      if (opts.kind === "step") {
        const x0 = xs(p.x0), x1 = xs(p.x1);
        d += `${joined ? "L" : "M"}${x0},${y}L${x1},${y}`;
        marks.push([(x0 + x1) / 2, y]);
      } else {
        const x = xs(p.x);
        d += `${joined ? "L" : "M"}${x},${y}`;
        marks.push([x, y]);
      }
      joined = true;
    }
    group.append(svgEl("path", { d }));
    for (const [cx, cy] of marks) {
      group.append(s.marker === "square"
        ? svgEl("rect", { x: cx - 4, y: cy - 4, width: 8, height: 8 })
        : svgEl("circle", { cx, cy, r: 4 }));
    }
    svg.append(group);
  }
  return svg;
}

// A legend as a list, with a small sample of each series' line and marker.
function chartLegend(series) {
  const ul = document.createElement("ul");
  ul.className = "legend";
  for (const s of series) {
    const li = document.createElement("li");
    const sample = svgEl("svg", { viewBox: "0 0 36 12", width: 36, height: 12, "aria-hidden": "true" });
    const group = svgEl("g", { class: `series ${s.cls}` });
    group.append(svgEl("path", { d: "M2,6L34,6" }));
    group.append(s.marker === "square" ? svgEl("rect", { x: 14, y: 2, width: 8, height: 8 }) : svgEl("circle", { cx: 18, cy: 6, r: 4 }));
    sample.append(group);
    li.append(sample, ` ${s.name}`);
    ul.append(li);
  }
  return ul;
}

// A visible data table: caption, column headers and rows of cell text or nodes.
function dataTable(caption, headers, rows) {
  const table = document.createElement("table");
  const cap = document.createElement("caption");
  cap.textContent = caption;
  table.append(cap);
  const head = table.createTHead().insertRow();
  for (const h of headers) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = h;
    head.append(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const cell of row) tr.insertCell().append(cell === null || cell === undefined ? "" : cell);
  }
  return table;
}

function mmss(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function fixed(v, digits = 1) {
  return v === null || v === undefined ? "not measured" : Number(v).toFixed(digits);
}
