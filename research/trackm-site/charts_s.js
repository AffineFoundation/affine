/** Track S SVG charts — equilibrium-centric views, same visual language. */

import {
  esc, pct, fmtClock, frame, emptyNote, yGrid, baseline, refLine,
  GOLD, BONE, ACCENT, TICK_FILL, MONO, W, H, PAD_L, PAD_R, PAD_T, PAD_B,
} from "./charts.js?v=3";

const X0 = PAD_L;
const XW = W - PAD_L - PAD_R;

/** Linear x over round numbers; integer tick labels. */
function roundAxis(pts) {
  const rs = pts.map((p) => p.round ?? 0);
  const r0 = Math.min(...rs);
  const r1 = Math.max(...rs, r0 + 1);
  const xAt = (r) => X0 + ((r - r0) / (r1 - r0)) * XW;
  const step = Math.max(1, Math.round((r1 - r0) / 6));
  let ticks = "";
  for (let r = r0; r <= r1; r += step) {
    ticks += `<text x="${xAt(r)}" y="${H - PAD_B + 16}" fill="${TICK_FILL}"
      font-family="${MONO}" font-size="10" text-anchor="middle">r${r}</text>`;
  }
  return { xAt, ticks };
}

function linePath(pts, xAt, yAt, val, color, dashed = false) {
  const seg = pts.filter((p) => val(p) != null);
  if (!seg.length) return "";
  const d = seg.map((p, i) =>
    `${i ? "L" : "M"} ${xAt(p.round).toFixed(1)} ${yAt(val(p)).toFixed(1)}`).join(" ");
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.75"
    ${dashed ? 'stroke-dasharray="4 3"' : ""}/>`;
}

function dot(x, y, color, tip, r = 2.6) {
  return `<g class="chart-hit" data-tip="${esc(tip)}">
    <circle cx="${x}" cy="${y}" r="8" fill="transparent"/>
    <circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>
  </g>`;
}

/* (a) fool rate per round: bold 0.5 equilibrium + 0.45–0.55 noise band */
export function drawFoolEquilibrium(svg, d) {
  const pts = (d.rounds || []).filter((p) => p.fool != null);
  if (!pts.length) { emptyNote(svg, "no scored rounds yet"); return; }
  frame(svg);

  const vals = pts.map((p) => p.fool);
  const lo = Math.min(0.40, ...vals) - 0.02;
  const hi = Math.max(0.60, ...vals) + 0.02;
  const yAt = (v) => PAD_T + ((hi - v) / (hi - lo)) * (H - PAD_T - PAD_B);
  const { xAt, ticks } = roundAxis(pts);

  const band = `<rect x="${X0}" y="${yAt(0.55)}" width="${XW}"
      height="${yAt(0.45) - yAt(0.55)}" fill="rgba(90,200,250,0.055)"/>
    <text x="${W - PAD_R - 4}" y="${yAt(0.55) + 11}" text-anchor="end"
      fill="rgba(90,200,250,0.5)" font-family="${MONO}" font-size="9">noise band 0.45–0.55</text>`;
  const eq = `<line x1="${X0}" x2="${W - PAD_R}" y1="${yAt(0.5)}" y2="${yAt(0.5)}"
      stroke="${ACCENT}" stroke-width="1.8" opacity="0.85"/>
    <text x="${X0 + 4}" y="${yAt(0.5) - 6}" fill="${ACCENT}"
      font-family="${MONO}" font-size="10">0.5 equilibrium</text>`;

  const yTicks = [];
  for (let v = Math.ceil(lo * 20) / 20; v <= hi + 1e-9; v += 0.05) yTicks.push(v);

  const dots = pts.map((p) => dot(xAt(p.round), yAt(p.fool), GOLD,
    `round ${p.round} · fool ${pct(p.fool, 2)} · best ${pct(p.fool_best, 2)} · dver ${p.dver ?? "—"} · g ${p.g ?? "—"} · ${fmtClock(p.t)} UTC`)).join("");

  const last = pts[pts.length - 1];
  svg.innerHTML = `${band}${yGrid(yAt, yTicks, (v) => v.toFixed(2))}${baseline()}${ticks}${eq}
    ${linePath(pts, xAt, yAt, (p) => p.fool, GOLD)}${dots}
    <text x="${X0}" y="${PAD_T - 12}" fill="${GOLD}" font-family="${MONO}"
      font-size="10">— fool rate · last ${pct(last.fool, 2)}</text>`;
}

/* (b) judge held-out accuracies (live / ctrl / tt) per D update, vs 0.5 */
export function drawHeldAcc(svg, d) {
  const ups = (d.d_updates || []).filter((u) =>
    u.held_live != null || u.held_ctrl != null || u.held_tt != null);
  if (!ups.length) { emptyNote(svg, "no D updates measured yet"); return; }
  frame(svg);

  const all = ups.flatMap((u) => [u.held_live, u.held_ctrl, u.held_tt])
    .filter((v) => v != null);
  const lo = Math.min(0.25, ...all) - 0.04;
  const hi = Math.max(0.65, ...all) + 0.04;
  const yAt = (v) => PAD_T + ((hi - v) / (hi - lo)) * (H - PAD_T - PAD_B);
  const slot = XW / ups.length;
  const xAt = (i) => X0 + slot * (i + 0.5);

  const series = [
    ["held_live", GOLD, "live (D vs G)"],
    ["held_ctrl", ACCENT, "ctrl (zero-adapter G)"],
    ["held_tt", BONE, "tt (teacher vs teacher)"],
  ];

  let body = "";
  for (const [key, color] of series) {
    const seg = ups.map((u, i) => ({ u, i })).filter(({ u }) => u[key] != null);
    if (!seg.length) continue;
    body += `<path d="${seg.map(({ u, i }, j) =>
      `${j ? "L" : "M"} ${xAt(i).toFixed(1)} ${yAt(u[key]).toFixed(1)}`).join(" ")}"
      fill="none" stroke="${color}" stroke-width="1.5" opacity="0.85"/>`;
    body += seg.map(({ u, i }) => dot(xAt(i), yAt(u[key]), color,
      `${u.dver ?? "—"} · ${key} ${pct(u[key], 2)} · n=${u.n ?? "—"} · train_acc ${u.train_acc != null ? pct(u.train_acc, 1) : "—"} · ${fmtClock(u.t)} UTC`)).join("");
  }

  const labels = ups.map((u, i) => `<text x="${xAt(i)}" y="${H - PAD_B + 16}"
    fill="${TICK_FILL}" font-family="${MONO}" font-size="10"
    text-anchor="middle">${esc(u.dver ?? "—")}</text>`).join("");
  const legend = series.map(([, color, name], i) =>
    `<text x="${X0 + i * 165}" y="${PAD_T - 12}" fill="${color}"
      font-family="${MONO}" font-size="10">— ${name}</text>`).join("");
  const yTicks = [0.3, 0.4, 0.5, 0.6].filter((v) => v > lo && v < hi);

  svg.innerHTML = `${yGrid(yAt, yTicks, (v) => v.toFixed(1))}${baseline()}
    ${refLine(yAt(0.5), "rgba(229,229,229,0.5)", "0.5 = nothing to learn")}
    ${labels}${body}${legend}`;
}

/* (c) typicality (typ_lp) per round */
export function drawTypicality(svg, d) {
  const pts = (d.rounds || []).filter((p) => p.typ_lp != null);
  if (!pts.length) { emptyNote(svg, "no typicality measurements yet"); return; }
  frame(svg);

  const vals = pts.map((p) => p.typ_lp);
  const span = Math.max(...vals) - Math.min(...vals) || 0.1;
  const lo = Math.min(...vals) - span * 0.25;
  const hi = Math.max(...vals) + span * 0.25;
  const yAt = (v) => PAD_T + ((hi - v) / (hi - lo)) * (H - PAD_T - PAD_B);
  const { xAt, ticks } = roundAxis(pts);

  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const dots = pts.map((p) => dot(xAt(p.round), yAt(p.typ_lp), GOLD,
    `round ${p.round} · typ_lp ${p.typ_lp.toFixed(4)} · think_len ${p.think_len ?? "—"} · ${fmtClock(p.t)} UTC`)).join("");
  const last = pts[pts.length - 1];

  svg.innerHTML = `${yGrid(yAt, yTicks, (v) => v.toFixed(2))}${baseline()}${ticks}
    ${linePath(pts, xAt, yAt, (p) => p.typ_lp, GOLD)}${dots}
    <text x="${X0}" y="${PAD_T - 12}" fill="${GOLD}" font-family="${MONO}"
      font-size="10">— typ_lp · last ${last.typ_lp.toFixed(4)}</text>`;
}

/* (d) agreement overlay: agree_tok (G vs teacher) against tt_tok (teacher vs teacher) */
export function drawAgreement(svg, d) {
  const pts = (d.rounds || []).filter((p) =>
    p.agree_tok != null || p.tt_tok != null);
  if (!pts.length) { emptyNote(svg, "no agreement measurements yet"); return; }
  frame(svg);

  const vals = pts.flatMap((p) => [p.agree_tok, p.tt_tok])
    .filter((v) => v != null);
  const lo = Math.max(0, Math.min(...vals) - 0.06);
  const hi = Math.min(1, Math.max(...vals) + 0.06);
  const yAt = (v) => PAD_T + ((hi - v) / (hi - lo)) * (H - PAD_T - PAD_B);
  const { xAt, ticks } = roundAxis(pts);

  const dots =
    pts.filter((p) => p.agree_tok != null).map((p) => dot(xAt(p.round), yAt(p.agree_tok), GOLD,
      `round ${p.round} · agree_tok (G vs T) ${pct(p.agree_tok, 1)} · ${fmtClock(p.t)} UTC`)).join("") +
    pts.filter((p) => p.tt_tok != null).map((p) => dot(xAt(p.round), yAt(p.tt_tok), BONE,
      `round ${p.round} · tt_tok (T vs T) ${pct(p.tt_tok, 1)} · ${fmtClock(p.t)} UTC`)).join("");

  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);

  svg.innerHTML = `${yGrid(yAt, yTicks, (v) => v.toFixed(2))}${baseline()}${ticks}
    ${linePath(pts, xAt, yAt, (p) => p.agree_tok, GOLD)}
    ${linePath(pts, xAt, yAt, (p) => p.tt_tok, BONE, true)}
    ${dots}
    <text x="${X0}" y="${PAD_T - 12}" fill="${GOLD}" font-family="${MONO}"
      font-size="10">— agree_tok (G vs teacher)</text>
    <text x="${X0 + 200}" y="${PAD_T - 12}" fill="${BONE}" font-family="${MONO}"
      font-size="10">┄ tt_tok (teacher vs teacher)</text>`;
}
