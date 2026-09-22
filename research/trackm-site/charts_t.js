/** Track T SVG charts — SWE-over-wall-clock headline + fool/judge view. */

import {
  esc, pct, fmtClock, frame, emptyNote, yGrid, baseline, refLine,
  GOLD, BONE, ACCENT, TICK_FILL, MONO, W, H, PAD_L, PAD_R, PAD_T, PAD_B,
} from "./charts.js?v=3";

const X0 = PAD_L;
const XW = W - PAD_L - PAD_R;

function timeAxis(ts) {
  const t0 = Math.min(...ts);
  const t1 = Math.max(...ts, t0 + 60_000);
  const xAt = (t) => X0 + ((t - t0) / (t1 - t0)) * XW;
  const ticks = Array.from({ length: 5 }, (_, i) => t0 + ((t1 - t0) * i) / 4)
    .map((t) => `<text x="${xAt(t)}" y="${H - PAD_B + 16}" fill="${TICK_FILL}"
      font-family="${MONO}" font-size="10" text-anchor="middle">${fmtClock(new Date(t).toISOString())}</text>`)
    .join("") + `<text x="${W - PAD_R}" y="${H - PAD_B + 28}" fill="${TICK_FILL}"
      font-family="${MONO}" font-size="9" text-anchor="end">UTC</text>`;
  return { xAt, ticks };
}

/* headline: SWE score over wall-clock */
export function drawSweHeadline(svg, d) {
  frame(svg);
  const proxies = (d.bench_proxy || []).filter((p) => p.frac != null);
  const fulls = (d.bench_full || []).filter((p) => p.score != null);
  const teach = d.swe_teacher ?? 0.3133;
  const sbase = d.swe_student_baseline;

  const hi = Math.max(0.40, teach + 0.06,
    ...proxies.map((p) => p.frac + 0.04),
    ...fulls.map((p) => p.score + 0.06));
  const yAt = (v) => PAD_T + ((hi - v) / hi) * (H - PAD_T - PAD_B);

  const refs = `${refLine(yAt(teach), GOLD, `teacher ${teach.toFixed(4)}`)}
    ${sbase != null
      ? refLine(yAt(sbase), "rgba(229,229,229,0.45)", `student raw ${sbase.toFixed(4)}`, "start")
      : `<g><line x1="${X0}" x2="${W - PAD_R}" y1="${H - PAD_B - 14}" y2="${H - PAD_B - 14}"
          stroke="rgba(229,229,229,0.25)" stroke-width="1" stroke-dasharray="2 5"/>
        <text x="${X0 + 4}" y="${H - PAD_B - 19}" fill="rgba(229,229,229,0.35)"
          font-family="${MONO}" font-size="10">student raw baseline — awaiting first full bench (position unknown)</text></g>`}`;
  const grid = yGrid(yAt, [0, 0.1, 0.2, 0.3, 0.4].filter((v) => v <= hi),
    (v) => v.toFixed(1));

  if (!proxies.length && !fulls.length) {
    emptyNote(svg, "no bench results yet — headline populates at the first proxy-16 panel",
      `${grid}${baseline()}${refs}`);
    return;
  }

  const { xAt, ticks } = timeAxis([...proxies, ...fulls].map((p) => Date.parse(p.t)));

  const proxyDots = proxies.map((p) => `
    <g class="chart-hit" data-tip="${esc(`proxy-16 · round ${p.round ?? "—"} · ${p.num}/${p.den} (${pct(p.frac, 1)}) · ${p.gpu_hours ?? "—"} gpu-h · ${fmtClock(p.t)} UTC`)}">
      <circle cx="${xAt(Date.parse(p.t))}" cy="${yAt(p.frac)}" r="8" fill="transparent"/>
      <circle cx="${xAt(Date.parse(p.t))}" cy="${yAt(p.frac)}" r="2.6"
        fill="${BONE}" opacity="0.65"/>
    </g>`).join("");

  const fullMarks = fulls.map((p) => {
    const x = xAt(Date.parse(p.t));
    const n = p.panel || 150;
    const se = Math.sqrt(Math.max(p.score * (1 - p.score), 1e-6) / n);
    const ciLo = Math.max(0, p.score - 1.96 * se);
    const ciHi = Math.min(1, p.score + 1.96 * se);
    const color = p.baseline ? "rgba(229,229,229,0.75)" : GOLD;
    const tip = `full panel · ${p.ckpt ?? "—"} · ${p.score.toFixed(4)} (${p.resolved ?? "—"}/${n})`
      + ` · p vs baseline ${p.p_vs_baseline ?? "—"} · ${p.gpu_hours ?? "—"} gpu-h · ${fmtClock(p.t)} UTC`;
    return `<g class="chart-hit" data-tip="${esc(tip)}">
      <circle cx="${x}" cy="${yAt(p.score)}" r="9" fill="transparent"/>
      <line x1="${x}" x2="${x}" y1="${yAt(ciHi)}" y2="${yAt(ciLo)}"
        stroke="${color}" stroke-width="1.4" opacity="0.8"/>
      <line x1="${x - 4}" x2="${x + 4}" y1="${yAt(ciHi)}" y2="${yAt(ciHi)}" stroke="${color}" stroke-width="1.2"/>
      <line x1="${x - 4}" x2="${x + 4}" y1="${yAt(ciLo)}" y2="${yAt(ciLo)}" stroke="${color}" stroke-width="1.2"/>
      <circle cx="${x}" cy="${yAt(p.score)}" r="4.2" fill="${color}"/>
      <text x="${x}" y="${yAt(p.score) - 14}" fill="${color}" font-family="${MONO}"
        font-size="10" text-anchor="middle">${p.score.toFixed(3)}${p.baseline ? " (raw)" : ""}</text>
    </g>`;
  }).join("");

  const legend = `
    <text x="${X0}" y="${PAD_T - 12}" fill="${BONE}" font-family="${MONO}"
      font-size="10" opacity="0.8">· proxy-16</text>
    <text x="${X0 + 90}" y="${PAD_T - 12}" fill="${GOLD}" font-family="${MONO}"
      font-size="10">● full panel ±95% CI</text>`;

  svg.innerHTML = `${grid}${baseline()}${ticks}${refs}${proxyDots}${fullMarks}${legend}`;
}

/* fool rate per round + judge held-out accuracy at each retrain */
export function drawFoolJudge(svg, d) {
  const rounds = (d.rounds || []).filter((p) => p.fool != null);
  const judges = (d.judges || []).filter((j) => j.held_acc != null);
  if (!rounds.length && !judges.length) {
    emptyNote(svg, "no rounds logged yet");
    return;
  }
  frame(svg);

  const { xAt, ticks } = timeAxis([...rounds, ...judges].map((p) => Date.parse(p.t)));
  const vals = [...rounds.map((p) => p.fool), ...judges.map((j) => j.held_acc)];
  const lo = Math.max(0, Math.min(0.4, ...vals) - 0.04);
  const hi = Math.min(1, Math.max(0.6, ...vals) + 0.04);
  const yAt = (v) => PAD_T + ((hi - v) / (hi - lo)) * (H - PAD_T - PAD_B);

  const foolPath = rounds.length ? `<path d="${rounds.map((p, i) =>
    `${i ? "L" : "M"} ${xAt(Date.parse(p.t)).toFixed(1)} ${yAt(p.fool).toFixed(1)}`).join(" ")}"
    fill="none" stroke="${GOLD}" stroke-width="1.75"/>` : "";

  const foolDots = rounds.map((p) => `
    <g class="chart-hit" data-tip="${esc(`round ${p.round} · fool ${pct(p.fool, 2)} · dver ${p.dver ?? "—"} · ${fmtClock(p.t)} UTC`)}">
      <circle cx="${xAt(Date.parse(p.t))}" cy="${yAt(p.fool)}" r="8" fill="transparent"/>
      <circle cx="${xAt(Date.parse(p.t))}" cy="${yAt(p.fool)}" r="2.6" fill="${GOLD}"/>
    </g>`).join("");

  const judgeDots = judges.map((j) => `
    <g class="chart-hit" data-tip="${esc(`judge retrain · ${j.dver ?? "—"} · held_acc ${pct(j.held_acc, 2)} · ${fmtClock(j.t)} UTC`)}">
      <circle cx="${xAt(Date.parse(j.t))}" cy="${yAt(j.held_acc)}" r="9" fill="transparent"/>
      <circle cx="${xAt(Date.parse(j.t))}" cy="${yAt(j.held_acc)}" r="3.6" fill="${BONE}"/>
      <text x="${xAt(Date.parse(j.t))}" y="${yAt(j.held_acc) - 10}" fill="${BONE}"
        font-family="${MONO}" font-size="10" text-anchor="middle">${esc(j.dver ?? "")}</text>
    </g>`).join("");

  const yTicks = [0.4, 0.5, 0.6, 0.7, 0.8].filter((v) => v > lo && v < hi);
  const legend = `
    <text x="${X0}" y="${PAD_T - 12}" fill="${GOLD}" font-family="${MONO}"
      font-size="10">— fool rate</text>
    <text x="${X0 + 110}" y="${PAD_T - 12}" fill="${BONE}" font-family="${MONO}"
      font-size="10">● judge held-out acc</text>`;

  svg.innerHTML = `${yGrid(yAt, yTicks, (v) => v.toFixed(1))}${baseline()}${ticks}
    ${refLine(yAt(0.5), ACCENT, "0.5 = indistinguishable")}
    ${foolPath}${foolDots}${judgeDots}${legend}`;
}
