/** Track M SVG charts — keeps the Affine dashboard visual language. */

export const esc = (s) =>
  String(s ?? "—").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const GOLD = "#f3c449";
const BONE = "#c6bda8";
const ACCENT = "#5ac8fa";
const QUAR = "rgba(229,229,229,0.30)";   // quarantined marks
const WARN = "rgba(245,230,99,0.5)";
const MONO = "IBM Plex Mono, monospace";
const TICK_FILL = "rgba(229,229,229,0.45)";
const GRID = 'stroke="rgba(255,255,255,0.03)" stroke-width="1" stroke-dasharray="2 4"';

const W = 760;
const H = 300;
const PAD_L = 56;
const PAD_R = 24;
const PAD_T = 28;
const PAD_B = 34;

export const pct = (v, d = 1) =>
  v == null || Number.isNaN(Number(v)) ? "—" : `${(Number(v) * 100).toFixed(d)}%`;

export function fmtTime(iso) {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return String(iso);
  const d = new Date(t);
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mi = String(d.getUTCMinutes()).padStart(2, "0");
  return `${mm}/${dd} ${hh}:${mi}`;
}

export function fmtClock(iso) {
  if (!iso) return "—";
  const d = new Date(Date.parse(iso));
  return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`;
}

export function fmtAgo(iso) {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  if (sec < 86400) return `${(sec / 3600).toFixed(1)}h`;
  return `${(sec / 86400).toFixed(1)}d`;
}

function frame(svg) {
  svg.setAttribute("width", String(W));
  svg.setAttribute("height", String(H));
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
}

function emptyNote(svg, msg, extra = "") {
  frame(svg);
  svg.innerHTML = `${extra}<text x="${(PAD_L + W - PAD_R) / 2}" y="${H / 2}" text-anchor="middle"
    fill="rgba(229,229,229,0.35)" font-family="${MONO}" font-size="12">${esc(msg)}</text>`;
}

function yGrid(yAt, ticks, fmt) {
  return ticks.map((v) => {
    const y = yAt(v);
    return `<g>
      <line x1="${PAD_L}" x2="${W - PAD_R}" y1="${y}" y2="${y}" ${GRID}/>
      <text x="${PAD_L - 10}" y="${y + 3}" fill="${TICK_FILL}"
        font-family="${MONO}" font-size="10" text-anchor="end">${fmt(v)}</text>
    </g>`;
  }).join("");
}

function baseline() {
  const y = H - PAD_B;
  return `<line x1="${PAD_L}" x2="${W - PAD_R}" y1="${y}" y2="${y}"
    stroke="rgba(255,255,255,0.08)" stroke-width="1"/>`;
}

function refLine(y, color, label, anchor = "end") {
  const tx = anchor === "end" ? W - PAD_R : PAD_L + 4;
  return `<g>
    <line x1="${PAD_L}" x2="${W - PAD_R}" y1="${y}" y2="${y}"
      stroke="${color}" stroke-width="1" stroke-dasharray="2 5" opacity="0.7"/>
    <text x="${tx}" y="${y - 5}" fill="${color}" font-family="${MONO}"
      font-size="10" text-anchor="${anchor}">${esc(label)}</text>
  </g>`;
}

/** Path segments split on the quarantined flag so bad data reads grey. */
function segmented(pts, xAt, yAt, val, liveColor, dashed = false) {
  let out = "";
  let run = [];
  let quar = null;
  const flush = () => {
    if (run.length < 1) return;
    const dPath = run.map((p, i) =>
      `${i ? "L" : "M"} ${xAt(p).toFixed(1)} ${yAt(val(p)).toFixed(1)}`).join(" ");
    out += `<path d="${dPath}" fill="none" stroke="${quar ? QUAR : liveColor}"
      stroke-width="1.75" ${dashed ? 'stroke-dasharray="4 3"' : ""}
      ${quar ? 'opacity="0.8"' : ""}/>`;
  };
  for (const p of pts) {
    if (val(p) == null) continue;
    if (quar === null) quar = p.quarantined;
    if (p.quarantined !== quar) {
      // No bridge across the boundary: a visible gap, not a fake trend line.
      flush();
      run = [];
      quar = p.quarantined;
    }
    run.push(p);
  }
  flush();
  return out;
}

/* (a) fool rate over time: king eval batches + miner training rounds */
export function drawReignFoolRate(svg, data) {
  frame(svg);
  const evals = (data.reign_series || []).filter((p) => p.king != null);
  const miner = (data.miner_series || []).filter((p) => p.fool_local != null);
  const challs = (data.reign_series || []).filter((p) => p.challenger != null);
  if (!evals.length && !miner.length) {
    emptyNote(svg, "no eval batches yet");
    return;
  }

  const allT = [...evals, ...miner, ...challs].map((p) => Date.parse(p.t));
  const t0 = Math.min(...allT);
  const t1 = Math.max(...allT, t0 + 60000);
  const vals = [
    ...evals.map((p) => p.king),
    ...miner.map((p) => p.fool_local),
    ...challs.map((p) => p.challenger),
  ];
  const liveKing = evals.filter((p) => !p.quarantined);
  const lastKing = liveKing[liveKing.length - 1];
  const bar = lastKing ? lastKing.king + (data.margin_pp ?? 0.03) : null;
  const lo = Math.max(0, Math.min(...vals) - 0.04);
  const hi = Math.max(...vals, bar ?? 0) + 0.04;
  const span = hi - lo || 1;
  const xAt = (p) => PAD_L + ((Date.parse(p.t) - t0) / (t1 - t0)) * (W - PAD_L - PAD_R);
  const yAt = (v) => PAD_T + ((hi - v) / span) * (H - PAD_T - PAD_B);

  // Quarantine window shading.
  let quarBand = "";
  const q = data.quarantine;
  if (q) {
    const qx0 = Math.max(PAD_L, xAt({ t: q.start }));
    const qx1 = Math.min(W - PAD_R, xAt({ t: q.end }));
    if (qx1 > qx0) {
      quarBand = `<rect x="${qx0}" y="${PAD_T}" width="${qx1 - qx0}"
          height="${H - PAD_T - PAD_B}" fill="rgba(245,230,99,0.045)"/>
        <text x="${(qx0 + qx1) / 2}" y="${PAD_T + 12}" text-anchor="middle"
          fill="${WARN}" font-family="${MONO}" font-size="9">quarantined · judge no-op</text>`;
    }
  }

  const ticks = Array.from({ length: 5 }, (_, i) => lo + (span * i) / 4);
  const xTicks = Array.from({ length: 5 }, (_, i) => t0 + ((t1 - t0) * i) / 4)
    .map((t) => `<text x="${xAt({ t: new Date(t).toISOString() })}" y="${H - PAD_B + 16}"
      fill="${TICK_FILL}" font-family="${MONO}" font-size="10"
      text-anchor="middle">${fmtClock(new Date(t).toISOString())}</text>`).join("");

  const kingPath = segmented(evals, xAt, yAt, (p) => p.king, GOLD);
  const challPath = segmented(challs, xAt, yAt, (p) => p.challenger, ACCENT);
  const minerPath = segmented(miner, xAt, yAt, (p) => p.fool_local, ACCENT, true);

  const dot = (p, v, color, tip) => `
    <g class="chart-hit" data-tip="${esc(tip)}">
      <circle cx="${xAt(p)}" cy="${yAt(v)}" r="8" fill="transparent"/>
      <circle cx="${xAt(p)}" cy="${yAt(v)}" r="${p.quarantined ? 2 : 2.6}"
        fill="${p.quarantined ? QUAR : color}"/>
    </g>`;
  const dots =
    evals.map((p) => dot(p, p.king, GOLD,
      `king · batch ${p.batch} · fool ${pct(p.king)}${p.quarantined ? " · QUARANTINED" : ""} · ${fmtClock(p.t)} UTC`)).join("") +
    challs.map((p) => dot(p, p.challenger, ACCENT,
      `challenger · batch ${p.batch} · fool ${pct(p.challenger)}${p.quarantined ? " · QUARANTINED" : ""} · ${fmtClock(p.t)} UTC`)).join("") +
    miner.map((p) => dot(p, p.fool_local, ACCENT,
      `miner train · round ${p.round} · local fool ${pct(p.fool_local)}${p.quarantined ? " · QUARANTINED" : ""} · ${fmtClock(p.t)} UTC`)).join("");

  const lastCh = challs.filter((p) => !p.quarantined).pop();
  const lastMiner = miner.filter((p) => !p.quarantined).pop();
  const legend = `
    <text x="${PAD_L}" y="${PAD_T - 12}" fill="${GOLD}" font-family="${MONO}"
      font-size="10">— king ${lastKing ? pct(lastKing.king) : "—"}</text>
    <text x="${PAD_L + 130}" y="${PAD_T - 12}" fill="${ACCENT}" font-family="${MONO}"
      font-size="10">${lastCh ? `— challenger ${pct(lastCh.challenger)}`
        : `┄ miner train ${lastMiner ? pct(lastMiner.fool_local) : "—"}`}</text>
    <text x="${PAD_L + 300}" y="${PAD_T - 12}" fill="${QUAR}" font-family="${MONO}"
      font-size="10">— quarantined</text>`;

  svg.innerHTML = `${quarBand}${yGrid(yAt, ticks, (v) => pct(v, 0))}${baseline()}${xTicks}
    ${bar != null ? refLine(yAt(bar), GOLD, `+3pp bar ${pct(bar, 1)}`) : ""}
    ${kingPath}${challPath}${minerPath}${dots}${legend}
    <text x="${W - PAD_R}" y="${H - PAD_B + 28}" fill="${TICK_FILL}"
      font-family="${MONO}" font-size="9" text-anchor="end">UTC</text>`;
}

/* (b) ratchet: king fool rate vs each fresh judge at reign start */
export function drawRatchet(svg, data) {
  frame(svg);
  const entries = data.ratchet || [];
  if (!entries.length) {
    emptyNote(svg, "no ratchet points yet");
    return;
  }
  const hi = Math.max(...entries.map((e) => e.value ?? 0), 0.25) * 1.25;
  const yAt = (v) => PAD_T + ((hi - v) / hi) * (H - PAD_T - PAD_B);
  const n = entries.length;
  const slot = (W - PAD_L - PAD_R) / n;
  const barW = Math.min(slot * 0.5, 72);

  const lastLiveIdx = entries.reduce(
    (acc, e, i) => (!e.quarantined ? i : acc), -1);
  const ticks = Array.from({ length: 5 }, (_, i) => (hi * i) / 4);
  const bars = entries.map((e, i) => {
    const x = PAD_L + slot * (i + 0.5);
    const y = yAt(e.value ?? 0);
    const fill = e.quarantined ? QUAR : (i === lastLiveIdx ? GOLD : BONE);
    const tip = `reign ${e.reign} · ${esc(e.dver)} · fool vs fresh judge ${pct(e.value)} · n=${e.n}${e.quarantined ? " · QUARANTINED (judge no-op)" : ""}`;
    return `<g class="chart-hit" data-tip="${tip}">
      <rect x="${x - barW / 2}" y="${y}" width="${barW}" height="${H - PAD_B - y}"
        rx="1" fill="${fill}" opacity="${e.quarantined ? 0.7 : 0.95}"/>
      <text x="${x}" y="${y - 7}" fill="${e.quarantined ? QUAR : (i === lastLiveIdx ? GOLD : TICK_FILL)}"
        font-family="${MONO}" font-size="10" text-anchor="middle">${pct(e.value)}${e.quarantined ? " ⚠" : ""}</text>
      <text x="${x}" y="${H - PAD_B + 16}" fill="${TICK_FILL}" font-family="${MONO}"
        font-size="10" text-anchor="middle">R${e.reign} ${esc(e.dver)}</text>
    </g>`;
  }).join("");

  const note = entries.every((e) => e.quarantined)
    ? `<text x="${(PAD_L + W - PAD_R) / 2}" y="${PAD_T + 4}" text-anchor="middle"
        fill="${WARN}" font-family="${MONO}" font-size="10">only quarantined points so far — fresh measurement pending under corrected judge</text>`
    : "";

  svg.innerHTML = `${yGrid(yAt, ticks, (v) => pct(v, 0))}${baseline()}${bars}${note}`;
}

/* (c) judge held-out A/B accuracy per version */
export function drawJudgeAcc(svg, data) {
  frame(svg);
  const entries = (data.judges || []).filter((j) => j.held_acc != null);
  if (!entries.length) {
    emptyNote(svg, "no judge measurements yet");
    return;
  }
  const lo = 0.45;
  const hi = Math.max(...entries.map((j) => j.held_acc), 0.9) + 0.05;
  const span = hi - lo;
  const yAt = (v) => PAD_T + ((hi - v) / span) * (H - PAD_T - PAD_B);
  const n = entries.length;
  const slot = (W - PAD_L - PAD_R) / n;
  const xAt = (i) => PAD_L + slot * (i + 0.5);

  // Publish gate band 0.6–0.9 (from the crown-rule header).
  const band = `<rect x="${PAD_L}" y="${yAt(0.9)}" width="${W - PAD_L - PAD_R}"
    height="${yAt(0.6) - yAt(0.9)}" fill="rgba(198,189,168,0.045)"/>
    <text x="${W - PAD_R - 4}" y="${yAt(0.9) + 12}" text-anchor="end"
      fill="rgba(198,189,168,0.45)" font-family="${MONO}" font-size="9">publish gate 0.6–0.9</text>`;

  const ticks = [0.5, 0.6, 0.7, 0.8, 0.9];
  const dots = entries.map((j, i) => {
    const color = j.quarantined ? QUAR : (j.zero_shot ? TICK_FILL : BONE);
    const cur = i === n - 1 && !j.quarantined && !j.zero_shot;
    const tip = `${esc(j.version)} · held-out acc ${pct(j.held_acc)}${j.matched != null ? ` · matched ${pct(j.matched, 0)}` : ""}${j.quarantined ? " · QUARANTINED (serving no-op)" : ""} · ${fmtClock(j.at)} UTC`;
    return `<g class="chart-hit" data-tip="${tip}">
      <circle cx="${xAt(i)}" cy="${yAt(j.held_acc)}" r="9" fill="transparent"/>
      <circle cx="${xAt(i)}" cy="${yAt(j.held_acc)}" r="${cur ? 4.5 : 3}"
        fill="${cur ? GOLD : color}"/>
      <text x="${xAt(i)}" y="${yAt(j.held_acc) - 10}" fill="${cur ? GOLD : color}"
        font-family="${MONO}" font-size="10" text-anchor="middle">${pct(j.held_acc)}${j.quarantined ? " ⚠" : ""}</text>
      <text x="${xAt(i)}" y="${H - PAD_B + 16}" fill="${TICK_FILL}" font-family="${MONO}"
        font-size="10" text-anchor="middle">${esc(j.version)}</text>
    </g>`;
  }).join("");

  svg.innerHTML = `${band}${yGrid(yAt, ticks, (v) => v.toFixed(1))}${baseline()}
    ${refLine(yAt(0.5), ACCENT, "0.5 = perfect distillation")}
    ${dots}`;
}

/* (d) SWE-bench per crowned checkpoint */
export function drawSwe(svg, data) {
  frame(svg);
  const entries = data.swe || [];
  const base = data.swe_baseline ?? 13.33;
  const teach = data.swe_teacher ?? 31.33;
  const hi = 35;
  const yAt = (v) => PAD_T + ((hi - v) / hi) * (H - PAD_T - PAD_B);
  const refs = `${refLine(yAt(teach), GOLD, `teacher ${teach.toFixed(2)}%`)}
    ${refLine(yAt(base), "rgba(229,229,229,0.4)", `base ${base.toFixed(2)}%`, "start")}`;

  if (!entries.length) {
    const reign = data.current?.reign ?? 0;
    emptyNote(svg,
      `no crowned checkpoints benched yet — reign ${reign} in progress`,
      `${yGrid(yAt, [0, 10, 20, 30], (v) => `${v}%`)}${baseline()}${refs}`);
    return;
  }

  const n = entries.length;
  const slot = (W - PAD_L - PAD_R) / n;
  const barW = Math.min(slot * 0.5, 72);
  const bars = entries.map((e, i) => {
    const score = e.score ?? 0;
    const x = PAD_L + slot * (i + 0.5);
    const y = yAt(score);
    const cur = i === n - 1;
    const tip = `reign ${e.reign} · SWE-bench ${score.toFixed(1)}% · ${fmtTime(e.t)} UTC`;
    return `<g class="chart-hit" data-tip="${esc(tip)}">
      <rect x="${x - barW / 2}" y="${y}" width="${barW}" height="${H - PAD_B - y}"
        rx="1" fill="${cur ? GOLD : BONE}" opacity="${cur ? 1 : 0.92}"/>
      <text x="${x}" y="${y - 7}" fill="${cur ? GOLD : TICK_FILL}" font-family="${MONO}"
        font-size="10" text-anchor="middle">${score.toFixed(1)}</text>
      <text x="${x}" y="${H - PAD_B + 16}" fill="${TICK_FILL}" font-family="${MONO}"
        font-size="10" text-anchor="middle">R${e.reign}</text>
    </g>`;
  }).join("");

  svg.innerHTML = `${yGrid(yAt, [0, 10, 20, 30], (v) => `${v}%`)}${baseline()}${refs}${bars}`;
}

/* mechanism diagram: teacher + miner → frozen judge → crown → retrain loop */
export function drawMechDiagram(svg) {
  const box = (x, y, w, h, title, sub, color = "rgba(255,255,255,0.10)", titleColor = "#e5e5e5") => `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="rgba(255,255,255,0.02)"
      stroke="${color}" stroke-width="1"/>
    <text x="${x + w / 2}" y="${y + 24}" text-anchor="middle" fill="${titleColor}"
      font-family="${MONO}" font-size="12" letter-spacing="1.5">${esc(title)}</text>
    ${sub.map((s, i) => `<text x="${x + w / 2}" y="${y + 44 + i * 15}" text-anchor="middle"
      fill="rgba(229,229,229,0.45)" font-family="${MONO}" font-size="10">${esc(s)}</text>`).join("")}`;

  const arrow = (x1, y1, x2, y2, label, color = "rgba(229,229,229,0.35)", dashed = false) => {
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    return `
    <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}"
      stroke-width="1" ${dashed ? 'stroke-dasharray="3 4"' : ""} marker-end="url(#arr)"/>
    ${label ? `<text x="${mx}" y="${my - 7}" text-anchor="middle" fill="${color}"
      font-family="${MONO}" font-size="9.5">${esc(label)}</text>` : ""}`;
  };

  svg.innerHTML = `
    <defs>
      <marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7"
        markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L8 4 L0 8 z" fill="rgba(229,229,229,0.35)"/>
      </marker>
    </defs>

    ${box(20, 30, 190, 82, "TEACHER", ["Qwen3.8-27B", "fixed · public"], "rgba(243,196,73,0.45)", GOLD)}
    ${box(20, 180, 190, 82, "MINER (KING)", ["Qwen3.6-35B-A3B", "~3B active"], "rgba(90,200,250,0.4)", ACCENT)}

    ${box(360, 100, 210, 96, "JUDGE D", ["27B base + LoRA", "frozen for the reign", "A/B: which is teacher?"],
      "rgba(198,189,168,0.45)", BONE)}

    ${box(720, 30, 210, 82, "FOOL RATE", ["P(judge wrong)", "king score"], "rgba(243,196,73,0.45)", GOLD)}
    ${box(720, 180, 210, 82, "CROWN EVENT", ["+3pp over 400 turns", "→ retrain D from scratch", "→ publish · freeze"],
      "rgba(245,230,99,0.4)", "#f5e663")}

    ${arrow(210, 71, 360, 124, "rollout (thought, action)")}
    ${arrow(210, 221, 360, 172, "rollout (thought, action)")}
    ${arrow(570, 124, 720, 71, "guess")}
    ${arrow(570, 172, 720, 221, "dethroned?")}
    ${arrow(720, 240, 465, 240, "", "rgba(245,230,99,0.4)", true)}
    <line x1="465" y1="240" x2="465" y2="196" stroke="rgba(245,230,99,0.4)"
      stroke-width="1" stroke-dasharray="3 4" marker-end="url(#arr)"/>
    <text x="590" y="258" text-anchor="middle" fill="rgba(245,230,99,0.55)"
      font-family="${MONO}" font-size="9.5">new judge D-n+1 · trained on all eval data</text>

    <text x="480" y="288" text-anchor="middle" fill="rgba(229,229,229,0.35)"
      font-family="${MONO}" font-size="10">a king that matches the teacher pins every fresh judge at 0.5 — the only stable throne is real distillation</text>`;
}
