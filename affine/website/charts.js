/** SVG hero / detail charts — keeps the existing Affine visual language. */

export const esc = (s) =>
  String(s ?? "—").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const short = (s, n = 18) => {
  const v = String(s ?? "");
  return v.length > n ? `${v.slice(0, n)}…` : v || "—";
};

/* ---------- model display names ---------- */

const ROMAN_TABLE = [
  [1000, "M"], [900, "CM"], [500, "D"], [400, "CD"],
  [100, "C"], [90, "XC"], [50, "L"], [40, "XL"],
  [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"],
];

export function romanNumeral(n) {
  let v = Math.floor(Number(n));
  if (!Number.isFinite(v) || v < 1) return String(n);
  let out = "";
  for (const [val, sym] of ROMAN_TABLE) {
    while (v >= val) {
      out += sym;
      v -= val;
    }
  }
  return out;
}

/** Kings are numbered from reign 0 (genesis) → Affine-I, reign 1 → Affine-II… */
export const kingName = (reignNumber) =>
  `Affine-${romanNumeral(Number(reignNumber) + 1)}`;

// Reign lookup: repo → reign_number (and hotkey → reign_number as a fallback
// for rows with no repo). Rebuilt from every snapshot by setReignLookup.
const reignByRepo = new Map();
const reignByHotkey = new Map();

export function setReignLookup(members, genesisRepo) {
  reignByRepo.clear();
  reignByHotkey.clear();
  if (genesisRepo) reignByRepo.set(genesisRepo, 0);
  for (const m of members || []) {
    if (m?.reign_number == null) continue;
    if (m.repo) reignByRepo.set(m.repo, m.reign_number);
    if (m.hotkey) reignByHotkey.set(m.hotkey, m.reign_number);
  }
}

/**
 * Kings display as Affine-<roman>; everything else as Affine-<hotkey[0:5]>.
 * Repo is the primary key (a king hotkey can later submit a different model);
 * hotkey lookup only applies when the row has no repo at all.
 */
export function modelDisplayName(repo, hotkey, reignNumber) {
  let rn = reignNumber;
  if (rn == null && repo && reignByRepo.has(repo)) rn = reignByRepo.get(repo);
  if (rn == null && !repo && hotkey && reignByHotkey.has(hotkey)) {
    rn = reignByHotkey.get(hotkey);
  }
  if (rn != null) return kingName(rn);
  if (hotkey) return `Affine-${String(hotkey).slice(0, 5)}`;
  return repo ? String(repo) : "—";
}

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

/** Compact relative age for live chips (e.g. "12m", "3h", "2d"). */
export function fmtAge(iso) {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h`;
  return `${Math.floor(sec / 86400)}d`;
}

export function fmtTao(v, digits = 4) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${Number(v).toFixed(digits)} τ`;
}

export function fmtUsd(v, digits = 0) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (digits === 0 && Math.abs(n) >= 100) {
    return `$${Math.round(n).toLocaleString("en-US")}`;
  }
  return `$${n.toFixed(digits === 0 ? 2 : digits)}`;
}

export function fmtDuration(sec) {
  if (sec == null || Number.isNaN(Number(sec))) return "—";
  const s = Math.max(0, Math.round(Number(sec)));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `${h}h ${rm}m` : `${h}h`;
}

export function fmtAlpha(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (n >= 100) return `${n.toFixed(0)} α`;
  if (n >= 10) return `${n.toFixed(1)} α`;
  return `${n.toFixed(2)} α`;
}

export function fmtZ(z) {
  if (z == null || Number.isNaN(Number(z))) return "—";
  const n = Number(z);
  return `${n > 0 ? "+" : ""}${n.toFixed(2)}`;
}

export function reignMembers(d) {
  const fromReign = d?.reign?.members;
  if (Array.isArray(fromReign) && fromReign.length) return fromReign;
  const chain = d?.reign_chain || [];
  const king = d?.king;
  if (!king && !chain.length) return [];
  const members = [];
  if (king) {
    members.push({
      reign_number: king.reign_number, repo: king.repo, revision: king.revision,
      hotkey: king.hotkey, crowned_at: king.crowned_at, score: king.score,
      current: true, earning: true,
    });
  }
  for (const hk of chain) {
    if (king && hk === king.hotkey) continue;
    members.push({
      hotkey: hk, repo: "", revision: "", current: false, earning: true,
    });
  }
  const weight_bps = Math.floor(10000 / Math.max(members.length, 1));
  return members.map((m) => ({ ...m, weight_bps, earning: m.earning !== false }));
}

export function duelPoints(history) {
  return (history || [])
    .filter((r) => r.event !== "failed")
    .filter((r) => r.z != null || r.event === "crowned")
    .slice()
    .reverse();
}

/** Tooltip name: display name plus the real repo string for hover discovery. */
function duelTipName(p) {
  const name = modelDisplayName(p?.repo, p?.hotkey, p?.reign_number);
  return p?.repo && p.repo !== name ? `${name} (${p.repo})` : name;
}

/** Marks with a challenge id navigate to the duel page on click. */
function duelCidAttr(p) {
  return p?.challenge_id ? ` data-cid="${esc(p.challenge_id)}"` : "";
}

/**
 * X axis: crownings are named moments; every other duel is a bare tick.
 * Reign numerals are dropped when two crowns land closer than the label is
 * wide — back-to-back crownings otherwise smear into one glyph run.
 */
const CROWN_LABEL_GAP = 18;

function duelAxisMarks(points, xAt, yBase) {
  let lastLabelX = -Infinity;
  return points.map((p, i) => {
    const x = xAt(i);
    if (p?.event !== "crowned") {
      return `<line x1="${x}" x2="${x}" y1="${yBase}" y2="${yBase + 4}"
        stroke="rgba(229,229,229,0.22)" stroke-width="1"/>`;
    }
    const label = p.reign_number != null ? romanNumeral(p.reign_number + 1) : "#?";
    const show = x - lastLabelX >= CROWN_LABEL_GAP;
    if (show) lastLabelX = x;
    return `<line x1="${x}" x2="${x}" y1="${yBase}" y2="${yBase + 6}"
        stroke="${GOLD}" stroke-width="1"/>
      ${show ? `<text x="${x}" y="${yBase + 18}" text-anchor="middle" fill="${GOLD}"
        font-family="${MONO}" font-size="10">${esc(label)}</text>` : ""}`;
  }).join("");
}

/* ---------- hero charts (different from affine.io env bars) ---------- */

export function chartWidth() {
  return Math.max(window.innerWidth || 960, 320);
}

export function fmtScore(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  const abs = Math.abs(n);
  if (abs >= 10) return n.toFixed(1);
  if (abs >= 1) return n.toFixed(2);
  return n.toFixed(3);
}

export function drawDuelZ(svg, history, { width: widthOpt, height: heightOpt } = {}) {
  const points = duelPoints(history);
  const width = Math.max(widthOpt || chartWidth(), 280);
  const height = heightOpt || 320;
  // Same gutters as drawDuelScores so the stacked heroes align.
  const padL = 56;
  const padR = 24;
  const padT = 28;
  const padB = 36;
  const n = Math.max(points.length, 1);
  const slot = (width - padL - padR) / n;
  // Scale bars down with the slot so dense histories keep a visible gap
  // (a fixed 10px floor fused bars into a block past ~60 duels).
  const barW = Math.max(1.5, Math.min(slot * 0.6, 64, slot - 1.5));

  const zs = points.map((p) =>
    p.event === "crowned" ? Math.max(Number(p.z) || 0, 3) : Number(p.z) || 0);
  let min = Math.min(-1, ...zs, 0);
  let max = Math.max(3.5, ...zs, 1);
  max = Math.max(max, 3.2);
  const yAt = (v) => padT + ((max - v) / (max - min || 1)) * (height - padT - padB);
  const xAt = (i) => padL + slot * (i + 0.5);

  const ticks = [];
  const step = max - min > 8 ? 2 : 1;
  for (let v = Math.ceil(min); v <= Math.floor(max); v += step) ticks.push(v);
  if (!ticks.includes(0)) ticks.push(0);
  ticks.sort((a, b) => a - b);

  const gold = "#f3c449";
  const bar = "#c6bda8";
  const mono = "IBM Plex Mono, monospace";

  // Crown threshold in z-units = k_sigma (contract: 2.0 since 2026-08-11).
  const kCrown = 2;
  const grid = ticks.filter((v) => v !== kCrown).map((v) => {
    const y = yAt(v);
    return `<g>
      <line x1="${padL}" x2="${width - padR}" y1="${y}" y2="${y}"
        stroke="${v === 0 ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)"}"
        stroke-width="1" ${v === 0 ? "" : 'stroke-dasharray="2 4"'}/>
      <text x="${padL - 10}" y="${y + 3}" fill="rgba(229,229,229,0.45)"
        font-family="${mono}" font-size="10" text-anchor="end">${v.toFixed(0)}</text>
    </g>`;
  }).join("");

  const yCrown = yAt(kCrown);
  const crownLine = `<g>
    <line x1="${padL}" x2="${width - padR}" y1="${yCrown}" y2="${yCrown}"
      stroke="${gold}" stroke-width="1" stroke-dasharray="2 5" opacity="0.7"/>
    <text x="${padL - 10}" y="${yCrown + 3}" fill="${gold}" font-family="${mono}"
      font-size="10" text-anchor="end">${kCrown}σ</text>
  </g>`;

  const columns = points.map((p, i) => {
    const z = zs[i];
    const x = xAt(i);
    const y0 = yAt(0);
    const y1 = yAt(z);
    const top = Math.min(y0, y1);
    const h = Math.max(Math.abs(y0 - y1), 2);
    const crowned = p.event === "crowned";
    const fill = crowned ? gold : (z >= 0 ? bar : "rgba(255,71,71,0.55)");
    const zLabel = fmtZ(p.event === "crowned" && p.z == null ? kCrown : p.z);
    const tip = `${duelTipName(p)} · ${p.event} · z=${zLabel} · ${fmtTime(p.at)}`;
    return `<g class="duel-hit" data-tip="${esc(tip)}"${duelCidAttr(p)}>
      <rect x="${x - barW / 2}" y="${top}" width="${barW}" height="${h}" rx="1" fill="${fill}"
        opacity="${crowned ? 1 : 0.92}"/>
    </g>`;
  }).join("");
  const axis = duelAxisMarks(points, xAt, height - padB);

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${grid}${crownLine}${columns}${axis}`;
}

export function drawDuelScores(svg, history,
                               { width: widthOpt, height: heightOpt } = {}) {
  // Best absolute Reason per duel — max(king, challenger), reg-price style.
  const points = (history || [])
    .filter((r) => r.event !== "failed")
    .filter((r) =>
      r.challenger || r.king || r.score != null || r.score_king != null
      || r.event === "crowned" || r.z != null)
    .slice()
    .reverse();
  const width = Math.max(widthOpt || chartWidth(), 280);
  const height = heightOpt || 320;
  const padL = 56;
  const padR = 24;
  const padT = 28;
  const padB = 36;
  const n = Math.max(points.length, 1);
  const slot = (width - padL - padR) / n;
  const xAt = (i) => padL + slot * (i + 0.5);
  const gold = "#f3c449";
  const mono = "IBM Plex Mono, monospace";

  const scoreOf = (p, key) => {
    const v = p[key];
    return v != null && Number.isFinite(Number(v)) ? Number(v) : null;
  };
  // Per-side mean Reason, same accessor as the sides pane: v3 rows publish
  // `reason`, pre-fork rows carried the identical quantity as `mean_lambda2`.
  // The top-level `score`/`score_king` fields are a last resort only — on
  // pre-fork rows they hold the retired S* mix, not Reason.
  const bestScore = (p) => {
    const vals = [
      reasonOf(p, "challenger") ?? scoreOf(p, "score"),
      reasonOf(p, "king") ?? scoreOf(p, "score_king"),
    ].filter((v) => v != null);
    return vals.length ? Math.max(...vals) : null;
  };

  const series = points.map((p, i) => ({ p, i, v: bestScore(p) }))
    .filter((pt) => pt.v != null);
  const scores = series.map((pt) => pt.v);

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (!scores.length) {
    svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle"
      fill="rgba(229,229,229,0.35)" font-family="${mono}" font-size="12">no Reason recorded yet</text>`;
    return;
  }

  let lo = Math.min(...scores);
  let hi = Math.max(...scores);
  if (hi === lo) {
    lo -= Math.abs(lo) * 0.2 || 0.02;
    hi += Math.abs(hi) * 0.2 || 0.02;
  } else {
    const pad = (hi - lo) * 0.18;
    lo -= pad;
    hi += pad;
  }
  const span = hi - lo || 1;
  const yAt = (v) => padT + ((hi - v) / span) * (height - padT - padB);
  const yFloor = height - padB;

  const ticks = Array.from({ length: 5 }, (_, i) => lo + (span * i) / 4);
  const grid = ticks.map((v) => {
    const y = yAt(v);
    return `<g>
      <line x1="${padL}" x2="${width - padR}" y1="${y}" y2="${y}"
        stroke="rgba(255,255,255,0.04)" stroke-dasharray="2 4"/>
      <text x="${padL - 10}" y="${y + 3}" text-anchor="end" fill="rgba(229,229,229,0.45)"
        font-family="${mono}" font-size="10">${fmtScore(v)}</text>
    </g>`;
  }).join("");

  const line = series.map((pt, k) =>
    `${k ? "L" : "M"} ${xAt(pt.i)} ${yAt(pt.v)}`).join(" ");
  const area = series.length >= 2
    ? `${line} L ${xAt(series[series.length - 1].i)} ${yFloor} L ${xAt(series[0].i)} ${yFloor} Z`
    : "";

  const last = series[series.length - 1];
  const tip = last
    ? `<circle cx="${xAt(last.i)}" cy="${yAt(last.v)}" r="3.5" fill="${gold}"/>
       <text x="${xAt(last.i) - 8}" y="${yAt(last.v) - 10}" text-anchor="end"
         fill="${gold}" font-family="${mono}" font-size="11">${esc(fmtScore(last.v))}</text>`
    : "";

  const labels = points.map((p, i) => {
    const crowned = p.event === "crowned";
    const v = bestScore(p);
    if (v == null) return "";
    const x = xAt(i);
    const chall = scoreOf(p, "score");
    const king = scoreOf(p, "score_king");
    const who = (chall != null && (king == null || chall >= king)) ? "challenger" : "king";
    const tip = `${duelTipName(p)} · best=${fmtScore(v)} (${who}) · ${fmtTime(p.at)}`;
    return `<g class="duel-hit" data-tip="${esc(tip)}"${duelCidAttr(p)}>
      <circle cx="${x}" cy="${yAt(v)}" r="9" fill="transparent"/>
      <circle cx="${x}" cy="${yAt(v)}" r="${crowned ? 4.5 : 2.5}"
        fill="${gold}" opacity="${crowned ? 1 : 0.85}"/>
    </g>`;
  }).join("");
  const axis = duelAxisMarks(points, xAt, height - padB);

  svg.innerHTML = `${grid}
    ${area ? `<path d="${area}" fill="rgba(243,196,73,0.08)"/>` : ""}
    <path d="${line}" fill="none" stroke="${gold}" stroke-width="1.75"/>
    ${labels}${tip}${axis}`;
}

/* ---------- per-duel gate grid ---------- */

const GOLD = "#f3c449";
const BONE = "#c6bda8";
const MONO = "IBM Plex Mono, monospace";

const num = (v) => {
  const n = Number(v);
  return v == null || !Number.isFinite(n) ? null : n;
};
const sideVal = (p, which, key) => num((p?.[which] || {})[key]);
const fmtPct = (v) => (v == null ? "—" : `${(Number(v) * 100).toFixed(0)}%`);
const fmtRatio = (v) => (v == null ? "—" : Number(v).toFixed(2));
const fmtInt = (v) => (v == null ? "—" : String(Math.round(Number(v))));

/**
 * Scored duels plus crownings, oldest → newest (history arrives newest first).
 * Crownings carry no gate payload but stay on the axis so every pane lines up
 * with the same moments as the hero charts.
 */
export function gatePoints(history) {
  return (history || [])
    .filter((r) => r.challenger || r.king || r.event === "crowned")
    .slice()
    .reverse();
}

/**
 * Everything the duel measures per side, one pane each. Since the Reason v3
 * fork only Reason is scored — every other pane is TELEMETRY: measured and
 * published in the verdict so the axis can be studied, never affecting the
 * score. The dashed threshold lines on telemetry panes are the retired S* v2
 * gate values, kept as visual context for pre-fork history. `keep: false`
 * lines only enter the y-domain when the data already reaches them.
 */
/** Per-side Reason: v3 verdicts publish `reason`; pre-fork rows carried the
 * identical quantity as `mean_lambda2`. */
const reasonOf = (p, side) =>
  sideVal(p, side, "reason") ?? sideVal(p, side, "mean_lambda2");

/**
 * Mean L1lift per side (telemetry). v3 verdicts publish the raw mean
 * directly; pre-fork verdicts published mean Λ2 + the mean mix, so the
 * clipped term is recovered exactly as (mix − Λ2) / w.
 */
const l1Term = (p, side) => {
  const direct = sideVal(p, side, "mean_l1lift");
  if (direct != null) return direct;
  const mix = sideVal(p, side, "mean_mix");
  const l2 = sideVal(p, side, "mean_lambda2");
  if (mix == null || l2 == null) return null;
  const w = Number(p.gates?.l1_weight) || 1;
  return (mix - l2) / w;
};

const TELEMETRY_NOTE = `<p><em>Telemetry: measured and published in every
  verdict for study — it does not affect the score or the crown. The dashed
  threshold is the retired S* v2 gate value, kept as context for pre-fork
  history.</em></p>`;

export const GATE_METRICS = [
  {
    id: "reason",
    title: "Reason (sides)",
    caption: "per-side mean lpC(y_C|z_A) − lpC(y_C|∅) — the score",
    detail: `<p><code>Reason = lpC(y_C|z_A) − lpC(y_C|∅)</code>: how much the
      miner's thought helps <em>the teacher</em> predict its own action. The
      teacher is the anchor — the miner is never judged by another model's
      opinion of its prose, only by whether its reasoning measurably transfers
      into the teacher.</p>
      <p>This is the whole score. Positive means the thought carried real
      information about what to do next; near zero means it was decoration.
      The hero chart shows the best side per duel; this pane shows both
      sides.</p>`,
    fmt: fmtScore,
    lines: [{ label: "", at: () => 0, faint: true }],
    series: [
      { label: "challenger", color: GOLD, get: (p) => reasonOf(p, "challenger") },
      { label: "king", color: BONE, get: (p) => reasonOf(p, "king") },
    ],
  },
  {
    id: "l1lift",
    title: "L1lift",
    caption: "telemetry · miner-side lift lpA(y_C|z_A) − lpA(y_C|∅)",
    detail: `<p><code>L1lift = lpA(y_C|z_A) − lpA(y_C|∅)</code>: how much the
      miner's thought helps <em>the miner itself</em> predict the teacher's
      action. Positive is the natural signature of a faithful distill — the
      model genuinely anticipates what the teacher will do.</p>
      <p>Under S* v2 this term was part of the ranked mix; since Reason v3 it
      is recorded only. v3 verdicts publish the raw mean directly; pre-fork
      points are recovered as <code>(S − Λ2) / w</code>.</p>${TELEMETRY_NOTE}`,
    fmt: fmtScore,
    lines: [{ label: "", at: () => 0, faint: true }],
    series: [
      { label: "challenger", color: GOLD, get: (p) => l1Term(p, "challenger") },
      { label: "king", color: BONE, get: (p) => l1Term(p, "king") },
    ],
  },
  {
    id: "eta",
    title: "η sufficiency",
    caption: "telemetry · Λ2(z_A) / Λ2(z_C) — fraction of teacher thinking replaced",
    detail: `<p><code>η = Λ2(z_A) / Λ2(z_C)</code> =
      <code>Reason / (lpC(y_C|z_C) − lpC(y_C|∅))</code>: how much of the
      teacher's own thinking the miner's thought replaces on the same
      reference action. Climbing η across reigns is the capability-slope
      signature; kings that crown while η sits flat are budget burn.</p>
      <p>Denominator is taken from the teacher reference
      (<code>lpC_yc_zc</code> / <code>lp_own</code>) already scored when refs
      are built — no extra GPU echo. Undefined on pairs where the teacher's
      own lift is ~0; the published mean skips those.</p>${TELEMETRY_NOTE}`,
    fmt: fmtScore,
    lines: [
      { label: "1", at: () => 1.0, faint: true },
      { label: "", at: () => 0, faint: true },
    ],
    series: [
      { label: "challenger", color: GOLD, get: (p) => sideVal(p, "challenger", "mean_eta") },
      { label: "king", color: BONE, get: (p) => sideVal(p, "king", "mean_eta") },
    ],
  },
  {
    id: "gate-pass",
    title: "causality",
    caption: "telemetry · pairs whose thought earns ≥ τ lift without leaking",
    detail: `<p>Share of the slice's pairs where the miner's thought
      <code>z_A</code> contains no fuzzy copy of its own action and injecting
      it raises the miner's own logprob of that action by at least
      <code>τ = 0.02</code>: <code>lpA(y_A|z_A) − lpA(y_A|∅) ≥ τ</code>.</p>
      <p>Formerly gate 1 of S* v2 (invalid below γ = 0.30). A silent or
      payload-stuffing miner shows up here first — under Reason v3 such a
      miner simply scores ≈ 0 and loses the duel on its own.</p>${TELEMETRY_NOTE}`,
    fmt: fmtPct,
    domain: [0, 1],
    lines: [{ label: "γ", at: (g) => g.gamma ?? 0.3, faint: true }],
    series: [
      { label: "challenger", color: GOLD, get: (p) => sideVal(p, "challenger", "gate_pass_rate") },
      { label: "king", color: BONE, get: (p) => sideVal(p, "king", "gate_pass_rate") },
    ],
  },
  {
    id: "bank",
    title: "prior bank",
    caption: "telemetry · pairs beating the published generic priors",
    detail: `<p>Share of pairs whose teacher lift beats the published prior bank
      — a fixed set of generic, context-free thoughts. Above zero means the
      miner's thought helped the teacher <em>more than a canned prior would
      have</em>, so the lift is specific to this turn rather than to sounding
      thoughtful in general.</p>
      <p>Formerly gate 2 of S* v2 (invalid below γ_bank = 0.08). This is the
      pane to watch for paraphrase-prior strategies: they tie genesis on raw
      Reason but sit at zero here.</p>${TELEMETRY_NOTE}`,
    fmt: fmtPct,
    domain: [0, 1],
    lines: [{ label: "γb", at: (g) => g.gamma_bank ?? 0.08, faint: true }],
    series: [
      { label: "challenger", color: GOLD, get: (p) => sideVal(p, "challenger", "bank_frac") },
      { label: "king", color: BONE, get: (p) => sideVal(p, "king", "bank_frac") },
    ],
  },
  {
    id: "calib",
    title: "calibration r",
    caption: "telemetry · mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)|",
    detail: `<p>Calibration diagnostic on the miner's own logprobs:
      <code>r = mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)|</code>. Values below 1
      mean injecting the thought made the teacher's action cheaper for the
      miner to predict — mean <code>L1lift > 0</code>, the distill signature.
      Live distills measure 0.72–0.81; the teacher scoring itself sits near
      0.35.</p>
      <p>Formerly gate 3 of S* v2 (invalid outside [0.3, 4.0]) — a defense of
      the L1 channel, which is no longer scored.</p>${TELEMETRY_NOTE}`,
    fmt: fmtRatio,
    lines: [
      { label: "r_lo", at: (g) => g.r_lo ?? 0.3, faint: true },
      { label: "r_hi", at: (g) => g.r_hi ?? 4.0, keep: false, faint: true },
    ],
    series: [
      { label: "challenger", color: GOLD, get: (p) => sideVal(p, "challenger", "calib_ratio") },
      { label: "king", color: BONE, get: (p) => sideVal(p, "king", "calib_ratio") },
    ],
  },
  {
    id: "baseline",
    title: "empty baseline",
    caption: "telemetry · challenger mean|lpA(y_C|∅)| ÷ king's",
    detail: `<p>The challenger's empty-context baseline
      <code>mean|lpA(y_C|∅)|</code> divided by the king's on the same slice.
      Both sides see identical turns and identical teacher actions, so an
      honest challenger lands near the king — the observed honest maximum is
      1.14×.</p>
      <p>Formerly gate 3b of S* v2 (invalid above 1.25×) — it closed free
      L1lift minting via baseline sabotage, an attack on a channel that is no
      longer scored.</p>${TELEMETRY_NOTE}`,
    fmt: fmtRatio,
    lines: [
      { label: "band", at: (g) => g.baseline_band ?? 1.25, faint: true },
      { label: "", at: () => 1.0, faint: true },
    ],
    series: [
      {
        label: "ratio",
        color: GOLD,
        get: (p) => {
          const c = sideVal(p, "challenger", "baseline_abs");
          const k = sideVal(p, "king", "baseline_abs");
          return c != null && k ? c / k : null;
        },
      },
    ],
  },
  {
    id: "se",
    title: "paired SE",
    caption: "spread of the per-turn Reason difference",
    detail: `<p>Standard error of the per-turn <code>Reason_c − Reason_k</code>
      differences, i.e. how much this slice disagreed with itself. It sets the
      height of the bar the challenger has to clear, since crowning needs
      <code>margin > k·SE</code>.</p>
      <p>Since Reason v3 there is no floor under it — the duel is purely
      relative to the slice's own noise. Pre-fork duels floored SE at
      <code>min_se = 0.005</code>.</p>`,
    fmt: fmtScore,
    lines: [],
    series: [{ label: "SE", color: GOLD, get: (p) => num(p.se) }],
  },
  {
    id: "len-thought",
    title: "thought length",
    caption: "mean chars of z per side · dashed = teacher",
    detail: `<p>Mean character length of the thoughts each side produced on the
      slice, with the teacher's own thoughts (dashed) as the anchor. Style
      pressure shows up here first: short-style models sit far below the
      teacher, verbose padders far above.</p>
      <p>Recorded since Reason v3 as telemetry — length is not scored, but the
      gap to the teacher is exactly the axis where style-vs-content strategies
      separate.</p>`,
    fmt: fmtInt,
    lines: [],
    series: [
      { label: "challenger", color: GOLD, get: (p) => sideVal(p, "challenger", "mean_len_z") },
      { label: "king", color: BONE, get: (p) => sideVal(p, "king", "mean_len_z") },
      { label: "teacher", color: BONE, dash: true, get: (p) => num(p.teacher?.mean_len_z) },
    ],
  },
  {
    id: "len-action",
    title: "action length",
    caption: "mean chars of y per side · dashed = teacher",
    detail: `<p>Mean character length of the actions each side emitted, against
      the teacher's own actions (dashed). Actions are scored by the teacher's
      likelihood of <em>its own</em> reference action, so action length is pure
      telemetry — but a side whose actions look nothing like the teacher's
      shows up here immediately.</p>`,
    fmt: fmtInt,
    lines: [],
    series: [
      { label: "challenger", color: GOLD, get: (p) => sideVal(p, "challenger", "mean_len_y") },
      { label: "king", color: BONE, get: (p) => sideVal(p, "king", "mean_len_y") },
      { label: "teacher", color: BONE, dash: true, get: (p) => num(p.teacher?.mean_len_y) },
    ],
  },
  {
    id: "len-delta",
    title: "length delta",
    caption: "miner − teacher mean chars · solid = thought · dashed = action",
    detail: `<p>How far each side's output length sits from the teacher's on
      the same slice: <code>Δlen = mean_len(miner) − mean_len(teacher)</code>.
      Solid lines are thought deltas, dashed are action deltas; zero means
      teacher-shaped output.</p>
      <p>A faithful distill should drift toward zero from either direction.
      Persistent large negative thought deltas are the short-style signature;
      large positive ones are padding.</p>`,
    fmt: fmtInt,
    lines: [{ label: "", at: () => 0, faint: true }],
    series: [
      { label: "chall Δz", color: GOLD, get: (p) => sideVal(p, "challenger", "len_z_delta") },
      { label: "king Δz", color: BONE, get: (p) => sideVal(p, "king", "len_z_delta") },
      { label: "chall Δy", color: GOLD, dash: true, get: (p) => sideVal(p, "challenger", "len_y_delta") },
      { label: "king Δy", color: BONE, dash: true, get: (p) => sideVal(p, "king", "len_y_delta") },
    ],
  },
  {
    id: "turns",
    title: "paired turns",
    caption: "turns where both sides produced a scorable pair",
    detail: `<p>Gold is the number of turns that actually entered the paired test;
      bone is the slice the duel was handed. The slice is 80 turns seeded by
      <code>blake2b(reveal_block_hash ‖ hotkey)</code>, so no one can know it
      before reveal and anyone can re-derive it afterwards.</p>
      <p>A shortfall means turns dropped out — a side failed to emit a parsable
      action, or a rollout returned non-finite logprobs. Large gaps make the SE
      wider and the crown harder, which is the intended behaviour.</p>`,
    fmt: fmtInt,
    lines: [],
    series: [
      { label: "paired", color: GOLD, get: (p) => num(p.n_paired_turns) },
      { label: "slice", color: BONE, dash: true, get: (p) => sideVal(p, "challenger", "n_turns") },
    ],
  },
  {
    id: "duration",
    title: "duel time",
    caption: "gold = enqueue → verdict · dashed = scoring only",
    detail: `<p>Gold is wall clock from the validator handing the challenge to
      the eval pod through to the published verdict: model download, vLLM
      warm-up, the injectability probe, then teacher references and both sides
      scored concurrently across the slice. The dashed line (Reason v3
      verdicts) is the scoring portion alone, measured on the pod.</p>
      <p>It is an operations signal, not part of the score. Spikes usually mean
      a large checkpoint pull or a cold engine rather than anything about the
      model's quality.</p>`,
    fmt: fmtDuration,
    lines: [],
    series: [
      { label: "duration", color: GOLD, get: (p) => num(p.duration_s) },
      { label: "scoring", color: BONE, dash: true, get: (p) => num(p.duel_seconds) },
    ],
  },
];

/**
 * The two hero charts, described the same way as the grid panes so the expand
 * view can treat every chart on the page uniformly. Since Reason v3 the
 * heroes are the two quantities the contract actually consists of: the score
 * (Reason) and the crown rule (Margin vs k·SE).
 */
export const HERO_CHARTS = [
  {
    id: "hero-score",
    title: "Reason",
    caption: "best of king and challenger each duel",
    detail: `<p>The higher of the two sides' mean <code>Reason</code> in each
      duel, oldest to newest. It is the level the subnet is currently
      distilling at: it steps up when a stronger model takes the crown and
      drifts with slice difficulty in between.</p>
      <p><code>Reason = lpC(y_C|z_A) − lpC(y_C|∅)</code>: how much the miner's
      thought raises the frozen teacher's likelihood of reproducing its own
      action, measured on SWE-style trajectories. The miner's own weights
      never enter the ranked quantity. This single term is the entire score —
      raw Λ2 tracked swe-rebench as well as the retired mix on the research
      panel (Spearman <code>+0.847</code> vs <code>+0.844</code> @ 15).</p>`,
    draw: (svg, history, opts) => drawDuelScores(svg, history, opts),
  },
  {
    id: "hero-margin",
    title: "Margin",
    caption: "paired z vs king · dashed = 2σ dethrone threshold",
    detail: `<p>The duel itself, in units of its own noise: each bar is
      <code>z = mean(Reason_c − Reason_k) / SE</code> against the reigning
      king. Because the mean is paired per turn, turn difficulty cancels — a
      hard slice hurts both sides equally. Gold bars are crownings, bone is a
      challenger that scored above the king without clearing the bar, red is
      a loss.</p>
      <p>The dotted gold line at 2σ is the dethrone threshold (since
      2026-08-11; was 3σ at Reason v3 launch). Clearing it is the whole crown
      rule under Reason v3, so every bar reads directly as how close that
      challenger came to the crown. There is no absolute floor: the test is
      purely relative to the slice's own noise. Pre-fork duels additionally
      required the retired <code>δ</code> floor.</p>`,
    draw: (svg, history, opts) => drawDuelZ(svg, history, opts),
  },
];

function lastGates(points) {
  for (let i = points.length - 1; i >= 0; i--) {
    if (points[i]?.gates) return points[i].gates;
  }
  return {};
}

/** One pane of the gate grid: every duel on x, one value per side on y. */
export function drawGateMetric(svg, points, metric,
                               { width: widthOpt, height: heightOpt } = {}) {
  const width = Math.max(widthOpt || 320, 240);
  const height = heightOpt || 168;
  const padL = 46;
  const padR = 12;
  const padT = 12;
  const padB = 22;

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!points.length) {
    svg.innerHTML = `<text x="${padL}" y="${height / 2}" fill="rgba(229,229,229,0.35)"
      font-family="${MONO}" font-size="11">no duels yet</text>`;
    return;
  }

  const gates = lastGates(points);
  const cols = metric.series.map((s) => points.map((p) => s.get(p)));
  const lines = (metric.lines || [])
    .map((l) => ({ ...l, v: num(l.at(gates)) }))
    .filter((l) => l.v != null);

  const vals = cols.flat().filter((v) => v != null);
  let lo;
  let hi;
  if (metric.domain) {
    [lo, hi] = metric.domain;
  } else {
    const anchors = lines.filter((l) => l.keep !== false).map((l) => l.v);
    const all = vals.concat(anchors);
    lo = all.length ? Math.min(...all) : 0;
    hi = all.length ? Math.max(...all) : 1;
    const pad = (hi - lo) * 0.15 || Math.abs(hi) * 0.2 || 1;
    lo -= pad;
    hi += pad;
  }
  const n = points.length;
  const slot = (width - padL - padR) / n;
  const xAt = (i) => padL + slot * (i + 0.5);
  const yAt = (v) => padT + ((hi - v) / ((hi - lo) || 1)) * (height - padT - padB);
  const inView = (v) => v >= lo && v <= hi;

  const grid = [hi, (hi + lo) / 2, lo].map((v) => {
    const y = yAt(v);
    return `<g>
      <line x1="${padL}" x2="${width - padR}" y1="${y}" y2="${y}"
        stroke="rgba(255,255,255,0.04)" stroke-width="1" stroke-dasharray="2 4"/>
      <text x="${padL - 6}" y="${y + 3}" text-anchor="end" fill="rgba(229,229,229,0.4)"
        font-family="${MONO}" font-size="9">${esc(metric.fmt(v))}</text>
    </g>`;
  }).join("");

  const thresholds = lines.filter((l) => inView(l.v)).map((l) => {
    const y = yAt(l.v);
    const color = l.faint ? "rgba(229,229,229,0.18)" : GOLD;
    return `<g>
      <line x1="${padL}" x2="${width - padR}" y1="${y}" y2="${y}"
        stroke="${color}" stroke-width="1" stroke-dasharray="2 5"
        opacity="${l.faint ? 1 : 0.75}"/>
      ${l.label ? `<text x="${width - padR}" y="${y - 4}" text-anchor="end" fill="${GOLD}"
        font-family="${MONO}" font-size="9" opacity="0.8">${esc(l.label)}</text>` : ""}
    </g>`;
  }).join("");

  // Crownings get a full-height hairline so every pane reads against the
  // same moments as the hero charts.
  const crowns = points.map((p, i) => (p.event === "crowned"
    ? `<line x1="${xAt(i)}" x2="${xAt(i)}" y1="${padT}" y2="${height - padB}"
        stroke="${GOLD}" stroke-width="1" opacity="0.18"/>`
    : "")).join("");

  const paths = metric.series.map((s, si) => {
    let d = "";
    let open = false;
    cols[si].forEach((v, i) => {
      if (v == null) {
        open = false;
        return;
      }
      d += `${open ? " L" : " M"} ${xAt(i)} ${yAt(Math.min(Math.max(v, lo), hi))}`;
      open = true;
    });
    return d
      ? `<path d="${d.trim()}" fill="none" stroke="${s.color}" stroke-width="1.4"
          opacity="${s.dash ? 0.55 : 0.9}" ${s.dash ? 'stroke-dasharray="3 3"' : ""}/>`
      : "";
  }).join("");

  const dots = metric.series.map((s, si) => cols[si].map((v, i) => {
    if (v == null || s.dash) return "";
    const p = points[i];
    const tip = `${duelTipName(p)} · ${s.label} ${metric.fmt(v)} · ${fmtTime(p.at)}`;
    const cy = yAt(Math.min(Math.max(v, lo), hi));
    return `<g class="duel-hit" data-tip="${esc(tip)}"${duelCidAttr(p)}>
      <circle cx="${xAt(i)}" cy="${cy}" r="8" fill="transparent"/>
      <circle cx="${xAt(i)}" cy="${cy}"
        r="${p.event === "crowned" ? 3 : 2}" fill="${s.color}"/>
    </g>`;
  }).join("")).join("");

  const axis = duelAxisMarks(points, xAt, height - padB);

  svg.innerHTML = `${grid}${crowns}${thresholds}${paths}${dots}${axis}`;
}

/* ---------- duel page charts (per-turn samples of one duel) ---------- */

const turnShort = (tid) => {
  const s = String(tid ?? "—");
  return s.length > 34 ? `${s.slice(0, 34)}…` : s;
};

/** Reason v3 series publish delta_reason / *_reason; pre-fork series carried
 * the mix. One accessor keeps every duel page rendering across the fork. */
const pairedDelta = (p) => p.delta_reason ?? p.delta_mix;
const pairedChall = (p) => p.challenger_reason ?? p.challenger_mix;
const pairedKing = (p) => p.king_reason ?? p.king_mix;

/** Per-turn ΔReason bars in slice order — which turns won or lost the duel. */
export function drawDeltaBars(svg, paired, { width: widthOpt, height: heightOpt } = {}) {
  const pts = (paired || []).filter((p) => pairedDelta(p) != null);
  const width = Math.max(widthOpt || 560, 320);
  const height = heightOpt || 240;
  const padL = 52;
  const padR = 14;
  const padT = 16;
  const padB = 18;
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!pts.length) {
    svg.innerHTML = `<text x="${padL}" y="${height / 2}" fill="rgba(229,229,229,0.35)"
      font-family="${MONO}" font-size="11">no paired turns</text>`;
    return;
  }
  const vals = pts.map((p) => Number(pairedDelta(p)));
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  let lo = Math.min(...vals, 0);
  let hi = Math.max(...vals, 0);
  const pad = (hi - lo) * 0.1 || 0.01;
  lo -= pad;
  hi += pad;
  const n = pts.length;
  const slot = (width - padL - padR) / n;
  const barW = Math.max(2, Math.min(slot * 0.7, 18));
  const yAt = (v) => padT + ((hi - v) / ((hi - lo) || 1)) * (height - padT - padB);
  const y0 = yAt(0);

  const labels = [hi, 0, lo].map((v) => `
    <text x="${padL - 8}" y="${yAt(v) + 3}" text-anchor="end" fill="rgba(229,229,229,0.4)"
      font-family="${MONO}" font-size="9">${esc(fmtScore(v))}</text>`).join("");

  const bars = pts.map((p, i) => {
    const v = Number(pairedDelta(p));
    const x = padL + slot * (i + 0.5);
    const y1 = yAt(v);
    const tip = `${turnShort(p.turn_id)} · ΔReason ${fmtScore(v)} · chall ${fmtScore(pairedChall(p))} vs king ${fmtScore(pairedKing(p))}`;
    return `<g class="duel-hit" data-tip="${esc(tip)}">
      <rect x="${x - barW / 2}" y="${Math.min(y0, y1)}" width="${barW}"
        height="${Math.max(Math.abs(y0 - y1), 1.5)}" rx="1"
        fill="${v >= 0 ? GOLD : "rgba(255,71,71,0.6)"}" opacity="0.9"/>
    </g>`;
  }).join("");

  svg.innerHTML = `
    <line x1="${padL}" x2="${width - padR}" y1="${y0}" y2="${y0}"
      stroke="rgba(255,255,255,0.14)" stroke-width="1"/>
    <line x1="${padL}" x2="${width - padR}" y1="${yAt(mean)}" y2="${yAt(mean)}"
      stroke="${GOLD}" stroke-width="1" stroke-dasharray="2 5" opacity="0.7"/>
    <text x="${width - padR}" y="${yAt(mean) - 5}" text-anchor="end" fill="${GOLD}"
      font-family="${MONO}" font-size="9" opacity="0.85">mean ${esc(fmtScore(mean))}</text>
    ${labels}${bars}`;
}

/** Challenger vs king Reason per turn; above the diagonal = challenger better. */
export function drawSideScatter(svg, paired, { width: widthOpt, height: heightOpt } = {}) {
  const pts = (paired || []).filter(
    (p) => pairedChall(p) != null && pairedKing(p) != null);
  const width = Math.max(widthOpt || 560, 320);
  const height = heightOpt || 240;
  const padL = 52;
  const padR = 14;
  const padT = 16;
  const padB = 30;
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!pts.length) {
    svg.innerHTML = `<text x="${padL}" y="${height / 2}" fill="rgba(229,229,229,0.35)"
      font-family="${MONO}" font-size="11">no paired turns</text>`;
    return;
  }
  const all = pts.flatMap((p) => [Number(pairedChall(p)), Number(pairedKing(p))]);
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 0.01;
  lo -= pad;
  hi += pad;
  const xAt = (v) => padL + ((v - lo) / ((hi - lo) || 1)) * (width - padL - padR);
  const yAt = (v) => padT + ((hi - v) / ((hi - lo) || 1)) * (height - padT - padB);

  const dots = pts.map((p) => {
    const c = Number(pairedChall(p));
    const k = Number(pairedKing(p));
    const win = c >= k;
    const tip = `${turnShort(p.turn_id)} · chall ${fmtScore(c)} vs king ${fmtScore(k)} · Δ ${fmtScore(c - k)}`;
    return `<g class="duel-hit" data-tip="${esc(tip)}">
      <circle cx="${xAt(k)}" cy="${yAt(c)}" r="7" fill="transparent"/>
      <circle cx="${xAt(k)}" cy="${yAt(c)}" r="3"
        fill="${win ? GOLD : "rgba(255,71,71,0.65)"}" opacity="0.85"/>
    </g>`;
  }).join("");

  svg.innerHTML = `
    <line x1="${xAt(lo)}" y1="${yAt(lo)}" x2="${xAt(hi)}" y2="${yAt(hi)}"
      stroke="rgba(255,255,255,0.16)" stroke-width="1" stroke-dasharray="4 4"/>
    <text x="${width / 2}" y="${height - 8}" text-anchor="middle"
      fill="rgba(229,229,229,0.4)" font-family="${MONO}" font-size="10">king Reason</text>
    <text x="14" y="${height / 2}" fill="rgba(229,229,229,0.4)" font-family="${MONO}"
      font-size="10" transform="rotate(-90 14 ${height / 2})">challenger Reason</text>
    ${dots}`;
}

/** Reason vs L1lift for both sides — the score against its telemetry twin. */
export function drawPairScatter(svg, series, { width: widthOpt, height: heightOpt } = {}) {
  const pts = [
    ...(series?.challenger || []).map((p) => ({ ...p, side: "challenger" })),
    ...(series?.king || []).map((p) => ({ ...p, side: "king" })),
  ].filter((p) => p.lambda2 != null && p.l1lift != null);
  const width = Math.max(widthOpt || 560, 320);
  const height = heightOpt || 240;
  const padL = 52;
  const padR = 14;
  const padT = 16;
  const padB = 30;
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!pts.length) {
    svg.innerHTML = `<text x="${padL}" y="${height / 2}" fill="rgba(229,229,229,0.35)"
      font-family="${MONO}" font-size="11">no pair series for this duel</text>`;
    return;
  }
  const xs = pts.map((p) => Number(p.lambda2));
  const ys = pts.map((p) => Number(p.l1lift));
  let x0 = Math.min(...xs, 0);
  let x1 = Math.max(...xs, 0);
  let y0 = Math.min(...ys, 0);
  let y1 = Math.max(...ys, 0);
  const xp = (x1 - x0) * 0.1 || 0.02;
  const yp = (y1 - y0) * 0.1 || 0.02;
  x0 -= xp; x1 += xp; y0 -= yp; y1 += yp;
  const xAt = (v) => padL + ((v - x0) / ((x1 - x0) || 1)) * (width - padL - padR);
  const yAt = (v) => padT + ((y1 - v) / ((y1 - y0) || 1)) * (height - padT - padB);

  const dots = pts.map((p) => {
    const chall = p.side === "challenger";
    const tip = `${p.side} · ${turnShort(p.turn_id)} · Reason ${fmtScore(p.lambda2)} · L1 ${fmtScore(p.l1lift)} · causality ${p.gate_ok ? "ok" : "fail"}`;
    return `<g class="duel-hit" data-tip="${esc(tip)}">
      <circle cx="${xAt(p.lambda2)}" cy="${yAt(p.l1lift)}" r="7" fill="transparent"/>
      <circle cx="${xAt(p.lambda2)}" cy="${yAt(p.l1lift)}" r="3"
        fill="${chall ? GOLD : BONE}" opacity="${p.gate_ok ? 0.85 : 0.9}"
        ${p.gate_ok ? "" : 'stroke="rgba(255,71,71,0.9)" stroke-width="1.4"'}/>
    </g>`;
  }).join("");

  svg.innerHTML = `
    <line x1="${padL}" x2="${width - padR}" y1="${yAt(0)}" y2="${yAt(0)}"
      stroke="rgba(255,255,255,0.12)"/>
    <line x1="${xAt(0)}" x2="${xAt(0)}" y1="${padT}" y2="${height - padB}"
      stroke="rgba(255,255,255,0.12)"/>
    <text x="${width / 2}" y="${height - 8}" text-anchor="middle"
      fill="rgba(229,229,229,0.4)" font-family="${MONO}" font-size="10">Reason</text>
    <text x="14" y="${height / 2}" fill="rgba(229,229,229,0.4)" font-family="${MONO}"
      font-size="10" transform="rotate(-90 14 ${height / 2})">L1lift</text>
    ${dots}`;
}

export function fmtDelta(cur, prev) {
  if (cur == null || prev == null) return "";
  const d = Number(cur) - Number(prev);
  if (!Number.isFinite(d) || d === 0) return "";
  const sign = d > 0 ? "+" : "";
  return `${sign}${fmtScore(d)}`;
}

export function drawReignChain(svg, d) {
  // Oldest → newest so absolute Reason climbs left-to-right over time.
  const members = [...reignMembers(d)].reverse();
  const width = chartWidth();
  const height = 320;
  const padL = 56;
  const padR = 20;
  const padT = 36;
  const padB = 72;
  const n = Math.max(members.length, 1);
  const slot = (width - padL - padR) / n;
  const barW = Math.max(18, Math.min(slot * 0.45, 72));
  const mono = "IBM Plex Mono, monospace";

  const scores = members.map((m) =>
    m.score != null && Number.isFinite(Number(m.score)) ? Number(m.score) : null);
  const known = scores.filter((s) => s != null);
  let lo = known.length ? Math.min(...known) : 0;
  let hi = known.length ? Math.max(...known) : 1;
  if (hi === lo) {
    lo -= Math.abs(lo) * 0.15 || 0.05;
    hi += Math.abs(hi) * 0.15 || 0.05;
  } else {
    const pad = (hi - lo) * 0.18;
    lo -= pad;
    hi += pad * 0.35;
  }
  // Keep a floor under the axis so short bars still read.
  const span = hi - lo || 1;
  const yAt = (v) => padT + ((hi - v) / span) * (height - padT - padB);
  const y0 = yAt(lo);

  const tickCount = 4;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) =>
    lo + (span * i) / tickCount);
  const grid = ticks.map((v) => {
    const y = yAt(v);
    return `<g>
      <line x1="${padL}" x2="${width - padR}" y1="${y}" y2="${y}"
        stroke="rgba(255,255,255,0.04)" stroke-dasharray="2 4"/>
      <text x="${padL - 10}" y="${y + 3}" text-anchor="end" fill="rgba(229,229,229,0.45)"
        font-family="${mono}" font-size="10">${fmtScore(v)}</text>
    </g>`;
  }).join("");

  const cols = members.map((m, i) => {
    const score = scores[i];
    const x = padL + slot * (i + 0.5);
    const current = !!m.current;
    const fill = current ? "#BF9939" : "#C6BDA8";
    const label = m.reign_number != null ? `#${m.reign_number}` : "prior";
    const repo = (m.repo || "").split("/").pop() || short(m.hotkey, 10);
    const prev = i > 0 ? scores[i - 1] : null;
    const delta = fmtDelta(score, prev);
    if (score == null) {
      return `<g>
        <title>${esc(m.repo || m.hotkey)} · Reason unknown</title>
        <text x="${x}" y="${y0 - 8}" text-anchor="middle" fill="rgba(229,229,229,0.35)"
          font-family="${mono}" font-size="10">—</text>
        <text x="${x}" y="${y0 + 18}" text-anchor="middle" fill="${current ? "#FFC93C" : "#e5e5e5"}"
          font-family="${mono}" font-size="11">${esc(label)}</text>
        <text x="${x}" y="${y0 + 34}" text-anchor="middle" fill="rgba(229,229,229,0.45)"
          font-family="${mono}" font-size="9">${esc(short(repo, 16))}</text>
      </g>`;
    }
    const y = yAt(score);
    const h = Math.max(2, y0 - y);
    return `<g>
      <title>${esc(m.repo || m.hotkey)} · Reason=${fmtScore(score)}</title>
      <rect x="${x - barW / 2}" y="${y}" width="${barW}" height="${h}" rx="1" fill="${fill}"/>
      <text x="${x}" y="${y - 8}" text-anchor="middle" fill="${current ? "#FFC93C" : "#e5e5e5"}"
        font-family="${mono}" font-size="10">${fmtScore(score)}</text>
      ${delta ? `<text x="${x}" y="${y - 22}" text-anchor="middle"
        fill="${Number(score) - Number(prev) >= 0 ? "#5ac8fa" : "rgba(255,71,71,0.7)"}"
        font-family="${mono}" font-size="9">${esc(delta)}</text>` : ""}
      <text x="${x}" y="${y0 + 18}" text-anchor="middle" fill="${current ? "#FFC93C" : "#e5e5e5"}"
        font-family="${mono}" font-size="11">${esc(label)}</text>
      <text x="${x}" y="${y0 + 34}" text-anchor="middle" fill="rgba(229,229,229,0.45)"
        font-family="${mono}" font-size="9">${esc(short(repo, 16))}</text>
      ${current ? `<text x="${x}" y="${y0 + 48}" text-anchor="middle" fill="#FFC93C"
        font-family="${mono}" font-size="9">CURRENT</text>` : ""}
    </g>`;
  }).join("");

  let line = "";
  if (known.length > 1) {
    const path = scores.reduce((acc, s, i) => {
      if (s == null) return acc;
      const cmd = acc ? "L" : "M";
      return `${acc}${acc ? " " : ""}${cmd} ${padL + slot * (i + 0.5)} ${yAt(s)}`;
    }, "");
    line = `<path d="${path}" fill="none" stroke="#f3c449" stroke-width="1.5" opacity="0.5"/>`;
  }

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = grid + cols + line;
}

/** SN registration burn (τ) over time — TMC history, oldest → newest. */
export function drawRegPrice(svg, history, { width: widthOpt, height: heightOpt } = {}) {
  const points = Array.isArray(history?.points) ? history.points : [];
  const width = Math.max(widthOpt || chartWidth(), 280);
  const height = heightOpt || 200;
  const padL = 52;
  const padR = 16;
  const padT = 18;
  const padB = 36;
  const mono = "IBM Plex Mono, monospace";
  const gold = "#f3c449";

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (points.length < 2) {
    svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle"
      fill="rgba(229,229,229,0.35)" font-family="${mono}" font-size="12">loading registration history…</text>`;
    return;
  }

  const ys = points.map((p) => Number(p.reg_tao)).filter((v) => Number.isFinite(v));
  let lo = Math.min(...ys);
  let hi = Math.max(...ys);
  if (hi === lo) {
    lo = Math.max(0, lo * 0.9);
    hi = hi * 1.1 || 1;
  } else {
    const pad = (hi - lo) * 0.12;
    lo = Math.max(0, lo - pad);
    hi += pad;
  }
  const span = hi - lo || 1;
  const yAt = (v) => padT + ((hi - v) / span) * (height - padT - padB);
  const xAt = (i) => padL + (i / (points.length - 1)) * (width - padL - padR);

  const ticks = Array.from({ length: 4 }, (_, i) => lo + (span * i) / 3);
  const grid = ticks.map((v) => {
    const y = yAt(v);
    return `<g>
      <line x1="${padL}" x2="${width - padR}" y1="${y}" y2="${y}"
        stroke="rgba(255,255,255,0.04)" stroke-dasharray="2 4"/>
      <text x="${padL - 8}" y="${y + 3}" text-anchor="end" fill="rgba(229,229,229,0.45)"
        font-family="${mono}" font-size="10">${v.toFixed(v >= 10 ? 1 : 2)}</text>
    </g>`;
  }).join("");

  const line = points.reduce((acc, p, i) => {
    const v = Number(p.reg_tao);
    if (!Number.isFinite(v)) return acc;
    const cmd = acc ? "L" : "M";
    return `${acc}${acc ? " " : ""}${cmd} ${xAt(i)} ${yAt(v)}`;
  }, "");

  // Soft fill under the line.
  const area = line
    ? `${line} L ${xAt(points.length - 1)} ${height - padB} L ${xAt(0)} ${height - padB} Z`
    : "";

  const first = points[0];
  const last = points[points.length - 1];
  const xLabels = `
    <text x="${padL}" y="${height - 10}" text-anchor="start" fill="rgba(229,229,229,0.4)"
      font-family="${mono}" font-size="10">${esc(fmtTime(first.t))}</text>
    <text x="${width - padR}" y="${height - 10}" text-anchor="end" fill="rgba(229,229,229,0.4)"
      font-family="${mono}" font-size="10">${esc(fmtTime(last.t))}</text>`;

  const tip = Number.isFinite(Number(last.reg_tao))
    ? `<circle cx="${xAt(points.length - 1)}" cy="${yAt(Number(last.reg_tao))}" r="3.5"
         fill="${gold}"/>
       <text x="${xAt(points.length - 1) - 8}" y="${yAt(Number(last.reg_tao)) - 10}"
         text-anchor="end" fill="${gold}" font-family="${mono}" font-size="11">${esc(fmtTao(last.reg_tao, 3))}</text>`
    : "";

  svg.innerHTML = `${grid}
    ${area ? `<path d="${area}" fill="rgba(243,196,73,0.08)"/>` : ""}
    <path d="${line}" fill="none" stroke="${gold}" stroke-width="1.75"/>
    ${tip}${xLabels}`;
}

