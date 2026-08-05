/** SVG hero / detail charts — keeps the existing Affine visual language. */

export const esc = (s) =>
  String(s ?? "—").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const short = (s, n = 18) => {
  const v = String(s ?? "");
  return v.length > n ? `${v.slice(0, n)}…` : v || "—";
};

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
      current: true,
    });
  }
  for (const hk of chain) {
    if (king && hk === king.hotkey) continue;
    members.push({ hotkey: hk, repo: "", revision: "", current: false });
  }
  const weight_bps = Math.floor(10000 / Math.max(members.length, 1));
  return members.map((m) => ({ ...m, weight_bps }));
}

export function duelPoints(history) {
  return (history || [])
    .filter((r) => r.event !== "failed")
    .filter((r) => r.z != null || r.event === "crowned")
    .slice()
    .reverse();
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

export function drawDuelZ(svg, history) {
  const points = duelPoints(history);
  const width = chartWidth();
  const height = 360;
  const padL = 52;
  const padR = 20;
  const padT = 28;
  const padB = 52;
  const n = Math.max(points.length, 1);
  const slot = (width - padL - padR) / n;
  const barW = Math.max(10, Math.min(slot * 0.55, 64));

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
  if (!ticks.includes(3)) ticks.push(3);
  ticks.sort((a, b) => a - b);

  const gold = "#f3c449";
  const bar = "#c6bda8";
  const mono = "IBM Plex Mono, monospace";

  const grid = ticks.map((v) => {
    const y = yAt(v);
    const major = v === 0 || v === 3;
    return `<g>
      <line x1="${padL}" x2="${width - padR}" y1="${y}" y2="${y}"
        stroke="${major ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)"}"
        stroke-width="1" ${v === 3 ? 'stroke-dasharray="4 4"' : v !== 0 ? 'stroke-dasharray="2 4"' : ""}/>
      <text x="${padL - 10}" y="${y + 3}" fill="rgba(229,229,229,0.45)"
        font-family="${mono}" font-size="10" text-anchor="end">${v === 3 ? "3σ" : v.toFixed(0)}</text>
    </g>`;
  }).join("");

  const columns = points.map((p, i) => {
    const z = zs[i];
    const x = xAt(i);
    const y0 = yAt(0);
    const y1 = yAt(z);
    const top = Math.min(y0, y1);
    const h = Math.max(Math.abs(y0 - y1), 2);
    const crowned = p.event === "crowned";
    const fill = crowned ? gold : (z >= 0 ? bar : "rgba(255,71,71,0.55)");
    const label = crowned
      ? `#${p.reign_number ?? "?"}`
      : (p.repo || "").split("/").pop()?.slice(0, 12) || "duel";
    const zLabel = fmtZ(p.event === "crowned" && p.z == null ? 3 : p.z);
    const showDate = slot >= 56;
    return `<g>
      <title>${esc(p.repo || "")} · ${esc(p.event)} · z=${zLabel}</title>
      <rect x="${x - barW / 2}" y="${top}" width="${barW}" height="${h}" rx="1" fill="${fill}"
        opacity="${crowned ? 1 : 0.92}"/>
      <text x="${x}" y="${top - 8}" text-anchor="middle" fill="${crowned ? gold : "#e5e5e5"}"
        font-family="${mono}" font-size="10" font-weight="${crowned ? 700 : 400}">${esc(zLabel)}</text>
      <text x="${x}" y="${height - padB + 16}" text-anchor="middle"
        fill="${crowned ? gold : "#e5e5e5"}" font-family="${mono}" font-size="10">${esc(label)}</text>
      ${showDate ? `<text x="${x}" y="${height - padB + 32}" text-anchor="middle"
        fill="rgba(229,229,229,0.35)" font-family="${mono}" font-size="9">${esc(fmtTime(p.at))}</text>` : ""}
    </g>`;
  }).join("");

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${grid}${columns}`;
}

export function drawDuelScores(svg, history) {
  // Absolute S* only — paired king / challenger bars per duel.
  const points = duelPoints(history).filter((p) =>
    p.score != null || p.score_king != null || p.event === "crowned");
  const width = chartWidth();
  const height = 360;
  const padL = 56;
  const padR = 20;
  const padT = 36;
  const padB = 56;
  const n = Math.max(points.length, 1);
  const slot = (width - padL - padR) / n;
  const pairW = Math.max(16, Math.min(slot * 0.55, 72));
  const barW = Math.max(6, pairW * 0.42);

  const scores = points.flatMap((p) => {
    const out = [];
    if (p.score != null && Number.isFinite(Number(p.score))) out.push(Number(p.score));
    if (p.score_king != null && Number.isFinite(Number(p.score_king))) out.push(Number(p.score_king));
    return out;
  });
  let lo = scores.length ? Math.min(...scores) : -0.05;
  let hi = scores.length ? Math.max(...scores) : 0;
  if (hi === lo) {
    lo -= Math.abs(lo) * 0.2 || 0.02;
    hi += Math.abs(hi) * 0.2 || 0.02;
  } else {
    const pad = (hi - lo) * 0.18;
    lo -= pad;
    hi += pad * 0.35;
  }
  const span = hi - lo || 1;
  const yAt = (v) => padT + ((hi - v) / span) * (height - padT - padB);
  const yBase = yAt(lo);
  const xAt = (i) => padL + slot * (i + 0.5);

  const gold = "#f3c449";
  const accent = "#5ac8fa";
  const kingFill = "#6a655c";
  const mono = "IBM Plex Mono, monospace";

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

  const legend = `<g font-family="${mono}" font-size="9">
    <rect x="${padL}" y="8" width="8" height="8" fill="${kingFill}"/>
    <text x="${padL + 12}" y="16" fill="rgba(229,229,229,0.45)">king</text>
    <rect x="${padL + 52}" y="8" width="8" height="8" fill="${accent}"/>
    <text x="${padL + 64}" y="16" fill="rgba(229,229,229,0.45)">challenger</text>
    <rect x="${padL + 142}" y="8" width="8" height="8" fill="${gold}"/>
    <text x="${padL + 154}" y="16" fill="rgba(229,229,229,0.45)">crowned</text>
  </g>`;

  const columns = points.map((p, i) => {
    const x = xAt(i);
    const crowned = p.event === "crowned";
    const chall = p.score != null && Number.isFinite(Number(p.score)) ? Number(p.score) : null;
    const king = p.score_king != null && Number.isFinite(Number(p.score_king)) ? Number(p.score_king) : null;
    const label = crowned
      ? `#${p.reign_number ?? "?"}`
      : (p.repo || "").split("/").pop()?.slice(0, 12) || "duel";
    const showDate = slot >= 64;
    const gap = 2;
    const kingX = x - barW - gap / 2;
    const challX = x + gap / 2;
    const challFill = crowned ? gold : accent;

    let bars = "";
    if (king != null) {
      const y = yAt(king);
      bars += `<rect x="${kingX}" y="${y}" width="${barW}" height="${Math.max(2, yBase - y)}"
        rx="1" fill="${kingFill}" opacity="0.9"/>
        <text x="${kingX + barW / 2}" y="${y - 6}" text-anchor="middle"
          fill="rgba(229,229,229,0.4)" font-family="${mono}" font-size="8">${fmtScore(king)}</text>`;
    }
    if (chall != null) {
      const y = yAt(chall);
      bars += `<rect x="${challX}" y="${y}" width="${barW}" height="${Math.max(2, yBase - y)}"
        rx="1" fill="${challFill}"/>
        <text x="${challX + barW / 2}" y="${y - 6}" text-anchor="middle"
          fill="${crowned ? gold : accent}" font-family="${mono}" font-size="8">${fmtScore(chall)}</text>`;
    }
    if (king == null && chall == null) {
      bars = `<text x="${x}" y="${yBase - 8}" text-anchor="middle" fill="rgba(229,229,229,0.3)"
        font-family="${mono}" font-size="10">—</text>`;
    }

    const delta = (king != null && chall != null)
      ? fmtDelta(chall, king)
      : "";

    return `<g>
      <title>${esc(p.repo || "")} · chall=${fmtScore(chall)} · king=${fmtScore(king)}</title>
      ${bars}
      ${delta ? `<text x="${x}" y="${padT - 4}" text-anchor="middle"
        fill="${Number(chall) - Number(king) >= 0 ? accent : "rgba(255,71,71,0.7)"}"
        font-family="${mono}" font-size="9">${esc(delta)}</text>` : ""}
      <text x="${x}" y="${height - padB + 16}" text-anchor="middle"
        fill="${crowned ? gold : "#e5e5e5"}" font-family="${mono}" font-size="10">${esc(label)}</text>
      ${showDate ? `<text x="${x}" y="${height - padB + 32}" text-anchor="middle"
        fill="rgba(229,229,229,0.35)" font-family="${mono}" font-size="9">${esc(fmtTime(p.at))}</text>` : ""}
    </g>`;
  }).join("");

  // Challenger trend line across duels.
  let line = "";
  const challPath = points.reduce((acc, p, i) => {
    if (p.score == null || !Number.isFinite(Number(p.score))) return acc;
    const cmd = acc ? "L" : "M";
    return `${acc}${acc ? " " : ""}${cmd} ${xAt(i)} ${yAt(Number(p.score))}`;
  }, "");
  if (challPath) {
    line = `<path d="${challPath}" fill="none" stroke="${accent}" stroke-width="1.25" opacity="0.45"/>`;
  }

  if (!points.length) {
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle"
      fill="rgba(229,229,229,0.35)" font-family="${mono}" font-size="12">no absolute S* recorded yet</text>`;
    return;
  }

  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${legend}${grid}${columns}${line}`;
}

export function fmtDelta(cur, prev) {
  if (cur == null || prev == null) return "";
  const d = Number(cur) - Number(prev);
  if (!Number.isFinite(d) || d === 0) return "";
  const sign = d > 0 ? "+" : "";
  return `${sign}${fmtScore(d)}`;
}

export function drawReignChain(svg, d) {
  // Oldest → newest so absolute S* climbs left-to-right over time.
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
        <title>${esc(m.repo || m.hotkey)} · S* unknown</title>
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
      <title>${esc(m.repo || m.hotkey)} · S*=${fmtScore(score)}</title>
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

/** Compact per-suite score bars across benched models. */
export function drawBenchSuite(svg, suite, models, { width } = {}) {
  const w = Math.max(width || 320, 220);
  const height = 148;
  const padL = 40;
  const padR = 12;
  const padT = 16;
  const padB = 40;
  const mono = "IBM Plex Mono, monospace";
  const accent = "#5ac8fa";
  const fail = "rgba(255,71,71,0.55)";
  const dim = "rgba(229,229,229,0.28)";

  const rows = (models || []).map((m) => {
    const r = m?.suites?.[suite];
    const score = r?.ok && r.score != null && Number.isFinite(Number(r.score))
      ? Number(r.score) : null;
    const failed = r && r.ok === false;
    const label = m.label || (m.model_repo || "").split("/").pop() || "model";
    return { label, score, failed, finished_at: r?.finished_at };
  });

  svg.setAttribute("width", String(w));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${w} ${height}`);

  if (!rows.length) {
    svg.innerHTML = `<text x="${w / 2}" y="${height / 2}" text-anchor="middle"
      fill="rgba(229,229,229,0.35)" font-family="${mono}" font-size="11">no runs yet</text>`;
    return;
  }

  const scores = rows.map((r) => r.score).filter((s) => s != null);
  let lo = 0;
  let hi = scores.length ? Math.max(...scores) : 1;
  if (hi <= 0) hi = 1;
  if (scores.length && Math.min(...scores) < 0) {
    lo = Math.min(0, ...scores);
  }
  // Leave headroom above the tallest bar.
  hi = hi + (hi - lo) * 0.18 || 1;
  const span = hi - lo || 1;
  const yAt = (v) => padT + ((hi - v) / span) * (height - padT - padB);
  const y0 = yAt(Math.max(lo, 0));
  const n = rows.length;
  const slot = (w - padL - padR) / n;
  const barW = Math.max(10, Math.min(slot * 0.55, 36));

  const ticks = [lo, lo + span / 2, hi];
  const grid = ticks.map((v) => {
    const y = yAt(v);
    return `<g>
      <line x1="${padL}" x2="${w - padR}" y1="${y}" y2="${y}"
        stroke="rgba(255,255,255,0.04)" stroke-dasharray="2 4"/>
      <text x="${padL - 6}" y="${y + 3}" text-anchor="end" fill="rgba(229,229,229,0.4)"
        font-family="${mono}" font-size="9">${fmtScore(v)}</text>
    </g>`;
  }).join("");

  const bars = rows.map((r, i) => {
    const x = padL + slot * (i + 0.5);
    const label = short(r.label, 10);
    if (r.score != null) {
      const y = yAt(r.score);
      const h = Math.max(2, y0 - y);
      return `<g>
        <title>${esc(r.label)} · ${fmtScore(r.score)}</title>
        <rect x="${x - barW / 2}" y="${y}" width="${barW}" height="${h}"
          rx="1" fill="${accent}"/>
        <text x="${x}" y="${y - 5}" text-anchor="middle" fill="${accent}"
          font-family="${mono}" font-size="9">${esc(fmtScore(r.score))}</text>
        <text x="${x}" y="${height - 14}" text-anchor="middle" fill="#e5e5e5"
          font-family="${mono}" font-size="9">${esc(label)}</text>
      </g>`;
    }
    if (r.failed) {
      return `<g>
        <title>${esc(r.label)} · failed</title>
        <line x1="${x - barW / 2}" x2="${x + barW / 2}" y1="${y0}" y2="${y0}"
          stroke="${fail}" stroke-width="2"/>
        <text x="${x}" y="${y0 - 8}" text-anchor="middle" fill="${fail}"
          font-family="${mono}" font-size="9">fail</text>
        <text x="${x}" y="${height - 14}" text-anchor="middle" fill="${dim}"
          font-family="${mono}" font-size="9">${esc(label)}</text>
      </g>`;
    }
    return `<g>
      <text x="${x}" y="${y0 - 8}" text-anchor="middle" fill="${dim}"
        font-family="${mono}" font-size="9">—</text>
      <text x="${x}" y="${height - 14}" text-anchor="middle" fill="${dim}"
        font-family="${mono}" font-size="9">${esc(label)}</text>
    </g>`;
  }).join("");

  svg.innerHTML = `${grid}${bars}`;
}

/** SN registration burn (τ) over time — TMC history, oldest → newest. */
export function drawRegPrice(svg, history) {
  const points = Array.isArray(history?.points) ? history.points : [];
  const width = chartWidth();
  const height = 200;
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

