const REFRESH_MS = 15000;
const $ = (id) => document.getElementById(id);

let filter = "";
let heroTab = "duel";
let cache = { dashboard: null, benchmarks: null, history: null };

async function getJSON(path) {
  try {
    const r = await fetch(`${path}?t=${Date.now()}`, { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

const esc = (s) =>
  String(s ?? "—").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const short = (s, n = 18) => {
  const v = String(s ?? "");
  return v.length > n ? `${v.slice(0, n)}…` : v || "—";
};

function fmtTime(iso) {
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

function fmtZ(z) {
  if (z == null || Number.isNaN(Number(z))) return "—";
  const n = Number(z);
  return `${n > 0 ? "+" : ""}${n.toFixed(2)}`;
}

function hubUrl(repo) {
  return repo ? `https://huggingface.co/${repo}` : null;
}

function badge(kind, text) {
  return `<span class="badge ${kind}">${esc(text)}</span>`;
}

function matches(row, q) {
  if (!q) return true;
  const hay = [
    row.event, row.repo, row.hotkey, row.error_code,
    row.rejection_reason, row.challenge_id,
  ].join(" ").toLowerCase();
  return hay.includes(q);
}

function reignMembers(d) {
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

function duelPoints(history) {
  return (history || [])
    .filter((r) => r.event !== "failed")
    .filter((r) => r.z != null || r.event === "crowned")
    .slice()
    .reverse();
}

/* ---------- hero charts (different from affine.io env bars) ---------- */

function chartWidth() {
  return Math.max(window.innerWidth || 960, 320);
}

function fmtScore(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  const abs = Math.abs(n);
  if (abs >= 10) return n.toFixed(1);
  if (abs >= 1) return n.toFixed(2);
  return n.toFixed(3);
}

function drawDuelZ(svg, history) {
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

function drawDuelScores(svg, history) {
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

function fmtDelta(cur, prev) {
  if (cur == null || prev == null) return "";
  const d = Number(cur) - Number(prev);
  if (!Number.isFinite(d) || d === 0) return "";
  const sign = d > 0 ? "+" : "";
  return `${sign}${fmtScore(d)}`;
}

function drawReignChain(svg, d) {
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

function renderHero() {
  const svg = $("hero-chart");
  if (!svg) return;
  switch (heroTab) {
    case "score":
      $("hero-caption").textContent =
        "Absolute S* per duel · gray = king · blue = challenger · gold = coronation · Δ above = chall − king";
      drawDuelScores(svg, cache.history);
      break;
    case "reign":
      $("hero-caption").textContent =
        "Reign evolution · absolute S* at coronation · gold = current · delta vs prior king";
      drawReignChain(svg, cache.dashboard);
      break;
    case "duel":
    default:
      $("hero-caption").textContent =
        "Paired duel z vs king · gold = coronation · dashed = 3σ dethrone threshold";
      drawDuelZ(svg, cache.history);
      break;
  }

  const d = cache.dashboard;
  const k = d?.king;
  const stats = d?.stats || {};
  const q = (d?.queue || []).length;
  const name = k?.repo?.split("/").pop() || "—";
  $("hero-king-row").innerHTML = [
    ["champion", k ? `<b class="gold">${esc(name)}</b>` : "<b>—</b>"],
    ["reign", `<b>${k ? `#${esc(k.reign_number)}` : "—"}</b>`],
    ["queue", `<b>${esc(q)}</b>`],
    ["duels", `<b>${esc(stats.duels ?? "—")}</b>`],
    ["phase", `<b>${esc(d?.phase?.name ?? "—")}</b>`],
  ].map(([lab, val]) =>
    `<span class="stat-chip"><span class="k">${lab}</span>${val}</span>`).join("");
}

/* ---------- sections ---------- */

function renderReign(d) {
  const members = reignMembers(d);
  const size = d?.reign?.size ?? 5;
  if (!members.length) {
    $("reign-meta").textContent = "burn";
    $("reign-wrap").innerHTML = `<div class="empty">no weight holders — emissions burn</div>`;
    return;
  }
  const pct = ((members[0].weight_bps || 0) / 100).toFixed(0);
  $("reign-meta").textContent = `last ${size} · ${members.length} active · ${pct}% each`;
  $("reign-wrap").innerHTML = `<table class="data-table">
    <thead><tr>
      <th>reign</th><th>model</th><th>revision</th><th>hotkey</th><th>crowned</th><th class="r">S*</th><th class="r">weight</th>
    </tr></thead>
    <tbody>${members.map((m) => {
      const url = hubUrl(m.repo);
      const model = m.repo
        ? (url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(m.repo)}</a>` : esc(m.repo))
        : "—";
      const wPct = ((m.weight_bps || 0) / 100).toFixed(0);
      return `<tr class="${m.current ? "current" : ""}">
        <td class="${m.current ? "gold" : "dim"}">${m.reign_number != null ? `#${esc(m.reign_number)}` : "prior"}</td>
        <td>${model}</td>
        <td class="dim">${esc(short(m.revision, 12))}</td>
        <td title="${esc(m.hotkey)}">${esc(short(m.hotkey, 20))}</td>
        <td class="when">${m.crowned_at ? esc(fmtTime(m.crowned_at)) : "—"}</td>
        <td class="r ${m.current ? "gold" : ""}">${esc(fmtScore(m.score))}</td>
        <td class="r"><span class="weight-cell">${esc(wPct)}% <span class="bar"><i style="width:${esc(wPct)}%"></i></span></span></td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

function renderChallenge(d) {
  const ce = d?.current_eval;
  if (!ce) {
    $("challenge-meta").textContent = "idle";
    $("challenge-wrap").innerHTML = `<div class="empty">no duel in flight</div>`;
    return;
  }
  $("challenge-meta").textContent = ce.stage || "running";
  const progress = ce.progress
    ? Object.entries(ce.progress).map(([k, v]) => `${k}: ${v}`).join(" · ")
    : "—";
  $("challenge-wrap").innerHTML = `<div class="kv-grid">
    <div class="kv"><span class="k">challenge</span><span class="v">${esc(ce.challenge_id)}</span></div>
    <div class="kv"><span class="k">challenger</span><span class="v"><a href="${esc(hubUrl(ce.repo) || "#")}" target="_blank" rel="noopener">${esc(ce.repo)}</a></span></div>
    <div class="kv"><span class="k">stage</span><span class="v gold">${esc(ce.stage)}</span></div>
    <div class="kv"><span class="k">progress</span><span class="v dim">${esc(progress)}</span></div>
  </div>`;
}

function renderQueue(d) {
  const q = d?.queue || [];
  const ce = d?.current_eval;
  $("queue-meta").textContent = q.length ? `${q.length} pending` : "queue idle";
  if (!q.length && !ce) {
    $("queue-wrap").innerHTML = `<div class="empty">empty</div>`;
    return;
  }
  const rows = [];
  if (ce) {
    rows.push({
      status: "evaluating", id: ce.challenge_id, repo: ce.repo,
      hotkey: ce.hotkey || "—", queued: "now", retries: "—",
    });
  }
  for (const e of q) {
    rows.push({
      status: "queued", id: e.challenge_id, repo: e.repo,
      hotkey: e.hotkey, queued: fmtTime(e.queued_at), retries: e.retry_count ?? 0,
    });
  }
  $("queue-wrap").innerHTML = `<table class="data-table">
    <thead><tr>
      <th>status</th><th>id</th><th>model</th><th>hotkey</th><th>queued</th><th class="r">retries</th>
    </tr></thead>
    <tbody>${rows.map((r) => {
      const url = hubUrl(r.repo);
      return `<tr class="${r.status === "evaluating" ? "current" : ""}">
        <td>${badge(r.status, r.status)}</td>
        <td>${esc(short(r.id, 14))}</td>
        <td>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(r.repo)}</a>` : esc(r.repo)}</td>
        <td>${esc(short(r.hotkey, 18))}</td>
        <td class="when">${esc(r.queued)}</td>
        <td class="r">${esc(r.retries)}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

function renderBenchmarks(b) {
  const el = $("benchmarks-wrap");
  if (!b?.models?.length) {
    el.innerHTML = `<div class="empty">no benchmark results yet</div>`;
    return;
  }
  const suites = [...new Set(b.models.flatMap((m) => Object.keys(m.suites || {})))].sort();
  el.innerHTML = `<table class="data-table">
    <thead><tr>
      <th>model</th>${suites.map((s) => `<th class="r">${esc(s.replace(/^tau2_/, ""))}</th>`).join("")}
    </tr></thead>
    <tbody>${b.models.map((m) => `<tr>
      <td><a href="${esc(hubUrl(m.model_repo) || "#")}" target="_blank" rel="noopener">${esc(m.label || m.model_repo)}</a></td>
      ${suites.map((s) => {
        const r = m.suites?.[s];
        if (!r) return `<td class="r dim">—</td>`;
        return r.ok
          ? `<td class="r ok">${esc(Number(r.score).toFixed(3))}</td>`
          : `<td class="r bad">fail</td>`;
      }).join("")}
    </tr>`).join("")}</tbody>
  </table>`;
}

function outcomeBadge(r) {
  if (r.event === "crowned") return badge("crowned", `crowned #${r.reign_number ?? "?"}`);
  if (r.event === "failed") return badge("failed", r.error_code || "failed");
  if (r.accepted) return badge("accepted", "accepted");
  if (r.accepted === false) return badge("rejected", "rejected");
  return badge("queued", r.event || "event");
}

function renderHistory(h) {
  const rows = (h || [])
    .filter((r) => r.event !== "failed")
    .filter((r) => matches(r, filter));
  $("history-meta").textContent = `${rows.length} shown`;
  if (!rows.length) {
    $("history-wrap").innerHTML = `<div class="empty">empty</div>`;
    return;
  }
  $("history-wrap").innerHTML = `<table class="data-table">
    <thead><tr>
      <th>when</th><th>event</th><th>model</th><th>outcome</th><th class="r">z</th><th class="r">S*</th><th class="r">king S*</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 80).map((r) => {
      const url = hubUrl(r.repo);
      const zClass = r.z == null ? "" : Number(r.z) >= 0 ? "ok" : "bad";
      return `<tr class="${r.event === "crowned" ? "current" : ""}">
        <td class="when">${esc(fmtTime(r.at))}</td>
        <td>${esc(r.event)}</td>
        <td>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(r.repo)}</a>` : esc(r.repo)}</td>
        <td>${outcomeBadge(r)}</td>
        <td class="r ${zClass}">${esc(fmtZ(r.z))}</td>
        <td class="r">${esc(fmtScore(r.score))}</td>
        <td class="r dim">${esc(fmtScore(r.score_king))}</td>
        <td class="dim">${esc(short(r.rejection_reason || r.error_detail || "—", 48))}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

function renderFails(h) {
  const fails = (h || []).filter((r) =>
    r.event === "failed" || (r.accepted === false && r.event !== "crowned"));
  const rows = fails.filter((r) => matches(r, filter));
  $("fails-meta").textContent = `${rows.length} shown`;
  if (!rows.length) {
    $("fails-wrap").innerHTML = `<div class="empty">none</div>`;
    return;
  }
  $("fails-wrap").innerHTML = `<table class="data-table">
    <thead><tr>
      <th>when</th><th>model</th><th>code</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 60).map((r) => {
      const url = hubUrl(r.repo);
      return `<tr>
        <td class="when">${esc(fmtTime(r.at))}</td>
        <td>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(r.repo)}</a>` : esc(r.repo)}</td>
        <td class="bad">${esc(r.error_code || r.rejection_reason || "reject")}</td>
        <td class="dim">${esc(short(r.error_detail || r.rejection_reason || "—", 64))}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

function renderAll() {
  const d = cache.dashboard;
  renderHero();
  if (d) {
    renderReign(d);
    renderChallenge(d);
    renderQueue(d);
  }
  renderBenchmarks(cache.benchmarks);
  renderHistory(cache.history);
  renderFails(cache.history);
}

async function refresh() {
  const [d, b, h] = await Promise.all([
    getJSON("data/dashboard.json"),
    getJSON("data/benchmarks.json"),
    getJSON("data/history.json"),
  ]);
  cache.dashboard = d;
  cache.benchmarks = b;
  cache.history = h;
  renderAll();
}

function wire() {
  document.querySelectorAll(".hero-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      heroTab = btn.dataset.tab;
      document.querySelectorAll(".hero-tab").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      renderHero();
    });
  });
  $("filter-input")?.addEventListener("input", (e) => {
    filter = e.target.value.trim().toLowerCase();
    renderHistory(cache.history);
    renderFails(cache.history);
  });
  window.addEventListener("resize", () => renderHero());
}

async function boot() {
  wire();
  await refresh();
  setInterval(refresh, REFRESH_MS);
}

boot();
