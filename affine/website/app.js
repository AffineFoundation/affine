import {
  fetchBenchmarks,
  fetchDuel,
  fetchDuelSeries,
  fetchHistory,
  fetchRegHistory,
  fingerprint,
  watchSnapshot,
} from "./api.js?v=27";
import {
  GATE_METRICS,
  drawDuelScores,
  drawDuelZ,
  drawGateMetric,
  drawRegPrice,
  esc,
  gatePoints,
  fmtAge,
  fmtAlpha,
  fmtDuration,
  fmtScore,
  fmtTao,
  fmtTime,
  fmtUsd,
  fmtZ,
  modelDisplayName,
  reignMembers,
  setReignLookup,
  short,
} from "./charts.js?v=27";

const $ = (id) => document.getElementById(id);

let filter = "";
let cache = { dashboard: null, benchmarks: null, history: null, regHistory: null };
let fps = { dashboard: "", benchmarks: "", history: "", hero: "", reg: "", gates: "" };
let closeWatch = null;

const hubUrl = (repo) => (repo ? `https://huggingface.co/${repo}` : null);
const tmcHotkeyUrl = (hk) =>
  (hk ? `https://taomarketcap.com/hotkey/${encodeURIComponent(hk)}` : null);
// Models are not stored on Hippius — only per-duel eval archives are. The
// model name keeps its HF href; duel artifacts get their own Hippius link in
// the detail panel (hippiusEvalUrl).
const hippiusEvalUrl = (cid) =>
  (cid
    ? `https://s3.hippius.com/affine-sn120/evals/${encodeURIComponent(cid)}.json.gz`
    : null);

// Kings render as Affine-<roman>, everything else as Affine-<hotkey[0:5]>;
// the real repo string stays discoverable via the title attribute.
function modelLink(repo, hotkey, reignNumber) {
  if (!repo && !hotkey) return "—";
  const name = modelDisplayName(repo, hotkey, reignNumber);
  const title = repo || hotkey || "";
  const url = hubUrl(repo);
  return url
    ? `<a href="${esc(url)}" target="_blank" rel="noopener" title="${esc(title)}" onclick="event.stopPropagation()">${esc(name)}</a>`
    : `<span title="${esc(title)}">${esc(name)}</span>`;
}

function hotkeyLink(hk) {
  const url = tmcHotkeyUrl(hk);
  if (!hk) return "—";
  const label = short(hk, 18);
  return url
    ? `<a href="${esc(url)}" target="_blank" rel="noopener" title="${esc(hk)}" onclick="event.stopPropagation()">${esc(label)}</a>`
    : `<span title="${esc(hk)}">${esc(label)}</span>`;
}

function badge(kind, text) {
  return `<span class="badge ${kind}">${esc(text)}</span>`;
}

function chartWidth() {
  return Math.max(window.innerWidth || 960, 320);
}

/* ---------- hero ---------- */

function paneWidth(svg) {
  const host = svg?.closest(".hero-pane") || svg?.parentElement;
  const w = host?.clientWidth || Math.floor(chartWidth() / 2);
  return Math.max(w - 20, 280);
}

function renderHero(force = false) {
  const scoreSvg = $("hero-chart-score");
  const zSvg = $("hero-chart-z");
  if (!scoreSvg || !zSvg) return;
  const d = cache.dashboard;
  // Market bar is independent of chart dirty-checks.
  renderMarketBar(d);

  const key = `${fps.history}|${fps.dashboard}|${chartWidth()}`;
  if (!force && key === fps.hero) return;
  fps.hero = key;

  drawDuelScores(scoreSvg, cache.history, { width: paneWidth(scoreSvg) });
  drawDuelZ(zSvg, cache.history, { width: paneWidth(zSvg) });
}

function renderMarketBar(d) {
  const el = $("market-bar-inner");
  if (!el) return;
  const market = d?.market;
  if (!market) {
    el.innerHTML = `<span class="market-item dim">SN120 · waiting on TaoMarketCap</span>`;
    return;
  }
  const weightsSrc = market.weights_source === "validator"
    ? "set by validator" : "commit (TMC)";
  const weightsTitle = market.weights_committed_at
    ? `last weights ${weightsSrc} ${market.weights_committed_at}`
    : "last weights commit (TMC)";
  const updated = market.updated_at
    ? `TMC updated ${fmtAge(market.updated_at)} ago`
    : "TaoMarketCap";
  el.innerHTML = [
    `<span class="market-item"><span class="k">SN</span><b>120</b></span>`,
    `<span class="market-item"><span class="k">price</span><b class="gold">${esc(fmtTao(market.price_tao, 4))}</b></span>`,
    `<span class="market-item"><span class="k">reg</span><b>${esc(fmtTao(market.reg_cost_tao, 3))}</b></span>`,
    `<span class="market-item" title="${esc(weightsTitle)}"><span class="k">weights</span><b>${esc(fmtAge(market.weights_committed_at))}</b></span>`,
    market.block_number != null
      ? `<span class="market-item"><span class="k">block</span><b>${esc(market.block_number)}</b></span>`
      : "",
    `<span class="market-item dim" title="${esc(updated)}">tmc</span>`,
  ].filter(Boolean).join("");
}

/* ---------- sections ---------- */

// Advisory bench references shown as deltas in the reign table. Baseline is
// the stock Qwen the Albedo kings are fine-tuned from; genesis is the seed
// king. Both are matched in the bench payload by label (repo as fallback).
const BENCH_BASELINE = { label: "baseline", repo: "Qwen/Qwen3.6-35B-A3B" };
const BENCH_GENESIS = {
  label: "reign-0",
  repo: "dendriteholdings/albedo-qwen3.6-35b-king-genesis",
};

// Genesis (reign 0 → Affine-I) is known statically; the rest of the reign
// lookup arrives with the first snapshot (see applySnapshot).
setReignLookup([], BENCH_GENESIS.repo);

function benchInfo(b) {
  const suite = (Array.isArray(b?.suites) && b.suites[0]) || "swe_rebench_lite";
  const scores = new Map();
  for (const m of b?.models || []) {
    const s = m?.suites?.[suite]?.score;
    if (s != null && Number.isFinite(Number(s))) {
      scores.set(m.model_repo, Number(s));
    }
  }
  const refScore = (ref) => {
    const hit = (b?.models || []).find(
      (m) => m.label === ref.label || m.model_repo === ref.repo);
    const s = hit?.suites?.[suite]?.score;
    return s != null && Number.isFinite(Number(s)) ? Number(s) : null;
  };
  return { suite, scores, qwen: refScore(BENCH_BASELINE), genesis: refScore(BENCH_GENESIS) };
}

function deltaCell(score, ref, fmt) {
  if (score == null || ref == null) return `<td class="r dim">—</td>`;
  const d = score - ref;
  const cls = d > 0 ? "ok" : d < 0 ? "bad" : "dim";
  return `<td class="r ${cls}">${esc(fmt(d, ref))}</td>`;
}

const fmtRelPct = (d, ref) =>
  ref === 0 ? "—" : `${d > 0 ? "+" : ""}${((d / ref) * 100).toFixed(0)}%`;
const fmtAbsDelta = (d) => `${d > 0 ? "+" : ""}${d.toFixed(2)}`;

function renderReign(d) {
  const members = reignMembers(d);
  if (!members.length) {
    $("reign-meta").textContent = "burn";
    $("reign-wrap").innerHTML = `<div class="empty">no weight holders — emissions burn</div>`;
    return;
  }
  const bench = benchInfo(cache.benchmarks);
  const earners = members.filter((m) => m.earning || (m.weight_bps || 0) > 0);
  const pct = earners.length
    ? ((earners[0].weight_bps || 0) / 100).toFixed(0)
    : "0";
  const benchBits = [];
  if (bench.qwen != null) benchBits.push(`qwen ${fmtScore(bench.qwen)}`);
  if (bench.genesis != null) benchBits.push(`Affine-I ${fmtScore(bench.genesis)}`);
  $("reign-meta").textContent =
    `${members.length} kings · ${earners.length} earning · ${pct}% each`
    + (benchBits.length ? ` · swe: ${benchBits.join(" / ")}` : "");
  $("reign-wrap").innerHTML = `<table class="data-table">
    <thead><tr>
      <th>reign</th><th>crowned</th><th>uid</th><th>model</th><th>hotkey</th>
      <th class="r">swe</th><th class="r">vs qwen</th><th class="r">vs Affine-I</th>
      <th class="r">S*</th><th class="r">α/day</th><th class="r">$/day</th><th class="r">weight</th>
    </tr></thead>
    <tbody>${members.map((m) => {
      const earning = m.earning || (m.weight_bps || 0) > 0;
      const wPct = ((m.weight_bps || 0) / 100).toFixed(0);
      const alpha = earning ? fmtAlpha(m.alpha_per_day) : "—";
      const usd = earning ? fmtUsd(m.usd_per_day) : "—";
      const swe = bench.scores.has(m.repo) ? bench.scores.get(m.repo) : null;
      return `<tr class="${m.current ? "current" : ""}">
        <td class="${m.current ? "gold" : "dim"}">${m.reign_number != null ? `#${esc(m.reign_number)}` : "prior"}</td>
        <td class="when">${m.crowned_at ? esc(fmtTime(m.crowned_at)) : "—"}</td>
        <td class="dim">${m.uid != null ? esc(m.uid) : "—"}</td>
        <td>${modelLink(m.repo, m.hotkey, m.reign_number)}</td>
        <td>${hotkeyLink(m.hotkey)}</td>
        <td class="r ${swe != null ? "" : "dim"}">${esc(swe != null ? fmtScore(swe) : "—")}</td>
        ${deltaCell(swe, bench.qwen, fmtRelPct)}
        ${deltaCell(swe, bench.genesis, fmtAbsDelta)}
        <td class="r ${m.current ? "gold" : ""}">${esc(fmtScore(m.score))}</td>
        <td class="r ${earning ? "gold" : "dim"}">${esc(alpha)}</td>
        <td class="r ${earning ? "" : "dim"}">${esc(usd)}</td>
        <td class="r">${earning
          ? `<span class="weight-cell">${esc(wPct)}% <span class="bar"><i style="width:${esc(wPct)}%"></i></span></span>`
          : m.inaccessible
            ? `<span class="bad" title="model repo gone/gated on HF — forfeits payout while dark">gated</span>`
            : `<span class="dim">—</span>`}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

function renderRegPrice(force = false) {
  const svg = $("reg-price-chart");
  if (!svg) return;
  const hist = cache.regHistory;
  const points = hist?.points || [];
  const last = points.length ? points[points.length - 1] : null;
  const key = `${fps.reg}|${chartWidth()}|${last?.reg_tao ?? ""}`;
  if (!force && key === fps.regRender) return;
  fps.regRender = key;
  const meta = $("reg-price-meta");
  if (meta) {
    if (last?.reg_tao != null) {
      const n = points.length;
      meta.textContent = `${fmtTao(last.reg_tao, 3)} · ${n} pts · tmc`;
    } else {
      meta.textContent = "tmc burn history";
    }
  }
  drawRegPrice(svg, hist);
}

function renderGates(force = false) {
  const wrap = $("gates-wrap");
  if (!wrap) return;
  const key = `${fps.history}|${chartWidth()}`;
  if (!force && key === fps.gates) return;
  fps.gates = key;

  const points = gatePoints(cache.history);
  const meta = $("gates-meta");
  if (meta) {
    meta.textContent = points.length
      ? `${points.length} scored duels · challenger gold · king bone · dashed = gate`
      : "waiting on a scored duel";
  }
  if (!points.length) {
    wrap.innerHTML = `<div class="empty">no scored duels yet</div>`;
    return;
  }
  if (!wrap.querySelector(".metric-pane")) {
    wrap.innerHTML = GATE_METRICS.map((m) => `
      <div class="metric-pane" id="metric-${esc(m.id)}">
        <div class="metric-head">
          <span class="metric-title">${esc(m.title)}</span>
          <span class="metric-caption">${esc(m.caption)}</span>
        </div>
        <svg role="img" aria-label="${esc(m.title)} per duel"></svg>
      </div>`).join("");
  }
  for (const m of GATE_METRICS) {
    const pane = $(`metric-${m.id}`);
    const svg = pane?.querySelector("svg");
    if (!svg) continue;
    drawGateMetric(svg, points, m, {
      width: Math.max((pane.clientWidth || 320) - 18, 240),
    });
  }
}

function intakeBadge(decision) {
  const d = String(decision || "");
  if (d === "enqueued") return badge("accepted", "enqueued");
  if (d.startsWith("rejected")) return badge("failed", d.replace(/^rejected_/, ""));
  if (d.startsWith("skipped")) return badge("rejected", d.replace(/^skipped_/, ""));
  return badge("queued", d || "intake");
}

function renderIntake(d) {
  const el = $("intake-wrap");
  const meta = $("intake-meta");
  if (!el) return;
  const rows = [...(d?.intake || [])].reverse();
  const stats = d?.stats || {};
  if (meta) {
    const total = stats.enqueued_total ?? stats.queued;
    meta.textContent = total != null
      ? `${rows.length} recent · ${total} enqueued all-time`
      : "reveal → decision · not the duel queue";
  }
  if (!rows.length) {
    el.innerHTML = `<div class="empty">no reveal decisions yet — LastCommitment alone does not appear here</div>`;
    return;
  }
  el.innerHTML = `<table class="data-table">
    <thead><tr>
      <th>when</th><th>decision</th><th>model</th><th>hotkey</th><th>block</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 40).map((r) => {
      const cid = r.challenge_id || "";
      return `<tr class="${cid ? "row-link" : ""}" ${cid ? `data-cid="${esc(cid)}"` : ""}>
        <td class="when">${esc(fmtTime(r.at))}</td>
        <td>${intakeBadge(r.decision)}</td>
        <td>${modelLink(r.repo, r.hotkey)}</td>
        <td>${hotkeyLink(r.hotkey)}</td>
        <td class="dim">${esc(r.block ?? "—")}</td>
        <td class="dim">${esc(short(r.detail || r.challenge_id || "—", 64))}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

function renderQueue(d) {
  const q = d?.queue || [];
  const ce = d?.current_eval;
  const pending = q.length;
  const bits = [];
  if (ce) bits.push(`evaluating ${ce.challenge_id || ""}`.trim());
  bits.push(pending ? `${pending} pending` : "idle");
  $("queue-meta").textContent = bits.join(" · ");
  if (!q.length && !ce) {
    $("queue-wrap").innerHTML = `<div class="empty">empty — commits/reveals show under intake, not here</div>`;
    return;
  }
  const rows = [];
  if (ce) {
    rows.push({
      status: "evaluating", id: ce.challenge_id, repo: ce.repo,
      hotkey: ce.hotkey || "", queued: "now", retries: "—",
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
    <tbody>${rows.map((r) => `<tr class="${r.status === "evaluating" ? "current" : ""}">
        <td>${badge(r.status, r.status)}</td>
        <td>${esc(short(r.id, 14))}</td>
        <td>${modelLink(r.repo, r.hotkey)}</td>
        <td>${hotkeyLink(r.hotkey)}</td>
        <td class="when">${esc(r.queued)}</td>
        <td class="r">${esc(r.retries)}</td>
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
    .filter((r) => {
      if (!filter) return true;
      const hay = [r.event, r.repo, r.hotkey, r.error_code,
        r.rejection_reason, r.challenge_id].join(" ").toLowerCase();
      return hay.includes(filter);
    });
  $("history-meta").textContent = `${rows.length} shown`;
  if (!rows.length) {
    $("history-wrap").innerHTML = `<div class="empty">empty</div>`;
    return;
  }
  $("history-wrap").innerHTML = `<table class="data-table">
    <thead><tr>
      <th>when</th><th>event</th><th>model</th><th>hotkey</th><th>outcome</th>
      <th class="r">dur</th><th class="r">z</th><th class="r">S*</th><th class="r">king S*</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 80).map((r) => {
      const zClass = r.z == null ? "" : Number(r.z) >= 0 ? "ok" : "bad";
      const cid = r.challenge_id || "";
      return `<tr class="row-link ${r.event === "crowned" ? "current" : ""}" data-cid="${esc(cid)}">
        <td class="when">${esc(fmtTime(r.at))}</td>
        <td>${esc(r.event)}</td>
        <td>${modelLink(r.repo, r.hotkey, r.reign_number)}</td>
        <td>${hotkeyLink(r.hotkey)}</td>
        <td>${outcomeBadge(r)}</td>
        <td class="r dim">${esc(fmtDuration(r.duration_s))}</td>
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
  const rows = fails.filter((r) => {
    if (!filter) return true;
    const hay = [r.event, r.repo, r.hotkey, r.error_code,
      r.rejection_reason, r.challenge_id].join(" ").toLowerCase();
    return hay.includes(filter);
  });
  $("fails-meta").textContent = `${rows.length} shown`;
  if (!rows.length) {
    $("fails-wrap").innerHTML = `<div class="empty">none</div>`;
    return;
  }
  $("fails-wrap").innerHTML = `<table class="data-table">
    <thead><tr>
      <th>when</th><th>uid</th><th>model</th><th>hotkey</th><th class="r">dur</th><th>code</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 60).map((r) => {
      const cid = r.challenge_id || "";
      return `<tr class="row-link" data-cid="${esc(cid)}">
        <td class="when">${esc(fmtTime(r.at))}</td>
        <td class="dim">${r.uid != null ? esc(r.uid) : "—"}</td>
        <td>${modelLink(r.repo, r.hotkey)}</td>
        <td>${hotkeyLink(r.hotkey)}</td>
        <td class="r dim">${esc(fmtDuration(r.duration_s))}</td>
        <td class="bad">${esc(r.error_code || r.rejection_reason || "reject")}</td>
        <td class="dim">${esc(short(r.error_detail || r.rejection_reason || "—", 64))}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

function renderSnapshotSections() {
  const d = cache.dashboard;
  if (!d) return;
  renderReign(d);
  renderIntake(d);
  renderQueue(d);
}

function renderAll() {
  renderHero();
  renderSnapshotSections();
  renderGates();
  renderHistory(cache.history);
  renderFails(cache.history);
}

/* ---------- duel detail panel ---------- */

function drawSeriesScatter(svg, series) {
  const pts = (series?.challenger || []).filter(
    (p) => p.lambda2 != null && p.l1lift != null);
  const width = Math.min(560, (window.innerWidth || 600) - 48);
  const height = 240;
  const pad = { l: 44, r: 16, t: 16, b: 36 };
  const mono = "IBM Plex Mono, monospace";
  if (!pts.length) {
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle"
      fill="rgba(229,229,229,0.35)" font-family="${mono}" font-size="11">no pair series for this duel</text>`;
    return;
  }
  const xs = pts.map((p) => Number(p.lambda2));
  const ys = pts.map((p) => Number(p.l1lift));
  let x0 = Math.min(...xs, 0);
  let x1 = Math.max(...xs, 0);
  let y0 = Math.min(...ys, 0);
  let y1 = Math.max(...ys, 0);
  const xpad = (x1 - x0) * 0.12 || 0.05;
  const ypad = (y1 - y0) * 0.12 || 0.05;
  x0 -= xpad; x1 += xpad; y0 -= ypad; y1 += ypad;
  const xAt = (v) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (width - pad.l - pad.r);
  const yAt = (v) => pad.t + ((y1 - v) / (y1 - y0 || 1)) * (height - pad.t - pad.b);
  const axes = `
    <line x1="${pad.l}" x2="${width - pad.r}" y1="${yAt(0)}" y2="${yAt(0)}"
      stroke="rgba(255,255,255,0.12)"/>
    <line x1="${xAt(0)}" x2="${xAt(0)}" y1="${pad.t}" y2="${height - pad.b}"
      stroke="rgba(255,255,255,0.12)"/>
    <text x="${width / 2}" y="${height - 8}" text-anchor="middle" fill="rgba(229,229,229,0.4)"
      font-family="${mono}" font-size="10">Λ2</text>
    <text x="12" y="${height / 2}" fill="rgba(229,229,229,0.4)"
      font-family="${mono}" font-size="10" transform="rotate(-90 12 ${height / 2})">L1lift</text>`;
  const dots = pts.map((p) => {
    const fill = p.gate_ok ? "#5ac8fa" : "rgba(255,71,71,0.7)";
    return `<circle cx="${xAt(p.lambda2)}" cy="${yAt(p.l1lift)}" r="3.5" fill="${fill}" opacity="0.85">
      <title>Λ2=${fmtScore(p.lambda2)} L1=${fmtScore(p.l1lift)} gate=${p.gate_ok}</title>
    </circle>`;
  }).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.innerHTML = axes + dots;
}

function failureDetail(duel) {
  const f = duel?.failure || {};
  return {
    code: f.code || duel?.error_code || duel?.rejection_reason || duel?.event || "failed",
    detail: f.detail || duel?.error_detail || duel?.rejection_reason || "",
    at: f.at || duel?.at,
    repo: f.repo || duel?.repo,
    hotkey: f.hotkey || duel?.hotkey,
    revision: f.revision || duel?.revision,
  };
}

function isFailureDuel(duel) {
  return duel?.event === "failed"
    || duel?.accepted === false
    || Boolean(duel?.error_code)
    || Boolean(duel?.failure);
}

async function openDuel(challengeId) {
  if (!challengeId) return;
  const panel = $("duel-panel");
  const body = $("duel-panel-body");
  const title = $("duel-panel-title");
  panel.hidden = false;
  title.textContent = challengeId;
  body.innerHTML = `<div class="empty">loading…</div>`;
  const [duel, series] = await Promise.all([
    fetchDuel(challengeId),
    fetchDuelSeries(challengeId).catch(() => null),
  ]);
  if (!duel || duel.error) {
    // Fall back to the slim history row so fails still open with detail.
    const row = (cache.history || []).find((r) => r.challenge_id === challengeId);
    if (!row) {
      body.innerHTML = `<div class="empty">no detail for ${esc(challengeId)}</div>`;
      return;
    }
    renderDuelBody(body, row, null);
    return;
  }
  renderDuelBody(body, duel, series && !series.error ? series : null);
}

function renderDuelBody(body, duel, series) {
  const fail = isFailureDuel(duel);
  const info = failureDetail(duel);
  const ch = duel.challenger || {};
  const kg = duel.king || {};
  const gates = duel.gates || {};
  if (fail && !duel.has_series && duel.z == null && duel.score == null) {
    body.innerHTML = `
      <div class="kv-grid">
        <div class="kv"><span class="k">when</span><span class="v">${esc(fmtTime(info.at))}</span></div>
        <div class="kv"><span class="k">code</span><span class="v bad">${esc(info.code)}</span></div>
        <div class="kv"><span class="k">uid</span><span class="v">${duel.uid != null ? esc(duel.uid) : "—"}</span></div>
        <div class="kv"><span class="k">duration</span><span class="v">${esc(fmtDuration(duel.duration_s))}</span></div>
        <div class="kv"><span class="k">model</span><span class="v">${modelLink(info.repo || duel.repo, info.hotkey || duel.hotkey, duel.reign_number)}</span></div>
        <div class="kv"><span class="k">revision</span><span class="v mono">${esc(info.revision || "—")}</span></div>
        <div class="kv"><span class="k">hotkey</span><span class="v">${hotkeyLink(info.hotkey || duel.hotkey)}</span></div>
        <div class="kv"><span class="k">challenge</span><span class="v mono">${esc(duel.challenge_id || "—")}</span></div>
      </div>
      <div class="fail-log-block">
        <div class="section-head">
          <h3 class="section-title">failure detail</h3>
          <span class="section-right note">validator log</span>
        </div>
        <pre class="fail-log">${esc(info.detail || "no detail recorded")}</pre>
      </div>`;
    return;
  }

  const outcome = duel.event === "crowned"
    ? `crowned #${duel.reign_number ?? "?"}`
    : (info.code || duel.event || "—");
  body.innerHTML = `
    <div class="kv-grid">
      <div class="kv"><span class="k">model</span><span class="v">${modelLink(duel.repo || info.repo, duel.hotkey || info.hotkey, duel.reign_number)}</span></div>
      <div class="kv"><span class="k">hotkey</span><span class="v">${hotkeyLink(duel.hotkey || info.hotkey)}</span></div>
      <div class="kv"><span class="k">uid</span><span class="v">${duel.uid != null ? esc(duel.uid) : "—"}</span></div>
      <div class="kv"><span class="k">duration</span><span class="v">${esc(fmtDuration(duel.duration_s))}</span></div>
      <div class="kv"><span class="k">outcome</span><span class="v ${fail ? "bad" : ""}">${esc(outcome)}</span></div>
      <div class="kv"><span class="k">z</span><span class="v ${Number(duel.z) >= 0 ? "ok" : "bad"}">${esc(fmtZ(duel.z))}</span></div>
      <div class="kv"><span class="k">margin</span><span class="v">${esc(fmtScore(duel.margin))} · se ${esc(fmtScore(duel.se))}</span></div>
      <div class="kv"><span class="k">S* chall</span><span class="v">${esc(fmtScore(duel.score ?? ch.S))}</span></div>
      <div class="kv"><span class="k">S* king</span><span class="v">${esc(fmtScore(duel.score_king ?? kg.S))}</span></div>
      <div class="kv"><span class="k">gate pass</span><span class="v">${esc(fmtScore(ch.gate_pass_rate))} / bank ${esc(fmtScore(ch.bank_frac))}</span></div>
      <div class="kv"><span class="k">thresholds</span><span class="v dim">kσ=${esc(gates.k_sigma ?? 3)} · δ=${esc(gates.min_margin ?? "—")}</span></div>
      ${duel.has_artifact && duel.challenge_id ? `
      <div class="kv"><span class="k">artifact</span><span class="v"><a href="${esc(hippiusEvalUrl(duel.challenge_id))}" target="_blank" rel="noopener" title="full duel record on Hippius (rollouts, teacher refs, logprobs)">hippius · evals/${esc(short(duel.challenge_id, 14))}.json.gz</a></span></div>` : ""}
    </div>
    ${fail && info.detail ? `
      <div class="fail-log-block">
        <div class="section-head">
          <h3 class="section-title">failure detail</h3>
          <span class="section-right note">validator log</span>
        </div>
        <pre class="fail-log">${esc(info.detail)}</pre>
      </div>` : ""}
    <div class="duel-chart-block">
      <div class="section-head"><h3 class="section-title">Λ2 vs L1lift</h3>
        <span class="section-right note">blue = gate ok · red = gate fail</span></div>
      <svg id="duel-series-chart" role="img" aria-label="pair series"></svg>
    </div>`;
  const svg = $("duel-series-chart");
  if (svg) drawSeriesScatter(svg, series);
}

function closeDuel() {
  $("duel-panel").hidden = true;
}

/* ---------- data wiring ---------- */

function applySnapshot(snap) {
  if (!snap) return;
  const fp = fingerprint({
    generated_at: snap.generated_at,
    phase: snap.phase,
    current_eval: snap.current_eval,
    king: snap.king,
    queue: snap.queue,
    intake: snap.intake,
    stats: snap.stats,
    reign: snap.reign,
    market: snap.market,
  });
  if (fp === fps.dashboard) return;
  fps.dashboard = fp;
  cache.dashboard = snap;
  setReignLookup(reignMembers(snap), BENCH_GENESIS.repo);
  renderMarketBar(snap);
  renderHero();
  renderSnapshotSections();
  // Display names depend on the reign lookup — refresh the history tables so
  // king rows rendered before the first snapshot pick up their roman names.
  renderHistory(cache.history);
  renderFails(cache.history);
}

async function refreshHistoryAndBench() {
  const [h, b, reg] = await Promise.all([
    fetchHistory({ limit: 100, q: filter }),
    fetchBenchmarks(),
    fetchRegHistory(),
  ]);
  const hfp = fingerprint(h);
  if (hfp !== fps.history) {
    fps.history = hfp;
    cache.history = h;
    fps.hero = "";
    renderHero(true);
    renderGates(true);
    renderHistory(cache.history);
    renderFails(cache.history);
  }
  const bfp = fingerprint(b);
  if (b && bfp !== fps.benchmarks) {
    fps.benchmarks = bfp;
    cache.benchmarks = b;
    // Bench scores render inside the reign table now.
    if (cache.dashboard) renderReign(cache.dashboard);
  }
  if (reg?.points?.length) {
    const rfp = fingerprint({
      updated_at: reg.updated_at,
      n: reg.points.length,
      last: reg.points[reg.points.length - 1],
    });
    if (rfp !== fps.reg) {
      fps.reg = rfp;
      cache.regHistory = reg;
      renderRegPrice(true);
    }
  }
}

function wire() {
  $("filter-input")?.addEventListener("input", (e) => {
    filter = e.target.value.trim().toLowerCase();
    refreshHistoryAndBench();
  });
  window.addEventListener("resize", () => {
    fps.hero = "";
    renderHero(true);
    renderGates(true);
    renderRegPrice(true);
  });
  $("intake-wrap")?.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-cid]");
    if (!tr || e.target.closest("a")) return;
    openDuel(tr.dataset.cid);
  });
  $("history-wrap")?.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-cid]");
    if (!tr || e.target.closest("a")) return;
    openDuel(tr.dataset.cid);
  });
  $("fails-wrap")?.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-cid]");
    if (!tr || e.target.closest("a")) return;
    openDuel(tr.dataset.cid);
  });
  $("duel-panel-close")?.addEventListener("click", closeDuel);
  $("duel-panel-backdrop")?.addEventListener("click", closeDuel);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDuel();
  });
}

async function boot() {
  wire();
  await refreshHistoryAndBench();
  closeWatch = watchSnapshot(applySnapshot, {
    onStatus: (s) => {
      const el = $("live-status");
      if (el) el.textContent = s;
    },
  });
  // History grows slower than live snapshot — refresh on an interval.
  setInterval(refreshHistoryAndBench, 15000);
}

boot();
