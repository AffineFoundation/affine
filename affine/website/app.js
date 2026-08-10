import {
  currentMode,
  duelTurnUrl,
  fetchBenchmarks,
  fetchDuel,
  fetchDuelLog,
  fetchDuelSeries,
  fetchDuelTurn,
  fetchHistory,
  fetchRegHistory,
  fingerprint,
  watchSnapshot,
} from "./api.js?v=44";
import {
  GATE_METRICS,
  HERO_CHARTS,
  drawDeltaBars,
  drawDuelScores,
  drawDuelZ,
  drawGateMetric,
  drawPairScatter,
  drawRegPrice,
  drawSideScatter,
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
} from "./charts.js?v=44";

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
  const marginSvg = $("hero-chart-margin");
  if (!scoreSvg || !marginSvg) return;
  const d = cache.dashboard;
  // Market bar is independent of chart dirty-checks.
  renderMarketBar(d);

  const key = `${fps.history}|${fps.dashboard}|${chartWidth()}`;
  if (!force && key === fps.hero) return;
  fps.hero = key;

  drawDuelScores(scoreSvg, cache.history, { width: paneWidth(scoreSvg) });
  drawDuelZ(marginSvg, cache.history,
    { width: paneWidth(marginSvg), height: 320 });
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
  el.innerHTML = [
    `<span class="market-item"><span class="k">SN</span><b>120</b></span>`,
    `<span class="market-item"><span class="k">price</span><b class="gold">${esc(fmtTao(market.price_tao, 4))}</b></span>`,
    `<span class="market-item"><span class="k">reg</span><b>${esc(fmtTao(market.reg_cost_tao, 3))}</b></span>`,
    `<span class="market-item" title="${esc(weightsTitle)}"><span class="k">weights</span><b>${esc(fmtAge(market.weights_committed_at))}</b></span>`,
    market.block_number != null
      ? `<span class="market-item"><span class="k">block</span><b>${esc(market.block_number)}</b></span>`
      : "",
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
      <th class="r">Reason</th><th class="r">α/day</th><th class="r">$/day</th><th class="r">weight</th>
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
  const pane = svg.closest(".metric-pane");
  drawRegPrice(svg, hist, {
    width: Math.max((pane?.clientWidth || 320) - 20, 280),
    height: 320,
  });
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
      ? `${points.length} scored duels · measured, not scored · challenger gold · king bone`
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
          <div class="metric-head-row">
            <span class="metric-title">${esc(m.title)}</span>
            <button type="button" class="expand-btn" data-chart="${esc(m.id)}"
              title="expand ${esc(m.title)}"
              aria-label="Expand ${esc(m.title)} chart">⤢</button>
          </div>
          <span class="metric-caption">${esc(m.caption)}</span>
        </div>
        <svg role="img" aria-label="${esc(m.title)} per duel"></svg>
      </div>`).join("") + `
      <div class="metric-pane" id="metric-reg-price">
        <div class="metric-head">
          <div class="metric-head-row">
            <span class="metric-title">registration cost</span>
            <button type="button" class="expand-btn" data-chart="reg-price"
              title="expand registration cost"
              aria-label="Expand registration cost chart">⤢</button>
          </div>
          <span class="metric-caption" id="reg-price-meta">tmc burn history</span>
        </div>
        <svg id="reg-price-chart" role="img"
          aria-label="SN120 registration burn history"></svg>
      </div>`;
    renderRegPrice(true);
  }
  for (const m of GATE_METRICS) {
    const pane = $(`metric-${m.id}`);
    const svg = pane?.querySelector("svg");
    if (!svg) continue;
    drawGateMetric(svg, points, m, {
      width: Math.max((pane.clientWidth || 320) - 20, 280),
      height: 320,
    });
  }
}

/* ---------- expanded chart ---------- */

// Every chart on the page, keyed by the data-chart on its expand button.
/** Series of the duel page currently on screen (for expanded chart modals). */
const duelPage = { cid: null, series: null };

const DUEL_CHART_SPECS = [
  {
    id: "duel-delta",
    title: "ΔReason per turn",
    caption: "challenger − king, slice order · gold = challenger won the turn",
    detail: `<p>Each bar is one turn of the seeded slice:
      <code>ΔReason = Reason_challenger − Reason_king</code> on that turn,
      averaged over the turn's scored pairs. Gold bars are turns the
      challenger won, red bars turns the king won.</p>
      <p>The dashed line is the mean of these bars — exactly the duel's
      <code>margin</code>. The crown rule asks that this mean clear
      <code>k·SE</code> of its own spread, so a duel is won by consistent
      per-turn advantage, not a few outlier turns.</p>`,
    render: (svg, width) =>
      drawDeltaBars(svg, duelPage.series?.paired || [], { width, height: 440 }),
  },
  {
    id: "duel-sides",
    title: "challenger vs king",
    caption: "per-turn Reason · above the dashed diagonal = challenger better",
    detail: `<p>Every dot is one paired turn, placed at (king Reason,
      challenger Reason). The dashed diagonal is parity: dots above it are
      turns where the challenger's score beat the king's on the same prompt
      with the same teacher references.</p>
      <p>This view separates "the challenger is better" (cloud shifted above
      the diagonal) from "the turns are just easier/harder" (cloud sliding
      along the diagonal): only distance from the diagonal wins duels.</p>`,
    render: (svg, width) =>
      drawSideScatter(svg, duelPage.series?.paired || [], { width, height: 440 }),
  },
  {
    id: "duel-pair",
    title: "Reason vs L1lift",
    caption: "gold = challenger · bone = king · red ring = causality fail",
    detail: `<p>The score against its telemetry twin, per turn.
      <code>Reason = lpC(y_C|z_A) − lpC(y_C|∅)</code> is the teacher-side lift
      the miner's thought gives the teacher's own reference action — the whole
      score. <code>L1lift = lpA(y_C|z_A) − lpA(y_C|∅)</code> is the same lift
      measured on the miner's own logprobs — recorded, not scored.</p>
      <p>Gold dots are challenger turns, bone dots king turns; a red ring
      marks turns whose pairs failed the (telemetry) causality/leakage check.
      A faithful distill drifts up and to the right of its opponent.</p>`,
    render: (svg, width) =>
      drawPairScatter(svg, duelPage.series, { width, height: 440 }),
  },
];

const CHART_SPECS = new Map([
  ...DUEL_CHART_SPECS.map((c) => [c.id, c]),
  ...HERO_CHARTS.map((c) => [c.id, {
    ...c,
    render: (svg, width) => c.draw(svg, cache.history, { width, height: 460 }),
  }]),
  ...GATE_METRICS.map((m) => [m.id, {
    ...m,
    render: (svg, width) =>
      drawGateMetric(svg, gatePoints(cache.history), m, { width, height: 420 }),
  }]),
  ["reg-price", {
    id: "reg-price",
    title: "registration cost",
    caption: "SN120 registration burn (τ) · TMC history",
    detail: `<p>The TAO burned to register a hotkey on SN120 over time, sourced
      from TaoMarketCap and downsampled by affine-dash. Registration cost
      doubles on each registration and decays toward a floor between them, so
      spikes are registration bursts — miners piling in — and the long decays
      are quiet periods.</p>
      <p>It matters to miners because a duel slot requires a registered hotkey:
      reg cost is effectively the price of a seat at the table. A rising curve
      means the subnet is contested right now; a decayed one means entry is
      cheap.</p>`,
    render: (svg, width) => drawRegPrice(svg, cache.regHistory, { width, height: 460 }),
  }],
]);

let openChartId = null;

function drawOpenChart() {
  const spec = CHART_SPECS.get(openChartId);
  const svg = $("chart-modal-svg");
  if (!spec || !svg) return;
  const host = svg.parentElement;
  spec.render(svg, Math.max((host?.clientWidth || 900) - 2, 320));
}

function openChart(id) {
  const spec = CHART_SPECS.get(id);
  const modal = $("chart-modal");
  if (!spec || !modal) return;
  openChartId = id;
  modal.hidden = false;
  $("chart-modal-title").textContent = spec.title;
  $("chart-modal-caption").textContent = spec.caption || "";
  // detail is authored markup from charts.js, never user input.
  $("chart-modal-detail").innerHTML = spec.detail || "";
  drawOpenChart();
}

function closeChart() {
  openChartId = null;
  const modal = $("chart-modal");
  if (modal) modal.hidden = true;
}

/** Instant cursor-following readout for chart marks (native <title> is too slow). */
function wireChartTip() {
  const tip = document.createElement("div");
  tip.id = "chart-tip";
  tip.hidden = true;
  document.body.appendChild(tip);
  document.addEventListener("mousemove", (e) => {
    const g = e.target instanceof Element && e.target.closest(".duel-hit[data-tip]");
    if (!g) {
      tip.hidden = true;
      return;
    }
    tip.textContent = g.dataset.tip;
    tip.hidden = false;
    const pad = 14;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }, { passive: true });
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
      <th>when</th><th>age</th><th>event</th><th>model</th><th>hotkey</th><th>outcome</th>
      <th class="r">dur</th><th class="r">z</th><th class="r">Reason</th><th class="r">king Reason</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 80).map((r) => {
      const zClass = r.z == null ? "" : Number(r.z) >= 0 ? "ok" : "bad";
      const cid = r.challenge_id || "";
      return `<tr class="row-link ${r.event === "crowned" ? "current" : ""}" data-cid="${esc(cid)}">
        <td class="when">${esc(fmtTime(r.at))}</td>
        <td class="dim" title="${esc(fmtTime(r.at))}">${r.at ? `${esc(fmtAge(r.at))} ago` : "—"}</td>
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
      <th>when</th><th>age</th><th>uid</th><th>model</th><th>hotkey</th><th class="r">dur</th><th>code</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 60).map((r) => {
      const cid = r.challenge_id || "";
      return `<tr class="row-link" data-cid="${esc(cid)}">
        <td class="when">${esc(fmtTime(r.at))}</td>
        <td class="dim" title="${esc(fmtTime(r.at))}">${r.at ? `${esc(fmtAge(r.at))} ago` : "—"}</td>
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

/* ---------- duel page (#duel/<challenge_id>) ---------- */

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

function openDuel(challengeId) {
  if (!challengeId) return;
  closeChart();
  location.hash = `duel/${encodeURIComponent(challengeId)}`;
}

function duelHashCid() {
  const m = location.hash.match(/^#duel\/(.+)$/);
  return m ? decodeURIComponent(m[1]) : null;
}

function closeDuelPage() {
  // Prefer real back so front page scroll position is restored.
  if (window.history.length > 1) window.history.back();
  else location.hash = "";
}

function route() {
  const cid = duelHashCid();
  const page = $("duel-page");
  if (!page) return;
  document.body.classList.toggle("duel-open", Boolean(cid));
  page.hidden = !cid;
  if (!cid) {
    page.dataset.cid = "";
    return;
  }
  window.scrollTo(0, 0);
  if (page.dataset.cid !== cid) {
    page.dataset.cid = cid;
    renderDuelPage(cid);
  }
}

const duelChartWidth = (el) =>
  Math.max((el?.closest(".duel-chart-pane")?.clientWidth || 560) - 26, 320);

function copyToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).catch(() => copyFallback(text));
    return;
  }
  copyFallback(text);
}

function copyFallback(text) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
}

/** Icon button that copies the FULL value even when the display truncates. */
function copyBtn(value) {
  if (value == null || value === "") return "";
  return `<button type="button" class="copy-btn" data-copy="${esc(String(value))}"
    title="copy" aria-label="Copy value">⧉</button>`;
}

async function renderDuelPage(cid) {
  const body = $("duel-page-body");
  const title = $("duel-page-title");
  if (!body) return;
  if (title) title.textContent = cid;
  body.innerHTML = `<div class="empty">loading duel record…</div>`;
  const [duelRaw, seriesRaw, logRaw] = await Promise.all([
    fetchDuel(cid).catch(() => null),
    fetchDuelSeries(cid).catch(() => null),
    fetchDuelLog(cid).catch(() => null),
  ]);
  if (duelHashCid() !== cid) return; // user navigated away mid-fetch
  let duel = duelRaw && !duelRaw.error ? duelRaw : null;
  if (!duel) {
    duel = (cache.history || []).find((r) => r.challenge_id === cid) || null;
  }
  if (!duel) {
    body.innerHTML = `<div class="empty">no record for ${esc(cid)} — it may have rotated out of the recent history window</div>`;
    // Allow a retry once history arrives (deep links land before first fetch).
    const page = $("duel-page");
    if (page) page.dataset.cid = "";
    return;
  }
  const series = seriesRaw && !seriesRaw.error ? seriesRaw : null;
  const logLines = (logRaw && !logRaw.error && Array.isArray(logRaw.lines))
    ? logRaw.lines : null;
  duelPage.cid = cid;
  duelPage.series = series;
  body.innerHTML = duelPageHtml(duel, series, logLines);
  const paired = series?.paired || [];
  const delta = $("duel-chart-delta");
  if (delta) drawDeltaBars(delta, paired, { width: duelChartWidth(delta) });
  const sides = $("duel-chart-sides");
  if (sides) drawSideScatter(sides, paired, { width: duelChartWidth(sides) });
  const pair = $("duel-chart-pair");
  if (pair) drawPairScatter(pair, series, { width: duelChartWidth(pair) });
}

/** Pre-fork verdicts stamp `gates`; Reason v3 verdicts stamp `duel_params`. */
function duelParams(duel) {
  return duel.duel_params || duel.gates || {};
}

function isPreFork(duel) {
  return Boolean(duel.gates) && !duel.duel_params;
}

function verdictSummary(duel) {
  const params = duelParams(duel);
  const k = Number(params.k_sigma ?? 3);
  // δ floor existed only pre-fork; v3 duels are purely relative.
  const delta = isPreFork(duel) && params.min_margin != null
    ? Number(params.min_margin) : null;
  const bar = duel.se != null ? k * Number(duel.se) : null;
  const name = modelDisplayName(duel.repo, duel.hotkey, duel.reign_number);
  if (isFailureDuel(duel) && duel.z == null && duel.margin == null) {
    return `${name} never reached a paired verdict — the duel failed with `
      + `“${failureDetail(duel).code}” before scoring completed.`;
  }
  const m = duel.margin != null ? fmtScore(duel.margin) : "—";
  const need = [
    bar != null ? `${k}·SE ≈ ${fmtScore(bar)}` : `${k}·SE`,
    delta != null ? `δ = ${fmtScore(delta)}` : null,
  ].filter(Boolean).join(" and ");
  if (duel.event === "crowned" || duel.challenger_wins) {
    return `${name} beat the king: paired Reason margin ${m} cleared ${need} `
      + `(z = ${fmtZ(duel.z)})${duel.event === "crowned" ? ` — crowned reign #${duel.reign_number ?? "?"}` : ""}.`;
  }
  if (duel.rejection_reason) {
    return `${name} was rejected before the margin test: ${duel.rejection_reason}.`;
  }
  return `${name} did not dethrone the king: paired Reason margin ${m} `
    + `(z = ${fmtZ(duel.z)}) needed to clear ${need}.`;
}

/** One side-by-side row of the challenger/king comparison table. */
function sideRow(label, tip, cv, kv, thr, cPass, kPass, fmt = fmtScore) {
  const cell = (v, pass) => {
    const cls = v == null ? "dim" : pass == null ? "" : pass ? "ok" : "bad";
    return `<td class="num ${cls}">${v == null ? "—" : esc(fmt(v))}</td>`;
  };
  return `<tr>
    <td class="mono" ${tip ? `title="${esc(tip)}"` : ""}>${esc(label)}</td>
    ${cell(cv, cPass)}${cell(kv, kPass)}
    <td class="dim">${thr ? esc(thr) : ""}</td>
  </tr>`;
}

function sidesTableHtml(duel) {
  const ch = duel.challenger || {};
  const kg = duel.king || {};
  const chReason = ch.reason ?? ch.mean_lambda2 ?? ch.S;
  const kgReason = kg.reason ?? kg.mean_lambda2 ?? kg.S;
  if (chReason == null && kgReason == null && duel.score == null) {
    return `<div class="empty">no per-side scores recorded for this duel</div>`;
  }
  const pct = (v) => `${(Number(v) * 100).toFixed(0)}%`;
  const pre = isPreFork(duel);
  const g = duel.gates || {};
  const inBand = (v, lo, hi) =>
    v == null || lo == null ? null : Number(v) >= Number(lo) && Number(v) <= Number(hi);
  const gte = (v, t) => (v == null || t == null ? null : Number(v) >= Number(t));
  const baselineOk = ch.baseline_abs != null && kg.baseline_abs && g.baseline_band != null
    ? Number(ch.baseline_abs) <= Number(g.baseline_band) * Number(kg.baseline_abs)
    : null;
  // Pre-fork rows keep their pass/fail coloring and gate thresholds; Reason v3
  // rows render the same quantities as plain telemetry.
  const rows = [
    ...(pre ? [
      sideRow("valid", "all S* v2 gates passed for this side", ch.valid, kg.valid,
        "must be true", ch.valid === true, kg.valid === true,
        (v) => (v ? "yes" : "NO")),
      sideRow("S* (legacy mix)", "retired ranking score: mean(Λ2 + clip(L1lift, ±0.1))",
        duel.score ?? ch.S, duel.score_king ?? kg.S, "higher wins"),
    ] : []),
    sideRow("Reason", "the score: mean lpC(y_C|z_A) − lpC(y_C|∅)",
      chReason, kgReason, "higher wins"),
    sideRow("L1lift", "miner-side lift lpA(y_C|z_A) − lpA(y_C|∅)",
      ch.mean_l1lift, kg.mean_l1lift, pre ? "" : "telemetry"),
    sideRow("causality pass", "share of pairs passing leakage + causality",
      ch.gate_pass_rate, kg.gate_pass_rate,
      pre ? (g.gamma != null ? `≥ ${g.gamma}` : "≥ γ") : "telemetry",
      pre ? gte(ch.gate_pass_rate, g.gamma) : null,
      pre ? gte(kg.gate_pass_rate, g.gamma) : null, pct),
    sideRow("prior bank", "share of pairs beating the published prior thoughts",
      ch.bank_frac, kg.bank_frac,
      pre ? (g.gamma_bank != null ? `≥ ${g.gamma_bank}` : "≥ γ_bank") : "telemetry",
      pre ? gte(ch.bank_frac, g.gamma_bank) : null,
      pre ? gte(kg.bank_frac, g.gamma_bank) : null, pct),
    sideRow("calibration r", "mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)|",
      ch.calib_ratio, kg.calib_ratio,
      pre ? (g.r_lo != null ? `${g.r_lo} – ${g.r_hi}` : "in band") : "telemetry",
      pre ? inBand(ch.calib_ratio, g.r_lo, g.r_hi) : null,
      pre ? inBand(kg.calib_ratio, g.r_lo, g.r_hi) : null),
    sideRow("empty baseline", "mean|lpA(y_C|∅)|",
      ch.baseline_abs, kg.baseline_abs,
      pre ? (g.baseline_band != null ? `chall ≤ ${g.baseline_band}× king` : "banded")
        : "telemetry",
      pre ? baselineOk : null, null),
    sideRow("thought chars", "mean length of z_A (chars)",
      ch.mean_len_z, kg.mean_len_z, "telemetry", null, null,
      (v) => String(Math.round(Number(v)))),
    sideRow("action chars", "mean length of y_A (chars)",
      ch.mean_len_y, kg.mean_len_y, "telemetry", null, null,
      (v) => String(Math.round(Number(v)))),
    sideRow("Δ len vs teacher", "thought-length delta miner − teacher (chars)",
      ch.len_z_delta, kg.len_z_delta, "telemetry", null, null,
      (v) => String(Math.round(Number(v)))),
    sideRow("turns scored", "", ch.n_turns, kg.n_turns, "", null, null, String),
    sideRow("pairs scored", "", ch.n_pairs, kg.n_pairs, "", null, null, String),
  ].join("");
  return `<div class="table-wrap"><table class="data-table sides-table">
    <thead><tr><th>metric</th><th class="num">challenger</th>
      <th class="num">king</th><th>${pre ? "gate" : "role"}</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

const TURNS_COLSPAN = 9;

function turnsTableHtml(duel, paired) {
  if (!paired.length) {
    return `<div class="empty">no per-turn series — the eval artifact is not published (older duel or failure before scoring)</div>`;
  }
  const cid = duel.challenge_id || "";
  const dOf = (p) => p.delta_reason ?? p.delta_mix;
  const sorted = [...paired].sort(
    (a, b) => (dOf(b) ?? -Infinity) - (dOf(a) ?? -Infinity));
  const rows = sorted.map((p) => {
    const d = dOf(p);
    const tid = p.turn_id ?? "";
    return `<tr class="turn-row" data-turn="${esc(tid)}"
        title="click to view the rollout texts for this turn">
      <td class="mono">${esc(short(tid || "—", 44))}</td>
      <td class="num ${d == null ? "dim" : d >= 0 ? "ok" : "bad"}">${d == null ? "—" : esc(fmtScore(d))}</td>
      <td class="num">${esc(fmtScore(p.challenger_reason ?? p.challenger_mix))}</td>
      <td class="num">${esc(fmtScore(p.king_reason ?? p.king_mix))}</td>
      <td class="num">${esc(fmtScore(p.challenger_l1lift))}</td>
      <td class="num">${esc(fmtScore(p.king_l1lift))}</td>
      <td class="${p.challenger_gate_ok ? "ok" : "bad"}">${p.challenger_gate_ok ? "ok" : "fail"}</td>
      <td class="${p.king_gate_ok ? "ok" : "bad"}">${p.king_gate_ok ? "ok" : "fail"}</td>
      <td>${cid && tid ? `<a class="raw-link" href="${esc(duelTurnUrl(cid, tid))}"
        target="_blank" rel="noopener" title="raw JSON for this exact turn">json ↗</a>` : ""}</td>
    </tr>`;
  }).join("");
  return `<div class="table-wrap table-scroll"><table class="data-table turns-table">
    <thead><tr><th>turn</th><th class="num">ΔReason</th>
      <th class="num">chall Reason</th><th class="num">king Reason</th>
      <th class="num">chall L1 ·t</th><th class="num">king L1 ·t</th>
      <th>chall caus ·t</th><th>king caus ·t</th><th>raw</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

/* ---------- inline rollout viewer (one turn's actual texts) ---------- */

function rolloutPairHtml(p, i) {
  const chips = `
    <span class="chip">Reason ${esc(fmtScore(p.reason ?? p.lambda2))}</span>
    <span class="chip">L1 ${esc(fmtScore(p.l1lift))} ·t</span>
    <span class="chip ${p.gate_ok ? "ok" : "bad"}">caus ${p.gate_ok ? "ok" : "fail"} ·t</span>`;
  return `<div class="rollout-pair">
    <div class="rollout-pair-head"><span class="dim">pair ${i + 1}</span>${chips}</div>
    <div class="rollout-text"><span class="k">thought z_A</span>
      <pre>${esc(p.thought ?? "—")}</pre></div>
    <div class="rollout-text"><span class="k">action y_A</span>
      <pre>${esc(p.action ?? "—")}</pre></div>
  </div>`;
}

function rolloutDetailHtml(detail) {
  const side = (label, cls, s) => !s ? "" : `
    <div class="rollout-side">
      <div class="rollout-side-head ${cls}">${esc(label)}
        <span class="dim">· ${s.valid ? "valid" : "INVALID"} · bank ${esc(fmtScore(s.bank_frac))} · ${esc(String(s.n_pairs ?? (s.pairs || []).length))} pairs</span></div>
      ${(s.pairs || []).map(rolloutPairHtml).join("")}
    </div>`;
  const refs = (detail.teacher_refs || []).map((r, i) => `
    <div class="rollout-pair">
      <div class="rollout-pair-head"><span class="dim">reference ${i + 1}</span>
        <span class="chip">lp own ${esc(fmtScore(r.lp_own))}</span>
        <span class="chip">lp ∅ ${esc(fmtScore(r.lp_empty))}</span></div>
      <div class="rollout-text"><span class="k">teacher thought z_C</span>
        <pre>${esc(r.thought ?? "—")}</pre></div>
      <div class="rollout-text"><span class="k">teacher action y_C</span>
        <pre>${esc(r.action ?? "—")}</pre></div>
    </div>`).join("");
  return `<div class="rollout-detail">
    ${side("challenger", "gold", detail.challenger)}
    ${side("king", "", detail.king)}
    ${refs ? `<div class="rollout-side">
      <div class="rollout-side-head dim">teacher references (the y_C both sides were scored against)</div>
      ${refs}</div>` : ""}
  </div>`;
}

async function toggleRolloutRow(tr) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains("rollout-row")) {
    next.remove();
    return;
  }
  const cid = duelPage.cid;
  const tid = tr.dataset.turn;
  if (!cid || !tid) return;
  const holder = document.createElement("tr");
  holder.className = "rollout-row";
  holder.innerHTML = `<td colspan="${TURNS_COLSPAN}"><div class="empty">loading rollout…</div></td>`;
  tr.after(holder);
  if (currentMode() !== "api") {
    holder.firstElementChild.innerHTML = `<div class="empty">rollout texts need the live API —
      in static mode download the full artifact:
      <a href="${esc(hippiusEvalUrl(cid))}" target="_blank" rel="noopener">evals/${esc(cid)}.json.gz</a></div>`;
    return;
  }
  const detail = await fetchDuelTurn(cid, tid).catch(() => null);
  if (!holder.isConnected) return;
  if (!detail || detail.error) {
    holder.firstElementChild.innerHTML = `<div class="empty">no rollout detail for this turn</div>`;
    return;
  }
  holder.firstElementChild.innerHTML = rolloutDetailHtml(detail);
}

function duelExplainHtml() {
  const specs = [
    ...HERO_CHARTS.map((c) => ({ title: c.title, caption: c.caption, detail: c.detail })),
    ...GATE_METRICS.map((m) => ({ title: m.title, caption: m.caption, detail: m.detail })),
  ];
  // detail is trusted static HTML from charts.js (same strings as the modal).
  return specs.map((s) => `
    <details class="explain-item">
      <summary><span class="mono">${esc(s.title)}</span>
        <span class="dim">· ${esc(s.caption)}</span></summary>
      <div class="explain-detail">${s.detail}</div>
    </details>`).join("");
}

function duelPageHtml(duel, series, logLines) {
  const fail = isFailureDuel(duel);
  const info = failureDetail(duel);
  const slice = duel.slice || series?.slice || {};
  const outcome = duel.event === "crowned"
    ? `crowned #${duel.reign_number ?? "?"}`
    : fail ? (info.code || "failed") : (duel.event || "—");
  const meta = $("duel-page-meta");
  if (meta) {
    // Lost-but-clean verdicts get the neutral badge — green would read as a win.
    meta.innerHTML = `<span class="badge ${duel.event === "crowned" ? "crowned" : fail ? "rejected" : duel.challenger_wins ? "accepted" : "queued"}">${esc(outcome)}</span>`;
  }
  const paired = series?.paired || [];
  const artifactLink = duel.challenge_id && (duel.has_artifact || paired.length)
    ? `<a href="${esc(hippiusEvalUrl(duel.challenge_id))}" target="_blank" rel="noopener">evals/${esc(short(duel.challenge_id, 18))}.json.gz</a>`
    : `<span class="dim">not published</span>`;

  const repo = duel.repo || info.repo;
  const hotkey = duel.hotkey || info.hotkey;
  const revision = duel.revision || info.revision;
  const overview = `
    <div class="kv-grid duel-overview">
      <div class="kv"><span class="k">challenger</span><span class="v">${modelLink(repo, hotkey, duel.reign_number)}${copyBtn(repo)}</span></div>
      <div class="kv"><span class="k">hotkey</span><span class="v">${hotkeyLink(hotkey)}${copyBtn(hotkey)}</span></div>
      <div class="kv"><span class="k">uid</span><span class="v">${duel.uid != null ? esc(duel.uid) : "—"}</span></div>
      <div class="kv"><span class="k">revision</span><span class="v mono">${esc(short(revision || "—", 14))}${copyBtn(revision)}</span></div>
      <div class="kv"><span class="k">challenge</span><span class="v mono">${esc(duel.challenge_id || "—")}${copyBtn(duel.challenge_id)}</span></div>
      <div class="kv"><span class="k">when</span><span class="v" title="${esc(fmtTime(duel.at))}">${esc(fmtTime(duel.at))} · ${esc(fmtAge(duel.at))}</span></div>
      <div class="kv"><span class="k">duration</span><span class="v">${esc(fmtDuration(duel.duration_s))}</span></div>
      <div class="kv"><span class="k">paired turns</span><span class="v">${esc(duel.n_paired_turns ?? paired.length ?? "—")}</span></div>
      <div class="kv"><span class="k">artifact</span><span class="v">${artifactLink}${duel.challenge_id ? copyBtn(hippiusEvalUrl(duel.challenge_id)) : ""}</span></div>
      ${slice.seed != null ? `<div class="kv"><span class="k">slice seed</span><span class="v mono" title="derived from the reveal block hash — miners cannot precompute the slice">${esc(slice.seed)}${copyBtn(slice.seed)}</span></div>` : ""}
      ${slice.block_hash ? `<div class="kv"><span class="k">block hash</span><span class="v mono" title="${esc(slice.block_hash)}">${esc(short(slice.block_hash, 18))}${copyBtn(slice.block_hash)}</span></div>` : ""}
      ${slice.digest ? `<div class="kv"><span class="k">corpus digest</span><span class="v mono" title="${esc(slice.digest)}">${esc(short(slice.digest, 18))}${copyBtn(slice.digest)}</span></div>` : ""}
    </div>`;

  const params = duelParams(duel);
  const pre = isPreFork(duel);
  const kSE = duel.se != null ? Number(params.k_sigma ?? 3) * Number(duel.se) : null;
  const marginOk = duel.margin != null && kSE != null
    ? Number(duel.margin) > kSE
      && (!pre || params.min_margin == null
          || Number(duel.margin) > Number(params.min_margin))
    : null;
  const verdict = `
    <p class="duel-verdict-line">${esc(verdictSummary(duel))}</p>
    <div class="kv-grid">
      <div class="kv"><span class="k">z</span><span class="v ${Number(duel.z) >= 0 ? "ok" : "bad"}">${esc(fmtZ(duel.z))}</span></div>
      <div class="kv"><span class="k">margin ⟨Reason_c − Reason_k⟩</span><span class="v ${marginOk == null ? "" : marginOk ? "ok" : "bad"}">${esc(fmtScore(duel.margin))}</span></div>
      <div class="kv"><span class="k">SE</span><span class="v">${esc(fmtScore(duel.se))}</span></div>
      <div class="kv"><span class="k">${esc(String(params.k_sigma ?? 3))}·SE bar</span><span class="v">${kSE == null ? "—" : esc(fmtScore(kSE))}</span></div>
      ${pre ? `<div class="kv"><span class="k">δ noise floor (legacy)</span><span class="v">${esc(fmtScore(params.min_margin))}</span></div>` : ""}
      ${duel.duel_seconds != null ? `<div class="kv"><span class="k">scoring time</span><span class="v">${esc(fmtDuration(duel.duel_seconds))}</span></div>` : ""}
      <div class="kv"><span class="k">challenger wins</span><span class="v ${duel.challenger_wins ? "ok" : "bad"}">${duel.challenger_wins == null ? "—" : duel.challenger_wins ? "yes" : "no"}</span></div>
    </div>`;

  const failBlock = fail && info.detail ? `
    <div class="duel-block">
      <div class="section-head"><h3 class="section-title">failure detail</h3>
        <span class="section-right note">${esc(info.code || "")}</span></div>
      <pre class="fail-log">${esc(info.detail)}</pre>
    </div>` : "";

  const chartPane = (spec, svgId, ariaLabel) => `
    <div class="duel-chart-pane">
      <div class="metric-head">
        <div class="metric-head-row">
          <span class="metric-title">${esc(spec.title)}</span>
          <button type="button" class="expand-btn" data-chart="${esc(spec.id)}"
            title="expand ${esc(spec.title)}"
            aria-label="Expand ${esc(spec.title)} chart">⤢</button>
        </div>
        <span class="metric-caption">${esc(spec.caption)}</span>
      </div>
      <svg id="${svgId}" role="img" aria-label="${ariaLabel}"></svg>
    </div>`;
  const charts = `
    <div class="duel-chart-grid">
      ${chartPane(DUEL_CHART_SPECS[0], "duel-chart-delta", "delta mix per turn")}
      ${chartPane(DUEL_CHART_SPECS[1], "duel-chart-sides", "challenger vs king mix")}
      ${chartPane(DUEL_CHART_SPECS[2], "duel-chart-pair", "reason vs l1lift")}
    </div>`;

  const logBlock = `
    <div class="duel-block">
      <div class="section-head"><h3 class="section-title">validator log</h3>
        <span class="section-right note">lines mentioning ${esc(duel.challenge_id || "this duel")} · redacted</span></div>
      ${logLines && logLines.length
        ? `<pre class="fail-log duel-log">${esc(logLines.join("\n"))}</pre>`
        : `<div class="empty">log not available${logLines ? " for this duel" : " in static mode"} — full log: <a href="https://s3.hippius.com/affine-sn120/data/validator_log.txt" target="_blank" rel="noopener">validator_log.txt</a></div>`}
    </div>`;

  const auditBlock = `
    <div class="duel-block">
      <div class="section-head"><h3 class="section-title">verify this result</h3>
        <span class="section-right note">every number on this page is recomputable from public data</span></div>
      <div class="audit-note">
        <ol>
          <li><strong>The slice cannot be cherry-picked.</strong> The 80 turns
            were selected by a seed derived from the finney block hash of the
            miner's reveal block${slice.block_hash ? ` (<code>${esc(short(slice.block_hash, 18))}</code>, shown above)` : ""} —
            it does not exist before the commitment lands on chain, so neither
            the miner nor the validator can precompute or steer the slice.</li>
          <li><strong>The corpus is content-addressed.</strong> The turn corpus
            is sha-pinned in <a href="/api/v1/contract" target="_blank" rel="noopener">the chain contract</a>
            (affine.toml) and mirrored on Hugging Face (Dataset link in the
            nav)${slice.digest ? ` — this duel's slice digest is <code>${esc(short(slice.digest, 18))}</code>` : ""}.
            Re-derive the slice from seed + corpus and you get these exact turn ids.</li>
          <li><strong>The raw evidence is published.</strong> The artifact
            (${artifactLink}) contains every scored pair: the miner's thought
            <code>z_A</code>, its action <code>y_A</code>, the teacher references
            <code>y_C</code>, and all logprob components — the same data the
            rollout rows above are rendered from.</li>
          <li><strong>The math is replayable.</strong> Teacher-force the pinned
            teacher (see the contract) over the published texts to reproduce
            the logprobs, then recompute
            <code>Reason = lpC(y_C|z_A) − lpC(y_C|∅)</code> and the paired
            <code>z = mean(Reason_c − Reason_k) / SE</code>. The exact scoring
            code ships in the repo — see <a href="/llms.txt" target="_blank" rel="noopener">llms.txt</a>
            → <code>code/affine/score.py</code>.</li>
          <li><strong>The verdict follows mechanically.</strong> Crown iff
            margin &gt; ${esc(String(params.k_sigma ?? 3))}·SE${pre
              ? ` (this pre-fork duel additionally required margin &gt; δ = ${esc(fmtScore(params.min_margin))} and every S* v2 gate)`
              : " — that is the whole rule"}. No judge, no discretion.</li>
        </ol>
      </div>
    </div>`;

  return `
    ${overview}
    <div class="duel-block">
      <div class="section-head"><h3 class="section-title">verdict</h3>
        <span class="section-right note">${pre
          ? `crown rule (pre-fork S* v2): z > ${esc(String(params.k_sigma ?? 3))} AND margin > δ, both sides gate-valid`
          : `crown rule: margin > ${esc(String(params.k_sigma ?? 3))}·SE — nothing else`}</span></div>
      ${verdict}
    </div>
    ${failBlock}
    <div class="duel-block">
      <div class="section-head"><h3 class="section-title">challenger vs king</h3>
        <span class="section-right note">${pre ? "per-side gates and scores" : "Reason (the score) + telemetry (·t = measured, not scored)"}</span></div>
      ${sidesTableHtml(duel)}
    </div>
    <div class="duel-block">
      <div class="section-head"><h3 class="section-title">samples</h3>
        <span class="section-right note">${esc(String(paired.length))} paired turns from the seeded slice</span></div>
      ${charts}
    </div>
    <div class="duel-block">
      <div class="section-head"><h3 class="section-title">rollouts</h3>
        <span class="section-right note">sorted by ΔReason · full thoughts/actions/logprobs in the artifact: ${artifactLink}</span></div>
      ${turnsTableHtml(duel, paired)}
    </div>
    ${logBlock}
    ${auditBlock}
    <div class="duel-block">
      <div class="section-head"><h3 class="section-title">how to read these numbers</h3>
        <span class="section-right note">same definitions as the charts on the front page</span></div>
      ${duelExplainHtml()}
    </div>`;
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
  const navKing = $("nav-king");
  if (navKing && snap.king?.repo) {
    navKing.textContent =
      modelDisplayName(snap.king.repo, snap.king.hotkey, snap.king.reign_number);
    navKing.href = hubUrl(snap.king.repo);
  }
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
    drawOpenChart();
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
    drawOpenChart();
  });
  document.addEventListener("click", (e) => {
    const copy = e.target.closest(".copy-btn");
    if (copy) {
      e.preventDefault();
      copyToClipboard(copy.dataset.copy || "");
      copy.textContent = "✓";
      copy.classList.add("copied");
      setTimeout(() => {
        copy.textContent = "⧉";
        copy.classList.remove("copied");
      }, 1200);
      return;
    }
    const btn = e.target.closest(".expand-btn");
    if (btn) {
      e.preventDefault();
      openChart(btn.dataset.chart);
      return;
    }
    const turn = e.target.closest("tr.turn-row");
    if (turn && !e.target.closest("a")) {
      toggleRolloutRow(turn);
      return;
    }
    // Any chart mark carrying a challenge id opens its duel page.
    const hit = e.target.closest(".duel-hit[data-cid]");
    if (hit) openDuel(hit.dataset.cid);
  });
  // Section TOC + collapsible lower sections. Charts drawn while a section
  // is display:none get a fallback width, so expanding the telemetry grid
  // forces a redraw at the real pane width.
  const expandSection = (id) => {
    const sec = document.getElementById(id);
    if (!sec || !sec.classList.contains("collapsed")) return;
    sec.classList.remove("collapsed");
    if (id === "gates") {
      renderGates(true);
      renderRegPrice(true);
    }
  };
  document.querySelectorAll(".toc-link").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const id = (a.getAttribute("href") || "").slice(1);
      if (duelHashCid()) location.hash = "";
      expandSection(id);
      setTimeout(() => {
        document.getElementById(id)
          ?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 60);
    });
  });
  document.querySelectorAll(".section.collapsible > .section-head").forEach((head) => {
    head.addEventListener("click", (e) => {
      if (e.target.closest("a, button, input")) return;
      const sec = head.parentElement;
      if (sec.classList.contains("collapsed")) expandSection(sec.id);
      else sec.classList.add("collapsed");
    });
  });
  $("chart-modal-close")?.addEventListener("click", closeChart);
  $("chart-modal-backdrop")?.addEventListener("click", closeChart);
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
  $("duel-back")?.addEventListener("click", closeDuelPage);
  window.addEventListener("hashchange", route);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (openChartId) closeChart();
    else if (duelHashCid()) closeDuelPage();
  });
}

async function boot() {
  wire();
  wireChartTip();
  route(); // deep link straight to a duel page (#duel/<cid>)
  await refreshHistoryAndBench();
  route(); // retry the duel render now that history is cached (static mode)
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
