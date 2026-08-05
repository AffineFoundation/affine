import {
  fetchBenchmarks,
  fetchDuel,
  fetchDuelSeries,
  fetchHistory,
  fetchRegHistory,
  fingerprint,
  watchSnapshot,
} from "./api.js?v=7";
import {
  drawBenchSuite,
  drawDuelScores,
  drawDuelZ,
  drawRegPrice,
  drawReignChain,
  esc,
  fmtAge,
  fmtScore,
  fmtTao,
  fmtTime,
  fmtZ,
  reignMembers,
  short,
} from "./charts.js?v=7";

const $ = (id) => document.getElementById(id);

let filter = "";
let heroTab = "duel";
let cache = { dashboard: null, benchmarks: null, history: null, regHistory: null };
let fps = { dashboard: "", benchmarks: "", history: "", hero: "", reg: "" };
let closeWatch = null;

const hubUrl = (repo) => (repo ? `https://huggingface.co/${repo}` : null);

function badge(kind, text) {
  return `<span class="badge ${kind}">${esc(text)}</span>`;
}

function chartWidth() {
  return Math.max(window.innerWidth || 960, 320);
}

/* ---------- hero ---------- */

function renderHero(force = false) {
  const svg = $("hero-chart");
  if (!svg) return;
  const d = cache.dashboard;
  // Market bar is independent of chart dirty-checks.
  renderMarketBar(d);

  const key = `${heroTab}|${fps.history}|${fps.dashboard}|${chartWidth()}`;
  if (!force && key === fps.hero) return;
  fps.hero = key;

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

  const k = d?.king;
  const stats = d?.stats || {};
  const q = (d?.queue || []).length;
  const name = k?.repo?.split("/").pop() || "—";
  $("hero-king-row").innerHTML = [
    ["champion", k ? `<b class="gold">${esc(name)}</b>` : "<b>—</b>"],
    ["reign", `<b>${k ? `#${esc(k.reign_number)}` : "—"}</b>`],
    ["queue", `<b>${esc(q)}</b>`],
    ["duels", `<b>${esc(stats.duels ?? stats.accepted ?? "—")}</b>`],
    ["phase", `<b>${esc(d?.phase?.name ?? "—")}</b>`],
  ].map(([lab, val]) =>
    `<span class="stat-chip"><span class="k">${lab}</span>${val}</span>`).join("");
}

function renderMarketBar(d) {
  const el = $("market-bar-inner");
  if (!el) return;
  const market = d?.market;
  if (!market) {
    el.innerHTML = `<span class="market-item dim">SN120 · waiting on TaoMarketCap</span>`;
    return;
  }
  const weightsTitle = market.weights_committed_at
    ? `last weights commit ${market.weights_committed_at}`
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
  ].filter(Boolean).join('<span class="market-sep" aria-hidden="true">·</span>');
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

function benchSuites(b) {
  // Prefer contract suites (what the subnet is following). Fall back to
  // whatever appears in results if the payload omitted `suites`.
  const fromCfg = Array.isArray(b?.suites) ? b.suites.filter(Boolean) : [];
  if (fromCfg.length) return fromCfg;
  return [...new Set((b?.models || []).flatMap((m) => Object.keys(m.suites || {})))];
}

function suiteLabel(suite) {
  return String(suite || "").replace(/^tau2_/, "").replace(/_/g, " ");
}

function renderBenchmarks(b) {
  const el = $("benchmarks-wrap");
  const suites = benchSuites(b);
  const models = b?.models || [];
  if (!suites.length) {
    el.innerHTML = `<div class="empty">no benchmark suites configured</div>`;
    return;
  }
  const active = b?.active || [];
  el.innerHTML = `<div class="bench-grid">${suites.map((suite) => {
    const scored = models.filter((m) => m.suites?.[suite]?.ok && m.suites[suite].score != null).length;
    const failed = models.filter((m) => m.suites?.[suite]?.ok === false).length;
    const running = active.filter((j) => j.suite === suite).length;
    const bits = [];
    if (scored) bits.push(`${scored} scored`);
    if (failed) bits.push(`${failed} fail`);
    if (running) bits.push(`${running} running`);
    if (!bits.length) bits.push("awaiting runs");
    return `<div class="bench-card" data-suite="${esc(suite)}">
      <div class="bench-card-head">
        <span class="bench-card-title">${esc(suiteLabel(suite))}</span>
        <span class="bench-card-meta">${esc(bits.join(" · "))}</span>
      </div>
      <svg class="bench-suite-chart" data-suite="${esc(suite)}"
        role="img" aria-label="${esc(suiteLabel(suite))} scores"></svg>
    </div>`;
  }).join("")}</div>`;

  el.querySelectorAll("svg.bench-suite-chart").forEach((svg) => {
    const suite = svg.dataset.suite;
    const host = svg.parentElement;
    const w = Math.max((host?.clientWidth || 280) - 20, 200);
    drawBenchSuite(svg, suite, models, { width: w });
  });
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
      <th>when</th><th>event</th><th>model</th><th>outcome</th><th class="r">z</th><th class="r">S*</th><th class="r">king S*</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 80).map((r) => {
      const url = hubUrl(r.repo);
      const zClass = r.z == null ? "" : Number(r.z) >= 0 ? "ok" : "bad";
      const cid = r.challenge_id || "";
      return `<tr class="row-link ${r.event === "crowned" ? "current" : ""}" data-cid="${esc(cid)}">
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
      <th>when</th><th>model</th><th>code</th><th>detail</th>
    </tr></thead>
    <tbody>${rows.slice(0, 60).map((r) => {
      const url = hubUrl(r.repo);
      const cid = r.challenge_id || "";
      return `<tr class="row-link" data-cid="${esc(cid)}">
        <td class="when">${esc(fmtTime(r.at))}</td>
        <td>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(r.repo)}</a>` : esc(r.repo)}</td>
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
  renderChallenge(d);
  renderQueue(d);
}

function renderAll() {
  renderHero();
  renderSnapshotSections();
  renderBenchmarks(cache.benchmarks);
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
  const url = hubUrl(duel.repo || info.repo);

  if (fail && !duel.has_series && duel.z == null && duel.score == null) {
    body.innerHTML = `
      <div class="kv-grid">
        <div class="kv"><span class="k">when</span><span class="v">${esc(fmtTime(info.at))}</span></div>
        <div class="kv"><span class="k">code</span><span class="v bad">${esc(info.code)}</span></div>
        <div class="kv"><span class="k">model</span><span class="v">${
          url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(info.repo || "—")}</a>`
              : esc(info.repo || "—")}</span></div>
        <div class="kv"><span class="k">revision</span><span class="v mono">${esc(info.revision || "—")}</span></div>
        <div class="kv"><span class="k">hotkey</span><span class="v mono">${esc(info.hotkey || "—")}</span></div>
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
      <div class="kv"><span class="k">model</span><span class="v">${
        url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(duel.repo || "—")}</a>`
            : esc(duel.repo || "—")}</span></div>
      <div class="kv"><span class="k">outcome</span><span class="v ${fail ? "bad" : ""}">${esc(outcome)}</span></div>
      <div class="kv"><span class="k">z</span><span class="v ${Number(duel.z) >= 0 ? "ok" : "bad"}">${esc(fmtZ(duel.z))}</span></div>
      <div class="kv"><span class="k">margin</span><span class="v">${esc(fmtScore(duel.margin))} · se ${esc(fmtScore(duel.se))}</span></div>
      <div class="kv"><span class="k">S* chall</span><span class="v">${esc(fmtScore(duel.score ?? ch.S))}</span></div>
      <div class="kv"><span class="k">S* king</span><span class="v">${esc(fmtScore(duel.score_king ?? kg.S))}</span></div>
      <div class="kv"><span class="k">gate pass</span><span class="v">${esc(fmtScore(ch.gate_pass_rate))} / bank ${esc(fmtScore(ch.bank_frac))}</span></div>
      <div class="kv"><span class="k">thresholds</span><span class="v dim">kσ=${esc(gates.k_sigma ?? 3)} · δ=${esc(gates.min_margin ?? "—")}</span></div>
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
    stats: snap.stats,
    reign: snap.reign,
    market: snap.market,
  });
  if (fp === fps.dashboard) return;
  fps.dashboard = fp;
  cache.dashboard = snap;
  renderMarketBar(snap);
  renderHero();
  renderSnapshotSections();
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
    renderHistory(cache.history);
    renderFails(cache.history);
  }
  const bfp = fingerprint(b);
  if (b && bfp !== fps.benchmarks) {
    fps.benchmarks = bfp;
    cache.benchmarks = b;
    renderBenchmarks(cache.benchmarks);
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
  document.querySelectorAll(".hero-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      heroTab = btn.dataset.tab;
      document.querySelectorAll(".hero-tab").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      fps.hero = "";
      renderHero(true);
    });
  });
  $("filter-input")?.addEventListener("input", (e) => {
    filter = e.target.value.trim().toLowerCase();
    refreshHistoryAndBench();
  });
  window.addEventListener("resize", () => {
    fps.hero = "";
    renderHero(true);
    renderRegPrice(true);
    if (cache.benchmarks) renderBenchmarks(cache.benchmarks);
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
