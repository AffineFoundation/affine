/** Track M dashboard — fetches data.json, regenerated every 2 minutes from
 * trackM_status.log by refresh_data.sh (mock fallback if the log vanishes).
 * The page re-polls data.json every 90s. */

import {
  esc, pct, fmtTime, fmtClock, fmtAgo,
  drawReignFoolRate, drawRatchet, drawJudgeAcc, drawSwe, drawMechDiagram,
} from "./charts.js?v=2";

const $ = (id) => document.getElementById(id);
const POLL_MS = 90_000;

/** "Qwen/Qwen3.6-35B-A3B" → "Qwen3.6-35B-A3B" for tight chips. */
const shortModel = (s) => (s ? String(s).split("/").pop() : "—");

function renderStatusBar(d) {
  const c = d.current;
  const state = c.evals_state || "—";
  const paused = state !== "running";
  const q = d.quarantine;
  $("status-bar").innerHTML = `
    <span class="market-item"><span class="k">reign</span><b class="gold">R${c.reign}</b></span>
    <span class="market-item"><span class="k">king</span><b class="gold">${esc(shortModel(c.king))}</b></span>
    <span class="market-item"><span class="k">judge</span><b>${esc(c.judge_version ?? "—")}</b></span>
    <span class="market-item"><span class="k">fool rate</span><b>${pct(c.king_fool_rate)}</b></span>
    <span class="market-item"><span class="k">evals</span>
      <b class="${paused ? "warn" : "ok"}">${esc(state.toUpperCase())}</b></span>
    ${q ? `<span class="market-item" title="${esc(q.note)}"><span class="k">quarantine</span>
      <b class="warn">${fmtClock(q.start)}–${fmtClock(q.end)} UTC</b></span>` : ""}
    <span class="market-item"><span class="k">updated</span>
      <b>${fmtClock(d.log_end)} UTC</b></span>`;
}

function renderArena(d) {
  const c = d.current;
  const need = d.margin_pp ?? 0.03;
  const state = c.evals_state || "—";
  const paused = state !== "running";
  const gap = (c.challenger_fool_rate != null && c.king_fool_rate != null)
    ? c.challenger_fool_rate - c.king_fool_rate : null;
  const minerQuar = c.miner_fool_local == null && c.miner_round != null;

  $("arena-meta").textContent =
    `dethrone: +${(need * 100).toFixed(0)}pp over ≥${d.min_turns} paired turns`;
  $("arena-wrap").innerHTML = `<div class="kv-grid">
    <div class="kv"><span class="k">reign</span><span class="v gold">R${c.reign}</span>
      <span class="sub">since ${fmtTime(c.reign_started_at)} UTC (corrected publish)</span></div>
    <div class="kv"><span class="k">king</span><span class="v gold">${esc(shortModel(c.king))}</span>
      <span class="sub">${c.reign === 0 ? "genesis — base model" : "crowned checkpoint"}</span></div>
    <div class="kv"><span class="k">judge</span><span class="v">${esc(c.judge_version ?? "—")}</span>
      <span class="sub">frozen · held-out acc ${pct(c.judge_acc)}</span></div>
    <div class="kv"><span class="k">king fool rate</span>
      <span class="v ${c.king_fool_rate != null ? "gold" : "dim"}">${pct(c.king_fool_rate)}</span>
      <span class="sub">${c.king_fool_rate != null
        ? `latest batch vs frozen ${esc(c.judge_version)}`
        : `awaiting evals under ${esc(c.judge_version ?? "new judge")}`}</span></div>
    <div class="kv"><span class="k">challenger</span>
      <span class="v ${c.challenger_fool_rate != null ? "accent" : "dim"}">${c.challenger_fool_rate != null ? pct(c.challenger_fool_rate) : "none yet"}</span>
      <span class="sub">${c.challenger_fool_rate != null
        ? esc(shortModel(c.challenger) || "in duel") : "no submission duelling"}</span></div>
    <div class="kv"><span class="k">gap to crown</span>
      <span class="v ${gap == null ? "dim" : gap >= need ? "ok" : ""}">${gap == null ? "—" : `${gap >= 0 ? "+" : ""}${(gap * 100).toFixed(1)}pp`}</span>
      <span class="sub">needs +${(need * 100).toFixed(0)}pp</span></div>
    <div class="kv"><span class="k">time in reign</span><span class="v">${fmtAgo(c.reign_started_at)}</span>
      <span class="sub">${c.paired_turns ?? 0} paired duel turns</span></div>
    <div class="kv"><span class="k">miner training</span>
      <span class="v ${minerQuar ? "dim" : "accent"}">${c.miner_round != null ? `round ${c.miner_round}` : "—"}</span>
      <span class="sub">${minerQuar
        ? "reset to base after quarantine"
        : c.miner_fool_local != null ? `local fool ${pct(c.miner_fool_local)}` : "warming up"}</span></div>
    <div class="kv"><span class="k">evals</span>
      <span class="v ${paused ? "bad" : "ok"}">${esc(state.toUpperCase())}</span>
      <span class="sub">${paused ? "judge not frozen-live" : "judge frozen for the reign"}</span></div>
  </div>`;
}

function renderCrownLog(d) {
  const crowns = d.crowns || [];
  if (!crowns.length) {
    $("crown-meta").textContent = "0 crowns";
    $("crown-wrap").innerHTML = `<div class="empty">no crowns yet — reign ${d.current?.reign ?? 0} in progress
      (dethrone needs +${((d.margin_pp ?? 0.03) * 100).toFixed(0)}pp over ≥${d.min_turns} paired turns)</div>`;
    return;
  }
  // Defensive mapping: CROWN line format may evolve; show what we have.
  const rows = [...crowns].reverse().map((c, i) => `<tr class="${i === 0 ? "current" : ""}">
      <td><span class="king-gold">R${c.reign ?? "—"}</span></td>
      <td><span class="king-gold">${esc(shortModel(c.king ?? c.challenger ?? "—"))}</span></td>
      <td class="when">${fmtTime(c.t)} UTC</td>
      <td class="r">${c.margin != null ? `+${(c.margin * 100).toFixed(1)}pp` : "—"}</td>
      <td>${esc(c.dver ?? c.judge ?? "—")}</td>
      <td class="r">${c.retrain_hours != null ? `${c.retrain_hours}h` : "—"}</td>
      <td class="r">${pct(c.judge_acc)}</td>
      <td class="r">${c.n ?? "—"}</td>
      <td>${i === 0
        ? '<span class="badge reigning">reigning</span>'
        : '<span class="badge crowned">crowned</span>'}</td>
    </tr>`).join("");
  $("crown-meta").textContent = `${crowns.length} crowns`;
  $("crown-wrap").innerHTML = `<table class="data-table">
    <thead><tr>
      <th>reign</th><th>king</th><th>crowned at</th><th class="r">margin</th>
      <th>judge</th><th class="r">retrain</th><th class="r">judge acc</th>
      <th class="r">turns</th><th>status</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

/* shared tooltip for .chart-hit marks (original dashboard behavior) */
function wireTips() {
  const tip = $("chart-tip");
  document.addEventListener("mousemove", (e) => {
    const hit = e.target.closest?.(".chart-hit");
    if (!hit || !hit.dataset.tip) { tip.hidden = true; return; }
    tip.textContent = hit.dataset.tip;
    tip.hidden = false;
    const pad = 14;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  });
}

function render(d) {
  const live = d.source === "live";
  const badge = $("mock-badge");
  if (badge) badge.hidden = live;

  $("hero-teacher").textContent = d.teacher;
  $("mech-teacher").textContent = d.teacher;
  $("mech-miner").textContent = d.miner_base;
  $("charts-meta").textContent =
    `source: ${d.source} · log through ${fmtClock(d.log_end)} UTC · regenerated ${fmtClock(d.generated_at)} UTC`;
  $("footer-note").textContent =
    `Track M · GAD arena · SN120 · ${live ? "live run" : "mock dataset"} · data refreshes every 2 min`;

  renderStatusBar(d);
  renderArena(d);
  renderCrownLog(d);
  drawReignFoolRate($("chart-reign"), d);
  drawRatchet($("chart-ratchet"), d);
  drawJudgeAcc($("chart-judge"), d);
  drawSwe($("chart-swe"), d);
}

async function refresh() {
  try {
    const res = await fetch(`data.json?t=${Date.now()}`, { cache: "no-store" });
    render(await res.json());
  } catch (err) {
    $("status-bar").innerHTML =
      '<span class="market-item dim">failed to load data.json — retrying</span>';
  }
}

wireTips();
drawMechDiagram($("mech-diagram"));
refresh();
setInterval(refresh, POLL_MS);
