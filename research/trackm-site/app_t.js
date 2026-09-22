/** Track T dashboard — fetches dataT.json (regenerated every 2 minutes).
 * The run is just launching: a missing log renders an honest
 * "experiment starting up" state and the refresh loop picks the log up
 * automatically once it appears. Page re-polls every 90s. */

import { esc, pct, fmtTime, fmtClock, fmtAgo } from "./charts.js?v=3";
import { drawSweHeadline, drawFoolJudge } from "./charts_t.js?v=1";

const $ = (id) => document.getElementById(id);
const POLL_MS = 90_000;

function renderStatusBar(d) {
  const b = d.banner || {};
  $("status-bar").innerHTML = `
    <span class="market-item"><span class="k">round</span><b class="gold">${b.rounds ?? "—"}</b></span>
    <span class="market-item"><span class="k">fool</span><b class="gold">${pct(b.last_fool, 2)}</b></span>
    <span class="market-item"><span class="k">judge</span><b>${esc(b.judge_version ?? "—")}</b></span>
    <span class="market-item"><span class="k">gpu-hours</span><b>${b.gpu_hours != null ? b.gpu_hours.toFixed(1) : "—"}</b></span>
    <span class="market-item"><span class="k">benches</span>
      <b>${b.benches ? `${b.benches.proxy} proxy · ${b.benches.full} full` : "—"}</b></span>
    <span class="market-item"><span class="k">updated</span><b>${fmtClock(d.log_end)} UTC</b></span>`;
}

function renderBanner(d) {
  const b = d.banner || {};
  $("run-meta").textContent =
    `teacher full-panel reference: ${(d.swe_teacher ?? 0.3133).toFixed(4)}`;
  $("run-wrap").innerHTML = `<div class="kv-grid">
    <div class="kv"><span class="k">rounds</span><span class="v gold">${b.rounds ?? "—"}</span>
      <span class="sub">adversarial rounds completed</span></div>
    <div class="kv"><span class="k">time in run</span><span class="v">${fmtAgo(b.started_at)}</span>
      <span class="sub">since ${fmtTime(b.started_at)} UTC</span></div>
    <div class="kv"><span class="k">judge version</span><span class="v">${esc(b.judge_version ?? "—")}</span>
      <span class="sub">online discriminator</span></div>
    <div class="kv"><span class="k">gpu-hours</span>
      <span class="v">${b.gpu_hours != null ? b.gpu_hours.toFixed(1) : "—"}</span>
      <span class="sub">as of the latest bench line</span></div>
    <div class="kv"><span class="k">fool rate</span><span class="v gold">${pct(b.last_fool, 2)}</span>
      <span class="sub">latest round</span></div>
    <div class="kv"><span class="k">student raw baseline</span>
      <span class="v ${d.swe_student_baseline != null ? "" : "dim"}">${d.swe_student_baseline != null ? d.swe_student_baseline.toFixed(4) : "not benched yet"}</span>
      <span class="sub">full-panel SWE, untrained student</span></div>
  </div>`;
}

function renderStartingUp(d) {
  $("status-bar").innerHTML = `
    <span class="market-item"><span class="k">state</span><b class="warn">STARTING UP</b></span>
    <span class="market-item dim">waiting for trackT_status.log — checked ${fmtClock(d.generated_at)} UTC</span>`;
  $("run-meta").textContent = "no log yet";
  $("run-wrap").innerHTML = `<div class="empty">
    experiment starting up — trackT_status.log has not appeared yet.
    The data refresh loop (every 2 min) picks it up automatically the moment the run starts logging;
    this page re-polls every 90 s. Nothing below is mock data — it is simply empty.</div>`;
  $("headline-meta").textContent = "reference lines only — no bench results yet";
  $("charts-meta").textContent = "no rounds yet";
  $("runlog-meta").textContent = "0 events";
  $("runlog-wrap").innerHTML = '<div class="empty">no events yet</div>';
}

function renderRunLog(d) {
  const evs = d.events || [];
  if (!evs.length) {
    $("runlog-meta").textContent = "0 events";
    $("runlog-wrap").innerHTML = '<div class="empty">no events yet</div>';
    return;
  }
  const rows = [...evs].reverse().map((e) => `<tr>
      <td class="when">${fmtTime(e.t)} UTC</td>
      <td>${esc(e.src)}</td>
      <td class="${e.sev === "bad" ? "sev-bad" : e.sev === "warn" ? "sev-warn" : ""}">${esc(e.msg)}</td>
      <td class="r">${e.count > 1 ? `×${e.count}` : ""}</td>
    </tr>`).join("");
  $("runlog-meta").textContent =
    `${evs.length} events — judge retrains, bench verdicts, incidents`;
  $("runlog-wrap").innerHTML = `<table class="data-table">
    <thead><tr><th>time</th><th>src</th><th>event</th><th class="r">repeats</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function render(d) {
  const starting = d.source !== "live";
  $("starting-badge").hidden = !starting;
  $("footer-note").textContent =
    `Track T · pure-GAN run · SN120 · ${starting ? "starting up" : "live run"} · data refreshes every 2 min`;

  if (starting) {
    renderStartingUp(d);
  } else {
    $("headline-meta").textContent =
      `log through ${fmtClock(d.log_end)} UTC · regenerated ${fmtClock(d.generated_at)} UTC`;
    $("charts-meta").textContent =
      `${(d.rounds || []).length} rounds · ${(d.judges || []).length} judge retrains`;
    renderStatusBar(d);
    renderBanner(d);
    renderRunLog(d);
  }
  // Headline always draws: reference lines render even with zero data.
  drawSweHeadline($("chart-swe-t"), d);
  drawFoolJudge($("chart-fool-t"), d);
}

/* shared tooltip for .chart-hit marks */
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

async function refresh() {
  try {
    const res = await fetch(`dataT.json?t=${Date.now()}`, { cache: "no-store" });
    render(await res.json());
  } catch (err) {
    $("status-bar").innerHTML =
      '<span class="market-item dim">failed to load dataT.json — retrying</span>';
  }
}

wireTips();
refresh();
setInterval(refresh, POLL_MS);
