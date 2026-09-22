/** Track S dashboard — fetches dataS.json (regenerated every 2 minutes from
 * trackS_status.log); the page re-polls every 90s. */

import { esc, pct, fmtTime, fmtClock, fmtAgo } from "./charts.js?v=3";
import {
  drawFoolEquilibrium, drawHeldAcc, drawTypicality, drawAgreement,
} from "./charts_s.js?v=1";

const $ = (id) => document.getElementById(id);
const POLL_MS = 90_000;

const VERDICT_CLASS = { STABLE: "stable", DRIFT: "drift", LEAK: "leak" };

const fmtDur = (s) => {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  return m ? `${m}m${String(Math.round(s % 60)).padStart(2, "0")}s` : `${Math.round(s)}s`;
};

function renderStatusBar(d) {
  const c = d.current || {};
  const v = d.verdict || {};
  const cls = VERDICT_CLASS[v.label] || "dim";
  $("status-bar").innerHTML = `
    <span class="market-item"><span class="k">round</span><b class="gold">${c.round ?? "—"}</b></span>
    <span class="market-item"><span class="k">fool</span><b class="gold">${pct(c.fool, 2)}</b></span>
    <span class="market-item"><span class="k">mean₅</span><b>${v.mean_fool5 != null ? pct(v.mean_fool5, 2) : "—"}</b></span>
    <span class="market-item" title="${esc(v.reason)}"><span class="k">verdict</span>
      <b class="${cls === "stable" ? "ok" : cls === "dim" ? "" : "warn"}">${esc(v.label ?? "—")}</b></span>
    <span class="market-item"><span class="k">judge</span><b>${esc(c.dver ?? "—")}</b></span>
    <span class="market-item"><span class="k">held live/ctrl/tt</span>
      <b>${pct(c.held_live, 1)} / ${pct(c.held_ctrl, 1)} / ${pct(c.held_tt, 1)}</b></span>
    <span class="market-item"><span class="k">updated</span><b>${fmtClock(d.log_end)} UTC</b></span>`;
}

function renderEquilibrium(d) {
  const c = d.current || {};
  const v = d.verdict || {};
  const badge = $("verdict-badge");
  badge.textContent = v.label ?? "PENDING";
  badge.className = `verdict-badge ${VERDICT_CLASS[v.label] || "pending"}`;
  badge.title = v.reason || "computed from recent rounds";

  $("equilibrium-meta").textContent =
    `G = exact teacher copy · truth is 0.5 everywhere · verdict window: last ${v.n ?? 5} rounds`;
  $("equilibrium-wrap").innerHTML = `<div class="kv-grid">
    <div class="kv"><span class="k">round</span><span class="v gold">${c.round ?? "—"}</span>
      <span class="sub">${c.avg_wall_s ? `~${fmtDur(c.avg_wall_s)} per round` : "warming up"}</span></div>
    <div class="kv"><span class="k">fool rate</span><span class="v gold">${pct(c.fool, 2)}</span>
      <span class="sub">latest round · truth 0.5</span></div>
    <div class="kv"><span class="k">mean fool (5r)</span>
      <span class="v ${v.label === "STABLE" ? "ok" : v.label ? "bad" : "dim"}">${v.mean_fool5 != null ? pct(v.mean_fool5, 2) : "—"}</span>
      <span class="sub">stable band 0.45–0.55</span></div>
    <div class="kv"><span class="k">judge</span><span class="v">${esc(c.dver ?? "—")}</span>
      <span class="sub">${c.d_updated_at ? `updated ${fmtTime(c.d_updated_at)} UTC` : "online · every 3rd round"}</span></div>
    <div class="kv"><span class="k">held live</span><span class="v">${pct(c.held_live, 2)}</span>
      <span class="sub">D vs current G · truth 0.5</span></div>
    <div class="kv"><span class="k">held ctrl / tt</span>
      <span class="v">${pct(c.held_ctrl, 2)} / ${pct(c.held_tt, 2)}</span>
      <span class="sub">leak alarms — sustained &gt;0.58 = artifact</span></div>
    <div class="kv"><span class="k">typicality</span>
      <span class="v">${c.typ_lp != null ? c.typ_lp.toFixed(4) : "—"}</span>
      <span class="sub">typ_lp · think_len ${c.think_len ?? "—"}</span></div>
    <div class="kv"><span class="k">g adapter</span>
      <span class="v accent">${esc(c.g ?? "—")}</span>
      <span class="sub">${c.eff != null ? `eff vs base ${c.eff}` : "eff — (kept previous)"}</span></div>
    <div class="kv"><span class="k">time in run</span><span class="v">${fmtAgo(d.log_start)}</span>
      <span class="sub">since ${fmtTime(d.log_start)} UTC</span></div>
  </div>`;
}

function renderRunLog(d) {
  const evs = d.events || [];
  if (!evs.length) {
    $("runlog-meta").textContent = "0 events";
    $("runlog-wrap").innerHTML = '<div class="empty">no infra events yet</div>';
    return;
  }
  const rows = [...evs].reverse().map((e) => `<tr>
      <td class="when">${fmtTime(e.t)} UTC</td>
      <td>${esc(e.src)}</td>
      <td class="${e.sev === "bad" ? "sev-bad" : e.sev === "warn" ? "sev-warn" : ""}">${esc(e.msg)}</td>
      <td class="r">${e.count > 1 ? `×${e.count}` : ""}</td>
    </tr>`).join("");
  $("runlog-meta").textContent = `${evs.length} events (consecutive repeats collapsed)`;
  $("runlog-wrap").innerHTML = `<table class="data-table">
    <thead><tr><th>time</th><th>src</th><th>event</th><th class="r">repeats</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function render(d) {
  if (d.source === "missing") {
    $("status-bar").innerHTML =
      '<span class="market-item dim">Track S — waiting for trackS_status.log to appear</span>';
    $("equilibrium-wrap").innerHTML =
      '<div class="empty">no run log yet — the parser picks it up automatically once the file appears</div>';
    return;
  }
  $("charts-meta").textContent =
    `source: ${d.source} · log through ${fmtClock(d.log_end)} UTC · regenerated ${fmtClock(d.generated_at)} UTC`;
  $("footer-note").textContent =
    `Track S · self-play control · SN120 · live run · data refreshes every 2 min`;

  renderStatusBar(d);
  renderEquilibrium(d);
  renderRunLog(d);
  drawFoolEquilibrium($("chart-fool"), d);
  drawHeldAcc($("chart-held"), d);
  drawTypicality($("chart-typ"), d);
  drawAgreement($("chart-agree"), d);
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
    const res = await fetch(`dataS.json?t=${Date.now()}`, { cache: "no-store" });
    render(await res.json());
  } catch (err) {
    $("status-bar").innerHTML =
      '<span class="market-item dim">failed to load dataS.json — retrying</span>';
  }
}

wireTips();
refresh();
setInterval(refresh, POLL_MS);
