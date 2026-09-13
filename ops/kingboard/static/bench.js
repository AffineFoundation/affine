/* Kingboard "Benchmarks" tab: renders /api/benchsuite.json (one scorecard per
   benchmark-suite run, written by ops/benchsuite/publish.py). Tab switching
   is by URL hash (#envs | #benchmarks) so links are shareable. */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs, ...kids) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    for (const kid of kids) {
      if (kid === null || kid === undefined) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  };
  const pct = (x, d = 1) => (x === null || x === undefined) ? "–" : (100 * x).toFixed(d) + "%";
  const ci = (side) => side && side.ci95 ? ` [${pct(side.ci95[0], 0)}–${pct(side.ci95[1], 0)}]` : "";
  const mins = (s) => (s === null || s === undefined) ? "–" : s < 5400 ? Math.round(s / 60) + " min" : (s / 3600).toFixed(1) + " h";
  const kfmt = (n) => (n === null || n === undefined) ? "–" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "k" : String(n);

  const state = { runs: [], runId: null };

  function showTab(name) {
    document.querySelectorAll("#tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === name));
    $("#tab-envs").classList.toggle("hidden", name !== "envs");
    $("#tab-benchmarks").classList.toggle("hidden", name !== "benchmarks");
    if (name === "benchmarks" && !state.runs.length) load();
  }

  function runLabel(r) {
    const k = r.king || {};
    if (k.model) return `${r.run_id} · ${k.label || k.model} (Prime Inference)`;
    const reign = k.reign !== undefined ? `reign ${k.reign}` : (k.label || "");
    const digest = k.digest ? `king-${String(k.digest).slice(0, 12)}` : "";
    return `${r.run_id} · ${reign} ${digest}`.trim();
  }
  const colLabel = (r) => { const k = (r && r.king) || {}; return k.model ? (k.label || k.model) : k.reign !== undefined ? `reign ${k.reign}` : (k.label || r.run_id); };

  function renderRunSelect() {
    const sel = $("#bench-run");
    sel.innerHTML = "";
    for (const r of state.runs) sel.append(el("option", { value: r.run_id }, runLabel(r)));
    if (!state.runId || !state.runs.some((r) => r.run_id === state.runId)) state.runId = state.runs[0] && state.runs[0].run_id;
    sel.value = state.runId || "";
  }

  function current() { return state.runs.find((r) => r.run_id === state.runId); }

  function renderTable() {
    const r = current();
    const tbody = $("#bench-table tbody");
    tbody.innerHTML = "";
    if (!r) { $("#bench-meta").textContent = "no benchmark runs published yet"; return; }
    const w = r.where || {};
    const ident = r.identical_to ? `IDENTICAL WEIGHTS — same ${r.identical_to.n_tensors} tensors as reign ${r.identical_to.reign} (${r.identical_to.run_id}); numbers shown are that run's, no new pass · ` : "";
    $("#bench-meta").textContent = `${r.rows.length} rows · ${ident}${r.status === "partial" ? "PARTIAL (still running) · " : ""}run created ${(r.created_at || "").replace("T", " ")} · ` +
      `${w.provider || ""} ${w.gpu || ""} · teacher ${(r.teacher || {}).hf_repo || ""}` +
      (r.prime_spent_usd !== undefined && r.prime_spent_usd !== null ? ` · pod cost ≈ $${r.prime_spent_usd}` : "");
    const rows = [...r.rows].sort((a, b) => (a.group || "").localeCompare(b.group || "") || a.env.localeCompare(b.env) || a.temperature - b.temperature);
    for (const row of rows) {
      const k = row.king, t = row.teacher;
      const d = row.delta;
      const dcls = d === null || d === undefined ? "" : d > 0.005 ? "good" : d < -0.005 ? "bad" : "";
      tbody.append(el("tr", { title: row.note || "" },
        el("td", {}, row.env),
        el("td", { class: "muted" }, row.group || ""),
        el("td", { class: "num" }, row.temperature === 0 ? "0" : String(row.temperature)),
        el("td", { class: "num" }, k ? k.n : (t ? t.n : "–")),
        el("td", { class: "num king-col" }, k ? pct(k.score) + ci(k) : "–"),
        el("td", { class: "num teacher-col" }, t ? pct(t.score) + ci(t) : "–"),
        el("td", { class: "num " + dcls }, d === null || d === undefined ? "–" : (d > 0 ? "+" : "") + (100 * d).toFixed(1) + " pt"),
        el("td", { class: "num muted", title: "score over rollouts that finished inside the time/context budget" },
          k && k.finished_only ? `${pct(k.finished_only.score)} (n=${k.finished_only.n})` : "–"),
        el("td", { class: "num muted" }, k ? kfmt(k.completion_tokens) : "–"),
        el("td", { class: "num muted" }, k && k.finish_length_frac !== undefined ? pct(k.finish_length_frac, 0) : "–"),
        el("td", { class: "num muted" }, k ? mins(k.wall_seconds) : "–")));
    }
    const sk = (r.skipped || []).map((s) => `${s.env}: ${s.why}`);
    $("#bench-skipped").textContent = sk.length ? "Not run — " + sk.join(" · ") : "";
    $("#bench-links").innerHTML = "";
    if (r.r2_prefix) {
      $("#bench-links").append("Every rollout (prompts, replies, grades, tokens, timing): ",
        el("a", { href: `https://data.affine.io/${r.r2_prefix}index.json`, target: "_blank" }, `data.affine.io/${r.r2_prefix}`));
    }
  }

  function renderHistory() {
    const thead = $("#bench-history thead"), tbody = $("#bench-history tbody");
    thead.innerHTML = ""; tbody.innerHTML = "";
    const runs = [...state.runs].sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""));
    if (!runs.length) return;
    const envs = [...new Set(runs.flatMap((r) => r.rows.filter((x) => x.temperature === 0).map((x) => x.env)))].sort();
    const head = el("tr", {}, el("th", {}, "benchmark"));
    for (const r of runs) {
      const k = r.king || {};
      head.append(el("th", { class: "num", title: r.run_id }, colLabel(r)));
    }
    head.append(el("th", { class: "num" }, "teacher (latest)"));
    thead.append(head);
    const latest = runs[runs.length - 1];
    for (const env of envs) {
      const tr = el("tr", {}, el("td", {}, env));
      for (const r of runs) {
        const row = r.rows.find((x) => x.env === env && x.temperature === 0);
        tr.append(el("td", { class: "num" }, row && row.king ? pct(row.king.score) : "–"));
      }
      const lrow = latest.rows.find((x) => x.env === env && x.temperature === 0);
      tr.append(el("td", { class: "num teacher-col" }, lrow && lrow.teacher ? pct(lrow.teacher.score) : "–"));
      tbody.append(tr);
    }
  }

  async function load() {
    try {
      const res = await fetch("/api/benchsuite.json", { cache: "no-store" });
      const data = await res.json();
      state.runs = data.runs || [];
    } catch (e) {
      state.runs = [];
    }
    renderRunSelect();
    renderTable();
    renderHistory();
  }

  $("#bench-run").addEventListener("change", (e) => { state.runId = e.target.value; renderTable(); });
  window.addEventListener("hashchange", () => showTab(location.hash === "#benchmarks" ? "benchmarks" : "envs"));
  showTab(location.hash === "#benchmarks" ? "benchmarks" : "envs");
  setInterval(() => { if (location.hash === "#benchmarks") load(); }, 300000);
})();
