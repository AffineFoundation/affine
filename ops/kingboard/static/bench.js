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
    $("#tab-challengers").classList.toggle("hidden", name !== "challengers");
    if ((name === "benchmarks" || name === "challengers") && !state.runs.length) load();
  }

  // -- Challengers tab: challenger cards (mode "challenger") vs the king they duelled --
  const CHAL_ENVS = ["aime25", "math500", "gpqa-diamond", "mmlu-pro", "ifbench", "ifeval", "humaneval", "livecodebench", "bfcl-v3"];
  function kingCardFor(card) {
    const d = card.duel || {};
    const byDigest = state.runs.filter((r) => r.mode !== "challenger" && r.mode !== "comparables" && r.king && r.king.digest === d.vs_king_digest && r.status !== "skipped_identical_weights");
    if (byDigest.length) return byDigest.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))[0];
    const byReign = state.runs.filter((r) => r.mode !== "challenger" && r.king && String(r.king.reign) === String(d.vs_reign));
    return byReign.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))[0] || null;
  }
  function renderChallengers() {
    const thead = $("#chal-table thead"), tbody = $("#chal-table tbody");
    thead.innerHTML = ""; tbody.innerHTML = "";
    const cards = state.runs.filter((r) => r.mode === "challenger" || (r.king && r.king.duel));
    if (!cards.length) { $("#chal-note").textContent = "no challenger cards yet"; return; }
    const envs = CHAL_ENVS.filter((e) => cards.some((c) => c.rows.some((x) => x.env === e && x.temperature === 0)));
    const head = el("tr", {}, el("th", {}, "challenger"), el("th", { class: "num" }, "duel margin"), el("th", { class: "num" }, "z"), el("th", {}, "vs king"), el("th", {}, "status"));
    for (const e of envs) head.append(el("th", { class: "num" }, e));
    head.append(el("th", { class: "num" }, "mean Δ"));
    thead.append(head);
    cards.sort((a, b) => ((b.duel || {}).margin || 0) - ((a.duel || {}).margin || 0));
    for (const c of cards) {
      const d = c.duel || {}; const base = kingCardFor(c);
      const tr = el("tr", {}, el("td", { title: c.run_id }, (c.king && c.king.label) || c.run_id),
        el("td", { class: "num" }, d.margin !== undefined && d.margin !== null ? (d.margin > 0 ? "+" : "") + Number(d.margin).toFixed(5) : "–"),
        el("td", { class: "num" }, d.z !== undefined && d.z !== null ? Number(d.z).toFixed(2) : "–"),
        el("td", { class: "muted" }, base ? `reign ${base.king.reign}` : (d.vs_reign ? `reign ${d.vs_reign} (no card)` : "–")),
        el("td", { class: "muted" }, c.status === "partial" ? "running" : c.status));
      const deltas = [];
      for (const e of envs) {
        const row = c.rows.find((x) => x.env === e && x.temperature === 0 && x.king);
        const brow = base ? base.rows.find((x) => x.env === e && x.temperature === 0 && x.king) : null;
        if (!row) { tr.append(el("td", { class: "num muted" }, "…")); continue; }
        if (!brow) { tr.append(el("td", { class: "num" }, pct(row.king.score))); continue; }
        const delta = 100 * (row.king.score - brow.king.score), hw = 100 * (brow.king.ci95[1] - brow.king.ci95[0]) / 2;
        deltas.push(delta);
        const cls = Math.abs(delta) > hw ? (delta > 0 ? "good" : "bad") : "";
        tr.append(el("td", { class: "num " + cls, title: `${pct(row.king.score)} vs king ${pct(brow.king.score)} [±${hw.toFixed(1)}]` }, `${delta > 0 ? "+" : ""}${delta.toFixed(1)}${Math.abs(delta) > hw ? "*" : ""}`));
      }
      tr.append(el("td", { class: "num" }, deltas.length ? `${(deltas.reduce((a, b) => a + b, 0) / deltas.length).toFixed(1)} pt` : "–"));
      tbody.append(tr);
    }
    $("#chal-note").textContent = "* = outside the king card's 95% interval. Rows sorted by duel margin (the meter's order); if the meter tracked the benchmarks, mean Δ would fall down the table.";
  }

  function runLabel(r) {
    const k = r.king || {};
    if (k.model) return `${r.run_id} · ${k.label || k.model} (Prime Inference)`;
    const reign = k.reign !== undefined ? `reign ${k.reign}` : (k.label || "");
    const digest = k.digest ? `king-${String(k.digest).slice(0, 12)}` : "";
    return `${r.run_id} · ${reign}${k.uncrowned ? " (removed)" : ""} ${digest}`.trim();
  }
  const colLabel = (r) => { const k = (r && r.king) || {}; return k.model ? (k.label || k.model) : k.reign !== undefined ? `reign ${k.reign}${k.uncrowned ? " (removed)" : ""}` : (k.label || r.run_id); };

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
      // Per gold class (When2Call): accuracy per class and how often each side
      // answered with the tool call — on a non-tool class that is the
      // "called a tool when none was needed" rate.
      const kb = (k && k.by_class) || {}, tb = (t && t.by_class) || {};
      const classes = [...new Set([...Object.keys(kb), ...Object.keys(tb)])].sort()
        .filter((c) => !row.show_classes || row.show_classes.includes(c));
      for (const cls of classes) {
        const kc = kb[cls], tc = tb[cls];
        const dd = kc && tc ? kc.score - tc.score : null;
        const toolRate = (c) => c && c.metrics && c.metrics.pred_tool_call !== undefined ? ` · →tool ${pct(c.metrics.pred_tool_call, 0)}` : "";
        tbody.append(el("tr", { class: "subrow", title: `${row.env}: rows whose gold answer is "${cls}"` },
          el("td", { class: "muted" }, `  ↳ ${cls}`),
          el("td", { class: "muted" }, "gold class"),
          el("td", { class: "num muted" }, ""),
          el("td", { class: "num muted" }, (kc || tc || {}).n),
          el("td", { class: "num muted" }, kc ? pct(kc.score) + ci(kc) + toolRate(kc) : "–"),
          el("td", { class: "num muted" }, tc ? pct(tc.score) + ci(tc) + toolRate(tc) : "–"),
          el("td", { class: "num muted" }, dd === null ? "–" : (dd > 0 ? "+" : "") + (100 * dd).toFixed(1) + " pt"),
          el("td", { class: "num muted" }, ""), el("td", { class: "num muted" }, ""),
          el("td", { class: "num muted" }, ""), el("td", { class: "num muted" }, "")));
      }
    }
    const sk = (r.skipped || []).map((s) => `${s.env}: ${s.why}`);
    $("#bench-skipped").textContent = sk.length ? "Not run — " + sk.join(" · ") : "";
    $("#bench-links").innerHTML = "";
    if (r.prime_evals && Object.keys(r.prime_evals).length) {
      $("#bench-links").append(el("span", {}, `Prime Evals: ${Object.keys(r.prime_evals).length} runs on account ${r.prime_evals_account || "arbos"} · `),
        el("a", { href: "https://app.primeintellect.ai/dashboard/evaluations", target: "_blank" }, "Evaluations tab"), " · ");
    }
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
    renderChallengers();
  }

  $("#bench-run").addEventListener("change", (e) => { state.runId = e.target.value; renderTable(); });
  const tabOf = () => location.hash === "#benchmarks" ? "benchmarks" : location.hash === "#challengers" ? "challengers" : "envs";
  window.addEventListener("hashchange", () => showTab(tabOf()));
  showTab(tabOf());
  setInterval(() => { if (tabOf() !== "envs") load(); }, 300000);
})();
