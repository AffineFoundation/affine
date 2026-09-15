/* Kingboard landing view: the model x (benchmark | environment) matrix from
   /api/matrix (build.py::build_matrix). Rows = teacher, genesis, kings in
   reign order (newest first); value = average score 0-100; tint = gap to the
   teacher row. Column headers sort; the first header restores reign order. */
(function () {
  "use strict";

  const REFRESH_MS = 300000;
  const TINT_FULL_PT = 40;      // |delta| in points at which the tint saturates
  const TINT_MAX_ALPHA = 0.42;

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
  const fmt = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
  const signed = (x, d = 1) => (x === null || x === undefined) ? "–" : (x > 0 ? "+" : "") + Number(x).toFixed(d);
  const isoShort = (iso) => iso ? String(iso).replace("T", " ").replace(/(\+00:00|Z)$/, "").slice(0, 16) + " UTC" : "–";
  const dayShort = (iso) => iso ? String(iso).replace("T", " ").slice(5, 16) : "–";   // MM-DD HH:MM
  const num = (x) => (x === null || x === undefined) ? "–" : Number(x).toLocaleString();

  const state = { matrix: null, sort: { key: null, desc: true } };

  function tint(delta, scale = 1) {
    if (delta === null || delta === undefined) return "";
    const a = scale * Math.min(TINT_MAX_ALPHA, Math.abs(delta) / TINT_FULL_PT * TINT_MAX_ALPHA);
    if (a < 0.02) return "";
    return delta > 0 ? `background-color: rgba(68, 255, 154, ${a.toFixed(3)})`
                     : `background-color: rgba(255, 71, 71, ${a.toFixed(3)})`;
  }

  // -- tooltip ------------------------------------------------------------------
  const tip = $("#chart-tip");
  function showTip(evt, lines) {
    tip.replaceChildren(...lines.map((l) => el("div", {}, ...(Array.isArray(l) ? l : [l]))));
    tip.hidden = false;
    moveTip(evt);
  }
  function moveTip(evt) {
    const pad = 14;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    tip.style.left = `${Math.max(4, x)}px`;
    tip.style.top = `${Math.max(4, y)}px`;
  }
  function hideTip() { tip.hidden = true; }

  function cellTip(row, col, cell, teacherCell) {
    const lines = [[el("b", {}, row.label), el("span", { class: "dim" }, ` · ${col.label}`)]];
    if (!cell || cell.score === null || cell.score === undefined) {
      lines.push(cell && cell.reason ? cell.reason : "no measurement");
      if (col.kind === "bench") lines.push(el("span", { class: "dim" }, "no benchmark card for this model yet"));
      return lines;
    }
    if (col.kind === "total") {
      lines.push([el("b", {}, fmt(cell.score)), ` · mean of ${cell.n_cols} columns (${cell.n_bench} benchmarks, ${cell.n_env} environments)`]);
      if (cell.teacher_same_cols !== undefined) {
        lines.push(`teacher on the same ${cell.n_same_cols} columns: ${fmt(cell.teacher_same_cols)} → Δ ${signed(cell.delta)} pt`);
      }
      return lines;
    }
    const ci = cell.lo !== null && cell.lo !== undefined ? ` [${fmt(cell.lo)}–${fmt(cell.hi)}] 95% CI` : "";
    lines.push([el("b", {}, fmt(cell.score)), ci]);
    if (col.kind === "bench") {
      lines.push(`n = ${num(cell.n)} tasks · greedy T=0${cell.metric === "finished_only" ? " · finished-only" : ""}`);
      if (cell.all_rollouts !== null && cell.all_rollouts !== undefined) lines.push(`all rollouts (timeouts count as failed): ${fmt(cell.all_rollouts)}`);
      lines.push(el("span", { class: "dim" }, `card ${cell.run_id}${cell.mode ? " · " + cell.mode : ""}${cell.reused_from ? " · teacher cells reused from " + cell.reused_from : ""}`));
    } else {
      lines.push(`${num(cell.solved)} solved / ${num(cell.n)} graded (${num(cell.rollouts)} rollouts, ${num(cell.errored)} errored)`);
      lines.push(el("span", { class: "dim" }, `datagen rollouts on ${col.env}${col.env_id ? " · " + col.env_id : ""}`));
    }
    if (row.kind !== "teacher" && teacherCell && teacherCell.score !== null && teacherCell.score !== undefined) {
      lines.push(`teacher ${fmt(teacherCell.score)} → Δ ${signed(cell.delta)} pt`);
    }
    if (col.note) lines.push(el("span", { class: "dim" }, col.note));
    return lines;
  }

  // -- sparkline per column: kings in reign order, oldest -> newest --------
  function sparkline(col, rows) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 84 20");
    svg.setAttribute("class", "mini");
    const kings = rows.filter((r) => r.kind === "king" && !r.revoked).slice().sort((a, b) => a.order - b.order).reverse();
    const pts = kings.map((r, i) => ({ i, r, c: r.cells[col.key] })).filter((p) => p.c && p.c.score !== null && p.c.score !== undefined);
    if (pts.length < 2) return svg;
    const teacher = rows.find((r) => r.kind === "teacher");
    const tc = teacher && teacher.cells[col.key];
    const vals = pts.map((p) => p.c.score).concat(tc && tc.score !== null && tc.score !== undefined ? [tc.score] : []);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const span = Math.max(hi - lo, 5);
    const x = (i) => 3 + i * (78 / Math.max(1, kings.length - 1));
    const y = (v) => 17 - 14 * ((v - lo) / span);
    const mk = (tag, attrs) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };
    if (tc && tc.score !== null && tc.score !== undefined) {
      svg.append(mk("line", { x1: 0, x2: 84, y1: y(tc.score).toFixed(1), y2: y(tc.score).toFixed(1), stroke: "rgba(90,200,250,0.55)", "stroke-width": 1, "stroke-dasharray": "2 2" }));
    }
    svg.append(mk("path", { d: pts.map((p, k) => `${k ? "L" : "M"}${x(p.i).toFixed(1)},${y(p.c.score).toFixed(1)}`).join(" "), fill: "none", stroke: "#f3c449", "stroke-width": 1.2 }));
    for (const p of pts) {
      const c = mk("circle", { cx: x(p.i).toFixed(1), cy: y(p.c.score).toFixed(1), r: p.r.current ? 2.2 : 1.5, fill: "#f3c449" });
      const t = document.createElementNS(ns, "title");
      t.textContent = `${p.r.label}: ${fmt(p.c.score)}`;
      c.append(t);
      svg.append(c);
    }
    return svg;
  }

  // -- render -----------------------------------------------------------------------
  function sortedRows(m) {
    const rows = m.rows.slice();
    const teacher = rows.find((r) => r.kind === "teacher");
    const rest = rows.filter((r) => r !== teacher);
    const key = state.sort.key;
    if (key) {
      const val = (r) => { const c = r.cells[key]; return c && c.score !== null && c.score !== undefined ? c.score : null; };
      rest.sort((a, b) => {
        const va = val(a), vb = val(b);
        if (va === null && vb === null) return a.order - b.order;
        if (va === null) return 1;
        if (vb === null) return -1;
        return (vb - va) * (state.sort.desc ? 1 : -1) || a.order - b.order;
      });
    } else {
      rest.sort((a, b) => a.order - b.order);
    }
    return teacher ? [teacher, ...rest] : rest;
  }

  function render() {
    const m = state.matrix;
    if (!m) return;
    const thead = $("#matrix thead"), tbody = $("#matrix tbody");
    const cols = m.columns;
    const bench = cols.filter((c) => c.kind === "bench"), envs = cols.filter((c) => c.kind === "env");
    const teacher = m.rows.find((r) => r.kind === "teacher") || { cells: {} };
    const firstEnv = envs.length ? envs[0].key : null;

    // header: group row, column row, sparkline row
    const groups = el("tr", { class: "groups" },
      el("th", { class: "model sticky" }, ""),
      el("th", { class: "total" }, ""),
      el("th", { colspan: bench.length || 1 }, `held-out benchmarks (${bench.length})`),
      el("th", { colspan: envs.length || 1, class: "gap" }, `datagen environments (${envs.length})`));
    const head = el("tr", { class: "cols" });
    const modelTh = el("th", { class: "model sticky" + (state.sort.key ? "" : " sorted"), title: "reign order (newest first)" }, "model");
    modelTh.addEventListener("click", () => { state.sort = { key: null, desc: true }; render(); });
    head.append(modelTh);
    for (const c of cols) {
      const cls = ["col", c.kind === "total" ? "total" : "", c.key === firstEnv ? "first-env" : "",
        state.sort.key === c.key ? "sorted" : "", state.sort.key === c.key && state.sort.desc ? "desc" : ""].filter(Boolean).join(" ");
      const th = el("th", { class: cls, title: (c.note || c.env_id || c.label) + (state.sort.key === c.key ? (state.sort.desc ? " · sorted high → low" : " · sorted low → high") : " · click to sort") },
        el("span", { class: "lbl" }, c.label + (state.sort.key === c.key ? (state.sort.desc ? " ▼" : " ▲") : "")));
      th.addEventListener("click", () => {
        if (state.sort.key === c.key) state.sort.desc = !state.sort.desc;
        else state.sort = { key: c.key, desc: true };
        render();
      });
      head.append(th);
    }
    const spark = el("tr", { class: "spark" }, el("th", { class: "model sticky" }, el("span", { class: "dim" }, "kings over time · dashed = teacher")));
    for (const c of cols) spark.append(el("th", { class: (c.kind === "total" ? "total" : "") + (c.key === firstEnv ? " first-env" : "") }, sparkline(c, m.rows)));
    thead.replaceChildren(groups, head, spark);

    // body
    const out = [];
    for (const r of sortedRows(m)) {
      const tr = el("tr", { class: [r.kind, r.current ? "current" : "", r.revoked ? "revoked" : ""].filter(Boolean).join(" ") });
      const badge = r.current ? el("span", { class: "badge crowned" }, "current")
        : r.revoked ? el("span", { class: "badge rejected", title: r.revoked_reason }, "removed") : null;
      const sub = r.kind === "king"
        ? `king-${r.digest12} · crowned ${r.revoked ? dayShort(r.crowned_at) + " · removed " + dayShort(r.revoked_at) : isoShort(r.crowned_at)}`
        : r.sub;
      tr.append(el("td", { class: "model", title: r.revoked ? r.revoked_reason : (r.hotkey ? "hotkey " + r.hotkey : "") },
        el("span", { class: "name" }, r.label), badge, el("span", { class: "sub" }, sub)));
      for (const c of cols) {
        const cell = r.cells[c.key];
        const has = cell && cell.score !== null && cell.score !== undefined;
        const cls = ["cell", c.kind === "total" ? "total" : "", c.key === firstEnv ? "first-env" : "",
          has ? "" : "blank", r.kind === "teacher" ? "teacher-ref" : ""].filter(Boolean).join(" ");
        // removed reigns keep their tint at half strength: on the record, not in the race
        const style = has && r.kind !== "teacher" ? tint(cell.delta, r.revoked ? 0.5 : 1) : "";
        const td = el("td", { class: cls, style: style || null }, has ? fmt(cell.score) : "·");
        td.addEventListener("mouseenter", (e) => showTip(e, cellTip(r, c, cell, teacher.cells[c.key])));
        td.addEventListener("mousemove", moveTip);
        td.addEventListener("mouseleave", hideTip);
        tr.append(td);
      }
      out.push(tr);
    }
    tbody.replaceChildren(...out);

    // meta, legend, definitions
    const kings = m.rows.filter((r) => r.kind === "king");
    const withData = m.rows.filter((r) => r.n_cells > 0).length;
    $("#matrix-meta").textContent = `${kings.length} reigns (${kings.filter((r) => r.revoked).length} removed) · ${withData} of ${m.rows.length} rows with data · ${bench.length} benchmarks · ${envs.length} environments · ${m.n_cards} cards · built ${isoShort(m.generated_at)}`;
    const ramp = el("span", { class: "ramp" });
    for (const d of [-40, -30, -20, -10, -4, 4, 10, 20, 30, 40]) ramp.append(el("i", { style: tint(d) || "background-color: transparent", title: `${signed(d, 0)} pt vs teacher` }));
    $("#matrix-legend").replaceChildren(
      el("span", {}, "cell tint = score − teacher:"), el("span", {}, "−40"), ramp, el("span", {}, "+40 pt"),
      el("span", {}, " · "), el("span", { class: "badge crowned" }, "current"), el("span", {}, "reigning king"),
      el("span", { class: "badge rejected" }, "removed"), el("span", {}, "crown revoked by the operator (row kept for the record)"),
      el("span", {}, " · blank = no measurement"));
    $("#matrix-definitions").replaceChildren(...Object.entries(m.definitions || {}).map(([k, v]) => el("li", {}, el("b", {}, k + ": "), v)));
    const cards = (m.cards || []).map((c) => `${c.run_id}${c.reign !== null && c.reign !== undefined ? " (reign " + c.reign + ")" : c.label ? " (" + c.label + ")" : ""}${c.used ? "" : " — not a model row"}`);
    $("#matrix-cards").textContent = cards.length ? "benchmark cards read: " + cards.join(" · ") : "no benchmark cards found";
  }

  async function load() {
    try {
      const res = await fetch(`/api/matrix?t=${Date.now()}`, { cache: "no-store" });
      if (res.status === 503) { $("#matrix-meta").textContent = "first build running — the matrix appears after the next refresh pass"; return; }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.matrix = await res.json();
      render();
    } catch (e) {
      $("#matrix-meta").textContent = `load failed: ${e.message}`;
    }
  }

  load();
  setInterval(load, REFRESH_MS);
})();
