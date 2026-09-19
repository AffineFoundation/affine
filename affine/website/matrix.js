/** Kings vs teacher — performance over time (#kings on the main page).
 *
 * Renders /api/v1/matrix (ops/kingboard/build.py::build_matrix) as two
 * stacked tables with the same rows (teacher, genesis, kings newest first):
 *   1. held-out benchmarks   — total + one column per benchmark card env
 *   2. datagen environments  — total + one column per environment, grouped
 * Value = average score 0–100, tint = gap to the teacher. Row names come
 * from charts.js kingName, the Reign table's source. Monitoring only.
 */

import { fetchDatasetTable, fetchMatrix } from "./api.js?v=69";
import { kingName } from "./charts.js?v=75";

const REFRESH_MS = 300000;
const TINT_FULL_PT = 40;      // |Δ| in points where the tint saturates
const TINT_MAX_ALPHA = 0.40;

const TABLES = [
  { id: "kings-bench", title: "held-out benchmarks", total: "total:bench", kind: "bench",
    caption: "greedy T=0 · never in D · score = tasks passed" },
  { id: "kings-env", title: "datagen environments", total: "total:env", kind: "env",
    caption: "solve rate over sampled (T=0.8) rollouts · king seat + coverage backfill · teacher = teacher_* rollouts · greyed = < 30 rollouts" },
];

const state = {
  matrix: null,
  dataset: null,
  sort: {},             // table id -> { key, desc }
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 1) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const signed = (x, d = 1) => (x == null ? "–" : `${x > 0 ? "+" : ""}${Number(x).toFixed(d)}`);
const num = (x) => (x == null ? "–" : Number(x).toLocaleString());
const when = (iso) => (iso ? String(iso).replace("T", " ").slice(0, 16) + " UTC" : "–");

function tint(delta) {
  if (delta == null) return "";
  const a = Math.min(TINT_MAX_ALPHA, (Math.abs(delta) / TINT_FULL_PT) * TINT_MAX_ALPHA);
  if (a < 0.02) return "";
  return delta > 0
    ? `background-color: rgba(68, 255, 154, ${a.toFixed(3)})`
    : `background-color: rgba(255, 71, 71, ${a.toFixed(3)})`;
}

// Same names as the Reign table: reign n → Affine-<roman(n + 1)> (charts.js
// kingName); the teacher and the genesis seed keep their plain labels.
function rowName(r) {
  if (r.kind === "king" && r.reign != null) return kingName(r.reign);
  return r.label;
}

function rowTip(r) {
  if (r.kind === "reference") return r.tip || `${r.model} — reference model; not a king`;
  if (r.kind === "teacher") return `${r.model} — the frozen teacher (the score's fixed point)`;
  if (r.kind === "genesis") return `${r.model} — reign 0 (${kingName(0)}), the seed king; never won a duel`;
  return `${kingName(r.reign)} · reign ${r.reign} · king-${r.digest12}\ncrowned ${when(r.crowned_at)}`
    + (r.challenge_id ? ` · ${r.challenge_id}` : "")
    + (r.hotkey ? `\nhotkey ${r.hotkey}` : "")
    + (r.current ? "\ncurrent king" : "");
}

function cellTip(row, col, cell, teacherCell) {
  const head = `${rowName(row)} · ${col.label}${col.kind === "env" && col.group ? ` (${col.group})` : ""}`;
  if (cell && cell.running) {
    return `${head}\n${col.kind === "bench" ? "benchmark set" : "cell"} ${cell.state || "running"}`
      + (cell.eta ? `\nETA ≈ ${when(cell.eta)} (pass average per cell; sandbox sets take longer)` : "\nETA: first cell not finished yet")
      + `\npass ${cell.run_id}`;
  }
  if (!cell || cell.score == null) {
    return `${head}\n${cell?.reason || "no measurement"}`
      + (col.kind === "bench" ? "\nnever run for this model (no benchmark card)" : "");
  }
  const lines = [head];
  if (col.kind.startsWith("total")) {
    lines.push(`${fmt(cell.score)} · mean of ${cell.n_cols} columns`);
    if (cell.teacher_same_cols != null) {
      lines.push(`teacher on the same ${cell.n_same_cols} columns: ${fmt(cell.teacher_same_cols)} → Δ ${signed(cell.delta)} pt`);
    }
    return lines.join("\n");
  }
  const ci = cell.lo != null ? ` [${fmt(cell.lo)}–${fmt(cell.hi)}] 95% CI` : "";
  lines.push(`${fmt(cell.score)}${ci}`);
  if (col.kind === "bench") {
    lines.push(`n = ${num(cell.n)} tasks · greedy T=0${cell.metric === "finished_only" ? " · finished-only" : ""}`);
    if (cell.all_rollouts != null) lines.push(`all rollouts (timeouts count as failed): ${fmt(cell.all_rollouts)}`);
    if (cell.cap_bound) lines.push(`‡ cap-bound: ${Math.round(100 * cell.cap_frac)}% of rollouts hit the completion cap (scored 0); lower bound`);
    if (cell.graded === "llm_judge") lines.push(`⚖ judge-graded: ${judgeText(cell.judge)} — advisory, never part of the score`);
    lines.push(`card ${cell.run_id}${cell.mode ? ` · ${cell.mode}` : ""}`);
  } else {
    lines.push(`${num(cell.solved)} solved / ${num(cell.n)} graded (${num(cell.rollouts)} rollouts, ${num(cell.errored)} errored)${cell.temp ? ` · ${cell.temp} T=0.8` : ""}`);
    if (cell.low_n) lines.push(`fewer than 30 graded rollouts — provisional until the backfill lands`);
    if (cell.greedy && cell.greedy.n) lines.push(`greedy T=0 (not pooled in): ${cell.greedy.score != null ? fmt(cell.greedy.score) : "–"} over ${num(cell.greedy.n)} graded`);
    lines.push(`datagen rollouts on ${col.env}${col.env_id ? ` · ${col.env_id}` : ""}`);
  }
  if (row.kind !== "teacher" && teacherCell && teacherCell.score != null) {
    lines.push(`teacher ${fmt(teacherCell.score)} → Δ ${signed(cell.delta)} pt`);
  }
  if (col.note) lines.push(col.note);
  return lines.join("\n");
}

const judgeText = (j) => j && (j.model || j.via)
  ? `LLM judge ${j.model || "?"}${j.via ? ` via ${j.via}` : ""}${j.temperature != null ? ` (T=${j.temperature})` : ""}`
  : "LLM judge";

function tableColumns(m, spec) {
  return m.columns.filter((c) => c.kind === spec.kind);
}

const hasValue = (r, key) => r.cells[key] != null && r.cells[key].score != null;
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

/** Common column set = the columns every displayed row WITH data has a
 * value for (rows still empty or running do not shrink it). */
function commonColumns(rows, cols) {
  const dataRows = rows.filter((r) => cols.some((c) => hasValue(r, c.key)));
  if (!dataRows.length) return [];
  return cols.filter((c) => dataRows.every((r) => hasValue(r, c.key)));
}

function totals(rows, cols, common, teacher) {
  const over = (r, cs) => (cs.length && cs.every((c) => hasValue(r, c.key)) ? mean(cs.map((c) => r.cells[c.key].score)) : null);
  const tCommon = teacher ? over(teacher, common) : null;
  const tFull = teacher ? over(teacher, cols) : null;
  const out = new Map();
  for (const r of rows) {
    const c = over(r, common), f = over(r, cols);
    out.set(r.key, {
      common: c, full: f,
      commonDelta: c != null && tCommon != null && r.kind !== "teacher" ? c - tCommon : null,
      fullDelta: f != null && tFull != null && r.kind !== "teacher" ? f - tFull : null,
      running: c == null && cols.some((cc) => (r.cells[cc.key] || {}).running),
    });
  }
  return out;
}

function visibleRows(m, spec) {
  // the API's row set: teacher, kings newest first, genesis last (kings
  // without any measurement are under m.hidden, never rendered)
  const teacher = m.rows.find((r) => r.kind === "teacher");
  const rest = m.rows.filter((r) => r !== teacher);
  const sort = state.sort[spec.id] || { key: null, desc: true };
  const cols = tableColumns(m, spec).filter((c) => !c.advisory);
  const tot = totals([...(teacher ? [teacher] : []), ...rest], cols, commonColumns([...(teacher ? [teacher] : []), ...rest], cols), teacher);
  const val = (r) => sort.key === "total" ? tot.get(r.key).common
    : sort.key === "full" ? tot.get(r.key).full
    : (r.cells[sort.key] && r.cells[sort.key].score != null ? r.cells[sort.key].score : null);
  rest.sort((a, b) => {
    if (!sort.key) return a.order - b.order;
    const va = val(a), vb = val(b);
    if (va == null && vb == null) return a.order - b.order;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (vb - va) * (sort.desc ? 1 : -1) || a.order - b.order;
  });
  return teacher ? [teacher, ...rest] : rest;
}

// Environment columns carry their fold group; a separator goes before the
// first column of each group and a label row names the groups.
function groupBlocks(cols) {
  const blocks = [];
  for (const c of cols) {
    const name = c.kind === "env" ? (c.group || "other") : c.kind === "bench" ? (c.group === "agentic" ? "agentic" : "") : "";
    const last = blocks[blocks.length - 1];
    if (last && last.name === name) { last.n += 1; continue; }
    blocks.push({ name, n: 1, first: c.key });
  }
  return blocks;
}

function renderTable(m, spec) {
  const wrap = $(`${spec.id}-wrap`);
  if (!wrap) return;
  const cols = tableColumns(m, spec);
  const rows = visibleRows(m, spec);
  const teacher = m.rows.find((r) => r.kind === "teacher") || { cells: {} };
  const sort = state.sort[spec.id] || { key: null, desc: true };
  // judge-graded (advisory) columns are shown but never enter total / full
  const scoreCols = cols.filter((c) => !c.advisory);
  const common = commonColumns(rows, scoreCols);
  const tot = totals(rows, scoreCols, common, m.rows.find((r) => r.kind === "teacher"));
  const commonTitle = `mean over the ${common.length} columns every displayed row with data has a value for:\n`
    + (common.map((c) => c.short || c.abbr || c.label).join(", ") || "(none)")
    + `\nrecomputed as rows change; Δ vs the teacher on the same columns${scoreCols.length < cols.length ? "; judge-graded columns excluded" : ""}`;
  const fullTitle = `mean over all ${scoreCols.length} scored columns — only for rows that have every column; blank otherwise`;
  const allBlocks = groupBlocks(cols);
  // the environments table always shows its fold groups; the benchmarks table
  // gets a block row only once an agentic column exists
  const blocks = spec.kind === "env" || allBlocks.some((b) => b.name === "agentic") ? allBlocks : [];
  const sepAt = new Set(blocks.filter((b) => b.name).map((b) => b.first));
  const mark = (key) => (sort.key === key ? (sort.desc ? " ▾" : " ▴") : "");

  const groupRow = blocks.length
    ? `<tr class="blocks"><th class="model"></th><th></th><th></th>${blocks.map((b) =>
      `<th class="block${sepAt.has(b.first) ? " sep" : ""}" colspan="${b.n}" title="${esc(b.name)}">${esc(b.name.replace("_", " "))}</th>`).join("")}</tr>`
    : "";
  const head = `<tr><th class="model${sort.key ? "" : " sorted"}" data-sort="" title="reign order (newest first)">model</th>`
    + `<th class="col total${sort.key === "total" ? " sorted" : ""}" data-sort="total" title="${esc(commonTitle)}\nclick to sort">total${mark("total")}</th>`
    + `<th class="col full${sort.key === "full" ? " sorted" : ""}" data-sort="full" title="${esc(fullTitle)}\nclick to sort">full${mark("full")}</th>`
    + cols.map((c) => {
      const cls = ["col", c.kind, sepAt.has(c.key) ? "sep" : "", sort.key === c.key ? "sorted" : ""].filter(Boolean).join(" ");
      const judge = c.advisory ? `\n⚖ judge-graded: ${judgeText(c.judge)} — advisory, never part of the score` : "";
      const title = `${c.label}${c.kind === "env" && c.group ? ` · ${c.group}` : ""}${c.kind === "bench" && c.group ? ` · ${c.group}` : ""}${c.kind === "bench" && c.n ? ` · n = ${c.n}` : ""}${judge}${c.note ? `\n${c.note}` : ""}\nclick to sort`;
      return `<th class="${cls}${c.advisory ? " advisory" : ""}" data-sort="${esc(c.key)}" title="${esc(title)}">${esc(c.short || c.abbr || c.label)}${c.advisory ? `<span class="mk judge">⚖</span>` : ""}${mark(c.key)}</th>`;
    }).join("") + "</tr>";

  const body = rows.map((r) => {
    const cls = [r.kind, r.current ? "current" : ""].filter(Boolean).join(" ");
    const t = tot.get(r.key);
    const d = spec.kind === "env" ? 0 : 1;
    const totalTip = `${rowName(r)} · total (common columns)\n`
      + (t.common != null ? `${fmt(t.common)} over ${common.length} columns` + (t.commonDelta != null ? ` · teacher on the same columns → Δ ${signed(t.commonDelta)} pt` : "")
        : t.running ? "benchmark pass running — total appears when its cells land" : "no value on one of the common columns")
      + `\n${common.map((c) => c.short || c.abbr || c.label).join(", ")}`;
    const fullTip = `${rowName(r)} · full total\n` + (t.full != null ? `${fmt(t.full)} over all ${scoreCols.length} columns` + (t.fullDelta != null ? ` · Δ ${signed(t.fullDelta)} pt vs teacher` : "")
      : `blank: the row has ${scoreCols.filter((c) => hasValue(r, c.key)).length} of ${scoreCols.length} scored columns`);
    const totalTd = `<td class="cell total${t.common == null ? " blank" : ""} duel-hit" data-tip="${esc(totalTip)}"`
      + `${t.common != null && r.kind !== "teacher" ? ` style="${tint(t.commonDelta)}"` : ""}>${t.common != null ? fmt(t.common, d) : t.running ? "…" : "·"}</td>`;
    const fullTd = `<td class="cell full${t.full == null ? " blank" : ""} duel-hit" data-tip="${esc(fullTip)}">${t.full != null ? fmt(t.full, d) : "·"}</td>`;
    const cells = cols.map((c) => {
      const cell = r.cells[c.key];
      const has = cell && cell.score != null;
      const running = !has && cell && cell.running;
      const tcls = ["cell", c.kind, sepAt.has(c.key) ? "sep" : "", has ? (cell.low_n ? "lown" : "") : running ? "running" : "blank"].filter(Boolean).join(" ");
      const style = has && r.kind !== "teacher" ? tint(cell.delta) : "";
      const marks = has ? `${cell.cap_bound ? `<span class="mk cap">‡</span>` : ""}${cell.graded === "llm_judge" ? `<span class="mk judge">⚖</span>` : ""}` : "";
      return `<td class="${tcls} duel-hit" data-tip="${esc(cellTip(r, c, cell, teacher.cells[c.key]))}"`
        + `${style ? ` style="${style}"` : ""}>${has ? fmt(cell.score, d) + marks : running ? "…" : "·"}</td>`;
    }).join("");
    return `<tr class="${cls}"><td class="model duel-hit" data-tip="${esc(rowTip(r))}">`
      + `<span class="name">${esc(rowName(r))}</span>${r.current ? `<i class="cur" title="current king"></i>` : ""}</td>${totalTd}${fullTd}${cells}</tr>`;
  }).join("");

  wrap.innerHTML = `<table class="data-table kings ${spec.kind}"><thead>${groupRow}${head}</thead><tbody>${body}</tbody></table>`;
}

// -- Dataset D: columns = sources (+ king groups), rows = metrics ----------
const compact = (n) => {
  if (n == null) return "·";
  const v = Number(n);
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e4) return `${Math.round(v / 1e3)}k`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return `${Math.round(v)}`;
};

function datasetValue(row, col) {
  const v = col[row.key];
  if (row.fmt === "bool") return v == null ? "·" : (v ? "yes" : "no");
  if (v == null) return "·";
  if (row.fmt === "pct") return `${(100 * v).toFixed(v < 0.1 ? 1 : 0)}%`;
  if (row.fmt === "score") return `${Math.round(v)}`;
  if (row.fmt === "draws") return v >= 10 ? `${Math.round(v)}` : v.toFixed(1);
  return compact(v);
}

function datasetTip(row, col) {
  const gm = col.group_meta || {};
  const head = `${col.label}${col.kind === "env" ? ` (${col.group})` : " (fold group)"} · ${row.label}`;
  const v = col[row.key];
  const lines = [head];
  if (row.key === "turns_per_duel") {
    lines.push(v == null ? "no draws published" : `${Number(v).toFixed(2)} turns of ${state.dataset.header.n_per_duel ?? 1300} per duel`);
    if (gm.draws_per_duel != null) lines.push(`group ${col.group}: ${Number(gm.draws_per_duel).toFixed(1)} draws per duel = share ${(100 * gm.share).toFixed(1)}% (target ${gm.static_mix != null ? (100 * gm.static_mix).toFixed(0) + "%" : "–"})`);
    if (col.strata_share_in_group != null) lines.push(`this source holds ${(100 * col.strata_share_in_group).toFixed(1)}% of the group's strata`);
    lines.push(col.turns_per_duel_note || "");
  } else if (row.key === "supply_limited") {
    lines.push(v == null ? "no cap for this group" : `${v ? "yes" : "no"} — group strata ${num(gm.strata)} vs cap ${num(gm.cap)} (${gm.cap_source || ""})`);
    if (gm.share != null) lines.push(`slice share ${(100 * gm.share).toFixed(1)}% vs [mix] target ${gm.static_mix != null ? (100 * gm.static_mix).toFixed(0) + "%" : "–"}`);
  } else if (row.key === "curriculum_share") {
    lines.push(v == null ? "no curriculum weight for this group" : `${(100 * v).toFixed(2)}% of the slice for group ${col.group} under the v1.2 shadow rule (not applied yet)`);
  } else if (row.key === "king_solve") {
    lines.push(v == null ? "no graded king rollouts on this source" : `${Number(v).toFixed(1)}% solved over ${num(col.king_solve_n)} graded rollouts (current king)`);
  } else if (row.key === "turns" || row.key === "strata") {
    lines.push(v == null ? "not in D" : `${num(v)} ${row.key} in D` + (col.kind === "env" ? ` under group ${col.group}` : ""));
    if (gm.turns != null) lines.push(`group ${col.group}: ${num(gm.turns)} turns, ${num(gm.strata)} strata (slice keys${gm.sub_strata_k > 1 ? `, sub-strata k = ${gm.sub_strata_k}` : ""})`);
  } else {
    lines.push(v == null ? "not a datagen source (fold-routed group)" : `${num(v)} rollouts`);
  }
  if (row.note) lines.push(row.note);
  return lines.filter(Boolean).join("\n");
}

function renderDataset() {
  const d = state.dataset;
  const wrap = $("kings-dataset-wrap");
  if (!d || !wrap) return;
  const cols = d.columns || [];
  const blocks = [];
  for (const c of cols) {
    const name = c.kind === "env" ? (c.group || "other") : "king groups";
    const last = blocks[blocks.length - 1];
    if (last && last.name === name) { last.n += 1; continue; }
    blocks.push({ name, n: 1, first: c.key });
  }
  const sepAt = new Set(blocks.map((b) => b.first));
  const groupRow = `<tr class="blocks"><th class="model"></th>${blocks.map((b) =>
    `<th class="block sep" colspan="${b.n}" title="${esc(b.name)}">${esc(b.name.replace("_", " "))}</th>`).join("")}</tr>`;
  const head = `<tr><th class="model">metric</th>${cols.map((c) => {
    const gm = c.group_meta || {};
    const title = `${c.label}${c.kind === "env" ? ` · group ${c.group}${c.env_id ? ` · ${c.env_id}` : ""}` : " · fold group (routed from king / teacher rollouts)"}`
      + `${gm.turns != null ? `\ngroup: ${num(gm.turns)} turns · ${num(gm.strata)} strata · ${Number(gm.draws_per_duel || 0).toFixed(1)} draws/duel` : ""}`;
    return `<th class="col ${c.kind}${sepAt.has(c.key) ? " sep" : ""}" title="${esc(title)}">${esc(c.abbr)}</th>`;
  }).join("")}</tr>`;
  const body = (d.rows || []).map((r) => {
    const cells = cols.map((c) => {
      const v = c[r.key];
      const cls = ["cell", c.kind, sepAt.has(c.key) ? "sep" : "", v == null ? "blank" : "",
        r.fmt === "bool" && v === true ? "warn" : ""].filter(Boolean).join(" ");
      return `<td class="${cls} duel-hit" data-tip="${esc(datasetTip(r, c))}">${esc(datasetValue(r, c))}</td>`;
    }).join("");
    return `<tr class="metric${r.headline ? " headline" : ""}"><td class="model duel-hit" data-tip="${esc(`${r.label}\n${r.note || ""}`)}"><span class="name">${esc(r.short || r.label)}</span></td>${cells}</tr>`;
  }).join("");
  wrap.innerHTML = `<table class="data-table kings dataset"><thead>${groupRow}${head}</thead><tbody>${body}</tbody></table>`;
  const h = d.header || {};
  const rec = h.recurrence || {};
  const meta = $("kings-dataset-meta");
  if (meta) {
    meta.textContent = `D = ${num(h.n_turns)} turns · ${num(h.n_strata)} strata · epoch ${h.epoch ?? "–"}`
      + `${h.manifest_sha12 ? ` · manifest ${h.manifest_sha12}` : ""}`
      + ` · ${h.n_per_duel ?? 1300} turns per duel`
      + `${rec.rollout_overlap != null ? ` · simulated recurrence between two duels: ${(100 * rec.rollout_overlap).toFixed(1)}% of rollouts, ${(100 * rec.turn_overlap).toFixed(1)}% of turns` : ""}`
      + `${h.curriculum_mode ? ` · curriculum ${h.curriculum_mode} (${h.curriculum_rule || ""}, for epoch ${h.curriculum_for_epoch ?? "–"})` : ""}`;
  }
}

function render() {
  const m = state.matrix;
  if (!m) return;
  for (const spec of TABLES) renderTable(m, spec);
  renderDataset();   // no-op on affine.io (no #kings-dataset-wrap); api/v1/dataset_table stays served
}

function wire() {
  for (const spec of TABLES) {
    $(`${spec.id}-wrap`)?.addEventListener("click", (e) => {
      const th = e.target instanceof Element && e.target.closest("th[data-sort]");
      if (!th) return;
      const key = th.dataset.sort;
      const cur = state.sort[spec.id] || { key: null, desc: true };
      if (!key) state.sort[spec.id] = { key: null, desc: true };
      else if (cur.key === key) state.sort[spec.id] = { key, desc: !cur.desc };
      else state.sort[spec.id] = { key, desc: true };
      render();
    });
  }
}

async function refresh() {
  const wantDataset = Boolean($("kings-dataset-wrap"));
  const [m, d] = await Promise.all([fetchMatrix().catch(() => null), wantDataset ? fetchDatasetTable().catch(() => null) : null]);
  if (d && Array.isArray(d.columns)) state.dataset = d;
  else if (wantDataset && !state.dataset) {
    $("kings-dataset-wrap").innerHTML = `<div class="empty">dataset table not built yet</div>`;
  }
  if (!m || !Array.isArray(m.rows)) {
    if (!state.matrix) {
      for (const spec of TABLES) {
        const wrap = $(`${spec.id}-wrap`);
        if (wrap) wrap.innerHTML = `<div class="empty">matrix not built yet</div>`;
      }
    }
    return;
  }
  state.matrix = m;
  render();
}

export function initMatrix() {
  if (!$("kings-bench-wrap")) return;
  wire();
  refresh();
  setInterval(refresh, REFRESH_MS);
}
