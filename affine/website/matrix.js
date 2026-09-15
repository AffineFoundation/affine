/** Kings vs teacher — performance over time (#kings on the main page).
 *
 * Renders /api/v1/matrix (ops/kingboard/build.py::build_matrix): one row per
 * model in reign order (teacher, genesis, kings newest first), one column
 * per held-out benchmark and per datagen environment group, value = average
 * score 0–100, tint = gap to the teacher. Monitoring only — never scored.
 */

import { fetchMatrix } from "./api.js?v=68";

const REFRESH_MS = 300000;
const TINT_FULL_PT = 40;      // |Δ| in points where the tint saturates
const TINT_MAX_ALPHA = 0.40;

const state = {
  matrix: null,
  sort: { key: null, desc: true },
  showAll: false,       // rows without any measurement
  expandEnvs: false,    // per-environment columns instead of one per group
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

function rowTip(r) {
  if (r.kind === "teacher") return `${r.model} — the frozen teacher (the score's fixed point)`;
  if (r.kind === "genesis") return `${r.model} — reign 0, the seed king; never won a duel`;
  return `reign ${r.reign} · king-${r.digest12}\ncrowned ${when(r.crowned_at)}`
    + (r.challenge_id ? ` · ${r.challenge_id}` : "")
    + (r.hotkey ? `\nhotkey ${r.hotkey}` : "")
    + (r.current ? "\ncurrent king" : "");
}

function cellTip(row, col, cell, teacherCell) {
  const head = `${row.label} · ${col.label}`;
  if (!cell || cell.score == null) {
    return `${head}\n${cell?.reason || "no measurement"}`
      + (col.kind === "bench" ? "\nno benchmark card for this model yet" : "");
  }
  const lines = [head];
  if (col.kind === "total") {
    lines.push(`${fmt(cell.score)} · mean of ${cell.n_cols} columns (${cell.n_bench} benchmarks, ${cell.n_env} environment groups)`);
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
    lines.push(`card ${cell.run_id}${cell.mode ? ` · ${cell.mode}` : ""}`);
  } else if (col.kind === "group") {
    lines.push(`${num(cell.solved)} solved / ${num(cell.n)} graded over ${cell.envs.length} environments (${num(cell.rollouts)} rollouts)`);
    lines.push(cell.envs.join(", "));
  } else {
    lines.push(`${num(cell.solved)} solved / ${num(cell.n)} graded (${num(cell.rollouts)} rollouts, ${num(cell.errored)} errored)`);
    lines.push(`datagen rollouts on ${col.env}${col.env_id ? ` · ${col.env_id}` : ""}`);
  }
  if (row.kind !== "teacher" && teacherCell && teacherCell.score != null) {
    lines.push(`teacher ${fmt(teacherCell.score)} → Δ ${signed(cell.delta)} pt`);
  }
  if (col.note && col.kind !== "group") lines.push(col.note);
  return lines.join("\n");
}

function visibleColumns(m) {
  const kinds = state.expandEnvs ? ["total", "bench", "env"] : ["total", "bench", "group"];
  return m.columns.filter((c) => kinds.includes(c.kind));
}

function visibleRows(m) {
  const teacher = m.rows.find((r) => r.kind === "teacher");
  let rest = m.rows.filter((r) => r !== teacher);
  if (!state.showAll) rest = rest.filter((r) => r.n_cells > 0 || r.current || r.kind === "genesis");
  const key = state.sort.key;
  const val = (r) => (r.cells[key] && r.cells[key].score != null ? r.cells[key].score : null);
  rest.sort((a, b) => {
    if (!key) return a.order - b.order;
    const va = val(a), vb = val(b);
    if (va == null && vb == null) return a.order - b.order;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (vb - va) * (state.sort.desc ? 1 : -1) || a.order - b.order;
  });
  return teacher ? [teacher, ...rest] : rest;
}

function render() {
  const m = state.matrix;
  const wrap = $("kings-wrap");
  if (!m || !wrap) return;
  const cols = visibleColumns(m);
  const rows = visibleRows(m);
  const teacher = m.rows.find((r) => r.kind === "teacher") || { cells: {} };
  const firstEnvKey = (cols.find((c) => c.kind === "group" || c.kind === "env") || {}).key;
  const sortMark = (key) => (state.sort.key === key ? (state.sort.desc ? " ▾" : " ▴") : "");

  const head = [
    `<th class="model${state.sort.key ? "" : " sorted"}" data-sort="" title="reign order (newest first)">model</th>`,
    ...cols.map((c) => {
      const cls = ["r", "col", c.kind, c.key === firstEnvKey ? "first-env" : "",
        state.sort.key === c.key ? "sorted" : ""].filter(Boolean).join(" ");
      const title = `${c.label}${c.kind === "bench" && c.n ? ` · n = ${c.n}` : ""}${c.note ? `\n${c.note}` : ""}\nclick to sort`;
      return `<th class="${cls}" data-sort="${esc(c.key)}" title="${esc(title)}">${esc(c.abbr || c.label)}${sortMark(c.key)}</th>`;
    }),
  ].join("");

  const body = rows.map((r) => {
    const cls = [r.kind, r.current ? "current" : ""].filter(Boolean).join(" ");
    const cells = cols.map((c) => {
      const cell = r.cells[c.key];
      const has = cell && cell.score != null;
      const tcls = ["r", "cell", c.kind, c.key === firstEnvKey ? "first-env" : "", has ? "" : "blank"].filter(Boolean).join(" ");
      const style = has && r.kind !== "teacher" ? tint(cell.delta) : "";
      return `<td class="${tcls} duel-hit" data-tip="${esc(cellTip(r, c, cell, teacher.cells[c.key]))}"`
        + `${style ? ` style="${style}"` : ""}>${has ? fmt(cell.score) : "·"}</td>`;
    }).join("");
    return `<tr class="${cls}"><td class="model duel-hit" data-tip="${esc(rowTip(r))}">`
      + `<span class="name">${esc(r.label)}</span>${r.current ? `<i class="cur" title="current king"></i>` : ""}</td>${cells}</tr>`;
  }).join("");

  wrap.innerHTML = `<table class="data-table kings"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

  const kings = m.rows.filter((r) => r.kind === "king");
  const hidden = m.rows.length - rows.length;
  const meta = $("kings-meta");
  if (meta) {
    meta.textContent = `${kings.length} reigns · ${m.columns.filter((c) => c.kind === "bench").length} benchmarks · `
      + `${m.columns.filter((c) => c.kind === "env").length} environments in ${m.columns.filter((c) => c.kind === "group").length} groups`
      + ` · built ${when(m.generated_at)}`;
  }
  const showAll = $("kings-show-all");
  if (showAll) {
    showAll.textContent = state.showAll ? "hide reigns without data" : `show all reigns${hidden ? ` (+${hidden})` : ""}`;
    showAll.hidden = !hidden && !state.showAll;
  }
  const expand = $("kings-expand");
  if (expand) expand.textContent = state.expandEnvs ? "group environments" : "expand environments";
}

function wire() {
  const wrap = $("kings-wrap");
  if (!wrap) return;
  wrap.addEventListener("click", (e) => {
    const th = e.target instanceof Element && e.target.closest("th[data-sort]");
    if (!th) return;
    const key = th.dataset.sort;
    if (!key) state.sort = { key: null, desc: true };
    else if (state.sort.key === key) state.sort.desc = !state.sort.desc;
    else state.sort = { key, desc: true };
    render();
  });
  $("kings-show-all")?.addEventListener("click", () => { state.showAll = !state.showAll; render(); });
  $("kings-expand")?.addEventListener("click", () => { state.expandEnvs = !state.expandEnvs; render(); });
}

async function refresh() {
  const m = await fetchMatrix().catch(() => null);
  if (!m || !Array.isArray(m.rows)) {
    if (!state.matrix) {
      const wrap = $("kings-wrap");
      if (wrap) wrap.innerHTML = `<div class="empty">matrix not built yet</div>`;
    }
    return;
  }
  state.matrix = m;
  render();
}

export function initMatrix() {
  if (!$("kings-wrap")) return;
  wire();
  refresh();
  setInterval(refresh, REFRESH_MS);
}
