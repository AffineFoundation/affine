/** Kings vs teacher — performance over time (#kings on the main page).
 *
 * Renders /api/v1/matrix (ops/kingboard/build.py::build_matrix) as two
 * stacked tables with the same rows (teacher, genesis, kings newest first):
 *   1. held-out benchmarks   — total + one column per benchmark card env
 *   2. datagen environments  — total + one column per environment, grouped
 * Value = average score 0–100, tint = gap to the teacher. Row names come
 * from charts.js kingName, the Reign table's source. Monitoring only.
 */

import { fetchMatrix } from "./api.js?v=68";
import { kingName } from "./charts.js?v=73";

const REFRESH_MS = 300000;
const TINT_FULL_PT = 40;      // |Δ| in points where the tint saturates
const TINT_MAX_ALPHA = 0.40;

const TABLES = [
  { id: "kings-bench", title: "held-out benchmarks", total: "total:bench", kind: "bench",
    caption: "greedy T=0 · never in D · score = tasks passed" },
  { id: "kings-env", title: "datagen environments", total: "total:env", kind: "env",
    caption: "solve rate over the king seat's rollouts · teacher = teacher_* rollouts" },
];

const state = {
  matrix: null,
  sort: {},             // table id -> { key, desc }
  showAll: false,       // rows without any measurement
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
  if (r.kind === "teacher") return `${r.model} — the frozen teacher (the score's fixed point)`;
  if (r.kind === "genesis") return `${r.model} — reign 0 (${kingName(0)}), the seed king; never won a duel`;
  return `${kingName(r.reign)} · reign ${r.reign} · king-${r.digest12}\ncrowned ${when(r.crowned_at)}`
    + (r.challenge_id ? ` · ${r.challenge_id}` : "")
    + (r.hotkey ? `\nhotkey ${r.hotkey}` : "")
    + (r.current ? "\ncurrent king" : "");
}

function cellTip(row, col, cell, teacherCell) {
  const head = `${rowName(row)} · ${col.label}${col.kind === "env" && col.group ? ` (${col.group})` : ""}`;
  if (!cell || cell.score == null) {
    return `${head}\n${cell?.reason || "no measurement"}`
      + (col.kind === "bench" ? "\nno benchmark card for this model yet" : "");
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
    lines.push(`card ${cell.run_id}${cell.mode ? ` · ${cell.mode}` : ""}`);
  } else {
    lines.push(`${num(cell.solved)} solved / ${num(cell.n)} graded (${num(cell.rollouts)} rollouts, ${num(cell.errored)} errored)`);
    lines.push(`datagen rollouts on ${col.env}${col.env_id ? ` · ${col.env_id}` : ""}`);
  }
  if (row.kind !== "teacher" && teacherCell && teacherCell.score != null) {
    lines.push(`teacher ${fmt(teacherCell.score)} → Δ ${signed(cell.delta)} pt`);
  }
  if (col.note) lines.push(col.note);
  return lines.join("\n");
}

function tableColumns(m, spec) {
  const total = m.columns.find((c) => c.key === spec.total);
  return [total, ...m.columns.filter((c) => c.kind === spec.kind)].filter(Boolean);
}

function visibleRows(m, spec) {
  const teacher = m.rows.find((r) => r.kind === "teacher");
  let rest = m.rows.filter((r) => r !== teacher);
  if (!state.showAll) {
    rest = rest.filter((r) => r.current || r.kind === "genesis"
      || Object.keys(r.cells).some((k) => k.startsWith(`${spec.kind}:`) && r.cells[k].score != null));
  }
  const sort = state.sort[spec.id] || { key: null, desc: true };
  const val = (r) => (r.cells[sort.key] && r.cells[sort.key].score != null ? r.cells[sort.key].score : null);
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
    const name = c.kind === "env" ? (c.group || "other") : "";
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
  const blocks = spec.kind === "env" ? groupBlocks(cols) : [];
  const sepAt = new Set(blocks.filter((b) => b.name).map((b) => b.first));
  const mark = (key) => (sort.key === key ? (sort.desc ? " ▾" : " ▴") : "");

  const groupRow = blocks.length
    ? `<tr class="blocks"><th class="model"></th>${blocks.map((b) =>
      `<th class="block${sepAt.has(b.first) ? " sep" : ""}" colspan="${b.n}" title="${esc(b.name)}">${esc(b.name.replace("_", " "))}</th>`).join("")}</tr>`
    : "";
  const head = `<tr><th class="model${sort.key ? "" : " sorted"}" data-sort="" title="reign order (newest first)">model</th>`
    + cols.map((c) => {
      const cls = ["col", c.kind, sepAt.has(c.key) ? "sep" : "", sort.key === c.key ? "sorted" : ""].filter(Boolean).join(" ");
      const title = `${c.label}${c.kind === "env" && c.group ? ` · ${c.group}` : ""}${c.kind === "bench" && c.n ? ` · n = ${c.n}` : ""}${c.note ? `\n${c.note}` : ""}\nclick to sort`;
      return `<th class="${cls}" data-sort="${esc(c.key)}" title="${esc(title)}">${esc(c.short || c.abbr || c.label)}${mark(c.key)}</th>`;
    }).join("") + "</tr>";

  const body = rows.map((r) => {
    const cls = [r.kind, r.current ? "current" : ""].filter(Boolean).join(" ");
    const cells = cols.map((c) => {
      const cell = r.cells[c.key];
      const has = cell && cell.score != null;
      const tcls = ["cell", c.kind, sepAt.has(c.key) ? "sep" : "", has ? "" : "blank"].filter(Boolean).join(" ");
      const style = has && r.kind !== "teacher" ? tint(cell.delta) : "";
      return `<td class="${tcls} duel-hit" data-tip="${esc(cellTip(r, c, cell, teacher.cells[c.key]))}"`
        + `${style ? ` style="${style}"` : ""}>${has ? fmt(cell.score, spec.kind === "env" ? 0 : 1) : "·"}</td>`;
    }).join("");
    return `<tr class="${cls}"><td class="model duel-hit" data-tip="${esc(rowTip(r))}">`
      + `<span class="name">${esc(rowName(r))}</span>${r.current ? `<i class="cur" title="current king"></i>` : ""}</td>${cells}</tr>`;
  }).join("");

  wrap.innerHTML = `<table class="data-table kings ${spec.kind}"><thead>${groupRow}${head}</thead><tbody>${body}</tbody></table>`;
  const meta = $(`${spec.id}-meta`);
  if (meta) meta.textContent = `${cols.length - 1} columns · ${rows.length} of ${m.rows.length} rows · ${spec.caption}`;
}

function render() {
  const m = state.matrix;
  if (!m) return;
  for (const spec of TABLES) renderTable(m, spec);
  const kings = m.rows.filter((r) => r.kind === "king");
  const meta = $("kings-meta");
  if (meta) meta.textContent = `${kings.length} reigns · ${m.columns.filter((c) => c.kind === "bench").length} benchmarks · ${m.columns.filter((c) => c.kind === "env").length} environments · built ${when(m.generated_at)}`;
  const withData = m.rows.filter((r) => r.n_cells > 0 || r.current || r.kind === "genesis" || r.kind === "teacher").length;
  const hidden = m.rows.length - withData;
  const showAll = $("kings-show-all");
  if (showAll) {
    showAll.textContent = state.showAll ? "hide reigns without data" : `show all reigns${hidden ? ` (+${hidden})` : ""}`;
    showAll.hidden = !hidden && !state.showAll;
  }
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
  $("kings-show-all")?.addEventListener("click", () => { state.showAll = !state.showAll; render(); });
}

async function refresh() {
  const m = await fetchMatrix().catch(() => null);
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
