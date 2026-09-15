/* Affine kingboard front end: renders /api/stats.json, refreshes every 60 s. */
(function () {
  "use strict";

  const REFRESH_S = 60;
  const GROUP_ORDER = ["coding", "terminal", "math", "tool_use", "nl2repo", "agent", "other"];

  const state = {
    stats: null,
    reign: null,                    // digest12 of the selected reign
    trendEnv: "__all__",
    sort: { env: { key: "source", desc: false }, harness: { key: "n", desc: true } },
    countdown: REFRESH_S,
  };

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs, ...kids) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    for (const kid of kids) {
      if (kid === null || kid === undefined) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  };

  const pct = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : (100 * x).toFixed(d) + "%";
  const num = (x) => (x === null || x === undefined) ? "–" : Number.isInteger(x) ? x.toLocaleString() : (+x).toFixed(1);
  const short = (s, n = 8) => s ? (s.length > 2 * n ? s.slice(0, n) + "…" + s.slice(-4) : s) : "–";
  const ago = (ts) => {
    if (!ts) return "–";
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 90) return Math.round(s) + " s ago";
    if (s < 5400) return Math.round(s / 60) + " min ago";
    if (s < 172800) return (s / 3600).toFixed(1) + " h ago";
    return (s / 86400).toFixed(1) + " d ago";
  };
  const isoShort = (iso) => iso ? iso.replace("T", " ").replace(/(\+00:00|Z)$/, " UTC").slice(0, 20) : "–";

  // -- header cards ------------------------------------------------------------
  function renderCards(st) {
    const k = st.king || {};
    const reign = currentReign();
    const cards = [
      el("div", { class: "card king" },
        el("div", { class: "k" }, "current king"),
        el("div", { class: "v" }, k.reign !== undefined && k.reign !== null ? `reign ${k.reign}` : "unknown",
          el("small", {}, k.digest12 ? `king-${k.digest12}` : "")),
        el("div", { class: "small muted mono" }, k.hotkey ? `hotkey ${short(k.hotkey, 6)}` : "",
          k.crowned_at ? ` · crowned ${isoShort(k.crowned_at)}` : "")),
      el("div", { class: "card" },
        el("div", { class: "k" }, "last rollout in the data"),
        el("div", { class: "v" }, ago(st.last_rollout_ts)),
        el("div", { class: "small muted" }, `manifest ${isoShort(st.manifest && st.manifest.published_at)}`)),
      el("div", { class: "card" },
        el("div", { class: "k" }, "king rollouts 1 h / 24 h"),
        el("div", { class: "v" }, `${num(st.recent.king_1h)} / ${num(st.recent.king_24h)}`),
        el("div", { class: "small muted" }, `all seats ${num(st.recent.all_1h)} / ${num(st.recent.all_24h)}`)),
      el("div", { class: "card" },
        el("div", { class: "k" }, "rollouts indexed"),
        el("div", { class: "v" }, num(st.counts.rollouts),
          el("small", {}, `${num(st.counts.king)} king · ${num(st.counts.teacher)} teacher`)),
        el("div", { class: "small muted" }, `${num(st.manifest && st.manifest.n_chunks)} chunks · stats ${ago(st.generated_ts)}`)),
    ];
    if (reign) {
      const t = reign.total;
      cards.push(el("div", { class: "card" },
        el("div", { class: "k" }, `selected reign overall`),
        el("div", { class: "v" }, pct(t.rate), el("small", {}, `${t.solved}/${t.graded} graded`)),
        el("div", { class: "small muted" }, `teacher ${pct(st.teacher.total.rate)} on ${num(st.teacher.total.graded)} graded`)));
    }
    const box = $("#cards");
    box.replaceChildren(...cards);
  }

  // -- reign selector ------------------------------------------------------------
  function currentReign() {
    const st = state.stats;
    if (!st) return null;
    return st.reigns.find((r) => r.digest12 === state.reign) || null;
  }

  function reignLabel(r) {
    const name = r.reign !== null && r.reign !== undefined ? `reign ${r.reign}` : "reign ? (not in state.json)";
    return `${name} · king-${r.digest12} · ${r.n_rollouts.toLocaleString()} rollouts${r.current ? " · current" : ""}`;
  }

  function renderReignSelect(st) {
    const sel = $("#reign");
    if (!state.reign || !st.reigns.some((r) => r.digest12 === state.reign)) {
      const cur = st.reigns.find((r) => r.current) || st.reigns[0];
      state.reign = cur ? cur.digest12 : null;
    }
    sel.replaceChildren(...st.reigns.map((r) => {
      const o = el("option", { value: r.digest12 }, reignLabel(r));
      if (r.digest12 === state.reign) o.selected = true;
      return o;
    }));
    const r = currentReign();
    $("#reign-meta").textContent = r
      ? `${r.hotkey ? "hotkey " + short(r.hotkey, 6) + " · " : ""}data ${isoShort(new Date(r.first_ts * 1000).toISOString())} → ${ago(r.last_ts)} · ${num(r.recent_24h)} rollouts in the last 24 h`
      : "no king rollouts found yet";
  }

  // -- tables ----------------------------------------------------------------------
  function rateCell(row, cls) {
    if (row.rate === null || row.rate === undefined) return el("td", { class: "num muted" }, "–");
    const bar = el("span", { class: "bar" + (cls || ""), style: `width:${Math.round(44 * row.rate)}px` });
    return el("td", { class: "num" }, bar, pct(row.rate),
      el("span", { class: "ci" }, `[${pct(row.lo, 0)}–${pct(row.hi, 0)}]`));
  }

  function sparkline(series) {
    // series: [{bucket, rate, n}] with bucket 0 = newest; draw oldest→newest.
    const pts = (series || []).filter((b) => b.rate !== null && b.rate !== undefined).sort((a, b) => b.bucket - a.bucket);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 110 22");
    if (pts.length === 0) return svg;
    const x = (b) => 4 + (13 - b.bucket) * (102 / 13);
    const y = (r) => 19 - 16 * r;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pts.map((p, i) => `${i ? "L" : "M"}${x(p).toFixed(1)},${y(p.rate).toFixed(1)}`).join(" "));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "var(--king)");
    path.setAttribute("stroke-width", "1.5");
    svg.append(path);
    for (const p of pts) {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", x(p).toFixed(1)); c.setAttribute("cy", y(p.rate).toFixed(1));
      c.setAttribute("r", Math.min(3, 1 + Math.log10(1 + p.n)).toFixed(1));
      c.setAttribute("fill", "var(--king)");
      const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
      t.textContent = `${p.bucket === 0 ? "last 24 h" : p.bucket + " d ago"}: ${pct(p.rate)} (n=${p.n})`;
      c.append(t);
      svg.append(c);
    }
    return svg;
  }

  function sortRows(rows, spec, extra) {
    const get = extra || ((r, k) => r[k]);
    const cmp = (a, b) => {
      const va = get(a, spec.key), vb = get(b, spec.key);
      const na = va === null || va === undefined, nb = vb === null || vb === undefined;
      if (na && nb) return 0;
      if (na) return 1;
      if (nb) return -1;
      if (typeof va === "string") return va.localeCompare(vb) * (spec.desc ? -1 : 1);
      return (va - vb) * (spec.desc ? -1 : 1);
    };
    return rows.slice().sort(cmp);
  }

  function renderEnvTable(st) {
    const r = currentReign();
    const tbody = $("#env-table tbody");
    const spec = state.sort.env;
    markSorted("#env-table", spec);
    if (!r) {
      tbody.replaceChildren(el("tr", {}, el("td", { colspan: 12, class: "empty" }, "No king rollouts in the trace store yet.")));
      return;
    }
    const get = (row, k) => k === "teacher_rate" ? (row.teacher ? row.teacher.rate : null) : row[k];
    const groups = new Map();
    for (const row of r.envs) {
      if (!groups.has(row.group)) groups.set(row.group, []);
      groups.get(row.group).push(row);
    }
    const order = [...groups.keys()].sort((a, b) => {
      const ia = GROUP_ORDER.indexOf(a), ib = GROUP_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
    const out = [];
    for (const g of order) {
      const rows = sortRows(groups.get(g), spec, get);
      const n = rows.reduce((s, x) => s + x.n, 0);
      const solved = rows.reduce((s, x) => s + x.solved, 0);
      const graded = rows.reduce((s, x) => s + x.graded, 0);
      const tSolved = rows.reduce((s, x) => s + (x.teacher ? x.teacher.solved : 0), 0);
      const tGraded = rows.reduce((s, x) => s + (x.teacher ? x.teacher.graded : 0), 0);
      out.push(el("tr", { class: "group" }, el("td", { colspan: 12 }, g,
        el("span", { class: "muted" }, `${n.toLocaleString()} king rollouts · king ${graded ? pct(solved / graded) : "–"} vs teacher ${tGraded ? pct(tSolved / tGraded) : "–"}`))));
      for (const row of rows) {
        const t = row.teacher;
        const delta = row.delta;
        out.push(el("tr", {},
          el("td", {}, row.source, el("span", { class: "env-id" }, `${row.env_id}${row.harnesses.length ? " · " + row.harnesses.join(", ") : ""}`)),
          el("td", { class: "num" }, num(row.n)),
          rateCell(row),
          el("td", { class: "num" }, num(row.solved)),
          el("td", { class: "num" }, num(row.failed)),
          el("td", { class: "num" + (row.errored ? " neg" : "") }, num(row.errored)),
          el("td", { class: "num" }, num(row.unscored)),
          el("td", { class: "num" }, num(row.median_turns)),
          el("td", { class: "num" }, pct(row.timeout_rate)),
          t && t.rate !== null && t.rate !== undefined
            ? el("td", { class: "num" }, el("span", { class: "bar t", style: `width:${Math.round(44 * t.rate)}px` }), pct(t.rate), el("span", { class: "n" }, `(${t.graded.toLocaleString()})`))
            : el("td", { class: "num muted" }, t ? `– (${t.n} ungraded)` : "no teacher data"),
          el("td", { class: "num " + (delta === null || delta === undefined ? "muted" : delta >= 0 ? "pos" : "neg") },
            delta === null || delta === undefined ? "–" : (delta >= 0 ? "+" : "") + (100 * delta).toFixed(1) + " pt"),
          el("td", { class: "spark" }, sparkline(r.trend[row.source]))));
      }
    }
    tbody.replaceChildren(...out);
  }

  function renderHarnessTable(st) {
    const r = currentReign();
    const tbody = $("#harness-table tbody");
    const spec = state.sort.harness;
    markSorted("#harness-table", spec);
    if (!r) { tbody.replaceChildren(); return; }
    const rows = sortRows(r.harnesses, spec);
    tbody.replaceChildren(...rows.map((row) => el("tr", {},
      el("td", {}, row.harness),
      el("td", { class: "num" }, num(row.n)),
      rateCell(row),
      el("td", { class: "num" }, num(row.solved)),
      el("td", { class: "num" }, num(row.failed)),
      el("td", { class: "num" + (row.errored ? " neg" : "") }, num(row.errored)),
      el("td", { class: "num" }, num(row.unscored)),
      el("td", { class: "num" }, num(row.median_turns)),
      el("td", { class: "num" }, pct(row.timeout_rate)))));
  }

  function markSorted(tableSel, spec) {
    document.querySelectorAll(`${tableSel} th`).forEach((th) => {
      th.classList.toggle("sorted", th.dataset.key === spec.key);
      th.classList.toggle("desc", th.dataset.key === spec.key && spec.desc);
    });
  }

  function bindSorting(tableSel, which, defaultDescKeys) {
    document.querySelectorAll(`${tableSel} th[data-key]`).forEach((th) => {
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        const spec = state.sort[which];
        if (spec.key === key) spec.desc = !spec.desc;
        else { spec.key = key; spec.desc = defaultDescKeys.includes(key); }
        render();
      });
    });
  }

  // -- trend ---------------------------------------------------------------------------
  function pooled(trendByEnv, envKey) {
    // -> [{bucket, n, solved, rate}] pooled over envs (or one env)
    const cells = new Map();
    const envs = envKey === "__all__" ? Object.keys(trendByEnv) : [envKey];
    for (const e of envs) {
      for (const b of trendByEnv[e] || []) {
        const c = cells.get(b.bucket) || { bucket: b.bucket, n: 0, solved: 0 };
        c.n += b.n; c.solved += b.solved;
        cells.set(b.bucket, c);
      }
    }
    return [...cells.values()].map((c) => ({ ...c, rate: c.n ? c.solved / c.n : null })).sort((a, b) => b.bucket - a.bucket);
  }

  function renderTrendSelect(st) {
    const sel = $("#trend-env");
    const envs = st.envs.slice().sort((a, b) => (a.group + a.source).localeCompare(b.group + b.source));
    const opts = [el("option", { value: "__all__" }, "all envs (pooled)")];
    for (const e of envs) opts.push(el("option", { value: e.source }, `${e.group} / ${e.source}`));
    sel.replaceChildren(...opts);
    sel.value = envs.some((e) => e.source === state.trendEnv) ? state.trendEnv : "__all__";
    state.trendEnv = sel.value;
  }

  function renderTrend(st) {
    const svg = $("#trend");
    const W = 640, H = 260, L = 44, R = 12, T = 14, B = 30;
    const ns = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs, text) => {
      const n = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      if (text !== undefined) n.textContent = text;
      return n;
    };
    svg.replaceChildren();
    const x = (bucket) => L + (13 - bucket) * ((W - L - R) / 13);
    const y = (rate) => T + (1 - rate) * (H - T - B);
    for (let i = 0; i <= 4; i++) {
      const yy = y(i / 4);
      svg.append(mk("line", { x1: L, x2: W - R, y1: yy, y2: yy, stroke: "#30363d", "stroke-width": 1 }));
      svg.append(mk("text", { x: L - 6, y: yy + 4, fill: "#8b949e", "font-size": 11, "text-anchor": "end" }, `${i * 25}%`));
    }
    for (const b of [13, 9, 6, 3, 0]) {
      const anchor = b === 13 ? "start" : b === 0 ? "end" : "middle";
      svg.append(mk("text", { x: x(b), y: H - 10, fill: "#8b949e", "font-size": 11, "text-anchor": anchor }, b === 0 ? "last 24 h" : `${b} d ago`));
    }
    const drawSeries = (series, color, width, dash) => {
      const pts = series.filter((p) => p.rate !== null);
      if (!pts.length) return;
      const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.bucket).toFixed(1)},${y(p.rate).toFixed(1)}`).join(" ");
      const path = mk("path", { d, fill: "none", stroke: color, "stroke-width": width });
      if (dash) path.setAttribute("stroke-dasharray", dash);
      svg.append(path);
      for (const p of pts) {
        const c = mk("circle", { cx: x(p.bucket), cy: y(p.rate), r: Math.min(5, 1.5 + Math.log10(1 + p.n)), fill: color });
        c.append(mk("title", {}, `${p.bucket === 0 ? "last 24 h" : p.bucket + " d ago"}: ${pct(p.rate)} of ${p.n} graded`));
        svg.append(c);
      }
    };
    const cur = currentReign();
    for (const r of st.reigns) {
      if (cur && r.digest12 === cur.digest12) continue;
      drawSeries(pooled(r.trend, state.trendEnv), "#6e7681", 1, "3 3");
    }
    drawSeries(pooled(st.teacher.trend, state.trendEnv), "#58a6ff", 2);
    if (cur) drawSeries(pooled(cur.trend, state.trendEnv), "#f2cc60", 2.5);
    const k = cur ? pooled(cur.trend, state.trendEnv) : [];
    const t = pooled(st.teacher.trend, state.trendEnv);
    const kn = k.reduce((s, p) => s + p.n, 0), tn = t.reduce((s, p) => s + p.n, 0);
    $("#trend-note").textContent = `graded rollouts in the last 14 days: king ${kn.toLocaleString()}, teacher ${tn.toLocaleString()}. Dot size grows with the bucket's sample size; buckets are 24 h wide and end at the last stats build.`;
  }

  // -- definitions ---------------------------------------------------------------------
  function renderDefinitions(st) {
    const d = st.definitions || {};
    $("#definitions").replaceChildren(...Object.entries(d).map(([k, v]) => el("li", {}, el("b", {}, k + ": "), v)));
    const stops = Object.entries(st.stop_conditions || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(" · ");
    $("#stops").textContent = `stop conditions over all rollouts: ${stops}`;
    $("#build-info").textContent = `Last stats build ${isoShort(st.generated_at)} (${st.build_seconds}s, read via ${st.manifest && st.manifest.read_mode}).`;
  }

  // -- main -------------------------------------------------------------------------------
  function render() {
    const st = state.stats;
    if (!st) return;
    renderCards(st);
    renderReignSelect(st);
    renderEnvTable(st);
    renderHarnessTable(st);
    renderTrendSelect(st);
    renderTrend(st);
    renderDefinitions(st);
  }

  async function load() {
    try {
      const res = await fetch(`/api/stats.json?t=${Date.now()}`, { cache: "no-store" });
      if (res.status === 503) {
        const body = await res.json();
        $("#cards").replaceChildren(el("div", { class: "card" }, el("div", { class: "k" }, "status"),
          el("div", { class: "v" }, "first build running"),
          el("div", { class: "small muted" }, body.builder && body.builder.running ? "reading the trace store…" : "waiting for the builder")));
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.stats = await res.json();
      render();
    } catch (e) {
      $("#refresh-note").textContent = `refresh failed: ${e.message}`;
    }
  }

  $("#reign").addEventListener("change", (e) => { state.reign = e.target.value; render(); });
  $("#trend-env").addEventListener("change", (e) => { state.trendEnv = e.target.value; renderTrend(state.stats); });
  bindSorting("#env-table", "env", ["n", "rate", "solved", "failed", "errored", "unscored", "median_turns", "timeout_rate", "teacher_rate", "delta"]);
  bindSorting("#harness-table", "harness", ["n", "rate", "solved", "failed", "errored", "unscored", "median_turns", "timeout_rate"]);

  load();
  setInterval(() => {
    state.countdown -= 1;
    if (state.countdown <= 0) { state.countdown = REFRESH_S; load(); }
    $("#countdown").textContent = state.countdown;
  }, 1000);
})();
