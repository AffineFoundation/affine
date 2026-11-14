/* SN120 v2 status site. One column schema and one renderer for every table, as on the v1 site (k1-tables.js).
   The benchmark table, its score cards and the chart tiles copy the structure of the v1 site (bench.js, k1.js, timing.js).
   A page reads one file. index.html reads site-data.json, which build.py writes from the publisher's public snapshot.
   v1.html (<body data-page="v1">) reads v1-data.json, the frozen v1 history that v1_archive.py wrote once at cutover.
   Privacy: build.py puts a batch link and a LoRA hash on an applied winner only; every other batch shows "private". */
(() => {
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const clip = (s, a = 6, b = 4) => { s = String(s || ''); return s.length > a + b + 1 ? s.slice(0, a) + '…' + (b ? s.slice(-b) : '') : s || '—'; };
  const sgn = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
  const dur = s => {
    const m = Math.round(Math.max(0, s) / 60), h = Math.floor(m / 60);
    return m < 1 ? '<1m' : m < 60 ? m + 'm' : h < 48 ? h + 'h' + (m % 60 ? ' ' + (m % 60) + 'm' : '') : Math.floor(h / 24) + 'd' + (h % 24 ? ' ' + (h % 24) + 'h' : '');
  };
  const utc = t => new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  const mute = (s, t) => `<span class="mute"${t ? ` title="${esc(t)}"` : ''}>${esc(s)}</span>`;
  const pct = x => x == null ? mute('—') : (100 * x).toFixed(1) + '%';
  const ago = t => t ? `<span title="${utc(t)}">${dur(Date.now() / 1000 - t)} ago</span>` : mute('—');
  const key = (v, a, b) => v ? `<button type="button" class="key" data-copy="${esc(v)}" title="${esc(v)} (click to copy)">${esc(clip(v, a, b))}</button>` : mute('—');
  const link = (u, s, t) => `<a href="${esc(u)}"${t ? ` title="${esc(t)}"` : ''}>${esc(s)}</a>`;
  const pill = (s, c, t) => `<span class="pill ${c}"${t ? ` title="${esc(t)}"` : ''}>${esc(s)}</span>`;
  const score = (x, d, t) => x == null ? mute('—') : `<span class="${x > 0 ? 'up' : x < 0 ? 'dn' : 'mute'}" title="${esc(t)}">${sgn(x, d)}</span>`;
  const RESULT = {
    winner: ['WINNER', 'up', 'Beat the head M by more than the winner bar.'],
    lost: ['LOST', 'dn', 'Had a fair tournament and did not win. The registration is used up.'],
    no_decision: ['NO DECISION', 'wait', 'Our infrastructure failed. Not charged; the seat goes back to the front of the queue.'],
    skip: ['SKIP', 'private', 'The file was missing or invalid when the round froze it. Not charged; the seat keeps its place.'],
    not_trained: ['NOT TRAINED', 'wait', 'The training time of the round ended before this batch was trained. Not charged; the seat is first in the next round.'],
    not_scored: ['NOT SCORED', 'wait', 'The time limit of the round came before the first look of the scoring. Not charged; the seat is first in the next round.'],
  };
  const why = r => r.reason ? ` The validator's text: "${r.reason}"` : '';
  const OUTCOME = {
    applied: r => pill('UID ' + r.winner_uid + ' applied', 'up', 'The winning LoRA is on the head now.'),
    null: r => mute('M stays', 'No LoRA cleared the bar. The head did not change.' + why(r)),
    no_decision: r => pill('no decision', 'wait', 'No LoRA was judged: a failure on our side, or the time of the round ended first. Nobody is charged.' + why(r)),
    abandoned: r => pill('abandoned', 'wait', 'The files of the named winner could not be verified. Nobody is charged.' + why(r)),
  };
  const FLAG = {
    leased: ['in round', 'up', 'In the round that runs now.'],
    front: ['front', 'wait', 'Sent back to the front, not charged: a failure on our side, or the time of the round ended first.'],
    demoted: ['back of queue', 'wait', 'Moved to the back: no valid file for too long. A seat with a valid file is never moved back.'],
    held: ['held', 'wait', 'Parked for the operator after repeated infrastructure failures. Not charged.'],
  };
  const V1 = document.body.dataset.page === 'v1', DATA = V1 ? 'v1-data.json' : 'site-data.json';
  let D = null, filter = '';

  const files = r => r.batch
    ? link(r.batch, 'batch.json', 'The winning batch, byte for byte') + (r.lora ? ' ' + key(r.lora, 6, 0) : '')
    : r.imported ? link('v1.html', 'v1 king', 'Carried over from v1 at cutover. The v1 history is on its own page.')
    : pill('private', 'private', 'Only a winner is published. This batch stays private.');
  const dagger = (c, t) => c ? `<sup title="${esc(t)}">†</sup>` : '';

  // h = header, w = width in px, n = numeric (right aligned), t = header tooltip, r = cell from a row
  const C = {
    rank: { h: '#', w: 36, n: 1, r: r => mute(r.rank) },
    line: { h: '#', w: 36, n: 1, r: r => mute(r.line), t: 'Place in line (1 = first).' },
    round: { h: 'Round', w: 92, r: r => r.file_url ? link(r.file_url, r.round_id, 'round.json') : esc(r.round_id || '—') },
    uid: { h: 'UID', w: 52, n: 1, r: r => esc(r.uid ?? '—') },
    hot: { h: 'Hotkey', w: 104, r: r => key(r.hotkey) },
    z: { h: 'z', w: 60, n: 1, r: r => score(r.z, 2, 'z = gain / standard error'), t: 'Gain over the head M, in standard errors.' },
    head: { h: 'Head', w: 132, r: r => key(r.head, 8, 0) + (r.current ? ' ' + pill('current', 'up') : ''), t: 'Head version id.' },
    files: { h: 'Files', w: 136, r: files, t: 'Published for a winner only: the batch, and the hash of the LoRA.' },
    reg: { h: 'Reg block', w: 88, n: 1, r: r => esc(r.registration_block ?? '—') },
    check: { h: 'File', w: 96, r: r => r.file_valid ? pill('valid', 'up', 'The stored file passes every check.') : mute('no valid file', 'This seat is skipped, is not charged, and keeps its place.'), t: 'Live check of the stored file. It never shows file content.' },
    flags: { h: 'Status', w: 170, r: r => r.flags.map(f => pill(...FLAG[f])).join(' ') || mute('waiting') },
    result: { h: 'Result', w: 112, r: r => pill(...(RESULT[r.status] || [r.status, 'private', ''])) },
    gain: { h: 'Gain', w: 76, n: 1, r: r => score(r.delta == null ? null : 100 * r.delta, 2, `Pass rate minus the pass rate of M, in percentage points (standard error ${(100 * r.se).toFixed(2)})`), t: 'LoRA pass rate minus head pass rate, in percentage points.' },
    bar: { h: 'vs bar', w: 68, n: 1, r: r => score(r.z == null ? null : r.z - D.z_bar, 2, `z minus the winner bar (${D.z_bar})`), t: 'z minus the winner bar. Above zero clears the bar.' },
    capHit: { h: 'Cap-hit', w: 68, n: 1, r: r => pct(r.cap_rate), t: 'Share of answers that hit the token cap. Such an answer is not passed.' },
    look: { h: 'Dropped', w: 68, n: 1, r: r => r.look == null ? mute('—') : 'look ' + (r.look + 1), t: 'The early look at which this LoRA was dropped, if any.' },
    sha: { h: 'Batch sha256', w: 100, r: r => key(r.sha, 8, 0) },
    at: { h: 'When', w: 92, n: 1, r: r => ago(r.at) },
    date: { h: 'Date (UTC)', w: 124, n: 1, r: r => r.at ? esc(utc(r.at).slice(0, 16)) : mute('—') },
    env: { h: 'Env', w: 60, r: r => esc(r.env) },
    seats: { h: 'Batches', w: 68, n: 1, r: r => esc(r.seats), t: 'Seats of the round: the batches with a valid file at round open. A batch whose training failed is counted.' },
    skipped: { h: 'Skipped', w: 68, n: 1, r: r => esc(r.skipped), t: 'Seats with no valid file at round open. Not charged.' },
    tasks: { h: 'Tasks', w: 64, n: 1, r: r => esc(r.tasks) },
    mPass: { h: 'M pass', w: 68, n: 1, r: r => pct(r.null_pass) },
    mCap: { h: 'M cap-hit', w: 76, n: 1, r: r => pct(r.null_cap) },
    outcome: { h: 'Outcome', w: 132, r: r => (OUTCOME[r.outcome] || (() => esc(r.outcome)))(r) },
    kind: { h: 'Kind', w: 84, r: r => r.folded ? mute('folded', 'The LoRAs are merged into the base.') : mute('stacked', 'The winning LoRA is stacked on the base.') },
    manifest: { h: 'Manifest', w: 84, r: r => r.manifest ? link(r.manifest, 'head.json', 'File hashes of this head') : mute('—') },
    epoch: { h: 'Epoch', w: 64, n: 1, r: r => 'E' + esc(r.epoch) },
    verdict: { h: 'Result', w: 92, r: r => pill(...({ green: ['GREEN', 'up', 'Applied to the head.'], red: ['RED', 'dn', 'Not applied.'] }[r.verdict] || [r.verdict, 'wait', 'Not decided when v1 stopped.'])) },
    share: { h: 'Weight', w: 68, n: 1, r: r => pct(r.share), t: 'Share of the miner weight when v1 stopped.' },
    batch: { h: 'Batch', w: 92, n: 1, r: r => r.tasks == null ? mute('—') : esc(r.tasks) + (r.rollouts == null ? '' : ' · ' + esc(r.rollouts)), t: 'Tasks in the batch, then rollouts.' },
    cPass: { h: 'Cand. pass', w: 84, n: 1, r: r => pct(r.c_pass), t: 'Pass rate of the trained candidate on the scoring sample.' },
    hPass: { h: 'Head pass', w: 84, n: 1, r: r => pct(r.h_pass), t: 'Pass rate of the head on the same sample.' },
    cand: { h: 'Candidate', w: 100, r: r => key(r.candidate, 8, 0), t: 'Hash of the trained candidate.' },
  };
  const T = {
    kings: { cols: ['rank', 'uid', 'hot', 'round', 'at', 'z', 'head', 'files'], empty: 'No king yet.' },
    queue: { cols: ['line', 'uid', 'hot', 'reg', 'check', 'flags'], empty: 'No registration waits in the queue.' },
    history: { cols: ['round', 'uid', 'hot', 'result', 'gain', 'z', 'bar', 'capHit', 'look', 'sha', 'files', 'at'], empty: 'No result yet.' },
    rounds: { cols: ['round', 'env', 'head', 'seats', 'skipped', 'tasks', 'mPass', 'mCap', 'outcome', 'at'], empty: 'No round yet.' },
    heads: { cols: ['head', 'round', 'kind', 'manifest', 'at'], empty: 'No head yet.' },
    v1kings: { cols: ['rank', 'uid', 'hot', 'epoch', 'z', 'share'], empty: 'No king.' },
    v1turns: { cols: ['epoch', 'uid', 'hot', 'verdict', 'z', 'cPass', 'hPass', 'capHit', 'batch', 'cand', 'head', 'date'], empty: 'No turn.' },
  };

  const table = n => document.querySelector(`#${n} table`);
  function shell(n) {
    const t = table(n), cols = T[n].cols.map(k => C[k]);
    t.querySelectorAll('colgroup,thead').forEach(el => el.remove());
    t.style.minWidth = cols.reduce((a, c) => a + c.w, 0) + 'px';
    t.insertAdjacentHTML('afterbegin', '<colgroup>' + cols.map(c => `<col style="width:${c.w}px">`).join('') + '</colgroup><thead><tr>'
      + cols.map(c => `<th class="${c.n ? 'num' : ''}"${c.t ? ` title="${esc(c.t)}"` : ''}>${esc(c.h)}</th>`).join('') + '</tr></thead>');
  }
  function fill(n, rows, reading, more) {
    const cols = T[n].cols, body = table(n).tBodies[0], box = body.closest('.scroll');
    body.innerHTML = (rows.length
      ? rows.map(r => '<tr>' + cols.map(k => `<td class="${C[k].n ? 'num' : ''}">${C[k].r(r)}</td>`).join('') + '</tr>').join('')
      : `<tr><td colspan="${cols.length}" class="empty">${esc(T[n].empty)}</td></tr>`)
      + (more ? `<tr class="more"><td colspan="${cols.length}">${esc(more)}</td></tr>` : '');
    document.querySelector(`#${n} .reading`).textContent = reading;
    if (box.classList.contains('wait')) { box.classList.remove('wait'); body.classList.add('fade'); }
  }

  /* One chart for every tile and every score card, as k1.js and bench.js of the v1 site: grid lines with their values, bars or a
     line with hollow points per series (a line breaks where a value is missing), one dashed reference line, the first and the last
     x label, and a tooltip for the position under the pointer. o.n positions; o.x(i) in 0..1 places them by time, else they are
     even slots. The box has a fixed size in the CSS, so a chart never moves the page. */
  const NS = 'http://www.w3.org/2000/svg';
  const svg = (t, a, text) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (text != null) e.textContent = text; return e; };
  const nice = (lo, hi, n, least) => {
    const raw = Math.max((hi - lo) / n, least || 0) || 1, p = 10 ** Math.floor(Math.log10(raw)), st = [1, 2, 5, 10].find(m => m * p >= raw - 1e-12) * p;
    lo = Math.floor(lo / st + 1e-9) * st; hi = Math.ceil(hi / st - 1e-9) * st;
    return [lo, hi > lo ? hi : lo + st, st];
  };
  function chart(box, o) {
    box.textContent = '';
    const W = box.clientWidth || 300, H = box.clientHeight || 180, P = o.pad || { l: 40, r: 10, t: 10, b: 20 };
    const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.name });
    const idx = [...Array(o.n).keys()], vals = o.series.flatMap(s => idx.map(s.v)).filter(v => v != null);
    box.append(g);
    if (!vals.length) { box.insertAdjacentHTML('beforeend', `<p class="empty">${esc(o.empty)}</p>`); return; }
    let [lo, hi, st] = nice(Math.min(...vals, ...(o.include || [])), Math.max(...vals, ...(o.include || [])), o.lines || 4, o.least);
    if (o.bound) { lo = Math.max(lo, o.bound[0]); hi = Math.min(hi, o.bound[1]); if (hi <= lo) lo = hi - st; }
    const y = v => P.t + (H - P.t - P.b) * (1 - (v - lo) / (hi - lo)), wide = W - P.l - P.r, slot = wide / Math.max(o.n, o.slots || 1);
    const xs = idx.map(o.x ? i => P.l + wide * o.x(i) : i => P.l + slot * (i + .5));
    for (let v = lo; v <= hi + st / 2; v += st) g.append(svg('line', { x1: P.l, x2: W - P.r, y1: y(v), y2: y(v), class: 'grid' }), svg('text', { x: P.l - 6, y: y(v) + 3, 'text-anchor': 'end' }, o.fmt(v)));
    if (o.ref != null) g.append(svg('line', { x1: P.l, x2: W - P.r, y1: y(o.ref), y2: y(o.ref), class: 'ref' }));
    const zero = y(Math.min(hi, Math.max(lo, 0))), bar = Math.max(2, Math.min(26, slot * .62));
    for (const s of o.series) {
      const has = idx.filter(i => s.v(i) != null), color = i => typeof s.c === 'function' ? s.c(i) : s.c;
      if (s.bar) {
        for (const i of has) {
          const h = Math.max(1.5, Math.abs(y(s.v(i)) - zero));
          g.append(svg('rect', { x: xs[i] - bar / 2, y: s.v(i) >= 0 ? zero - h : zero, width: bar, height: h, fill: s.hollow && s.hollow(i) ? 'none' : color(i), stroke: color(i) }));
        }
        continue;
      }
      let run = [];
      const line = () => {
        if (run.length > 1) g.append(svg('polyline', { points: run.map(i => xs[i].toFixed(1) + ',' + y(s.v(i)).toFixed(1)).join(' '), fill: 'none', stroke: color(run[0]), 'stroke-width': 1.5, 'stroke-linejoin': 'round' }));
        run = [];
      };
      for (const i of idx) s.v(i) == null ? line() : run.push(i);
      line();
      const r = has.length > 40 ? 1.6 : 2.4, last = has[has.length - 1];
      for (const i of has) g.append(svg('circle', { cx: xs[i], cy: y(s.v(i)), r: i === last ? r + .6 : r, fill: i === last ? color(i) : 'var(--bg)', stroke: color(i), 'stroke-width': 1.2 }));
    }
    g.append(svg('text', { x: P.l, y: H - 4, 'text-anchor': 'start' }, o.label(0)));
    if (o.n > 1) g.append(svg('text', { x: W - P.r, y: H - 4, 'text-anchor': 'end' }, o.label(o.n - 1)));
    const tip = document.createElement('output');
    tip.className = 'tip'; tip.hidden = true; box.append(tip);
    idx.forEach(i => {
      const l = i ? (xs[i - 1] + xs[i]) / 2 : P.l, r = i < o.n - 1 ? (xs[i] + xs[i + 1]) / 2 : W - P.r;
      const hit = svg('rect', { x: l, y: P.t, width: Math.max(1, r - l), height: H - P.t - P.b, fill: 'transparent' });
      hit.addEventListener('mouseenter', () => { tip.innerHTML = o.tip(i); tip.hidden = false; tip.style.left = Math.min(Math.max(0, xs[i] - tip.offsetWidth / 2), Math.max(0, W - tip.offsetWidth)) + 'px'; });
      hit.addEventListener('mouseleave', () => { tip.hidden = true; });
      g.append(hit);
    });
  }
  const tipOf = (title, rows) => `<strong>${esc(title)}</strong>` + rows.map(([k, v]) => `<span>${esc(k)}</span> ${esc(v)}`).join('\n');
  const share = (x, d = 1) => x == null ? '—' : (100 * x).toFixed(d) + '%';
  const whole = v => +(100 * v).toFixed(2) + '%';

  /* The benchmark table and its score cards, as bench.js of the v1 site. Columns: Model, Avg, then one per benchmark. The kings
     are the body, newest first, in a box about five rows deep. The reference models and the genesis head are a sticky footer, the
     genesis head last. The reference models are in the order of the v1 page: the largest first, then the higher average. As
     there, the footer holds only the ones that lead for their size; a link shows all. A cell is shaded by its place between the lowest and the highest score of its column, and a star marks the
     highest. A model with no score still has its row, and the name says the state of its run. The cards show, per column, the score
     of the genesis head and of each scored king over time. */
  const RUN = {
    queued: ['queued', 'wait', 'The benchmark run of this model has not started.'],
    running: ['running', 'wait', 'The benchmark run of this model is in progress. Its scores come one benchmark at a time.'],
    failed: ['failed', 'dn', 'The benchmark run of this model failed.'],
    none: ['no run yet', 'private', 'The benchmark file does not name this model yet.'],
  };
  RUN.evaluating = RUN.running;  // the word of the v1 benchmark file
  const AVG = { id: '_avg', label: 'Avg', in_avg: true };
  // Parameters of each reference model, in billions: the table PARAMS of bench.js of the v1 site, by the label of the model.
  const SIZE = {
    'qwen3.5-9b': 9.65, 'qwen3-8b': 8.19, 'deepseek-r1-0528-qwen3-8b': 8.19, 'r1-distill-qwen-7b': 7.62, 'r1-distill-llama-8b': 8.03,
    'nemotron-nano-9b-v2': 8.89, 'glm-z1-9b-0414': 9.4, 'qwen3.5-4b': 4.66, 'qwen3-14b': 14.77, 'qwen3-4b': 4.02, 'r1-distill-qwen-14b': 14.77,
    'openreasoning-nemotron-7b': 7.62, 'acereason-nemotron-1.1-7b': 7.62, 'skywork-or1-7b': 7.62, 'mimo-7b-rl': 7.83,
    'llama-3.1-nemotron-nano-8b-v1': 8.03, 'openthinker3-7b': 7.62, 'phi-4-reasoning': 14.66, 'phi-4-mini-reasoning': 3.84,
    'deepscaler-1.5b-preview': 1.78, 'r1-distill-qwen-1.5b': 1.78, 'gpt-oss-20b': 20.91,
  };
  const size = r => SIZE[String(r.label).toLowerCase()] ?? null;
  const STAR = '<svg class="star" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 .6l1.3 2.8 3 .3-2.3 2 .7 3L5 7.2 2.3 8.7l.7-3-2.3-2 3-.3z"/></svg>';
  const shade = f => { const k = Math.min(1, Math.max(0, (.9 - f) / .75)); return `hsl(${Math.round(130 * (1 - k ** .75))} 65% 45% / ${(.3 + .28 * Math.abs(.5 - k)).toFixed(2)})`; };
  const pending = r => ['queued', 'running', 'evaluating'].includes(r.status);
  let allRefs = false;
  function bench(b) {
    b = b || { columns: [], rows: [] };  // a data file that an older builder wrote may have no table
    const cols = [AVG, ...b.columns], val = (r, c) => c === AVG ? r.avg : r.cells[c.id] ? r.cells[c.id].score : null;
    const every = b.rows.filter(r => r.kind === 'king'), kings = V1 ? every.filter(r => Object.keys(r.cells).length) : every;
    const genesis = b.rows.filter(r => r.kind === 'genesis');
    const refs = b.rows.filter(r => r.kind !== 'king' && r.kind !== 'genesis').sort((x, y) => (size(y) ?? -1) - (size(x) ?? -1) || (y.avg ?? -1) - (x.avg ?? -1));
    // The rule of the v1 page: a reference model is in view when, in the average or in one counted benchmark, no reference
    // model of its size or smaller has a higher score. A model with no size in the table, or with no score, is not in view.
    const counted = cols.filter(c => c.in_avg), rated = refs.filter(r => size(r) != null && counted.some(c => val(r, c) != null));
    const leads = (r, c) => val(r, c) != null && !rated.some(o => o !== r && size(o) <= size(r) && val(o, c) > val(r, c));
    const lead = rated.filter(r => counted.some(c => leads(r, c))), pinned = allRefs || !lead.length ? refs : lead;
    const shown = [...kings, ...refs, ...genesis], span = {};
    for (const c of cols.filter(c => c.in_avg)) { const v = shown.map(r => val(r, c)).filter(x => x != null); if (v.length) span[c.id] = [Math.min(...v), Math.max(...v)]; }
    const cell = (r, c) => {
      const x = val(r, c), one = r.cells[c.id], s = span[c.id], avg = c === AVG ? ' avg' : '';
      if (x == null) return `<td class="num na${avg}">${pending(r) ? '…' : '—'}</td>`;
      const f = s && s[1] > s[0] ? (x - s[0]) / (s[1] - s[0]) : 1, top = s && Math.round(1000 * x) === Math.round(1000 * s[1]);
      const title = (top ? 'Highest of the column. ' : '') + (one ? (one.se == null ? '' : `±${(100 * one.se).toFixed(1)} · `) + `cap-hit ${share(one.cap_rate)}` : 'Mean of the counted benchmarks.');
      return `<td class="num${avg}"${s ? ` style="background:linear-gradient(${shade(f)},${shade(f)}),var(--bg)"` : ''} title="${esc(title)}">${top ? STAR : ''}${(100 * x).toFixed(1)}`
        + (one ? dagger(one.partial, `Partial run: ${one.samples} of ${one.expected} samples. A lost sample counts as wrong.`) : dagger(r.avg_partial, 'At least one score of this row is from a partial run.')) + '</td>';
    };
    const name = r => {
      const [first, ...rest] = r.kind === 'king' || r.kind === 'genesis' ? String(r.label).split(' ') : [r.label];
      return `<span class="who"><span>${esc(first)}${rest.length ? ' ' + mute(rest.join(' ')) : ''}</span>` + (r.status === 'scored' || !r.status ? '' : pill(...(RUN[r.status] || [r.status, 'wait', '']))) + '</span>';
    };
    const about = r => [r.label, size(r) && r.kind === 'reference' && `${size(r)} billion parameters`, r.head && 'Head ' + r.head, RUN[r.status] && RUN[r.status][2], r.at && 'Applied ' + utc(r.at), r.v1 && 'Scored by the benchmark service of v1; from the v1 history.'].filter(Boolean).join(' · ');
    const row = r => `<tr class="${esc(r.kind)}"><td class="name" title="${esc(about(r))}">${name(r)}</td>${cols.map(c => cell(r, c)).join('')}</tr>`;
    const suites = [...new Set(b.columns.map(c => c.suite).filter(Boolean))], t = table('bench'), box = t.closest('.scroll');
    t.querySelector('colgroup')?.remove();
    t.insertAdjacentHTML('afterbegin', `<colgroup><col style="width:200px"><col style="width:64px">${b.columns.map(() => '<col style="width:88px">').join('')}</colgroup>`);
    t.style.minWidth = 264 + 88 * b.columns.length + 'px';
    t.tHead.innerHTML = (suites.length > 1 ? `<tr class="suites"><th></th><th class="avg"></th>${suites.map(x => `<th class="suite" colspan="${b.columns.filter(c => c.suite === x).length}">${esc(x)}</th>`).join('')}</tr>` : '')
      + `<tr><th>Model</th><th class="num avg" title="Mean of the counted benchmarks, in percent. A row has it only when it has every counted benchmark.">Avg</th>`
      + b.columns.map(c => `<th class="num" title="${esc(c.label)}: ${c.in_avg ? 'score in percent.' : 'shown only; not counted in Avg, not shaded.'}">${esc(c.label)}</th>`).join('') + '</tr>';
    t.tBodies[0].innerHTML = kings.map(row).join('') || `<tr><td colspan="${cols.length}" class="empty">${V1 ? 'No king was scored.' : 'No king yet: no round has applied a winner.'}</td></tr>`;
    t.tFoot.innerHTML = pinned.map(row).join('') + genesis.map(row).join('');
    let h = t.tHead.offsetHeight + t.tFoot.offsetHeight + 2 + (box.scrollWidth > box.clientWidth ? 12 : 0);
    for (const r of [...t.tBodies[0].rows].slice(0, 5)) h += r.offsetHeight;
    box.style.height = h + 'px';
    if (box.classList.contains('wait')) { box.classList.remove('wait'); t.classList.add('fade'); }
    const scored = every.filter(r => r.status === 'scored').length, more = document.getElementById('refs');
    more.hidden = !lead.length || lead.length === refs.length;
    more.textContent = allRefs ? `Show the ${lead.length} reference models that lead for their size` : `Show all ${refs.length} reference models`;
    document.querySelector('#bench .reading').textContent = V1 ? `${scored} kings scored · ${every.length - scored} with no benchmark run`
      : `${scored} of ${every.length} kings scored · ` + (b.updated_at ? `updated ${dur(Date.now() / 1000 - b.updated_at)} ago` : 'no benchmark file yet');

    const past = [...(V1 ? [] : genesis), ...every].filter(r => r.at != null).sort((x, y) => x.at - y.at), cards = document.querySelector('#bench .sparks');
    cards.hidden = !b.columns.length;
    cards.innerHTML = cols.map(c => `<div class="spark"><div class="section-heading"><h3 title="${esc(c.label)}: the score of ${V1 ? '' : 'the genesis head and of '}each scored king, by the time it was applied">${esc(c.label)}</h3><span class="reading">—</span></div><div class="chart"></div></div>`).join('');
    cols.forEach((c, n) => {
      const pts = past.filter(r => val(r, c) != null), card = cards.children[n], t0 = pts.length && pts[0].at, t1 = pts.length && pts[pts.length - 1].at;
      if (pts.length) card.querySelector('.reading').textContent = share(val(pts[pts.length - 1], c));
      chart(card.querySelector('.chart'), {
        name: c.label + ' score of each king', n: pts.length, x: i => t1 > t0 ? (pts[i].at - t0) / (t1 - t0) : .5, pad: { l: 32, r: 6, t: 6, b: 16 }, lines: 2, least: .01, bound: [0, 1], fmt: whole,
        series: [{ v: i => val(pts[i], c), c: 'var(--performance)' }], label: i => String(pts[i].label).split(' ')[0], empty: 'No score yet.',
        tip: i => tipOf(pts[i].label, [[c.label, share(val(pts[i], c)) + (pts[i].cells[c.id] && pts[i].cells[c.id].se != null ? ' ±' + (100 * pts[i].cells[c.id].se).toFixed(1) : '')], ['applied', utc(pts[i].at)]]),
      });
    });
  }

  /* The chart tiles of the v1 site (k1.js, timing.js), on the closed rounds that the snapshot lists, the oldest at the left.
     v1 had one batch per turn; v2 has many per round, so "z" is the z of the applied winner, or the best z of a round with no winner. */
  const ENDED = { applied: r => `UID ${r.winner_uid} applied`, null: () => 'M stays', no_decision: () => 'no decision', abandoned: () => 'abandoned' };
  const TONE = { applied: 'var(--performance)', null: 'var(--rejected)' };
  function tiles(d) {
    const rs = d.rounds.slice().reverse(), ended = r => (ENDED[r.outcome] || (() => r.outcome))(r), took = r => r.at - r.opened_at;
    const tip = (r, rows) => tipOf(`${r.round_id} · ${ended(r)}`, r.reason ? [...rows, ['validator', r.reason]] : rows);
    const tile = (id, reading, o) => {
      const el = document.getElementById(id);
      el.querySelector('.reading').textContent = reading || '—';
      chart(el.querySelector('.chart'), { n: rs.length, slots: 10, label: i => rs[i].round_id, empty: 'No round is closed yet.', name: el.querySelector('h2').textContent, ...o });
    };
    const last = f => rs.filter(f).pop(), z = last(r => r.best_z != null), pass = last(r => r.null_pass != null), end = rs[rs.length - 1];
    tile('c-z', z && `${z.round_id} z ${sgn(z.best_z)}`, {
      include: [0, d.z_bar], ref: d.z_bar, fmt: v => sgn(v, v % 1 ? 1 : 0),
      series: [{ bar: 1, v: i => rs[i].best_z, c: i => TONE[rs[i].outcome] || 'var(--accepted)', hollow: i => !TONE[rs[i].outcome] }],
      tip: i => tip(rs[i], [[rs[i].outcome === 'applied' ? 'winner z' : 'best z', rs[i].best_z == null ? '—' : sgn(rs[i].best_z)], ['winner bar', 'z > ' + d.z_bar], ['batches', rs[i].seats]]),
    });
    tile('c-pass', pass && `${pass.round_id} M ${share(pass.null_pass)}`, {
      bound: [0, 1], least: .01, fmt: whole,
      series: [{ v: i => rs[i].null_pass, c: 'var(--accepted)' }, { v: i => rs[i].winner_pass, c: 'var(--performance)' }],
      tip: i => tip(rs[i], [['head M', share(rs[i].null_pass, 2)], ['winner', share(rs[i].winner_pass, 2)], ['M cap-hit', share(rs[i].null_cap, 2)], ['tasks', rs[i].tasks]]),
    });
    tile('c-batch', end && `${end.round_id} ${end.seats} of at most ${d.round.cap} · ${end.with_z} with a z`, {
      include: [0], least: 1, fmt: v => v,
      series: [{ bar: 1, v: i => rs[i].seats, c: 'var(--performance)', hollow: () => 1 }, { bar: 1, v: i => rs[i].with_z || null, c: 'var(--performance)' }],
      tip: i => tip(rs[i], [['batches', `${rs[i].seats} of at most ${d.round.cap}`], ['with a z', rs[i].with_z], ['skipped, no valid file', rs[i].skipped], ['tasks', rs[i].tasks]]),
    });
    const five = rs.slice(-5).map(took).sort((a, b) => a - b), mid = five.length >> 1;
    const unit = Math.max(0, ...rs.map(took)) >= 7200 ? ['h', 3600] : ['m', 60];
    tile('c-time', end && `${end.round_id} ${dur(took(end))} · median of the last ${five.length} ${dur(five.length % 2 ? five[mid] : (five[mid - 1] + five[mid]) / 2)}`, {
      include: [0], least: 1, fmt: v => v + unit[0],
      series: [{ bar: 1, v: i => took(rs[i]) / unit[1], c: i => TONE[rs[i].outcome] || 'var(--accepted)', hollow: i => !TONE[rs[i].outcome] }],
      tip: i => tip(rs[i], [['open to close', dur(took(rs[i]))], ['opened', utc(rs[i].opened_at)], ['closed', utc(rs[i].at)]]),
    });
  }
  const match = (rows, f) => f ? rows.filter(x => String(x.uid) === f || String(x.hotkey || '').toLowerCase().includes(f)) : rows;
  function renderV1() {
    const d = D, f = filter.trim().toLowerCase(), turns = match(d.turns, f);
    document.getElementById('archived').textContent = utc(d.archived_at);
    fill('v1kings', d.kings, `${d.kings.length} kings when v1 stopped`);
    fill('v1turns', turns, f ? `${turns.length} of ${d.turns.length} turns` : `${turns.length} turns · ${turns.filter(x => x.verdict === 'green').length} green`);
    bench(d.bench);
    status();
  }
  function render() {
    if (V1) return renderV1();
    const d = D, r = d.round, q = d.queue.rows, waits = q.filter(x => !x.flags.includes('leased')), ready = waits.filter(x => x.file_valid).length;
    const live = {
      id: r.id || '—', state: d.phase, seats: r.seats == null ? '—' : `${r.seats} / ${r.cap}`, tasks: r.tasks,
      head: clip(d.head, 8, 0), bar: 'z > ' + d.z_bar, cap: d.max_output_tokens.toLocaleString('en-US'), next: ready + ' ready',
    };
    for (const el of document.querySelectorAll('#round [data-f]')) el.textContent = live[el.dataset.f];
    document.querySelector('#round .reading').innerHTML = 'env ' + esc(d.env) + (r.opened_at ? ' · opened ' + ago(r.opened_at) : '');
    fill('kings', d.kings, `${d.kings.length} of ${d.kings_window} places`);
    // every count in the summary comes from the listed rows, so it can never disagree with them
    const hidden = d.queue.total - q.length;
    fill('queue', q, `${q.length - waits.length} in the round · ${ready} waiting with a valid file · ${waits.length - ready} waiting without`,
      hidden > 0 ? `+${hidden} more registrations wait behind these and are not listed.` : '');
    const f = filter.trim().toLowerCase(), hist = match(d.history, f);
    fill('history', hist, f ? `${hist.length} of ${d.history.length} results` : `${hist.length} results`);
    fill('rounds', d.rounds, `${d.rounds.length} rounds · ${d.rounds.filter(x => x.outcome === 'applied').length} applied`);
    fill('heads', d.heads, `${d.heads.length} versions`);
    bench(d.bench);
    tiles(d);
    status();
  }
  function status(failed) {
    const el = document.getElementById('status');
    el.dataset.state = !D || failed ? 'unavailable' : 'live';
    el.textContent = V1 ? (D ? `Archived ${utc(D.archived_at)}. This page does not change.` : 'The v1 archive is not on this site.')
      : !D ? 'No data yet.' : (failed ? 'Refresh failed; this is the last good data. ' : '') + `Last change published ${dur(Date.now() / 1000 - D.generated_at)} ago.`;
  }
  async function load(first) {
    try {
      const res = await (first || fetch(DATA + '?t=' + Date.now(), { cache: 'no-store' }));
      if (!res || !res.ok) throw new Error(DATA + ' ' + (res ? res.status : 'unreachable'));
      D = await res.json();
      render();
    } catch (e) { console.warn(e); status(true); }
  }
  document.addEventListener('click', e => {
    const b = e.target.closest && e.target.closest('.key');
    if (b && navigator.clipboard) navigator.clipboard.writeText(b.dataset.copy).then(() => { b.classList.add('copied'); setTimeout(() => b.classList.remove('copied'), 900); }, () => { });
  });
  document.getElementById('filter').addEventListener('input', e => { filter = e.target.value; if (D) render(); });
  document.getElementById('refs').addEventListener('click', () => { allRefs = !allRefs; if (D) bench(D.bench); });
  let resized = 0;  // a chart is drawn at the size of its box, so a new size draws again from the data in hand
  addEventListener('resize', () => { clearTimeout(resized); resized = setTimeout(() => D && render(), 200); });
  Object.keys(T).filter(table).forEach(shell);
  load(window.SITE_DATA);
  if (!V1) setInterval(load, 60000);
})();
