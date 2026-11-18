/* SN120 v2 status site. One column schema and one renderer for every table, as on the v1 site (k1-tables.js).
   The benchmark table, its score cards and the chart tiles copy the structure of the v1 site (bench.js, k1.js, timing.js).
   A page reads one file, and <body data-page> names the page. index.html (the benchmark table first) and round.html (the live
   round) read site-data.json, which build.py writes from the publisher's public snapshot. supply.html reads miner-supply.json,
   which supply.py writes. v1.html reads v1-data.json, the frozen v1 history that v1_archive.py wrote once at cutover.
   A part of the page is filled only where the page has it, so each page uses this one file.
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
  // A value that does not exist is a word, never an empty cell: `no(why)` is the word, and its tooltip says why.
  const no = (why, word = 'none') => mute(word, why);
  const pct = (x, gap = mute('—')) => x == null ? gap : (100 * x).toFixed(1) + '%';
  const ago = t => t ? `<span title="${utc(t)}">${dur(Date.now() / 1000 - t)} ago</span>` : mute('—');
  const key = (v, a, b) => v ? `<button type="button" class="key" data-copy="${esc(v)}" title="${esc(v)} (click to copy)">${esc(clip(v, a, b))}</button>` : mute('—');
  const link = (u, s, t) => `<a href="${esc(u)}"${t ? ` title="${esc(t)}"` : ''}>${esc(s)}</a>`;
  const pill = (s, c, t) => `<span class="pill ${c}"${t ? ` title="${esc(t)}"` : ''}>${esc(s)}</span>`;
  const score = (x, d, t, gap = mute('—')) => x == null ? gap : `<span class="${x > 0 ? 'up' : x < 0 ? 'dn' : 'mute'}" title="${esc(t)}">${sgn(x, d)}</span>`;
  const RESULT = {
    winner: ['WINNER', 'up', 'Beat the head M by more than the winner bar.'],
    lost: ['LOST', 'dn', 'Had a fair tournament and did not win. The registration is used up.'],
    no_decision: ['NO DECISION', 'wait', 'Our infrastructure failed. Not charged; the seat goes back to the front of the queue.'],
    skip: ['SKIP', 'private', 'The file was missing or invalid when the round froze it. Not charged; the seat keeps its place.'],
    not_trained: ['NOT TRAINED', 'wait', 'The training time of the round ended before this batch was trained. Not charged; the seat is first in the next round.'],
    not_scored: ['NOT SCORED', 'wait', 'The time limit of the round came before the first look of the scoring. Not charged; the seat is first in the next round.'],
  };
  // A round with no result gives no entry a status: the box failed, or the operator ended the round.
  const NO_RESULT = ['NO RESULT', 'wait', 'The round had no result: the box failed, not this seat. Not charged; no failure is counted for the seat, and it goes to the front of the queue.'];
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
  const PAGE = document.body.dataset.page || 'home', V1 = PAGE === 'v1', SUPPLY = PAGE === 'supply', DATA = V1 ? 'v1-data.json' : 'site-data.json';
  let D = null, filter = '';
  // The word for a row that has no score: a king carried over from v1, or a batch that was not scored. The v1 page keeps its dash.
  const unscored = r => V1 ? mute('—')
    : r.imported ? no('Carried over from v1: this king has no v2 score and no v2 head. Its v1 score is on the v1 history page.')
    : no('Not scored. ' + (r.status == null ? NO_RESULT : RESULT[r.status] || ['', '', 'The round gave this batch no score.'])[2]);
  const noM = r => no('The round ended before the head M was scored, so the round file has no value.' + why(r));

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
    z: { h: 'z', w: 60, n: 1, r: r => score(r.z, 2, 'z = gain / standard error', unscored(r)), t: 'Gain over the head M, in standard errors.' },
    head: { h: 'Head', w: 132, r: r => (r.head ? key(r.head, 8, 0) : unscored(r)) + (r.current ? ' ' + pill('current', 'up') : ''), t: 'Head version id.' },
    files: { h: 'Files', w: 136, r: files, t: 'Published for a winner only: the batch, and the hash of the LoRA.' },
    reg: { h: 'Reg block', w: 88, n: 1, r: r => esc(r.registration_block ?? '—') },
    check: { h: 'File', w: 96, r: r => r.file_valid ? pill('valid', 'up', 'The stored file passes every check.') : mute('no valid file', 'This seat is skipped, is not charged, and keeps its place.'), t: 'Live check of the stored file. It never shows file content.' },
    flags: { h: 'Status', w: 170, r: r => r.flags.map(f => pill(...FLAG[f])).join(' ') || mute('waiting') },
    result: { h: 'Result', w: 112, r: r => pill(...(r.status == null ? NO_RESULT : RESULT[r.status] || [r.status, 'private', ''])) },
    gain: { h: 'Gain', w: 76, n: 1, r: r => score(r.delta == null ? null : 100 * r.delta, 2, `Pass rate minus the pass rate of M, in percentage points (standard error ${(100 * r.se).toFixed(2)})`, unscored(r)), t: 'LoRA pass rate minus head pass rate, in percentage points.' },
    bar: { h: 'vs bar', w: 68, n: 1, r: r => score(r.z == null ? null : r.z - D.z_bar, 2, `z minus the winner bar (${D.z_bar})`, unscored(r)), t: 'z minus the winner bar. Above zero clears the bar.' },
    capHit: { h: 'Cap-hit', w: 68, n: 1, r: r => pct(r.cap_rate, unscored(r)), t: 'Share of answers that hit the token cap. Such an answer is not passed.' },
    look: { h: 'Dropped', w: 68, n: 1, r: r => r.look != null ? 'look ' + (r.look + 1) : r.z == null ? unscored(r) : no('Not dropped at an early look.', 'no'), t: 'The early look at which this LoRA was dropped, if any.' },
    sha: { h: 'Batch sha256', w: 100, r: r => key(r.sha, 8, 0) },
    at: { h: 'When', w: 92, n: 1, r: r => ago(r.at) },
    date: { h: 'Date (UTC)', w: 124, n: 1, r: r => r.at ? esc(utc(r.at).slice(0, 16)) : mute('—') },
    env: { h: 'Env', w: 60, r: r => esc(r.env) },
    seats: { h: 'Batches', w: 68, n: 1, r: r => esc(r.seats), t: 'Seats of the round: the batches with a valid file at round open. A batch whose training failed is in this number.' },
    skipped: { h: 'Skipped', w: 68, n: 1, r: r => esc(r.skipped), t: 'Seats with no valid file at round open. Not charged.' },
    tasks: { h: 'Tasks', w: 64, n: 1, r: r => esc(r.tasks) },
    mPass: { h: 'M pass', w: 68, n: 1, r: r => pct(r.null_pass, noM(r)) },
    mCap: { h: 'M cap-hit', w: 76, n: 1, r: r => pct(r.null_cap, noM(r)) },
    outcome: { h: 'Outcome', w: 132, r: r => (OUTCOME[r.outcome] || (() => esc(r.outcome)))(r) },
    kind: { h: 'Kind', w: 84, r: r => r.folded ? mute('folded', 'The LoRAs are merged into the base.') : mute('stacked', 'The winning LoRA is stacked on the base.') },
    manifest: { h: 'Manifest', w: 84, r: r => r.manifest ? link(r.manifest, 'head.json', 'File hashes of this head') : no('No manifest file of this head is on this site. The validator writes the manifest of a head only while that head is the current one.', 'no file') },
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
    if (!table(n)) return;
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
  const share = (x, d = 1, gap = 'none') => x == null ? gap : (100 * x).toFixed(d) + '%';
  const whole = v => +(100 * v).toFixed(2) + '%';

  /* The benchmark table and its score cards, as bench.js of the v1 site. Columns: Model, Avg, then one per benchmark. The kings
     are the body, newest first, in a box about five rows deep. The reference models and the genesis head are a sticky footer, the
     genesis head last. The reference models are in the order of the v1 page: the largest first, then the higher average. As
     there, the footer holds only the ones that lead for their size; a link shows all. A cell is shaded by its place between the lowest and the highest score of its column, and a star marks the
     highest. A model with no score still has its row, and the name says the state of its run. The cards show, per column, the score
     of the genesis head and of each scored king over time. */
  // The pill of the row, its colour, the tooltip, and the word of a cell that has no score. The page shows a state only when the
  // benchmark file gives the row that state: "queued", "evaluating" (shown as "running") or "failed". A model that the file does
  // not name, or a site with no benchmark file, has "no run yet".
  const RUN = {
    queued: ['queued', 'wait', 'The benchmark file lists this model as queued: no run of it is in progress now.', 'queued'],
    running: ['running', 'wait', 'The benchmark run of this model is in progress: the benchmark file says "evaluating".', 'running'],
    failed: ['failed', 'dn', 'The benchmark file says that the run of this model failed.', 'failed'],
    none: ['no run yet', 'private', 'No run yet: the benchmark file does not name this model.', 'no run yet'],
  };
  RUN.evaluating = RUN.running;  // the word of the benchmark file, for a data file whose builder left it as it is
  RUN.waiting = ['waiting', 'wait', 'The benchmark file lists this model as waiting: its run did not start.', 'waiting'];
  // The state of the machine of a row, where it says more than the state of the run: the run waits for the machine, or got none.
  const POD = {
    renting: ['renting', 'wait', 'The benchmark service rents a machine for the run of this model.', 'renting'],
    setup: ['setup', 'wait', 'The machine for the run of this model is in setup: the run did not start.', 'setup'],
    'rent-failed': ['failed', 'dn', 'The rent of a machine for the run of this model failed: the run did not start.', 'rent failed'],
  };
  const NO_FILE = 'No run yet: no benchmark file is published.';
  // Why the benchmark service starts no run of a row until the operator says so (`stopped`), and what a row waits for (`waiting_for`).
  const STOP = {
    held: ['held', 'dn', 'The run of this model ended and gave no value for a benchmark. The benchmark service starts no new run of it until the operator says so.', 'held'],
    expired: ['expired', 'dn', 'The machine of this model reached its time limit. The benchmark service starts no new run of it until the operator says so.', 'expired'],
    rent_failed: POD['rent-failed'],
  };
  const WAIT = {
    weights: 'The weight files of this model are not in the store yet: its run starts when they are.',
    source: 'The benchmark service cannot read the source of the weights of this model yet: it rents no machine for it.',
  };
  // The sampling of a score, from the settings profile that the benchmark file names. A profile that this page does not know is shown by its name.
  const PP = { 'qwen3.5-family': 'presence penalty 1.5', 'qwen3.5-family-pp0': 'presence penalty 0' };
  const pp = name => PP[name] || `profile ${name}`;
  const V1_KING = 'A king of v1, scored by the benchmark service of v2. It won no round of v2, so it has no round and no head here.';
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
  let allRefs = false;
  function bench(b) {
    if (!table('bench')) return;
    b = b || { columns: [], rows: [] };  // a data file that an older builder wrote may have no table
    const cols = [AVG, ...b.columns], val = (r, c) => c === AVG ? r.avg : r.cells[c.id] ? r.cells[c.id].score : null;
    // The state of a row that is not scored. With no benchmark file at all, the tooltip of "no run yet" says that. A state word
    // that this page does not know is shown as it is.
    const since = r => r.pod.since ? ` Since ${utc(r.pod.since)}.` : '';
    const run = r => r.status === 'scored' || !r.status ? undefined
      : r.pod && POD[r.pod.state] ? POD[r.pod.state].map((x, i) => i === 2 ? x + since(r) : x)
      : STOP[r.stopped] ? STOP[r.stopped]
      : WAIT[r.waiting_for] ? [r.status, 'wait', WAIT[r.waiting_for], r.status]
      : r.status === 'none' && !b.updated_at && !V1 ? [...RUN.none.slice(0, 2), NO_FILE, RUN.none[3]]
      : RUN[r.status] || [r.status, 'wait', `The benchmark file gives this model the state "${r.status}".`, 'no score'];
    const every = b.rows.filter(r => r.kind === 'king'), kings = V1 ? every.filter(r => Object.keys(r.cells).length) : every;
    const genesis = b.rows.filter(r => r.kind === 'genesis'), old = b.rows.filter(r => r.kind === 'v1-king');  // old: a king of v1 that the benchmark service of v2 scores
    const refs = b.rows.filter(r => r.kind !== 'king' && r.kind !== 'genesis' && r.kind !== 'v1-king').sort((x, y) => (size(y) ?? -1) - (size(x) ?? -1) || (y.avg ?? -1) - (x.avg ?? -1));
    // The rule of the v1 page: a reference model is in view when, in the average or in one counted benchmark, no reference
    // model of its size or smaller has a higher score. A model with no size in the table, or with no score, is not in view.
    const counted = cols.filter(c => c.in_avg), rated = refs.filter(r => size(r) != null && counted.some(c => val(r, c) != null));
    const leads = (r, c) => val(r, c) != null && !rated.some(o => o !== r && size(o) <= size(r) && val(o, c) > val(r, c));
    const lead = rated.filter(r => counted.some(c => leads(r, c))), pinned = allRefs || !lead.length ? refs : lead;
    const shown = [...kings, ...refs, ...genesis, ...old], span = {};
    for (const c of cols.filter(c => c.in_avg)) { const v = shown.map(r => val(r, c)).filter(x => x != null); if (v.length) span[c.id] = [Math.min(...v), Math.max(...v)]; }
    // The samples for each problem of a benchmark, from the rows on the page: a row of a king or of the genesis head names the count of
    // the run that its score is from, and a reference model has the count of its v1 run. A benchmark where the rows on the page do not
    // all have one count gets one sentence in the note and in the tooltip of its column head: who has which count. A value that was
    // computed from a longer run of the same model gets one more sentence: from how many of the samples, and where the full score is.
    const samples = n => `${n} sample${n === 1 ? '' : 's'}`, list = a => a.length > 1 ? `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}` : a[0];
    const longer = c => b.columns.find(o => o.full_of === c.id), first = r => r.kind === 'v1-king' ? 'the v1 king' : String(r.label).split(' ')[0];
    const part = (sets, has) => list(sets.map(([n, name]) => [n.filter(has).length, n.length, name]).filter(([m]) => m).map(([m, all, name]) => m < all ? `${m} of ${name}` : name));
    // The sampling of the scores of a row: the profiles of its cells. For a model with no score the file has the profile of its node, or
    // of the settings of the moment: the row tooltip shows it only while the run is in work, and the sentence of the note never.
    const had = r => [...new Set(Object.values(r.cells).map(x => x.profile).filter(Boolean))];
    const sampling = () => {
      const sets = [[kings, 'the kings'], [old, old.length > 1 ? 'the v1 kings' : 'the v1 king'], [genesis, genesis.length > 1 ? 'the genesis heads' : 'the genesis head']]
        .map(([rows, name]) => [rows.map(had).filter(a => a.length), name]);
      const all = [...new Set(sets.flatMap(([n]) => n.flat()))], both = shown.filter(r => had(r).length > 1);
      return all.length < 2 ? '' : 'Sampling: ' + all.map(p => `${pp(p)} for ${part(sets, a => a.includes(p))}`).join(', ') + '.'
        + (both.length ? ` ${list(both.map(first))} ${both.length > 1 ? 'have' : 'has'} scores of more than one sampling: each score names its own in its tooltip.` : '')
        + (pinned.some(r => !had(r).length) ? ' The data does not name the sampling of the reference models.' : '') + ' ';
    };
    const from = (one, all) => `${one.rollouts || 'a part'} of the ${all && all.rollouts ? all.rollouts + ' ' : ''}samples of each problem`;
    const differ = c => {
      if (c.full_of) return '';
      const sets = [[kings, 'the kings'], [old, old.length > 1 ? 'the v1 kings' : 'the v1 king'], [genesis, genesis.length > 1 ? 'the genesis heads' : 'the genesis head'], [pinned, 'the reference models']]
        .map(([rows, name]) => [rows.map(r => r.cells[c.id] && r.cells[c.id].rollouts).filter(Boolean), name]);
      const counts = [...new Set(sets.flatMap(([n]) => n))].sort((x, y) => x - y), each = c.problems ? `each of the ${c.problems} problems` : 'each problem';
      const who = k => part(sets, x => x === k);
      const wide = longer(c), made = [...kings, ...pinned, ...genesis, ...old].filter(r => r.cells[c.id] && r.cells[c.id].derived), many = made.length > 1;
      const how = r => from(r.cells[c.id], wide && r.cells[wide.id]), one = made.every(r => how(r) === how(made[0]));
      const all = wide && made.every(r => r.cells[wide.id]) && made[0].cells[wide.id].rollouts;
      const parts = [
        counts.length > 1 ? counts.map((k, i) => `${samples(k)}${i ? '' : ' for ' + each} for ${who(k)}`).join(', ') + '.' : '',
        made.length ? `The value${many ? 's' : ''} of ${list(made.map(first))} ${many ? 'were' : 'was'} computed from ${one ? how(made[0]) : 'a part of the samples of each problem'} of ${many ? 'their full runs' : 'its full run'}`
          + (wide && made.every(r => r.cells[wide.id]) ? `; the column "${wide.head}" beside it has the score of ${one && all ? 'all ' + all : many ? 'the full runs' : 'the full run'}.` : '.') : '',
      ].filter(Boolean);
      return parts.length ? `${c.label}: ${parts.join(' ')}` : '';
    };
    const cell = (r, c) => {
      const x = val(r, c), one = r.cells[c.id], s = span[c.id], avg = c === AVG ? ' avg' : '';
      if (x == null && c.full_of) return '<td class="num na" title="No full run of this model."></td>';
      if (x == null) {  // no score: the state of the run in a word (a word of more than 7 letters is too wide for Avg); a scored row can lack one benchmark, and then it has no average
        const [word, text] = run(r) ? [c === AVG && run(r)[3].length > 7 ? r.status === 'none' ? 'not run' : 'none' : run(r)[3], run(r)[2]] : c === AVG ? ['none', 'No average: a row has one only when it has every counted benchmark.'] : ['no score', `This model has no score in ${c.label}.`];
        return `<td class="num na${avg}" title="${esc(text)}">${esc(word)}</td>`;
      }
      const f = s && s[1] > s[0] ? (x - s[0]) / (s[1] - s[0]) : 1, top = s && Math.round(1000 * x) === Math.round(1000 * s[1]);
      const title = (top ? 'Highest of the column. ' : '') + (one ? (one.se == null ? '' : `±${(100 * one.se).toFixed(1)} · `) + `cap-hit ${share(one.cap_rate, 1, 'not recorded')}` + (one.rollouts ? ` · ${samples(one.rollouts)} for each problem` : '') + (one.profile ? ` · ${pp(one.profile)}` : '') + (one.derived ? ` · computed from ${from(one, longer(c) && r.cells[longer(c).id])} of the full run` : '') : 'Mean of the counted benchmarks.');
      return `<td class="num${avg}"${s ? ` style="background:linear-gradient(${shade(f)},${shade(f)}),var(--bg)"` : ''} title="${esc(title)}">${top ? STAR : ''}${(100 * x).toFixed(1)}`
        + (one ? dagger(one.partial, `Partial run: ${one.samples} of ${one.expected} samples. A lost sample counts as wrong.`) : dagger(r.avg_partial, 'At least one score of this row is from a partial run.')) + '</td>';
    };
    const name = r => {
      const [first, ...rest] = r.kind === 'king' || r.kind === 'genesis' ? String(r.label).split(' ') : r.kind === 'v1-king' && r.uid != null ? [r.label, 'UID ' + r.uid] : [r.label];
      return `<span class="who"><span>${esc(first)}${rest.length ? ' ' + mute(rest.join(' ')) : ''}</span>` + (run(r) ? pill(...run(r).slice(0, 3)) : '') + '</span>';
    };
    const about = r => [r.label, size(r) && r.kind === 'reference' && `${size(r)} billion parameters`, r.kind === 'v1-king' && (r.uid != null ? `UID ${r.uid} · ` : '') + (r.v1_epoch != null ? `v1 epoch ${r.v1_epoch} · ` : '') + V1_KING, r.head && 'Head ' + r.head, run(r) && run(r)[2], had(r).length ? 'Sampling: ' + list(had(r).map(pp)) : r.profile && run(r) === RUN.running && 'Sampling of its run: ' + pp(r.profile), r.pod && !(run(r) && POD[r.pod.state]) && `Machine: ${r.pod.state}.${since(r)}`, r.at && 'Applied ' + utc(r.at), r.v1 && 'Scored by the benchmark service of v1; from the v1 history.'].filter(Boolean).join(' · ');
    const row = r => `<tr class="${esc(r.kind)}"><td class="name" title="${esc(about(r))}">${name(r)}</td>${cols.map(c => cell(r, c)).join('')}</tr>`;
    const suites = [...new Set(b.columns.map(c => c.suite).filter(Boolean))], t = table('bench'), box = t.closest('.scroll');
    t.querySelector('colgroup')?.remove();
    t.insertAdjacentHTML('afterbegin', `<colgroup><col style="width:200px"><col style="width:64px">${b.columns.map(() => '<col style="width:88px">').join('')}</colgroup>`);
    t.style.minWidth = 264 + 88 * b.columns.length + 'px';
    t.tHead.innerHTML = (suites.length > 1 ? `<tr class="suites"><th></th><th class="avg"></th>${suites.map(x => `<th class="suite" colspan="${b.columns.filter(c => c.suite === x).length}">${esc(x)}</th>`).join('')}</tr>` : '')
      + `<tr><th>Model</th><th class="num avg" title="Mean of the counted benchmarks, in percent. A row has it only when it has every counted benchmark.">Avg</th>`
      + b.columns.map(c => `<th class="num" title="${esc(c.label)}: ${c.in_avg ? 'score in percent.' : 'shown only; not counted in Avg, not shaded.'}${c.full_of ? ' The score of the full run, for a model that has one.' : ''}${esc(differ(c) && ' ' + differ(c).slice(c.label.length + 2))}">${esc(c.head || c.label)}</th>`).join('') + '</tr>';
    t.tBodies[0].innerHTML = kings.map(row).join('') || `<tr><td colspan="${cols.length}" class="empty">${V1 ? 'No king was scored.' : 'No king yet: no round has applied a winner.'}</td></tr>`;
    t.tFoot.innerHTML = pinned.map(row).join('') + genesis.map(row).join('') + old.map(row).join('');
    let h = t.tHead.offsetHeight + t.tFoot.offsetHeight + 2 + (box.scrollWidth > box.clientWidth ? 12 : 0);
    for (const r of [...t.tBodies[0].rows].slice(0, 5)) h += r.offsetHeight;
    box.style.height = h + 'px';
    if (box.classList.contains('wait')) { box.classList.remove('wait'); t.classList.add('fade'); }
    const scored = every.filter(r => r.status === 'scored').length, more = document.getElementById('refs');
    more.hidden = !lead.length || lead.length === refs.length;
    if (more.previousSibling?.own !== 1) more.before(Object.assign(document.createTextNode(''), { own: 1 }));
    more.previousSibling.data = (old.length ? `v1 king: ${V1_KING[0].toLowerCase() + V1_KING.slice(1)} ` : '') + b.columns.map(c => differ(c) && differ(c) + ' ').join('') + sampling();
    more.textContent = allRefs ? `Show the ${lead.length} reference models that lead for their size` : `Show all ${refs.length} reference models`;
    document.querySelector('#bench .reading').textContent = V1 ? `${scored} kings scored · ${every.length - scored} with no benchmark run`
      : `${scored} of ${every.length} kings scored · ` + (b.updated_at ? `updated ${dur(Date.now() / 1000 - b.updated_at)} ago` : 'no benchmark run yet');

    const past = [...(V1 ? [] : genesis), ...every].filter(r => r.at != null).sort((x, y) => x.at - y.at), cards = document.querySelector('#bench .sparks');
    const charted = cols.filter(c => !c.full_of);  // the full run of a benchmark has its column in the table and no card
    cards.hidden = !b.columns.length;
    cards.innerHTML = charted.map(c => `<div class="spark"><div class="section-heading"><h3 title="${esc(c.label)}: the score of ${V1 ? '' : 'the genesis head and of '}each scored king, by the time it was applied">${esc(c.label)}</h3><span class="reading">${b.updated_at || V1 ? 'none' : 'no run yet'}</span></div><div class="chart"></div></div>`).join('');
    charted.forEach((c, n) => {
      const pts = past.filter(r => val(r, c) != null), card = cards.children[n], t0 = pts.length && pts[0].at, t1 = pts.length && pts[pts.length - 1].at;
      if (pts.length) card.querySelector('.reading').textContent = share(val(pts[pts.length - 1], c));
      chart(card.querySelector('.chart'), {
        name: c.label + ' score of each king', n: pts.length, x: i => t1 > t0 ? (pts[i].at - t0) / (t1 - t0) : .5, pad: { l: 32, r: 6, t: 6, b: 16 }, lines: 2, least: .01, bound: [0, 1], fmt: whole,
        series: [{ v: i => val(pts[i], c), c: 'var(--performance)' }], label: i => String(pts[i].label).split(' ')[0], empty: b.updated_at || V1 ? 'No score yet.' : 'No run yet: no benchmark file is published.',
        tip: i => tipOf(pts[i].label, [[c.label, share(val(pts[i], c)) + (pts[i].cells[c.id] && pts[i].cells[c.id].se != null ? ' ±' + (100 * pts[i].cells[c.id].se).toFixed(1) : '')], ['applied', utc(pts[i].at)]]),
      });
    });
  }

  /* The chart tiles of the v1 site (k1.js, timing.js), on the closed rounds that the snapshot lists, the oldest at the left.
     v1 had one batch per turn; v2 has many per round, so "z" is the z of the applied winner, or the best z of a round with no winner. */
  const ENDED = { applied: r => `UID ${r.winner_uid} applied`, null: () => 'M stays', no_decision: () => 'no decision', abandoned: () => 'abandoned' };
  const TONE = { applied: 'var(--performance)', null: 'var(--rejected)' };
  function tiles(d) {
    if (!document.getElementById('charts')) return;
    const rs = d.rounds.slice().reverse(), ended = r => (ENDED[r.outcome] || (() => r.outcome))(r), took = r => r.at - r.opened_at;
    const tip = (r, rows) => tipOf(`${r.round_id} · ${ended(r)}`, r.reason ? [...rows, ['validator', r.reason]] : rows);
    const tile = (id, reading, o) => {
      const el = document.getElementById(id);
      el.querySelector('.reading').textContent = reading || 'none';
      chart(el.querySelector('.chart'), { n: rs.length, slots: 10, label: i => rs[i].round_id, empty: 'No round is closed yet.', name: el.querySelector('h2').textContent, ...o });
    };
    const last = f => rs.filter(f).pop(), z = last(r => r.best_z != null), pass = last(r => r.null_pass != null), end = rs[rs.length - 1];
    tile('c-z', z && `${z.round_id} z ${sgn(z.best_z)}`, {
      include: [0, d.z_bar], ref: d.z_bar, fmt: v => sgn(v, v % 1 ? 1 : 0),
      series: [{ bar: 1, v: i => rs[i].best_z, c: i => TONE[rs[i].outcome] || 'var(--accepted)', hollow: i => !TONE[rs[i].outcome] }],
      tip: i => tip(rs[i], [[rs[i].outcome === 'applied' ? 'winner z' : 'best z', rs[i].best_z == null ? 'none: no batch was scored' : sgn(rs[i].best_z)], ['winner bar', 'z > ' + d.z_bar], ['batches', rs[i].seats]]),
    });
    tile('c-pass', pass && `${pass.round_id} M ${share(pass.null_pass)}`, {
      bound: [0, 1], least: .01, fmt: whole,
      series: [{ v: i => rs[i].null_pass, c: 'var(--accepted)' }, { v: i => rs[i].winner_pass, c: 'var(--performance)' }],
      tip: i => tip(rs[i], [['head M', share(rs[i].null_pass, 2, 'not scored')], ['winner', share(rs[i].winner_pass, 2, rs[i].outcome === 'applied' ? 'none' : 'no winner')], ['M cap-hit', share(rs[i].null_cap, 2, 'not scored')], ['tasks', rs[i].tasks]]),
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

  /* Miner supply: the counts of miner-supply.json, which supply.py writes each minute from the list of the object store. The
     section has its own file and its own timer, so it and the rest of the page never wait for each other. A count with "≥" is
     "at least": the record does not cover all of that window (it began later, or a run is missing). */
  let S = null;
  const KIND = [['uploads', 'Uploads per hour from R2 LastModified'], ['validated', 'Validated so far'], ['no_verdict', 'Uploads with no verdict yet'], ['not_valid', 'Not valid at the last check'], ['replaced', 'Replaced before a verdict']];
  function supply(failed) {
    const el = document.getElementById('supply'), foot = document.getElementById('status');
    foot.dataset.state = !S || failed ? 'unavailable' : 'live';
    foot.textContent = !S ? 'No counts yet.' : (failed ? 'Refresh failed; these are the last good counts. ' : '') + `Counts written ${dur(Date.now() / 1000 - S.generated_at)} ago.`;
    if (!S) { el.querySelector('.reading').textContent = 'The recorder has written no counts yet.'; return; }
    const age = Date.now() / 1000 - S.generated_at, w = S.waiting, hs = S.hours, part = 'At least: the record does not cover all of this time.';
    // every count but "no verdict yet" can miss a file that was replaced before the recorder saw it
    const n = (r, k) => (r.complete || k === 'no_verdict' ? '' : '≥ ') + r[k] + (k === 'uploads' || k === 'validated' ? ` · ${r[k + '_per_hour'].toFixed(2)}/h` : '');
    el.querySelector('.reading').innerHTML = (age > 300 ? `<span class="warn">The recorder's last run was ${dur(age)} ago: these counts are old.</span> · ` : '')
      + `counting since ${esc(utc(S.counting_since))}` + (S.gaps.length ? ` · ${S.gaps.length} gap${S.gaps.length > 1 ? 's' : ''} with no run in 24 h` : '');
    el.querySelector('table').innerHTML = '<thead><tr><th>Count</th>' + S.windows.map(r => `<th class="num">last ${r.hours} h</th>`).join('') + '</tr></thead><tbody>'
      + KIND.map(([k, name]) => `<tr><td title="${esc(S.labels[k])}">${esc(name)}</td>` + S.windows.map(r => `<td class="num"${r.complete ? '' : ` title="${part}"`}>${esc(n(r, k))}</td>`).join('') + '</tr>').join('') + '</tbody>';
    el.querySelector('[data-f=waiting]').innerHTML = `Valid files waiting now: <strong>${esc(w.valid)}</strong> `
      + (w.whole_line ? `of ${esc(w.hotkeys)} hotkeys in the waiting line.` : `in the first ${esc(w.seats)} of ${esc(w.line)} seats of the waiting line. The files of the other seats have no published verdict.`)
      + (w.checked_at ? ` Last file check of the validator: ${esc(utc(w.checked_at))}, ${dur(age + S.generated_at - w.checked_at)} ago.` : ' The validator has published no file check yet.')
      + (S.phase === 'IDLE' ? '' : ' A round runs now: the next file check comes when it is over.');
    if (S.round_time) el.querySelectorAll('[data-f=round-time]').forEach(x => { x.textContent = S.round_time.words; });
    const day = S.windows[S.windows.length - 1];
    el.querySelector('.tile .reading').textContent = `${n(day, 'uploads')} in ${day.hours} h`;
    chart(el.querySelector('.chart'), {
      n: hs.length, name: 'Miner uploads by hour', include: [0], least: 1, fmt: v => v, label: i => utc(hs[i].t).slice(11, 16) + ' UTC', empty: 'No count yet.',
      series: [{ bar: 1, v: i => hs[i].uploads, c: i => hs[i].complete ? 'var(--performance)' : 'var(--accepted)', hollow: () => 1 },
        { bar: 1, v: i => hs[i].validated || null, c: i => hs[i].complete ? 'var(--performance)' : 'var(--accepted)' }],
      tip: i => tipOf(utc(hs[i].t).slice(0, 16) + ' UTC, one hour' + (hs[i].open ? ' (not over)' : ''),
        [...KIND.map(([k, name]) => [name.toLowerCase(), hs[i][k]]), ['record', hs[i].complete ? 'covers this hour' : 'at least: not all of this hour']]),
    });
  }
  async function loadSupply(first) {
    let failed = false;
    try {
      const res = await (first || fetch('miner-supply.json?t=' + Date.now(), { cache: 'no-store' }));
      if (!res || !res.ok) throw new Error('miner-supply.json ' + (res ? res.status : 'unreachable'));
      S = await res.json();
    } catch (e) { console.warn(e); failed = true; }
    supply(failed);
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
      id: r.id || 'none open', state: d.phase, seats: r.seats == null ? 'no round' : `${r.seats} / ${r.cap}`, tasks: r.tasks,
      head: clip(d.head, 8, 0), bar: 'z > ' + d.z_bar, cap: d.max_output_tokens.toLocaleString('en-US'), next: ready + ' ready',
    };
    for (const el of document.querySelectorAll('#round [data-f]')) el.textContent = live[el.dataset.f];
    const open = document.querySelector('#round .reading');
    if (open) open.innerHTML = 'env ' + esc(d.env) + (r.opened_at ? ' · opened ' + ago(r.opened_at) : '');
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
  document.getElementById('filter')?.addEventListener('input', e => { filter = e.target.value; if (D) render(); });
  document.getElementById('refs')?.addEventListener('click', () => { allRefs = !allRefs; if (D) bench(D.bench); });
  let resized = 0;  // a chart is drawn at the size of its box, so a new size draws again from the data in hand
  addEventListener('resize', () => { clearTimeout(resized); resized = setTimeout(() => SUPPLY ? S && supply() : D && render(), 200); });
  Object.keys(T).filter(table).forEach(shell);
  const again = SUPPLY ? loadSupply : load;
  again(window.SITE_DATA);
  if (!V1) setInterval(again, 60000);
})();
