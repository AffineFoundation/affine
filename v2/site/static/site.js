/* SN120 v2 status site. One column schema and one renderer for every table, as on the v1 site (k1-tables.js).
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
  const OUTCOME = {
    applied: r => pill('UID ' + r.winner_uid + ' applied', 'up', 'The winning LoRA is on the head now.'),
    null: () => mute('M stays', 'No LoRA cleared the bar. The head did not change.'),
    no_decision: () => pill('no decision', 'wait', 'No LoRA was judged: a failure on our side, or the time of the round ended first. Nobody is charged.'),
    abandoned: () => pill('abandoned', 'wait', 'The files of the named winner could not be verified. Nobody is charged.'),
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
  const benchCell = id => r => {
    const c = r.cells[id];
    return c ? `<span title="±${(100 * c.se).toFixed(1)} · cap-hit ${(100 * c.cap_rate).toFixed(1)}%">${(100 * c.score).toFixed(1)}</span>`
      + dagger(c.partial, `Partial run: ${c.samples} of ${c.expected} samples. A lost sample counts as wrong.`) : mute('—');
  };

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
    env: { h: 'Env', w: 60, r: r => esc(r.env) },
    seats: { h: 'Batches', w: 68, n: 1, r: r => esc(r.seats), t: 'Batches trained and scored in the round.' },
    skipped: { h: 'Skipped', w: 68, n: 1, r: r => esc(r.skipped), t: 'Seats with no valid file at round open. Not charged.' },
    tasks: { h: 'Tasks', w: 64, n: 1, r: r => esc(r.tasks) },
    mPass: { h: 'M pass', w: 68, n: 1, r: r => pct(r.null_pass) },
    mCap: { h: 'M cap-hit', w: 76, n: 1, r: r => pct(r.null_cap) },
    outcome: { h: 'Outcome', w: 132, r: r => (OUTCOME[r.outcome] || (() => esc(r.outcome)))(r) },
    kind: { h: 'Kind', w: 84, r: r => r.folded ? mute('folded', 'The LoRAs are merged into the base.') : mute('stacked', 'The winning LoRA is stacked on the base.') },
    manifest: { h: 'Manifest', w: 84, r: r => r.manifest ? link(r.manifest, 'head.json', 'File hashes of this head') : mute('—') },
    model: { h: 'Model', w: 190, r: r => esc(r.label) + (r.status === 'scored' ? '' : ' ' + pill(r.status, 'wait')) },
    epoch: { h: 'Epoch', w: 64, n: 1, r: r => 'E' + esc(r.epoch) },
    verdict: { h: 'Result', w: 92, r: r => pill(...({ green: ['GREEN', 'up', 'Applied to the head.'], red: ['RED', 'dn', 'Not applied.'] }[r.verdict] || [r.verdict, 'wait', 'Not decided when v1 stopped.'])) },
    share: { h: 'Weight', w: 68, n: 1, r: r => pct(r.share), t: 'Share of the miner weight when v1 stopped.' },
    batch: { h: 'Batch', w: 92, n: 1, r: r => r.tasks == null ? mute('—') : esc(r.tasks) + (r.rollouts == null ? '' : ' · ' + esc(r.rollouts)), t: 'Tasks in the batch, then rollouts.' },
    cPass: { h: 'Cand. pass', w: 84, n: 1, r: r => pct(r.c_pass), t: 'Pass rate of the trained candidate on the scoring sample.' },
    hPass: { h: 'Head pass', w: 84, n: 1, r: r => pct(r.h_pass), t: 'Pass rate of the head on the same sample.' },
    cand: { h: 'Candidate', w: 100, r: r => key(r.candidate, 8, 0), t: 'Hash of the trained candidate.' },
    date: { h: 'Date (UTC)', w: 124, n: 1, r: r => r.at ? esc(utc(r.at).slice(0, 16)) : mute('—') },
    avg: { h: 'Avg', w: 64, n: 1, r: r => r.avg == null ? mute('—') : (100 * r.avg).toFixed(1) + dagger(r.avg_partial, 'At least one score of this row is from a partial run.'), t: 'Mean of the scored benchmarks of the row, in percent.' },
  };
  const T = {
    kings: { cols: ['rank', 'uid', 'hot', 'round', 'at', 'z', 'head', 'files'], empty: 'No king yet.' },
    queue: { cols: ['line', 'uid', 'hot', 'reg', 'check', 'flags'], empty: 'No registration waits in the queue.' },
    history: { cols: ['round', 'uid', 'hot', 'result', 'gain', 'z', 'bar', 'capHit', 'look', 'sha', 'files', 'at'], empty: 'No result yet.' },
    rounds: { cols: ['round', 'env', 'head', 'seats', 'skipped', 'tasks', 'mPass', 'mCap', 'outcome', 'at'], empty: 'No round yet.' },
    heads: { cols: ['head', 'round', 'kind', 'manifest', 'at'], empty: 'No head yet.' },
    bench: { cols: V1 ? ['model', 'date', 'avg'] : ['model', 'round', 'at', 'avg'], empty: 'No benchmark result yet.' },
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

  function bench(b) {
    const lead = T.bench.cols.filter(k => !k.startsWith('bench:'));
    T.bench.cols = [...lead, ...b.columns.map(c => {
      C['bench:' + c.id] = { h: c.label, w: 78, n: 1, r: benchCell(c.id), t: c.in_avg ? 'Score in percent.' : 'Shown only: not counted in Avg.' };
      return 'bench:' + c.id;
    })];
    shell('bench');
    const kings = b.rows.filter(x => x.kind === 'king').length;
    fill('bench', b.rows, !b.rows.length ? '' : V1 ? `${kings} kings` : `${kings} kings · updated ${dur(Date.now() / 1000 - b.updated_at)} ago`);
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
    bench(d.bench || { columns: [], rows: [] });
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
  Object.keys(T).filter(table).forEach(shell);
  load(window.SITE_DATA);
  if (!V1) setInterval(load, 60000);
})();
