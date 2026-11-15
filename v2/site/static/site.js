/* SN120 v2 status site. One column schema and one renderer for every table, as on the v1 site (k1-tables.js).
   The benchmark table, its score cards and the chart tiles copy the structure of the v1 site (bench.js, k1.js, timing.js).
   A page reads one file, and <body data-page> names the page. index.html (the benchmark table first) and round.html (the live
   round) read site-data.json, which build.py writes from the publisher's public snapshot. supply.html reads miner-supply.json,
   which supply.py writes. v1.html reads v1-data.json, the frozen v1 history that v1_archive.py wrote once at cutover.
   A part of the page is filled only where the page has it, so each page uses this one file.
   Privacy: build.py puts a submission link and a LoRA hash on an applied winner only; every other submission shows "private".
   Terms: submission = one LoRA upload; round = one evaluation cycle; batch = the submissions seated in a round; king = round winner. */
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
  // The small bar of "vs bar": its length is the distance to the bar, 22 px at 3 standard errors and more.
  const margin = (m, t) => `<span class="bar ${m > 0 ? 'up' : m < 0 ? 'dn' : 'mute'}" title="${esc(t)}"><i style="width:${Math.round(Math.min(1, Math.abs(m) / 3) * 22)}px"></i>${sgn(m)}</span>`;
  const RESULT = {
    winner: ['WINNER', 'up', 'Beat the head M by more than the winner bar.'],
    lost: ['LOST', 'dn', 'Had a fair tournament and did not win. This hotkey already had their one turn.'],
    no_decision: ['NO DECISION', 'wait', 'Our infrastructure failed. Not charged; the seat goes back to the front of the queue.'],
    skip: ['SKIP', 'private', 'The file was missing or invalid when the round froze it. Not charged; the seat keeps its place.'],
    not_trained: ['NOT TRAINED', 'wait', 'The training time of the round ended before this submission was trained. Not charged; the seat is first in the next round.'],
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
  const FILE_STATUS = {
    valid: ['valid', 'up', 'The stored file passes every check.'],
    not_valid: ['invalid', 'dn', 'The stored file failed a check.'],
    not_judged: ['not judged', 'wait', 'No verdict yet at the last file check, or no upload then.'],
    no_file: ['no file', 'wait', 'No file is stored for this seat.'],
  };
  const ZONE = {
    round: ['in round', 'up', 'Leased into the round that runs now.'],
    front: ['front', 'wait', FLAG.front[2]],
    back: ['back', 'wait', FLAG.demoted[2]],
    normal: ['normal', 'private', 'Ordinary place in the waiting line.'],
  };
  const QUEUE_PAGE = 50;
  let queuePage = 0, queueShowAll = false;
  const PAGE = document.body.dataset.page || 'home', V1 = PAGE === 'v1', SUPPLY = PAGE === 'supply', DATA = V1 ? 'v1-data.json' : 'site-data.json';
  let D = null, filter = '';
  // sort: which column of which table the visitor last chose, and the direction (1 = low to high, -1 = high to low). A second click on the same column flips the direction.
  const sorts = {}, held = {};  // held[n] = the last rows that fill() received for table n, before the visitor's sort
  let openKing = null;  // the hotkey of the king whose detail panel is open, or null
  let ROUND = {};  // the closed rounds of the data file, by round id: the link to the round file, and the cap-hit of M
  // The cap-hit limit of the round of a row, as the judge has it: the larger of `cap_hit_floor` and the cap-hit of M plus `cap_hit_margin`.
  // The page knows it only when the data file has the two settings and the cap-hit of M of that round. If not, no cell is marked.
  const capLimit = r => D.cap_hit_floor == null || D.cap_hit_margin == null || (ROUND[r.round_id] || {}).null_cap == null ? null : Math.max(D.cap_hit_floor, ROUND[r.round_id].null_cap + D.cap_hit_margin);
  const overCap = r => r.cap_rate != null && capLimit(r) != null && r.cap_rate > capLimit(r);
  // The word for a row that has no score: a king carried over from v1, or a submission that was not scored. The v1 page keeps its dash.
  const unscored = r => V1 ? mute('—')
    : r.imported ? no('Carried over from v1: this king has no v2 score and no v2 head. Its v1 score is on the v1 history page.')
    : no('Not scored. ' + (r.status == null ? NO_RESULT : RESULT[r.status] || ['', '', 'The round gave this submission no score.'])[2]);
  const noM = r => no('The round ended before the head M was scored, so the round file has no value.' + why(r));

  const files = r => r.batch
    ? link(r.batch, 'batch.json', 'The winning submission file, byte for byte') + (r.lora ? ' ' + key(r.lora, 6, 0) : '')
    : r.imported ? link('v1.html', 'v1 king', 'Carried over from v1 at cutover. The v1 history is on its own page.')
    : pill('private', 'private', 'Only a winner is published. This submission stays private.');
  const dagger = (c, t) => c ? `<sup title="${esc(t)}">†</sup>` : '';

  // Submission size and train settings. Only the winner's submission is public, so each other row says why it has no value.
  const closed = (r, what) => r.imported ? unscored(r)
    : r.batch ? no(`The winner's submission is public, but the data file of this page does not have ${what} yet. It is in batch.json.`, 'not read')
    : r.line != null ? pill('private', 'private', 'A file in the queue stays private. Only a winner submission is published.')
    : pill('private', 'private', 'Only a winner is published. This submission stays private.');
  const batchSize = r => r.batch_prompts == null ? closed(r, 'its size')
    : `<span title="${esc(r.batch_prompts)} prompts${r.batch_rollouts == null ? '' : ` · ${esc(r.batch_rollouts)} rollouts`}">${esc(r.batch_prompts)}${r.batch_rollouts == null || !r.batch_prompts ? '' : ' × ' + (r.batch_rollouts / r.batch_prompts).toFixed(1)}</span>`;
  // A setting shows in the cell only when the miner set it to a value that is not the default. The tooltip has each setting.
  const HP = { lora_lr: 'lr', sample_temperature: 'temp', target_modules: 'modules', group_size: 'group' };
  const setting = v => Array.isArray(v) ? v.join(', ') : typeof v === 'number' && v && Math.abs(v) < 1e-3 ? v.toExponential() : String(v);
  const HP_HELP = 'Train settings that the miner set to a value that is not the default. Each setting is in the tooltip. Public for a winner only.';
  const hparams = r => {
    if (r.train == null) return closed(r, 'its train settings');
    const known = D.train_settable || [], std = Object.fromEntries(known.map(x => [x.key, x.default])), keys = [...new Set([...known.map(x => x.key), ...Object.keys(r.train)])];
    const set = keys.filter(k => k in r.train && setting(r.train[k]) !== setting(std[k]));
    const all = keys.map(k => `${k} ${setting(k in r.train ? r.train[k] : std[k])}${!(k in std) ? '' : set.includes(k) ? ` (default ${setting(std[k])})` : ' (default)'}`).join('\n');
    return set.length ? `<span class="hp" title="${esc(all)}">${set.map((k, i) => `<span>${esc(HP[k] || k)} ${esc(setting(r.train[k]))}${i < set.length - 1 ? ' ·' : ''}</span>`).join(' ')}</span>`
      : mute('defaults', 'The miner set no train setting to a value that is not the default.' + (all ? '\n' + all : ''));
  };

  // h = header, w = width in px, n = numeric (right aligned), t = header tooltip, r = cell from a row,
  // s = value used to sort the column when the visitor clicks the header (null values sort last)
  const num = k => r => { const v = r[k]; return v == null || v !== v ? null : +v; };
  const text = k => r => { const v = r[k]; return v == null || v === '' ? null : String(v).toLowerCase(); };
  const C = {
    rank: { h: '#', w: 36, n: 1, s: num('rank'), r: r => mute(r.rank) },
    line: { h: '#', w: 36, n: 1, s: num('line'), r: r => mute(r.line), t: 'Place in line (1 = first).' },
    round: { h: 'Round', w: 92, s: text('round_id'), r: r => r.file_url || (ROUND[r.round_id] || {}).file_url ? link(r.file_url || ROUND[r.round_id].file_url, r.round_id, 'round.json') : esc(r.round_id || '—') },
    uid: { h: 'UID', w: 52, n: 1, s: num('uid'), r: r => esc(r.uid ?? '—') },
    hot: { h: 'Hotkey', w: 104, s: text('hotkey'), r: r => key(r.hotkey) },
    z: { h: 'z', w: 60, n: 1, s: num('z'), r: r => score(r.z, 2, 'z = gain / standard error', unscored(r)), t: 'Gain over the head M, in standard errors.' },
    head: { h: 'Head', w: 132, s: text('head'), r: r => (r.head ? key(r.head, 8, 0) : unscored(r)) + (r.current ? ' ' + pill('current', 'up') : ''), t: 'Head version id.' },
    files: { h: 'Files', w: 136, r: files, t: 'Published for a winner only: the submission file, and the hash of the LoRA.' },
    reg: { h: 'Reg block', w: 88, n: 1, s: num('registration_block'), r: r => esc(r.registration_block ?? '—') },
    check: { h: 'File', w: 96, s: r => r.file_valid ? 1 : 0, r: r => r.file_valid ? pill('valid', 'up', 'The stored file passes every check.') : mute('no valid file', 'This seat is skipped, is not charged, and keeps its place.'), t: 'Live check of the stored file. It never shows file content.' },
    flags: { h: 'Status', w: 170, s: r => (r.flags || []).join(' '), r: r => r.flags.map(f => pill(...FLAG[f])).join(' ') || mute('waiting') },
    fileStatus: { h: 'File status', w: 110, s: r => ({ valid: 0, not_valid: 1, not_judged: 2, no_file: 3 }[r.file_verdict] ?? 9), r: r => pill(...(FILE_STATUS[r.file_verdict] || [r.file_verdict || 'unknown', 'private', ''])), t: 'Live check of the stored file: valid, invalid, not judged, or no file. It never shows file content.' },
    zone: { h: 'Zone', w: 72, s: text('zone'), r: r => { const flags = r.flags || []; const z = r.zone || (flags.includes('leased') ? 'round' : flags.includes('front') ? 'front' : flags.includes('demoted') ? 'back' : 'normal'); return z === 'normal' ? mute('normal', ZONE.normal[2]) : pill(...(ZONE[z] || [z, 'private', ''])); }, t: 'Turn-order zone: front, normal, or back of queue.' },
    result: { h: 'Result', w: 112, s: text('status'), r: r => pill(...(r.status == null ? NO_RESULT : RESULT[r.status] || [r.status, 'private', ''])) },
    gain: { h: 'Gain', w: 76, n: 1, s: num('delta'), r: r => score(r.delta == null ? null : 100 * r.delta, 2, `Pass rate minus the pass rate of M, in percentage points (standard error ${(100 * r.se).toFixed(2)})`, r.rank && !r.imported ? no('The round of this king is no longer in the history of mining.json, so this page has no gain for it.', 'not listed') : unscored(r)), t: 'LoRA pass rate minus head pass rate, in percentage points.' },
    bar: { h: 'vs bar', w: 96, n: 1, s: r => r.z == null ? null : r.z - D.z_bar, r: r => r.z == null ? unscored(r) : margin(r.z - D.z_bar, `z ${r.z.toFixed(3)} against the winner bar z > ${D.z_bar}: ${r.z > D.z_bar ? 'cleared by' : 'short by'} ${Math.abs(r.z - D.z_bar).toFixed(3)}`), t: 'z minus the winner bar. Above zero clears the bar.' },
    capHit: { h: 'Cap-hit', w: 68, n: 1, s: num('cap_rate'), r: r => overCap(r) ? `<span class="dn" title="Above the cap-hit limit of this round, ${pct(capLimit(r))}: the larger of ${pct(D.cap_hit_floor)} and the cap-hit of M plus ${pct(D.cap_hit_margin)}. A LoRA above the limit cannot win.">${pct(r.cap_rate)}</span>` : pct(r.cap_rate, unscored(r)), t: 'Share of answers that hit the token cap. Such an answer is not passed. Red: above the cap-hit limit of the round.' },
    look: { h: 'Dropped', w: 68, n: 1, s: num('look'), r: r => r.look != null ? 'look ' + (r.look + 1) : r.z == null ? unscored(r) : no('Not dropped at an early look.', 'no'), t: 'Looks are the stages rollouts are scored in. The early look at which this LoRA was dropped, if any.' },
    size: { h: 'Size', w: 76, n: 1, s: num('batch_prompts'), r: batchSize, t: 'Training prompts × mean rollouts for each prompt. Public for a winner only.' },
    hp: { h: 'Hparams', w: 146, r: hparams, t: HP_HELP },
    hpWide: { h: 'Hparams', w: 210, r: hparams, t: HP_HELP },
    sha: { h: 'Submission sha256', w: 120, s: text('sha'), r: r => key(r.sha, 8, 0) },
    at: { h: 'When', w: 92, n: 1, s: num('at'), r: r => ago(r.at) },
    date: { h: 'Date (UTC)', w: 124, n: 1, s: num('at'), r: r => r.at ? esc(utc(r.at).slice(0, 16)) : mute('—') },
    env: { h: 'Env', w: 60, s: text('env'), r: r => esc(r.env) },
    seats: { h: 'Submissions', w: 88, n: 1, s: num('seats'), r: r => esc(r.seats), t: 'Submissions seated in the round: valid file at round open. A submission whose training failed is still counted. Together they are the batch of the round.' },
    skipped: { h: 'Skipped', w: 68, n: 1, s: num('skipped'), r: r => esc(r.skipped), t: 'Seats with no valid file at round open. Not charged.' },
    tasks: { h: 'Tasks', w: 64, n: 1, s: num('tasks'), r: r => esc(r.tasks), t: 'Eval problems for each submission in this round.' },
    mPass: { h: 'M pass', w: 68, n: 1, s: num('null_pass'), r: r => pct(r.null_pass, noM(r)) },
    mCap: { h: 'M cap-hit', w: 76, n: 1, s: num('null_cap'), r: r => pct(r.null_cap, noM(r)) },
    outcome: { h: 'Outcome', w: 132, s: text('outcome'), r: r => (OUTCOME[r.outcome] || (() => esc(r.outcome)))(r) },
    kind: { h: 'Kind', w: 84, s: r => r.folded ? 1 : 0, r: r => r.folded ? mute('folded', 'The LoRAs are merged into the base.') : mute('stacked', 'The winning LoRA is stacked on the base.') },
    manifest: { h: 'Manifest', w: 84, r: r => r.manifest ? link(r.manifest, 'head.json', 'File hashes of this head') : no('No manifest file of this head is on this site. The validator writes the manifest of a head only while that head is the current one.', 'no file') },
    epoch: { h: 'v1 turn', w: 64, n: 1, s: num('epoch'), r: r => 'E' + esc(r.epoch), t: 'v1 turn index (historical). v2 uses rounds, not epochs.' },
    verdict: { h: 'Result', w: 92, s: text('verdict'), r: r => pill(...({ green: ['GREEN', 'up', 'Applied to the head.'], red: ['RED', 'dn', 'Not applied.'] }[r.verdict] || [r.verdict, 'wait', 'Not decided when v1 stopped.'])) },
    share: { h: 'Weight', w: 68, n: 1, s: num('share'), r: r => pct(r.share), t: 'Share of the miner weight when v1 stopped.' },
    batch: { h: 'Tasks', w: 92, n: 1, s: num('tasks'), r: r => r.tasks == null ? mute('—') : esc(r.tasks) + (r.rollouts == null ? '' : ' · ' + esc(r.rollouts)), t: 'Tasks, then rollouts.' },
    cPass: { h: 'Cand. pass', w: 84, n: 1, s: num('c_pass'), r: r => pct(r.c_pass), t: 'Pass rate of the trained candidate on the scoring sample.' },
    hPass: { h: 'Head pass', w: 84, n: 1, s: num('h_pass'), r: r => pct(r.h_pass), t: 'Pass rate of the head on the same sample.' },
    cand: { h: 'Candidate', w: 100, s: text('candidate'), r: r => key(r.candidate, 8, 0), t: 'Hash of the trained candidate.' },
  };
  const T = {
    kings: { cols: ['rank', 'uid', 'hot', 'size', 'hpWide', 'round', 'at', 'z', 'gain', 'head', 'files'], empty: 'No king yet.' },
    queueRound: { cols: ['line', 'uid', 'hot', 'fileStatus', 'reg'], empty: 'No seat is in the running round.' },
    queueWait: { cols: ['line', 'uid', 'hot', 'fileStatus', 'zone', 'reg'], empty: 'No registration waits in the queue.' },
    history: { cols: ['round', 'uid', 'hot', 'size', 'hp', 'result', 'gain', 'z', 'bar', 'capHit', 'look', 'sha', 'files', 'at'], empty: 'No result yet.' },
    rounds: { cols: ['round', 'env', 'head', 'seats', 'skipped', 'tasks', 'mPass', 'mCap', 'outcome', 'at'], empty: 'No round yet.' },
    heads: { cols: ['head', 'round', 'kind', 'manifest', 'at'], empty: 'No head yet.' },
    v1kings: { cols: ['rank', 'uid', 'hot', 'epoch', 'z', 'share'], empty: 'No king.' },
    v1turns: { cols: ['epoch', 'uid', 'hot', 'verdict', 'z', 'cPass', 'hPass', 'capHit', 'batch', 'cand', 'head', 'date'], empty: 'No turn.' },
  };

  const tableRoot = n => document.querySelector(`[data-table="${n}"]`) || document.getElementById(n);
  const table = n => tableRoot(n)?.querySelector('table');
  function ordered(n, rows) {
    const cur = sorts[n];
    if (!cur || !C[cur.key]?.s) return rows;
    const key = C[cur.key].s, dir = cur.dir;
    return rows.map((r, i) => [key(r), i, r]).sort((a, b) => {
      if (a[0] == null && b[0] == null) return a[1] - b[1];
      if (a[0] == null) return 1;
      if (b[0] == null) return -1;
      return a[0] < b[0] ? -dir : a[0] > b[0] ? dir : a[1] - b[1];
    }).map(x => x[2]);
  }
  function shell(n) {
    const t = table(n); if (!t) return;
    const cols = T[n].cols.map(k => C[k]), cur = sorts[n];
    t.querySelectorAll('colgroup,thead').forEach(el => el.remove());
    t.style.minWidth = cols.reduce((a, c) => a + c.w, 0) + 'px';
    t.insertAdjacentHTML('afterbegin', '<colgroup>' + cols.map(c => `<col style="width:${c.w}px">`).join('') + '</colgroup><thead><tr>'
      + T[n].cols.map(k => {
        const c = C[k], on = cur && cur.key === k, tip = [c.t, c.s ? 'Click to sort.' : ''].filter(Boolean).join(' ');
        return `<th class="${c.n ? 'num' : ''}${c.s ? ' sort' : ''}${on ? ' sorted' : ''}"${c.s ? ` data-sort="${k}"` : ''}${tip ? ` title="${esc(tip)}"` : ''}${on ? ` aria-sort="${cur.dir > 0 ? 'ascending' : 'descending'}"` : ''}>${esc(c.h)}${on ? `<span class="sort-mark" aria-hidden="true">${cur.dir > 0 ? '↑' : '↓'}</span>` : ''}</th>`;
      }).join('') + '</tr></thead>');
  }
  function kingDetail(r) {
    const box = document.getElementById('king-detail');
    if (!box) return;
    if (!r) { box.hidden = true; box.innerHTML = ''; return; }
    const lines = [
      ['UID', esc(r.uid ?? '—')],
      ['Hotkey', r.hotkey ? `<button type="button" class="key" data-copy="${esc(r.hotkey)}" title="${esc(r.hotkey)} (click to copy)">${esc(r.hotkey)}</button>` : mute('—')],
      ['Round', r.file_url ? link(r.file_url, r.round_id || 'round.json', 'round.json') : esc(r.round_id || '—')],
      ['When', r.at ? esc(utc(r.at)) : mute('—')],
      ['z', r.z == null ? unscored(r) : esc(sgn(r.z))],
      ['Gain', r.delta == null ? unscored(r) : esc(sgn(100 * r.delta) + ' pp')],
      ['Head', r.head ? `<button type="button" class="key" data-copy="${esc(r.head)}" title="${esc(r.head)} (click to copy)">${esc(r.head)}</button>` : unscored(r)],
      ['Files', `<span class="files">${files(r)}</span>`],
    ];
    box.innerHTML = `<div class="detail-top"><h3>UID ${esc(r.uid ?? '—')}</h3><button type="button" class="link" data-close-detail>Close</button></div><dl>${lines.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;
    box.hidden = false;
  }
  function fill(n, rows, reading, more) {
    if (!table(n)) return;
    held[n] = { rows, reading, more };
    const cols = T[n].cols, body = table(n).tBodies[0], box = body.closest('.scroll'), root = tableRoot(n);
    let view = ordered(n, rows);
    if (n === 'queueWait') {
      const total = view.length, pages = Math.max(1, Math.ceil(total / QUEUE_PAGE));
      if (queuePage >= pages) queuePage = pages - 1;
      if (!queueShowAll && total > QUEUE_PAGE) {
        const start = queuePage * QUEUE_PAGE;
        view = view.slice(start, start + QUEUE_PAGE);
        more = `showing ${start + 1}–${start + view.length} of ${total} waiting`;
      } else if (total) {
        more = `showing 1–${total} of ${total} waiting`;
      } else {
        more = '';
      }
      const pager = document.getElementById('queue-pager');
      if (pager) {
        pager.hidden = total <= QUEUE_PAGE;
        if (total > QUEUE_PAGE) {
          pager.innerHTML = `<span class="pager-label">${esc(more)}</span>`
            + `<button type="button" data-qpage="prev" ${queueShowAll || queuePage <= 0 ? 'disabled' : ''}>Previous</button>`
            + `<button type="button" data-qpage="next" ${queueShowAll || queuePage >= pages - 1 ? 'disabled' : ''}>Next</button>`
            + `<button type="button" data-qpage="all">${queueShowAll ? 'Show 50 per page' : 'Show all'}</button>`;
          more = '';  // label lives in the pager, not as a table footnote
        }
      }
    }
    shell(n);  // refresh the sort mark on the header of this table
    body.innerHTML = (view.length
      ? view.map(r => `<tr${n === 'kings' && openKing && r.hotkey === openKing ? ' class="open"' : ''}${n === 'kings' && r.hotkey ? ` data-king="${esc(r.hotkey)}"` : ''}>` + cols.map(k => `<td class="${C[k].n ? 'num' : ''}">${C[k].r(r)}</td>`).join('') + '</tr>').join('')
      : `<tr><td colspan="${cols.length}" class="empty">${esc(T[n].empty)}</td></tr>`)
      + (more ? `<tr class="more"><td colspan="${cols.length}">${esc(more)}</td></tr>` : '');
    const readingEl = root?.querySelector('.reading');
    if (readingEl) readingEl.textContent = reading;
    if (box.classList.contains('wait')) { box.classList.remove('wait'); body.classList.add('fade'); }
    if (n === 'kings') kingDetail(openKing ? view.find(r => r.hotkey === openKing) || rows.find(r => r.hotkey === openKing) : null);
  }
  function queueTotals(summary) {
    const el = document.getElementById('queue-totals');
    if (!el) return;
    if (!summary) { el.innerHTML = ''; return; }
    const items = [
      ['In the running round', summary.in_round],
      ['Waiting', summary.waiting],
      ['Valid file', summary.valid],
      ['Invalid file', summary.not_valid],
      ['Not judged or no upload', summary.not_judged],
      ['Front', summary.front],
      ['Back of queue', summary.back],
    ];
    el.innerHTML = items.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');
  }

  /* One chart for every tile and every score card, as k1.js and bench.js of the v1 site: axis values (no grid lines), bars or a
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
  // the top of a tooltip in its chart: under the sticky header of the page when the top of the chart is behind that header
  const clear = (box, tip) => {
    const head = document.querySelector('body>header');
    if (!head || getComputedStyle(head).position !== 'sticky' || box.closest('.cx-open')) return '';
    const behind = head.getBoundingClientRect().bottom - box.getBoundingClientRect().top;
    return behind > 0 ? Math.max(4, Math.min(behind + 4, box.clientHeight - tip.offsetHeight)) + 'px' : '';
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
    for (let v = lo; v <= hi + st / 2; v += st) g.append(svg('text', { x: P.l - 6, y: y(v) + 3, 'text-anchor': 'end' }, o.fmt(v)));
    if (o.ref != null) g.append(svg('line', { x1: P.l, x2: W - P.r, y1: y(o.ref), y2: y(o.ref), class: 'ref' }));
    const zero = y(Math.min(hi, Math.max(lo, 0))), bar = Math.max(2, Math.min(26, slot * .62));
    for (const s of o.series) {
      const has = idx.filter(i => s.v(i) != null), color = i => typeof s.c === 'function' ? s.c(i) : s.c;
      if (s.plain) continue;
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
    if (o.marks) o.marks(g, { xs, y, bar, P, W, H });
    const tip = document.createElement('output');
    tip.className = 'tip'; tip.hidden = true; box.append(tip);
    idx.forEach(i => {
      const l = i ? (xs[i - 1] + xs[i]) / 2 : P.l, r = i < o.n - 1 ? (xs[i] + xs[i + 1]) / 2 : W - P.r;
      const hit = svg('rect', { x: l, y: P.t, width: Math.max(1, r - l), height: H - P.t - P.b, fill: 'transparent' });
      hit.addEventListener('mouseenter', () => { tip.innerHTML = o.tip(i); tip.hidden = false; tip.style.left = Math.min(Math.max(0, xs[i] - tip.offsetWidth / 2), Math.max(0, W - tip.offsetWidth)) + 'px'; });
      hit.addEventListener('mouseleave', () => { tip.hidden = true; });
      hit.addEventListener('mouseenter', () => { tip.style.top = clear(box, tip); });
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
  // The sampling of a score, from the settings profile that the benchmark file names: its presence penalty. One list for the tooltip
  // of a score, the note under the table and the second line of a row. The benchmark file gives only the name of a profile, so the
  // values are here; `profiles` of the data file wins when it is there. A profile that neither has is shown by its name.
  const PP = { 'qwen3.5-family': 1.5, 'qwen3.5-family-pp0': 0 };
  let PROFILES = {};
  const penalty = name => (PROFILES[name] || {}).presence_penalty ?? PP[name] ?? null;
  const pp = name => penalty(name) == null ? `profile ${name}` : `presence penalty ${penalty(name)}`;
  const ppShort = name => penalty(name) == null ? name : penalty(name) ? `pp ${penalty(name)}` : 'pp0';
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
  const size = r => r.size_b ?? SIZE[String(r.label).toLowerCase()] ?? null;
  // The short name of a reference model with a long label: the table SHORT of bench.js of the v1 site, by the label of the model.
  const SHORT = {
    'deepseek-r1-0528-qwen3-8b': 'R1-0528-Qwen3-8B', 'r1-distill-qwen-7b': 'R1-Qwen-7B', 'r1-distill-llama-8b': 'R1-Llama-8B', 'r1-distill-qwen-14b': 'R1-Qwen-14B',
    'r1-distill-qwen-1.5b': 'R1-Qwen-1.5B', 'nemotron-nano-9b-v2': 'Nemotron-Nano-9B', 'glm-z1-9b-0414': 'GLM-Z1-9B', 'openreasoning-nemotron-7b': 'OR-Nemotron-7B',
    'acereason-nemotron-1.1-7b': 'AceReason-7B', 'llama-3.1-nemotron-nano-8b-v1': 'Nemotron-Nano-8B', 'phi-4-reasoning': 'Phi-4-R-14B', 'phi-4-mini-reasoning': 'Phi-4-mini-R',
    'deepscaler-1.5b-preview': 'DeepScaleR-1.5B',
  };
  // The width of the column Model. At the width of a phone it is narrow and a name takes more lines, as on the v1 site. On a wider
  // screen it is as wide as the longest second line (`letters`, at 6 px a letter), up to the room that the box has with no sideways scroll.
  const nameWidth = (letters, room) => innerWidth <= 640 ? Math.min(Math.round(innerWidth * .46), 168) : Math.max(200, Math.min(Math.ceil(6.05 * letters) + 18, room));
  const STAR = '<svg class="star" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 .6l1.3 2.8 3 .3-2.3 2 .7 3L5 7.2 2.3 8.7l.7-3-2.3-2 3-.3z"/></svg>';
  const shade = f => { const k = Math.min(1, Math.max(0, (.9 - f) / .75)); return `hsl(${Math.round(130 * (1 - k ** .75))} 65% 45% / ${(.3 + .28 * Math.abs(.5 - k)).toFixed(2)})`; };
  let allRefs = false;
  // The reference rows of the home page, by label in small letters. A reference model that is not in this list stays in the data
  // and has its row on the v1 page.
  const HOME_REFS = ['qwen3-8b'];
  function bench(b) {
    PROFILES = b.profiles || {};
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
    const lead = rated.filter(r => counted.some(c => leads(r, c)));
    const pinned = V1 ? allRefs || !lead.length ? refs : lead : refs.filter(r => HOME_REFS.includes(String(r.label).toLowerCase()));
    // The home page: the reference rows and the newest king stay in view above the list, and the genesis head stays in view under it.
    // The list that scrolls has the older kings, newest first, and then the v1 king. The v1 page has no row above its list.
    const above = V1 ? [] : [...pinned, ...kings.slice(0, 1)], listed = V1 ? kings : [...kings.slice(1), ...old], under = V1 ? [...pinned, ...genesis, ...old] : genesis;
    const shown = [...kings, ...refs, ...genesis, ...old], inView = V1 ? shown : [...above, ...listed, ...under], span = {};  // span: the lowest and the highest score of a column, for the shading; and the highest of the rows in view, for the star
    for (const c of cols.filter(c => c.in_avg)) { const v = shown.map(r => val(r, c)).filter(x => x != null); if (v.length) span[c.id] = [Math.min(...v), Math.max(...v), Math.max(...inView.map(r => val(r, c)).filter(x => x != null))]; }
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
      const sets = [[kings, 'the kings'], [old, old.length > 1 ? 'the v1 kings' : 'the v1 king'], [genesis, genesis.length > 1 ? 'the genesis heads' : 'the genesis head'], [pinned, pinned.length > 1 ? 'the reference models' : 'the reference model']]
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
      const f = s && s[1] > s[0] ? (x - s[0]) / (s[1] - s[0]) : 1, top = s && Math.round(1000 * x) === Math.round(1000 * s[2]);
      const title = (top ? 'Highest of the column. ' : '') + (one ? (one.se == null ? '' : `±${(100 * one.se).toFixed(1)} · `) + `cap-hit ${share(one.cap_rate, 1, 'not recorded')}` + (one.rollouts ? ` · ${samples(one.rollouts)} for each problem` : '') + (one.profile ? ` · ${pp(one.profile)}` : '') + (one.derived ? ` · computed from ${from(one, longer(c) && r.cells[longer(c).id])} of the full run` : '') : 'Mean of the counted benchmarks.');
      return `<td class="num${avg}"${s ? ` style="background:linear-gradient(${shade(f)},${shade(f)}),var(--bg)"` : ''} title="${esc(title)}">${top ? STAR : ''}${(100 * x).toFixed(1)}`
        + (one ? dagger(one.partial, `Partial run: ${one.samples} of ${one.expected} samples. A lost sample counts as wrong.`) : dagger(r.avg_partial, 'At least one score of this row is from a partial run.')) + '</td>';
    };
    // The second line under a name, as on the v1 site: `budget 65,536 tok · cut-off 3.3% · pp 1.5`. The budget is the token budget
    // of the run. The cut-off is the mean cap-hit of the scores of the row. The sampling is the presence penalty of the settings profile
    // of the scores. Each part is there only when the data has it, so a row with none of them has one line.
    const meta = r => {
      const caps = Object.values(r.cells).map(x => x.cap_rate).filter(x => x != null), profiles = had(r).length ? had(r) : r.profile && run(r) === RUN.running ? [r.profile] : [];
      return [r.budget != null && `budget ${Number(r.budget).toLocaleString('en-US')} tok`, caps.length && `cut-off ${share(caps.reduce((a, x) => a + x, 0) / caps.length)}`, ...profiles.map(ppShort)].filter(Boolean);
    };
    const name = r => {
      const other = r.kind !== 'king' && r.kind !== 'genesis' && r.kind !== 'v1-king', billions = other && size(r) != null ? size(r).toFixed(1) + 'B' : '';
      const [first, ...rest] = r.kind === 'king' || r.kind === 'genesis' ? String(r.label).split(' ') : r.kind === 'v1-king' && r.uid != null ? [r.label, 'UID ' + r.uid] : [SHORT[String(r.label).toLowerCase()] || r.label, ...(billions ? [billions] : [])];
      const model = other && r.repo ? `<a href="https://huggingface.co/${esc(r.repo)}" target="_blank" rel="noopener" title="${esc(r.repo + (billions ? ` · ${billions} params` : ''))}">${esc(first)}</a>` : esc(first);
      return `<span class="who"><span>${model}${rest.length ? ' ' + mute(rest.join(' ')) : ''}</span>` + (run(r) ? pill(...run(r).slice(0, 3)) : '') + '</span>' + (meta(r).length ? `<span class="meta">${meta(r).map((x, i, all) => `<span>${esc(x)}${i < all.length - 1 ? ' ·' : ''}</span>`).join(' ')}</span>` : '');
    };
    const about = r => [r.label, r.repo && r.kind === 'reference' && `Hugging Face ${r.repo}${r.repo_rev ? ' at ' + String(r.repo_rev).slice(0, 8) : ''}`, r.budget != null && `budget ${Number(r.budget).toLocaleString('en-US')} tokens`, size(r) && r.kind === 'reference' && `${size(r)} billion parameters`, r.kind === 'v1-king' && (r.uid != null ? `UID ${r.uid} · ` : '') + (r.v1_epoch != null ? `v1 epoch ${r.v1_epoch} · ` : '') + V1_KING, r.head && 'Head ' + r.head, run(r) && run(r)[2], had(r).length ? 'Sampling: ' + list(had(r).map(pp)) : r.profile && run(r) === RUN.running && 'Sampling of its run: ' + pp(r.profile), r.pod && !(run(r) && POD[r.pod.state]) && `Machine: ${r.pod.state}.${since(r)}`, r.at && 'Applied ' + utc(r.at), r.v1 && 'Scored by the benchmark service of v1; from the v1 history.'].filter(Boolean).join(' · ');
    const row = r => `<tr class="${esc(r.kind)}"><td class="name" title="${esc(about(r))}">${name(r)}</td>${cols.map(c => cell(r, c)).join('')}</tr>`;
    const suites = [...new Set(b.columns.map(c => c.suite).filter(Boolean))], t = table('bench'), box = t.closest('.scroll');
    t.querySelector('colgroup')?.remove();
    const nameW = nameWidth(Math.max(0, ...[...kings, ...pinned, ...genesis, ...old].map(r => meta(r).join(' · ').length)), box.clientWidth - 64 - 88 * b.columns.length);
    t.insertAdjacentHTML('afterbegin', `<colgroup><col style="width:${nameW}px"><col style="width:64px">${b.columns.map(() => '<col style="width:88px">').join('')}</colgroup>`);
    t.style.minWidth = nameW + 64 + 88 * b.columns.length + 'px';
    t.tHead.innerHTML = (suites.length > 1 ? `<tr class="suites"><th></th><th class="avg"></th>${suites.map(x => `<th class="suite" colspan="${b.columns.filter(c => c.suite === x).length}">${esc(x)}</th>`).join('')}</tr>` : '')
      + `<tr><th>Model</th><th class="num avg" title="Mean of the counted benchmarks, in percent. A row has it only when it has every counted benchmark.">Avg</th>`
      + b.columns.map(c => `<th class="num" title="${esc(c.label)}: ${c.in_avg ? 'score in percent.' : 'shown only; not counted in Avg, not shaded.'}${c.full_of ? ' The score of the full run, for a model that has one.' : ''}${esc(differ(c) && ' ' + differ(c).slice(c.label.length + 2))}">${esc(c.head || c.label)}</th>`).join('') + '</tr>';
    const body = t.querySelector('tbody:not(.top)'), pin = t.querySelector('tbody.top') || body.insertAdjacentElement('beforebegin', Object.assign(document.createElement('tbody'), { className: 'top' }));
    pin.innerHTML = above.map(row).join('');
    body.innerHTML = (kings.length ? '' : `<tr><td colspan="${cols.length}" class="empty">${V1 ? 'No king was scored.' : 'No king yet: no round has applied a winner.'}</td></tr>`) + listed.map(row).join('');
    t.tFoot.innerHTML = under.map(row).join('');
    t.style.setProperty('--head', t.tHead.offsetHeight + 'px');
    let h = t.tHead.offsetHeight + pin.offsetHeight + t.tFoot.offsetHeight + 2 + (box.scrollWidth > box.clientWidth ? 12 : 0);
    for (const r of [...body.rows].slice(0, 5)) h += r.offsetHeight;
    box.style.height = h + 'px';
    if (!V1) { box.style.maxHeight = h + 'px'; box.style.height = 'auto'; }  // a limit, not a fixed height: a list of fewer rows has no scroll bar and no empty band
    if (box.classList.contains('wait')) { box.classList.remove('wait'); t.classList.add('fade'); }
    const scored = every.filter(r => r.status === 'scored').length, more = document.getElementById('refs');
    more.hidden = !V1 || !lead.length || lead.length === refs.length;
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
     v2 seats several submissions per round (the batch of the round); "z" is the z of the applied king, or the best z of a round with no winner. */
  const OPEN_PAD = { l: 58, r: 16, t: 18, b: 28 };  // the open tile has larger axis text
  const ENDED = { applied: r => `UID ${r.winner_uid} applied`, null: () => 'M stays', no_decision: () => 'no decision', abandoned: () => 'abandoned' };
  const TONE = { applied: 'var(--performance)', null: 'var(--rejected)' };
  const OUTCOMES = [['up', 'winner applied'], ['dn', 'no winner: M stays'], ['wait hollow', 'no decision']];
  // the phases of a round, in the order of the round: key in the data, name, legend mark, colour, help text
  const PHASES = [
    ['training', 'Training', 'up', 'var(--performance)', 'The time that the GPU box spent to train the submissions of the round.'],
    ['evaluation', 'Evaluation', 'wait', 'var(--accepted)', 'The time that the GPU box spent to score the head M and each LoRA of the round, by looks.'],
    ['promote', 'Promote', 'ink', 'var(--ink)', 'From the result of the round to the new head. Only a round with an applied winner has it.'],
    ['fold', 'Fold', 'mid', 'var(--muted)', 'The merge of the applied LoRAs into a new base, after the new head. Only a round with an applied winner has it.'],
    ['other', 'Other', 'low', 'var(--axis)', 'The rest of the round, which the round file gives to no phase: open, wait and close.'],
  ];
  const sci = v => (+v.toPrecision(3)).toExponential();  // 1e-5, 2.5e-7
  const tile = (id, reading, o) => {
    const el = document.getElementById(id), r = el.querySelector('.reading');
    r.textContent = reading || 'none';
    if (o.hint) r.title = o.hint; else r.removeAttribute('title');
    chart(el.querySelector('.chart'), { name: el.querySelector('h2').textContent, pad: el.classList.contains('cx-open') ? OPEN_PAD : null, ...o });
    return el;
  };
  const legend = (id, items) => {
    document.querySelector(`#${id} .legend`).innerHTML = items.map(([mark, text, help]) => `<span${help ? ` title="${esc(help)}"` : ''}><i class="sw ${mark}"></i>${esc(text)}</span>`).join('');
  };
  const titled = (id, text, help) => {
    const el = document.getElementById(id), h = el.querySelector('h2'), b = el.querySelector('.cx');
    if (h.textContent === text && h.title === help) return;
    h.textContent = text; h.title = help;
    if (b) b.setAttribute('aria-label', b.getAttribute('aria-label').split(' ')[0] + ' ' + text);
  };
  const SUBMISSIONS = ['Submissions per round', 'One bar per closed round. Outline: submissions seated in the round (the batch of that round, usually about 6 to 8). Filled: submissions that got a z. Hover a bar for the cap of the round, skipped seats and tasks.'];
  function tiles(d) {
    if (!document.getElementById('charts')) return;
    const rs = d.rounds.slice().reverse(), ended = r => (ENDED[r.outcome] || (() => r.outcome))(r), took = r => r.at - r.opened_at;
    const tip = (r, rows) => tipOf(`${r.round_id} · ${ended(r)}`, r.reason ? [...rows, ['validator', r.reason]] : rows);
    const round = o => ({ n: rs.length, slots: 10, label: i => rs[i].round_id, empty: 'No round is closed yet.', ...o });
    const tone = r => TONE[r.outcome] || 'var(--accepted)', undecided = r => !TONE[r.outcome];
    // as k1.js of v1 for a held turn: a dashed circle under the axis
    const ring = (g, x, at, color) => g.append(svg('circle', { cx: x, cy: at.H - at.P.b + 4.5, r: 2.5, fill: 'none', stroke: color, 'stroke-width': 1.2, 'stroke-dasharray': '2 1' }));
    const last = f => rs.filter(f).pop(), z = last(r => r.best_z != null), pass = last(r => r.null_pass != null), end = rs[rs.length - 1];
    tile('c-z', z && `${z.round_id} z ${sgn(z.best_z)}`, round({
      include: [0, d.z_bar], ref: d.z_bar, fmt: v => sgn(v, v % 1 ? 1 : 0),
      series: [{ bar: 1, v: i => rs[i].best_z, c: i => tone(rs[i]), hollow: i => undecided(rs[i]) }],
      tip: i => tip(rs[i], [[rs[i].outcome === 'applied' ? 'winner z' : 'best z', rs[i].best_z == null ? 'none: no submission was scored' : sgn(rs[i].best_z)], ['winner bar', 'z > ' + d.z_bar], ['submissions', rs[i].seats]]),
    }));
    tile('c-pass', pass && `${pass.round_id} M ${share(pass.null_pass)}`, round({
      bound: [0, 1], least: .01, fmt: whole,
      series: [{ v: i => rs[i].null_pass, c: 'var(--accepted)' }, { v: i => rs[i].winner_pass, c: 'var(--performance)' }],
      marks: (g, at) => rs.forEach((r, i) => undecided(r) && ring(g, at.xs[i], at, 'var(--accepted)')),
      tip: i => tip(rs[i], [['head M', share(rs[i].null_pass, 2, 'not scored')], ['winner', share(rs[i].winner_pass, 2, rs[i].outcome === 'applied' ? 'none' : 'no winner')], ['M cap-hit', share(rs[i].null_cap, 2, 'not scored')], ['tasks', rs[i].tasks]]),
    }));
    /* Submissions seated in each round (the batch of that round). Outline = seats; filled = submissions that got a z. */
    const seatTip = r => [['submissions', `${r.seats} of at most ${d.round.cap}`], ['with a z', r.with_z], ['skipped, no valid file', r.skipped], ['tasks', r.tasks]];
    titled('c-batch', ...SUBMISSIONS);
    legend('c-batch', [['up hollow', 'submissions seated'], ['up', 'submissions with a z']]);
    tile('c-batch', end && `${end.round_id} ${end.seats} of at most ${d.round.cap} · ${end.with_z} with a z`, round({
      include: [0], least: 1, fmt: v => v,
      hint: end && `Newest closed round ${end.round_id}: ${end.seats} submissions seated (cap ${d.round.cap}); ${end.with_z} with a z.`,
      series: [{ bar: 1, v: i => rs[i].seats, c: 'var(--performance)', hollow: () => 1 }, { bar: 1, v: i => rs[i].with_z || null, c: 'var(--performance)' }],
      tip: i => tip(rs[i], seatTip(rs[i])),
    }));
    const five = rs.slice(-5).map(took).sort((a, b) => a - b), mid = five.length >> 1, median = five.length && dur(five.length % 2 ? five[mid] : (five[mid - 1] + five[mid]) / 2);
    /* As timing.js of v1: a round much longer than the others does not set the scale. "Much longer" is more than 3 times the
       median and more than the median plus 30 minutes. The axis ends 15% above the longest of the other rounds, at a round
       value; the long bar is cut there, with a break mark and its real time above it. */
    const times = rs.map(took).sort((a, b) => a - b), med = times[times.length >> 1] || 0, usual = times.filter(v => v <= Math.max(3 * med, med + 1800));
    const unit = Math.max(0, ...usual) * 1.15 >= 7200 ? ['h', 3600] : ['m', 60];
    const cap = usual.length < times.length ? nice(0, Math.max(...usual) * 1.15 / unit[1], 4, 1)[1] : Infinity, cut = r => took(r) / unit[1] > cap;
    /* The bar of a round is split by phase, as v1, when the data has the phase times. A phase is drawn when it took 3 seconds
       or more (v1: 0.05 minutes). Each phase is one bar from zero to the sum up to that phase, the longest drawn first. A round
       with no phase times is one outline. Data with no phase times at all (an older builder) keeps one bar in the colour of
       the result. */
    const phased = rs.some(r => r.phases != null), spent = (r, k) => r.phases != null && r.phases[PHASES[k][0]] >= 3;
    const upTo = k => i => spent(rs[i], k) ? Math.min(cap, PHASES.slice(0, k + 1).reduce((sum, [name]) => sum + (rs[i].phases[name] || 0), 0) / unit[1]) : null;
    const full = i => Math.min(cap, took(rs[i]) / unit[1]);
    legend('c-time', !phased ? OUTCOMES : [...PHASES.filter((_, k) => rs.some(r => spent(r, k))).map(([, name, mark, , help]) => [mark, name, help]),
      ...(rs.some(r => r.phases == null) ? [['mid hollow', 'no phase times', 'The phase times of this round are not read yet. The bar is the time from round open to round close.']] : [])]);
    tile('c-time', end && `${end.round_id} ${dur(took(end))} · median of the last ${five.length} ${median}`, round({
      include: [0], least: 1, fmt: v => v + unit[0], bound: cap < Infinity ? [0, cap] : null,
      hint: end && `Newest closed round ${end.round_id}: ${dur(took(end))} from open to close. Median of the last ${five.length} closed rounds: ${median}.`,
      series: !phased ? [{ bar: 1, v: full, c: i => tone(rs[i]), hollow: i => undecided(rs[i]) }]
        : [{ bar: 1, v: i => rs[i].phases == null ? full(i) : null, c: 'var(--muted)', hollow: () => 1 }, ...PHASES.map((p, k) => ({ bar: 1, v: upTo(k), c: p[3] })).reverse()],
      tip: i => tip(rs[i], [[rs[i].phases != null ? 'total' : 'open to close', dur(took(rs[i]))], ...PHASES.filter((_, k) => spent(rs[i], k)).map(([name]) => [name, dur(rs[i].phases[name])]),
        ...(cut(rs[i]) ? [['bar', `cut at ${cap}${unit[0]}: much longer than the other rounds`]] : []), ['opened', utc(rs[i].opened_at)], ['closed', utc(rs[i].at)]]),
      marks: (g, { xs, y, bar, P, W }) => rs.forEach((r, i) => {
        if (!cut(r)) return;
        const x = xs[i] - bar / 2, yb = y(cap) + 6;
        g.append(svg('path', { d: `M${x - 1} ${yb + 3} L${xs[i]} ${yb - 1} L${x + bar + 1} ${yb + 3}`, stroke: 'var(--bg)', 'stroke-width': 2.5, fill: 'none' }),
          svg('text', { x: Math.min(Math.max(xs[i], P.l + 14), W - P.r - 14), y: P.t - 3, 'text-anchor': 'middle', class: 'over' }, dur(took(r)).replace(/ /g, '')));
      }),
    }));
    /* Learning rate, as the tile of k1.js of v1: one dot per round with a public rate, on a log axis. Only the winning
       submission is public. Its rate is the lora_lr of its submission file (batch.json), or the default of the contract when
       the file sets none. A round with no winner has a dashed circle under the axis: its rates stay private. */
    const set = (d.train_settable || []).find(x => x.key === 'lora_lr'), own = r => r.train != null && r.train.lora_lr != null;
    const rate = r => r.train == null ? null : own(r) ? r.train.lora_lr : set ? set.default : null, rated = last(r => rate(r) != null), lg = Math.log10;
    tile('c-lr', rated && `${rated.round_id} ${sci(rate(rated))}`, round({
      include: set ? [lg(set.low), lg(set.high)] : [], least: 1, fmt: v => sci(10 ** v), empty: 'No learning rate published yet.',
      hint: (rated ? `Newest public rate: ${sci(rate(rated))}, round ${rated.round_id}. ` : '') + (set ? `Contract of today: default ${sci(set.default)}; a miner can set ${sci(set.low)} to ${sci(set.high)}.` : ''),
      series: [{ plain: 1, v: i => rate(rs[i]) == null ? null : lg(rate(rs[i])) }, { plain: 1, v: () => set ? lg(set.default) : null }],
      marks: (g, at) => rs.forEach((r, i) => {
        if (rate(r) != null) g.append(svg('circle', { cx: at.xs[i], cy: at.y(lg(rate(r))), r: 3.2, fill: tone(r), stroke: tone(r) }));
        else if (r.outcome !== 'applied') ring(g, at.xs[i], at, tone(r));
      }),
      tip: i => tip(rs[i], rate(rs[i]) != null ? [['lora_lr', sci(rate(rs[i]))], ['source', own(rs[i]) ? 'set by the miner in the submission file' : 'default of the contract of today']]
        : [['lora_lr', rs[i].outcome === 'applied' ? 'not read yet' : 'private: the round has no winner']]),
    }));
    cost();
  }

  /* Registration cost, as regcost.js of the v1 site: the burn in TAO to register one UID on subnet 120, for the last 7 days. The
     page reads the feed reg-cost-recent.json of the site, as v1 did. A point is one hour: its start, the value at its end, its
     highest and its lowest value; an hour with no change has no point. The line is a step line with a faint band from the
     lowest to the highest value of each hour, and a dot at the value of now. The hover follows the pointer: a guide line and
     the dot move to the hour. A feed that does not answer, or that was written more than 2 hours ago, gives the empty tile. */
  const COST_FIELDS = ['hour_start_unix', 'close_tao', 'high_tao', 'low_tao'], COST_SPAN = 7 * 86400, COST_OLD = 7200;
  const dated = (t, o) => new Date(t * 1000).toLocaleString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC', ...o });
  let COST = null;
  function cost() {
    if (!document.getElementById('c-cost')) return;
    const s = COST && Date.now() / 1000 - COST.generated_at <= COST_OLD ? COST.sn120 : null;
    if (!s) { tile('c-cost', null, { n: 0, series: [], empty: 'No registration cost: the feed did not answer, or it is older than 2 hours.' }); return; }
    const [T, C, HI, LO] = COST_FIELDS.map(f => s.point_fields.indexOf(f)), now = s.current.as_of || COST.generated_at, from = now - COST_SPAN;
    const before = s.points.filter(p => p[T] < from).pop(), pts = s.points.filter(p => p[T] >= from && p[T] <= now).map(p => ({ t: p[T], v: p[C], hi: p[HI], lo: p[LO] }));
    if (before) pts.unshift({ t: from, v: before[C], hi: before[C], lo: before[C], held: before[T] });
    pts.push({ t: now, v: s.current.tao, hi: s.current.tao, lo: s.current.tao, now: 1 });
    // a log axis when the highest value is more than 150 times the lowest, as v1
    const top = Math.max(...pts.map(p => p.hi)), low = Math.min(...pts.map(p => p.lo)), log = low > 0 && top / low > 150, f = log ? Math.log10 : v => v;
    const tao = v => +v.toFixed(3) + ' TAO', when = t => dated(t, { year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) + ' UTC';
    tile('c-cost', s.current.tao.toFixed(3) + ' τ', {
      n: pts.length, x: i => (pts[i].t - from) / COST_SPAN, include: log ? [] : [0], least: log ? 1 : 0, fmt: log ? v => +(10 ** v).toPrecision(2) : v => +v.toFixed(2), label: i => dated(pts[i].t),
      hint: `Now ${s.current.tao} TAO` + (s.current.block ? ` at block ${s.current.block}` : '') + (s.min_burn_tao != null ? `. Lowest possible cost: ${s.min_burn_tao} TAO` : '') + `. Feed written ${when(COST.generated_at)}.`,
      series: [{ plain: 1, v: i => f(pts[i].v) }, { plain: 1, v: i => f(pts[i].hi) }, { plain: 1, v: i => f(pts[i].lo) }],
      tip: i => pts[i].now ? tipOf(tao(pts[i].v), [['now', when(pts[i].t)], ...(s.current.block ? [['block', s.current.block]] : [])])
        : pts[i].held ? tipOf(tao(pts[i].v), [['no change since', when(pts[i].held)]])
          : tipOf(tao(pts[i].v), [['hour from', when(pts[i].t)], ['hour range', `${+pts[i].lo.toFixed(3)}–${+pts[i].hi.toFixed(3)} TAO`]]),
      marks: (g, { xs, y, P, W, H }) => {
        const X = t => (P.l + (W - P.l - P.r) * (t - from) / COST_SPAN).toFixed(1), Y = v => y(f(v)).toFixed(1), end = pts.length - 1;
        const line = pts.map((p, i) => i ? `H${X(p.t)}V${Y(p.v)}` : `M${X(p.t)} ${Y(p.v)}`).join('');
        const band = pts.filter(p => !p.now && !p.held).map(p => `M${X(p.t)} ${Y(p.hi)}H${X(Math.min(now, p.t + 3600))}V${Y(p.lo)}H${X(p.t)}Z`).join('');
        const guide = svg('line', { y1: P.t, y2: H - P.b, class: 'guide', visibility: 'hidden' }), dot = svg('circle', { r: 3, fill: 'var(--ink)' });
        const move = i => {
          const on = i >= 0, at = on ? i : end;
          dot.setAttribute('cx', xs[at]); dot.setAttribute('cy', Y(pts[at].v));
          guide.setAttribute('x1', xs[at]); guide.setAttribute('x2', xs[at]); guide.setAttribute('visibility', on ? 'visible' : 'hidden');
        };
        g.append(svg('path', { d: band, fill: 'var(--ink)', 'fill-opacity': .18 }), svg('path', { d: line, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.5, 'stroke-linejoin': 'round' }), guide, dot);
        move(-1);
        g.addEventListener('mouseover', e => move([...g.querySelectorAll('rect[fill=transparent]')].indexOf(e.target)));
        g.addEventListener('mouseleave', () => move(-1));
      },
    });
  }
  async function loadCost() {
    let feed = null;
    try {
      const res = await fetch('reg-cost-recent.json?t=' + Date.now(), { cache: 'no-store' });
      feed = res.ok ? await res.json() : null;
    } catch (e) { feed = null; }
    const s = feed && feed.sn120;
    COST = s && typeof feed.generated_at === 'number' && Array.isArray(s.points) && s.current && typeof s.current.tao === 'number'
      && Array.isArray(s.point_fields) && COST_FIELDS.every(f => s.point_fields.includes(f)) ? feed : null;
    cost();
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
    ROUND = Object.fromEntries(D.rounds.map(x => [x.round_id, x]));
    const d = D, r = d.round, q = d.queue.rows || d.queue.waiting || [];
    const ready = (d.queue.summary && d.queue.summary.valid != null)
      ? d.queue.summary.valid
      : q.filter(x => !(x.flags || []).includes('leased') && (x.file_verdict === 'valid' || (x.file_verdict == null && x.file_valid))).length;
    const live = {
      id: r.id || 'none open', state: d.phase, seats: r.seats == null ? 'no round' : `${r.seats} / ${r.cap}`, tasks: r.tasks,
      head: clip(d.head, 8, 0), bar: 'z > ' + d.z_bar, cap: d.max_output_tokens.toLocaleString('en-US'), next: ready + ' ready',
    };
    for (const el of document.querySelectorAll('#round [data-f]')) el.textContent = live[el.dataset.f];
    const open = document.querySelector('#round .reading');
    if (open) open.innerHTML = 'env ' + esc(d.env) + (r.opened_at ? ' · opened ' + ago(r.opened_at) : '');
    // The gain of a king is the gain of its winning entry. A data file that has no gain on the king has it on that entry in the history.
    const won = Object.fromEntries(d.history.filter(x => x.status === 'winner').map(x => [x.round_id + ' ' + x.uid, x]));
    const kings = d.kings.map(k => k.delta != null || k.imported || !won[k.round_id + ' ' + k.uid] ? k : { ...k, delta: won[k.round_id + ' ' + k.uid].delta, se: won[k.round_id + ' ' + k.uid].se });
    fill('kings', kings, `${d.kings.length} of ${d.kings_window} places`);
    const qRound = d.queue.round || q.filter(x => (x.flags || []).includes('leased'));
    const qWait = d.queue.waiting || q.filter(x => !(x.flags || []).includes('leased'));
    const summary = d.queue.summary || {
      in_round: qRound.length, waiting: d.queue.total ?? qWait.length,
      valid: qWait.filter(x => x.file_verdict === 'valid' || (x.file_verdict == null && x.file_valid)).length,
      not_valid: qWait.filter(x => x.file_verdict === 'not_valid').length,
      not_judged: qWait.filter(x => x.file_verdict === 'not_judged' || x.file_verdict === 'no_file' || (x.file_verdict == null && !x.file_valid)).length,
      front: qWait.filter(x => x.zone === 'front' || (x.flags || []).includes('front')).length,
      back: qWait.filter(x => x.zone === 'back' || (x.flags || []).includes('demoted')).length,
    };
    queueTotals(summary);
    document.querySelector('#queue > .section-heading .reading').textContent =
      `${summary.in_round} in round · ${summary.waiting} waiting`;
    fill('queueRound', qRound, `${summary.in_round}`);
    fill('queueWait', qWait, `${summary.waiting}`);
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
  /* header and footer of the v1 site (network-desktop.js, top.js): the theme button, the prices, "Copy view", the old anchors */
  const root = document.documentElement, themeButton = document.getElementById('theme'), device = matchMedia('(prefers-color-scheme:light)');
  const savedTheme = () => { try { const t = localStorage.getItem('affine:theme'); return t === 'light' || t === 'dark' ? t : null; } catch { return null; } };
  function theme(next) {
    root.dataset.theme = next;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = next === 'light' ? '#ffffff' : '#000000';
    if (!themeButton) return;
    const other = next === 'light' ? 'dark' : 'light';
    themeButton.setAttribute('aria-pressed', String(next === 'light'));
    themeButton.setAttribute('aria-label', `Switch to ${other} mode`);
    themeButton.title = `Switch to ${other} mode`;
  }
  const deviceTheme = () => device.matches ? 'light' : 'dark';
  theme(savedTheme() || deviceTheme());
  themeButton?.addEventListener('click', () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    try { localStorage.setItem('affine:theme', next); } catch { /* the choice then holds for this page only */ }
    theme(next);
  });
  addEventListener('storage', e => { if (e.key === 'affine:theme') theme(savedTheme() || deviceTheme()); });
  device.addEventListener('change', () => { if (!savedTheme()) theme(deviceTheme()); });

  /* Prices of the header. The feed is incentive-live.json, the same file that the v1 header read; its prices come from
     the chain (Alpha in TAO) and from a price service (TAO in USD) and do not depend on the v1 game. A price is shown
     only when the time of that price (`tao_usd_at`, `alpha_tao_at`) is less than PRICE_MAX_AGE seconds before now.
     In all other cases the header shows no price: no dash and no message, and the price row is closed. The Alpha price
     in USD is the Alpha price in TAO times the TAO price, so it needs the two.
     A feed that gives an error status or a bad file is not asked again until the page is loaded again: the browser
     writes one console line for each such request, and one line for each page load is the most that this page causes.
     The prices in view still go out of view when they become too old.
     A request with no answer at all (no network) is asked again, but less and less often: PRICE_WAITS is the wait in
     minutes, by the count of such requests in a row. One answer brings the wait back to 1 minute. A page that is not
     in view (`document.hidden`) makes no request. */
  const PRICE_WAITS = [1, 1, 2, 4, 10];
  const PRICE_MAX_AGE = 7200;
  const priceRow = document.querySelector('.prices'), priceEl = { tao: document.getElementById('tao-usd'), alpha: document.getElementById('alpha-usd'), rate: document.getElementById('alpha-tao') };
  const usd = n => '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  let priceFeed = true, lastPrices = null, priceFails = 0;
  function showPrices(p) {
    if (!priceRow || !priceEl.tao || !priceEl.alpha || !priceEl.rate) return;
    const now = Date.now() / 1000;
    const fresh = (v, at) => typeof v === 'number' && isFinite(v) && v > 0 && typeof at === 'number' && now - at < PRICE_MAX_AGE && now - at > -300;
    const tao = p && fresh(p.tao_usd, p.tao_usd_at) ? p.tao_usd : null, alpha = tao !== null && fresh(p.alpha_tao, p.alpha_tao_at) ? p.alpha_tao : null;
    priceEl.tao.closest('div').hidden = tao === null;
    priceEl.alpha.closest('div').hidden = alpha === null;
    priceRow.hidden = tao === null;
    if (tao !== null) {
      priceEl.tao.textContent = usd(tao);
      priceEl.tao.title = `TAO in US dollars, read ${utc(p.tao_usd_at)}.`;
    }
    if (alpha !== null) {
      priceEl.alpha.textContent = usd(alpha * tao);
      priceEl.alpha.title = `Alpha of subnet 120 in US dollars: its price in TAO times the price of TAO. Read ${utc(p.alpha_tao_at)}.`;
      priceEl.rate.textContent = alpha.toFixed(5) + ' TAO';
    }
  }
  async function prices() {
    if (priceFeed && !document.hidden) {
      let res = null;
      try { res = await fetch('incentive-live.json?t=' + Date.now(), { cache: 'no-store' }); } catch { /* no network now: the wait grows */ }
      priceFails = res ? 0 : priceFails + 1;
      if (res) {
        const p = res.ok ? (await res.json().catch(() => null) || {}).incentive_prices : null;
        if (p && typeof p === 'object') lastPrices = p; else priceFeed = false;
      }
    }
    showPrices(lastPrices);
    setTimeout(prices, PRICE_WAITS[Math.min(priceFails, PRICE_WAITS.length - 1)] * 60000);
  }
  if (priceRow) prices();

  const copyView = document.getElementById('copy-view');
  copyView?.addEventListener('click', async () => {
    let done = false;
    try { await navigator.clipboard.writeText(location.href); done = true; } catch { /* no clipboard in this browser or context */ }
    copyView.textContent = done ? 'Copied' : 'Copy unavailable';
    setTimeout(() => { copyView.textContent = 'Copy view'; }, 1800);
  });

  /* Links to the v1 page used #k1-kings, #k1-queue and #k1-history. Each old id is put at the start of the section
     that now holds that table, so such a link opens the right place. */
  for (const id of ['kings', 'queue', 'history']) {
    const section = document.getElementById(id);
    if (!section || document.getElementById('k1-' + id)) continue;
    const mark = document.createElement('a');
    mark.id = 'k1-' + id;
    mark.setAttribute('aria-hidden', 'true');
    section.prepend(mark);
  }
  const oldAnchor = /^#k1-(kings|queue|history)$/.exec(location.hash);
  if (oldAnchor && document.getElementById(oldAnchor[1])) location.replace('#' + oldAnchor[1]);

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
    if (e.target.closest && e.target.closest('[data-close-detail]')) { openKing = null; if (held.kings) fill('kings', held.kings.rows, held.kings.reading, held.kings.more); return; }
    const qbtn = e.target.closest && e.target.closest('#queue-pager [data-qpage]');
    if (qbtn && held.queueWait) {
      const act = qbtn.dataset.qpage, total = held.queueWait.rows.length, pages = Math.max(1, Math.ceil(total / QUEUE_PAGE));
      if (act === 'all') queueShowAll = !queueShowAll;
      else if (act === 'prev' && !queueShowAll) queuePage = Math.max(0, queuePage - 1);
      else if (act === 'next' && !queueShowAll) queuePage = Math.min(pages - 1, queuePage + 1);
      fill('queueWait', held.queueWait.rows, held.queueWait.reading, held.queueWait.more);
      return;
    }
    const th = e.target.closest && e.target.closest('th[data-sort]');
    if (th) {
      const n = th.closest('[data-table]')?.dataset.table || th.closest('section')?.id, key = th.dataset.sort;
      if (n && T[n] && C[key]?.s && held[n]) {
        const cur = sorts[n];
        sorts[n] = { key, dir: cur && cur.key === key ? -cur.dir : (C[key].n ? -1 : 1) };
        fill(n, held[n].rows, held[n].reading, held[n].more);
      }
      return;
    }
    const row = e.target.closest && e.target.closest('#kings tbody tr[data-king]');
    if (row && !e.target.closest('a,button')) {
      openKing = openKing === row.dataset.king ? null : row.dataset.king;
      if (held.kings) fill('kings', held.kings.rows, held.kings.reading, held.kings.more);
    }
  });
  document.getElementById('filter')?.addEventListener('input', e => { filter = e.target.value; if (D) render(); });
  document.getElementById('refs')?.addEventListener('click', () => { allRefs = !allRefs; if (D) bench(D.bench); });
  /* Expand, as chart-expand.js of the v1 site. A corner button lifts its tile into a large box over the page: 90% of the width
     and 80% of the height, and the whole screen below 700 px. A holder of the same height keeps the place in the grid. The same
     button, then an X, a click outside the box, or Esc close it, and the focus goes back to the button. The chart is drawn
     again at each size by the code that draws it in the grid, so the refresh and the hover go on. */
  const CORNER = [['Expand', 'M9.5 2H14v4.5M14 2 9 7M6.5 14H2V9.5M2 14l5-5'], ['Close (Esc)', 'M3 3l10 10M13 3 3 13']];
  const corner = (tile, open) => {
    const b = tile.querySelector('.cx'), name = tile.querySelector('h2').textContent;
    b.title = CORNER[+open][0];
    b.setAttribute('aria-label', (open ? 'Close ' : 'Expand ') + name);
    b.innerHTML = `<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><path d="${CORNER[+open][1]}"/></svg>`;
  };
  let lifted = null;
  function expand(tile) {
    const was = lifted;
    if (!was && !tile) return;
    if (was) {
      lifted = null;
      was.holder.remove(); was.back.remove();
      was.tile.classList.remove('cx-open');
      for (const a of ['role', 'aria-modal', 'aria-label']) was.tile.removeAttribute(a);
      corner(was.tile, false);
    }
    if (tile && !(was && was.tile === tile)) {
      const holder = document.createElement('div'), back = document.createElement('div');
      holder.className = 'cx-holder'; holder.style.height = tile.offsetHeight + 'px';
      back.className = 'cx-back'; back.addEventListener('click', () => expand());
      tile.before(holder); document.body.append(back);
      tile.classList.add('cx-open');
      tile.setAttribute('role', 'dialog'); tile.setAttribute('aria-modal', 'true'); tile.setAttribute('aria-label', tile.querySelector('h2').textContent);
      lifted = { tile, holder, back };
      corner(tile, true);
    }
    document.documentElement.classList.toggle('cx-lock', !!lifted);
    if (D) tiles(D); else cost();
    (lifted || was).tile.querySelector('.cx').focus();
  }
  document.querySelectorAll('#charts .tile').forEach(tile => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'cx'; b.addEventListener('click', () => expand(tile));
    tile.append(b); corner(tile, false);
  });
  addEventListener('keydown', e => { if (e.key === 'Escape') expand(); });
  let resized = 0;  // a chart is drawn at the size of its box, so a new size draws again from the data in hand
  addEventListener('resize', () => { clearTimeout(resized); resized = setTimeout(() => SUPPLY ? S && supply() : D && render(), 200); });
  Object.keys(T).filter(table).forEach(shell);
  const again = SUPPLY ? loadSupply : load;
  again(window.SITE_DATA);
  if (!V1) setInterval(again, 60000);
  if (document.getElementById('c-cost')) { loadCost(); setInterval(loadCost, 900000); }
})();
