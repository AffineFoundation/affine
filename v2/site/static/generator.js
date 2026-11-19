/* Public Generator dashboard and Markdown guide. Same-origin data only; no credentials or private paths.
   The parsing functions have no DOM dependency and are tested with the checked-in public fixtures. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.GeneratorPage = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const EPOCH_SECONDS = 4 * 60 * 60;
  const POOL_RELEASE = 'https://github.com/AffineFoundation/affine/releases/tag/sn120-math-prior-v1';
  const MINER_KIT = 'https://github.com/AffineFoundation/affine/tree/v2/miner';

  function object(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : {};
  }

  function list(value) { return Array.isArray(value) ? value : []; }
  function finite(value) { return typeof value === 'number' && Number.isFinite(value) ? value : null; }
  function text(value) { return typeof value === 'string' && value ? value : null; }
  function count(value) { return Number.isInteger(value) && value >= 0 ? value : null; }
  function pick(source, keys) {
    for (const key of keys) if (source[key] !== undefined && source[key] !== null) return source[key];
    return null;
  }

  /* Public files use Unix seconds; milliseconds are also accepted for forward compatibility. */
  function timestamp(value) {
    if (typeof value === 'string' && value.trim()) {
      if (/^\d+(?:\.\d+)?$/.test(value)) value = Number(value);
      else {
        const milliseconds = Date.parse(value);
        return Number.isFinite(milliseconds) ? milliseconds : null;
      }
    }
    const number = finite(value);
    if (number === null || number < 0) return null;
    return number * (Math.abs(number) < 100000000000 ? 1000 : 1);
  }

  function epochTime(epoch) {
    const match = typeof epoch === 'string' && /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})Z$/.exec(epoch);
    if (!match) return null;
    const [, year, month, day, hour, minute] = match;
    if (Number(minute) !== 0 || Number(hour) % 4 !== 0) return null;
    const milliseconds = Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour));
    if (!Number.isFinite(milliseconds)) return null;
    const date = new Date(milliseconds);
    if (date.getUTCFullYear() !== Number(year) || date.getUTCMonth() !== Number(month) - 1 || date.getUTCDate() !== Number(day)) return null;
    return milliseconds;
  }

  function utc(value) {
    const milliseconds = timestamp(value);
    if (milliseconds === null) return '—';
    return new Date(milliseconds).toISOString().replace('.000', '').replace('T', ' ').replace('Z', ' UTC');
  }

  function epochUtc(epoch) {
    const milliseconds = epochTime(epoch);
    return milliseconds === null ? null : utc(milliseconds / 1000);
  }

  function countdown(milliseconds, now) {
    const remaining = Math.max(0, Math.floor((milliseconds - now) / 1000));
    const hours = Math.floor(remaining / 3600);
    const minutes = Math.floor((remaining % 3600) / 60);
    const seconds = remaining % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }

  function parseJSONL(raw) {
    if (raw === null || raw === undefined || raw === '') return [];
    if (Array.isArray(raw)) return raw.filter(row => row && typeof row === 'object' && !Array.isArray(row));
    if (typeof raw !== 'string') throw new TypeError('JSONL must be text');
    const rows = [];
    raw.split(/\r?\n/).forEach((line, index) => {
      if (!line.trim()) return;
      let row;
      try { row = JSON.parse(line); }
      catch (error) { throw new SyntaxError(`Invalid public JSONL on line ${index + 1}`); }
      if (!row || typeof row !== 'object' || Array.isArray(row)) throw new TypeError(`Public JSONL line ${index + 1} is not an object`);
      rows.push(row);
    });
    return rows;
  }

  function submissionCount(manifest, state) {
    manifest = object(manifest);
    state = object(state);
    const direct = pick(manifest, ['submission_count', 'n_submissions']);
    if (count(direct) !== null) return direct;
    for (const field of [manifest.submissions, manifest.tasks, manifest.counts, state.submissions, state.counts]) {
      if (count(field) !== null) return field;
      const nested = object(field);
      const value = pick(nested, ['submissions', 'submission_count', 'tasks']);
      if (count(value) !== null) return value;
    }
    let found = false;
    let total = 0;
    for (const rows of [list(manifest.submissions), list(manifest.tasks), list(manifest.hotkeys)]) {
      if (!rows.length) continue;
      for (const row of rows) {
        const value = pick(object(row), ['submitted', 'task_count', 'n_tasks']);
        const ids = object(row).task_ids;
        const number = count(value) !== null ? value : list(ids).length;
        if (!number) continue;
        found = true;
        total += number;
      }
      if (found) return total;
    }
    const byHotkey = object(manifest.submissions_by_hotkey);
    for (const value of Object.values(byHotkey)) {
      if (count(value) !== null) { total += value; found = true; }
      else if (count(object(value).submitted) !== null) { total += object(value).submitted; found = true; }
    }
    if (found) return total;
    return count(pick(state, ['submissions_this_epoch', 'submission_count', 'n_submitted']));
  }

  function linkCount(state, manifest) {
    for (const source of [object(manifest), object(state), object(object(state).counts)]) {
      const value = pick(source, ['link_holders', 'link_holder_count', 'n_link_holders', 'links', 'issued_links', 'active_links']);
      if (count(value) !== null) return value;
      if (Array.isArray(value)) return value.length;
      if (value && typeof value === 'object') return Object.keys(value).length;
    }
    return null;
  }

  function normalizeState(raw, manifest, now) {
    const state = object(raw);
    const epoch = text(pick(state, ['epoch', 'epoch_id', 'current_epoch', 'live_epoch'])) || text(object(state.epoch).id);
    let start = timestamp(pick(state, ['window_start', 'submission_start', 'opens_at', 'start', 'opened_at']));
    let deadline = timestamp(pick(state, ['deadline', 'deadline_at', 'closes_at', 'submission_deadline', 'window_end']));
    const epochDeadline = epochTime(epoch);
    if (start === null && deadline === null && epochDeadline !== null) {
      deadline = epochDeadline;
      start = deadline - EPOCH_SECONDS * 1000;
    }
    const king = text(pick(state, ['king', 'king_version', 'frozen_king', 'frozen_king_version', 'head_version', 'head']))
      || text(object(state.king).version) || text(object(state.king).version_id) || text(object(state.head).version_id);
    return {
      epoch, start, deadline, king,
      submissions: submissionCount(manifest, state),
      links: linkCount(state, manifest),
      scoringEpoch: text(pick(state, ['last_scored_epoch', 'scored_epoch', 'results_epoch', 'last_results_epoch', 'latest_scored_epoch']))
        || text(object(state.last_scored).epoch) || text(object(state.results).epoch) || text(object(state.scoring).last_scored_epoch),
      manifestEpoch: text(object(manifest).epoch) || epoch,
      manifestPublished: Boolean(manifest && typeof manifest === 'object' && Object.keys(manifest).length),
      status: text(pick(state, ['status', 'state', 'phase'])),
      now: finite(now) === null ? Date.now() : now,
    };
  }

  function normalizeResults(raw) {
    const document = object(raw);
    let rows = pick(document, ['hotkeys', 'by_hotkey', 'miners', 'submissions', 'per_hotkey', 'results']);
    if (Array.isArray(rows)) rows = rows.map(object);
    else {
      const keyed = object(rows);
      rows = Object.entries(keyed).map(([hotkey, value]) => ({hotkey, ...object(value)}));
    }
    return rows.map(row => ({
      hotkey: text(row.hotkey) || text(row.submitter) || '—',
      uid: count(row.uid),
      submitted: count(pick(row, ['submitted', 'n_submitted', 'task_count', 'tasks'])) ?? (Array.isArray(row.task_ids) ? row.task_ids.length : null),
      eligible: count(pick(row, ['eligible', 'n_eligible', 'eligible_count'])),
      scored: count(pick(row, ['scored', 'n_scored', 'scored_count'])),
      accepted: count(pick(row, ['accepted', 'n_accepted', 'accepted_count'])),
    })).sort((a, b) => (b.accepted ?? -1) - (a.accepted ?? -1) || a.hotkey.localeCompare(b.hotkey));
  }

  function normalizeAccepted(raw) {
    return parseJSONL(raw).map(row => ({
      id: text(pick(row, ['task_id', 'id'])) || '—',
      hotkey: text(pick(row, ['hotkey', 'submitter', 'submitter_hotkey'])) || '—',
      uid: count(row.uid),
      uploadedAt: timestamp(pick(row, ['uploaded_at', 'upload_time', 'uploaded', 'timestamp'])),
      passes: count(pick(row, ['s', 'passes', 'k'])),
      rollouts: count(pick(row, ['n', 'rollouts', 'samples'])),
      q: finite(row.q),
    })).sort((a, b) => (b.uploadedAt ?? -1) - (a.uploadedAt ?? -1) || a.id.localeCompare(b.id));
  }

  function normalizePayout(raw) {
    const document = object(raw);
    let rows = pick(document, ['weights', 'payout', 'vector', 'uids', 'payouts']);
    if (Array.isArray(rows)) rows = rows.map(object);
    else {
      const keyed = object(rows);
      rows = Object.entries(keyed).map(([uid, value]) => typeof value === 'number' ? {uid: Number(uid), weight: value} : {uid: Number(uid), ...object(value)});
    }
    rows = rows.map(row => ({
      uid: count(typeof row.uid === 'string' && /^\d+$/.test(row.uid) ? Number(row.uid) : row.uid),
      hotkey: text(row.hotkey),
      weight: finite(pick(row, ['weight', 'share', 'w'])),
      tasks: count(pick(row, ['tasks', 'accepted', 'accepted_tasks', 'n_accepted'])),
    })).filter(row => row.uid !== null && row.weight !== null && row.weight >= 0);
    if (!rows.length) {
      // The core payout document has claims, not a separate weight vector. Aggregate that
      // public record for display; the setter still owns registration and freshness checks.
      const claims = list(document.claims).map(object);
      const grouped = new Map();
      for (const claim of claims) {
        if (count(claim.uid) === null) continue;
        const row = grouped.get(claim.uid) || {uid: claim.uid, hotkey: text(claim.hotkey), weight: 0, tasks: 0};
        row.weight += 1 / claims.length;
        row.tasks++;
        grouped.set(claim.uid, row);
      }
      rows = [...grouped.values()];
    }
    const claims = list(document.claims).map(object);
    rows.forEach(row => {
      if (row.tasks === null) row.tasks = claims.filter(claim => claim.uid === row.uid).length;
      if (!row.hotkey) {
        const hotkeys = object(document.hotkeys);
        row.hotkey = text(hotkeys[row.uid]) || text(object(hotkeys[row.uid]).hotkey) || text(claims.find(claim => claim.uid === row.uid)?.hotkey);
      }
    });
    return {
      updatedAt: timestamp(document.updated_at),
      rows: rows.sort((a, b) => b.weight - a.weight || a.uid - b.uid),
    };
  }

  function cell(row, value, className) {
    const element = row.insertCell();
    if (className) element.className = className;
    if (value && typeof value === 'object' && value.nodeType) element.appendChild(value);
    else element.textContent = value ?? '—';
    return element;
  }

  function empty(table, message, columns) {
    const row = table.tBodies[0].insertRow();
    cell(row, message, 'empty').colSpan = columns;
  }

  function shortKey(value) {
    if (!value || value === '—') return '—';
    return value.length > 18 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
  }

  function copyKey(value, doc) {
    const button = doc.createElement('button');
    button.type = 'button';
    button.className = 'key';
    button.textContent = shortKey(value);
    button.title = value;
    button.setAttribute('aria-label', `Copy hotkey ${value}`);
    button.addEventListener('click', async () => {
      try {
        if (!rootNavigator().clipboard) return;
        await rootNavigator().clipboard.writeText(value);
        button.classList.add('copied');
        setTimeout(() => button.classList.remove('copied'), 1200);
      } catch (_) { /* A clipboard permission is optional; the full key remains in the title. */ }
    });
    return button;
  }

  function rootNavigator() { return typeof navigator === 'undefined' ? {} : navigator; }
  function identity(row, doc) {
    const span = doc.createElement('span');
    if (row.hotkey && row.hotkey !== '—') span.appendChild(copyKey(row.hotkey, doc));
    if (row.uid !== null && row.uid !== undefined) span.appendChild(doc.createTextNode(`${row.hotkey && row.hotkey !== '—' ? ' · ' : ''}UID ${row.uid}`));
    return span;
  }

  function renderState(view, doc) {
    for (const key of ['epoch', 'king', 'submissions', 'links']) {
      const element = doc.querySelector(`[data-f="${key}"]`);
      let value = view[key];
      if (key === 'submissions' && value === null) value = 'Not published';
      if (key === 'links' && value === null) value = 'Not published';
      if (key === 'king' && value === null) value = 'Freezes at deadline';
      element.textContent = value ?? '—';
      if (key === 'king' && view.king) element.title = view.king;
    }
    doc.getElementById('window').textContent = view.start !== null && view.deadline !== null
      ? `${utc(view.start)} → ${utc(view.deadline)} (deadline excluded)`
      : 'Submission window is not published yet';
    doc.getElementById('deadline').textContent = view.deadline !== null ? utc(view.deadline) : 'Deadline is not published yet';
    const note = doc.getElementById('epoch-note');
    note.textContent = view.manifestPublished
      ? `Manifest published for ${view.manifestEpoch || 'this epoch'}. The submission count is from the deadline snapshot. Only public data is shown.`
      : 'The deadline manifest has not been published. Counts from public link issuance are shown when available; no private submission data is read.';
    const tick = () => {
      const timer = doc.querySelector('[data-f="countdown"]');
      if (view.deadline === null) timer.textContent = 'Awaiting epoch state';
      else if (Date.now() >= view.deadline) timer.textContent = 'Closed · scoring';
      else timer.textContent = countdown(view.deadline, Date.now());
    };
    tick();
    const interval = setInterval(tick, 1000);
    if (doc.defaultView) doc.defaultView.addEventListener('pagehide', () => clearInterval(interval), {once: true});
    return interval;
  }

  function renderResults(rows, epoch, doc) {
    const body = doc.getElementById('hotkey-results').tBodies[0];
    body.replaceChildren();
    doc.getElementById('results-epoch').textContent = epoch || 'not scored yet';
    if (!rows.length) empty(doc.getElementById('hotkey-results'), epoch ? 'No scored hotkeys are published for this epoch.' : 'The first epoch has not been scored yet. Results will appear after publication.', 5);
    for (const row of rows) {
      const tr = body.insertRow();
      cell(tr, identity(row, doc));
      for (const key of ['submitted', 'eligible', 'scored', 'accepted']) cell(tr, row[key] ?? '—', 'num');
    }
    doc.getElementById('results-note').textContent = epoch
      ? 'Submitted: valid task IDs at the deadline snapshot. Eligible: IDs remaining after registration, activation, deduplication and pool checks. Scored: IDs with completed rollouts. Accepted: IDs admitted to the permanent task ledger.'
      : 'The first epoch has not been scored yet. Results will appear after scoring and publication.';
  }

  function renderAccepted(rows, epoch, doc) {
    const body = doc.getElementById('accepted-tasks').tBodies[0];
    body.replaceChildren();
    doc.getElementById('accepted-epoch').textContent = epoch || 'awaiting first scored epoch';
    if (!rows.length) empty(doc.getElementById('accepted-tasks'), epoch ? 'No accepted tasks in this scored epoch.' : 'No accepted tasks yet. The first epoch has not been scored.', 5);
    for (const row of rows) {
      const tr = body.insertRow();
      cell(tr, row.id).title = row.id;
      cell(tr, identity(row, doc));
      cell(tr, utc(row.uploadedAt));
      cell(tr, row.passes !== null && row.rollouts !== null ? `${row.passes} / ${row.rollouts}` : '—', 'num');
      cell(tr, row.q === null ? '—' : row.q.toFixed(4), 'num');
    }
  }

  function renderPayout(payout, doc) {
    const body = doc.getElementById('payout-rows').tBodies[0];
    body.replaceChildren();
    doc.getElementById('payout-updated').textContent = payout.updatedAt === null ? 'not published yet' : `updated ${utc(payout.updatedAt)}`;
    if (!payout.rows.length) empty(doc.getElementById('payout-rows'), 'No payout vector yet. Accepted tasks will appear after the first scored epoch.', 4);
    for (const row of payout.rows) {
      const tr = body.insertRow();
      cell(tr, row.uid, 'num');
      if (row.hotkey) cell(tr, copyKey(row.hotkey, doc)).title = row.hotkey;
      else cell(tr, '—');
      cell(tr, row.tasks, 'num');
      cell(tr, `${(row.weight * 100).toFixed(3)}%`, 'num');
    }
  }

  function publicRoot(doc) {
    const path = new URL(doc.baseURI).pathname;
    return path.endsWith('/generator/') ? './' : '/generator/';
  }

  async function fetchPublic(path, root, fetcher, asText) {
    const url = new URL(root + path, 'https://generator.invalid');
    if (url.origin !== 'https://generator.invalid') throw new TypeError('Only same-origin public paths are allowed');
    const response = await fetcher(url.pathname + '?t=' + Date.now(), {cache: 'no-store'});
    if (!response.ok) {
      if (response.status === 404) return null;
      throw new Error(`Public data unavailable (${response.status})`);
    }
    return asText ? response.text() : response.json();
  }

  async function load(fetcher, doc, root) {
    root = root || publicRoot(doc);
    let view = normalizeState(null, null);
    let results = null;
    let accepted = null;
    let payout = null;
    let state = null;
    let scoredEpoch = null;
    try {
      state = await fetchPublic('state.json', root, fetcher, false);
      view = normalizeState(state, null);
      const epochPattern = /^\d{8}T\d{4}Z$/;
      const manifest = view.epoch && epochPattern.test(view.epoch)
        ? await fetchPublic(`epochs/${view.epoch}/manifest.json`, root, fetcher, false).catch(() => null) : null;
      Object.assign(view, normalizeState(state, manifest));
      scoredEpoch = view.scoringEpoch;
      if (scoredEpoch && epochPattern.test(scoredEpoch)) {
        [results, accepted] = await Promise.all([
          fetchPublic(`epochs/${scoredEpoch}/results.json`, root, fetcher, false).catch(() => null),
          fetchPublic(`epochs/${scoredEpoch}/accepted.jsonl`, root, fetcher, true).catch(() => null),
        ]);
      } else scoredEpoch = null;
      payout = await fetchPublic('payout.json', root, fetcher, false).catch(() => null);
    } catch (_) {
      // A first epoch or a transient public read failure keeps the complete empty shell.
      view = normalizeState(state, null);
      results = accepted = payout = null;
    }
    renderState(view, doc);
    renderResults(results ? normalizeResults(results) : [], scoredEpoch, doc);
    renderAccepted(normalizeAccepted(accepted), scoredEpoch, doc);
    renderPayout(normalizePayout(payout), doc);
    const status = doc.getElementById('status');
    status.textContent = state ? 'Public data loaded' : 'Awaiting first public epoch';
    status.dataset.state = state ? 'available' : 'unavailable';
    return {state: view, results, accepted: normalizeAccepted(accepted), payout: normalizePayout(payout)};
  }

  /* Minimal, safe Markdown: inline code is protected, raw HTML is escaped, and only http(s) links are emitted.
     It intentionally supports headings, fences, lists, tables, quotes and paragraphs, not arbitrary HTML. */
  function inlineMarkdown(value, doc) {
    const fragment = doc.createDocumentFragment();
    const codes = [];
    const source = String(value).replace(/`([^`\n]+)`/g, (_, code) => `\u0000${codes.push(code) - 1}\u0000`);
    const token = /\[([^\]]+)\]\((https?:\/\/[^\s)"']+)\)|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*|\u0000(\d+)\u0000/g;
    let cursor = 0;
    let match;
    while ((match = token.exec(source)) !== null) {
      if (match.index > cursor) fragment.appendChild(doc.createTextNode(source.slice(cursor, match.index)));
      let element;
      if (match[1] !== undefined) {
        element = doc.createElement('a');
        const url = new URL(match[2], doc.baseURI);
        if (url.protocol !== 'http:' && url.protocol !== 'https:') {
          fragment.appendChild(doc.createTextNode(match[0]));
          cursor = match.index + match[0].length;
          continue;
        }
        element.href = url.href;
        element.rel = 'noopener noreferrer';
        element.textContent = match[1];
      } else if (match[3] !== undefined || match[4] !== undefined) {
        element = doc.createElement(match[3] !== undefined ? 'strong' : 'em');
        element.textContent = match[3] ?? match[4];
      } else {
        element = doc.createElement('code');
        element.textContent = codes[Number(match[5])];
      }
      fragment.appendChild(element);
      cursor = match.index + match[0].length;
    }
    if (cursor < source.length) fragment.appendChild(doc.createTextNode(source.slice(cursor)));
    return fragment;
  }

  function renderMarkdown(markdown, container, doc) {
    const lines = String(markdown).replace(/\r/g, '').split('\n');
    const fragment = doc.createDocumentFragment();
    let paragraph = [];
    let fence = null;
    let items = null;
    function flushParagraph() {
      if (!paragraph.length) return;
      const p = doc.createElement('p');
      p.appendChild(inlineMarkdown(paragraph.join(' '), doc));
      fragment.appendChild(p);
      paragraph = [];
    }
    function flushItems() {
      if (!items) return;
      const ul = doc.createElement(items.ordered ? 'ol' : 'ul');
      items.rows.forEach(value => {
        const li = doc.createElement('li');
        li.appendChild(inlineMarkdown(value, doc));
        ul.appendChild(li);
      });
      fragment.appendChild(ul);
      items = null;
    }
    for (let index = 0; index < lines.length; index++) {
      const line = lines[index];
      const opening = /^```([\w+-]*)\s*$/.exec(line);
      if (fence !== null) {
        if (opening) {
          const pre = doc.createElement('pre');
          const code = doc.createElement('code');
          code.textContent = fence.rows.join('\n');
          pre.appendChild(code);
          fragment.appendChild(pre);
          fence = null;
        } else fence.rows.push(line);
        continue;
      }
      if (opening) { flushParagraph(); flushItems(); fence = {rows: []}; continue; }
      if (!line.trim()) { flushParagraph(); flushItems(); continue; }
      const heading = /^(#{1,3})\s+(.+)$/.exec(line);
      if (heading) { flushParagraph(); flushItems(); const h = doc.createElement(`h${heading[1].length}`); h.appendChild(inlineMarkdown(heading[2], doc)); fragment.appendChild(h); continue; }
      const item = /^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/.exec(line);
      if (item) {
        flushParagraph();
        const ordered = Boolean(item[2]);
        if (!items || items.ordered !== ordered) { flushItems(); items = {ordered, rows: []}; }
        items.rows.push(item[3]);
        continue;
      }
      const quote = /^>\s?(.*)$/.exec(line);
      if (quote) { flushParagraph(); flushItems(); const block = doc.createElement('blockquote'); block.appendChild(inlineMarkdown(quote[1], doc)); fragment.appendChild(block); continue; }
      if (line.includes('|') && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[index + 1] || '') && lines[index + 1].includes('-')) {
        flushParagraph(); flushItems();
        const cells = value => value.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(value => value.trim());
        const headers = cells(line);
        const table = doc.createElement('table');
        const head = table.createTHead().insertRow();
        headers.forEach(value => { const th = doc.createElement('th'); th.appendChild(inlineMarkdown(value, doc)); head.appendChild(th); });
        const body = table.createTBody();
        index += 2;
        while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
          const row = body.insertRow();
          cells(lines[index]).forEach(value => { const td = row.insertCell(); td.appendChild(inlineMarkdown(value, doc)); });
          index++;
        }
        index--;
        fragment.appendChild(table);
        continue;
      }
      flushItems();
      paragraph.push(line.trim());
    }
    if (fence) { const pre = doc.createElement('pre'); pre.textContent = fence.rows.join('\n'); fragment.appendChild(pre); }
    flushParagraph(); flushItems();
    container.replaceChildren(fragment);
    container.querySelectorAll('h1').forEach(heading => { heading.id = heading.id || 'guide-content-title'; });
  }

  async function loadGuide(fetcher, doc, root) {
    const container = doc.getElementById('guide-content');
    const placeholder = container.cloneNode(true);
    try {
      const markdown = await fetchPublic('generator-guide.md', root || publicRoot(doc), fetcher, true);
      if (markdown !== null && markdown.trim()) renderMarkdown(markdown, container, doc);
    } catch (_) { /* Keep the launch placeholder if the guide is not yet available. */ }
    doc.getElementById('status').textContent = container.isEqualNode(placeholder) ? 'Miner guide pending publication' : 'Public miner guide';
  }

  function init(doc, fetcher) {
    if (doc.getElementById('guide-content')) return loadGuide(fetcher || fetch, doc);
    return load(fetcher || fetch, doc).catch(() => {
      const status = doc.getElementById('status');
      status.textContent = 'Public data is temporarily unavailable';
      status.dataset.state = 'unavailable';
    });
  }

  return {
    EPOCH_SECONDS, POOL_RELEASE, MINER_KIT, timestamp, epochTime, epochUtc, utc, countdown,
    parseJSONL, submissionCount, linkCount, normalizeState, normalizeResults, normalizeAccepted,
    normalizePayout, inlineMarkdown, renderMarkdown, load, loadGuide, init,
  };
});

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => window.GeneratorPage.init(document));
}
