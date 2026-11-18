/* Affine Slots page. Theme, prices and "Copy view" follow the v2 site header; slot state is rendered here. */
(() => {
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const clip = (s, a = 6, b = 4) => { s = String(s || ''); return s.length > a + b + 1 ? s.slice(0, a) + '…' + (b ? s.slice(-b) : '') : s || '—'; };
  const mute = (s, t) => `<span class="mute"${t ? ` title="${esc(t)}"` : ''}>${esc(s)}</span>`;
  const utc = t => {
    if (t == null || t === '') return '';
    const d = typeof t === 'number' ? new Date(t < 1e12 ? t * 1000 : t) : new Date(t);
    if (Number.isNaN(d.getTime())) return String(t);
    return d.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  };
  const ago = t => {
    const d = typeof t === 'number' ? new Date(t < 1e12 ? t * 1000 : t) : new Date(t);
    if (Number.isNaN(d.getTime())) return mute('—');
    const s = Math.max(0, (Date.now() - d.getTime()) / 1000);
    const m = Math.round(s / 60), h = Math.floor(m / 60);
    const label = m < 1 ? '<1m' : m < 60 ? m + 'm' : h < 48 ? h + 'h' + (m % 60 ? ' ' + (m % 60) + 'm' : '') : Math.floor(h / 24) + 'd' + (h % 24 ? ' ' + (h % 24) + 'h' : '');
    return `<span title="${esc(utc(t))}">${esc(label)} ago</span>`;
  };
  const key = (v, a = 6, b = 4) => v ? `<button type="button" class="key" data-copy="${esc(v)}" title="${esc(v)} (click to copy)">${esc(clip(v, a, b))}</button>` : mute('—');
  const pill = (s, c, t) => `<span class="pill ${c}"${t ? ` title="${esc(t)}"` : ''}>${esc(s)}</span>`;
  const score = x => {
    if (x == null || x === '') return mute('—', 'No score yet');
    const n = Number(x);
    if (!Number.isFinite(n)) return mute('—');
    if (n >= 0 && n <= 1) return `<span title="${esc(String(n))}">${(100 * n).toFixed(2)}%</span>`;
    return `<span>${esc(n.toFixed(4))}</span>`;
  };
  const num = x => (x == null || x === '') ? mute('—') : esc(String(x));
  const TENURE = 8;
  const STATUS = {
    king: ['KING', 'king', 'This slot holds the crown and receives this mechanism\'s emission.'],
    challenging: ['CHALLENGING', 'challenging', 'This slot has finished its tenure and is being compared to the king.'],
    active: ['ACTIVE', 'active', 'This slot is in its eight-cycle tenure.'],
    banned: ['BANNED', 'banned', 'This slot was banned for cheating, such as overfitting in its batches.'],
  };

  const root = document.documentElement, themeButton = document.getElementById('theme'), device = matchMedia('(prefers-color-scheme:light)');
  const savedTheme = () => { try { const t = localStorage.getItem('affine:theme'); return t === 'light' || t === 'dark' ? t : null; } catch { return null; } };
  const deviceTheme = () => device.matches ? 'light' : 'dark';
  const theme = next => {
    root.dataset.theme = next;
    const color = document.querySelector('meta[name="theme-color"]');
    if (color) color.content = next === 'light' ? '#ffffff' : '#000000';
    if (!themeButton) return;
    const other = next === 'light' ? 'dark' : 'light';
    themeButton.setAttribute('aria-pressed', String(next === 'light'));
    themeButton.setAttribute('aria-label', `Switch to ${other} mode`);
    themeButton.title = `Switch to ${other} mode`;
  };
  theme(savedTheme() || deviceTheme());
  themeButton?.addEventListener('click', () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    try { localStorage.setItem('affine:theme', next); } catch { /* the choice then holds for this page only */ }
    theme(next);
  });
  addEventListener('storage', e => { if (e.key === 'affine:theme') theme(savedTheme() || deviceTheme()); });
  device.addEventListener('change', () => { if (!savedTheme()) theme(deviceTheme()); });

  const copyView = document.getElementById('copy-view');
  copyView?.addEventListener('click', async () => {
    let done = false;
    try { await navigator.clipboard.writeText(location.href); done = true; } catch { /* no clipboard in this browser or context */ }
    copyView.textContent = done ? 'Copied' : 'Copy unavailable';
    setTimeout(() => { copyView.textContent = 'Copy view'; }, 1800);
  });

  document.addEventListener('click', e => {
    const btn = e.target.closest && e.target.closest('button.key');
    if (!btn) return;
    const v = btn.getAttribute('data-copy');
    if (!v) return;
    navigator.clipboard.writeText(v).then(() => {
      btn.classList.add('copied');
      setTimeout(() => btn.classList.remove('copied'), 1200);
    }).catch(() => {});
  });

  const priceRow = document.querySelector('.prices');
  const priceEl = { tao: document.getElementById('tao-usd'), alpha: document.getElementById('alpha-usd'), rate: document.getElementById('alpha-tao') };
  const usd = n => '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const prices = async () => {
    if (!priceRow || !priceEl.tao) return;
    try {
      const r = await fetch('incentive-live.json?t=' + Date.now(), { cache: 'no-store' });
      if (!r.ok) return;
      const d = await r.json();
      const tao = d.tao_usd, alphaUsd = d.alpha_usd, alphaTao = d.alpha_tao;
      const taoAt = d.tao_usd_at, alphaAt = d.alpha_tao_at;
      const fresh = t => { const n = Date.parse(t) || (typeof t === 'number' ? (t < 1e12 ? t * 1000 : t) : NaN); return Number.isFinite(n) && (Date.now() - n) < 7200 * 1000; };
      if (tao != null && priceEl.tao) { priceEl.tao.textContent = usd(tao); priceEl.tao.parentElement.parentElement.hidden = !fresh(taoAt) && taoAt != null ? true : false; if (taoAt == null) priceEl.tao.parentElement.parentElement.hidden = false; }
      if ((alphaUsd != null || alphaTao != null) && priceEl.alpha) {
        if (alphaUsd != null) priceEl.alpha.textContent = usd(alphaUsd);
        if (alphaTao != null && priceEl.rate) priceEl.rate.textContent = Number(alphaTao).toFixed(6) + ' τ';
        priceEl.alpha.parentElement.parentElement.hidden = false;
      }
    } catch { /* prices stay hidden */ }
  };
  if (priceRow) prices();

  const PAGE = document.body.dataset.page || 'slots';

  function renderMd(md) {
    const lines = md.replace(/\r\n/g, '\n').split('\n');
    const inline = s => esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" rel="noopener">$1</a>');
    const endList = () => { if (inList) { html += inList === 'ol' ? '</ol>' : '</ul>'; inList = false; } };
    let html = '';
    let inList = false, inPre = false, pre = [];
    for (const raw of lines) {
      if (raw.startsWith('```')) {
        if (inPre) { html += '<pre><code>' + esc(pre.join('\n')) + '</code></pre>'; pre = []; inPre = false; }
        else { endList(); inPre = true; }
        continue;
      }
      if (inPre) { pre.push(raw); continue; }
      if (/^\s*$/.test(raw)) { endList(); continue; }
      const h = raw.match(/^(#{1,3})\s+(.*)$/);
      if (h) { endList(); html += `<h${h[1].length + 1}>` + inline(h[2]) + `</h${h[1].length + 1}>`; continue; }
      const ol = raw.match(/^\s*\d+\.\s+(.*)$/);
      if (ol) {
        if (inList !== 'ol') { endList(); html += '<ol>'; inList = 'ol'; }
        html += '<li>' + inline(ol[1]) + '</li>';
        continue;
      }
      const ul = raw.match(/^\s*[-*]\s+(.*)$/);
      if (ul) {
        if (inList !== 'ul') { endList(); html += '<ul>'; inList = 'ul'; }
        html += '<li>' + inline(ul[1]) + '</li>';
        continue;
      }
      endList();
      html += '<p>' + inline(raw) + '</p>';
    }
    endList();
    if (inPre) html += '<pre><code>' + esc(pre.join('\n')) + '</code></pre>';
    return html;
  }

  if (PAGE === 'slots-guide') {
    const body = document.getElementById('guide-body');
    const foot = document.getElementById('status');
    const loadGuide = async () => {
      try {
        const r = await fetch('slots-guide.md?t=' + Date.now(), { cache: 'no-store' });
        if (!r.ok) return;
        const md = await r.text();
        if (!md || md.length < 20) return;
        if (body) body.innerHTML = renderMd(md);
        if (foot) foot.textContent = 'Guide';
      } catch { /* keep the built-in summary */ }
    };
    loadGuide();
    return;
  }

  let last = null, lastSource = null, lastFailed = false;

  const emptyRow = (cols, text) => `<tr><td class="empty" colspan="${cols}">${esc(text)}</td></tr>`;

  const tenureCell = n => {
    const done = Math.max(0, Math.min(TENURE, Number(n) || 0));
    const pct = (100 * done / TENURE).toFixed(0);
    return `<div class="tenure"><span>${done} / ${TENURE}</span><span class="tenure-bar" title="${done} of ${TENURE} cycles"><i style="width:${pct}%"></i></span></div>`;
  };

  const render = (S, meta) => {
    const king = S && S.king, slots = (S && S.slots) || [], queue = (S && S.queue) || [], history = (S && S.history) || [];
    const set = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };
    const setText = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };

    if (!king) {
      set('king-uid', mute('—', 'No king yet'));
      set('king-hotkey', mute('—'));
      set('king-slot', mute('—'));
      set('king-score', mute('—'));
      set('king-since', mute('—'));
    } else {
      set('king-uid', king.uid != null ? esc(String(king.uid)) : mute('—'));
      set('king-hotkey', key(king.hotkey, 8, 6));
      set('king-slot', king.slot != null ? esc(String(king.slot)) : mute('—'));
      set('king-score', score(king.score));
      set('king-since', king.since ? `<span title="${esc(utc(king.since))}">${esc(utc(king.since))}</span>` : mute('—'));
    }
    set('king-cycle', S && S.cycle != null ? esc(String(S.cycle)) : mute('—'));
    setText('king-reading', king && king.uid != null ? 'UID ' + king.uid : 'No king');

    const tb = document.querySelector('#slots-scroll tbody');
    if (tb) {
      if (!slots.length) tb.innerHTML = emptyRow(9, 'No slots in the current state.');
      else tb.innerHTML = slots.map(s => {
        const st = String(s.status || 'active').toLowerCase();
        const [label, cls, tip] = STATUS[st] || [st.toUpperCase(), '', ''];
        const isKing = st === 'king';
        return `<tr class="${isKing ? 'king' : ''}">
          <td>${s.slot != null ? esc(String(s.slot)) : mute('—')}</td>
          <td>${s.uid != null ? esc(String(s.uid)) : mute('—')}</td>
          <td>${key(s.hotkey)}</td>
          <td>${tenureCell(s.cycles_done)}</td>
          <td>${score(s.window_score)}</td>
          <td>${num(s.tasks_scored)}</td>
          <td>${num(s.batches)}</td>
          <td>${pill(label, cls, tip)}</td>
          <td>${s.joined_at ? ago(s.joined_at) : mute('—')}</td>
        </tr>`;
      }).join('');
    }
    setText('slots-reading', slots.length ? slots.length + ' of 8' : '—');

    const qb = document.querySelector('#queue-scroll tbody');
    if (qb) {
      if (!queue.length) qb.innerHTML = emptyRow(4, 'No one in the queue yet.');
      else qb.innerHTML = queue.map((q, i) => `<tr>
        <td>${i + 1}</td>
        <td>${q.uid != null ? esc(String(q.uid)) : mute('—')}</td>
        <td>${key(q.hotkey)}</td>
        <td>${q.requested_at ? `<span title="${esc(utc(q.requested_at))}">${esc(utc(q.requested_at))}</span>` : mute('—')}</td>
      </tr>`).join('');
    }
    setText('queue-reading', queue.length ? String(queue.length) : 'empty');

    const hb = document.querySelector('#history-scroll tbody');
    if (hb) {
      if (!history.length) hb.innerHTML = emptyRow(6, 'No challenges yet.');
      else hb.innerHTML = history.map(h => {
        const crowned = !!h.crowned;
        return `<tr>
          <td>${h.at ? `<span title="${esc(utc(h.at))}">${esc(utc(h.at))}</span>` : mute('—')}</td>
          <td>${h.challenger_uid != null ? esc(String(h.challenger_uid)) : mute('—')}</td>
          <td>${h.king_uid != null ? esc(String(h.king_uid)) : mute('—')}</td>
          <td>${score(h.challenger_score)}</td>
          <td>${score(h.king_score)}</td>
          <td>${crowned ? pill('CROWNED', 'king', 'The challenger took the crown.') : pill('DEFENDED', 'active', 'The sitting king kept the crown.')}</td>
        </tr>`;
      }).join('');
    }
    setText('history-reading', history.length ? String(history.length) : 'none');

    const foot = document.getElementById('status');
    if (foot) {
      const written = S && S.updated_at ? utc(S.updated_at) : null;
      const seed = !!(S && (S.preview === 'seed' || meta.source === 'seed'));
      foot.dataset.state = !S || lastFailed ? 'unavailable' : 'live';
      if (!S) foot.textContent = 'No slot state yet.';
      else if (seed) foot.textContent = 'Seed occupants. Waiting for live slot state' + (written ? '. Written ' + written + '.' : '.');
      else if (lastFailed) foot.textContent = 'Refresh failed; this is the last good state' + (written ? ', written ' + written + '.' : '.');
      else foot.textContent = 'Live' + (written ? '. State written ' + written + '.' : '.');
    }
  };

  const accept = doc => {
    if (!doc || typeof doc !== 'object') return null;
    if (!('slots' in doc) && !('king' in doc)) return null;
    return doc;
  };

  const pullJson = async (url) => {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error('http ' + r.status);
    return accept(await r.json());
  };

  const urls = async () => {
    const list = ['slots-state.json?t=' + Date.now()];
    try {
      const r = await fetch('slots-config.json?t=' + Date.now(), { cache: 'no-store' });
      if (r.ok) {
        const c = await r.json();
        if (c && c.state_url) list.unshift(c.state_url + (c.state_url.includes('?') ? '&' : '?') + 't=' + Date.now());
      }
    } catch { /* same-origin copy only */ }
    if (window.SLOTS_STATE_URL) list.unshift(window.SLOTS_STATE_URL);
    return list;
  };

  const load = async (first) => {
    let doc = null, source = 'live';
    if (first && window.SLOTS_STATE) {
      try {
        const r = await window.SLOTS_STATE;
        if (r && r.ok) doc = accept(await r.clone().json());
      } catch { /* fall through */ }
    }
    if (!doc) {
      const list = await urls();
      for (const url of list) {
        try {
          doc = await pullJson(url);
          if (doc) { source = url.startsWith('slots-state') ? 'local' : 'live'; break; }
        } catch { /* try the next URL */ }
      }
    }
    if (doc) {
      if (doc.preview === 'seed') source = 'seed';
      last = doc; lastSource = source; lastFailed = false;
      render(doc, { source });
    } else if (last) {
      lastFailed = true;
      render(last, { source: lastSource || 'live' });
    } else {
      lastFailed = true;
      render(null, { source: 'none' });
    }
  };

  load(true);
  setInterval(() => load(false), 60000);
})();
