/* Flip incentive panels (LIVE) — Affine visual language (network.css tokens). */
(function () {
  const ns = 'http://www.w3.org/2000/svg';
  const $ = (id) => document.getElementById(id);
  const fin = (v) => Number.isFinite(v);
  const fmt = (v, d = 2) => (fin(v) ? Number(v).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: 0 }) : '—');
  const pct = (v, d = 2) => (fin(v) ? (v * 100).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }) + '%' : '—');

  function el(tag, attrs, text) {
    const node = document.createElementNS(ns, tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) if (v != null && v !== false) node.setAttribute(k, String(v));
    if (text != null) node.textContent = text;
    return node;
  }
  function h(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function setLegend(id, items) {
    const L = $(id);
    if (!L) return;
    L.replaceChildren();
    for (const [label, color] of items) {
      const span = h('span');
      const i = h('i', 'swatch');
      i.style.background = color.startsWith('--') ? 'var(' + color + ')' : color;
      span.append(i, document.createTextNode(label));
      L.append(span);
    }
  }

  function aggregatePairs(shadow) {
    const pairs = shadow?.window?.pairs || [];
    return pairs.map((p) => ({
      label: 'E' + p.epoch_from + '→' + p.epoch_to,
      epoch_from: p.epoch_from,
      epoch_to: p.epoch_to,
      fail_to_pass: Number(p.n_fail_to_pass) || 0,
      pass_to_fail: Number(p.n_pass_to_fail) || 0,
      events: Number(p.n_events) || 0,
      net: (Number(p.n_fail_to_pass) || 0) - (Number(p.n_pass_to_fail) || 0),
    }));
  }

  function aggregateCreditByEpoch(shadow) {
    const map = new Map();
    for (const m of shadow?.miners || []) {
      const be = m.by_epoch || {};
      for (const [ep, row] of Object.entries(be)) {
        const e = Number(ep);
        if (!fin(e)) continue;
        const cur = map.get(e) || { epoch: e, credit: 0, fail_to_pass: 0, pass_to_fail: 0, events: 0 };
        cur.credit += Number(row.credit) || 0;
        cur.fail_to_pass += Number(row.fail_to_pass) || 0;
        cur.pass_to_fail += Number(row.pass_to_fail) || 0;
        cur.events += Number(row.events) || 0;
        map.set(e, cur);
      }
    }
    return [...map.values()].sort((a, b) => a.epoch - b.epoch);
  }

  function drawCredit(shadow) {
    const svg = $('flip-credit-chart');
    const empty = $('flip-credit-empty');
    const tip = $('flip-credit-tip');
    const value = $('flip-credit-value');
    const change = $('flip-credit-change');
    const context = $('flip-credit-context');
    if (!svg) return;

    const rows = aggregateCreditByEpoch(shadow);
    const pairs = aggregatePairs(shadow);
    svg.replaceChildren();
    tip.hidden = true;

    if (!rows.length) {
      empty.hidden = false;
      empty.textContent = 'No shadow flip epochs yet.';
      value.textContent = '—';
      return;
    }
    empty.hidden = true;

    const last = rows[rows.length - 1];
    const prev = rows.length > 1 ? rows[rows.length - 2] : null;
    value.textContent = fmt(last.credit, 2);
    if (change) {
      if (prev && fin(prev.credit)) {
        const d = last.credit - prev.credit;
        change.textContent = (d >= 0 ? '+' : '') + fmt(d, 2) + ' vs E' + prev.epoch;
        change.dataset.direction = d > 0 ? 'up' : d < 0 ? 'down' : '';
      } else {
        change.textContent = '';
        delete change.dataset.direction;
      }
    }
    if (context) {
      const w = shadow.window || {};
      const gen = shadow.generated_at_cr || '';
      context.textContent =
        'Network credit by published epoch · ' +
        (w.n_pairs || pairs.length) + ' pairs · ' +
        (w.n_miners || (shadow.miners || []).length) + ' miners · ' +
        (shadow.live_weights ? 'LIVE WEIGHTS' : 'live_weights: false') +
        (gen ? ' · ' + gen : '');
    }

    setLegend('flip-credit-legend', [
      ['Fail→pass events', '--performance'],
      ['Pass→fail events', '--rejected'],
      ['Net credit', '--incentive'],
    ]);

    const W = 640, H = 260;
    const L = 48, R = W - 16, T = 16, B = H - 36;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

    const n = rows.length;
    const x = (i) => L + ((R - L) * (n === 1 ? 0.5 : i / (n - 1)));
    const maxBar = Math.max(1, ...rows.map((r) => Math.max(r.fail_to_pass, r.pass_to_fail)));
    const credits = rows.map((r) => r.credit);
    const cMin = Math.min(0, ...credits);
    const cMax = Math.max(0.01, ...credits);
    const yBar = (v) => B - ((B - T) * 0.55 * (v / maxBar));
    const yLine = (v) => B - ((B - T) * ((v - cMin) / (cMax - cMin || 1)));

    // guides
    for (const t of [0, 0.5, 1]) {
      const yy = T + (B - T) * t;
      svg.append(el('line', { x1: L, x2: R, y1: yy, y2: yy, class: 'chart-guide', stroke: 'var(--line)', 'stroke-dasharray': t === 1 ? '0' : '3 4' }));
    }
    svg.append(el('line', { x1: L, x2: R, y1: B, y2: B, stroke: 'var(--axis)' }));

    const bw = Math.max(4, Math.min(18, ((R - L) / Math.max(n, 1)) * 0.28));
    rows.forEach((r, i) => {
      const cx = x(i);
      const hFp = Math.max(0, B - yBar(r.fail_to_pass));
      const hPf = Math.max(0, B - yBar(r.pass_to_fail));
      svg.append(el('rect', {
        x: cx - bw - 1, y: yBar(r.fail_to_pass), width: bw, height: hFp,
        fill: 'var(--performance)', opacity: 0.9, class: 'flip-bar',
      }));
      svg.append(el('rect', {
        x: cx + 1, y: yBar(r.pass_to_fail), width: bw, height: hPf,
        fill: 'var(--rejected)', opacity: 0.9, class: 'flip-bar',
      }));
    });

    let d = '';
    rows.forEach((r, i) => {
      const cx = x(i), cy = yLine(r.credit);
      d += (i ? ' L ' : 'M ') + cx + ' ' + cy;
      svg.append(el('circle', {
        cx, cy, r: i === n - 1 ? 3.75 : 2.5,
        class: 'plot-point' + (i === n - 1 ? ' is-current' : ''),
        fill: i === n - 1 ? 'var(--incentive)' : 'var(--bg)',
        stroke: 'var(--incentive)',
      }));
    });
    if (d) svg.append(el('path', { d, class: 'plot-line', stroke: 'var(--incentive)', fill: 'none', 'stroke-width': 2 }));

    rows.forEach((r, i) => {
      if (i === 0 || i === n - 1 || n <= 10 || i % Math.ceil(n / 8) === 0) {
        svg.append(el('text', { x: x(i), y: B + 18, 'text-anchor': 'middle' }, 'E' + r.epoch));
      }
    });
    svg.append(el('text', { x: L - 8, y: T + 4, 'text-anchor': 'end' }, fmt(maxBar, 0)));
    svg.append(el('text', { x: L - 8, y: B, 'text-anchor': 'end' }, '0'));

    rows.forEach((r, i) => {
      const hit = el('rect', {
        x: x(i) - ((R - L) / Math.max(n, 1)) / 2,
        y: T, width: (R - L) / Math.max(n, 1), height: B - T,
        fill: 'transparent', class: 'flip-hit',
      });
      const show = () => {
        tip.replaceChildren();
        tip.append(h('span', '', 'Epoch ' + r.epoch), h('b', '', 'credit ' + fmt(r.credit, 3)));
        tip.append(h('small', '', 'fail→pass ' + fmt(r.fail_to_pass, 0) + ' · pass→fail ' + fmt(r.pass_to_fail, 0) + ' · events ' + fmt(r.events, 0)));
        tip.hidden = false;
      };
      hit.addEventListener('pointerenter', show);
      hit.addEventListener('focus', show);
      hit.addEventListener('pointerleave', () => { tip.hidden = true; });
      hit.addEventListener('blur', () => { tip.hidden = true; });
      svg.append(hit);
    });
  }

  function drawShare(shadow) {
    const body = $('flip-share-body');
    const value = $('flip-share-value');
    const label = $('flip-share-label');
    const context = $('flip-share-context');
    if (!body) return;

    const miners = [...(shadow?.miners || [])].sort((a, b) => (b.share_now || 0) - (a.share_now || 0));
    const top = miners.filter((m) => (m.share_now || 0) > 0 || (m.n_events || 0) > 0).slice(0, 64);
    const show = top.length ? top : miners.slice(0, 32);

    value.textContent = String(show.length);
    if (label) label.textContent = 'listed';
    if (context) {
      const f = shadow?.formula || {};
      context.textContent =
        (shadow?.version || 'shadow') +
        ' · share_mode ' + (f.share_mode || '—') +
        ' · window pairs ' + (shadow?.window?.n_pairs ?? '—') +
        ' · live_weights ' + String(!!shadow?.live_weights);
    }

    setLegend('flip-share-legend', [
      ['Sorted by share_now', '--incentive'],
      ['Eligible = n_eligible tasks while failing', '--muted'],
    ]);

    body.replaceChildren();
    if (!show.length) {
      const tr = h('tr');
      const td = h('td', 'flip-loading', 'No miners in scores-shadow.json');
      td.colSpan = 6;
      tr.append(td);
      body.append(tr);
      return;
    }
    for (const m of show) {
      const tr = h('tr');
      const uid = h('td');
      uid.append(document.createTextNode(String(m.uid)));
      if (m.hotkey) {
        const hk = h('span', 'hotkey', String(m.hotkey).slice(0, 6) + '…');
        uid.append(hk);
      }
      tr.append(
        uid,
        Object.assign(h('td', 'num'), { textContent: pct(m.share_now, 2) }),
        Object.assign(h('td', 'num'), { textContent: fmt(m.flip_score, 3) }),
        Object.assign(h('td', 'num'), { textContent: fmt(m.n_fail_to_pass, 0) }),
        Object.assign(h('td', 'num'), { textContent: fmt(m.n_pass_to_fail, 0) }),
        Object.assign(h('td', 'num'), { textContent: fmt(m.n_eligible, 0) })
      );
      body.append(tr);
    }
  }

  async function loadFlipScores() {
    // LIVE cutover: scores.json is flip formula; scores-shadow.json kept as mirror.
    const urls = ['/scores.json', '/scores-shadow.json', '/data/scores-shadow.json'];
    let lastErr;
    for (const url of urls) {
      try {
        const r = await fetch(url, { cache: 'no-store', signal: AbortSignal.timeout(15000) });
        if (!r.ok) throw new Error(url + ' ' + r.status);
        const doc = await r.json();
        // Prefer docs that carry flip fields; skip pure v9 window-credit if somehow stale.
        const miners = doc.miners || [];
        if (miners.length && miners[0] && ('flip_score' in miners[0] || doc.live_weights === true || (doc.version || '').includes('flip'))) {
          return doc;
        }
        if ((doc.version || '').includes('flip') || doc.mode === 'live' || doc.mode === 'shadow') return doc;
        lastErr = new Error(url + ' not flip scores');
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error('flip scores unavailable');
  }

  async function refresh() {
    try {
      const shadow = await loadFlipScores();
      drawCredit(shadow);
      drawShare(shadow);
      const conn = $('connection');
      if (conn && conn.textContent === 'Connecting') {
        /* leave network.js to own connection status once data loads */
      }
    } catch (e) {
      console.error('flip', e);
      const empty = $('flip-credit-empty');
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'Flip scores unavailable.';
      }
      const body = $('flip-share-body');
      if (body) {
        body.replaceChildren();
        const tr = h('tr');
        const td = h('td', 'flip-loading', 'Flip scores unavailable.');
        td.colSpan = 6;
        tr.append(td);
        body.append(tr);
      }
    }
  }

  refresh();
  setInterval(refresh, 90_000);
  window.addEventListener('affine:theme', refresh);
})();
