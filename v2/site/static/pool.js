(() => {
'use strict';
const $ = id => document.getElementById(id);
const text = (tag, value, cls) => { const el = document.createElement(tag); el.textContent = value ?? '—'; if (cls) el.className = cls; return el; };
const number = n => new Intl.NumberFormat('en-US').format(n || 0);
const decimal = n => Number(n).toFixed(6);
const utc = seconds => Number(seconds) > 0 ? new Date(seconds * 1000).toISOString().replace('.000Z', ' UTC') : 'unknown';
function rows(id, make) { const body = $(id).tBodies[0]; body.replaceChildren(...make); }
function empty(message) { const tr = document.createElement('tr'); const td = text('td', message, 'empty'); td.colSpan = 20; tr.append(td); return tr; }
function composition(id, doc, total) {
  rows(id, Object.entries(doc || {}).sort((a,b) => b[1]-a[1] || a[0].localeCompare(b[0])).map(([name,count]) => {
    const tr = document.createElement('tr');
    tr.append(text('td',name), text('td',number(count),'num'), text('td',(100*count/Math.max(1,total)).toFixed(1)+'%','num'));
    return tr;
  }));
}
function histogram(doc, total) {
  const chart = $('score-chart'); chart.replaceChildren();
  const bins = doc || [], max = Math.max(1, ...bins.map(bin => bin.count));
  bins.forEach(bin => {
    const group = document.createElement('div'); group.className = 'bar-slot';
    const bar = document.createElement('button'); bar.type = 'button'; bar.className = 'hist-bar';
    bar.style.height = `${bin.count ? Math.max(2, 170*bin.count/max) : 0}px`;
    bar.setAttribute('aria-label', `${decimal(bin.min)} to ${decimal(bin.max)}: ${number(bin.count)} tasks`);
    bar.title = `${decimal(bin.min)}–${decimal(bin.max)}\n${number(bin.count)} tasks (${(100*bin.count/Math.max(1,total)).toFixed(1)}%)`;
    const label = text('span', decimal(bin.min)); label.className = 'bar-label';
    group.append(bar,label); chart.append(group);
  });
  rows('score-table', bins.map(bin => {
    const tr = document.createElement('tr');
    tr.append(text('td',`${decimal(bin.min)}–${decimal(bin.max)}`),text('td',number(bin.count),'num'),text('td',(100*bin.count/Math.max(1,total)).toFixed(1)+'%','num'));
    return tr;
  }));
}
function changes(doc) {
  const recent = [...(doc || [])].reverse();
  rows('round-changes', recent.length ? recent.map(round => {
    const tr = document.createElement('tr');
    const when = text('td',`${round.round_id} · ${utc(round.opened_at)}`);
    const reasons = Object.entries(round.retire_reason_counts || {}).map(([name,n])=>`${name}: ${number(n)}`).join(', ') || 'none';
    tr.append(when,text('td',number(round.active_size),'num'),text('td',number(round.admitted),'num'),text('td',number(round.retired),'num'),text('td',reasons));
    const cell = document.createElement('td'), link = text('a','rotation.json'); link.href = round.rotation; cell.append(link); tr.append(cell);
    return tr;
  }) : [empty('No ranked rotation is published yet.')]);
}
function rankedRows(candidates, limit) {
  return candidates.slice(0,limit).map((row,index) => {
    const tr = document.createElement('tr');
    tr.append(text('td',index+1,'num'),text('td',row.id),text('td',row.state),
      text('td',row.admitted ? 'admitted' : row.retired ? `retired: ${row.retire_reason}` : '—'),
      text('td',decimal(row.p),'num'),text('td',number(row.rounds_seen),'num'),text('td',decimal(row.score),'num'),
      text('td',(row.flags || []).join(', ') || '—'),text('td',number(row.head_passes),'num'),text('td',number(row.head_rollouts),'num'),
      text('td',`${row.source_prefix} / ${row.env}`));
    return tr;
  });
}
Promise.resolve(window.POOL_DATA).then(([summary, ranking]) => {
  if (!summary || summary.schema !== 'affine-pool-summary-v1') {
    $('status').textContent = 'No ranked pool published yet';
    for (const id of ['round-changes','next-up','task-rows']) rows(id,[empty('No ranked pool is published yet.')]);
    return;
  }
  for (const dd of document.querySelectorAll('#pool-stats dd')) dd.textContent = number(summary[dd.dataset.f]);
  $('pool-round').textContent = `${summary.round_id} · opened ${utc(summary.opened_at)}`;
  $('rotation-link').href = summary.rotation;
  $('ranking-link').href = summary.ranking;
  composition('sources',summary.active?.by_source_prefix,summary.active_size);
  composition('environments',summary.active?.by_env,summary.active_size);
  histogram(summary.active?.score_histogram,summary.active_size);
  changes(summary.recent_rounds);
  if (!ranking || ranking.round_id !== summary.round_id || !Array.isArray(ranking.candidates)) {
    $('status').textContent = 'Summary available; ranking not mirrored yet';
    for (const id of ['next-up','task-rows']) rows(id,[empty('The latest ranking file is not available yet.')]);
    return;
  }
  const candidates = ranking.candidates;
  const queueIds = new Set(summary.next_up || []);
  const upcoming = candidates.filter(row => queueIds.has(row.id)).slice(0,50);
  rows('next-up', upcoming.length ? upcoming.map((row,index) => {
    const tr = document.createElement('tr');
    tr.append(text('td',index+1,'num'),text('td',row.id),text('td',row.state),text('td',decimal(row.p),'num'),text('td',decimal(row.score),'num'),text('td',(row.flags||[]).join(', ')||'—'));
    return tr;
  }) : [empty('No outside candidates.')]);
  const filter = $('task-filter');
  function search() {
    const query = filter.value.trim().toLowerCase();
    const matches = candidates.filter(row => !query || [row.id,row.state,row.source_prefix,row.env,...(row.flags||[])].join(' ').toLowerCase().includes(query));
    rows('task-rows', matches.length ? rankedRows(matches,200) : [empty('No candidate matches this search.')]);
    $('search-count').textContent = `${number(matches.length)} of ${number(candidates.length)} candidates · showing ${Math.min(200,matches.length)}`;
  }
  filter.addEventListener('input',search); search();
  $('status').textContent = `${number(candidates.length)} public candidates · ${utc(summary.opened_at)}`;
}).catch(error => { $('status').textContent = `Unable to load pool: ${error.message}`; });
})();
