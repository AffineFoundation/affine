// Run: node v2/tests/site_charts.test.cjs [path/to/site.js]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = process.argv[2] || require('node:path').join(__dirname, '../site/static/site.js');
const source = fs.readFileSync(path, 'utf8');
const between = (a, b) => source.slice(source.indexOf(a), source.indexOf(b, source.indexOf(a)));
class Node {
  constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this._text = ''; this.innerHTML = ''; this.style = {}; this.classList = { contains: () => false }; }
  set textContent(v) { this._text = v; if (v === '') this.children = []; }
  get textContent() { return this._text; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
  append(...children) { this.children.push(...children); }
  addEventListener(type, fn) { ((this.events ||= {})[type] ||= []).push(fn); }
  dispatch(type) { for (const fn of this.events?.[type] || []) fn(); }
  insertAdjacentHTML(_, html) { this.children.push(Object.assign(new Node('p'), { innerHTML: html })); }
  querySelectorAll(selector) { return selector === 'rect[fill=transparent]' ? this.children.filter(x => x.attrs.fill === 'transparent') : []; }
}
class Tile {
  constructor(id) {
    this.id = id; this.reading = new Node('span'); this.chart = new Node('div'); this.legend = new Node('p');
    this.classList = { contains: () => false };
    this.chart.clientWidth = 300; this.chart.clientHeight = 180; this.chart.style = {};
    this.heading = new Node('h2'); this.heading.textContent = id;
  }
  querySelector(s) { return { '.reading': this.reading, '.chart': this.chart, '.legend': this.legend, h2: this.heading, '.cx': null }[s]; }
}
const tiles = Object.fromEntries(['c-z', 'c-pass', 'c-batch', 'c-time', 'c-lr'].map(id => [id, new Tile(id)]));
const document = {
  getElementById: id => id === 'charts' ? {} : tiles[id] || null,
  createElementNS: (_, tag) => new Node(tag), createElement: tag => new Node(tag),
  querySelector: selector => { const m = /^#(\S+) \.legend$/.exec(selector); return m ? tiles[m[1]].legend : null; },
};
const helpers = between('  const esc =', '  const RESULT =');
const chart = between('  const NS =', '  /* The benchmark table');
const round = between(source.includes('  const OPEN_PAD') ? '  const OPEN_PAD' : '  const ENDED', source.includes('  function cost()') ? '  function cost()' : '  const match =').replace('    cost();', '');
const code = `${helpers}\n${chart}\n${round}\nthis.draw = tiles;`;
const context = { document, console, createElement: document.createElement, createElementNS: document.createElementNS };
vm.runInNewContext(code, context);
const rounds = [
  { round_id: 'r000162', outcome: 'applied', best_z: .33, null_pass: .559, winner_pass: .6, winner_uid: 7, seats: 6, with_z: 5 },
  { round_id: 'r000163', outcome: 'applied', best_z: 21.87, null_pass: .5, winner_pass: .7, winner_uid: 829, seats: 6, with_z: 5, reverted: true },
  { round_id: 'r000165', outcome: 'no_decision', best_z: null, null_pass: 0, winner_pass: 0, winner_uid: null, seats: 0, with_z: 0 },
].map((r, i) => ({ opened_at: i * 60, at: (i + 1) * 60, tasks: 10, null_cap: 0, skipped: 0, ...r }));
context.draw({ rounds: rounds.slice().reverse(), z_bar: 1, round: { cap: 8 }, train_settable: [] });
assert.equal(tiles['c-z'].reading.textContent, 'r000162 z +0.33');
assert.equal(tiles['c-pass'].reading.textContent, 'r000163 M 50.0%');
const zSvg = tiles['c-z'].chart.children.find(n => n.tag === 'svg');
assert(!zSvg.children.some(n => n.tag === 'text' && /20|21|22/.test(n.textContent)), 'reverted z must not set the axis');
const zHit = zSvg.children.filter(n => n.attrs.fill === 'transparent')[1];
zHit.dispatch('mouseenter');
const zTip = tiles['c-z'].chart.children.find(n => n.tag === 'output').innerHTML;
assert(zTip.includes('r000163 · reverted'));
assert(zTip.includes('reverted: head reverted after a failed next-round confirmation'));
assert(!zTip.includes('21.87'));
const revertedBar = zSvg.children.find(n => n.tag === 'rect' && n.attrs['stroke-dasharray'] === '3 2');
assert(revertedBar && revertedBar.attrs.fill === 'none' && revertedBar.attrs.stroke === 'var(--rejected)');
assert(zSvg.children.some(n => n.tag === 'text' && n.textContent === 'reverted'));
const pSvg = tiles['c-pass'].chart.children.find(n => n.tag === 'svg');
assert(!pSvg.children.some(n => n.tag === 'circle' && n.attrs.cy === '150'), 'no-decision zero must not become a point');
assert(pSvg.children.some(n => n.tag === 'circle' && n.attrs['stroke-dasharray'] === '2 1'), 'no-decision ring remains');
const pHit = pSvg.children.filter(n => n.attrs.fill === 'transparent')[2];
pHit.dispatch('mouseenter');
const pTip = tiles['c-pass'].chart.children.find(n => n.tag === 'output').innerHTML;
assert(pTip.includes('no decision: not scored') && !pTip.includes('head M</span> 0.0%'));
console.log(`PASS ${path}: axis scaling, readings, gaps, reverted mark and tooltips`);
