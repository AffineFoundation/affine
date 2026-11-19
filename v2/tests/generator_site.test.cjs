/* From any directory, with jsdom available in NODE_PATH:
   npm install --prefix /tmp/affine-generator-test-deps --no-audit --no-fund jsdom@27
   NODE_PATH=/tmp/affine-generator-test-deps/node_modules node --test v2/tests/generator_site.test.cjs
   The page itself has no dependencies; jsdom is only this test's DOM harness. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const {JSDOM} = require('jsdom');

const ROOT = path.resolve(__dirname, '..');
const FIXTURES = path.join(__dirname, 'fixtures', 'generator');
const page = require(path.join(ROOT, 'site', 'static', 'generator.js'));
const read = relative => fs.readFileSync(path.join(FIXTURES, relative), 'utf8');
const json = relative => JSON.parse(read(relative));

function fixtureDOM(guide) {
  const file = guide ? 'generator-guide.html' : 'generator.html';
  const html = fs.readFileSync(path.join(ROOT, 'site', 'static', file), 'utf8');
  return new JSDOM(html, {url: 'https://affine.example/generator/'});
}

function fetchFixtures(overrides = {}) {
  const requested = [];
  return {
    requested,
    fetch: async url => {
      const pathname = new URL(url, 'https://affine.example').pathname;
      requested.push(pathname);
      assert.ok(pathname.startsWith('/generator/'), 'only same-origin Generator public files are fetched');
      const relative = pathname.slice('/generator/'.length);
      const contents = Object.hasOwn(overrides, relative) ? overrides[relative] : read(relative);
      if (contents === null) return {ok: false, status: 404};
      return {ok: true, status: 200, text: async () => contents, json: async () => JSON.parse(contents)};
    },
  };
}

test('all public JSON and JSONL fixtures parse without errors', () => {
  const state = page.normalizeState(json('state.json'), json('epochs/20261119T2000Z/manifest.json'), json('links.json'), Date.UTC(2026, 10, 19, 19, 20));
  assert.equal(state.epoch, '20261119T2000Z');
  assert.equal(state.start, Date.UTC(2026, 10, 19, 16));
  assert.equal(state.deadline, Date.UTC(2026, 10, 19, 20));
  assert.equal(state.submissions, 3);
  assert.equal(state.links, 2);
  assert.equal(state.king, null);
  assert.equal(page.utc(state.deadline), '2026-11-19 20:00:00 UTC');
  assert.equal(page.epochUtc(state.epoch), '2026-11-19 20:00:00 UTC');
  assert.equal(page.countdown(state.deadline, state.now), '00:40:00');

  const results = page.normalizeResults(json('epochs/20261119T1600Z/results.json'));
  assert.deepEqual(results.map(row => [row.uid, row.submitted, row.eligible, row.scored, row.accepted]).sort((a, b) => a[0] - b[0]), [[17, 2, 2, 2, 1], [23, 1, 1, 1, 1]]);
  const accepted = page.normalizeAccepted(read('epochs/20261119T1600Z/accepted.jsonl'));
  assert.equal(accepted.length, 2);
  assert.deepEqual(accepted.map(row => [row.id, row.uid, row.uploadedAt, row.passes, row.rollouts]), [
    ['00000000000000000003', 23, 1795097700000, 4, 8],
    ['00000000000000000002', 17, 1795096800000, 6, 16],
  ]);
  assert.equal(page.normalizeAccepted(read('accepted-all.jsonl')).length, 2);
  const payout = page.normalizePayout(json('payout.json'));
  assert.equal(payout.updatedAt, 1795116000000);
  assert.deepEqual(payout.rows.map(row => [row.uid, row.weight, row.tasks]), [[17, 0.5, 1], [23, 0.5, 1]]);
  const claimsOnly = page.normalizePayout({...json('payout.json'), weights: undefined});
  assert.deepEqual(claimsOnly.rows, []);
  const burn = page.normalizePayout({...json('payout.json'), weights: {'0': 1}});
  assert.deepEqual(burn.rows[0], {uid: 0, hotkey: null, weight: 1, tasks: null});
  const currentState = page.normalizeState({...json('state.json'), current_epoch: '20261120T0000Z'}, null, json('links.json'));
  assert.equal(currentState.scoringEpoch, '20261119T1600Z');
  assert.equal(currentState.submissions, null);
  assert.equal(page.submissionCount({submissions: []}, {}), null);
});

test('the dashboard renders every section from the fixture directory', async t => {
  const dom = fixtureDOM(false);
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const source = fetchFixtures();
  const data = await page.load(source.fetch, dom.window.document, '/generator/');
  const document = dom.window.document;
  assert.equal(document.querySelector('[data-f="epoch"]').textContent, '20261119T2000Z');
  assert.equal(document.querySelector('[data-f="submissions"]').textContent, '3');
  assert.equal(document.querySelector('[data-f="links"]').textContent, '2');
  assert.equal(document.querySelector('[data-f="king"]').textContent, 'fixture-king-version');
  assert.match(document.getElementById('window').textContent, /2026-11-19 16:00:00 UTC → 2026-11-19 20:00:00 UTC/);
  assert.equal(document.querySelectorAll('#hotkey-results tbody tr').length, 2);
  assert.equal(document.querySelectorAll('#accepted-tasks tbody tr').length, 2);
  assert.equal(document.querySelectorAll('#payout-rows tbody tr').length, 2);
  assert.equal(document.getElementById('results-epoch').textContent, '20261119T1600Z');
  assert.equal(data.accepted.length, 2);
  assert.equal(document.getElementById('status').textContent, 'Public data loaded');
  assert.deepEqual(new Set(source.requested), new Set([
    '/generator/epochs/20261119T1600Z/accepted.jsonl',
    '/generator/epochs/20261119T1600Z/results.json',
    '/generator/epochs/20261119T2000Z/manifest.json',
    '/generator/payout.json',
    '/generator/links.json',
    '/generator/state.json',
  ]));
});

test('the control plane publication schemas drive every dashboard section', () => {
  const state = page.normalizeState({
    version: 'affine-generator-state-v1', current_epoch: '20261120T0000Z',
    window_open: '2026-11-19T20:00:00Z', window_close: '2026-11-20T00:00:00Z',
    frozen_king_version: 'head-v1', last_epoch: '20261119T1600Z',
  }, {
    version: 'affine-generator-manifest-v1', epoch: '20261119T2000Z',
    files: [{valid: true, task_count: 8}, {valid: true, task_count: 5}, {valid: true, task_count: 0}],
  });
  assert.equal(state.start, Date.UTC(2026, 10, 19, 20));
  assert.equal(state.deadline, Date.UTC(2026, 10, 20));
  assert.equal(state.king, 'head-v1');
  assert.equal(state.submissions, 13);
  assert.equal(state.scoringEpoch, '20261119T1600Z');
});

test('first epoch, missing files and bad JSONL have graceful empty states', async t => {
  assert.deepEqual(page.normalizeAccepted(null), []);
  assert.deepEqual(page.normalizeAccepted(''), []);
  assert.deepEqual(page.normalizeResults({}), []);
  assert.deepEqual(page.normalizePayout(null).rows, []);
  assert.equal(page.normalizeState({}, null).epoch, null);
  assert.throws(() => page.parseJSONL('{not JSON}\n'), SyntaxError);
  assert.equal(page.timestamp('not-a-date'), null);
  assert.equal(page.epochTime('20260230T0000Z'), null);
  assert.equal(page.epochTime('20261119T1800Z'), null);
  assert.equal(page.submissionCount({}, {}), null);

  const emptyState = json('empty-state.json');
  assert.equal(emptyState.published_epochs, undefined);
  assert.equal(emptyState.last_epoch, undefined);
  const dom = fixtureDOM(false);
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const source = fetchFixtures({
    'state.json': JSON.stringify(emptyState),
    'payout.json': null,
    'links.json': null,
  });
  await page.load(source.fetch, dom.window.document, '/generator/');
  const document = dom.window.document;
  assert.equal(document.querySelector('[data-f="king"]').textContent, 'Freezes at deadline');
  assert.match(document.querySelector('#hotkey-results .empty').textContent, /first epoch has not been scored yet/);
  assert.match(document.querySelector('#accepted-tasks .empty').textContent, /No accepted tasks yet/);
  assert.match(document.querySelector('#payout-rows .empty').textContent, /No payout vector yet/);
  assert.equal(document.getElementById('status').textContent, 'Public data loaded');
  assert.ok(!source.requested.some(url => url.includes('/results.json')));
});

test('the Markdown fixture renders safely, including code, lists and tables', async t => {
  const dom = fixtureDOM(true);
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const source = fetchFixtures();
  await page.loadGuide(source.fetch, dom.window.document, './');
  const document = dom.window.document;
  assert.equal(document.querySelector('#guide-content h1').textContent, 'Generator miner guide');
  assert.equal(document.querySelectorAll('#guide-content ol li').length, 3);
  assert.equal(document.querySelector('#guide-content pre code').textContent, 'affine-generator --help');
  assert.equal(document.querySelectorAll('#guide-content table tbody tr').length, 2);
  assert.equal(document.querySelector('#guide-content a').getAttribute('rel'), 'noopener noreferrer');
  assert.equal(document.querySelectorAll('#guide-content script').length, 0);
  assert.match(document.getElementById('guide-content').textContent, /<script>alert\("not executable"\)<\/script>/);
  assert.equal(document.getElementById('status').textContent, 'Public miner guide');
  const code = dom.window.document.createElement('p');
  code.appendChild(page.inlineMarkdown('`<img src=x onerror=alert(1)>` [bad](javascript:alert(1))', document));
  assert.equal(code.querySelectorAll('img, a, script').length, 0);
  assert.equal(code.querySelector('code').textContent, '<img src=x onerror=alert(1)>');
});

test('the actual PR 296 guide renders its links and code blocks', () => {
  const dom = new JSDOM('', {url: 'https://affine.example/generator-guide/'});
  const container = dom.window.document.createElement('article');
  const guide = fs.readFileSync(path.join(ROOT, 'site', 'static', 'generator-guide.md'), 'utf8');
  page.renderMarkdown(guide, container, dom.window.document);
  assert.equal(container.querySelector('h1').textContent, 'Generator miner guide');
  assert.ok(container.querySelectorAll('h1,h2').length >= 10);
  assert.ok(container.querySelectorAll('pre').length >= 10);
  assert.ok(container.querySelector('a[href="https://github.com/AffineFoundation/affine/blob/bb/generator-integration/v2/miner/generator_miner.py"]'));
  assert.ok(container.querySelector('a[href="https://affine.io/generator"]'));
  assert.equal(container.querySelectorAll('script').length, 0);
  dom.window.close();
});

test('missing guide keeps the static launch placeholder', async t => {
  const dom = fixtureDOM(true);
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  await page.loadGuide(async () => ({ok: false, status: 404}), dom.window.document, '/generator/');
  assert.match(dom.window.document.getElementById('guide-content').textContent, /Loading the miner-kit guide from the same origin/);
  assert.equal(dom.window.document.getElementById('status').textContent, 'Miner guide pending publication');
});
