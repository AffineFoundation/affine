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
  const state = page.normalizeState(json('state.json'), json('epochs/20261119T2000Z/manifest.json'), Date.UTC(2026, 10, 19, 19, 20));
  assert.equal(state.epoch, '20261119T2000Z');
  assert.equal(state.start, Date.UTC(2026, 10, 19, 16));
  assert.equal(state.deadline, Date.UTC(2026, 10, 19, 20));
  assert.equal(state.submissions, 13);
  assert.equal(state.links, 3);
  assert.equal(state.king, '20261119T1600Z:8f1e6a0c9d2b');
  assert.equal(page.utc(state.deadline), '2026-11-19 20:00:00 UTC');
  assert.equal(page.epochUtc(state.epoch), '2026-11-19 20:00:00 UTC');
  assert.equal(page.countdown(state.deadline, state.now), '00:40:00');

  const results = page.normalizeResults(json('epochs/20261119T1600Z/results.json'));
  assert.deepEqual(results.map(row => [row.uid, row.submitted, row.eligible, row.scored, row.accepted]), [[17, 10, 9, 8, 3], [23, 3, 2, 2, 1]]);
  const accepted = page.normalizeAccepted(read('epochs/20261119T1600Z/accepted.jsonl'));
  assert.equal(accepted.length, 2);
  assert.deepEqual(accepted[0], {id: 'abcdef0123456789abcd', hotkey: '5FixtureMinerBeta0000000000000000000000000000000', uid: 23, uploadedAt: 1795090500000, passes: 9, rollouts: 16, q: 0.9328});
  assert.equal(page.normalizeAccepted(read('accepted-all.jsonl')).length, 2);
  const payout = page.normalizePayout(json('payout.json'));
  assert.equal(payout.updatedAt, 1795116000000);
  assert.deepEqual(payout.rows.map(row => [row.uid, row.weight, row.tasks]), [[17, 0.75, 3], [23, 0.25, 1]]);
  const claimsOnly = page.normalizePayout({...json('payout.json'), weights: undefined});
  assert.deepEqual(claimsOnly.rows.map(row => [row.uid, row.weight, row.tasks]), [[17, 0.75, 3], [23, 0.25, 1]]);
  const realShape = page.normalizeResults({by_hotkey: {'5Miner': {uid: 1, submitted: 2, eligible: 1, scored: 1, accepted: 1}}});
  assert.equal(realShape[0].hotkey, '5Miner');
  assert.equal(page.normalizeState({epoch: '20261120T0000Z'}, null).deadline, Date.UTC(2026, 10, 20));
  assert.equal(page.submissionCount({submissions: []}, {}), null);
});

test('the dashboard renders every section from the fixture directory', async t => {
  const dom = fixtureDOM(false);
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const source = fetchFixtures();
  const data = await page.load(source.fetch, dom.window.document, '/generator/');
  const document = dom.window.document;
  assert.equal(document.querySelector('[data-f="epoch"]').textContent, '20261119T2000Z');
  assert.equal(document.querySelector('[data-f="submissions"]').textContent, '13');
  assert.equal(document.querySelector('[data-f="links"]').textContent, '3');
  assert.match(document.getElementById('window').textContent, /2026-11-19 16:00:00 UTC → 2026-11-19 20:00:00 UTC/);
  assert.equal(document.querySelectorAll('#hotkey-results tbody tr').length, 2);
  assert.equal(document.querySelectorAll('#accepted-tasks tbody tr').length, 2);
  assert.equal(document.querySelectorAll('#payout-rows tbody tr').length, 2);
  assert.equal(document.getElementById('results-epoch').textContent, '20261119T1600Z');
  assert.equal(data.accepted.length, 2);
  assert.equal(document.getElementById('status').textContent, 'Public data loaded');
  assert.deepEqual(source.requested.sort(), [
    '/generator/epochs/20261119T1600Z/accepted.jsonl',
    '/generator/epochs/20261119T1600Z/results.json',
    '/generator/epochs/20261119T2000Z/manifest.json',
    '/generator/payout.json',
    '/generator/state.json',
  ]);
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

  const emptyState = {...json('state.json'), epoch: '20261120T0000Z', last_scored_epoch: null, king_version: null};
  const dom = fixtureDOM(false);
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const source = fetchFixtures({
    'state.json': JSON.stringify(emptyState),
    'payout.json': null,
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
  await page.loadGuide(source.fetch, dom.window.document, '/generator/');
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

test('missing guide keeps the static launch placeholder', async t => {
  const dom = fixtureDOM(true);
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  await page.loadGuide(async () => ({ok: false, status: 404}), dom.window.document, '/generator/');
  assert.match(dom.window.document.getElementById('guide-content').textContent, /guide will be published here/);
  assert.equal(dom.window.document.getElementById('status').textContent, 'Miner guide pending publication');
});
