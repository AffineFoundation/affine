const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const series = require('./public/performance-series.js');

const common = {
  evaluation_split: 'train', training_run_id: 'fixed-run', dataset_id: 'fixed-data',
  taskset_hash: 'fixed-tasks', seed: 'fixed-seed', count: 128, requested_count: 128,
  experiment_id: 'fixed-study', runtime_profile_sha256: 'fixed-profile',
  output_token_budget: 2048,
  fixed_initial_trained: {mask_sha256: 'fixed55', checkpoint: 'fixed-initial', optimizer_step: 67, count: 55},
};
const upstream = {...common, phase: 'baseline', checkpoint: 'upstream', training_steps: 0};
const genesis = {...common, phase: 'baseline', checkpoint: 'genesis', training_steps: 0};
const current = {...common, phase: 'latest', checkpoint: 'current', training_steps: 68};
const next = {...common, phase: 'latest', checkpoint: 'next', training_steps: 72};
assert.equal(series.comparableTransition(upstream, genesis), false);
assert.equal(series.comparableTransition(genesis, upstream), false);
assert.equal(series.comparableTransition(genesis, current), true);
assert.equal(series.comparableTransition(undefined, upstream), false);
assert.equal(series.comparableTransition(genesis, {...current, measurement_gap: true}), false);
assert.equal(series.comparableTransition(genesis, {...current, runtime_profile_sha256: 'other'}), false);
assert.deepEqual(series.segments([upstream, genesis, current], [0, 1, 2]), [[0], [1, 2]]);
assert.deepEqual(series.segments([upstream, genesis, current, next], [0, 1, 2, 3]), [[0], [1, 2, 3]]);
assert.deepEqual(series.segments([genesis, {...current, measurement_gap: true}], [0, 1]), [[0], [1]]);

// Both the visible change badge and line must use the same comparison rule.
const network = fs.readFileSync(path.join(__dirname, 'public/network.js'), 'utf8');
assert.ok(network.includes('window.AffinePerformanceSeries.comparableTransition(rows[index-1],row)'));
const heldoutBefore = {evaluation_split: 'heldout', dataset_id: 'heldout', phase: 'before'};
const heldoutAfter = {...heldoutBefore, phase: 'after'};
assert.equal(series.comparableTransition(heldoutBefore, heldoutAfter), true);

function measuredTrain(row, epoch, timestamp, solved) {
  return {...row, epoch_id: epoch, timestamp, status: 'complete', env_id: 'affine_math',
    successes: 96, mean_reward: 96 / 128, completed_count: 128, error_count: 0,
    comparability_passed: true, generation_origin: 'fresh-operator-generation',
    fixed_initial_trained: {...common.fixed_initial_trained, mask_sha256: 'a'.repeat(64),
      checkpoint: 'b'.repeat(64), solved, wrong: 55 - solved, unresolved: 0, errors: 0},
  };
}
const historyInput = [measuredTrain(next, 'e72', 4, 42), measuredTrain(genesis, 'e0', 3, 41),
  measuredTrain(current, 'e68', 2, 39), measuredTrain(upstream, 'e0', 1, 43)];
const history = series.history({training_evaluations: historyInput},
  [{id: 'e0'}, {id: 'e68'}, {id: 'e72'}], 'train');
assert.deepEqual(history.map(row => row.checkpoint), ['upstream', 'genesis', 'current', 'next']);
assert.deepEqual(history.map(row => series.metric(row).count), [55, 55, 55, 55]);
assert.deepEqual(series.segments(history, [0, 1, 2, 3]), [[0], [1, 2, 3]]);

// Execute the real tooltip function, with pure presentation dependencies.
const detailsStart = network.indexOf('  function evaluationDetails(row) {');
const detailsEnd = network.indexOf('  function validOutcomes(row) {', detailsStart);
assert.ok(detailsStart >= 0 && detailsEnd > detailsStart);
const details = new Function('evaluationPhase', 'evaluationMetric', 'count', 'number',
  'epochName', 'percent', 'evaluationValue', 'utc',
  network.slice(detailsStart, detailsEnd) + '\nreturn evaluationDetails;')(
    () => 'Evaluation', row => row.fixed_initial_trained,
    value => Number.isInteger(value) && value >= 0, String, () => 'Epoch 91',
    value => `${100 * value}%`, () => 0.5, String);
const description = details({...genesis,
  reference_label: 'Actual run genesis — optimizer step 0 of this run; earlier training retained',
  checkpoint: 'b'.repeat(64), successes: 96, actual_trained_task_count: 0,
  completed_count: 128, output_token_budget: 2048, timestamp: 1,
  fixed_initial_trained: {...common.fixed_initial_trained, successes: 43},
  exposure_groups: {untrained: {count: 128, solved: 96}},
});
assert.ok(description.includes('0 tasks trained in this run'));
assert.ok(description.includes('Not yet trained in this run: 96 / 128 solved'));
assert.ok(description.includes('earlier training retained'));
assert.ok(description.includes('trained by optimizer step 67 in this run'));
assert.ok(!description.includes('0 tasks previously trained'));
console.log('Passed: baseline separation, metric continuity, missing/profile-changed records, unchanged held-out comparisons');
