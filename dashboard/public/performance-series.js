/* Pure series selection shared by both viewport layouts. */
(function (root) {
  'use strict';
  const integer = value => Number.isInteger(value) && value >= 0;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  function cohortKey(row) {
    return JSON.stringify([row.evaluation_split || 'heldout', row.training_run_id,
      row.dataset_id, row.taskset_hash, row.fixed_task_ids, row.seed,
      row.count, row.requested_count, row.harness, row.environment_version,
      row.model, row.model_runtime_revision, row.output_token_budget,
      row.policy_kind, row.sampling_policy, row.experiment_id,
      row.runtime_profile_sha256]);
  }
  function measured(row) {
    return row.status === 'complete' && finite(row.timestamp) &&
      integer(row.count) && row.count > 0 && integer(row.successes) &&
      row.successes <= row.count && finite(row.mean_reward) &&
      row.mean_reward >= 0 && row.mean_reward <= 1 &&
      (row.evaluation_split !== 'train' ||
        (row.generation_origin === 'fresh-operator-generation' && row.comparability_passed === true &&
         row.completed_count === row.requested_count &&
         row.count === row.requested_count && row.error_count === 0 &&
         row.mean_reward === row.successes / row.count));
  }
  function history(snapshot, epochs, split) {
    const order = new Map(epochs.map((row, index) => [row.id, index]));
    const supplied = split === 'train' ? snapshot.training_evaluations : snapshot.evaluations;
    const source = Array.isArray(supplied) ? supplied : [];
    const rows = source.filter(row => row && order.has(row.epoch_id) &&
      row.env_id === 'affine_math' && finite(row.timestamp) &&
      (split === 'train' ? row.evaluation_split === 'train' : row.evaluation_split !== 'train'))
      .sort((a, b) => order.get(a.epoch_id) - order.get(b.epoch_id) ||
        (split === 'train' ? Number(b.phase === 'baseline') - Number(a.phase === 'baseline') : 0) ||
        a.timestamp - b.timestamp);
    const latest = new Map();
    for (const row of rows) {
      // A fresh baseline and current checkpoint can belong to the same epoch.
      const key = split === 'train'
        ? JSON.stringify([cohortKey(row), row.checkpoint || row.epoch_id, row.phase])
        : row.epoch_id;
      latest.set(key, row);
    }
    const results = [], setups = new Map();
    let previous = null, gap = false;
    for (const row of latest.values()) {
      if (!measured(row)) { gap = true; continue; }
      const key = cohortKey(row);
      if (!setups.has(key)) setups.set(key, setups.size + 1);
      results.push({...row, display_setup: setups.get(key),
        setup_changed: previous !== null && key !== cohortKey(previous),
        measurement_gap: gap});
      previous = row; gap = false;
    }
    return results;
  }
  function segments(rows, coordinates) {
    const groups = [];
    rows.forEach((row, index) => {
      if (!index || row.measurement_gap || cohortKey(row) !== cohortKey(rows[index - 1])) groups.push([]);
      groups[groups.length - 1].push(coordinates[index]);
    });
    return groups;
  }
  const api = {cohortKey, measured, history, segments};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AffinePerformanceSeries = api;
})(typeof globalThis === 'undefined' ? this : globalThis);
