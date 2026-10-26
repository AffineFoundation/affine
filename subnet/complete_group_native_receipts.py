"""Compact trusted native receipts without duplicating committed trajectories.

These helpers neither grade nor authenticate signatures. The coordinator calls
compact_native_grades on its own native result; the trainer expands only a
signed coordinator receipt after authenticating the original document bytes.
"""
from copy import deepcopy

from .complete_group_sampling import digest, need, sha

FULL = 'complete-group-native-grading-v1'
COMPACT = 'complete-group-native-grading-compact-v1'
FIELDS = {'version', 'batch_sha256', 'manifest_sha256', 'miner', 'task_hash',
          'prompt', 'results', 'sampling_verified'}
ROW_FIELDS = {'nonce', 'classification', 'reward', 'stop_reason', 'output_sha256',
              'native_classification', 'done', 'observations'}


def _validate(grade, version):
    need(type(grade) is dict and set(grade) == FIELDS and grade['version'] == version,
         'exact versioned native grading receipt')
    for key in ('batch_sha256', 'manifest_sha256', 'miner', 'task_hash'):
        sha(grade[key], 'native ' + key)
    need(grade['sampling_verified'] is False, 'native grades do not certify sampling')
    prompt = grade['prompt']
    need(type(prompt) is list and 0 < len(prompt) <= 65536 and
         all(type(t) is int and 0 <= t < 1_000_000 for t in prompt), 'bounded native prompt')
    need(type(grade['results']) is list and 2 <= len(grade['results']) <= 128,
         'bounded complete native results')
    fields = ROW_FIELDS | ({'prompt', 'output', 'task_hash'} if version == FULL else set())
    for nonce, row in enumerate(grade['results']):
        need(type(row) is dict and set(row) == fields and type(row['nonce']) is int
             and row['nonce'] == nonce, 'exact ordered native receipt nonce')
        sha(row['output_sha256'], 'native output digest')
        need(row['classification'] in ('positive', 'negative', 'neutral') and
             type(row['reward']) in (int, float) and
             row['reward'] == float(row['classification'] == 'positive') and
             row['stop_reason'] in ('eos', 'budget') and row['done'] is True,
             'explicit native outcome and stopping receipt')


def compact_native_grades(grade):
    _validate(grade, FULL)
    for row in grade['results']:
        need(row['prompt'] == grade['prompt'] and row['task_hash'] == grade['task_hash']
             and digest(row['output']) == row['output_sha256'], 'original native prompt/task/output')
    result = deepcopy({k: v for k, v in grade.items() if k != 'results'})
    result['version'] = COMPACT
    result['results'] = [{k: deepcopy(row[k]) for k in ROW_FIELDS} for row in grade['results']]
    return result


def expand_native_grades(grade, batch):
    """Recover token lists solely from a document matching the signed hashes."""
    _validate(grade, COMPACT)
    need(type(batch) is dict and digest(batch) == grade['batch_sha256'] and
         batch['miner'] == grade['miner'] and batch['task_hash'] == grade['task_hash'],
         'compact native receipt binds original batch/miner/task')
    rolls = batch['rollouts']
    need(type(rolls) is list and len(rolls) == len(grade['results']), 'all native receipt attempts retained')
    results = []
    for nonce, (row, rollout) in enumerate(zip(grade['results'], rolls, strict=True)):
        need(type(rollout['seed']) is int and rollout['seed'] == nonce and
             rollout['task_hash'] == grade['task_hash'] and len(rollout['turns']) == 1,
             'original native attempt/task/turn')
        turn = rollout['turns'][0]
        need(turn['prompt'] == grade['prompt'] and digest(turn['output']) == row['output_sha256'],
             'committed token hash differs from signed native receipt')
        results.append(dict(deepcopy(row), prompt=list(grade['prompt']),
                            output=list(turn['output']), task_hash=grade['task_hash']))
    result = deepcopy({k: v for k, v in grade.items() if k != 'results'})
    result.update(version=FULL, results=results)
    return result
