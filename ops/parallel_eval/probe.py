"""Exact-plan probe coverage admission, separate from the unchanged live scorer.

Never trust a RESULT's summary counters as proof of a complete eval. This only
checks transport/coverage integrity; it does not change grading or thresholds.
Returned evidence contains neither draw indices nor request randomness.
"""


def exact_probe(result, indices, checkpoint_id, *, env='affine_gatepool', k=8):
    if type(k) is not int or k != 8:
        raise ValueError('live candidate rollout count required')
    plan = list(indices)
    if (not plan or any(type(i) is not int or i < 0 for i in plan)
            or len(set(plan)) != len(plan)):
        raise ValueError('unique ordinary task indices required')
    expected = len(plan) * k
    if (result.get('checkpoint_id') != checkpoint_id
            or result.get('partial') is not False
            or type(result.get('samples')) is not int or result['samples'] != expected
            or type(result.get('planned_samples')) is not int or result['planned_samples'] != expected):
        raise ValueError('complete exact-checkpoint probe required')
    envs = result.get('envs')
    if not isinstance(envs, dict) or set(envs) != {env}:
        raise ValueError('exact probe environment required')
    row = envs[env]
    cells = row.get('per_task')
    if not isinstance(cells, dict) or set(cells) != {str(i) for i in plan}:
        raise ValueError('actual probe cells do not match frozen draw')
    if (type(row.get('samples')) is not int or row['samples'] != expected
            or type(row.get('tasks')) is not int or row['tasks'] != len(plan)):
        raise ValueError('environment summary disagrees with frozen plan')
    for cell in cells.values():
        if (not isinstance(cell, dict) or type(cell.get('n')) is not int
                or cell['n'] != k or not isinstance(cell.get('L'), list)
                or len(cell['L']) != k):
            raise ValueError('every drawn task needs all candidate rollouts')
    return dict(version='pe-exact-probe-coverage-v1', complete=True,
                tasks=len(plan), samples=expected, rollouts_per_task=k,
                checkpoint=checkpoint_id, environment=env)
