"""Compare two update reports of the trainer (report.private.json, as train_pit.py writes it) on the same batch.

    python -m ops.parallel_eval.pit.compare_train_reports A.json B.json [--top 10] [--json]

Use: the sharded 8x RTX 5090 update of an epoch reported a preclip gradient norm 11% above the B300's. Both
trainers write, for every rollout, the numbers the loss used. This tool reads the two reports and answers, without
a GPU:

  inputs    do both updates have the same batch, checkpoint, fp32 master, signed settings and turn parameters?
  rollouts  which rollouts were trained with another mask: another derail cut (derail_index, loss_tokens) or other
            tokens outside the sampler support (support_outside_tokens)? For the rollouts with the same kept
            tokens: how far apart are their mean log-probs (the forward alone)?
  weight    how much of the gradient those rollouts can own (see `weight`).

It does not read gradients or weights, so it cannot prove a cause. It shows whether a discrete difference in the
loss exists and whether it is large enough to matter. The same report from two packings of one trainer on one
machine must show no rollout difference at all.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

INPUT_KEYS = ('input', 'data_sha256', 'group_count', 'rollout_count', 'config', 'loss_aggregation', 'unresolved_filter')
TOTAL_KEYS = ('preclip_gradient_norm', 'objective_loss', 'trained_rollouts', 'masked_rollouts', 'trained_tokens',
              'loss_tokens', 'derailed_rollouts')
MASK_KEYS = ('derail_index', 'loss_tokens', 'support_outside_tokens')


def _flat(value: Any, prefix: str = '') -> dict:
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            out.update(_flat(item, '%s.%s' % (prefix, key) if prefix else str(key)))
        return out
    return {prefix: value}


def _turn(report: Mapping[str, Any]) -> dict:
    params = report.get('train_params') or {}
    return {k: params.get(k) for k in ('lr', 'step_mult', 'knobs', 'knobs_applied', 'lr_multiplier')}


def _master(report: Mapping[str, Any]) -> dict:
    fp = ((report.get('optimizer') or {}).get('sgd') or {}).get('fp32_master') or {}
    return {k: fp.get(k) for k in ('input_master_digest', 'input_bf16_sha256', 'source')}


def pg_tokens(row: Mapping[str, Any]) -> int:
    """Tokens of one rollout in the policy-gradient mask: kept by the derail mask and inside the sampler support."""
    return int(row['loss_tokens']) - int(row.get('support_outside_tokens') or 0)


def weight(row: Mapping[str, Any], token_mean_share: Optional[float]) -> float:
    """Model of the squared size of one rollout's gradient, up to one factor common to all rollouts.

    The live loss is a mean over rollouts of advantage x (mean over the rollout's policy-gradient tokens). If the
    token gradients are uncorrelated and of equal size, a mean over n tokens has a squared size of 1 / n. So a
    rollout that the derail mask cuts to few tokens weighs more than a long one. `token_mean_share` is the factor
    of the loss_agg = token_mean knob (rollouts x tokens of this rollout / tokens of all); None when it is off."""
    n = pg_tokens(row)
    if n <= 0:
        return 0.0
    scale = float(row['advantage']) * (1.0 if token_mean_share is None else token_mean_share)
    return scale * scale / n


def _weights(report: Mapping[str, Any]) -> dict:
    rows = report['trajectories']
    knobs = (report.get('train_params') or {}).get('knobs') or {}
    total = sum(int(r['tokens']) for r in rows)
    share = (lambda r: len(rows) * int(r['tokens']) / total) if knobs.get('loss_agg') == 'token_mean' else (lambda r: None)
    return {(r['task_id'], r['nonce']): weight(r, share(r)) for r in rows}


def is_update_report(report: Any) -> bool:
    """True for the trainer report of a finished update that has the numbers of every rollout."""
    return isinstance(report, Mapping) and report.get('status') == 'updated' and isinstance(report.get('trajectories'), list)


def input_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    """-> {input: [value in A, value in B]} for each input that differs between two update reports.

    The inputs are the batch, the checkpoint, the fp32 master, the signed settings and the turn parameters."""
    flat_a, flat_b = (_flat(dict({k: r.get(k) for k in INPUT_KEYS}, turn=_turn(r), fp32_master=_master(r))) for r in (a, b))
    return {key: [flat_a.get(key), flat_b.get(key)]
            for key in sorted(set(flat_a) | set(flat_b)) if flat_a.get(key) != flat_b.get(key)}


def compare(a: Mapping[str, Any], b: Mapping[str, Any], *, top: int = 10) -> dict:
    """-> the comparison of two update reports as plain data (see the module docstring)."""
    for report in (a, b):
        if not is_update_report(report):
            raise ValueError('an update report with status "updated" and its trajectories is required')
    inputs = input_differences(a, b)
    totals = {}
    for key in TOTAL_KEYS:
        x, y = a.get(key), b.get(key)
        totals[key] = dict(a=x, b=y, ratio=(y / x if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x else None))
    rows_a = {(r['task_id'], r['nonce']): r for r in a['trajectories']}
    rows_b = {(r['task_id'], r['nonce']): r for r in b['trajectories']}
    shared = sorted(set(rows_a) & set(rows_b))
    w_a, w_b = _weights(a), _weights(b)
    sum_a, sum_b = math.fsum(w_a.values()), math.fsum(w_b.values())
    changed, drift, dot = [], [], 0.0
    for key in shared:
        x, y = rows_a[key], rows_b[key]
        if x['loss_tokens'] == y['loss_tokens']:      # the same kept tokens: the difference is the forward alone
            drift.append(abs(float(x['old_logprob_mean']) - float(y['old_logprob_mean'])))
        n_a, n_b = pg_tokens(x), pg_tokens(y)
        if n_a > 0 and n_b > 0:      # the two masks share at most min(n_a, n_b) tokens (a derail mask is a prefix)
            dot += math.sqrt(w_a[key] * w_b[key]) * min(n_a, n_b) / math.sqrt(n_a * n_b) * (
                1.0 if float(x['advantage']) * float(y['advantage']) >= 0 else -1.0)
        differs = [field for field in MASK_KEYS + ('advantage', 'stop', 'tokens') if x.get(field) != y.get(field)]
        if differs:
            changed.append(dict(
                task_id=key[0], nonce=key[1], differs=differs, tokens=[x['tokens'], y['tokens']], stop=[x['stop'], y['stop']],
                advantage=[x['advantage'], y['advantage']], derail_index=[x['derail_index'], y['derail_index']],
                loss_tokens=[x['loss_tokens'], y['loss_tokens']], pg_tokens=[n_a, n_b],
                old_logprob_mean=[x['old_logprob_mean'], y['old_logprob_mean']],
                weight_share=[w_a[key] / sum_a if sum_a else 0.0, w_b[key] / sum_b if sum_b else 0.0]))
    changed.sort(key=lambda row: -abs(row['weight_share'][1] - row['weight_share'][0]))
    masks = [row for row in changed if any(field in row['differs'] for field in MASK_KEYS)]
    norm = totals['preclip_gradient_norm']['ratio']
    return dict(
        inputs_different=inputs,
        totals=totals,
        microbatch=[a.get('microbatch') and {k: v for k, v in a['microbatch'].items() if k != 'ranks'},
                    b.get('microbatch') and {k: v for k, v in b['microbatch'].items() if k != 'ranks'}],
        rollouts=dict(
            shared=len(shared), only_a=len(set(rows_a) - set(rows_b)), only_b=len(set(rows_b) - set(rows_a)),
            different=len(changed), different_mask=len(masks),
            different_by_field={field: sum(field in row['differs'] for row in changed)
                                for field in MASK_KEYS + ('advantage', 'stop', 'tokens')},
            same_kept_tokens=len(drift),
            old_logprob_mean_abs_difference=dict(max=max(drift) if drift else None,
                                                 median=statistics.median(drift) if drift else None)),
        weight_model=dict(
            assumption='token gradients uncorrelated and of equal size: weight = advantage^2 / policy-gradient tokens',
            different_mask_share=[sum(row['weight_share'][0] for row in masks), sum(row['weight_share'][1] for row in masks)],
            modelled_norm_ratio=math.sqrt(sum_b / sum_a) if sum_a > 0 and sum_b > 0 else None,
            modelled_cosine=dot / math.sqrt(sum_a * sum_b) if sum_a > 0 and sum_b > 0 else None,
            measured_norm_ratio=norm),
        top=changed[:top])


def render(out: Mapping[str, Any]) -> str:
    lines = ['inputs']
    if out['inputs_different']:
        lines += ['  DIFFERENT %s: %r | %r' % (key, x, y) for key, (x, y) in out['inputs_different'].items()]
    else:
        lines.append('  the same batch, checkpoint, fp32 master, settings and turn parameters')
    lines.append('totals (A | B | B/A)')
    for key, row in out['totals'].items():
        lines.append('  %-24s %r | %r | %s' % (key, row['a'], row['b'], '-' if row['ratio'] is None else '%.6f' % row['ratio']))
    lines.append('  microbatch A %r' % (out['microbatch'][0],))
    lines.append('  microbatch B %r' % (out['microbatch'][1],))
    r = out['rollouts']
    lines.append('rollouts: %d in both, %d only in A, %d only in B' % (r['shared'], r['only_a'], r['only_b']))
    lines.append('  with another mask: %d (%s)' % (r['different_mask'], ', '.join(
        '%s %d' % (field, r['different_by_field'][field]) for field in MASK_KEYS)))
    lines.append('  with the same mask and another advantage, stop or length: %d' % (r['different'] - r['different_mask']))
    d = r['old_logprob_mean_abs_difference']
    lines.append('  mean log-prob over the kept tokens, |A - B|, for the %d rollouts with the same kept tokens: max %s, median %s' % (
        r['same_kept_tokens'], '-' if d['max'] is None else '%.3e' % d['max'], '-' if d['median'] is None else '%.3e' % d['median']))
    m = out['weight_model']
    lines.append('weight model (%s)' % m['assumption'])
    lines.append('  rollouts with another mask own %.1f%% of the weight in A and %.1f%% in B' % (
        100 * m['different_mask_share'][0], 100 * m['different_mask_share'][1]))
    fmt = lambda v: '-' if v is None else '%.4f' % v
    lines.append('  norm ratio B/A: modelled %s, measured %s' % (fmt(m['modelled_norm_ratio']), fmt(m['measured_norm_ratio'])))
    lines.append('  cosine of the two gradients, modelled: %s' % fmt(m['modelled_cosine']))
    if out['top']:
        lines.append('rollouts that differ, by change of weight share')
        for row in out['top']:
            lines.append('  %s nonce %s: %s; derail_index %r | %r, loss tokens %r | %r, policy-gradient tokens %r | %r, '
                         'advantage %.4f | %.4f, weight share %.2f%% | %.2f%%' % (
                             str(row['task_id'])[:12], row['nonce'], ', '.join(row['differs']), *row['derail_index'],
                             *row['loss_tokens'], *row['pg_tokens'], *row['advantage'],
                             100 * row['weight_share'][0], 100 * row['weight_share'][1]))
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('a', help='report.private.json of update A (for example the B300 update)')
    ap.add_argument('b', help='report.private.json of update B (for example the sharded update)')
    ap.add_argument('--top', type=int, default=10, help='rollouts to list (default 10)')
    ap.add_argument('--json', action='store_true', help='print the comparison as JSON')
    args = ap.parse_args(argv)
    out = compare(json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text()), top=args.top)
    print(json.dumps(out, indent=1, default=str) if args.json else render(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
