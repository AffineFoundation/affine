"""Head-vs-head parity of two probe serving layouts.

A serving layout may change speed only. This harness checks that. It runs one
checkpoint (the HEAD) under two layouts, on the same tasks and with the same
seeds, and compares the pass rate of every task.

Steps:

  plan      write one probe plan: the tasks, the seed, the rollouts per task.
            Both sides run this file.
  run-side  on one box: serve the checkpoint under one layout, run the plan,
            bring back a "side record". Run it once per layout. The two boxes
            can differ: the live layout on B300, TP2 on RTX 5090. The
            checkpoint must be on the box already, under <root>/ckpt/<id>.
  compare   read two side records. Report the pass-rate delta of every task,
            and decide if the layouts differ.

The same request seed does not give the same tokens under two layouts: batches
differ, and so does the order of float operations. So the two sides are two
samples of the same model, and ``compare`` is a statistical test. One task has
few rollouts (8 per side in the gate), so the tests pool the tasks:

  shift       the sum of the per-task deltas against its exact spread when the
              layouts are equal (each task is one stratum; this is the
              Cochran-Mantel-Haenszel test). It finds a layout that moves the
              pass rates one way.
  dispersion  the spread of the per-task deltas against the spread that
              sampling alone gives. It finds a layout that moves some tasks up
              and other tasks down.
  tail        the number of tasks with a small Fisher p-value against the
              exact chance of that many when sampling alone acts.
  symptoms    the same shift test on unresolved answers, output-cap hits and
              grader errors.

With the default ``z_crit`` of 3, two equal layouts are called different in
about 1 comparison of 100 (0.7% measured on 20000 simulated pairs).

Verdicts:

  different               a test is beyond ``z_crit`` (default 3).
  parity                  no test is, and the interval of the mean delta lies
                          inside ``max_shift``.
  no_difference_detected  no test is, and no ``max_shift`` was given. The
                          report says how small a shift this run can see.
  inconclusive            no test is, but the run is partial or cannot resolve
                          ``max_shift``. Use more tasks.
  not_comparable          the two sides did not run the same plan.

The trainer update is tracked too. The preclip gradient norm is the size of the
gradient of one update, before the trainer limits it to ``max_gradient_norm``.
``compare`` reads the trainer report of each side (``--train-a``, ``--train-b``:
``report.private.json`` of ``train_pit.py``, for the update that the trainer of
that side made from one batch). The report then has the norm of each side and
the ratio B / A. The norm is a tracked number, not a test: it does not change
the verdict or the exit code. ``compare_train_reports.py`` compares the same two
updates rollout by rollout.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import sys
import time
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ops.parallel_eval.live.pe_probe_plan import build_probe_plan

from . import GATE_K_CAND, GATE_MAX_TOKENS
from . import compare_train_reports as CTR
from . import probe_layout as PL
from . import probe_serve as PS
from .box import _load_public_harness, normalize_meta_pod
from .box_job import RemoteBoxJobStore
from .shadow_compare import GRAD_NORM_REL_TOL, gradient_norm_within_tol

REPORT_VERSION = 'pit-probe-parity-v1'
Z_CRIT = 3.0
TAIL_ALPHA = 0.01
# Exit codes of ``compare``.
EXIT = {'parity': 0, 'no_difference_detected': 0, 'different': 1, 'inconclusive': 2, 'not_comparable': 2}
# Count fields of one task cell in a probe result, and what each one counts.
COUNTS = {'pass': 'c', 'unresolved': 'u', 'budget_hit': 'b', 'error': 'x'}
# Keys under which a document holds a trainer report or its summary: ``TRAINING-RESULT.json`` of a live epoch
# (payload, report, training), the result of ``train_v834e``, the training report that the host stores for an
# item, ``markers/train.done``.
UPDATE_HOLDERS = ('payload', 'report_doc', 'training_report', 'report', 'training', 'training_report_summary', 'summary')
# The job of one update: the checkpoint that it starts from, the batch, and the optimizer step.
UPDATE_JOB = ('input_checkpoint', 'data_sha256', 'optimizer_step_before')
UPDATE_NUMBERS = ('objective_loss', 'trained_rollouts', 'masked_rollouts', 'trained_tokens', 'loss_tokens',
                  'derailed_rollouts')


class ParityError(ValueError):
    """The two inputs cannot be one head-vs-head comparison."""


# ------------------------------------------------------------------ exact statistics of one task
@lru_cache(maxsize=None)
def null_task(n_a: int, n_b: int, m: int) -> Tuple[Tuple[float, ...], float, float]:
    """One task when the layouts are equal: ``m`` hits fall at random on ``n_a + n_b`` rollouts.

    Returns (pmf of the hits of side B, variance of the rate delta, variance of the squared standard delta).
    """
    total = math.comb(n_a + n_b, m)
    pmf = tuple(
        math.comb(n_b, x) * math.comb(n_a, m - x) / total if 0 <= m - x <= n_a else 0.0 for x in range(n_b + 1))
    deltas = [x / n_b - (m - x) / n_a for x in range(n_b + 1)]
    var = sum(p * d * d for p, d in zip(pmf, deltas))
    var_sq = sum(p * (d * d / var) ** 2 for p, d in zip(pmf, deltas)) - 1.0 if var > 0 else 0.0
    return pmf, var, var_sq


def fisher_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p-value of one task: side A has ``hits_a`` of ``n_a``, side B ``hits_b`` of ``n_b``."""
    pmf = null_task(n_a, n_b, hits_a + hits_b)[0]
    return min(1.0, sum(p for p in pmf if p <= pmf[hits_b] * (1 + 1e-9)))


@lru_cache(maxsize=None)
def _tail_chance(n_a: int, n_b: int, m: int, alpha: float) -> float:
    """Chance that the Fisher p-value of one task is at most ``alpha`` when the layouts are equal."""
    pmf = null_task(n_a, n_b, m)[0]
    return sum(p for x, p in enumerate(pmf) if p > 0 and fisher_p(m - x, n_a, x, n_b) <= alpha)


def _z(value: float, variance: float) -> Optional[float]:
    return value / math.sqrt(variance) if variance > 0 else None


def _sum_of_squares_z(stat: float, mean: float, variance: float) -> Optional[float]:
    """Where a sum of squares lies on the normal scale.

    A sum of squares has a long right tail, so (stat - mean) / sd calls too many runs different. This uses the
    chi-square with the same mean and variance, through the Wilson-Hilferty cube root.
    """
    if mean <= 0 or variance <= 0:
        return None
    freedom = 2 * mean * mean / variance
    return ((stat / mean) ** (1 / 3) - (1 - 2 / (9 * freedom))) / math.sqrt(2 / (9 * freedom))


def _count_tail(chances: Sequence[float], seen: int) -> float:
    """Exact chance of ``seen`` or more events when event i has chance ``chances[i]`` (Poisson binomial)."""
    if seen <= 0:
        return 1.0
    below = [1.0] + [0.0] * (seen - 1)                 # chance of exactly j events so far, j < seen
    for q in chances:
        if q > 0:
            for j in range(seen - 1, 0, -1):
                below[j] = below[j] * (1 - q) + below[j - 1] * q
            below[0] *= 1 - q
    return min(1.0, max(0.0, 1.0 - sum(below)))


# ------------------------------------------------------------------ inputs
def load_side(doc: Mapping[str, Any], *, tag: Optional[str] = None, default_tag: str = 'side') -> Dict[str, Any]:
    """A side record from ``probe_serve.py side``, or a bare probe result (``result.json`` of a run)."""
    if not isinstance(doc, Mapping):
        raise ParityError('a side must be a JSON object')
    if doc.get('version') == PS.SIDE_VERSION:
        result, serve = doc.get('result'), doc.get('serve') or {}
        side = dict(tag=tag or doc.get('tag'), plan=doc.get('plan'), layout=serve.get('layout'), serve_s=serve.get('serve_s'),
                    gpus=serve.get('gpus'))
    elif isinstance(doc.get('envs'), Mapping) and 'checkpoint_id' in doc:
        result = doc
        serve = doc.get('serve_layout') or {}       # the eval stage of a shadow box adds this to its probe result
        side = dict(tag=tag, plan=None, layout=serve.get('layout'), serve_s=serve.get('serve_s'), gpus=serve.get('gpus'))
    else:
        raise ParityError('not a side record and not a probe result')
    if not isinstance(result, Mapping) or not isinstance(result.get('envs'), Mapping):
        raise ParityError('the side has no probe result')
    cells: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for env_id, env in result['envs'].items():
        for index, cell in (env.get('per_task') or {}).items():
            if isinstance(cell, Mapping) and cell.get('n'):
                cells[(env_id, int(index))] = cell
    side.update(
        tag=side['tag'] or default_tag, checkpoint_id=result.get('checkpoint_id'), seed=result.get('seed'), cells=cells,
        partial=bool(result.get('partial')), samples=result.get('samples'), planned_samples=result.get('planned_samples'),
        wall_seconds=result.get('wall_seconds'), rollouts_per_min=result.get('rollouts_per_min'),
        replicas=result.get('replicas'),
    )
    return side


def _plan_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[str]:
    """Why the two sides did not run one plan; empty when they did."""
    why: List[str] = []
    if a['seed'] != b['seed']:
        why.append('the seeds differ')
    pa, pb = a['plan'], b['plan']
    if pa and pb and pa.get('plan_sha256') != pb.get('plan_sha256'):
        ea = {e['env_id']: e for e in pa.get('envs') or []}
        eb = {e['env_id']: e for e in pb.get('envs') or []}
        if sorted(ea) != sorted(eb):
            why.append('the environments differ')
        for env_id in sorted(set(ea) & set(eb)):
            for key, what in (('indices_sha256', 'tasks'), ('k', 'rollouts per task'), ('max_tokens', 'output cap')):
                if ea[env_id].get(key) != eb[env_id].get(key):
                    why.append('%s: the %s differ' % (env_id, what))
        if not why:
            why.append('the plans differ (harness or sampling settings)')
    return why


# ------------------------------------------------------------------ the trainer update of each side
def _above_zero(value: Any) -> Optional[float]:
    """``value`` as a number above zero, else None. A box summary has 0 where the trainer report had no norm."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def find_update(doc: Any) -> Mapping[str, Any]:
    """The trainer report in ``doc``, or the summary of one when ``doc`` has no more.

    ``doc`` is ``report.private.json`` of ``train_pit.py``, or a document that holds it (``UPDATE_HOLDERS``): for
    the live trainer, ``TRAINING-RESULT.json`` of the epoch.
    """
    found: List[Mapping[str, Any]] = []
    queue = [doc]
    while queue:
        item = queue.pop(0)
        if isinstance(item, Mapping) and not any(item is seen for seen in found):
            found.append(item)
            queue.extend(item.get(key) for key in UPDATE_HOLDERS)
    # The trainer report has the job of the update. A summary has the norm alone, as ``gradient_norm``.
    for key in ('preclip_gradient_norm', 'gradient_norm'):
        for item in found:
            if _above_zero(item.get(key)) is not None:
                return item
    raise ParityError('the document has no gradient norm of a trainer update')


def update_summary(report: Mapping[str, Any], *, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """What the parity report keeps of one trainer update: the norm, the limit of the trainer, the job."""
    norm = _above_zero(report.get('preclip_gradient_norm')) or _above_zero(report.get('gradient_norm'))
    metrics, config, micro = (
        report.get(key) if isinstance(report.get(key), Mapping) else {} for key in ('epoch_metrics', 'config', 'microbatch'))
    settings = config.get('settings') if isinstance(config.get('settings'), Mapping) else {}
    limit = _above_zero(metrics.get('max_gradient_norm')) or _above_zero(settings.get('max_gradient_norm'))
    out = dict(
        preclip_gradient_norm=norm, max_gradient_norm=limit, above_limit=(norm > limit if limit else None),
        engine=micro.get('engine') or report.get('engine'), ranks=micro.get('world'), status=report.get('status'))
    out.update({key: report.get(key) for key in UPDATE_NUMBERS + UPDATE_JOB})
    start = out['input_checkpoint']
    out['from_this_checkpoint'] = (start == checkpoint_id) if start and checkpoint_id else None
    return out


def track_preclip(doc_a: Any = None, doc_b: Any = None, *, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """The preclip gradient norm of the trainer update of each side, and the ratio B / A.

    ``doc_a`` and ``doc_b`` are documents for ``find_update``; None when a side has no trainer report.

    same_job     True: the two updates start from one checkpoint, with one batch, at one optimizer step.
                 False: one of these differs; a ratio then has no meaning, and there is none.
                 None: the documents do not say.
    differences  each input that differs. For two trainer reports these are the names that
                 ``compare_train_reports`` gives (settings, turn parameters and fp32 master too).
    within_shadow_tolerance
                 the check of the gradient norm in the shadow pass criterion (``shadow_compare``), on this pair.

    The norm is a tracked number. It has no part in the verdict (``in_verdict`` is False).
    """
    reports = [None if doc is None else find_update(doc) for doc in (doc_a, doc_b)]
    a, b = (None if report is None else update_summary(report, checkpoint_id=checkpoint_id) for report in reports)
    out: Dict[str, Any] = dict(
        in_verdict=False, a=a, b=b, same_job=None, differences=[], ratio_b_over_a=None, relative_difference=None,
        shadow_tolerance=GRAD_NORM_REL_TOL, within_shadow_tolerance=None)
    if a is None or b is None:
        return out
    in_both = [key for key in UPDATE_JOB if a[key] is not None and b[key] is not None]
    job_differs = [key for key in in_both if a[key] != b[key]]
    if job_differs:
        out['same_job'] = False
    elif {'input_checkpoint', 'data_sha256'} <= set(in_both):
        out['same_job'] = True
    whole = all(CTR.is_update_report(report) for report in reports)
    out['differences'] = sorted(CTR.input_differences(*reports)) if whole else job_differs
    if out['same_job'] is not False:
        norm_a, norm_b = a['preclip_gradient_norm'], b['preclip_gradient_norm']
        out.update(
            ratio_b_over_a=norm_b / norm_a, relative_difference=abs(norm_b - norm_a) / max(norm_a, norm_b),
            within_shadow_tolerance=gradient_norm_within_tol(dict(gradient_norm=norm_b), dict(gradient_norm=norm_a))[0])
    return out


# ------------------------------------------------------------------ the comparison
def _stratified(pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]], field: str) -> Dict[str, Any]:
    """Shift test of one count field over all tasks: rates of both sides, mean delta, z."""
    total, variance = 0.0, 0.0
    hits_a = hits_b = n_a = n_b = 0
    for ca, cb in pairs:
        ha, hb = int(ca.get(field) or 0), int(cb.get(field) or 0)
        total += hb / cb['n'] - ha / ca['n']
        variance += null_task(ca['n'], cb['n'], ha + hb)[1]
        hits_a, hits_b, n_a, n_b = hits_a + ha, hits_b + hb, n_a + ca['n'], n_b + cb['n']
    return dict(a=hits_a / n_a, b=hits_b / n_b, mean_delta=total / len(pairs), z=_z(total, variance),
                null_se=math.sqrt(variance) / len(pairs))


def compare(doc_a: Mapping[str, Any], doc_b: Mapping[str, Any], *, tag_a: Optional[str] = None,
            tag_b: Optional[str] = None, max_shift: Optional[float] = None, z_crit: float = Z_CRIT,
            tail_alpha: float = TAIL_ALPHA, train_a: Any = None, train_b: Any = None) -> Dict[str, Any]:
    """Compare two sides of one checkpoint. Side A is the reference; a delta is B minus A.

    ``train_a`` and ``train_b`` are the trainer reports of the two sides. The report tracks their preclip
    gradient norm under the key ``preclip`` (see ``track_preclip``). They do not change the verdict.
    """
    a, b = load_side(doc_a, tag=tag_a, default_tag='A'), load_side(doc_b, tag=tag_b, default_tag='B')
    if not a['checkpoint_id'] or a['checkpoint_id'] != b['checkpoint_id']:
        raise ParityError('the sides are not one checkpoint: %s and %s' % (
            str(a['checkpoint_id'])[:12], str(b['checkpoint_id'])[:12]))
    preclip = track_preclip(train_a, train_b, checkpoint_id=a['checkpoint_id'])
    shared = sorted(set(a['cells']) & set(b['cells']))
    if not shared:
        raise ParityError('the sides share no finished task')
    pairs = [(a['cells'][key], b['cells'][key]) for key in shared]
    not_comparable = _plan_differences(a, b)
    if any(ca['n'] != cb['n'] for ca, cb in pairs) and not (a['partial'] or b['partial']):
        not_comparable.append('rollouts per task differ')

    tasks: List[Dict[str, Any]] = []
    disp, disp_var, informative = 0.0, 0.0, 0
    tail_seen, tail_chances = 0, []
    ci_var = 0.0
    same_len = len_pairs = 0
    for (env_id, index), (ca, cb) in zip(shared, pairs):
        na, nb, ha, hb = ca['n'], cb['n'], int(ca.get('c') or 0), int(cb.get('c') or 0)
        delta = hb / nb - ha / na
        _, var, var_sq = null_task(na, nb, ha + hb)
        p = fisher_p(ha, na, hb, nb)
        if var > 0:
            informative += 1
            disp += delta * delta / var
            disp_var += var_sq
        tail_seen += p <= tail_alpha
        tail_chances.append(_tail_chance(na, nb, ha + hb, tail_alpha))
        # Unbiased spread of each side's rate, for the interval of the mean delta.
        ci_var += (ha / na) * (1 - ha / na) / max(na - 1, 1) + (hb / nb) * (1 - hb / nb) / max(nb - 1, 1)
        la, lb = ca.get('L'), cb.get('L')
        if isinstance(la, list) and isinstance(lb, list) and len(la) == len(lb):
            same_len += sum(x == y for x, y in zip(la, lb))
            len_pairs += len(la)
        tasks.append(dict(env_id=env_id, index=index, a=[ha, na], b=[hb, nb], delta=delta, p=p))

    count = len(pairs)
    rates = {name: _stratified(pairs, field) for name, field in COUNTS.items()}
    shift = rates['pass']
    se = math.sqrt(ci_var) / count
    interval = [shift['mean_delta'] - z_crit * se, shift['mean_delta'] + z_crit * se]
    dispersion = dict(
        tasks=informative, ratio=(disp / informative if informative else None),
        z=_sum_of_squares_z(disp, informative, disp_var))
    # A count of rare events is far from normal, so its tail chance is exact.
    tail = dict(alpha=tail_alpha, tasks=tail_seen, expected=sum(tail_chances), p=_count_tail(tail_chances, tail_seen))
    one_sided = 1.0 - NormalDist().cdf(z_crit)

    reasons: List[str] = []
    for name, stat in rates.items():
        if stat['z'] is not None and abs(stat['z']) >= z_crit:
            reasons.append('%s rate: %+.4f (z %+.2f)' % (name.replace('_', ' '), stat['mean_delta'], stat['z']))
    if dispersion['z'] is not None and dispersion['z'] >= z_crit:
        reasons.append('per-task deltas spread %.2f times what sampling gives (z %+.2f)' % (
            dispersion['ratio'], dispersion['z']))
    if tail['p'] <= one_sided:
        reasons.append('%d tasks with Fisher p <= %g; sampling gives %.1f (chance %.2g)' % (
            tail_seen, tail_alpha, tail['expected'], tail['p']))
    different = bool(reasons)

    incomplete = [side['tag'] for side in (a, b) if side['partial']]
    only_a, only_b = len(set(a['cells']) - set(b['cells'])), len(set(b['cells']) - set(a['cells']))
    if not_comparable:
        verdict, reasons = 'not_comparable', not_comparable + reasons
    elif different:
        verdict = 'different'
    elif incomplete or only_a or only_b:
        verdict = 'inconclusive'
        reasons.append('partial run: %s' % (', '.join(incomplete) or 'the task sets differ'))
    elif max_shift is None:
        verdict = 'no_difference_detected'
    elif max(abs(interval[0]), abs(interval[1])) <= max_shift:
        verdict = 'parity'
    else:
        verdict = 'inconclusive'
        reasons.append('the interval of the mean delta is wider than max_shift %g' % max_shift)

    def about(side: Mapping[str, Any]) -> Dict[str, Any]:
        keys = ('tag', 'layout', 'gpus', 'replicas', 'partial', 'samples', 'planned_samples', 'wall_seconds',
                'rollouts_per_min', 'serve_s')
        return dict({k: side[k] for k in keys}, tasks=len(side['cells']),
                    mean_out_tokens=_mean_tokens(side['cells'].values()))

    walls = (a['wall_seconds'], b['wall_seconds'])
    return dict(
        version=REPORT_VERSION, checkpoint_id=a['checkpoint_id'], seed=a['seed'], verdict=verdict, reasons=reasons,
        z_crit=z_crit, max_shift=max_shift, a=about(a), b=about(b),
        tasks_compared=count, tasks_only_a=only_a, tasks_only_b=only_b,
        rates=rates, mean_delta=shift['mean_delta'], mean_delta_se=se, mean_delta_interval=interval,
        # The smallest true shift that this run finds 4 times in 5.
        detectable_shift=(z_crit + 0.84) * shift['null_se'],
        dispersion=dispersion, tail=tail,
        same_length_share=(same_len / len_pairs if len_pairs else None),
        speedup_b_over_a=(walls[0] / walls[1] if all(walls) and not incomplete else None),
        preclip=preclip,
        tasks=sorted(tasks, key=lambda row: (row['p'], -abs(row['delta']), row['env_id'], row['index'])),
    )


def _mean_tokens(cells: Any) -> Optional[float]:
    cells = list(cells)
    samples = sum(cell['n'] for cell in cells)
    return sum(int(cell.get('tok') or 0) for cell in cells) / samples if samples else None


def _render_preclip(preclip: Mapping[str, Any]) -> List[str]:
    """Text lines of the tracked preclip gradient norm."""
    sides = [(tag, preclip.get(key)) for tag, key in (('A', 'a'), ('B', 'b'))]
    title = 'preclip gradient norm of the trainer update (tracked; not in the verdict): '
    if not any(side for _, side in sides):
        return [title + 'no trainer report (give --train-a and --train-b)']

    def one(tag: str, side: Optional[Mapping[str, Any]]) -> str:
        if not side:
            return '%s no trainer report' % tag
        how = [str(side['engine'])] if side.get('engine') else []
        how += ['%s ranks' % side['ranks']] if side.get('ranks') else []
        return '%s %.6g%s' % (tag, side['preclip_gradient_norm'], ' [%s]' % ', '.join(how) if how else '')

    first = title + ', '.join(one(tag, side) for tag, side in sides)
    lines: List[str] = []
    if preclip.get('ratio_b_over_a') is not None:
        first += ', B / A %.4f' % preclip['ratio_b_over_a']
        lines.append('  relative difference %.2f%%, %s the %.2f%% of the shadow pass criterion' % (
            100 * preclip['relative_difference'], 'within' if preclip['within_shadow_tolerance'] else 'above',
            100 * preclip['shadow_tolerance']))
    elif preclip.get('same_job') is False:
        first += '; no ratio: the two updates are not one job'
    if all(side for _, side in sides):
        if preclip.get('same_job') is None:
            lines.append('  the reports do not say if the two updates are one job (one checkpoint, one batch, one step)')
        if preclip.get('differences'):
            lines.append('  inputs of the two updates that differ: %s' % ', '.join(preclip['differences']))
    elsewhere = [tag for tag, side in sides if side and side.get('from_this_checkpoint') is False]
    if elsewhere:
        lines.append('  %s not start from the checkpoint of the two sides' % (
            'the updates of A and B do' if len(elsewhere) == 2 else 'the update of %s does' % elsewhere[0]))
    limits = sorted({side['max_gradient_norm'] for _, side in sides if side and side.get('max_gradient_norm')})
    if limits:
        lines.append('  the trainer limits the norm to %s: %s' % (' / '.join('%g' % limit for limit in limits), ', '.join(
            '%s %s' % (tag, 'above the limit' if side['above_limit'] else 'not above the limit')
            for tag, side in sides if side and side.get('max_gradient_norm'))))
    return [first] + lines


def render(report: Mapping[str, Any], *, top: int = 20) -> str:
    """The report as text: the verdict, the pooled numbers, then the ``top`` tasks that differ most."""
    a, b = report['a'], report['b']

    def z(value: Optional[float]) -> str:
        return 'n/a' if value is None else '%+.2f' % value

    def layout(side: Mapping[str, Any]) -> str:
        lay = side.get('layout') or {}
        if not lay:
            return 'layout not recorded'
        return '%s: %s GPU per replica, vLLM %s, max-num-seqs %s, gmu %s, prefix caching %s' % (
            lay.get('name'), lay.get('gpus_per_replica'), lay.get('vllm_version'), lay.get('max_num_seqs'),
            lay.get('gpu_memory_utilization'), 'on' if lay.get('prefix_caching') else 'off')

    lines = [
        'probe parity of checkpoint %s, seed %s' % (str(report['checkpoint_id'])[:16], report['seed']),
        '  A = %s  [%s]' % (a['tag'], layout(a)),
        '  B = %s  [%s]' % (b['tag'], layout(b)),
        'verdict: %s' % report['verdict'],
    ]
    lines += ['  - %s' % reason for reason in report['reasons']]
    lines.append('tasks compared: %d (only in A: %d, only in B: %d)' % (
        report['tasks_compared'], report['tasks_only_a'], report['tasks_only_b']))
    lines.append('%-12s %9s %9s %9s %7s' % ('rate', 'A', 'B', 'B - A', 'z'))
    for name, stat in report['rates'].items():
        lines.append('%-12s %9.4f %9.4f %+9.4f %7s' % (
            name.replace('_', ' '), stat['a'], stat['b'], stat['mean_delta'], z(stat['z'])))
    lines.append('mean pass-rate delta: %+.4f, interval at z %.1f: [%+.4f, %+.4f]' % (
        report['mean_delta'], report['z_crit'], *report['mean_delta_interval']))
    lines.append('smallest shift this run finds 4 times in 5: %.4f' % report['detectable_shift'])
    disp, tail = report['dispersion'], report['tail']
    lines.append('dispersion: %s times the sampling spread over %d tasks (z %s)' % (
        'n/a' if disp['ratio'] is None else '%.2f' % disp['ratio'], disp['tasks'], z(disp['z'])))
    lines.append('tasks with Fisher p <= %g: %d, sampling gives %.1f (chance of that many: %.2g)' % (
        tail['alpha'], tail['tasks'], tail['expected'], tail['p']))
    if report['same_length_share'] is not None:
        lines.append('rollouts with the same output length on both sides: %.1f%%' % (100 * report['same_length_share']))
    for side in (a, b):
        if side.get('wall_seconds'):
            lines.append('%s: %.0f s probe wall, %s replicas, mean output %.0f tokens' % (
                side['tag'], side['wall_seconds'], side.get('replicas'), side.get('mean_out_tokens') or 0))
    if report['speedup_b_over_a']:
        lines.append('B is %.2f times as fast as A' % report['speedup_b_over_a'])
    lines += _render_preclip(report.get('preclip') or {})
    lines.append('per-task pass rates, largest differences first (%d of %d):' % (
        min(top, len(report['tasks'])), len(report['tasks'])))
    lines.append('  %-18s %7s %7s %7s %8s %9s' % ('env', 'task', 'A', 'B', 'B - A', 'Fisher p'))
    for row in report['tasks'][:top]:
        lines.append('  %-18s %7d %7s %7s %+8.3f %9.2g' % (
            row['env_id'][:18], row['index'], '%d/%d' % tuple(row['a']), '%d/%d' % tuple(row['b']), row['delta'], row['p']))
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ the plan and the box runs
def draw_indices(pool_size: int, tasks: int, seed: str) -> List[int]:
    """``tasks`` different task indices below ``pool_size``; the same seed gives the same tasks."""
    if not 0 < tasks <= pool_size:
        raise ParityError('cannot draw %d tasks from a pool of %d' % (tasks, pool_size))
    return sorted(random.Random('probe-parity:%s' % seed).sample(range(pool_size), tasks))


def build_plan(checkpoint: str, indices: Sequence[int], seed: str, *, harness: Mapping[str, Any],
               k: int = GATE_K_CAND, max_tokens: int = GATE_MAX_TOKENS, env_id: str = 'affine_gatepool') -> Dict[str, Any]:
    """The probe plan of both sides, from the builder that the gate uses."""
    if not isinstance(harness, Mapping) or 'harness' not in harness:
        raise ParityError('the public harness is required: without it the worker reads a file that only live boxes have')
    return build_probe_plan(checkpoint, [int(i) for i in indices], seed, k=k, max_tokens=max_tokens,
                            env_id=env_id, run_name='parity-%s-%s' % (checkpoint[:12], seed), harness=dict(harness))


def run_side(pod: Any, *, checkpoint: str, plan: Mapping[str, Any], layout: Any, work_dir: str,
             tag: Optional[str] = None, root: str = PL.BOX_ROOT, box_args: Sequence[str] = (),
             poll_s: float = 30.0, timeout_s: Optional[float] = None,
             sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """Run one side on a box and return its record. ``pod.run`` runs a shell command on the box.

    The side runs detached on the box, so a lost connection does not end it.
    """
    layout = PL.layout_from(layout)
    tag = tag or layout.name
    store = RemoteBoxJobStore(pod)
    side_dir = '%s/%s' % (work_dir.rstrip('/'), tag)
    store.ensure_dir(side_dir)
    tool = PS.ship(store.write_text, side_dir, PL.ProbePlan((layout,)))['probe_serve.py']
    plan_path, layout_path, out_path = ('%s/%s' % (side_dir, name) for name in ('plan.json', 'layout.json', 'side.json'))
    store.write_text(plan_path, json.dumps(dict(plan), sort_keys=True) + '\n')
    store.write_text(layout_path, json.dumps(layout.as_dict(), sort_keys=True) + '\n')
    words = ['python3', tool, '--root', root, *box_args, 'side', '--checkpoint', checkpoint, '--plan', plan_path,
             '--layout', layout_path, '--out', out_path, '--tag', tag]
    pid = store.launch_detached(side_dir, stage='parity-side', command=' '.join(shlex.quote(w) for w in words))
    deadline = time.time() + (timeout_s or PL.ProbePlan((layout,)).eval_timeout_s())
    while True:
        running = store.pid_alive(pid)
        text = store.read_text(out_path)
        if text:
            return json.loads(text)
        if not running:
            raise ParityError('side %s ended without a record: %s' % (
                tag, (store.tail_text('%s/job.log' % side_dir, 20) or '').strip()[-600:]))
        if time.time() >= deadline:
            raise ParityError('side %s did not finish in time; it still runs on the box' % tag)
        sleep(poll_s)


# ------------------------------------------------------------------ command line
def _read(path: str) -> Any:
    return json.loads(Path(path).read_text())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    pl = sub.add_parser('plan', help='write the probe plan that both sides run')
    pl.add_argument('--checkpoint', required=True, help='id of the checkpoint (HEAD) that both sides serve')
    pl.add_argument('--seed', required=True, help='plan seed; every request seed derives from it')
    pl.add_argument('--indices', help='JSON file with the list of task indices')
    pl.add_argument('--tasks', type=int, help='number of tasks to draw when --indices is not given')
    pl.add_argument('--pool-size', type=int, help='the draw takes task indices below this number')
    pl.add_argument('--k', type=int, default=GATE_K_CAND, help='rollouts per task')
    pl.add_argument('--max-tokens', type=int, default=GATE_MAX_TOKENS, help='output cap of one rollout')
    pl.add_argument('--env', default='affine_gatepool', help='environment id on the box')
    pl.add_argument('--harness', help='public_harness.json; default is the harness of this host')
    pl.add_argument('--out', required=True)

    rs = sub.add_parser('run-side', help='run one layout on one box and bring back the side record; '
                                         'the checkpoint must be on the box under <root>/ckpt/<id>')
    rs.add_argument('--pod', required=True, help='JSON file with the box handle (the rent record: id, ip, port)')
    rs.add_argument('--plan', required=True)
    rs.add_argument('--layout', required=True, help='a preset name (%s) or a JSON file with layout settings' % ', '.join(sorted(PL.PRESETS)))
    rs.add_argument('--tag', help='name of this side; default is the layout name')
    rs.add_argument('--work-dir', default='/workspace/pst/probe-parity', help='directory on the box')
    rs.add_argument('--out', required=True, help='where the side record goes on this host')

    cp = sub.add_parser('compare', help='compare two side records')
    cp.add_argument('a', help='side record or probe result of the reference layout')
    cp.add_argument('b', help='side record or probe result of the layout under test')
    cp.add_argument('--max-shift', type=float, help='largest mean pass-rate shift that still counts as parity, as a fraction')
    cp.add_argument('--z-crit', type=float, default=Z_CRIT, help='a test beyond this z says the layouts differ')
    cp.add_argument('--top', type=int, default=20, help='tasks in the text report')
    cp.add_argument('--out', help='write the full report (every task) as JSON')
    cp.add_argument('--train-a', help='trainer report of the update that the trainer of side A made: report.private.json '
                                      'of train_pit.py, or a document that holds it (TRAINING-RESULT.json of a live '
                                      'epoch). The report then tracks the gradient norm before clipping of each side; '
                                      'the verdict does not change')
    cp.add_argument('--train-b', help='the same for side B. The two updates must be of one batch')

    args = ap.parse_args(argv)
    try:
        if args.cmd == 'plan':
            if args.indices:
                indices = _read(args.indices)
            elif args.tasks and args.pool_size:
                indices = draw_indices(args.pool_size, args.tasks, args.seed)
            else:
                raise ParityError('give --indices, or --tasks with --pool-size')
            harness = _read(args.harness) if args.harness else _load_public_harness()
            plan = build_plan(args.checkpoint, indices, args.seed, harness=harness, k=args.k,
                              max_tokens=args.max_tokens, env_id=args.env)
            Path(args.out).write_text(json.dumps(plan, sort_keys=True) + '\n')
            print(json.dumps(dict(out=args.out, plan=PS.plan_identity(plan))))
            return 0
        if args.cmd == 'run-side':
            pod = normalize_meta_pod(_read(args.pod))
            if pod is None:
                raise ParityError('%s is not a usable box handle' % args.pod)
            layout = _read(args.layout) if Path(args.layout).is_file() else args.layout
            record = run_side(pod, checkpoint=_read(args.plan)['checkpoint_id'], plan=_read(args.plan), layout=layout,
                              work_dir=args.work_dir, tag=args.tag)
            Path(args.out).write_text(json.dumps(record, sort_keys=True) + '\n')
            print(json.dumps(dict(out=args.out, status=record['status'], samples=record['result'].get('samples'),
                                  partial=record['result'].get('partial'))))
            return 0
        report = compare(_read(args.a), _read(args.b), max_shift=args.max_shift, z_crit=args.z_crit,
                         train_a=_read(args.train_a) if args.train_a else None,
                         train_b=_read(args.train_b) if args.train_b else None)
        if args.out:
            Path(args.out).write_text(json.dumps(report, sort_keys=True) + '\n')
        sys.stdout.write(render(report, top=args.top))
        return EXIT[report['verdict']]
    except (ParityError, PL.LayoutError, PS.ServeError) as exc:
        print('probe parity: %s' % exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
