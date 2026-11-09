"""PIT scoring seam (BRIEF §1.8 / DESIGN §5).

Wraps the prior-v1 estimator path. Host RA ``pe_score.py`` is preferred when
importable; this module provides the git-side API including
``score_epoch_shadow``, which **must** record the box's own candidate hash —
never the live epoch's ``candidate`` field.

Live host signature (validator ``pe_score.score``):
  ``score(epoch, rid, result, flag=None)`` — positional; returns flat
  prior-v1 fields. PIT adapts that to nest ``heldout`` for gate emit.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Mapping, Optional

from .shadow_compare import evaluate_shadow_pass

# Flat fields live pe_score.score puts on the decision; gate emit needs them
# nested under heldout (PR96-REVIEW-2 blocker 1 / residual blocker 8).
_HELDOUT_FLAT_KEYS = (
    'set_size',
    'bar_z',
    'k_cand',
    'head_rollouts',
    'scoring',
    'settings_hash',
    'baseline',
    'max_tokens',
    'scoring_mode',
    'tau_run',
    'sigma_run_floor',
    'baseline_done_at_cr',
    'round_id',
)


class ScoreError(ValueError):
    """Fail-closed scoring error."""


def _hex64(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ScoreError('%s must be hex64' % field)
    if any(c not in '0123456789abcdef' for c in value):
        raise ScoreError('%s must be lowercase hex' % field)
    return value


def load_host_pe_score(module_path: Optional[str] = None):
    """Best-effort import of host RA pe_score. Returns None if unavailable."""
    candidates = [module_path] if module_path else []
    candidates.extend([
        'pe_score',
        'ops.parallel_eval.live.pe_score',
    ])
    for name in candidates:
        if not name:
            continue
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _evaluated_of(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    ev = raw.get('evaluated')
    return ev if isinstance(ev, Mapping) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def nest_heldout_fields(raw: Mapping[str, Any], *, round_id: str) -> Dict[str, Any]:
    """Copy live score fields into ``heldout`` for gate DECISION emit.

    Real live ``pe_score.score`` (BV.estimate + flat update) shape:
      top-level: decision, reasons, evaluated, z_bar, round_id, set_size,
      k_cand, head_rollouts, scoring, settings_hash, max_tokens, tau_run,
      baseline_done_at_cr, plan_seed, …
      evaluated: z, n, tau_run, sigma_run_floor, scoring, …
    ``sigma_run_floor`` is **not** top-level — only under ``evaluated``.
    ``baseline`` / ``scoring_mode`` are also absent at top-level (gate hook
    stamps them when writing DECISION; PIT must fill them here).
    """
    out = dict(raw)
    heldout = dict(out.get('heldout') or {})
    evaluated = _evaluated_of(out)

    for key in _HELDOUT_FLAT_KEYS:
        if key in heldout and heldout[key] is not None:
            continue

        if key == 'round_id':
            heldout[key] = _first_present(
                round_id or None, out.get('round_id'), heldout.get('round_id'),
            )
            continue

        if key == 'bar_z':
            zb = out.get('z_bar')
            zb_value = zb.get('value') if isinstance(zb, Mapping) else (
                zb if zb is not None and not isinstance(zb, Mapping) else None
            )
            heldout[key] = _first_present(
                out.get('bar_z'), zb_value, out.get('z_min'),
            )
            continue

        if key == 'baseline':
            # Live pe_score never stamps this; gate hook hardcodes prior-v1.
            heldout[key] = _first_present(out.get('baseline'), 'prior-v1')
            continue

        if key == 'scoring_mode':
            # Live pe_score omits this; PE refuses same-run so default separate.
            sm = out.get('scoring_mode')
            if sm is None and isinstance(out.get('probe'), Mapping):
                sm = out['probe'].get('scoring_mode')
            heldout[key] = _first_present(sm, 'separate')
            continue

        if key == 'sigma_run_floor':
            # Real live return: only under evaluated (not top-level).
            heldout[key] = _first_present(
                out.get('sigma_run_floor'),
                evaluated.get('sigma_run_floor'),
                0.0,
            )
            continue

        if key == 'tau_run':
            # Top-level from pe_score.update; also mirrored under evaluated.
            heldout[key] = _first_present(
                out.get('tau_run'),
                evaluated.get('tau_run'),
                0.0,
            )
            continue

        if key == 'scoring':
            heldout[key] = _first_present(
                out.get('scoring'),
                evaluated.get('scoring'),
            )
            continue

        # Remaining flat keys (set_size, k_cand, head_rollouts, settings_hash,
        # max_tokens, baseline_done_at_cr) are top-level on the live return.
        # Fall back to evaluated only if a future live shape moves them.
        heldout[key] = _first_present(out.get(key), evaluated.get(key))

    if round_id and not heldout.get('round_id'):
        heldout['round_id'] = round_id
    out['heldout'] = heldout
    # decision=None from live probe-error guard → no_decision (never green).
    dec = out.get('decision')
    if dec is None or dec == '':
        out['decision'] = 'no_decision'
    return out


def call_live_host_score(
    host_score: Callable[..., Mapping[str, Any]],
    *,
    epoch: int,
    round_id: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Invoke live ``score(epoch, rid, result)`` (positional) and nest heldout.

    Falls back to keyword ``round_id=`` only for non-live test doubles that
    declare that parameter — never pass ``round_id=`` to the live module.
    """
    rid = str(round_id)
    try:
        params = inspect.signature(host_score).parameters
    except (TypeError, ValueError):
        params = {}
    # Live: (epoch, rid, result, flag=None) — call positionally.
    if 'rid' in params or (list(params)[:2] == ['epoch', 'rid']):
        raw = host_score(int(epoch), rid, result)
    elif 'round_id' in params:
        # PIT/test doubles that accept round_id=
        raw = host_score(epoch=int(epoch), round_id=rid, result=result)
    else:
        # Last resort: positional live shape (also covers builtins / wraps).
        try:
            raw = host_score(int(epoch), rid, result)
        except TypeError:
            raw = host_score(epoch=int(epoch), round_id=rid, result=result)
    if not isinstance(raw, Mapping):
        raise ScoreError('host score must return a mapping')
    return nest_heldout_fields(raw, round_id=rid)


def score(
    *,
    epoch: int,
    round_id: str,
    result: Mapping[str, Any],
    candidate_id: str,
    head_id: str,
    estimate_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    host_score: Optional[Callable[..., Mapping[str, Any]]] = None,
    require_host: bool = False,
) -> Dict[str, Any]:
    """Score one PIT probe result.

    ``candidate_id`` is the box-produced candidate and is copied into the
    returned decision verbatim. Callers must not pass the live epoch candidate.
    """
    candidate_id = _hex64(candidate_id, 'candidate_id')
    head_id = _hex64(head_id, 'head_id')
    if host_score is not None:
        out = call_live_host_score(
            host_score, epoch=epoch, round_id=round_id, result=result,
        )
    elif estimate_fn is not None:
        out = nest_heldout_fields(
            dict(estimate_fn(result=result, candidate_id=candidate_id, head_id=head_id)),
            round_id=round_id,
        )
    else:
        embedded = result.get('score')
        if require_host:
            raise ScoreError('live scoring refuses embedded result.score without host scorer')
        if not isinstance(embedded, Mapping):
            raise ScoreError('no scorer available and result.score missing')
        out = nest_heldout_fields(dict(embedded), round_id=round_id)
    decision = str(out.get('decision') or '').lower()
    if decision not in ('green', 'red', 'pending', 'no_decision'):
        raise ScoreError('scorer returned invalid decision')
    # HARD: recorded candidate is always the box candidate.
    out['candidate'] = candidate_id
    out['candidate_id'] = candidate_id
    out['head'] = head_id
    out['epoch'] = int(epoch)
    out['round_id'] = round_id
    out['source'] = 'pit'
    # Promote evaluated.z to top-level z when live estimate nests it.
    if out.get('z') is None and isinstance(out.get('evaluated'), Mapping):
        if out['evaluated'].get('z') is not None:
            out['z'] = out['evaluated']['z']
    return out


def score_epoch_shadow(
    *,
    epoch: int,
    round_id: str,
    box_result: Mapping[str, Any],
    box_candidate_id: str,
    box_head_id: str,
    live_decision: Optional[Mapping[str, Any]] = None,
    estimate_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    host_score: Optional[Callable[..., Mapping[str, Any]]] = None,
    box_report: Optional[Mapping[str, Any]] = None,
    live_report: Optional[Mapping[str, Any]] = None,
    box_meta: Optional[Mapping[str, Any]] = None,
    live_meta: Optional[Mapping[str, Any]] = None,
    train_gpu: Optional[str] = None,
    eval_gpu: Optional[str] = None,
) -> Dict[str, Any]:
    """Shadow compare: score the **box** candidate and diff against live.

    Critical fix: the scored/recorded candidate hash is ``box_candidate_id``,
    not ``live_decision['candidate']``. Using the live epoch candidate here
    would attribute the wrong model to the PIT shadow run.

    Pass criterion (E1041 noise-tolerant): equal input hashes, exact rollouts/
    tokens, update size within 1e-3 rel, grad norm within 5%, same draw/seed/
    head baseline, disagreement ratio in [0.8, 1.25], |Δz| < 2.0. Equal
    candidate hashes / equal z are **not** required.
    """
    box_candidate_id = _hex64(box_candidate_id, 'box_candidate_id')
    scored = score(
        epoch=epoch,
        round_id=round_id,
        result=box_result,
        candidate_id=box_candidate_id,
        head_id=box_head_id,
        estimate_fn=estimate_fn,
        host_score=host_score,
    )
    if scored.get('candidate') != box_candidate_id:
        raise ScoreError('score_epoch_shadow refused non-box candidate hash')

    live = dict(live_decision or {})
    live_cand = live.get('candidate') or live.get('candidate_id')
    box_z = scored.get('z')
    live_z = live.get('z') or (live.get('evaluated') or {}).get('z')
    train_gpu = train_gpu or (box_result or {}).get('train_gpu') or (box_meta or {}).get('train_gpu')
    eval_gpu = eval_gpu or (box_result or {}).get('eval_gpu') or (box_meta or {}).get('eval_gpu')
    live_train_gpu = (live_meta or {}).get('train_gpu') or live.get('train_gpu')
    live_eval_gpu = (live_meta or {}).get('eval_gpu') or live.get('eval_gpu')

    box_pt = ((box_result.get('envs') or {}).get('affine_gatepool') or {}).get('per_task')
    if box_pt is None and isinstance(box_result.get('per_task'), Mapping):
        box_pt = box_result.get('per_task')
    live_pt = live.get('per_task') or ((live.get('probe') or {}).get('per_task') if isinstance(live.get('probe'), Mapping) else None)
    if live_pt is None and isinstance(live.get('envs'), Mapping):
        live_pt = ((live.get('envs') or {}).get('affine_gatepool') or {}).get('per_task')

    b_meta = dict(box_meta or {})
    b_meta.setdefault('round_id', round_id)
    if scored.get('heldout'):
        b_meta.setdefault('heldout', scored.get('heldout'))
        b_meta.setdefault('plan_seed', (scored.get('heldout') or {}).get('plan_seed') or scored.get('plan_seed'))
    l_meta = dict(live_meta or {})
    if live.get('heldout'):
        l_meta.setdefault('heldout', live.get('heldout'))
        l_meta.setdefault('round_id', (live.get('heldout') or {}).get('round_id') or live.get('round_id'))
        l_meta.setdefault('plan_seed', live.get('plan_seed') or (live.get('heldout') or {}).get('plan_seed'))

    verdict = evaluate_shadow_pass(
        box_report=box_report or (box_meta or {}).get('training_report') or {},
        live_report=live_report or live.get('training_report') or {},
        box_meta=b_meta,
        live_meta=l_meta,
        box_per_task=box_pt if isinstance(box_pt, Mapping) else {},
        live_per_task=live_pt if isinstance(live_pt, Mapping) else {},
        box_z=box_z,
        live_z=live_z,
        train_gpu=train_gpu,
        eval_gpu=eval_gpu,
        live_train_gpu=live_train_gpu,
        live_eval_gpu=live_eval_gpu,
    )

    comparison = {
        'epoch': epoch,
        'box_candidate': box_candidate_id,
        'live_candidate': live_cand,
        'candidates_differ': bool(live_cand) and live_cand != box_candidate_id,
        'box_decision': scored.get('decision'),
        'live_decision': live.get('decision'),
        'box_z': box_z,
        'live_z': live_z,
        'delta_z': (
            float(box_z) - float(live_z)
            if box_z is not None and live_z is not None else None
        ),
        'train_gpu': train_gpu,
        'eval_gpu': eval_gpu,
        'live_train_gpu': live_train_gpu,
        'live_eval_gpu': live_eval_gpu,
        'agreement': verdict.get('passed'),
        'shadow_pass': verdict,
    }
    if live.get('decision') and scored.get('decision') in ('green', 'red'):
        live_dec = str(live.get('decision')).lower()
        box_dec = str(scored.get('decision')).lower()
        comparison['decision_label_match'] = (live_dec == box_dec) or (
            live_dec == 'green' and box_dec == 'green'
        )

    return {
        'shadow': True,
        'scored': scored,
        'comparison': comparison,
        'candidate': box_candidate_id,
        'shadow_pass': verdict,
    }
