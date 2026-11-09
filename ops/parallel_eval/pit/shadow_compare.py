"""Shadow SCORE-vs-live pass criteria (Jacob 2026-11-09, post-E1041).

H200 (live) vs B300 (PIT) + nondeterministic all-reduce means candidate hashes
and z are not required to match. Pass when:

* equal input hashes
* exact rollouts and tokens
* update size within 1e-3 relative
* gradient norm within 5%
* same draw, seed and head baseline
* per-task disagreement ratio in [0.8, 1.25]
* |z_box - z_live| < 2.0
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

UPDATE_SIZE_REL_TOL = 1e-3
GRAD_NORM_REL_TOL = 0.05
DISAGREE_RATIO_LO = 0.8
DISAGREE_RATIO_HI = 1.25
Z_GAP_MAX = 2.0


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rel_close(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None:
        return False
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= tol


def _as_hash_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, str] = {}
    for k, v in value.items():
        if isinstance(v, str) and len(v) >= 8:
            out[str(k)] = v
        elif isinstance(v, Mapping):
            # nested document digests
            nested = v.get('sha256') or v.get('digest') or v.get('id')
            if isinstance(nested, str):
                out[str(k)] = nested
    return out


def equal_input_hashes(
    box_report: Mapping[str, Any],
    live_report: Mapping[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    box_h = _as_hash_map(box_report.get('source_file_hashes') or box_report.get('file_summary'))
    live_h = _as_hash_map(live_report.get('source_file_hashes') or live_report.get('file_summary'))
    # Compare intersection of keys; require non-empty overlap and exact match.
    keys = sorted(set(box_h) & set(live_h))
    if not keys:
        # Fall back to whole-map equality when both sides use the same schema.
        ok = bool(box_h) and box_h == live_h
        return ok, {'box': box_h, 'live': live_h, 'shared_keys': keys, 'ok': ok}
    mismatches = {k: (box_h[k], live_h[k]) for k in keys if box_h[k] != live_h[k]}
    ok = not mismatches
    return ok, {'shared_keys': keys, 'mismatches': mismatches, 'ok': ok}


def exact_rollouts_and_tokens(
    box_report: Mapping[str, Any],
    live_report: Mapping[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    def _pack(rep: Mapping[str, Any]) -> Dict[str, Any]:
        batching = rep.get('batching') if isinstance(rep.get('batching'), Mapping) else {}
        tokens = rep.get('tokens') if isinstance(rep.get('tokens'), Mapping) else {}
        fs = rep.get('file_summary') if isinstance(rep.get('file_summary'), Mapping) else {}
        return {
            'tasks': batching.get('tasks') or fs.get('tasks'),
            'rollouts': batching.get('rollouts') or fs.get('rollouts'),
            'groups': batching.get('groups') or fs.get('groups'),
            'prompt_tokens': tokens.get('prompt_tokens') or fs.get('prompt_tokens'),
            'output_tokens': tokens.get('output_tokens') or fs.get('output_tokens'),
            'loss_tokens': tokens.get('loss_tokens') or fs.get('loss_tokens'),
            'total_tokens': tokens.get('total_tokens') or fs.get('total_tokens'),
        }

    box_p, live_p = _pack(box_report), _pack(live_report)
    keys = [k for k in box_p if box_p[k] is not None and live_p.get(k) is not None]
    mismatches = {k: (box_p[k], live_p[k]) for k in keys if box_p[k] != live_p[k]}
    ok = bool(keys) and not mismatches
    return ok, {'box': box_p, 'live': live_p, 'compared_keys': keys, 'mismatches': mismatches, 'ok': ok}


def update_size_within_tol(
    box_report: Mapping[str, Any],
    live_report: Mapping[str, Any],
    *,
    tol: float = UPDATE_SIZE_REL_TOL,
) -> Tuple[bool, Dict[str, Any]]:
    b = _num(box_report.get('update_size'))
    if b is None:
        b = _num(((box_report.get('sgd') or {}) if isinstance(box_report.get('sgd'), Mapping) else {}).get('raw_update_norm'))
    l = _num(live_report.get('update_size'))
    if l is None:
        l = _num(((live_report.get('sgd') or {}) if isinstance(live_report.get('sgd'), Mapping) else {}).get('raw_update_norm'))
    ok = _rel_close(b, l, tol)
    return ok, {'box': b, 'live': l, 'tol': tol, 'ok': ok}


def gradient_norm_within_tol(
    box_report: Mapping[str, Any],
    live_report: Mapping[str, Any],
    *,
    tol: float = GRAD_NORM_REL_TOL,
) -> Tuple[bool, Dict[str, Any]]:
    b = _num(box_report.get('gradient_norm') or box_report.get('preclip_gradient_norm'))
    l = _num(live_report.get('gradient_norm') or live_report.get('preclip_gradient_norm'))
    ok = _rel_close(b, l, tol)
    return ok, {'box': b, 'live': l, 'tol': tol, 'ok': ok}


def same_draw_seed_baseline(
    box_meta: Mapping[str, Any],
    live_meta: Mapping[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    checks: Dict[str, Any] = {}
    for key in ('round_id', 'plan_seed', 'draw_commitment', 'head_baseline', 'baseline_settings_hash'):
        bv, lv = box_meta.get(key), live_meta.get(key)
        if bv is None and lv is None:
            continue
        checks[key] = {'box': bv, 'live': lv, 'ok': bv == lv and bv is not None}
    # Fill missing sides from heldout nesting (both sides independently).
    for side, meta in (('box', box_meta), ('live', live_meta)):
        held = meta.get('heldout') if isinstance(meta.get('heldout'), Mapping) else {}
        if held.get('round_id'):
            row = checks.setdefault('round_id', {})
            if row.get(side) is None:
                row[side] = held.get('round_id')
        if held.get('settings_hash'):
            row = checks.setdefault('baseline_settings_hash', {})
            if row.get(side) is None:
                row[side] = held.get('settings_hash')
        if held.get('plan_seed'):
            row = checks.setdefault('plan_seed', {})
            if row.get(side) is None:
                row[side] = held.get('plan_seed')
    # Recompute ok flags after heldout fill.
    for row in checks.values():
        row['ok'] = row.get('box') is not None and row.get('box') == row.get('live')
    ok = bool(checks) and all(r.get('ok') for r in checks.values())
    return ok, {'checks': checks, 'ok': ok}


def _task_pass_rate(cell: Mapping[str, Any]) -> Optional[float]:
    n = cell.get('n')
    c = cell.get('c')
    if n is None or c is None:
        return None
    try:
        n_f, c_f = float(n), float(c)
    except (TypeError, ValueError):
        return None
    if n_f <= 0:
        return None
    return c_f / n_f


def per_task_disagreement_ratio(
    box_per_task: Mapping[str, Any],
    live_per_task: Mapping[str, Any],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Directional disagreement symmetry: box_only_wins / live_only_wins.

    A value near 1.0 means disagreements are balanced (same-model eval noise).
    """
    if not isinstance(box_per_task, Mapping) or not isinstance(live_per_task, Mapping):
        return None, {'ok': False, 'why': 'missing-per-task'}
    box_only = live_only = 0
    shared = 0
    for key in set(box_per_task) & set(live_per_task):
        b_cell = box_per_task.get(key) or {}
        l_cell = live_per_task.get(key) or {}
        if not isinstance(b_cell, Mapping) or not isinstance(l_cell, Mapping):
            continue
        br, lr = _task_pass_rate(b_cell), _task_pass_rate(l_cell)
        if br is None or lr is None:
            continue
        shared += 1
        if br > lr + 1e-12:
            box_only += 1
        elif lr > br + 1e-12:
            live_only += 1
    if shared == 0:
        return None, {'ok': False, 'why': 'no-shared-tasks'}
    if live_only == 0 and box_only == 0:
        ratio = 1.0
    elif live_only == 0:
        ratio = float('inf')
    else:
        ratio = float(box_only) / float(live_only)
    ok = DISAGREE_RATIO_LO <= ratio <= DISAGREE_RATIO_HI
    return ratio, {
        'box_only_wins': box_only,
        'live_only_wins': live_only,
        'shared_tasks': shared,
        'ratio': ratio,
        'ok': ok,
        'lo': DISAGREE_RATIO_LO,
        'hi': DISAGREE_RATIO_HI,
    }


def z_gap_under(
    box_z: Any,
    live_z: Any,
    *,
    max_gap: float = Z_GAP_MAX,
) -> Tuple[bool, Dict[str, Any]]:
    bz, lz = _num(box_z), _num(live_z)
    if bz is None or lz is None:
        return False, {'box_z': bz, 'live_z': lz, 'gap': None, 'ok': False}
    gap = abs(bz - lz)
    ok = gap < max_gap
    return ok, {'box_z': bz, 'live_z': lz, 'gap': gap, 'max_gap': max_gap, 'ok': ok}


def evaluate_shadow_pass(
    *,
    box_report: Optional[Mapping[str, Any]] = None,
    live_report: Optional[Mapping[str, Any]] = None,
    box_meta: Optional[Mapping[str, Any]] = None,
    live_meta: Optional[Mapping[str, Any]] = None,
    box_per_task: Optional[Mapping[str, Any]] = None,
    live_per_task: Optional[Mapping[str, Any]] = None,
    box_z: Any = None,
    live_z: Any = None,
    train_gpu: Optional[str] = None,
    eval_gpu: Optional[str] = None,
    live_train_gpu: Optional[str] = None,
    live_eval_gpu: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a SCORE-vs-live verdict. Does **not** require equal candidate hashes."""
    box_report = dict(box_report or {})
    live_report = dict(live_report or {})
    box_meta = dict(box_meta or {})
    live_meta = dict(live_meta or {})

    checks: Dict[str, Any] = {}
    ok_input, checks['input_hashes'] = equal_input_hashes(box_report, live_report)
    ok_roll, checks['rollouts_tokens'] = exact_rollouts_and_tokens(box_report, live_report)
    ok_upd, checks['update_size'] = update_size_within_tol(box_report, live_report)
    ok_grad, checks['gradient_norm'] = gradient_norm_within_tol(box_report, live_report)
    ok_draw, checks['draw_seed_baseline'] = same_draw_seed_baseline(box_meta, live_meta)
    ratio, checks['disagreement_ratio'] = per_task_disagreement_ratio(
        box_per_task or {}, live_per_task or {},
    )
    ok_z, checks['z_gap'] = z_gap_under(box_z, live_z)

    required = (
        ok_input, ok_roll, ok_upd, ok_grad, ok_draw,
        bool(checks['disagreement_ratio'].get('ok')), ok_z,
    )
    passed = all(required)
    return dict(
        pass_criterion='e1041-noise-tolerant-v1',
        passed=passed,
        # Explicitly NOT requiring equal candidate hashes / equal z.
        require_equal_candidate_hashes=False,
        require_equal_z=False,
        checks=checks,
        disagreement_ratio=ratio,
        train_gpu=train_gpu,
        eval_gpu=eval_gpu,
        live_train_gpu=live_train_gpu,
        live_eval_gpu=live_eval_gpu,
    )
