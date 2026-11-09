"""PIT scoring seam (BRIEF §1.8 / DESIGN §5).

Wraps the prior-v1 estimator path. Host RA ``pe_score.py`` is preferred when
importable; this module provides the git-side API including
``score_epoch_shadow``, which **must** record the box's own candidate hash —
never the live epoch's ``candidate`` field.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Mapping, Optional


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
        out = dict(host_score(
            epoch=epoch, round_id=round_id, result=result,
            candidate_id=candidate_id, head_id=head_id,
        ))
    elif estimate_fn is not None:
        out = dict(estimate_fn(result=result, candidate_id=candidate_id, head_id=head_id))
    else:
        embedded = result.get('score')
        if require_host:
            raise ScoreError('live scoring refuses embedded result.score without host scorer')
        if not isinstance(embedded, Mapping):
            raise ScoreError('no scorer available and result.score missing')
        out = dict(embedded)
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
) -> Dict[str, Any]:
    """Shadow compare: score the **box** candidate and diff against live.

    Critical fix: the scored/recorded candidate hash is ``box_candidate_id``,
    not ``live_decision['candidate']``. Using the live epoch candidate here
    would attribute the wrong model to the PIT shadow run.
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
    comparison = {
        'epoch': epoch,
        'box_candidate': box_candidate_id,
        'live_candidate': live_cand,
        'candidates_differ': bool(live_cand) and live_cand != box_candidate_id,
        'box_decision': scored.get('decision'),
        'live_decision': live.get('decision'),
        'box_z': scored.get('z'),
        'live_z': live.get('z') or (live.get('evaluated') or {}).get('z'),
        'agreement': None,
    }
    if live.get('decision') and scored.get('decision') in ('green', 'red'):
        # Normalize live green → comparable label
        live_dec = str(live.get('decision')).lower()
        box_dec = str(scored.get('decision')).lower()
        comparison['agreement'] = (live_dec == box_dec) or (
            live_dec == 'green' and box_dec == 'green'
        )
        bz = comparison['box_z']
        lz = comparison['live_z']
        if bz is not None and lz is not None:
            comparison['delta_z'] = float(bz) - float(lz)

    return {
        'shadow': True,
        'scored': scored,
        'comparison': comparison,
        # Explicit top-level field so callers/logs cannot accidentally read live.
        'candidate': box_candidate_id,
    }
