#!/usr/bin/env python3
"""Host round-seed / plan-seed provider for PIT (PR #96).

Derives the candidate plan seed exactly like ``baseline_v1.cand_seed``:

    sha256(f\"{secret}:prior-baseline:cand:{epoch}\")

Never logs or returns the raw round seed. Fail-closed if SAMPLE / ROUND /
seed are missing. Also exposes the host-private task ``indices`` list for
``pe_lane.run_probe`` (never persist indices on ITEM.json).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


class RoundSeedError(RuntimeError):
    """Fail-closed: round seed / indices unavailable."""


def _rj(path: Path, default=None):
    try:
        return __import__('json').loads(path.read_text())
    except Exception:
        return default


def _gate_epochs() -> Path:
    env = os.environ.get('PE_GATE_EP')
    if env:
        p = Path(env)
        return p if p.name == 'epochs' else p / 'epochs'
    gate = os.environ.get('HG_GATE_STATE')
    if gate:
        return Path(gate) / 'epochs'
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261026-promotion-gate/state/epochs'
    )


def _heldout_root() -> Path:
    env = os.environ.get('PE_HELDOUT')
    if env:
        return Path(env)
    gate = os.environ.get('HG_GATE_STATE')
    if gate:
        return Path(gate) / 'heldout'
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261026-promotion-gate/state/heldout'
    )


def _cand_seed(secret: str, epoch: int) -> str:
    """Exact ``baseline_v1.cand_seed`` formula (no import of gate code required)."""
    if not isinstance(secret, str) or not secret:
        raise RoundSeedError('round secret seed missing')
    if type(epoch) is not int or epoch < 0:
        raise RoundSeedError('epoch must be non-negative int')
    return hashlib.sha256(
        ('%s:prior-baseline:cand:%d' % (secret, int(epoch))).encode()
    ).hexdigest()


def _sample_for_epoch(epoch: int) -> Mapping[str, Any]:
    sm = _rj(_gate_epochs() / str(int(epoch)) / 'SAMPLE.private.json')
    if not isinstance(sm, Mapping) or not sm.get('round_id'):
        raise RoundSeedError('SAMPLE.private.json missing for epoch %s' % epoch)
    return sm


def _round_private(rid: str) -> Mapping[str, Any]:
    r = _rj(_heldout_root() / str(rid) / 'ROUND.private.json')
    if not isinstance(r, Mapping):
        raise RoundSeedError('ROUND.private.json missing for round %s' % rid)
    if r.get('seed') is None:
        raise RoundSeedError('round %s seed not ready' % rid)
    if not r.get('indices'):
        raise RoundSeedError('round %s indices not ready' % rid)
    return r


def plan_seed_for_epoch(epoch: int) -> str:
    """Public plan-seed handle for epoch ``e`` (never the raw secret)."""
    sm = _sample_for_epoch(int(epoch))
    rid = str(sm['round_id'])
    r = _round_private(rid)
    # Prefer baseline_v1 when importable so bit-identity is guaranteed.
    try:
        gate_code = os.environ.get('GATE_CODE') or (
            '/home/const/subnet120-rewrite/state/root-audits/'
            '20261026-promotion-gate/code'
        )
        if gate_code and gate_code not in sys.path:
            sys.path.insert(0, str(gate_code))
        import baseline_v1 as BV  # type: ignore
        seed = BV.cand_seed(r['seed'], int(epoch))
    except Exception:
        seed = _cand_seed(str(r['seed']), int(epoch))
    if not isinstance(seed, str) or len(seed) != 64:
        raise RoundSeedError('plan seed derivation failed for epoch %s' % epoch)
    return seed


# Aliases expected by host_wiring.
plan_seed_fn = plan_seed_for_epoch
cand_seed_for_epoch = plan_seed_for_epoch


def indices_for_epoch(epoch: int) -> List[int]:
    """Host-private task indices for the epoch's bound round (fail closed)."""
    sm = _sample_for_epoch(int(epoch))
    rid = str(sm['round_id'])
    r = _round_private(rid)
    idx = list(r['indices'])
    sm_idx = sm.get('indices')
    if sm_idx is not None and list(sm_idx) != idx:
        raise RoundSeedError('round draw != epoch sample for e%s' % epoch)
    if not idx:
        raise RoundSeedError('empty indices for epoch %s' % epoch)
    return [int(x) for x in idx]


def indices_for_round(round_id: str) -> List[int]:
    r = _round_private(str(round_id))
    return [int(x) for x in r['indices']]


def round_id_for_epoch(epoch: int) -> str:
    return str(_sample_for_epoch(int(epoch))['round_id'])
