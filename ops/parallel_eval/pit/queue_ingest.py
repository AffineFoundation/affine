"""Ingest lifecycle dispatch queue + HEAD-CHAIN into the PIT controller (blocker 4).

Reads ``PE_LIFE_STATE/pending/e*.json``, K1 turn metadata, and durable HEAD-CHAIN /
CURRENT for binding head. Idempotent: skips epochs already present in the store.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, TYPE_CHECKING

from .expected_master import lookup_expected_master

if TYPE_CHECKING:
    from .controller import Controller
    from .durable import DurableStore


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _pe_state_root() -> Path:
    env = os.environ.get('PE_STATE')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/20261106-parallel-cutover/state'
    )


def _life_root() -> Path:
    env = os.environ.get('PE_LIFE_STATE')
    if env:
        return Path(env)
    pe_life = os.environ.get('PE_LIFE')
    if pe_life:
        return Path(pe_life)
    return Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/life')


def _gate_epochs_root() -> Path:
    env = os.environ.get('PE_GATE_EP')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261026-promotion-gate/state/epochs'
    )


def _head_chain_tip() -> tuple[Optional[str], int]:
    st = _pe_state_root()
    chain = _rj(st / 'HEAD-CHAIN.json', {}) or {}
    applied = list(chain.get('applied') or [])
    if applied:
        tip = str(applied[-1].get('new_head') or '')
        epoch = int(applied[-1].get('epoch_index') or applied[-1].get('epoch') or 0)
        if len(tip) == 64:
            return tip, epoch
    genesis = chain.get('genesis')
    if isinstance(genesis, str) and len(genesis) == 64:
        cur = _rj(st / 'CURRENT.json', {}) or {}
        epoch = int(cur.get('epoch_index') or 0)
        return genesis, epoch
    cur = _rj(st / 'CURRENT.json', {}) or {}
    head = (cur.get('checkpoint') or {}).get('id')
    if isinstance(head, str) and len(head) == 64:
        return head, int(cur.get('epoch_index') or 0)
    return None, 0


def _master_for_head(head: str) -> Optional[str]:
    st = _pe_state_root()
    index = _rj(st / 'MASTER-INDEX.json', {}) or {}
    if isinstance(index, Mapping):
        dig = index.get(head)
        if isinstance(dig, str) and len(dig) == 64:
            return dig
    return lookup_expected_master(head, index=index if isinstance(index, Mapping) else None)


def _k1_meta(epoch: int) -> Dict[str, Any]:
    life = _life_root()
    turn = _rj(life / 'epochs' / str(epoch) / 'K1-TURN.json')
    if not turn:
        turn = _rj(_gate_epochs_root().parent.parent / 'life' / 'epochs' / str(epoch) / 'K1-TURN.json')
    if not turn:
        return {}
    uid = int(turn.get('uid') or 0)
    hotkey = str(turn.get('hotkey') or turn.get('hotkey_prefix') or '')[:16]
    reg_idx = turn.get('queue_position') or turn.get('reg_block') or turn.get('registration_index')
    if reg_idx is None:
        reg_idx = epoch
    return dict(uid=uid, hotkey_prefix=hotkey, registration_index=int(reg_idx))


def _pending_epochs() -> List[Dict[str, Any]]:
    pending_dir = _life_root() / 'pending'
    if not pending_dir.is_dir():
        pending_dir = _pe_state_root() / 'life' / 'pending'
    rows: List[Dict[str, Any]] = []
    for path in sorted(pending_dir.glob('e*.json')):
        doc = _rj(path)
        if not doc:
            continue
        epoch = int(doc.get('epoch') or path.stem.lstrip('e') or -1)
        if epoch < 0:
            continue
        rows.append(dict(
            epoch=epoch,
            training_head=doc.get('training_head') or doc.get('head'),
            candidate=doc.get('candidate'),
            path=str(path),
            raw=doc,
        ))
    return rows


def sync(
    controller: 'Controller',
    store: Optional['DurableStore'] = None,
) -> Dict[str, Any]:
    """Enqueue new pending epochs and refresh controller head from durable state."""
    known_epochs: Set[int] = {int(i.epoch) for i in controller.items}
    if store is not None:
        known_epochs |= {int(i.epoch) for i in store.list_items()}

    enqueued: List[int] = []
    for row in _pending_epochs():
        epoch = int(row['epoch'])
        if epoch in known_epochs:
            continue
        meta = _k1_meta(epoch)
        item = controller.enqueue(
            epoch=epoch,
            registration_index=int(meta.get('registration_index') or epoch),
            uid=int(meta.get('uid') or 0),
            hotkey_prefix=str(meta.get('hotkey_prefix') or ''),
        )
        item.meta = dict(item.meta or {}, ingest_source='pending', pending_path=row.get('path'))
        known_epochs.add(epoch)
        enqueued.append(epoch)
        if store is not None:
            store.save_item(item)

    head, head_epoch = _head_chain_tip()
    head_set = False
    if head:
        master = _master_for_head(head)
        if master:
            cur = _rj(_pe_state_root() / 'CURRENT.json', {}) or {}
            round_id = str(cur.get('round_id') or controller.current_round_id)
            controller.set_head(head, epoch=head_epoch, master_digest=master, round_id=round_id)
            head_set = True

    return dict(enqueued=enqueued, head=head, head_epoch=head_epoch, head_set=head_set)
