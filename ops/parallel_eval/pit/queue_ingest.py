"""Ingest incoming batches into the PIT controller (Jacob 2026-11-09).

PIT owns dispatch: queued batches go straight here with a global box cap.
Sources (idempotent):
  1. ``PIT_STATE/inbox/*.json`` — host-written batch tickets (preferred)
  2. K1 turns under lifecycle epochs (uid/hotkey/reg) not yet queued
  3. Optional legacy ``life/pending/e*.json`` (ignored when ``pit_claimed``)

Head is refreshed from HEAD-CHAIN / CURRENT so each spawn binds the live head.
Lifecycle never feeds work by claiming epochs.
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


def _pit_state_root() -> Path:
    env = os.environ.get('PIT_STATE')
    if env:
        return Path(env)
    return _pe_state_root() / 'pit'


def _life_root() -> Path:
    env = os.environ.get('PE_LIFE_STATE')
    if env:
        return Path(env)
    pe_life = os.environ.get('PE_LIFE')
    if pe_life:
        return Path(pe_life)
    return Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')


def _gate_epochs_root() -> Path:
    env = os.environ.get('PE_GATE_EP')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261026-promotion-gate/state/epochs'
    )


def _k1_turns_path() -> Path:
    env = os.environ.get('K1_TURNS')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261027-k1-per-uid/state/K1-TURNS.jsonl'
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
    # Lifecycle CURRENT is the live head under PIT cutover.
    for cur_path in (
        _life_root() / 'CURRENT.json',
        Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/CURRENT.json'),
        st / 'CURRENT.json',
    ):
        cur = _rj(cur_path, {}) or {}
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


def _inbox_batches() -> List[Dict[str, Any]]:
    """Host PIT inbox: one JSON ticket per queued batch."""
    rows: List[Dict[str, Any]] = []
    for inbox in (_pit_state_root() / 'inbox', _pe_state_root() / 'pit' / 'inbox'):
        if not inbox.is_dir():
            continue
        for path in sorted(inbox.glob('*.json')):
            doc = _rj(path)
            if not isinstance(doc, Mapping):
                continue
            epoch = int(doc.get('epoch') or doc.get('batch_epoch') or -1)
            if epoch < 0:
                continue
            batch_path = (
                doc.get('batch_path') or doc.get('batch') or doc.get('batch_file')
                or doc.get('turn_path')
            )
            rows.append(dict(
                epoch=epoch,
                registration_index=int(doc.get('registration_index') or doc.get('queue_position') or epoch),
                uid=int(doc.get('uid') or 0),
                hotkey_prefix=str(doc.get('hotkey_prefix') or doc.get('hotkey') or '')[:16],
                path=str(path),
                batch_path=batch_path,
                source='inbox',
                raw=dict(doc),
            ))
    return rows


def _k1_batch_rows() -> List[Dict[str, Any]]:
    """Recent K1 turns as batch tickets (uid/hotkey/epoch)."""
    rows: List[Dict[str, Any]] = []
    # Per-epoch K1-TURN.json under life epochs.
    life_epochs = _life_root() / 'epochs'
    if life_epochs.is_dir():
        for d in sorted(life_epochs.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not d.name.isdigit():
                continue
            turn = _rj(d / 'K1-TURN.json')
            if not turn:
                continue
            epoch = int(turn.get('epoch') or d.name)
            batch_path = (
                turn.get('batch_path') or turn.get('batch') or turn.get('batch_file')
                or turn.get('path')
            )
            # Common layout: life/epochs/<e>/batch.json next to K1-TURN.json.
            if not batch_path:
                for cand in (d / 'batch.json', d / 'BATCH.json', d / 'turn.batch.json'):
                    if cand.is_file():
                        batch_path = str(cand)
                        break
            rows.append(dict(
                epoch=epoch,
                registration_index=int(turn.get('registration_index') or turn.get('queue_position') or epoch),
                uid=int(turn.get('uid') or 0),
                hotkey_prefix=str(turn.get('hotkey') or turn.get('hotkey_prefix') or '')[:16],
                path=str(d / 'K1-TURN.json'),
                batch_path=batch_path,
                source='k1-turn',
                raw=dict(turn),
            ))
    # Append-only turns log (best-effort tail).
    turns = _k1_turns_path()
    if turns.is_file():
        try:
            lines = turns.read_text().splitlines()[-200:]
        except Exception:
            lines = []
        for line in lines:
            try:
                turn = json.loads(line)
            except Exception:
                continue
            epoch = int(turn.get('epoch') or -1)
            if epoch < 0:
                continue
            rows.append(dict(
                epoch=epoch,
                registration_index=int(turn.get('registration_index') or turn.get('queue_position') or epoch),
                uid=int(turn.get('uid') or 0),
                hotkey_prefix=str(turn.get('hotkey') or turn.get('hotkey_prefix') or '')[:16],
                path=str(turns),
                source='k1-turns-jsonl',
                raw=dict(turn),
            ))
    return rows


def _legacy_pending_rows() -> List[Dict[str, Any]]:
    """Ignore lifecycle pe_lifecycle claim stubs (pit_claimed); keep real trainer pending."""
    pending_dir = _life_root() / 'pending'
    if not pending_dir.is_dir():
        pending_dir = _pe_state_root() / 'life' / 'pending'
    rows: List[Dict[str, Any]] = []
    if not pending_dir.is_dir():
        return rows
    for path in sorted(pending_dir.glob('e*.json')):
        doc = _rj(path)
        if not doc or doc.get('pit_claimed'):
            continue
        epoch = int(doc.get('epoch') or path.stem.lstrip('e') or -1)
        if epoch < 0:
            continue
        rows.append(dict(
            epoch=epoch,
            registration_index=int(doc.get('registration_index') or epoch),
            uid=int(doc.get('uid') or 0),
            hotkey_prefix=str(doc.get('hotkey_prefix') or '')[:16],
            path=str(path),
            source='pending',
            raw=dict(doc),
        ))
    return rows


def sync(
    controller: 'Controller',
    store: Optional['DurableStore'] = None,
) -> Dict[str, Any]:
    """Enqueue new batches and refresh controller head from durable state."""
    known_epochs: Set[int] = {int(i.epoch) for i in controller.items}
    if store is not None:
        known_epochs |= {int(i.epoch) for i in store.list_items()}

    tickets: List[Dict[str, Any]] = []
    tickets.extend(_inbox_batches())
    tickets.extend(_k1_batch_rows())
    tickets.extend(_legacy_pending_rows())
    # Prefer inbox over duplicates of the same epoch.
    seen: Set[int] = set()
    ordered: List[Dict[str, Any]] = []
    for row in tickets:
        e = int(row['epoch'])
        if e in seen:
            continue
        seen.add(e)
        ordered.append(row)
    ordered.sort(key=lambda r: (int(r['registration_index']), int(r['epoch'])))

    enqueued: List[int] = []
    for row in ordered:
        epoch = int(row['epoch'])
        if epoch in known_epochs:
            continue
        item = controller.enqueue(
            epoch=epoch,
            registration_index=int(row.get('registration_index') or epoch),
            uid=int(row.get('uid') or 0),
            hotkey_prefix=str(row.get('hotkey_prefix') or ''),
        )
        item.meta = dict(
            item.meta or {},
            ingest_source=row.get('source'),
            batch_path=row.get('batch_path') or row.get('path'),
            ticket_path=row.get('path'),
        )
        known_epochs.add(epoch)
        enqueued.append(epoch)
        if store is not None:
            store.save_item(item)
            store.enqueue(item.item_id)

    head, head_epoch = _head_chain_tip()
    head_set = False
    if head:
        master = _master_for_head(head)
        if master:
            cur = _rj(_life_root() / 'CURRENT.json', {}) or {}
            round_id = str(cur.get('round_id') or controller.current_round_id)
            controller.set_head(head, epoch=head_epoch, master_digest=master, round_id=round_id)
            head_set = True

    return dict(
        enqueued=enqueued,
        head=head,
        head_epoch=head_epoch,
        head_set=head_set,
        sources=['inbox', 'k1', 'pending'],
    )
