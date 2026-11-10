#!/usr/bin/env python3
"""Sanctioned PIT GREEN rerun (Jacob 2026-11-10).

Operator CLI writes a ``RERUN-REQUEST``; the host consumes it under ``pit.lock``.
The CLI never edits item / gate / lifecycle files directly.

Refuse unless ALL hold:
  * item released, or no PIT-RESULT and no live box
  * never applied (not in HEAD-CHAIN / APPLY)
  * nothing paid
  * gate head unchanged since original bind; candidate not in HEAD-CHAIN
  * miner batch file exists and sha256 == K1-TURN ``file_sha256``

On accept: archive attempt-1 (SAMPLE/DECISION/EVALUATED/COMMIT, PIT-RESULT,
old item), write ``RERUN.json`` with hashes, fresh prior_gate draw on the
CURRENT round with seed from (round, epoch, attempt), enqueue attempt=2 with
``priority`` so it binds ahead of registration-index FIFO. Not a new K1 turn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .cutover_apply import head_chain_has_candidate
from .durable import DurableStore, item_from_dict, pit_state_root
from .life_paths import pe_state_root
from .queue_ingest import write_pit_result
from .state import ItemRecord, ItemState, RENTED_STATES

RERUN_PRIORITY = 1_000_000
RERUN_ATTEMPT = 2
GATE_ARCHIVE_NAMES = (
    'SAMPLE.private.json',
    'DECISION.json',
    'EVALUATED.private.json',
    'COMMIT.json',
)


class RerunError(RuntimeError):
    """Refuse or fail a sanctioned rerun."""


def _rj(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _wj(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _sha_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def life_root() -> Path:
    for key in ('LIFE_ROOT', 'PE_LIFE'):
        env = os.environ.get(key)
        if env:
            return Path(env)
    raise RerunError('LIFE_ROOT/PE_LIFE unset (fail closed)')


def gate_epochs_root() -> Path:
    env = os.environ.get('PE_GATE_EP')
    if env:
        p = Path(env)
        return p if p.name == 'epochs' else p / 'epochs'
    gate = os.environ.get('HG_GATE_STATE')
    if gate:
        return Path(gate) / 'epochs'
    raise RerunError('PE_GATE_EP/HG_GATE_STATE unset (fail closed)')


def heldout_root() -> Path:
    env = os.environ.get('PE_HELDOUT')
    if env:
        return Path(env)
    gate = os.environ.get('HG_GATE_STATE')
    if gate:
        return Path(gate) / 'heldout'
    raise RerunError('PE_HELDOUT/HG_GATE_STATE unset (fail closed)')


def rerun_dir(store_root: Path, epoch: int) -> Path:
    return Path(store_root) / 'rerun' / ('e%d' % int(epoch))


def request_path(store_root: Path, epoch: int) -> Path:
    return Path(store_root) / 'rerun' / ('RERUN-REQUEST-e%d.json' % int(epoch))


def current_head_id() -> Optional[str]:
    try:
        chain = _rj(pe_state_root() / 'HEAD-CHAIN.json') or {}
    except Exception:
        chain = {}
    if isinstance(chain, Mapping):
        head = chain.get('head') or chain.get('current_head')
        if isinstance(head, str) and len(head) == 64:
            return head
        applied = chain.get('applied') or []
        if applied and isinstance(applied[-1], Mapping):
            cand = applied[-1].get('new_head') or applied[-1].get('candidate')
            if isinstance(cand, str) and len(cand) == 64:
                return cand
    return None


def find_item_for_epoch(items: Sequence[ItemRecord], epoch: int) -> Optional[ItemRecord]:
    matches = [i for i in items if int(i.epoch) == int(epoch)]
    if not matches:
        return None
    # Prefer non-archived live items; highest attempt last.
    matches.sort(key=lambda i: (int(i.attempt or 0), i.item_id))
    return matches[-1]


def batch_file_for_turn(epoch_dir: Path, turn: Mapping[str, Any]) -> Optional[Path]:
    """Miner free-file snapshot: ``epochs/<e>/ff/<hotkey>.json``."""
    hotkey = turn.get('hotkey')
    if not isinstance(hotkey, str) or not hotkey:
        return None
    path = Path(epoch_dir) / 'ff' / ('%s.json' % hotkey)
    return path if path.is_file() else None


def rerun_draw_seed(round_seed: str, epoch: int, attempt: int) -> str:
    return hashlib.sha256(
        ('%s:prior-baseline:rerun-draw:%d:%d' % (round_seed, int(epoch), int(attempt))).encode()
    ).hexdigest()


def rerun_cand_seed(round_seed: str, epoch: int, attempt: int) -> str:
    """Fresh candidate plan seed — never equals attempt-1 ``cand_seed(seed, epoch)``."""
    return hashlib.sha256(
        ('%s:prior-baseline:cand:%d:attempt:%d' % (round_seed, int(epoch), int(attempt))).encode()
    ).hexdigest()


def _default_prior_draw(pi: Mapping[str, Any], seed: str) -> List[str]:
    try:
        import baseline_v1 as BV  # type: ignore
        return list(BV.draw(dict(pi), seed))
    except Exception as exc:
        raise RerunError('prior_gate/baseline_v1.draw unavailable: %s' % str(exc)[:200]) from exc


def _probe_index(task_id: str, cfg: Optional[Mapping[str, Any]] = None) -> int:
    try:
        import baseline_v1 as BV  # type: ignore
        return int(BV.probe_index(task_id, cfg))
    except Exception:
        # Hermetic fallback: stable int from task id digits / hash.
        digits = ''.join(c for c in str(task_id) if c.isdigit())
        if digits:
            return int(digits) % 10_000_000
        return int(hashlib.sha256(str(task_id).encode()).hexdigest()[:8], 16) % 10_000_000


@dataclass
class RerunCheck:
    ok: bool
    reasons: List[str]
    item: Optional[ItemRecord] = None
    turn: Optional[Dict[str, Any]] = None
    batch_path: Optional[str] = None
    sample: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    bound_head: Optional[str] = None
    candidate_id: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            'ok': self.ok,
            'reasons': list(self.reasons),
            'item_id': None if self.item is None else self.item.item_id,
            'batch_path': self.batch_path,
            'bound_head': self.bound_head,
            'candidate_id': self.candidate_id,
        }


def preflight(
    epoch: int,
    *,
    store: Optional[DurableStore] = None,
    items: Optional[Sequence[ItemRecord]] = None,
) -> RerunCheck:
    """Read-only refuse checks (CLI + host). Never mutates state."""
    reasons: List[str] = []
    life = life_root()
    gate_ep = gate_epochs_root()
    epoch_dir = life / 'epochs' / str(int(epoch))
    gate_dir = gate_ep / str(int(epoch))
    turn = _rj(epoch_dir / 'K1-TURN.json')
    if not isinstance(turn, Mapping) or turn.get('status') != 'turn':
        reasons.append('K1-TURN missing or not status=turn')
        turn_doc: Optional[Dict[str, Any]] = None
    else:
        turn_doc = dict(turn)

    store = store or DurableStore()
    item_list = list(items) if items is not None else store.list_items()
    item = find_item_for_epoch(item_list, int(epoch))

    pit_result = epoch_dir / 'PIT-RESULT.json'
    has_pit_result = pit_result.is_file()
    live_box = bool(
        item is not None
        and item.box_id
        and item.state in RENTED_STATES
    )
    released = bool(item is not None and item.state is ItemState.RELEASED)
    if not (released or (not has_pit_result and not live_box)):
        reasons.append(
            'item not released and (PIT-RESULT present or live box held)'
        )

    sample = _rj(gate_dir / 'SAMPLE.private.json')
    decision = _rj(gate_dir / 'DECISION.json')
    bound_head = None
    candidate_id = None
    if item is not None:
        bound_head = item.bound_head
        candidate_id = item.candidate_id
    if isinstance(sample, Mapping):
        bound_head = bound_head or sample.get('head')
    if isinstance(decision, Mapping):
        candidate_id = candidate_id or decision.get('candidate')
        bound_head = bound_head or decision.get('head')

    # Never applied / not in HEAD-CHAIN.
    if candidate_id and head_chain_has_candidate(str(candidate_id)):
        reasons.append('candidate is in HEAD-CHAIN (already applied)')
    apply_ready = pe_state_root() / 'apply'
    if candidate_id and apply_ready.is_dir():
        for p in apply_ready.glob('*.json'):
            doc = _rj(p)
            if isinstance(doc, Mapping) and doc.get('candidate') == candidate_id:
                if doc.get('state') in ('applied', 'ready', 'APPLY'):
                    reasons.append('candidate has APPLY artifact (%s)' % p.name)
                    break
    greens = _rj(pe_state_root() / 'GREENS.json') or {}
    for row in (greens.get('pending') or []) + (greens.get('applied') or []):
        if isinstance(row, Mapping) and row.get('candidate') == candidate_id:
            if row.get('state') == 'applied' or row.get('applied'):
                reasons.append('candidate listed applied in GREENS.json')
                break

    # Nothing paid: item flag, finalized settlement, or consumed K1 join.
    # A green DECISION.paid alone is not enough — lost GREENs may carry that
    # flag without ever settling; refuse only when settlement completed.
    if item is not None and item.paid:
        reasons.append('item.paid is true')
    finalized = epoch_dir / 'FINALIZED.json'
    promo = epoch_dir / 'PROMOTION-GATE.json'
    promo_doc = _rj(promo) if promo.is_file() else None
    settled = finalized.is_file() or (
        isinstance(promo_doc, Mapping)
        and (promo_doc.get('applied') or promo_doc.get('decision') in ('applied', 'green'))
    )
    if isinstance(decision, Mapping) and decision.get('paid') is True and settled:
        reasons.append('DECISION.paid with FINALIZED/PROMOTION-GATE settlement')
    for key in ('K1_JOINS', 'K1_STATE'):
        root = os.environ.get(key)
        if not root:
            continue
        join = Path(root) / ('%d.json' % int(epoch))
        if not join.is_file():
            join = Path(root) / 'joins' / ('%d.json' % int(epoch))
        if join.is_file():
            reasons.append('K1 join exists (turn already consumed)')
            break

    # Head unchanged since original bind; candidate not in HEAD-CHAIN (above).
    head_now = current_head_id()
    if bound_head and head_now and bound_head != head_now:
        # Head may have advanced via OTHER candidates; refuse only when the
        # original bind head is no longer the gate/chain head *and* we cannot
        # confirm the gate SAMPLE head still matches bind. SAMPLE.head is the
        # authoritative bind-time head.
        sample_head = sample.get('head') if isinstance(sample, Mapping) else None
        if sample_head and sample_head != bound_head:
            reasons.append('gate SAMPLE head diverged from bound head')
        # If SAMPLE still names the bind head but chain head moved via another
        # GREEN, still refuse — rebinding must use CURRENT head via fresh SAMPLE.
        # Operator intent: "gate head is unchanged since the original bind".
        if sample_head and head_now and sample_head != head_now:
            reasons.append(
                'gate/chain head moved since original bind (%s -> %s)'
                % (str(sample_head)[:12], str(head_now)[:12])
            )

    # Batch file sha == K1-TURN file_sha256.
    batch_path = None
    if turn_doc is not None:
        want = turn_doc.get('file_sha256')
        if not isinstance(want, str) or len(want) != 64:
            reasons.append('K1-TURN.file_sha256 missing')
        else:
            bf = batch_file_for_turn(epoch_dir, turn_doc)
            if bf is None:
                reasons.append('miner batch file missing (epochs/<e>/ff/<hotkey>.json)')
            else:
                got = _sha_file(bf)
                batch_path = str(bf)
                if got != want:
                    reasons.append('batch sha256 != K1-TURN.file_sha256')

    # Already have an active attempt-2 / pending request handled elsewhere.
    if item is not None and int(item.attempt or 0) >= RERUN_ATTEMPT and item.state is ItemState.QUEUED:
        reasons.append('attempt-%d item already queued' % RERUN_ATTEMPT)

    ok = not reasons
    return RerunCheck(
        ok=ok,
        reasons=reasons,
        item=item,
        turn=turn_doc,
        batch_path=batch_path,
        sample=dict(sample) if isinstance(sample, Mapping) else None,
        decision=dict(decision) if isinstance(decision, Mapping) else None,
        bound_head=bound_head if isinstance(bound_head, str) else None,
        candidate_id=candidate_id if isinstance(candidate_id, str) else None,
    )


def write_rerun_request(
    *,
    epoch: int,
    operator: str,
    reason: str,
    store_root: Optional[Path] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """CLI path: validate + write RERUN-REQUEST only (no archive / item edits)."""
    if not operator or not str(operator).strip():
        raise RerunError('operator required')
    if not reason or not str(reason).strip():
        raise RerunError('reason required')
    root = Path(store_root) if store_root else pit_state_root()
    store = DurableStore(root)
    check = preflight(int(epoch), store=store)
    if not check.ok:
        raise RerunError('refuse rerun e%d: %s' % (int(epoch), '; '.join(check.reasons)))
    path = request_path(root, int(epoch))
    if path.is_file():
        existing = _rj(path) or {}
        if existing.get('status') in ('requested', 'consumed'):
            raise RerunError(
                'RERUN-REQUEST already exists for e%d (status=%s)'
                % (int(epoch), existing.get('status'))
            )
    at = float(now if now is not None else time.time())
    doc = {
        'epoch': int(epoch),
        'operator': str(operator).strip(),
        'reason': str(reason).strip(),
        'status': 'requested',
        'requested_at': at,
        'preflight': check.as_dict(),
    }
    _wj(path, doc, 0o600)
    return doc


def _archive_file(src: Path, dest_dir: Path) -> Optional[Dict[str, str]]:
    if not src.is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        raise RerunError('archive destination exists: %s' % dest)
    shutil.move(str(src), str(dest))
    return {'path': str(dest), 'sha256': _sha_file(dest) or ''}


def archive_attempt1(
    epoch: int,
    *,
    store: DurableStore,
    item: Optional[ItemRecord],
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Move gate/lifecycle/item artifacts into attempt-1/; write RERUN.json hashes."""
    life = life_root()
    gate_dir = gate_epochs_root() / str(int(epoch))
    epoch_dir = life / 'epochs' / str(int(epoch))
    rr_root = rerun_dir(store.root, int(epoch))
    gate_arch = gate_dir / 'attempt-1'
    life_arch = epoch_dir / 'attempt-1'
    item_arch = rr_root / 'attempt-1'
    hashes: Dict[str, Any] = {'gate': {}, 'lifecycle': {}, 'item': None}
    for name in GATE_ARCHIVE_NAMES:
        meta = _archive_file(gate_dir / name, gate_arch)
        if meta:
            hashes['gate'][name] = meta
    pit_meta = _archive_file(epoch_dir / 'PIT-RESULT.json', life_arch)
    if pit_meta:
        hashes['lifecycle']['PIT-RESULT.json'] = pit_meta
    if item is not None:
        src = store.items_dir / ('%s.json' % item.item_id)
        item_arch.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            dest = item_arch / ('%s.json' % item.item_id)
            if dest.exists():
                raise RerunError('item archive exists: %s' % dest)
            shutil.move(str(src), str(dest))
            hashes['item'] = {
                'item_id': item.item_id,
                'path': str(dest),
                'sha256': _sha_file(dest) or '',
            }
        # Drop from controller memory is host's job; remove queue entry.
        q = [x for x in store.queue() if x != item.item_id]
        _wj(store.queue_path(), q, 0o600)
    at = float(now if now is not None else time.time())
    doc = {
        'epoch': int(epoch),
        'attempt_archived': 1,
        'attempt_next': RERUN_ATTEMPT,
        'archived_at': at,
        'hashes': hashes,
        'old_item_id': None if item is None else item.item_id,
    }
    _wj(rr_root / 'RERUN.json', doc, 0o600)
    _wj(gate_dir / 'RERUN.json', doc, 0o600)
    return doc


def write_fresh_sample(
    epoch: int,
    *,
    attempt: int = RERUN_ATTEMPT,
    draw_fn: Optional[Callable[[Mapping[str, Any], str], Sequence[str]]] = None,
    round_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fresh prior_gate draw on the CURRENT round; seed from (round, epoch, attempt)."""
    if round_provider is not None:
        round_doc = dict(round_provider())
    else:
        round_doc = _load_current_round()
    rid = str(round_doc['id'])
    round_seed = str(round_doc.get('seed') or '')
    if not round_seed:
        raise RerunError('current round seed missing')
    head = str(round_doc.get('head') or '')
    if len(head) != 64:
        raise RerunError('current round head missing')
    draw_seed = rerun_draw_seed(round_seed, int(epoch), int(attempt))
    plan = rerun_cand_seed(round_seed, int(epoch), int(attempt))
    # Never equal the attempt-1 cand_seed(seed, epoch).
    classic = hashlib.sha256(
        ('%s:prior-baseline:cand:%d' % (round_seed, int(epoch))).encode()
    ).hexdigest()
    if plan == classic:
        raise RerunError('rerun cand_seed collided with attempt-1 seed')

    pi = round_doc.get('pi')
    if not isinstance(pi, Mapping):
        pi_path = heldout_root() / rid / 'PI.private.json'
        pi = _rj(pi_path)
    if not isinstance(pi, Mapping) or not pi.get('pi'):
        # Fall back: reuse round indices but still tag fresh plan_seed/attempt.
        # Prefer real prior draw when PI is present.
        ids = None
        indices = list(round_doc.get('indices') or [])
        if not indices:
            raise RerunError('current round has no indices/PI for fresh draw')
    else:
        drawer = draw_fn or _default_prior_draw
        pi_map = pi.get('pi') if 'pi' in pi and isinstance(pi.get('pi'), Mapping) else pi
        ids = list(drawer(pi_map, draw_seed))
        cfg = round_doc.get('cfg') or (pi.get('cfg') if isinstance(pi, Mapping) else None)
        indices = [_probe_index(t, cfg if isinstance(cfg, Mapping) else None) for t in ids]

    if not indices:
        raise RerunError('fresh draw produced empty indices')
    at = time.time()
    sample = {
        'version': 'pit-rerun-v1',
        'epoch': int(epoch),
        'head': head,
        'seed': draw_seed,  # draw seed handle (not round secret)
        'indices': [int(x) for x in indices],
        'drawn_at': at,
        'heldout': True,
        'prior_v1': True,
        'round_id': rid,
        'attempt': int(attempt),
        'plan_seed': plan,
        'rerun': True,
        'ids': list(ids) if ids is not None else None,
    }
    gate_dir = gate_epochs_root() / str(int(epoch))
    # Refuse clobbering a non-archived SAMPLE.
    live = gate_dir / 'SAMPLE.private.json'
    if live.is_file():
        raise RerunError('SAMPLE.private.json still present (archive attempt-1 first)')
    _wj(live, sample, 0o600)
    commit = {
        'version': 'pit-rerun-v1',
        'epoch': int(epoch),
        'head': head,
        'sample_size': len(indices),
        'heldout_round': rid,
        'attempt': int(attempt),
        'rerun': True,
        'drawn_at': at,
        'plan_seed': plan,
    }
    _wj(gate_dir / 'COMMIT.json', commit, 0o644)
    return sample


def _load_current_round() -> Dict[str, Any]:
    root = heldout_root()
    # Prefer CURRENT.json pointer when present.
    cur = _rj(root / 'CURRENT.json') or _rj(root / 'current.json')
    rid = None
    if isinstance(cur, Mapping):
        rid = cur.get('id') or cur.get('round_id')
    if not rid:
        # Newest ROUND.private.json by mtime.
        cands = sorted(root.glob('*/ROUND.private.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise RerunError('no heldout ROUND.private.json for current round')
        rid = cands[0].parent.name
    doc = _rj(root / str(rid) / 'ROUND.private.json')
    if not isinstance(doc, Mapping):
        raise RerunError('ROUND.private.json missing for current round %s' % rid)
    out = dict(doc)
    out.setdefault('id', rid)
    return out


def enqueue_rerun_item(
    *,
    epoch: int,
    store: DurableStore,
    ctl: Any,
    old_item: Optional[ItemRecord],
    sample: Mapping[str, Any],
    operator: str,
    reason: str,
) -> ItemRecord:
    """Create attempt=2 QUEUED item with priority; not a new K1 turn."""
    turn = _rj(life_root() / 'epochs' / str(int(epoch)) / 'K1-TURN.json') or {}
    uid = int((old_item.uid if old_item else turn.get('uid')) or 0)
    hotkey = (
        (old_item.hotkey_prefix if old_item else '')
        or str(turn.get('hotkey') or '')[:16]
    )
    reg = int(old_item.registration_index if old_item else epoch)
    old_id = old_item.item_id if old_item else None
    item_id = 'pit-rerun-e%d-a%d' % (int(epoch), RERUN_ATTEMPT)
    rec = ItemRecord(
        item_id=item_id,
        registration_index=reg,
        epoch=int(epoch),
        uid=uid,
        hotkey_prefix=hotkey,
        state=ItemState.QUEUED,
        attempt=RERUN_ATTEMPT,
        priority=RERUN_PRIORITY,
        rerun_of=old_id,
        shadow=bool(ctl.config.is_shadow()) if getattr(ctl, 'config', None) else True,
        meta={
            'rerun': True,
            'rerun_operator': operator,
            'rerun_reason': reason,
            'rerun_round_id': sample.get('round_id'),
            'rerun_plan_seed': sample.get('plan_seed'),
            'k1_turn_reuse': True,  # not a new K1 submission turn
        },
    )
    if hasattr(ctl, 'enqueue'):
        rec = ctl.enqueue(
            epoch=int(epoch),
            registration_index=reg,
            uid=uid,
            hotkey_prefix=hotkey,
            item_id=item_id,
            attempt=RERUN_ATTEMPT,
            priority=RERUN_PRIORITY,
            rerun_of=old_id,
            meta=rec.meta,
        )
    else:
        ctl.items.append(rec)
    store.save_item(rec)
    store.enqueue(rec.item_id)
    return rec


def consume_rerun_request(
    epoch: int,
    *,
    store: DurableStore,
    ctl: Any,
    draw_fn: Optional[Callable[..., Sequence[str]]] = None,
    round_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Host path under pit.lock: archive → fresh SAMPLE → enqueue attempt 2."""
    path = request_path(store.root, int(epoch))
    req = _rj(path)
    if not isinstance(req, Mapping) or req.get('status') != 'requested':
        raise RerunError('no pending RERUN-REQUEST for e%d' % int(epoch))
    check = preflight(int(epoch), store=store, items=list(ctl.items))
    if not check.ok:
        req = dict(req)
        req['status'] = 'refused'
        req['refused_at'] = float(now if now is not None else time.time())
        req['refuse_reasons'] = list(check.reasons)
        _wj(path, req, 0o600)
        raise RerunError('refuse consume e%d: %s' % (int(epoch), '; '.join(check.reasons)))

    old_item = check.item
    # Remove old item from controller memory before archive.
    if old_item is not None:
        ctl.items = [i for i in list(ctl.items) if i.item_id != old_item.item_id]

    archive = archive_attempt1(
        int(epoch), store=store, item=old_item, now=now,
    )
    sample = write_fresh_sample(
        int(epoch),
        attempt=RERUN_ATTEMPT,
        draw_fn=draw_fn,
        round_provider=round_provider,
    )
    new_item = enqueue_rerun_item(
        epoch=int(epoch),
        store=store,
        ctl=ctl,
        old_item=old_item,
        sample=sample,
        operator=str(req.get('operator') or ''),
        reason=str(req.get('reason') or ''),
    )
    # Placeholder so queue_ingest cannot re-enqueue the epoch while attempt 2 runs.
    # Overwritten by the normal emit path when attempt 2 finishes.
    write_pit_result(
        life_root() / 'epochs' / str(int(epoch)),
        decision='rerun_active',
        item_id=new_item.item_id,
        extra={
            'attempt': RERUN_ATTEMPT,
            'rerun': True,
            'rerun_of': old_item.item_id if old_item else None,
        },
    )
    done = dict(req)
    done.update({
        'status': 'consumed',
        'consumed_at': float(now if now is not None else time.time()),
        'archive': archive,
        'new_item_id': new_item.item_id,
        'sample_round_id': sample.get('round_id'),
        'plan_seed': sample.get('plan_seed'),
    })
    _wj(path, done, 0o600)
    return done


def consume_pending_requests(
    *,
    store: DurableStore,
    ctl: Any,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    root = store.root / 'rerun'
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(root.glob('RERUN-REQUEST-e*.json')):
        doc = _rj(path)
        if not isinstance(doc, Mapping) or doc.get('status') != 'requested':
            continue
        epoch = int(doc['epoch'])
        try:
            out.append(consume_rerun_request(epoch, store=store, ctl=ctl, **kwargs))
        except RerunError as exc:
            out.append({'epoch': epoch, 'ok': False, 'error': str(exc)[:400]})
    return out


def decision_archived(epoch: int) -> bool:
    """True when attempt-1 archived a DECISION (gate_writer may accept a fresh one)."""
    gate_dir = gate_epochs_root() / str(int(epoch))
    return (gate_dir / 'attempt-1' / 'DECISION.json').is_file()


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='python -m ops.parallel_eval.pit.rerun')
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--operator', required=True)
    p.add_argument('--reason', required=True)
    p.add_argument('--dry-run', action='store_true', help='preflight only; do not write request')
    args = p.parse_args(list(argv) if argv is not None else None)
    try:
        check = preflight(int(args.epoch))
        if args.dry_run:
            print(json.dumps({'dry_run': True, **check.as_dict()}, indent=1, sort_keys=True))
            return 0 if check.ok else 2
        doc = write_rerun_request(
            epoch=int(args.epoch),
            operator=str(args.operator),
            reason=str(args.reason),
        )
        print(json.dumps(doc, indent=1, sort_keys=True, default=str))
        return 0
    except RerunError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
