#!/usr/bin/env python3
"""Sanctioned PIT GREEN rerun (Jacob 2026-11-10).

Operator CLI writes a ``RERUN-REQUEST``; the host consumes it under ``pit.lock``.
The CLI never edits item / gate / lifecycle files directly.

Refuse unless ALL hold:
  * item released, or no PIT-RESULT and no live box
  * never settled/applied (HEAD-CHAIN / APPLY / GREENS state=applied /
    consumed K1 join or payout) — ``item.paid`` alone is not settlement
  * miner batch file exists and sha256 == K1-TURN ``file_sha256``

Reruns bind the CURRENT round/head (Jacob): SAMPLE uses the round's frozen
task list; only the candidate plan seed is fresh per attempt.

On accept: archive attempt-1 (SAMPLE/DECISION/EVALUATED/COMMIT, PIT-RESULT,
old item), write ``RERUN.json`` with hashes, write SAMPLE on the CURRENT round
with frozen indices + attempt plan seed, enqueue attempt=2 with ``priority``.
Not a new K1 turn. ``RERUN-ACTIVE.json`` (not PIT-RESULT) blocks re-ingest so
max_pending still counts the epoch.
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
from .durable import DurableStore, pit_state_root
from .life_paths import pe_state_root
from .state import (
    ItemRecord,
    ItemState,
    RENTED_STATES,
    apply_transition,
    can_transition,
    holds_rental,
)

RERUN_PRIORITY = 1_000_000
# First sanctioned GREEN rerun (attempt-1 = original). Box-retry ``item.attempt``
# is independent and starts at 0 on every new rerun item.
RERUN_ATTEMPT = 2
# Meta keys copied from the prior item so HostBoxIO gets a real K1/epoch path.
RERUN_BATCH_META_KEYS = ('batch_path', 'epoch_dir', 'ticket_path', 'ingest_source')
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


def rerun_attempt_of(item: Optional[ItemRecord]) -> int:
    """Sanctioned SAMPLE/plan_seed attempt number (not the box-retry counter)."""
    if item is None:
        return 0
    meta = item.meta or {}
    raw = meta.get('rerun_attempt')
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    # Legacy items stamped attempt=RERUN_ATTEMPT before the counter split.
    if item.rerun_of or meta.get('rerun'):
        try:
            return int(item.attempt or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def find_item_for_epoch(items: Sequence[ItemRecord], epoch: int) -> Optional[ItemRecord]:
    matches = [i for i in items if int(i.epoch) == int(epoch)]
    if not matches:
        return None
    # Prefer non-archived live items; highest rerun_attempt, then box attempt.
    matches.sort(key=lambda i: (rerun_attempt_of(i), int(i.attempt or 0), i.item_id))
    return matches[-1]


def is_failed_rerun_without_result(item: Optional[ItemRecord], epoch: int) -> bool:
    """True when a prior rerun item died without PIT-RESULT (replaceable)."""
    if item is None:
        return False
    meta = item.meta or {}
    if not (item.rerun_of or meta.get('rerun') or rerun_attempt_of(item) >= RERUN_ATTEMPT):
        return False
    epoch_dir = life_root() / 'epochs' / str(int(epoch))
    if (epoch_dir / 'PIT-RESULT.json').is_file():
        return False
    if item.state in (ItemState.ABORTED, ItemState.FAILED):
        return True
    if item.state is ItemState.RELEASED:
        # Exhausted abort → RELEASED; RERUN-ACTIVE still marks an unfinished rerun.
        if (epoch_dir / 'RERUN-ACTIVE.json').is_file():
            return True
        if item.last_error or meta.get('aborted') or item.decision in ('no_decision',):
            return True
    return False


def next_rerun_attempt(item: Optional[ItemRecord]) -> int:
    """Next SAMPLE/plan_seed attempt: 2 for first rerun, else prior+1."""
    cur = rerun_attempt_of(item)
    if cur < RERUN_ATTEMPT:
        return RERUN_ATTEMPT
    return cur + 1


def _verify_batch_path_sha(
    batch_path: Path,
    turn: Mapping[str, Any],
) -> str:
    """Require batch_path exists and sha256 == K1-TURN.file_sha256.

    When ``batch_path`` is the epoch dir (k1-turn ingest), verify the miner
    free-file under ``ff/<hotkey>.json``. When it is a file, hash that file.
    """
    want = turn.get('file_sha256')
    if not isinstance(want, str) or len(want) != 64:
        raise RerunError('K1-TURN.file_sha256 missing')
    if not batch_path.exists():
        raise RerunError('batch_path missing: %s' % batch_path)
    if batch_path.is_dir():
        bf = batch_file_for_turn(batch_path, turn)
        if bf is None:
            raise RerunError(
                'miner batch file missing under batch_path %s' % batch_path
            )
        got = _sha_file(bf)
    else:
        got = _sha_file(batch_path)
    if got != want:
        raise RerunError('batch_path sha256 != K1-TURN.file_sha256')
    return want


def copy_rerun_batch_meta(
    old_item: Optional[ItemRecord],
    *,
    epoch: int,
    turn: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Copy batch_path/epoch_dir/ticket_path/ingest_source; verify before queue."""
    old_meta = dict(old_item.meta or {}) if old_item is not None else {}
    epoch_dir = life_root() / 'epochs' / str(int(epoch))
    batch_path = old_meta.get('batch_path') or str(epoch_dir)
    epoch_dir_s = old_meta.get('epoch_dir') or str(epoch_dir)
    ticket_path = old_meta.get('ticket_path')
    ingest_source = old_meta.get('ingest_source') or 'k1-turn'
    turn_doc = turn
    if turn_doc is None:
        raw = _rj(epoch_dir / 'K1-TURN.json')
        if not isinstance(raw, Mapping):
            raise RerunError('K1-TURN missing; cannot verify batch_path')
        turn_doc = raw
    _verify_batch_path_sha(Path(str(batch_path)), turn_doc)
    if not Path(str(epoch_dir_s)).exists():
        raise RerunError('epoch_dir missing: %s' % epoch_dir_s)
    out: Dict[str, Any] = {
        'batch_path': str(batch_path),
        'epoch_dir': str(epoch_dir_s),
        'ingest_source': ingest_source,
    }
    if ticket_path:
        out['ticket_path'] = str(ticket_path)
    return out


def _iter_greens_rows(greens: Any) -> List[Mapping[str, Any]]:
    """Live GREENS.json is a list of rows; keep dict ``pending``/``applied`` support."""
    rows: List[Mapping[str, Any]] = []
    if isinstance(greens, list):
        for row in greens:
            if isinstance(row, Mapping):
                rows.append(row)
        return rows
    if isinstance(greens, Mapping):
        for key in ('pending', 'applied', 'rows', 'items'):
            for row in greens.get(key) or []:
                if isinstance(row, Mapping):
                    rows.append(row)
        # Flat dict of candidate → row.
        if not rows:
            for val in greens.values():
                if isinstance(val, Mapping) and val.get('candidate'):
                    rows.append(val)
    return rows


def _greens_candidate_applied(greens: Any, candidate_id: str) -> bool:
    """True only when a GREENS row for candidate has ``state == 'applied'``."""
    if not candidate_id:
        return False
    for row in _iter_greens_rows(greens):
        if row.get('candidate') != candidate_id:
            continue
        if row.get('state') == 'applied':
            return True
    return False


def _k1_join_consumed(epoch: int) -> bool:
    for key in ('K1_JOINS', 'K1_STATE'):
        root = os.environ.get(key)
        if not root:
            continue
        join = Path(root) / ('%d.json' % int(epoch))
        if not join.is_file():
            join = Path(root) / 'joins' / ('%d.json' % int(epoch))
        if join.is_file():
            return True
    return False


def batch_file_for_turn(epoch_dir: Path, turn: Mapping[str, Any]) -> Optional[Path]:
    """Miner free-file snapshot: ``epochs/<e>/ff/<hotkey>.json``."""
    hotkey = turn.get('hotkey')
    if not isinstance(hotkey, str) or not hotkey:
        return None
    path = Path(epoch_dir) / 'ff' / ('%s.json' % hotkey)
    return path if path.is_file() else None


def rerun_cand_seed(round_seed: str, epoch: int, attempt: int) -> str:
    """Fresh candidate plan seed — never equals attempt-1 ``cand_seed(seed, epoch)``."""
    return hashlib.sha256(
        ('%s:prior-baseline:cand:%d:attempt:%d' % (round_seed, int(epoch), int(attempt))).encode()
    ).hexdigest()


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

    # Real settlement / apply only — not item.paid (run_host sets that on every GREEN).
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
    greens_raw = _rj(pe_state_root() / 'GREENS.json')
    if candidate_id and _greens_candidate_applied(greens_raw, str(candidate_id)):
        reasons.append('candidate listed applied in GREENS.json')

    # Consumed K1 join / payout record for the epoch.
    if _k1_join_consumed(int(epoch)):
        reasons.append('K1 join exists (turn already consumed)')
    payout = epoch_dir / 'PAYOUT.json'
    if payout.is_file():
        reasons.append('PAYOUT.json present (epoch already paid out)')

    # Jacob: reruns bind the CURRENT round/head — do not refuse on head move.

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

    # Active (non-terminal) rerun item already queued / in flight.
    # ABORTED / FAILED / RELEASED without PIT-RESULT → replaceable; ok.
    if item is not None and rerun_attempt_of(item) >= RERUN_ATTEMPT:
        if item.state is ItemState.QUEUED or item.state in RENTED_STATES:
            reasons.append(
                'rerun attempt-%d item already active (%s)'
                % (rerun_attempt_of(item), item.state.value)
            )

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
    replace = is_failed_rerun_without_result(check.item, int(epoch))
    next_attempt = next_rerun_attempt(check.item) if replace else RERUN_ATTEMPT
    if path.is_file():
        existing = _rj(path) or {}
        status = existing.get('status')
        if status == 'requested':
            raise RerunError(
                'RERUN-REQUEST already exists for e%d (status=%s)'
                % (int(epoch), status)
            )
        if status == 'consumed' and not replace:
            raise RerunError(
                'RERUN-REQUEST already exists for e%d (status=%s)'
                % (int(epoch), status)
            )
        # consumed + replaceable failed rerun → overwrite for attempt N+1.
    at = float(now if now is not None else time.time())
    doc = {
        'epoch': int(epoch),
        'operator': str(operator).strip(),
        'reason': str(reason).strip(),
        'status': 'requested',
        'requested_at': at,
        'preflight': check.as_dict(),
        'rerun_attempt': int(next_attempt),
        'replace_failed_rerun': bool(replace),
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


def _archive_already_present(epoch: int, store: DurableStore) -> Optional[Dict[str, Any]]:
    """Return existing RERUN.json when attempt-1 archive is durable; else None."""
    rr_root = rerun_dir(store.root, int(epoch))
    gate_arch = gate_epochs_root() / str(int(epoch)) / 'attempt-1'
    rr_path = rr_root / 'RERUN.json'
    if rr_path.is_file():
        doc = _rj(rr_path)
        if isinstance(doc, Mapping) and doc.get('hashes'):
            return dict(doc)
        raise RerunError(
            'RERUN.json exists for e%d but hashes are empty/corrupt — refuse rewrite'
            % int(epoch)
        )
    if gate_arch.is_dir() and any(gate_arch.iterdir()):
        raise RerunError(
            'attempt-1/ already exists for e%d without RERUN.json — refuse rewrite'
            % int(epoch)
        )
    return None


def archive_attempt1(
    epoch: int,
    *,
    store: DurableStore,
    item: Optional[ItemRecord],
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Move gate/lifecycle/item artifacts into attempt-1/; write RERUN.json hashes.

    Refuses to rewrite when ``RERUN.json`` or ``attempt-1/`` already exists
    (preserves audit hashes if a later step fails).
    """
    existing = _archive_already_present(int(epoch), store)
    if existing is not None:
        raise RerunError(
            'archive already present for e%d — refuse rewrite of RERUN.json hashes'
            % int(epoch)
        )
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
    round_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """SAMPLE on the CURRENT round's frozen task list; fresh attempt plan seed.

    Head baseline is built once per round over the frozen indices — a rerun
    must reuse those indices so z compares candidate and head on the same tasks.
    Only ``plan_seed`` (candidate sampling) is derived from (round, epoch, attempt).
    """
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
    indices = [int(x) for x in (round_doc.get('indices') or [])]
    if not indices:
        raise RerunError('current round has no frozen indices')
    plan = rerun_cand_seed(round_seed, int(epoch), int(attempt))
    classic = hashlib.sha256(
        ('%s:prior-baseline:cand:%d' % (round_seed, int(epoch))).encode()
    ).hexdigest()
    if plan == classic:
        raise RerunError('rerun cand_seed collided with attempt-1 seed')
    at = time.time()
    sample = {
        'version': 'pit-rerun-v1',
        'epoch': int(epoch),
        'head': head,
        # Round seed (same as prior_gate SAMPLE) — not a per-attempt task redraw.
        'seed': round_seed,
        'indices': indices,
        'drawn_at': at,
        'heldout': True,
        'prior_v1': True,
        'round_id': rid,
        'attempt': int(attempt),
        'plan_seed': plan,
        'rerun': True,
        'ids': list(round_doc['ids']) if round_doc.get('ids') is not None else None,
    }
    gate_dir = gate_epochs_root() / str(int(epoch))
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
        'commitment': round_doc.get('draw_commitment') or round_doc.get('commitment'),
    }
    _wj(gate_dir / 'COMMIT.json', commit, 0o644)
    return sample


def write_rerun_active(
    epoch: int,
    *,
    item_id: str,
    rerun_of: Optional[str] = None,
    rerun_attempt: int = RERUN_ATTEMPT,
) -> Path:
    """Marker that ingest skips without freeing max_pending (PIT-RESULT absent)."""
    path = life_root() / 'epochs' / str(int(epoch)) / 'RERUN-ACTIVE.json'
    _wj(path, {
        'status': 'rerun-active',
        'epoch': int(epoch),
        'item_id': item_id,
        'attempt': int(rerun_attempt),
        'rerun_attempt': int(rerun_attempt),
        'rerun_of': rerun_of,
        'at': time.time(),
    }, 0o600)
    return path


def archive_failed_rerun_sample(
    epoch: int,
    *,
    failed_attempt: int,
) -> Optional[Dict[str, Any]]:
    """Move live SAMPLE/COMMIT from a failed rerun into attempt-N/ (audit keep)."""
    gate_dir = gate_epochs_root() / str(int(epoch))
    dest = gate_dir / ('attempt-%d' % int(failed_attempt))
    hashes: Dict[str, Any] = {}
    for name in ('SAMPLE.private.json', 'COMMIT.json', 'DECISION.json', 'EVALUATED.private.json'):
        meta = _archive_file(gate_dir / name, dest)
        if meta:
            hashes[name] = meta
    return hashes or None


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
    rerun_attempt: int = RERUN_ATTEMPT,
) -> ItemRecord:
    """Create a QUEUED rerun item with priority; not a new K1 turn.

    ``rerun_attempt`` drives SAMPLE / plan_seed / item_id (a2, a3, …).
    Box-retry ``attempt`` always starts at 0 and does not count toward
    MAX_ATTEMPTS until the box fails and the controller increments it.
    """
    turn_raw = _rj(life_root() / 'epochs' / str(int(epoch)) / 'K1-TURN.json') or {}
    if not isinstance(turn_raw, Mapping):
        turn_raw = {}
    uid = int((old_item.uid if old_item else turn_raw.get('uid')) or 0)
    hotkey = (
        (old_item.hotkey_prefix if old_item else '')
        or str(turn_raw.get('hotkey') or '')[:16]
    )
    reg = int(old_item.registration_index if old_item else epoch)
    old_id = old_item.item_id if old_item else None
    ra = int(rerun_attempt)
    if ra < RERUN_ATTEMPT:
        raise RerunError('rerun_attempt must be >= %d' % RERUN_ATTEMPT)
    item_id = 'pit-rerun-e%d-a%d' % (int(epoch), ra)
    batch_meta = copy_rerun_batch_meta(old_item, epoch=int(epoch), turn=turn_raw)
    meta = {
        'rerun': True,
        'rerun_attempt': ra,
        'rerun_operator': operator,
        'rerun_reason': reason,
        'rerun_round_id': sample.get('round_id'),
        'rerun_plan_seed': sample.get('plan_seed'),
        'k1_turn_reuse': True,  # not a new K1 submission turn
        **batch_meta,
    }
    # Box attempt starts at 0 — independent of rerun_attempt / MAX_ATTEMPTS budget.
    box_attempt = 0
    rec = ItemRecord(
        item_id=item_id,
        registration_index=reg,
        epoch=int(epoch),
        uid=uid,
        hotkey_prefix=hotkey,
        state=ItemState.QUEUED,
        attempt=box_attempt,
        priority=RERUN_PRIORITY,
        rerun_of=old_id,
        shadow=bool(ctl.config.is_shadow()) if getattr(ctl, 'config', None) else True,
        meta=meta,
    )
    if hasattr(ctl, 'enqueue'):
        rec = ctl.enqueue(
            epoch=int(epoch),
            registration_index=reg,
            uid=uid,
            hotkey_prefix=hotkey,
            item_id=item_id,
            attempt=box_attempt,
            priority=RERUN_PRIORITY,
            rerun_of=old_id,
            meta=meta,
        )
    else:
        ctl.items.append(rec)
    store.save_item(rec)
    # Caller (consume under store.locked()) already holds pit.lock — never
    # re-enter via store.enqueue() (fcntl deadlock on a second fd).
    store._enqueue_unlocked(rec.item_id)
    return rec


def consume_rerun_request(
    epoch: int,
    *,
    store: DurableStore,
    ctl: Any,
    round_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Host path under pit.lock: archive → SAMPLE → enqueue rerun item.

    First rerun archives attempt-1 and enqueues rerun_attempt=2.
    Replacing an ABORTED/FAILED rerun without PIT-RESULT archives the failed
    SAMPLE into attempt-N/ and enqueues rerun_attempt=N+1 with a fresh seed.

    If archive already succeeded (RERUN.json with hashes) but SAMPLE/enqueue
    did not, resume without rewriting hashes. Never clobber a partial archive.
    """
    path = request_path(store.root, int(epoch))
    req = _rj(path)
    if not isinstance(req, Mapping) or req.get('status') != 'requested':
        raise RerunError('no pending RERUN-REQUEST for e%d' % int(epoch))
    check = preflight(int(epoch), store=store, items=list(ctl.items))
    if not check.ok:
        # Allow resume when archive already landed (preflight may see no item).
        archived = _archive_already_present(int(epoch), store)
        if archived is None:
            req = dict(req)
            req['status'] = 'refused'
            req['refused_at'] = float(now if now is not None else time.time())
            req['refuse_reasons'] = list(check.reasons)
            _wj(path, req, 0o600)
            raise RerunError('refuse consume e%d: %s' % (int(epoch), '; '.join(check.reasons)))

    old_item = check.item
    replace = is_failed_rerun_without_result(old_item, int(epoch))
    target_attempt = int(req.get('rerun_attempt') or 0) or next_rerun_attempt(old_item)
    if replace and target_attempt < RERUN_ATTEMPT + 1:
        target_attempt = next_rerun_attempt(old_item)

    if old_item is not None:
        ctl.items = [i for i in list(ctl.items) if i.item_id != old_item.item_id]

    archived = _archive_already_present(int(epoch), store)
    failed_sample_archive: Optional[Dict[str, Any]] = None
    if replace:
        if archived is None:
            raise RerunError(
                'replace-failed-rerun e%d requires existing attempt-1 RERUN.json'
                % int(epoch)
            )
        archive = archived
        # Keep attempt-1 audit; park the failed attempt's SAMPLE/COMMIT.
        failed_n = rerun_attempt_of(old_item) or RERUN_ATTEMPT
        failed_sample_archive = archive_failed_rerun_sample(
            int(epoch), failed_attempt=failed_n,
        )
        # Archive the failed item record under attempt-N/.
        if old_item is not None:
            item_arch = rerun_dir(store.root, int(epoch)) / ('attempt-%d' % failed_n)
            src = store.items_dir / ('%s.json' % old_item.item_id)
            if src.is_file():
                item_arch.mkdir(parents=True, exist_ok=True)
                dest = item_arch / ('%s.json' % old_item.item_id)
                if not dest.exists():
                    shutil.move(str(src), str(dest))
            q = [x for x in store.queue() if x != old_item.item_id]
            _wj(store.queue_path(), q, 0o600)
        # Extend RERUN.json with replace history (never rewrite attempt-1 hashes).
        rr_root = rerun_dir(store.root, int(epoch))
        hist = list(archive.get('replace_history') or [])
        hist.append({
            'failed_rerun_attempt': failed_n,
            'failed_item_id': old_item.item_id if old_item else None,
            'next_rerun_attempt': target_attempt,
            'sample_archive': failed_sample_archive,
            'at': float(now if now is not None else time.time()),
        })
        archive = dict(archive)
        archive['replace_history'] = hist
        archive['attempt_next'] = target_attempt
        _wj(rr_root / 'RERUN.json', archive, 0o600)
        gate_rr = gate_epochs_root() / str(int(epoch)) / 'RERUN.json'
        if gate_rr.is_file():
            _wj(gate_rr, archive, 0o600)
    elif archived is not None:
        archive = archived
        if old_item is None and archive.get('old_item_id'):
            # Reconstruct minimal pointer for enqueue meta.
            old_item = ItemRecord(
                item_id=str(archive['old_item_id']),
                registration_index=int(epoch),
                epoch=int(epoch),
                state=ItemState.RELEASED,
            )
            # Restore batch meta from archived item json when available.
            arch_item = (
                rerun_dir(store.root, int(epoch)) / 'attempt-1'
                / ('%s.json' % archive['old_item_id'])
            )
            if arch_item.is_file():
                adoc = _rj(arch_item)
                if isinstance(adoc, Mapping) and isinstance(adoc.get('meta'), Mapping):
                    old_item.meta = dict(adoc['meta'])
                    for key in ('uid', 'hotkey_prefix', 'registration_index'):
                        if adoc.get(key) is not None:
                            setattr(old_item, key, adoc[key])
    else:
        archive = archive_attempt1(
            int(epoch), store=store, item=old_item, now=now,
        )

    gate_dir = gate_epochs_root() / str(int(epoch))
    sample_path = gate_dir / 'SAMPLE.private.json'
    if sample_path.is_file() and not replace:
        sample = _rj(sample_path)
        if not isinstance(sample, Mapping) or int(sample.get('attempt') or 0) < target_attempt:
            raise RerunError('live SAMPLE present but not attempt-%d' % target_attempt)
        sample = dict(sample)
    else:
        # replace archives SAMPLE first; first-rerun resume may still need write.
        if sample_path.is_file() and replace:
            # Should have been archived; refuse to clobber if still present.
            raise RerunError(
                'SAMPLE.private.json still present after failed-rerun archive'
            )
        sample = write_fresh_sample(
            int(epoch),
            attempt=target_attempt,
            round_provider=round_provider,
        )

    def _is_active_target(i: ItemRecord) -> bool:
        if int(i.epoch) != int(epoch):
            return False
        if rerun_attempt_of(i) != target_attempt:
            return False
        # Only reuse a live/queued item for this attempt — never an aborted one.
        return i.state is ItemState.QUEUED or i.state in RENTED_STATES

    # Idempotent enqueue: reuse existing active target-attempt item if present.
    existing = [i for i in list(ctl.items) if _is_active_target(i)]
    if existing:
        new_item = existing[0]
    else:
        on_disk = [i for i in store.list_items() if _is_active_target(i)]
        if on_disk:
            new_item = on_disk[0]
            if all(i.item_id != new_item.item_id for i in ctl.items):
                ctl.items.append(new_item)
        else:
            new_item = enqueue_rerun_item(
                epoch=int(epoch),
                store=store,
                ctl=ctl,
                old_item=old_item,
                sample=sample,
                operator=str(req.get('operator') or ''),
                reason=str(req.get('reason') or ''),
                rerun_attempt=target_attempt,
            )

    # Separate marker: ingest skips; PIT-RESULT stays absent for max_pending.
    epoch_dir = life_root() / 'epochs' / str(int(epoch))
    write_rerun_active(
        int(epoch),
        item_id=new_item.item_id,
        rerun_of=old_item.item_id if old_item else archive.get('old_item_id'),
        rerun_attempt=target_attempt,
    )
    if (epoch_dir / 'PIT-RESULT.json').is_file():
        # Must not hide the rerun from backpressure / k1 readers.
        raise RerunError(
            'PIT-RESULT.json present during active rerun e%d — refuse '
            '(use RERUN-ACTIVE only)' % int(epoch)
        )

    done = dict(req)
    done.update({
        'status': 'consumed',
        'consumed_at': float(now if now is not None else time.time()),
        'archive': archive,
        'new_item_id': new_item.item_id,
        'sample_round_id': sample.get('round_id'),
        'plan_seed': sample.get('plan_seed'),
        'rerun_attempt': target_attempt,
        'box_attempt': int(new_item.attempt or 0),
        'replace_failed_rerun': bool(replace),
        'resumed': archived is not None and not replace,
    })
    _wj(path, done, 0o600)
    return done


def consume_pending_requests(
    *,
    store: DurableStore,
    ctl: Any,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Consume each pending RERUN-REQUEST; one failure must not stall the tick."""
    root = store.root / 'rerun'
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(root.glob('RERUN-REQUEST-e*.json')):
        doc = _rj(path)
        if not isinstance(doc, Mapping) or doc.get('status') != 'requested':
            continue
        try:
            epoch = int(doc['epoch'])
        except (TypeError, ValueError, KeyError) as exc:
            failed = dict(doc)
            failed.update({
                'status': 'failed',
                'error': 'malformed RERUN-REQUEST epoch: %s' % str(exc)[:200],
                'failed_at': time.time(),
            })
            try:
                _wj(path, failed, 0o600)
            except Exception:
                pass
            out.append({'ok': False, 'error': failed['error'], 'path': str(path)})
            continue
        try:
            out.append(consume_rerun_request(epoch, store=store, ctl=ctl, **kwargs))
        except Exception as exc:
            # Broad catch: AttributeError on GREENS shape etc. must not kill the tick.
            err = str(exc)[:400]
            try:
                failed = dict(doc)
                failed.update({
                    'status': 'failed',
                    'error': err,
                    'failed_at': time.time(),
                })
                _wj(path, failed, 0o600)
            except Exception:
                pass
            out.append({'epoch': epoch, 'ok': False, 'error': err})
    return out


def decision_archived(epoch: int) -> bool:
    """True when attempt-1 archived a DECISION (gate_writer may accept a fresh one)."""
    gate_dir = gate_epochs_root() / str(int(epoch))
    return (gate_dir / 'attempt-1' / 'DECISION.json').is_file()


def cancel_request_path(root: Path, epoch: int) -> Path:
    return root / 'rerun' / ('CANCEL-REQUEST-e%d.json' % int(epoch))


def write_cancel_request(
    *,
    epoch: int,
    operator: str,
    reason: str,
    store_root: Optional[Path] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Operator CANCEL-REQUEST: drop a pending/released rerun item cleanly."""
    if not operator or not str(operator).strip():
        raise RerunError('operator required')
    if not reason or not str(reason).strip():
        raise RerunError('reason required')
    root = Path(store_root) if store_root else pit_state_root()
    store = DurableStore(root)
    items = [i for i in (store.list_items() or []) if int(i.epoch) == int(epoch)]
    if not items:
        raise RerunError('no PIT item for epoch %d' % int(epoch))
    # Prefer active rerun / queued attempt-2+.
    target = None
    for it in items:
        if int(getattr(it, 'priority', 0) or 0) >= RERUN_PRIORITY or rerun_attempt_of(it) >= RERUN_ATTEMPT:
            target = it
            break
    if target is None:
        items.sort(key=lambda i: float((i.meta or {}).get('decided_at_mono') or 0), reverse=True)
        target = items[0]
    if target.state not in (ItemState.QUEUED, ItemState.RELEASED, ItemState.FAILED, ItemState.ABORTED, ItemState.NO_DECISION):
        if holds_rental(target) or target.state in RENTED_STATES:
            raise RerunError(
                'refuse cancel e%d: item still holds a box (state=%s)'
                % (int(epoch), target.state.value)
            )
    path = cancel_request_path(root, int(epoch))
    if path.is_file():
        existing = _rj(path) or {}
        if existing.get('status') in ('requested', 'consumed'):
            raise RerunError(
                'CANCEL-REQUEST already exists for e%d (status=%s)'
                % (int(epoch), existing.get('status'))
            )
    at = float(now if now is not None else time.time())
    doc = {
        'epoch': int(epoch),
        'operator': str(operator).strip(),
        'reason': str(reason).strip(),
        'status': 'requested',
        'requested_at': at,
        'item_id': target.item_id,
        'state': target.state.value,
        'decision': target.decision,
    }
    _wj(path, doc, 0o600)
    return doc


def consume_cancel_request(
    path: Path,
    *,
    store: DurableStore,
    ctl: Any,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Host path under pit.lock: drop the named rerun item from the queue."""
    doc = _rj(path) or {}
    if doc.get('status') != 'requested':
        return {'ok': False, 'error': 'not-requested', 'path': str(path)}
    epoch = int(doc['epoch'])
    at = float(now if now is not None else time.time())
    item = None
    for it in list(ctl.items):
        if it.item_id == doc.get('item_id') or int(it.epoch) == epoch:
            item = it
            break
    if item is None:
        doc['status'] = 'refused'
        doc['refused_at'] = at
        doc['error'] = 'item-missing'
        _wj(path, doc, 0o600)
        return {'ok': False, 'error': 'item-missing', 'epoch': epoch}
    if holds_rental(item) or item.state in RENTED_STATES:
        doc['status'] = 'refused'
        doc['refused_at'] = at
        doc['error'] = 'still-rented'
        _wj(path, doc, 0o600)
        return {'ok': False, 'error': 'still-rented', 'epoch': epoch}

    # Terminalize: leave audit + remove from active queue.
    meta = dict(item.meta or {})
    meta.update({
        'rerun_cancelled': True,
        'rerun_cancelled_at': at,
        'rerun_cancel_operator': str(doc.get('operator') or '')[:120],
        'rerun_cancel_reason': str(doc.get('reason') or '')[:400],
        'gate_emit_ready': True,
    })
    if item.state is ItemState.QUEUED and can_transition(item, ItemState.ABORTED):
        item = apply_transition(
            item, ItemState.ABORTED, reason='rerun-cancel',
            updates={
                'decision': item.decision or 'no_decision',
                'paid': False,
                'meta': meta,
            },
        )
    elif item.state is ItemState.DECIDED and can_transition(item, ItemState.RELEASED):
        item = apply_transition(
            item, ItemState.RELEASED, reason='rerun-cancel',
            updates={'box_id': None, 'box_name': None, 'meta': meta},
        )
    else:
        item.meta = meta
    ctl.items = [item if i.item_id == item.item_id else i for i in ctl.items]
    if hasattr(ctl, '_persist'):
        ctl._persist(item)
    else:
        store.save_item(item)

    audit = {
        'epoch': epoch,
        'item_id': item.item_id,
        'operator': doc.get('operator'),
        'reason': doc.get('reason'),
        'cancelled_at': at,
        'state': item.state.value,
        'decision': item.decision,
    }
    audit_dir = store.root / 'rerun' / ('e%d' % epoch)
    _wj(audit_dir / 'CANCEL.json', audit, 0o600)
    # Drop RERUN-ACTIVE so re-ingest is allowed again.
    active = store.root / 'RERUN-ACTIVE.json'
    try:
        if active.is_file():
            adoc = _rj(active) or {}
            if int(adoc.get('epoch') or -1) == epoch:
                active.unlink()
    except Exception:
        pass
    # Clear pending RERUN-REQUEST if still requested.
    try:
        rp = request_path(store.root, epoch)
        if rp.is_file():
            rdoc = _rj(rp) or {}
            if rdoc.get('status') == 'requested':
                rdoc['status'] = 'cancelled'
                rdoc['cancelled_at'] = at
                _wj(rp, rdoc, 0o600)
    except Exception:
        pass
    doc['status'] = 'consumed'
    doc['consumed_at'] = at
    doc['audit'] = audit
    _wj(path, doc, 0o600)
    return {'ok': True, 'epoch': epoch, 'item_id': item.item_id, 'note': 'cancelled'}


def consume_pending_cancels(*, store: DurableStore, ctl: Any) -> List[Dict[str, Any]]:
    root = store.root / 'rerun'
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(root.glob('CANCEL-REQUEST-e*.json')):
        doc = _rj(path) or {}
        if doc.get('status') != 'requested':
            continue
        try:
            out.append(consume_cancel_request(path, store=store, ctl=ctl))
        except Exception as exc:
            out.append({'ok': False, 'path': str(path), 'error': str(exc)[:300]})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='python -m ops.parallel_eval.pit.rerun')
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--operator', required=True)
    p.add_argument('--reason', required=True)
    p.add_argument('--dry-run', action='store_true', help='preflight only; do not write request')
    p.add_argument('--cancel', action='store_true', help='write CANCEL-REQUEST instead of RERUN-REQUEST')
    args = p.parse_args(list(argv) if argv is not None else None)
    try:
        if args.cancel:
            if args.dry_run:
                print(json.dumps({
                    'dry_run': True, 'cancel': True, 'epoch': int(args.epoch),
                }, indent=1, sort_keys=True))
                return 0
            doc = write_cancel_request(
                epoch=int(args.epoch),
                operator=str(args.operator),
                reason=str(args.reason),
            )
            print(json.dumps(doc, indent=1, sort_keys=True, default=str))
            return 0
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
