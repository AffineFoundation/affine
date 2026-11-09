"""Durable PIT item + decision state for the host daemon.

Files under PIT_STATE (default env, else PE_STATE/pit):
  items/<item_id>.json   — ItemRecord snapshots (0600)
  DECISIONS.json         — hash-chained DecisionLog
  queue.json             — registration order
  claims/PIT-CLAIM-e<N>.json — skip marker so pe_live avoids PE-lane double-dispatch
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional

from .decisions import DecisionLog
from .state import ItemRecord, ItemState


def pit_state_root(explicit: Optional[str] = None) -> Path:
    """Single PIT_STATE source of truth shared with pe_live claim checks.

    Fail closed when both PIT_STATE and PE_STATE are set but disagree
    (PIT_STATE must equal PE_STATE/pit).
    """
    if explicit:
        return Path(explicit)
    pit = os.environ.get('PIT_STATE')
    pe = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if pit and pe:
        expected = Path(pe) / 'pit'
        if Path(pit).resolve() != expected.resolve():
            raise RuntimeError(
                'PIT_STATE=%s disagrees with PE_STATE/pit=%s (fail closed)' % (pit, expected)
            )
    if pit:
        return Path(pit)
    if pe:
        return Path(pe) / 'pit'
    return Path('/home/const/subnet120-rewrite/state/root-audits/20261109-serverless/state')


def _wj(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def item_to_dict(item: ItemRecord) -> dict:
    return item.snapshot()


def item_from_dict(raw: dict) -> ItemRecord:
    state = raw.get('state') or ItemState.QUEUED.value
    if isinstance(state, ItemState):
        st = state
    else:
        st = ItemState(str(state))
    return ItemRecord(
        item_id=str(raw['item_id']),
        registration_index=int(raw.get('registration_index') or 0),
        epoch=int(raw['epoch']),
        uid=int(raw.get('uid') or 0),
        hotkey_prefix=str(raw.get('hotkey_prefix') or ''),
        state=st,
        bound_head=raw.get('bound_head'),
        bound_master_digest=raw.get('bound_master_digest'),
        bound_round_id=raw.get('bound_round_id'),
        bound_plan_seed=raw.get('bound_plan_seed'),
        candidate_id=raw.get('candidate_id'),
        box_id=raw.get('box_id'),
        box_name=raw.get('box_name'),
        attempt=int(raw.get('attempt') or 0),
        sealed=bool(raw.get('sealed')),
        decision=raw.get('decision'),
        z=raw.get('z'),
        fresh=raw.get('fresh'),
        merge_blocked=bool(raw.get('merge_blocked') or raw.get('stranded')),
        stranded=bool(raw.get('stranded') or raw.get('merge_blocked')),
        paid=bool(raw.get('paid')),
        result_stored=bool(raw.get('result_stored')),
        shadow=bool(raw.get('shadow', True)),
        last_error=raw.get('last_error'),
        meta=dict(raw.get('meta') or {}),
    )


class DurableStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else pit_state_root()
        self.items_dir = self.root / 'items'
        self.claims_dir = self.root / 'claims'
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.claims_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / 'DECISIONS.json'
        self._lock_path = self.root / 'pit.lock'
        self.decisions = self._load_decisions()

    def _load_decisions(self) -> DecisionLog:
        if self.log_path.is_file():
            return DecisionLog.load(self.log_path, lock_path=self._lock_path)
        return DecisionLog(lock_path=self._lock_path)

    def persist_decisions(self) -> None:
        self.decisions.save(self.log_path)

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            yield

    def save_item(self, item: ItemRecord) -> Path:
        path = self.items_dir / ('%s.json' % item.item_id)
        _wj(path, item_to_dict(item), 0o600)
        return path

    def load_item(self, item_id: str) -> Optional[ItemRecord]:
        raw = _rj(self.items_dir / ('%s.json' % item_id))
        if not raw:
            return None
        return item_from_dict(raw)

    def list_items(self) -> List[ItemRecord]:
        out = []
        for p in sorted(self.items_dir.glob('*.json')):
            raw = _rj(p)
            if raw:
                out.append(item_from_dict(raw))
        return out

    def write_claim(self, epoch: int, *, candidate: str, head: str, item_id: str) -> Path:
        path = self.claims_dir / ('PIT-CLAIM-e%d.json' % int(epoch))
        doc = dict(
            epoch=int(epoch), candidate=candidate, training_head=head,
            item_id=item_id, source='pit', at=time.time(),
        )
        _wj(path, doc, 0o600)
        # pe_live looks under PE_LIFE_STATE and PIT_STATE root for PIT-CLAIM.
        # Only mirror when env vars are non-empty (Path('') == '.' is truthy).
        pe_state_env = os.environ.get('PE_STATE') or ''
        pe_life_env = os.environ.get('PE_LIFE_STATE') or ''
        if pe_state_env:
            _wj(Path(pe_state_env) / ('PIT-CLAIM-e%d.json' % int(epoch)), doc, 0o600)
        if pe_life_env:
            _wj(Path(pe_life_env) / ('PIT-CLAIM-e%d.json' % int(epoch)), doc, 0o600)
        _wj(self.root / ('PIT-CLAIM-e%d.json' % int(epoch)), doc, 0o600)
        return path

    def queue_path(self) -> Path:
        return self.root / 'queue.json'

    def enqueue(self, item_id: str) -> None:
        with self.locked():
            q = _rj(self.queue_path(), []) or []
            if item_id not in q:
                q.append(item_id)
            _wj(self.queue_path(), q, 0o600)

    def queue(self) -> List[str]:
        return list(_rj(self.queue_path(), []) or [])
