"""Declare a decided GREEN lost to infra (operator CLI + host consume).

Jacob 2026-11-10: when the first GREEN from head H cannot apply (box dead /
apply failed ≥30m without verified master), declare ``lost_infra``. The GREEN
stays paid, gets no HEAD, and leaves the first-GREEN slot so a later same-head
waiter with a verified master can apply.

CLI writes ``LOST-INFRA-REQUEST-e{N}.json``; host consumes under ``pit.lock``.
Never weakens fp32 / ROOT checks. Never promotes on a score timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .durable import DurableStore, pit_state_root
from .state import ItemRecord, ItemState


class LostInfraError(RuntimeError):
    """Operator / host lost-infra refusal."""


GREEN_APPLY_LOST_INFRA_S = 30 * 60


def _wj(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _rj(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def request_dir(root: Optional[Path] = None) -> Path:
    base = Path(root) if root else pit_state_root()
    return base / 'lost-infra'


def request_path(root: Optional[Path], epoch: int) -> Path:
    return request_dir(root) / ('LOST-INFRA-REQUEST-e%d.json' % int(epoch))


def audit_path(root: Optional[Path], epoch: int) -> Path:
    return request_dir(root) / ('e%d' % int(epoch)) / 'LOST-INFRA.json'


def find_item_for_epoch(store: DurableStore, epoch: int) -> Optional[ItemRecord]:
    items = list(store.list_items() or [])
    hits = [i for i in items if int(i.epoch) == int(epoch)]
    if not hits:
        return None
    # Prefer decided green_fresh / green_waiting still holding the slot.
    pref = [
        i for i in hits
        if i.decision in ('green_fresh', 'green', 'green_waiting')
        and i.state is ItemState.DECIDED
    ]
    if pref:
        pref.sort(key=lambda i: float((i.meta or {}).get('decided_at_mono') or 0))
        return pref[0]
    hits.sort(key=lambda i: float((i.meta or {}).get('decided_at_mono') or 0), reverse=True)
    return hits[0]


def write_lost_infra_request(
    *,
    epoch: int,
    operator: str,
    reason: str,
    store_root: Optional[Path] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    if not operator or not str(operator).strip():
        raise LostInfraError('operator required')
    if not reason or not str(reason).strip():
        raise LostInfraError('reason required')
    root = Path(store_root) if store_root else pit_state_root()
    store = DurableStore(root)
    item = find_item_for_epoch(store, int(epoch))
    if item is None:
        raise LostInfraError('no PIT item for epoch %d' % int(epoch))
    if item.decision not in ('green_fresh', 'green', 'green_waiting'):
        raise LostInfraError(
            'e%d decision=%r is not a GREEN holding the apply path'
            % (int(epoch), item.decision)
        )
    if (item.meta or {}).get('lost_infra') or item.decision == 'lost_infra':
        raise LostInfraError('e%d already declared lost_infra' % int(epoch))
    if (item.meta or {}).get('head_advanced'):
        raise LostInfraError('e%d already advanced HEAD — refuse lost_infra' % int(epoch))
    path = request_path(root, int(epoch))
    if path.is_file():
        existing = _rj(path) or {}
        if existing.get('status') in ('requested', 'consumed'):
            raise LostInfraError(
                'LOST-INFRA-REQUEST already exists for e%d (status=%s)'
                % (int(epoch), existing.get('status'))
            )
    at = float(now if now is not None else time.time())
    doc = {
        'epoch': int(epoch),
        'operator': str(operator).strip(),
        'reason': str(reason).strip(),
        'status': 'requested',
        'requested_at': at,
        'item_id': item.item_id,
        'candidate_id': item.candidate_id,
        'bound_head': item.bound_head,
        'decision': item.decision,
    }
    _wj(path, doc, 0o600)
    return doc


def list_pending_requests(root: Path) -> List[Path]:
    d = request_dir(root)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob('LOST-INFRA-REQUEST-e*.json')):
        doc = _rj(p) or {}
        if doc.get('status') == 'requested':
            out.append(p)
    return out


def stamp_item_lost_infra(
    item: ItemRecord,
    *,
    reason: str,
    now: float,
    operator: str = 'host',
) -> ItemRecord:
    """Mark item lost_infra: stays paid, no HEAD, leaves first-GREEN slot."""
    meta = dict(item.meta or {})
    meta.update({
        'lost_infra': True,
        'lost_infra_at': float(now),
        'lost_infra_reason': str(reason)[:400],
        'lost_infra_operator': str(operator)[:120],
        'green_apply_pending': False,
        'gate_emit_ready': True,
        'settlement': 'infra_loss_no_head',
    })
    meta.pop('apply_retry_at', None)
    meta.pop('waiting_on_leader', None)
    # Stay DECIDED until release; only stamp fields (DECIDED→DECIDED illegal).
    item.decision = 'lost_infra'
    item.paid = True
    item.fresh = False
    item.merge_blocked = True
    item.stranded = True
    item.meta = meta
    return item


def consume_lost_infra_request(
    path: Path,
    *,
    store: DurableStore,
    ctl: Any,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Host path under pit.lock: declare the named GREEN lost_infra."""
    doc = _rj(path) or {}
    if doc.get('status') != 'requested':
        return {'ok': False, 'error': 'not-requested', 'path': str(path)}
    epoch = int(doc['epoch'])
    at = float(now if now is not None else time.time())
    item = None
    for it in list(ctl.items):
        if it.item_id == doc.get('item_id') or int(it.epoch) == epoch:
            if it.decision in ('green_fresh', 'green', 'green_waiting', 'lost_infra'):
                item = it
                break
    if item is None:
        item = find_item_for_epoch(store, epoch)
    if item is None:
        doc['status'] = 'refused'
        doc['refused_at'] = at
        doc['error'] = 'item-missing'
        _wj(path, doc, 0o600)
        return {'ok': False, 'error': 'item-missing', 'epoch': epoch}
    if (item.meta or {}).get('head_advanced'):
        doc['status'] = 'refused'
        doc['refused_at'] = at
        doc['error'] = 'already-head-advanced'
        _wj(path, doc, 0o600)
        return {'ok': False, 'error': 'already-head-advanced', 'epoch': epoch}

    updated = stamp_item_lost_infra(
        item,
        reason=str(doc.get('reason') or 'operator-declare-lost'),
        now=at,
        operator=str(doc.get('operator') or 'operator'),
    )
    # Controller owns promote-waiter + release.
    if hasattr(ctl, 'declare_lost_infra'):
        updated, note = ctl.declare_lost_infra(
            updated, reason=str(doc.get('reason') or 'operator-declare-lost'),
            operator=str(doc.get('operator') or 'operator'),
        )
    else:
        note = 'stamped'
        ctl.items = [updated if i.item_id == updated.item_id else i for i in ctl.items]
        if hasattr(ctl, '_persist'):
            ctl._persist(updated)

    audit = {
        'epoch': epoch,
        'item_id': updated.item_id,
        'candidate_id': updated.candidate_id,
        'bound_head': updated.bound_head,
        'operator': doc.get('operator'),
        'reason': doc.get('reason'),
        'declared_at': at,
        'decision': updated.decision,
        'paid': updated.paid,
        'note': note,
    }
    _wj(audit_path(store.root, epoch), audit, 0o600)
    doc['status'] = 'consumed'
    doc['consumed_at'] = at
    doc['audit'] = audit
    _wj(path, doc, 0o600)
    return {'ok': True, 'epoch': epoch, 'item_id': updated.item_id, 'note': note}


def consume_pending_lost_infra(*, store: DurableStore, ctl: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in list_pending_requests(store.root):
        try:
            out.append(consume_lost_infra_request(path, store=store, ctl=ctl))
        except Exception as exc:
            out.append({'ok': False, 'path': str(path), 'error': str(exc)[:300]})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='python -m ops.parallel_eval.pit.lost_infra')
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--operator', required=True)
    p.add_argument('--reason', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(list(argv) if argv is not None else None)
    try:
        if args.dry_run:
            root = pit_state_root()
            store = DurableStore(root)
            item = find_item_for_epoch(store, int(args.epoch))
            print(json.dumps({
                'dry_run': True,
                'epoch': int(args.epoch),
                'item_id': getattr(item, 'item_id', None),
                'decision': getattr(item, 'decision', None),
                'candidate_id': getattr(item, 'candidate_id', None),
            }, indent=1, sort_keys=True))
            return 0 if item is not None else 2
        doc = write_lost_infra_request(
            epoch=int(args.epoch),
            operator=str(args.operator),
            reason=str(args.reason),
        )
        print(json.dumps(doc, indent=1, sort_keys=True, default=str))
        return 0
    except LostInfraError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
