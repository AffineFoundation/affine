#!/usr/bin/env python3
"""PIT live host daemon (Jacob 2026-11-09 decoupled dispatch).

Owns the batch queue under PIT_STATE (cap N boxes). Incoming batches go straight
here — lifecycle never holds waiting for PIT. Each box binds current head at
spawn; first fresh GREEN becomes HEAD; later old-head GREEN is paid+stranded;
everything else is destroyed. Writes PIT-CLAIM-eN so pe_live skips PE-lane
double-dispatch, emits gate-schema DECISION.json, applies GREEN_FRESH via
pe_apply.accept_pit_decision. k1/weights read those decisions.

ROOT path: delegated host key only (see ceremony_sign_host_key.py / docs/PIT.md).
Live config refuses root_mode=per_round. Never embeds key material.

Usage:
  PYTHONPATH=. python3 -m ops.parallel_eval.pit.run_host \\
    --flag /path/to/PROMOTION-GATE.json [--once] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from .apply_live import LiveApplicator
from .box import ScriptedBoxIO
from .config import load_pit_config
from .controller import Controller
from .cutover import plan_cutover
from .decision_emit import emit_decision, gate_epochs_root
from .durable import DurableStore, pit_state_root
from .gate_writer import HostGateWriter
from .host_wiring import LiveWiringError, build_live_host_bundle
from .queue_ingest import sync as queue_ingest_sync
def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_flag(path: Path) -> dict:
    raw = _rj(path, {}) or {}
    return raw


def _start_rent_reconcile(ctl: Controller) -> dict:
    ctl.rent.load_ledger()
    known = {
        str(i.box_id) for i in ctl.items
        if i.box_id and i.state.value not in ('released', 'aborted', 'queued')
    }
    listings = list(ctl.rent.list_pods() or []) if ctl.rent.list_pods is not None else []
    out = ctl.rent.reconcile(listings, item_box_ids=known)
    ctl.rent.save_ledger()
    return out


def build_controller(
    *,
    parallel_eval: Mapping[str, Any],
    store: DurableStore,
    dry_run: bool = False,
    allow_unwired: bool = False,
) -> Controller:
    cfg = load_pit_config(parallel_eval)
    if cfg.is_shadow() and allow_unwired:
        ctl = Controller.from_flag(dict(parallel_eval))
        for item in store.list_items():
            ctl.items.append(item)
        return ctl

    if cfg.is_live() and cfg.enabled:
        try:
            bundle = build_live_host_bundle(
                parallel_eval if 'pit' in parallel_eval else {'pit': (parallel_eval.get('pit') or cfg.as_dict())},
                validate_only=dry_run,
            )
        except LiveWiringError as exc:
            raise RuntimeError(str(exc)) from exc
        applicator = LiveApplicator(
            config=cfg,
            verify_uploads=bundle.verify_uploads,
            accept_pit_decision=bundle.pe_apply_accept,
            destroy_state=bundle.destroy_box_state_fn,
            upload_master=bundle.upload_master,
            upload_candidate=bundle.upload_candidate,
        )
        ctl = Controller(
            config=cfg,
            rent=bundle.rent,
            lane=bundle.lane,
            box_io_factory=bundle.box_io_factory,
            root_signer=bundle.root_signer,
            score_fn=bundle.score_fn,
            plan_seed_fn=bundle.plan_seed_fn,
            live_applicator=applicator,
            destroy_box_state_fn=bundle.destroy_box_state_fn,
        )
        ledger = store.root / 'rent-ledger.json'
        ctl.rent.ledger_path = str(ledger)
        if not dry_run:
            _start_rent_reconcile(ctl)
    else:
        ctl = Controller.from_flag(dict(parallel_eval))

    for item in store.list_items():
        ctl.items.append(item)
    return ctl


def tick_once(
    ctl: Controller,
    store: DurableStore,
    *,
    dry_run: bool = False,
) -> dict:
    plan = plan_cutover(ctl.config)
    ingest = queue_ingest_sync(ctl, store=store)
    for item in list(ctl.items):
        store.save_item(item)
        if item.bound_head and item.candidate_id and item.state.value not in ('released', 'aborted'):
            store.write_claim(
                item.epoch,
                candidate=item.candidate_id or '',
                head=item.bound_head or '',
                item_id=item.item_id,
            )
    if dry_run:
        return {'dry_run': True, 'items': len(ctl.items), 'plan': plan.as_dict(), 'ingest': ingest}

    summary: dict[str, Any] = {'plan': plan.as_dict(), 'advanced': [], 'ingest': ingest}
    if hasattr(ctl, 'tick'):
        out = ctl.tick()
        summary['tick'] = out
        ctl.rent.save_ledger()
    for item in list(ctl.items):
        store.save_item(item)
        if not ctl.config.may_write_live_artifacts():
            continue
        if item.meta.get('decision_emitted'):
            continue
        # GREEN DECISION.json only after verify/apply (gate_emit_ready). Never emit a
        # paid green for a candidate that failed upload verification.
        if not _gate_emit_ready(item):
            continue
        scored = dict(item.meta.get('scored') or {})
        if not scored and item.z is not None:
            scored = {'decision': item.decision, 'evaluated': {'z': item.z}}
        # HostGateWriter requires a complete heldout block; refuse incomplete.
        if not (scored.get('heldout') or {}).get('set_size'):
            summary.setdefault('emit_deferred', []).append({
                'epoch': item.epoch, 'why': 'heldout-incomplete',
            })
            continue
        gw = HostGateWriter(
            epochs_root=gate_epochs_root(),
            shadow=ctl.config.is_shadow(),
        )
        path = emit_decision(item, scored=scored, gate_writer=gw, config=ctl.config)
        if path:
            summary['advanced'].append({'epoch': item.epoch, 'decision': str(path)})
        store.decisions.append(
            epoch=item.epoch,
            uid=item.uid,
            hotkey_prefix=item.hotkey_prefix,
            binding_H=item.bound_head or '',
            binding_round_id=item.bound_round_id or '',
            candidate=item.candidate_id or '',
            decision=item.decision or 'no_decision',
            z=item.z,
            fresh=item.fresh,
            merge_blocked=item.merge_blocked,
            paid=item.paid,
            shadow=False,
            extra={'item_id': item.item_id},
        )
        store.persist_decisions()
        item.meta['decision_emitted'] = True
        store.save_item(item)
    return summary


def _gate_emit_ready(item) -> bool:
    """True when it is safe to write DECISION.json for this item."""
    d = item.decision
    if not d:
        return False
    meta = item.meta or {}
    if d == 'green_fresh':
        apply = meta.get('live_apply') or {}
        return bool(apply.get('applied') and apply.get('master_verified') and apply.get('candidate_verified'))
    if d in ('green_stale', 'red', 'no_decision'):
        # Terminal or post-finish; green_stale/red may emit from DECIDED after finish started.
        return bool(meta.get('gate_emit_ready')) or item.state.value in (
            'released', 'no_decision', 'decided',
        )
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='PIT live host daemon')
    ap.add_argument('--flag', required=True, help='PROMOTION-GATE / PE flag JSON')
    ap.add_argument('--state', default=None, help='PIT_STATE override')
    ap.add_argument('--once', action='store_true', help='single tick then exit')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-unwired', action='store_true', help='shadow only: skip live wiring validation')
    ap.add_argument('--interval', type=float, default=5.0)
    args = ap.parse_args(argv)

    flag_path = Path(args.flag)
    flag = load_flag(flag_path)
    pe = flag.get('parallel_eval') or flag
    if 'pit' not in pe and 'pit' in flag:
        pe = flag
    pe_body = pe if 'pit' in pe else {'pit': pe.get('pit', pe)}

    cfg = load_pit_config(pe_body)
    if args.allow_unwired and cfg.is_live():
        print(json.dumps({'error': '--allow-unwired is shadow-only'}), file=sys.stderr)
        return 2

    store = DurableStore(root=pit_state_root(args.state))
    try:
        ctl = build_controller(
            parallel_eval=pe_body,
            store=store,
            dry_run=args.dry_run,
            allow_unwired=args.allow_unwired,
        )
    except RuntimeError as exc:
        print(json.dumps({'error': str(exc)[:400]}), file=sys.stderr)
        return 2

    if ctl.config.is_live() and ctl.box_io_factory is ScriptedBoxIO:
        print(json.dumps({'error': 'live refuses ScriptedBoxIO'}), file=sys.stderr)
        return 2

    if args.once or args.dry_run:
        out = tick_once(ctl, store, dry_run=args.dry_run)
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
        return 0

    while True:
        try:
            tick_once(ctl, store, dry_run=False)
        except Exception as ex:
            print(json.dumps({'error': str(ex)[:400], 'at': time.time()}), flush=True)
        time.sleep(float(args.interval))


if __name__ == '__main__':
    raise SystemExit(main())
