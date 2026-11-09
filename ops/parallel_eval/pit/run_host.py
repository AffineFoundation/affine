#!/usr/bin/env python3
"""PIT live host daemon (CUTOVER-PLAN blocker 1).

Owns the item queue under PIT_STATE, writes PIT-CLAIM-eN so pe_live skips the
epoch, drives Controller ticks, emits gate-schema DECISION.json, and applies
GREEN_FRESH via pe_apply.accept_pit_decision.

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
from .config import load_pit_config
from .controller import Controller
from .cutover import plan_cutover
from .decision_emit import emit_decision
from .durable import DurableStore, pit_state_root
from .signer import LIVE_ROOT_MODE, load_delegated_host_signer


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_flag(path: Path) -> dict:
    raw = _rj(path, {}) or {}
    return raw


def bind_pe_apply_hooks(applicator: LiveApplicator) -> None:
    """Wire live pe_apply.accept_pit_decision when importable on the host."""
    try:
        pe_dir = Path(os.environ.get('PE_CODE') or '')
        if pe_dir and str(pe_dir) not in sys.path:
            sys.path.insert(0, str(pe_dir))
        import pe_apply  # type: ignore
        applicator.accept_pit_decision = pe_apply.accept_pit_decision
    except Exception:
        # Unit / dry-run hosts may lack the live module; apply_live still
        # requires verify_uploads in live (fail closed).
        pass


def build_controller(
    *,
    parallel_eval: Mapping[str, Any],
    store: DurableStore,
    dry_run: bool = False,
) -> Controller:
    cfg = load_pit_config(parallel_eval)
    if cfg.is_live() and not dry_run:
        # Live refuses anything but delegated host key (also enforced in load_pit_config).
        if cfg.root_mode != LIVE_ROOT_MODE:
            raise RuntimeError('live ROOT path is delegated_host only; refused root_mode=%r' % cfg.root_mode)
        signer = load_delegated_host_signer()
    else:
        signer = None
    ctl = Controller.from_flag(dict(parallel_eval), root_signer=signer)
    # Restore durable items into the in-memory controller.
    for item in store.list_items():
        ctl.items.append(item)
    if cfg.is_live() and ctl.live_applicator is not None and not dry_run:
        bind_pe_apply_hooks(ctl.live_applicator)
        # Fail-closed: leave verify_uploads unset until the host wires a real
        # verifier — apply_live will refuse GREEN without it.
    return ctl


def tick_once(
    ctl: Controller,
    store: DurableStore,
    *,
    dry_run: bool = False,
) -> dict:
    plan = plan_cutover(ctl.config)
    # Persist any in-memory items that were enqueued externally.
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
        return {'dry_run': True, 'items': len(ctl.items), 'plan': plan.as_dict()}
    # Drive one controller pass if available.
    summary: dict[str, Any] = {'plan': plan.as_dict(), 'advanced': []}
    if hasattr(ctl, 'tick'):
        out = ctl.tick()
        summary['tick'] = out
    for item in list(ctl.items):
        store.save_item(item)
        if item.decision and ctl.config.may_write_live_artifacts():
            if item.meta.get('decision_emitted'):
                continue
            scored = dict(item.meta.get('scored') or {})
            if not scored and item.z is not None:
                scored = {'decision': item.decision, 'evaluated': {'z': item.z}}
            path = emit_decision(item, scored=scored, config=ctl.config)
            if path:
                summary['advanced'].append({'epoch': item.epoch, 'decision': path.name})
            store.decisions.append(
                epoch=item.epoch,
                uid=item.uid,
                hotkey_prefix=item.hotkey_prefix,
                binding_H=item.bound_head or '',
                binding_round_id=item.bound_round_id or '',
                candidate=item.candidate_id or '',
                decision=item.decision,
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='PIT live host daemon')
    ap.add_argument('--flag', required=True, help='PROMOTION-GATE / PE flag JSON')
    ap.add_argument('--state', default=None, help='PIT_STATE override')
    ap.add_argument('--once', action='store_true', help='single tick then exit')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--interval', type=float, default=5.0)
    args = ap.parse_args(argv)

    flag_path = Path(args.flag)
    flag = load_flag(flag_path)
    pe = flag.get('parallel_eval') or flag
    # Accept either top-level {pit:...} or nested parallel_eval.
    if 'pit' not in pe and 'pit' in flag:
        pe = flag
    store = DurableStore(root=pit_state_root(args.state))
    ctl = build_controller(parallel_eval=pe if 'pit' in pe else {'pit': pe.get('pit', pe)}, store=store, dry_run=args.dry_run)

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
