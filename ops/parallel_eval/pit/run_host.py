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
from typing import Any, List, Mapping, Optional

from .apply_live import LiveApplicator
from .box import ScriptedBoxIO
from .config import load_pit_config
from .controller import Controller
from .cutover import plan_cutover
from .decision_emit import emit_decision, gate_epochs_root
from .durable import DurableStore, pit_state_root
from .gate_writer import GateWriteError, HostGateWriter, build_host_rounds
from .host_wiring import LiveWiringError, build_live_host_bundle
from .pe_score import nest_heldout_fields
from .queue_ingest import (
    IngestError,
    sync as queue_ingest_sync,
    write_ingest_alert,
    write_pit_result,
)


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_flag(path: Path) -> dict:
    raw = _rj(path, {}) or {}
    return raw


def _uploads_verified(item) -> bool:
    apply = (item.meta or {}).get('live_apply') or {}
    return bool(apply.get('candidate_verified') and apply.get('master_verified'))


def _item_needs_box(item) -> bool:
    """True until uploads are verified for GREEN, or while the item is undecided."""
    if item.state.value in ('released', 'aborted'):
        return False
    decision = (item.decision or '').lower()
    if decision in ('green_fresh', 'green'):
        return not _uploads_verified(item)
    if decision in ('red', 'green_stale', 'no_decision'):
        # Decision known and not a deferred-upload GREEN — box may be reaped after release.
        return bool(item.box_id) and item.state.value not in ('released',)
    # Not yet decided: keep the box.
    return item.state.value not in ('queued',)


def _bound_box_ids(ctl: Controller) -> set[str]:
    """Box ids still bound to in-flight / GREEN-pending items (never orphan-reap)."""
    known: set[str] = set()
    for item in ctl.items:
        if not _item_needs_box(item):
            continue
        if item.box_id:
            known.add(str(item.box_id))
        pod = (item.meta or {}).get('pod') or {}
        for key in ('id', 'box_id'):
            pid = pod.get(key)
            if pid:
                known.add(str(pid))
        name = item.box_name or pod.get('name')
        if name:
            known.add(str(name))
    return known


def _restore_box_bindings(ctl: Controller, listings: List[Any]) -> List[str]:
    """Re-attach box_id from meta/listing when a GREEN/undecided item lost it."""
    restored: List[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in listings:
        if not isinstance(row, Mapping):
            continue
        bid = str(row.get('id') or row.get('box_id') or '')
        name = str(row.get('name') or row.get('pod_name') or '')
        if bid:
            by_id[bid] = row
        if name:
            by_name[name] = row
    for item in ctl.items:
        if item.box_id or not _item_needs_box(item):
            continue
        pod = (item.meta or {}).get('pod') or {}
        bid = str(pod.get('id') or pod.get('box_id') or '')
        name = str(item.box_name or pod.get('name') or '')
        row = by_id.get(bid) or by_name.get(name)
        if row is None:
            continue
        item.box_id = str(row.get('id') or row.get('box_id') or bid or '')
        item.box_name = str(row.get('name') or row.get('pod_name') or name or '') or item.box_name
        if item.box_id:
            restored.append(item.box_id)
            ctl._persist(item)
    return restored


def _start_rent_reconcile(ctl: Controller) -> dict:
    """Adopt bound boxes; only reap listings with no in-flight / GREEN-pending item.

    Must run AFTER items are loaded onto the controller. Releasing a GREEN box
    whose candidate still lives only on the pod (candidate_on_box) destroys the
    only copy of C — never do that from startup orphan cleanup.
    """
    ctl.rent.load_ledger()
    listings = list(ctl.rent.list_pods() or []) if ctl.rent.list_pods is not None else []
    restored = _restore_box_bindings(ctl, listings)
    known = _bound_box_ids(ctl)
    # Names in known should also map to ids from listings for ledger adopt.
    for row in listings:
        if not isinstance(row, Mapping):
            continue
        bid = str(row.get('id') or row.get('box_id') or '')
        name = str(row.get('name') or row.get('pod_name') or '')
        if bid and (bid in known or name in known):
            known.add(bid)
        if name and (bid in known or name in known):
            known.add(name)
    out = ctl.rent.reconcile(listings, item_box_ids=known)
    out['restored_box_ids'] = restored
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
    if allow_unwired and cfg.is_shadow():
        ctl = Controller.from_flag(dict(parallel_eval))
        for item in store.list_items():
            ctl.items.append(item)
        return ctl

    # Real wiring for BOTH live and shadow: real pe_rent/pe_lane/pe_box/pe_score.
    # Shadow writes only to PIT_STATE (gate_writer.shadow=True); never live DECISION/HEAD.
    if cfg.enabled and (cfg.is_live() or cfg.is_shadow()):
        try:
            bundle = build_live_host_bundle(
                parallel_eval if 'pit' in parallel_eval else {'pit': (parallel_eval.get('pit') or cfg.as_dict())},
                validate_only=dry_run,
            )
        except LiveWiringError as exc:
            raise RuntimeError(str(exc)) from exc
        applicator = None
        if cfg.is_live():
            if bundle.root_signer is None:
                raise RuntimeError(
                    'live cutover refused: delegated host signer required '
                    '(AFFINE_PIT_HOST_KEY_FILE + AFFINE_PIT_HOST_CERT_FILE + AFFINE_PIT_ROOT_PUBKEY_HEX); '
                    'FixtureRootSigner is test-only'
                )
            applicator = LiveApplicator(
                config=cfg,
                verify_uploads=bundle.verify_uploads,
                accept_pit_decision=bundle.pe_apply_accept,
                destroy_state=bundle.destroy_box_state_fn,
                upload_master=bundle.upload_master,
                upload_candidate=bundle.upload_candidate,
                checkpoint_signer=bundle.root_signer,
            )
        ctl = Controller(
            config=cfg,
            rent=bundle.rent,
            lane=bundle.lane,
            box_io_factory=bundle.box_io_factory,
            root_signer=bundle.root_signer,
            score_fn=bundle.score_fn,
            plan_seed_fn=bundle.plan_seed_fn,
            indices_fn=bundle.indices_fn,
            round_id_fn=bundle.round_id_fn,
            live_applicator=applicator,
            destroy_box_state_fn=bundle.destroy_box_state_fn,
            durable_save_fn=store.save_item,
        )
        ledger = store.root / 'rent-ledger.json'
        ctl.rent.ledger_path = str(ledger)
    else:
        ctl = Controller.from_flag(dict(parallel_eval))

    # Load durable items BEFORE rent reconcile so GREEN/undecided boxes are
    # known and re-adopted — never orphan-released on host restart.
    for item in store.list_items():
        ctl.items.append(item)
    if (
        cfg.enabled
        and (cfg.is_live() or cfg.is_shadow())
        and not dry_run
        and getattr(ctl, 'rent', None) is not None
        and getattr(ctl.rent, 'ledger_path', None)
    ):
        _start_rent_reconcile(ctl)
    return ctl


def write_host_heartbeat(root: Path, *, now: Optional[float] = None) -> Path:
    """Affine-pit-host liveness marker for mining discovery under cutover.

    Discovery accepts this file (fresh ``at``) instead of a PE controller PID
    when ``parallel_eval.pit.cutover`` or ``WARM-OFF`` is set. Fail-closed
    elsewhere: other bindings still require the live lifecycle unit.
    """
    now = time.time() if now is None else float(now)
    path = Path(root) / 'HOST-HEARTBEAT.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        'version': 'affine-pit-host-heartbeat-v1',
        'at': now,
        'pid': os.getpid(),
    }
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def tick_once(
    ctl: Controller,
    store: DurableStore,
    *,
    dry_run: bool = False,
) -> dict:
    plan = plan_cutover(ctl.config)
    if not dry_run:
        write_host_heartbeat(store.root)
    try:
        ingest = queue_ingest_sync(ctl, store=store)
    except Exception as exc:
        # Never silently swallow ingest failures — a dead ingest must be visible.
        alert = write_ingest_alert(exc, root=store.root)
        print(
            json.dumps({
                'component': 'pit-run-host',
                'level': 'error',
                'event': 'ingest_failed',
                'error': str(exc)[:400],
                'alert': str(alert),
            }),
            file=sys.stderr,
            flush=True,
        )
        if isinstance(exc, IngestError):
            raise
        raise IngestError('tick_once ingest failed: %s' % str(exc)[:400]) from exc
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
        # Fill heldout from the live score shape (set_size / evaluated.n / …).
        scored = nest_heldout_fields(
            scored, round_id=str(item.bound_round_id or scored.get('round_id') or ''),
        )
        item.meta['scored'] = scored
        # HostGateWriter requires a complete heldout block — never silently skip.
        if not (scored.get('heldout') or {}).get('set_size'):
            print(
                json.dumps({
                    'component': 'pit-run-host',
                    'level': 'error',
                    'event': 'emit_heldout_incomplete',
                    'epoch': item.epoch,
                    'decision': item.decision,
                }),
                file=sys.stderr,
                flush=True,
            )
            continue
        try:
            gw = HostGateWriter(
                epochs_root=gate_epochs_root(),
                shadow=ctl.config.is_shadow(),
                rounds=build_host_rounds(),
            )
            path = emit_decision(item, scored=scored, gate_writer=gw, config=ctl.config)
        except (GateWriteError, RuntimeError, ValueError, OSError) as exc:
            # Keep decision_emitted unset so later ticks retry (e1055 recovery).
            print(
                json.dumps({
                    'component': 'pit-run-host',
                    'level': 'error',
                    'event': 'emit_decision_failed',
                    'epoch': item.epoch,
                    'item_id': item.item_id,
                    'error': str(exc)[:400],
                }),
                file=sys.stderr,
                flush=True,
            )
            item.meta = dict(item.meta or {}, last_emit_error=str(exc)[:300])
            store.save_item(item)
            continue
        if path:
            summary['advanced'].append({'epoch': item.epoch, 'decision': str(path)})
        # Miner-credit marker: queue_ingest skips once PIT-RESULT exists.
        epoch_dir = _epoch_dir_for_item(item)
        if epoch_dir:
            write_pit_result(
                Path(epoch_dir),
                decision=str(item.decision or 'no_decision'),
                item_id=item.item_id,
                extra={'paid': bool(item.paid), 'z': item.z},
            )
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
        # Once the decision is written, ensure the box is released (idempotent).
        if item.box_id:
            released, _note = ctl._release(item, reason='post-emit')
            ctl._replace(released)
            item = released
        store.save_item(item)
    return summary


def _epoch_dir_for_item(item) -> Optional[str]:
    meta = item.meta or {}
    if meta.get('epoch_dir'):
        return str(meta['epoch_dir'])
    for key in ('LIFE_ROOT', 'PE_LIFE'):
        root = os.environ.get(key)
        if root:
            return str(Path(root) / 'epochs' / str(int(item.epoch)))
    return None


def _gate_emit_ready(item) -> bool:
    """True when it is safe to write DECISION.json for this item.

    Do not treat bare ``decided`` as ready — that races emit before box release.
    ``gate_emit_ready`` is set in ``_finish_decision`` (after apply, with release).
    Recovery: decided+released with last_error still emits so turns/PIT-RESULT finish.
    """
    d = item.decision
    if not d:
        return False
    meta = item.meta or {}
    if d == 'green_fresh':
        apply = meta.get('live_apply') or {}
        return bool(
            apply.get('applied')
            and apply.get('master_verified')
            and apply.get('candidate_verified')
        )
    if d in ('green_stale', 'red', 'no_decision'):
        if meta.get('gate_emit_ready'):
            return True
        # e1055-style recovery: already released, decision known, emit unfinished.
        if item.state.value == 'released' and not meta.get('decision_emitted'):
            return True
        return False
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
