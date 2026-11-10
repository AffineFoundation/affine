"""Ingest incoming batches into the PIT controller (Jacob 2026-11-09).

PIT owns dispatch: queued batches go straight here with a global box cap.
Sources (idempotent):
  1. ``PIT_STATE/inbox/*.json`` — host-written batch tickets (preferred)
  2. K1 turns under lifecycle epochs (uid/hotkey/reg) not yet queued
  3. Optional legacy ``life/pending/e*.json`` (ignored when ``pit_claimed``)

Real K1-TURN.json has ``epoch`` as a run-id string
(e.g. ``affine-mimo9b-grpo-20261013-e001050``); the integer is ``epoch_index``.

Do **not** enqueue the pre-cutover backlog: ``epoch_index >= CURRENT.epoch_index``
unless the epoch has ``PIT-DISPATCHED.json`` and no ``PIT-RESULT.json`` (cutover
finalizes and advances CURRENT within ~1s, so just-dispatched epochs sit below
the floor until PIT finishes). Require TRAINING-JOB + TRAIN-DISPATCH-FF (or K1);
skip TRAINING-RESULT / PIT-RESULT. ``registration_index = epoch_index``.

Head is refreshed from HEAD-CHAIN / CURRENT so each spawn binds the live head.
Lifecycle never feeds work by claiming epochs.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, TYPE_CHECKING

from .durable import pit_state_root
from .expected_master import lookup_expected_master
from .life_paths import pe_state_root

if TYPE_CHECKING:
    from .controller import Controller
    from .durable import DurableStore


class IngestError(RuntimeError):
    """Loud, fail-visible ingest failure (never silently swallowed)."""


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _pe_state_root() -> Path:
    return pe_state_root()


def _pit_state_root() -> Path:
    # Shared fail-closed source of truth with run_host / pe_live claim checks.
    return pit_state_root()


def _life_root() -> Optional[Path]:
    """Training lifecycle root (CURRENT.json + epochs/), or None if unset.

    Uses ``LIFE_ROOT``, then ``PE_LIFE``. Never ``PE_LIFE_STATE`` — that env
    is the PE life-state dir (pending/ + apply/) shared with pe_live /
    ``life_paths`` in the same host process. Confusing the two queued the
    backlog when PE_LIFE_STATE had no CURRENT.json.
    """
    for key in ('LIFE_ROOT', 'PE_LIFE'):
        env = os.environ.get(key)
        if env:
            return Path(env)
    return None


def _require_life_root() -> Path:
    root = _life_root()
    if root is None:
        raise IngestError(
            'LIFE_ROOT (or PE_LIFE) unset: training lifecycle root with CURRENT.json '
            'required; PE_LIFE_STATE is the PE pending/apply dir and must not be used here'
        )
    return root


def _gate_epochs_root() -> Path:
    env = os.environ.get('PE_GATE_EP')
    if env:
        p = Path(env)
        return p if p.name == 'epochs' else p / 'epochs'
    gate = os.environ.get('HG_GATE_STATE')
    if gate:
        return Path(gate) / 'epochs'
    raise RuntimeError(
        'PE_GATE_EP/HG_GATE_STATE unset (fail closed; no host-path defaults)'
    )


def _sample_round_id(epoch: int) -> Optional[str]:
    """Host-side SAMPLE.private.json['round_id'] for epoch (never sent to the box)."""
    try:
        root = _gate_epochs_root()
    except RuntimeError:
        return None
    sm = _rj(root / str(int(epoch)) / 'SAMPLE.private.json')
    if isinstance(sm, Mapping):
        rid = sm.get('round_id')
        if isinstance(rid, str) and rid and rid not in ('round-fixture', 'fixture'):
            return rid
    return None


def write_pit_result(epoch_dir: Path, *, decision: str, item_id: str = '', extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Durable PIT-RESULT marker — queue_ingest skips the epoch once present."""
    doc = dict(
        status='pit-result',
        decision=decision,
        item_id=item_id,
        at=time.time(),
    )
    if extra:
        doc.update(dict(extra))
    path = Path(epoch_dir) / 'PIT-RESULT.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.tmp' % path.name)
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    tmp.replace(path)
    return path


def _k1_turns_path() -> Path:
    env = os.environ.get('K1_TURNS')
    if env:
        return Path(env)
    raise RuntimeError(
        'K1_TURNS unset (fail closed; no host-path defaults)'
    )


def _epoch_index_of(obj: Mapping[str, Any], *, dir_name: Optional[str] = None) -> Optional[int]:
    """Real K1-TURN: integer is ``epoch_index``; ``epoch`` is a run-id string.

    Fall back to a digit directory name. Never ``int(epoch)`` on a run-id.
    """
    raw = obj.get('epoch_index')
    if raw is None and dir_name and str(dir_name).isdigit():
        raw = dir_name
    if raw is None:
        # Inbox / pending tickets may still use integer ``epoch``.
        cand = obj.get('epoch')
        if isinstance(cand, int) or (isinstance(cand, str) and cand.isdigit()):
            raw = cand
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def _current_epoch_floor() -> int:
    """CURRENT.json epoch_index — backlog below this must not enqueue.

    Fail closed: missing LIFE_ROOT/PE_LIFE or missing/unreadable CURRENT ⇒
    ``IngestError`` (never floor=0).
    """
    errors: List[str] = []
    life = _life_root()
    if life is None:
        raise IngestError(
            'LIFE_ROOT (or PE_LIFE) unset: cannot read CURRENT.json floor; '
            'refusing enqueue (never floor=0). PE_LIFE_STATE is the PE '
            'pending/apply dir and must not be used here'
        )
    for cur_path in (
        life / 'CURRENT.json',
        _pe_state_root() / 'CURRENT.json',
    ):
        if not cur_path.is_file():
            errors.append('%s missing' % cur_path)
            continue
        cur = _rj(cur_path, None)
        if not isinstance(cur, Mapping):
            errors.append('%s unreadable' % cur_path)
            continue
        if cur.get('epoch_index') is None:
            errors.append('%s missing epoch_index' % cur_path)
            continue
        try:
            return int(cur['epoch_index'])
        except (TypeError, ValueError):
            errors.append('%s bad epoch_index' % cur_path)
            continue
    raise IngestError(
        'CURRENT.json missing/unreadable under LIFE_ROOT/PE_LIFE (and PE_STATE); '
        'refusing enqueue (never floor=0): %s' % '; '.join(errors[:4])
    )


def _pit_dispatched_open(epoch_dir: Optional[Path]) -> bool:
    """True when cutover marked PIT-DISPATCHED and PIT has not finished yet."""
    if epoch_dir is None:
        return False
    d = Path(epoch_dir)
    if not d.is_dir():
        return False
    return (
        (d / 'PIT-DISPATCHED.json').is_file()
        and not (d / 'PIT-RESULT.json').is_file()
    )


def _above_floor_or_pit_open(epoch: int, floor: int, epoch_dir: Optional[Path]) -> bool:
    """Floor gate: PIT-DISPATCHED-without-RESULT always eligible; else epoch >= floor."""
    if int(epoch) >= int(floor):
        return True
    return _pit_dispatched_open(epoch_dir)


def _epoch_dir_ready(epoch_dir: Path) -> bool:
    """Require JOB+DISPATCH; skip once a PIT/standing RESULT exists.

    Cutover finalizes epochs as no-update AFTER writing TRAINING-JOB + marking
    ``PIT-DISPATCHED.json``. Those FINALIZED epochs must still be ingestible
    until PIT finishes (``PIT-RESULT.json``). Non-PIT FINALIZED epochs stay skipped.
    """
    if not (epoch_dir / 'TRAINING-JOB.json').is_file():
        return False
    if not (
        (epoch_dir / 'TRAIN-DISPATCH-FF.json').is_file()
        or (epoch_dir / 'TRAIN-DISPATCH-K1.json').is_file()
    ):
        return False
    if (epoch_dir / 'TRAINING-RESULT.json').is_file():
        return False
    if (epoch_dir / 'PIT-RESULT.json').is_file():
        return False
    # Sanctioned rerun in flight: do not re-enqueue. PIT-RESULT stays absent so
    # max_pending / backpressure still count the epoch.
    if (epoch_dir / 'RERUN-ACTIVE.json').is_file():
        return False
    if (epoch_dir / 'FINALIZED.json').is_file():
        # After lifecycle finalize: only PIT-DISPATCHED epochs remain for PIT.
        if not (epoch_dir / 'PIT-DISPATCHED.json').is_file():
            return False
    return True


def training_job_checkpoint(epoch: int) -> Optional[str]:
    """TRAINING-JOB checkpoint.id for stale-head checks before rent.

    Returns None when LIFE_ROOT/PE_LIFE is unset or the job file is absent
    (hermetic unit tests); never consults PE_LIFE_STATE.
    """
    life = _life_root()
    if life is None:
        return None
    ed = life / 'epochs' / str(int(epoch))
    job = _rj(ed / 'TRAINING-JOB.json')
    if not isinstance(job, Mapping):
        return None
    payload = job.get('payload') if isinstance(job.get('payload'), Mapping) else job
    if not isinstance(payload, Mapping):
        return None
    man = payload.get('manifest')
    if isinstance(man, Mapping):
        inner = man.get('payload') if isinstance(man.get('payload'), Mapping) else man
        ck = (inner or {}).get('checkpoint') if isinstance(inner, Mapping) else None
        if isinstance(ck, Mapping):
            cid = ck.get('id')
            if isinstance(cid, str) and len(cid) == 64:
                return cid
    ck2 = payload.get('checkpoint')
    if isinstance(ck2, Mapping):
        cid = ck2.get('id')
        if isinstance(cid, str) and len(cid) == 64:
            return cid
    return None


def write_ingest_alert(exc: BaseException, *, root: Optional[Path] = None) -> Path:
    """Persist a loud ingest failure so a dead ingest is visible (health/alert)."""
    doc = {
        'ok': False,
        'at_unix': time.time(),
        'error': str(exc)[:800],
        'error_type': type(exc).__name__,
        'traceback': traceback.format_exc()[-4000:],
    }
    alert = Path('/tmp/pit-INGEST-ALERT.json')
    try:
        pit = root or _pit_state_root()
        pit.mkdir(parents=True, exist_ok=True)
        alert = pit / 'INGEST-ALERT.json'
        health = pit / 'INGEST-HEALTH.json'
        tmp = alert.with_name('.%s.%d.tmp' % (alert.name, os.getpid()))
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
        os.chmod(tmp, 0o600)
        os.replace(tmp, alert)
        health.write_text(json.dumps({
            'ok': False, 'at_unix': doc['at_unix'], 'last_error': doc['error'],
        }, indent=1, sort_keys=True) + '\n')
    except OSError as write_exc:
        print(
            json.dumps({
                'component': 'pit-queue-ingest',
                'level': 'error',
                'event': 'ingest_alert_write_failed',
                'error': str(write_exc)[:200],
            }),
            file=sys.stderr,
            flush=True,
        )
    print(
        json.dumps({
            'component': 'pit-queue-ingest',
            'level': 'error',
            'event': 'ingest_failed',
            'error': doc['error'],
            'alert': str(alert),
        }),
        file=sys.stderr,
        flush=True,
    )
    return alert


def write_ingest_ok(*, root: Optional[Path] = None, summary: Optional[Mapping[str, Any]] = None) -> None:
    doc = {
        'ok': True,
        'at_unix': time.time(),
        'summary': dict(summary or {}),
    }
    try:
        pit = root or _pit_state_root()
        pit.mkdir(parents=True, exist_ok=True)
        health = pit / 'INGEST-HEALTH.json'
        alert = pit / 'INGEST-ALERT.json'
        health.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + '\n')
        if alert.is_file():
            try:
                alert.unlink()
            except OSError:
                pass
    except OSError:
        # Health file is best-effort; never fail the ingest itself on path/perms.
        pass


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
    # Lifecycle CURRENT is the live head under PIT cutover (LIFE_ROOT / PE_LIFE).
    cur_paths = []
    life = _life_root()
    if life is not None:
        cur_paths.append(life / 'CURRENT.json')
    cur_paths.append(st / 'CURRENT.json')
    for cur_path in cur_paths:
        cur = _rj(cur_path, {}) or {}
        head = (cur.get('checkpoint') or {}).get('id')
        if isinstance(head, str) and len(head) == 64:
            return head, int(cur.get('epoch_index') or 0)
    return None, 0


def _fp32_master_index_dir() -> Optional[Path]:
    """Published k1/fp32-master index dir (same source pe_box verifies on the box)."""
    for key in ('PIT_FP32_MASTER_INDEX', 'AFFINE_FP32_MASTER_INDEX'):
        v = os.environ.get(key)
        if v:
            return Path(v)
    return None


def _published_master_manifest(head: str) -> Optional[Mapping[str, Any]]:
    idx = _fp32_master_index_dir()
    if idx is None:
        return None
    path = idx / ('%s.json' % head)
    doc = _rj(path)
    return doc if isinstance(doc, Mapping) else None


def _master_for_head(head: str) -> Optional[str]:
    """Resolve M(H): PE_STATE/MASTER-INDEX.json, then PIT_FP32_MASTER_INDEX/<head>.json.

    Prod does not write MASTER-INDEX.json — the published website index is the
    durable source (same file pe_box verifies on the box). Fail closed (return
    None + loud alert) when neither source has a digest.
    """
    st = _pe_state_root()
    index = _rj(st / 'MASTER-INDEX.json', {}) or {}
    if isinstance(index, Mapping):
        dig = index.get(head)
        if isinstance(dig, str) and len(dig) == 64:
            return dig

    pub = _published_master_manifest(head)
    if isinstance(pub, Mapping):
        dig = pub.get('master_digest') or pub.get('digest')
        if isinstance(dig, str) and len(dig) == 64:
            return dig

    dig = lookup_expected_master(
        head,
        public_manifest=pub if isinstance(pub, Mapping) else None,
        index=index if isinstance(index, Mapping) else None,
        alert=lambda kind, detail: print(
            json.dumps({
                'component': 'pit-queue-ingest',
                'level': 'error',
                'event': kind,
                'detail': dict(detail),
            }),
            file=sys.stderr,
            flush=True,
        ),
    )
    if dig:
        return dig

    print(
        json.dumps({
            'component': 'pit-queue-ingest',
            'level': 'error',
            'event': 'master_unpublished',
            'head': head[:16],
            'note': (
                'neither PE_STATE/MASTER-INDEX.json nor PIT_FP32_MASTER_INDEX/<head>.json '
                'has M(H); pause dequeue/bind until the master is published (fail closed)'
            ),
            'pit_fp32_master_index': str(_fp32_master_index_dir() or ''),
        }),
        file=sys.stderr,
        flush=True,
    )
    return None


def _inbox_batches(*, floor: int) -> List[Dict[str, Any]]:
    """Host PIT inbox: one JSON ticket per queued batch."""
    rows: List[Dict[str, Any]] = []
    life = _life_root()
    for inbox in (_pit_state_root() / 'inbox', _pe_state_root() / 'pit' / 'inbox'):
        if not inbox.is_dir():
            continue
        for path in sorted(inbox.glob('*.json')):
            doc = _rj(path)
            if not isinstance(doc, Mapping):
                continue
            epoch = _epoch_index_of(doc)
            if epoch is None:
                continue
            ed = (life / 'epochs' / str(epoch)) if life is not None else None
            if not _above_floor_or_pit_open(epoch, floor, ed):
                continue
            batch_path = (
                doc.get('batch_path') or doc.get('batch') or doc.get('batch_file')
                or doc.get('turn_path')
            )
            rows.append(dict(
                epoch=epoch,
                registration_index=epoch,
                uid=int(doc.get('uid') or 0),
                hotkey_prefix=str(doc.get('hotkey_prefix') or doc.get('hotkey') or '')[:16],
                path=str(path),
                batch_path=batch_path,
                source='inbox',
                raw=dict(doc),
            ))
    return rows


def _k1_batch_rows(*, floor: int) -> List[Dict[str, Any]]:
    """Recent K1 turns as batch tickets — real-shaped epoch_index, no backlog."""
    rows: List[Dict[str, Any]] = []
    life = _life_root()
    if life is None:
        return rows
    life_epochs = life / 'epochs'
    if life_epochs.is_dir():
        for d in sorted(
            life_epochs.iterdir(),
            key=lambda p: int(p.name) if p.name.isdigit() else -1,
        ):
            if not d.name.isdigit():
                continue
            turn = _rj(d / 'K1-TURN.json')
            if not isinstance(turn, Mapping):
                continue
            epoch = _epoch_index_of(turn, dir_name=d.name)
            if epoch is None:
                continue
            if not _above_floor_or_pit_open(epoch, floor, d):
                continue
            if not _epoch_dir_ready(d):
                continue
            rows.append(dict(
                epoch=epoch,
                registration_index=epoch,
                uid=int(turn.get('uid') or 0),
                hotkey_prefix=str(turn.get('hotkey') or turn.get('hotkey_prefix') or '')[:16],
                path=str(d / 'K1-TURN.json'),
                batch_path=str(d),
                source='k1-turn',
                raw=dict(turn),
            ))
    # Append-only turns log (best-effort tail) — still require epoch_index.
    try:
        turns = _k1_turns_path()
    except RuntimeError:
        # K1_TURNS unset: skip the jsonl source (life epochs above still apply).
        return rows
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
            if not isinstance(turn, Mapping):
                continue
            epoch = _epoch_index_of(turn)
            if epoch is None:
                continue
            ed = life / 'epochs' / str(epoch)
            if not _above_floor_or_pit_open(epoch, floor, ed if ed.is_dir() else None):
                continue
            if ed.is_dir() and not _epoch_dir_ready(ed):
                continue
            rows.append(dict(
                epoch=epoch,
                registration_index=epoch,
                uid=int(turn.get('uid') or 0),
                hotkey_prefix=str(turn.get('hotkey') or turn.get('hotkey_prefix') or '')[:16],
                path=str(turns),
                batch_path=str(ed) if ed.is_dir() else None,
                source='k1-turns-jsonl',
                raw=dict(turn),
            ))
    return rows


def _legacy_pending_rows(*, floor: int) -> List[Dict[str, Any]]:
    """Ignore lifecycle pe_lifecycle claim stubs (pit_claimed); keep real trainer pending."""
    life = _life_root()
    rows: List[Dict[str, Any]] = []
    if life is None:
        return rows
    pending_dir = life / 'pending'
    if not pending_dir.is_dir():
        pending_dir = _pe_state_root() / 'life' / 'pending'
    if not pending_dir.is_dir():
        return rows
    for path in sorted(pending_dir.glob('e*.json')):
        doc = _rj(path)
        if not doc or doc.get('pit_claimed'):
            continue
        if not isinstance(doc, Mapping):
            continue
        stem = path.stem.lstrip('e')
        epoch = _epoch_index_of(doc, dir_name=stem if stem.isdigit() else None)
        if epoch is None:
            continue
        ed = life / 'epochs' / str(epoch)
        if not _above_floor_or_pit_open(epoch, floor, ed if ed.is_dir() else None):
            continue
        if ed.is_dir() and not _epoch_dir_ready(ed):
            continue
        rows.append(dict(
            epoch=epoch,
            registration_index=epoch,
            uid=int(doc.get('uid') or 0),
            hotkey_prefix=str(doc.get('hotkey_prefix') or '')[:16],
            path=str(path),
            batch_path=str(ed) if ed.is_dir() else None,
            source='pending',
            raw=dict(doc),
        ))
    return rows


def sync(
    controller: 'Controller',
    store: Optional['DurableStore'] = None,
) -> Dict[str, Any]:
    """Enqueue new batches and refresh controller head from durable state.

    Raises ``IngestError`` on unexpected failure — callers must log loudly
    (see ``run_host.tick_once``) and write INGEST-ALERT / INGEST-HEALTH.
    """
    try:
        return _sync_inner(controller, store=store)
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError('queue_ingest.sync failed: %s' % str(exc)[:400]) from exc


def _sync_inner(
    controller: 'Controller',
    store: Optional['DurableStore'] = None,
) -> Dict[str, Any]:
    known_epochs: Set[int] = {int(i.epoch) for i in controller.items}
    if store is not None:
        known_epochs |= {int(i.epoch) for i in store.list_items()}

    floor = _current_epoch_floor()
    tickets: List[Dict[str, Any]] = []
    tickets.extend(_inbox_batches(floor=floor))
    tickets.extend(_k1_batch_rows(floor=floor))
    tickets.extend(_legacy_pending_rows(floor=floor))
    # Prefer inbox over duplicates of the same epoch.
    seen: Set[int] = set()
    ordered: List[Dict[str, Any]] = []
    for row in tickets:
        e = int(row['epoch'])
        if e in seen:
            continue
        seen.add(e)
        ordered.append(row)
    # Order by epoch_index (= registration_index), never queue_position.
    ordered.sort(key=lambda r: (int(r['registration_index']), int(r['epoch'])))

    enqueued: List[int] = []
    skipped_backlog = 0
    life = _require_life_root()
    for row in ordered:
        epoch = int(row['epoch'])
        ed = life / 'epochs' / str(epoch)
        if not _above_floor_or_pit_open(epoch, floor, ed if ed.is_dir() else None):
            skipped_backlog += 1
            continue
        if epoch in known_epochs:
            continue
        item = controller.enqueue(
            epoch=epoch,
            registration_index=int(row.get('registration_index') or epoch),
            uid=int(row.get('uid') or 0),
            hotkey_prefix=str(row.get('hotkey_prefix') or ''),
        )
        batch_path = row.get('batch_path') or row.get('path')
        item.meta = dict(
            item.meta or {},
            ingest_source=row.get('source'),
            batch_path=batch_path,
            ticket_path=row.get('path'),
            epoch_dir=str(ed),
        )
        known_epochs.add(epoch)
        enqueued.append(epoch)
        if store is not None:
            store.save_item(item)
            store.enqueue(item.item_id)

    head, head_epoch = _head_chain_tip()
    head_set = False
    master_missing = False
    master = None
    if head:
        master = _master_for_head(head)
        if master:
            # Prefer SAMPLE.private.json round_id for the head epoch (same source
            # live pe_score uses). Never invent a fixture; binding refuses if missing.
            round_id = _sample_round_id(head_epoch) or ''
            controller.set_head(
                head,
                epoch=head_epoch,
                master_digest=master,
                round_id=round_id or None,
            )
            head_set = True
        else:
            master_missing = True
            print(
                json.dumps({
                    'component': 'pit-queue-ingest',
                    'level': 'warning',
                    'event': 'head_set_paused_waiting_master',
                    'head': head[:16],
                    'head_epoch': head_epoch,
                    'note': (
                        'head known but M(H) unpublished — pause bind/rent until '
                        'PIT_FP32_MASTER_INDEX/<head>.json (or MASTER-INDEX) appears'
                    ),
                }),
                file=sys.stderr,
                flush=True,
            )

    out = dict(
        enqueued=enqueued,
        head=head,
        head_epoch=head_epoch,
        head_set=head_set,
        master=master,
        master_missing=master_missing,
        current_epoch_floor=floor,
        skipped_backlog=skipped_backlog,
        sources=['inbox', 'k1', 'pending'],
    )
    health_root = store.root if store is not None else None
    write_ingest_ok(root=health_root, summary=out)
    return out
