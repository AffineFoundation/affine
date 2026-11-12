"""Rent and release Lium boxes that have a name no live tool matches.

Terms:

* live tool: a program of the live subnet that finds its pods by name. All of
  them match names that start with ``affine-`` (the PE lanes, the warm pool,
  the PE controller and guard match ``affine-pe-``; ``replace_pod`` lists
  ``affine-`` pods).
* isolated box: a shadow box or a fakepass box. Its name starts with
  ``sn120-shadow-pit-`` or ``sn120-fakepass-pit-``.

Live ``pe_rent.rent`` and ``pe_rent.release`` refuse an isolated name, and
``our_pods`` does not list an isolated box. Thus an isolated box does not use an
eval-node slot of live, and no live program can adopt or release it. This module
is the rent path for those boxes.

``IsolatedRent`` does the same Lium calls as live ``pe_rent`` (``lium up`` with
8 GPUs and no backup, ``lium rm`` by pod id). It refuses a name that is not its
own. Its lock, pod ledger and events are files in ``state_dir``, which is never
a live directory.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from . import LIVE_NAME_ROOT, PREFIX_SHADOW

PREFIX_FAKEPASS = 'sn120-fakepass-pit-'
ISOLATED_PREFIXES = (PREFIX_SHADOW, PREFIX_FAKEPASS)
# Names of shadow and fakepass boxes before 2026-11-12. They start with the live
# PIT prefix; live PIT does not own, count or release a box with such a name.
LEGACY_ISOLATED_PREFIXES = ('affine-pe-pit-shadow-', 'affine-pe-pit-fakepass-')
RENT_TTL = '4h'


class IsolatedRentError(RuntimeError):
    """The isolated rent path refused an operation."""


def live_name_problem(name: Any) -> Optional[str]:
    """Why a live tool can match this name. None: no live tool matches it."""
    if not isinstance(name, str) or not name:
        return 'the name is empty'
    if name.lower().startswith(LIVE_NAME_ROOT):
        return 'the name starts with %r, and live tools match such names' % LIVE_NAME_ROOT
    return None


def pod_record(
    pod: Mapping[str, Any],
    *,
    gpu: Optional[str] = None,
    price: Any = None,
    country: Any = None,
    tier: Any = None,
    rented_at: Optional[float] = None,
) -> Dict[str, Any]:
    """One row of ``lium ps`` or ``lium up`` as the pod dict that the PE code uses."""
    ip = pod.get('ip') or pod.get('ssh_ip')
    port = pod.get('ssh_port') or pod.get('port')
    command = pod.get('ssh_command') or ''
    if (not ip or not port) and 'ssh' in command:
        found = re.search(r'-p\s+(\d+).*?root@([0-9.]+)', command)
        if found:
            port, ip = int(found.group(1)), found.group(2)
    at = float(rented_at if rented_at is not None else time.time())
    return dict(
        id=pod.get('id'), huid=pod.get('huid'), name=pod.get('name'), ip=ip, port=int(port or 0),
        gpu=gpu or pod.get('gpu_model') or pod.get('gpu_name'),
        price_h=price if price is not None else pod.get('price_per_hour'),
        country=country or pod.get('country_code'), tier=tier or pod.get('tier'),
        rented_at=at, status=pod.get('status'),
    )


@dataclass
class IsolatedRent:
    """Rent, list and release the boxes of one isolated name prefix."""

    prefix: str
    state_dir: Union[Path, Callable[[], Path]]      # a callable is asked when a state file is needed
    lium: Callable[..., Tuple[int, str, str]]
    lium_ps: Callable[[], List[Mapping[str, Any]]]
    pick_node: Callable[..., Tuple[Optional[str], Optional[Mapping[str, Any]]]]
    ssh_ok: Optional[Callable[[str, int, str], bool]] = None
    never_names: Tuple[str, ...] = ()
    gpu_order: Tuple[str, ...] = ()
    price_ceiling: Optional[float] = None
    ssh_tries: int = 60
    ssh_wait_s: float = 5.0
    adopt_wait_s: float = 360.0
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if not callable(self.state_dir):
            self.state_dir = Path(self.state_dir)
        problem = live_name_problem(self.prefix)
        if problem:
            raise IsolatedRentError('refused the prefix %r: %s' % (self.prefix, problem))

    # -- names -------------------------------------------------------------
    def owns(self, name: Any) -> bool:
        return isinstance(name, str) and name.startswith(self.prefix)

    def _never(self, *texts: Any) -> bool:
        return any(n in str(text or '') for text in texts for n in self.never_names)

    def _check_name(self, name: Any) -> None:
        if not self.owns(name):
            raise IsolatedRentError('refused the name %r: it does not start with %s' % (name, self.prefix))
        if self._never(name):
            raise IsolatedRentError('refused the name %r: it has a protected pod name in it' % (name,))

    # -- state files -------------------------------------------------------
    def _dir(self) -> Path:
        """The state directory. Listing pods does not need it; a rent, a release and an event do."""
        return Path(self.state_dir()) if callable(self.state_dir) else self.state_dir

    def _ledger_path(self) -> Path:
        return self._dir() / 'pods.json'

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            doc = json.loads(self._ledger_path().read_text())
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    def _save(self, ledger: Mapping[str, Any]) -> None:
        path = self._ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
        tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True, default=str) + '\n')
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def _event(self, event: str, **fields: Any) -> None:
        state = self._dir()
        state.mkdir(parents=True, exist_ok=True)
        row = dict(fields, event=event, at=float(self.now()), prefix=self.prefix)
        with open(state / 'events.jsonl', 'a') as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + '\n')

    # -- Lium --------------------------------------------------------------
    def pods(self) -> List[Mapping[str, Any]]:
        """The pods of this prefix that Lium lists. A live pod is never in this list."""
        return [p for p in (self.lium_ps() or []) if isinstance(p, Mapping) and self.owns(p.get('name'))]

    def _find(self, name: str, pod_id: Any = None) -> Optional[Mapping[str, Any]]:
        for pod in self.pods():
            if pod.get('name') == name or (pod_id and pod.get('id') == pod_id):
                return pod
        return None

    def rent(self, name: str, ttl: str = RENT_TTL, *, gpu: Optional[str] = None, **_ignored: Any) -> Optional[Dict[str, Any]]:
        """Rent one 8-GPU box with this name. -> the pod dict, or None when Lium has no offer or the rental failed."""
        self._check_name(name)
        state = self._dir()
        state.mkdir(parents=True, exist_ok=True)
        with open(state / 'rent.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prefer = [gpu] if gpu else (list(self.gpu_order) or None)
            got_gpu, row = self.pick_node(prefer, fallback=False, price_ceiling=self.price_ceiling)
            if not row:
                self._event('rent_no_supply', name=name, gpu=gpu)
                return None
            huid = row['huid']
            price, country, tier = row.get('price_per_hour'), row.get('country_code'), row.get('tier')
            started = float(self.now())
            self._event('rent_attempt', name=name, huid=huid, gpu=got_gpu, price_h=price)
            rc, out, err = self.lium(
                'up', huid, '-c', '8', '--name', name, '--verify-gpus', '--strict-gpus',
                '--no-volume-encryption', '--no-backup', '-y', '--json', '--timeout', '900',
                '--ttl', ttl, timeout=1000,
            )
            details = dict(gpu=got_gpu, price=price, country=country, tier=tier, rented_at=started)
            if rc != 0:
                tail = (err or out or '')[-400:]
                self._event('rent_failed', name=name, huid=huid, err=tail)
                if 'still being processed' not in tail and 'earlier request' not in tail:
                    return None
                # Lium accepted the request and answers later: wait for the pod with this name.
                pod = None
                deadline = float(self.now()) + float(self.adopt_wait_s)
                while pod is None and float(self.now()) < deadline:
                    self.sleep(15)
                    pod = self._find(name)
                if pod is None:
                    return None
                self._event('rent_adopted', name=name, id=pod.get('id'), huid=pod.get('huid'))
            else:
                try:
                    raw = json.loads(out)
                    pod = raw.get('pod') if isinstance(raw, dict) and 'pod' in raw else raw
                except ValueError:
                    pod = None
                if not isinstance(pod, Mapping) or not pod:
                    self.sleep(5)
                    pod = self._find(name)
                if not pod:
                    self._event('rent_parse_fail', name=name, out=(out or '')[-300:], err=(err or '')[-300:])
                    return None
            record = pod_record(pod, **details)
            if not record.get('id') or not record.get('ip') or not record.get('port'):
                self.sleep(3)
                listed = self._find(name, record.get('id'))
                if listed is not None:
                    record = pod_record(listed, **details)
            record.update(name=record.get('name') or name, executor=huid, auto_backup_disabled=True)
            if record['name'] != name:
                # Do not use and do not remove a pod that has a different name: only a pod of this
                # prefix is removed. The TTL of the rental ends it.
                self._event('rent_wrong_name', name=name, got=record['name'], id=record.get('id'))
                raise IsolatedRentError('Lium gave the pod the name %r, not %r' % (record['name'], name))
            for _ in range(int(self.ssh_tries) if self.ssh_ok is not None else 0):
                try:
                    if record.get('ip') and record.get('port') and self.ssh_ok(record['ip'], record['port'], record['id']):
                        break
                except Exception:
                    pass
                self.sleep(self.ssh_wait_s)
            record['ready_at'] = float(self.now())
            record['rent_to_ready_s'] = round(record['ready_at'] - started, 1)
            ledger = self._load()
            ledger[name] = record
            self._save(ledger)
            self._event('rented', name=name, id=record.get('id'), huid=record.get('huid'), gpu=got_gpu,
                        price_h=record.get('price_h'), rent_to_ready_s=record['rent_to_ready_s'])
            return record

    def release(self, name_or_id: str, reason: str = 'done') -> bool:
        """Remove one box of this prefix. -> True when Lium removed it. A pod of a different name is never removed."""
        ledger = self._load()
        record: Optional[Mapping[str, Any]] = None
        key = None
        for name, row in ledger.items():
            if name == name_or_id or row.get('id') == name_or_id or row.get('name') == name_or_id:
                record, key = row, name
                break
        listed: Optional[List[Mapping[str, Any]]]
        try:
            listed = [p for p in (self.lium_ps() or []) if isinstance(p, Mapping)]
        except Exception:
            listed = None
        if record is None and listed is not None:
            for pod in listed:
                if self.owns(pod.get('name')) and name_or_id in (pod.get('name'), pod.get('id')):
                    record, key = pod_record(pod), pod.get('name')
                    break
        if not record or not record.get('id'):
            self._event('release_missing', name=name_or_id, reason=reason)
            return False
        name = str(record.get('name') or '')
        now_listed = next((p for p in (listed or []) if p.get('id') == record['id']), None)
        foreign = now_listed is not None and not self.owns(now_listed.get('name'))
        if not self.owns(name) or foreign or self._never(name, record.get('huid')):
            self._event('release_refused', name=name, id=record.get('id'), reason=reason,
                        listed_name=(now_listed or {}).get('name'))
            return False
        rc, out, err = self.lium('rm', record['id'], '-y', timeout=120)
        ok = rc == 0
        self._event('released' if ok else 'release_failed', name=name, id=record['id'], reason=reason,
                    err=None if ok else (err or out or '')[-200:])
        if key in ledger:
            ledger[key] = dict(ledger[key], released=ok, released_at=float(self.now()), release_reason=reason)
            self._save(ledger)
        return ok


def bind_isolated_rent(
    mod: Any,
    *,
    prefix: str,
    state_dir: Union[Path, Callable[[], Path]],
    gpu_order: Tuple[str, ...] = (),
    price_ceiling: Optional[float] = None,
) -> IsolatedRent:
    """Build the isolated rent path on the Lium helpers of the host ``pe_rent`` module.

    Only ``lium``, ``lium_ps``, ``pick_node`` and the ssh check are used. Live
    ``rent``, ``release`` and ``our_pods`` are not: they are for live names.
    """
    missing = [n for n in ('lium', 'lium_ps', 'pick_node') if not callable(getattr(mod, n, None))]
    if missing:
        raise IsolatedRentError('host pe_rent module lacks %s' % ', '.join(missing))
    pods_tool = getattr(mod, 'R', None)
    ssh_ok = None
    if callable(getattr(pods_tool, 'ssh_ok', None)):
        ssh_ok = lambda ip, port, pod_id: pods_tool.ssh_ok(ip, port, pod_id)
    # The helpers are looked up at each call: a host that replaces one (a test fixture) is still used.
    return IsolatedRent(
        prefix=prefix,
        state_dir=state_dir,
        lium=lambda *args, **kw: mod.lium(*args, **kw),
        lium_ps=lambda: mod.lium_ps(),
        pick_node=lambda *args, **kw: mod.pick_node(*args, **kw),
        ssh_ok=ssh_ok,
        never_names=tuple(getattr(mod, 'NEVER_NAMES', ()) or ()),
        gpu_order=tuple(gpu_order),
        price_ceiling=None if price_ceiling is None else float(price_ceiling),
    )
