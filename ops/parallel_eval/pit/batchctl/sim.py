"""Fault-injection harness: a hostile fake world around the real controller.

Real code under test: ``Controller``, ``Effects``, ``Journal``, ``core``,
``guard``, ``layout``, ``timers``. Nothing of it is stubbed.

Fake parts: the object store (R2), the providers, the boxes with their jobs,
the batch source, the clock, and the file system under the journal. The
fakes do what the real parts can do on a bad day:

* a box dies, hangs, pauses and comes back as a zombie, uploads wrong bytes,
  leaves a file out, writes ``COMPLETE.json`` too early or for another fence;
* a store or provider request is lost, or is applied and its answer is lost,
  or lands much later (even after the controller that sent it is dead), or
  lands twice; one read gives wrong bytes, a wrong digest, or says that an
  object is missing when it is there;
* a rental makes no box, a late box, two boxes, the wrong GPU, or a price
  above the ceiling; a listing misses a box;
* an effect result is delivered late, twice, or never;
* the controller process dies at any call: before, in the middle of, or
  after any journal write or fsync, and around any store, provider, agent or
  source call. A crash can be a process kill (written bytes stay) or a power
  loss (bytes that were not fsynced can vanish; a rename that was not made
  durable can be undone).

``Auditor`` checks the invariants against the fake world's ground truth,
not against what the controller believes. Some checks run at the moment of
an action (for example at each pointer move). The rest run at the end.

The base rule is checked from three sides. What the controller sends: every
box of a batch gets the base of the first box. What the journal says: a
batch gets a new box only after its last box was lost before a decision, and
it gets one decision. What is true in the fake world: the verdict and the
payout are the ones against that first base, and no box is lost in a world
that has no fault.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import core as C
from .controller import Config, Controller, op_key
from .effects import APPLY_VERIFY_SHA256
from .guard import JOURNAL, StateRoot, box_name, own_prefix, parse_box_name, shadow_prefix
from .interfaces import (
    BatchRef, Blob, BoxError, BoxInfo, Digest, JobSpec, JudgeInvalid, NotFound, ObjectInfo,
    PreconditionFailed, ProviderError, RentUnavailable, SourceError, StoreError, TooLarge,
    VERDICT_GREEN, VERDICT_RED,
)
from .journal import FileOps, scan, verify_file
from .layout import (
    GENESIS_BATCH, META_FP32_MASTER_DIGEST, ROLE_AUX, ROLE_BF16_WEIGHTS, ROLE_FP32_MASTER,
    SCHEMA_HEAD, SCHEMA_STATE, Layout, StateManifest, complete_doc, heartbeat_body, is_hex64,
    is_safe_batch_id, outcome_body, parse_head, parse_manifest, sha256_hex,
)
from ...durable import canonical

PRICES = {'RTX5090': 12.0, 'H100': 20.0, 'B300': 40.0}
BOX_FAULTS = (
    'die', 'hang', 'pause', 'slow', 'corrupt_object', 'truncated_object', 'missing_object',
    'bad_result', 'garbage_result', 'early_complete', 'wrong_fence', 'wrong_base', 'no_state',
)
TAMPER_PENDING, TAMPER_CANONICAL, TAMPER_HEAD, TAMPER_MANIFEST = 'pending', 'canonical', 'head', 'manifest'
JUDGE_BUG_TRANSIENT, JUDGE_BUG_POISON = 'transient', 'poison'


class SimCrash(BaseException):
    """The controller process dies at this point."""


class AuditFailure(AssertionError):
    """An invariant was broken."""


class SimClock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass(frozen=True)
class Scenario:
    batches: int = 4
    bad_refs: int = 0
    max_boxes: int = 2
    max_infra_failures: int = 2
    green_pct: int = 60
    deterministic_training: bool = True
    dry_rungs: int = 0
    # Random faults: each fault site fires with ``fault_rate`` until
    # ``fault_budget`` faults were injected.
    fault_rate: float = 0.0
    fault_budget: int = 0
    # Random controller crashes at crash points.
    crash_rate: float = 0.0
    crash_budget: int = 0
    # Systematic crash: die at exactly this crash point (1-based).
    crash_at: Optional[int] = None
    # Then die once more, this many crash points into the recovery.
    crash_again: Optional[int] = None
    power_loss: Optional[bool] = None
    # A writer that is not the controller moves the pointer at this step.
    foreign_write_at: Optional[int] = None
    # The same, but in the worst moment: after the controller read the
    # pointer and before its own write arrives.
    foreign_race: bool = False
    # Someone damages one object in R2. ``pending``: a file of the state
    # that is about to become HEAD. ``canonical``: a file of a state that
    # was copied but has no decision yet. ``head``: a file of the current
    # HEAD. ``manifest``: the manifest of the current HEAD.
    tamper: Optional[str] = None
    breaker_failures: Optional[int] = None
    # An operator ends each halt of the infra circuit breaker.
    operator_resumes: bool = False
    max_cost_usd: Optional[float] = None
    apply_verify: str = APPLY_VERIFY_SHA256
    # Keep the journal on the real disk too (real fsync, real rename).
    real_disk: bool = False
    # A bug in the judge: ``transient`` or ``poison`` (see ``SimJudge``). An
    # operator fixes a poison bug after the controller has halted on it.
    judge_bug: Optional[str] = None
    # The batch source stops listing a batch once it is admitted.
    source_forgets: bool = False
    # Run the random scheduler first: effect results wait in a mailbox and
    # arrive in any order. Without it only the fair rounds run.
    chaos: bool = True
    chaos_steps: int = 500
    max_jump_s: float = 40.0
    max_rounds: int = 3000


# How often each fault site fires, relative to ``Scenario.fault_rate``. Sites
# that are passed very often get a low weight, so that the fault budget also
# reaches the rare sites (one roll per box, per rental, per launch).
FAULT_WEIGHTS = {
    'store.read_error': 0.25, 'store.bad_read': 0.25,
    'provider.list_error': 0.25, 'provider.list_misses_a_box': 0.25,
    'journal.short_write': 0.25, 'source.error': 0.25, 'box.put': 0.4,
    'box': 5.0, 'rent': 3.0, 'launch': 3.0, 'destroy': 3.0, 'chaos': 2.0,
}


class Faults:
    def __init__(self, rng: random.Random, rate: float, budget: int):
        self.rng = rng
        self.rate = rate
        self.budget = budget
        self.quiet = rate <= 0 or budget <= 0
        self.used: Dict[str, int] = collections.Counter()

    def _fires(self, site: str) -> bool:
        if self.quiet or self.budget <= 0:
            return False
        if self.rng.random() >= self.rate * FAULT_WEIGHTS.get(site, 1.0):
            return False
        self.budget -= 1
        return True

    def roll(self, site: str) -> bool:
        if not self._fires(site):
            return False
        self.used[site] += 1
        return True

    def pick(self, site: str, kinds: Sequence[str]) -> Optional[str]:
        if not self._fires(site):
            return None
        kind = self.rng.choice(list(kinds))
        self.used['%s:%s' % (site, kind)] += 1
        return kind


# ----------------------------------------------------------------------------
# File system under the journal
# ----------------------------------------------------------------------------

class MemFile:
    __slots__ = ('data', 'synced')

    def __init__(self, data: bytes = b'', synced: int = 0):
        self.data = bytearray(data)
        self.synced = synced


class MemOps(FileOps):
    """An in-memory file system with a crash model.

    ``synced`` is the length of a file that an fsync made durable. A rename
    is durable only after ``fsync_dir``.
    """

    def __init__(self, world: 'World'):
        self.w = world
        self.files: Dict[str, MemFile] = {}
        self.fds: Dict[int, str] = {}
        self.next_fd = 1000
        self.renames: List[Tuple[str, Optional[MemFile], str, MemFile]] = []

    def exists(self, path: Path) -> bool:
        return str(path) in self.files

    def read_bytes(self, path: Path) -> bytes:
        if str(path) not in self.files:
            raise FileNotFoundError(str(path))
        return bytes(self.files[str(path)].data)

    def open_append(self, path: Path) -> int:
        if str(path) not in self.files:
            raise FileNotFoundError(str(path))
        self.next_fd += 1
        self.fds[self.next_fd] = str(path)
        return self.next_fd

    def write(self, fd: int, data: bytes) -> int:
        data = bytes(data)
        self.w.point('journal.write.before')
        target = self.files[self.fds[fd]]
        count = len(data)
        if count > 1 and self.w.faults.roll('journal.short_write'):
            count = self.w.rng.randint(1, count - 1)
        if self.w.hit('journal.write.torn'):
            target.data += data[:self.w.rng.randint(0, count)]
            raise SimCrash('journal.write.torn')
        target.data += data[:count]
        self.w.point('journal.write.after')
        return count

    def fsync(self, fd: int) -> None:
        self.w.point('journal.fsync.before')
        target = self.files[self.fds[fd]]
        target.synced = len(target.data)
        self.w.point('journal.fsync.after')

    def close(self, fd: int) -> None:
        self.fds.pop(fd, None)

    def write_temp(self, path: Path, data: bytes) -> None:
        self.w.point('journal.temp.before')
        self.files[str(path)] = MemFile(bytes(data), len(data))
        self.w.point('journal.temp.after')

    def replace(self, src: Path, dst: Path) -> None:
        self.w.point('journal.rename.before')
        moved = self.files.pop(str(src))
        self.renames.append((str(dst), self.files.get(str(dst)), str(src), moved))
        self.files[str(dst)] = moved
        self.w.point('journal.rename.after')

    def fsync_dir(self, path: Path) -> None:
        self.w.point('journal.dirsync.before')
        self.renames.clear()
        self.w.point('journal.dirsync.after')

    def durable(self, path: str) -> bytes:
        """The bytes of a file that survive every kind of crash."""
        target = self.files.get(path)
        if target is None or any(item[0] == path for item in self.renames):
            return b''
        return bytes(target.data[:target.synced])

    def crash(self, rng: random.Random, power_loss: bool) -> None:
        self.fds.clear()
        if power_loss:
            for dst, old, src, moved in reversed(self.renames):
                if rng.random() < 0.5:
                    if old is None:
                        del self.files[dst]
                    else:
                        self.files[dst] = old
                    self.files[src] = moved
            for target in self.files.values():
                keep = rng.randint(target.synced, len(target.data))
                lost = len(target.data) - keep
                del target.data[keep:]
                if lost and rng.random() < 0.3:
                    # Blocks that were allocated but never written.
                    target.data += bytes(rng.randint(1, lost))
        self.renames.clear()


class DiskOps(MemOps):
    """The same crash model, with every call also done on the real disk.

    The real ``FileOps`` does the work: real appends, fsyncs and renames.
    Each read comes from the disk and must be equal to the model. After a
    crash the files on the disk are set to what the model says survived.
    """

    def __init__(self, world: 'World'):
        super().__init__(world)
        self.real = FileOps()
        self.real_fds: Dict[int, int] = {}

    def _same(self, got: Any, want: Any) -> Any:
        if got != want:
            self.w.audit.fail('the disk and the model of the disk differ')
        return got

    def exists(self, path: Path) -> bool:
        return self._same(self.real.exists(path), super().exists(path))

    def read_bytes(self, path: Path) -> bytes:
        return self._same(self.real.read_bytes(path), super().read_bytes(path))

    def open_append(self, path: Path) -> int:
        fd = super().open_append(path)
        self.real_fds[fd] = self.real.open_append(path)
        return fd

    def write(self, fd: int, data: bytes) -> int:
        target = self.files[self.fds[fd]]
        before = len(target.data)
        try:
            return super().write(fd, data)
        finally:
            # Also when the write is torn by a crash: the same bytes land.
            view = memoryview(bytes(target.data[before:]))
            while view:
                view = view[self.real.write(self.real_fds[fd], view):]

    def fsync(self, fd: int) -> None:
        self.w.point('journal.fsync.before')
        self.real.fsync(self.real_fds[fd])
        target = self.files[self.fds[fd]]
        target.synced = len(target.data)
        self.w.point('journal.fsync.after')

    def close(self, fd: int) -> None:
        super().close(fd)
        real = self.real_fds.pop(fd, None)
        if real is not None:
            self.real.close(real)

    def write_temp(self, path: Path, data: bytes) -> None:
        self.w.point('journal.temp.before')
        self.real.write_temp(path, data)
        self.files[str(path)] = MemFile(bytes(data), len(data))
        self.w.point('journal.temp.after')

    def replace(self, src: Path, dst: Path) -> None:
        self.w.point('journal.rename.before')
        self.real.replace(src, dst)
        moved = self.files.pop(str(src))
        self.renames.append((str(dst), self.files.get(str(dst)), str(src), moved))
        self.files[str(dst)] = moved
        self.w.point('journal.rename.after')

    def fsync_dir(self, path: Path) -> None:
        self.w.point('journal.dirsync.before')
        self.real.fsync_dir(path)
        self.renames.clear()
        self.w.point('journal.dirsync.after')

    def crash(self, rng: random.Random, power_loss: bool) -> None:
        for real in self.real_fds.values():
            self.real.close(real)
        self.real_fds.clear()
        known = set(self.files)
        super().crash(rng, power_loss)
        for path in sorted(known | set(self.files)):
            if path in self.files:
                with open(path, 'wb') as f:
                    f.write(bytes(self.files[path].data))
            elif os.path.exists(path):
                os.unlink(path)

    def check_disk(self) -> None:
        """The journal on the disk is the journal of the model, and it is whole."""
        path = Path(self.w.journal_path)
        self._same(self.real.read_bytes(path), bytes(self.files[str(path)].data))
        count, _last = verify_file(path)
        self._same(count > 0, True)
        on_disk = {str(path.parent / name) for name in os.listdir(path.parent)}
        self._same(set(self.files) <= on_disk, True)


# ----------------------------------------------------------------------------
# Object store
# ----------------------------------------------------------------------------

class RawStore:
    """The fake R2: atomic per request, strongly consistent."""

    def __init__(self) -> None:
        self.objs: Dict[str, bytes] = {}
        self.listeners: List[Callable[[str, str, Optional[bytes], Optional[bytes]], None]] = []

    @staticmethod
    def etag(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def _set(self, who: str, key: str, data: Optional[bytes]) -> None:
        before = self.objs.get(key)
        if data is None:
            self.objs.pop(key, None)
        else:
            self.objs[key] = data
        for listener in self.listeners:
            listener(who, key, before, data)

    def get(self, key: str, *, max_bytes: int) -> Blob:
        data = self.objs.get(key)
        if data is None:
            raise NotFound(key)
        if len(data) > max_bytes:
            raise TooLarge(key)
        return Blob(data, self.etag(data))

    def head(self, key: str) -> Optional[ObjectInfo]:
        data = self.objs.get(key)
        return None if data is None else ObjectInfo(key, len(data), self.etag(data))

    def digest(self, key: str) -> Digest:
        data = self.objs.get(key)
        if data is None:
            raise NotFound(key)
        return Digest(len(data), sha256_hex(data), self.etag(data))

    def put(self, key: str, data: bytes, *, if_none_match: bool = False,
            if_match: Optional[str] = None, who: str = 'box') -> str:
        current = self.objs.get(key)
        if if_none_match and current is not None:
            raise PreconditionFailed(key)
        if if_match is not None and (current is None or self.etag(current) != if_match):
            raise PreconditionFailed(key)
        self._set(who, key, bytes(data))
        return self.etag(data)

    def copy(self, src: str, dst: str, *, src_etag: str, who: str = 'box') -> None:
        current = self.objs.get(src)
        if current is None:
            raise NotFound(src)
        if self.etag(current) != src_etag:
            raise PreconditionFailed(src)
        self._set(who, dst, current)

    def delete(self, key: str, *, who: str = 'box') -> None:
        if key in self.objs:
            self._set(who, key, None)

    def list(self, prefix: str) -> List[ObjectInfo]:
        return [
            ObjectInfo(key, len(data), self.etag(data))
            for key, data in sorted(self.objs.items()) if key.startswith(prefix)
        ]


class ControllerStore:
    """The store as the controller reaches it: slow, lossy, and it can be
    the last thing the controller process ever does."""

    def __init__(self, world: 'World'):
        self.w = world
        self.raw = world.raw
        self.last_bad: Optional[str] = None

    def _read(self, name: str, fn: Callable[[], Any]) -> Any:
        self.w.point('store.%s.before' % name)
        if self.w.faults.roll('store.read_error'):
            raise StoreError('injected: read failed')
        out = fn()
        self.w.point('store.%s.after' % name)
        return out

    def _bad_read(self, key: str) -> Optional[str]:
        """One read of an object that is there goes wrong: the answer is
        ``missing`` or ``wrong``. The object itself is as it was, and the
        next read of the same key is right."""
        if self.last_bad == key:
            self.last_bad = None
            return None
        if not self.w.faults.roll('store.bad_read'):
            return None
        self.last_bad = key
        return self.w.rng.choice(('missing', 'wrong', 'wrong'))

    def _write(self, name: str, fn: Callable[[], Any]) -> Any:
        self.w.point('store.%s.before' % name)
        kind = self.w.faults.pick('store.write', ('lost', 'late', 'answer_lost', 'twice'))
        if kind == 'lost':
            raise StoreError('injected: request lost')
        if kind == 'late':
            self.w.later(fn)
            raise StoreError('injected: timeout, the request lands later')
        out = fn()
        self.w.point('store.%s.after' % name)
        if kind == 'answer_lost':
            raise StoreError('injected: answer lost')
        if kind == 'twice':
            self.w.later(fn)
        return out

    def get(self, key: str, *, max_bytes: int) -> Blob:
        out = self._read('get', lambda: self.raw.get(key, max_bytes=max_bytes))
        bad = self._bad_read(key)
        if bad == 'missing':
            raise NotFound(key)
        if bad == 'wrong':
            rng = self.w.rng
            data = bytearray(out.data)
            if data and rng.random() < 0.5:
                data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
            else:
                del data[rng.randrange(len(data) + 1):]
                data += b'\x00' if bytes(data) == out.data else b''
            return Blob(bytes(data), out.etag)
        return out

    def head(self, key: str) -> Optional[ObjectInfo]:
        return self._read('head', lambda: self.raw.head(key))

    def digest(self, key: str) -> Digest:
        out = self._read('digest', lambda: self.raw.digest(key))
        bad = self._bad_read(key)
        if bad == 'missing':
            raise NotFound(key)
        if bad == 'wrong':
            wrong = sha256_hex(b'bad read %d' % self.w.rng.getrandbits(32))
            return Digest(out.size, wrong, out.etag)
        return out

    def list(self, prefix: str) -> List[ObjectInfo]:
        return self._read('list', lambda: self.raw.list(prefix))

    def put(self, key: str, data: bytes, *, if_none_match: bool = False,
            if_match: Optional[str] = None) -> str:
        data = bytes(data)
        self.w.audit.controller_put(key, data)
        if key == self.w.layout.head and self.w.sc.foreign_race and not self.w.audit.foreign:
            if self.w.rng.random() < 0.5:
                self.w.foreign_pointer_write()
        return self._write('put', lambda: self.raw.put(
            key, data, if_none_match=if_none_match, if_match=if_match, who='controller',
        ))

    def copy(self, src: str, dst: str, *, src_etag: str) -> None:
        self.w.audit.controller_copy(src, dst)
        return self._write('copy', lambda: self.raw.copy(src, dst, src_etag=src_etag, who='controller'))

    def delete(self, key: str) -> None:
        self.w.audit.controller_delete(key)
        return self._write('delete', lambda: self.raw.delete(key, who='controller'))


# ----------------------------------------------------------------------------
# Provider, boxes and jobs
# ----------------------------------------------------------------------------

class SimBox:
    def __init__(self, box_id: str, name: str, provider: str, gpu: str, gpu_count: int,
                 tier: str, price: float, ours: bool):
        self.box_id = box_id
        self.name = name
        self.provider = provider
        self.gpu = gpu
        self.gpu_count = gpu_count
        self.tier = tier
        self.price = price
        self.ours = ours
        self.alive = True
        self.job: Optional['SimJob'] = None

    def info(self) -> BoxInfo:
        return BoxInfo(self.box_id, self.name, self.provider, self.gpu, self.gpu_count, self.tier, self.price)


class FakeProvider:
    def __init__(self, world: 'World', name: str):
        self.w = world
        self.name = name
        self.boxes: Dict[str, SimBox] = {}
        self.count = 0

    def make(self, name: str, gpu: str, tier: str, price: Optional[float] = None, gpu_count: int = 8) -> SimBox:
        self.count += 1
        box_id = '%s-box-%04d' % (self.name, self.count)
        ours = parse_box_name(self.w.run_id, name) is not None
        price = PRICES.get(gpu, 30.0) if price is None else price
        box = SimBox(box_id, name, self.name, gpu, gpu_count, tier, price, ours)
        self.boxes[box_id] = box
        return box

    def kill(self, box_id: str) -> None:
        box = self.boxes.get(box_id)
        if box is not None:
            box.alive = False

    def rent(self, *, name: str, gpu: str, tier: str, gpu_count: int, max_price_usd_h: float,
             ttl_s: float, timeout_s: float) -> BoxInfo:
        w = self.w
        w.audit.controller_rent(self.name, name)
        w.point('provider.rent.before')
        if (self.name, gpu, tier) in w.dry:
            raise RentUnavailable('no offer')
        kind = w.faults.pick('rent', (
            'unavailable', 'lost', 'answer_lost', 'late', 'twin', 'wrong_gpu', 'pricey', 'small',
        ))
        if kind == 'unavailable':
            raise RentUnavailable('injected: no offer')
        if kind == 'lost':
            raise ProviderError('injected: request lost')
        if kind == 'late':
            w.later(lambda: self.make(name, gpu, tier))
            raise ProviderError('injected: timeout, the box comes later')
        if kind == 'wrong_gpu':
            box = self.make(name, 'B200', tier)
        elif kind == 'pricey':
            box = self.make(name, gpu, tier, price=max_price_usd_h + 1.0)
        elif kind == 'small':
            box = self.make(name, gpu, tier, gpu_count=gpu_count - 1)
        else:
            box = self.make(name, gpu, tier)
        if kind == 'twin':
            self.make(name, gpu, tier)
        w.point('provider.rent.after')
        if kind == 'answer_lost':
            raise ProviderError('injected: answer lost')
        return box.info()

    def list(self) -> List[BoxInfo]:
        w = self.w
        w.point('provider.list.before')
        if w.faults.roll('provider.list_error'):
            raise ProviderError('injected: listing failed')
        alive = [box.info() for box in self.boxes.values() if box.alive]
        if alive and w.faults.roll('provider.list_misses_a_box'):
            alive.pop(w.rng.randrange(len(alive)))
        w.point('provider.list.after')
        return alive

    def destroy(self, box_id: str) -> None:
        w = self.w
        w.audit.controller_destroy(self, box_id)
        w.point('provider.destroy.before')
        kind = w.faults.pick('destroy', ('lost', 'late', 'answer_lost'))
        if kind == 'lost':
            raise ProviderError('injected: request lost')
        if kind == 'late':
            w.later(lambda: self.kill(box_id))
            raise ProviderError('injected: timeout, the box goes later')
        self.kill(box_id)
        w.point('provider.destroy.after')
        if kind == 'answer_lost':
            raise ProviderError('injected: answer lost')


class FakeAgent:
    def __init__(self, world: 'World'):
        self.w = world

    def launch(self, box_id: str, provider: str, spec: JobSpec) -> None:
        w = self.w
        w.audit.controller_launch(box_id, spec)
        w.point('agent.launch.before')
        box = w.providers[provider].boxes.get(box_id)
        if box is None or not box.alive:
            raise BoxError('box is not reachable')
        kind = w.faults.pick('launch', ('lost', 'answer_lost'))
        if kind == 'lost':
            raise BoxError('injected: request lost')
        if box.job is None:
            box.job = SimJob(w, box, spec)
        elif box.job.spec != spec:
            w.audit.fail('one box got two different jobs')
        w.point('agent.launch.after')
        if kind == 'answer_lost':
            raise BoxError('injected: answer lost')


@dataclass
class Package:
    """Ground truth of what a box uploaded for one fence."""

    batch_id: str
    fence: int
    base_state: str
    verdict: str
    result_sha: str
    result_ok: bool
    state_id: Optional[str]
    state_ok: bool
    unstable: bool


class SimJob:
    """The job of one box. Each ``step`` does one piece of work."""

    def __init__(self, world: 'World', box: SimBox, spec: JobSpec):
        self.w = world
        self.box = box
        self.spec = spec
        self.fault = world.faults.pick('box', BOX_FAULTS) or 'ok'
        self.cut = world.rng.randint(1, 9) if self.fault in ('die', 'hang', 'pause') else 0
        self.steps = 0
        self.done = False
        self.paused_until: Optional[float] = None
        self.unstable = False
        self.script = self._script()

    def step(self) -> None:
        if self.done or not self.box.alive:
            return
        if self.paused_until is not None:
            if self.w.clock.now() < self.paused_until:
                return
            self.paused_until = None
        self.steps += 1
        if self.steps == self.cut:
            if self.fault == 'die':
                self.box.alive = False
                self.done = True
                return
            if self.fault == 'hang':
                self.done = True
                return
            # A pause longer than the heartbeat grace: the lease expires and
            # the box then goes on as a zombie.
            self.paused_until = self.w.clock.now() + self.w.rng.uniform(70.0, 200.0)
            return
        try:
            next(self.script)
        except StopIteration:
            self.done = True

    def _put(self, key: str, data: bytes, *, payload: bool = True) -> None:
        w = self.w
        kind = w.faults.pick('box.put', ('late', 'twice'))
        if kind == 'late':
            # The box thinks the upload is done. It lands later, maybe after
            # the box is gone.
            w.later(lambda: w.raw.put(key, data, who='box'))
            self.unstable = self.unstable or payload
            return
        w.raw.put(key, data, who='box')
        if kind == 'twice':
            w.later(lambda: w.raw.put(key, data, who='box'))

    def _script(self):
        w, spec = self.w, self.spec
        lay = w.layout
        batch, fence, base = spec.batch_id, spec.fence, spec.base_state
        beats = [0]

        def beat(stage: str) -> None:
            beats[0] += 1
            self._put(
                lay.staging_heartbeat(batch, fence),
                heartbeat_body(fence=fence, seq=beats[0], stage=stage), payload=False,
            )

        # The box checks its base state itself and trains only on a verified
        # fp32 master (fail closed).
        problem = w.truth.state_problem(base)
        if problem is None:
            manifest = w.truth.manifest_of(base)
            if manifest.fp32_master_digest != spec.expected_fp32_master_digest:
                problem = 'the job names another fp32 master digest'
        if problem is not None:
            if not w.tampered:
                w.audit.fail('a box was sent to a base state that R2 does not hold: %s' % problem)
            return
        if not spec.staging_prefix == lay.staging_prefix(batch, fence):
            w.audit.fail('a job got a staging prefix that is not its own')
        beat('download')
        yield
        for _ in range(w.rng.randint(1, 3) + (12 if self.fault == 'slow' else 0)):
            beat('train')
            yield

        verdict = w.truth.verdict(batch, base)
        result = w.truth.result_bytes(batch, base, verdict, fence)
        files = w.truth.state_files(batch, base, fence) if verdict == VERDICT_GREEN else []
        manifest = w.truth.manifest(batch, base, files) if files else None
        claimed_fence, claimed_base, uploaded_result = fence, base, result
        result_ok, state_ok = True, manifest is not None
        if self.fault == 'no_state' and manifest is not None:
            manifest, files, state_ok = None, [], False
        elif self.fault == 'wrong_fence':
            claimed_fence, result_ok = fence + 1000, False
        elif self.fault == 'wrong_base':
            claimed_base, result_ok = 'f' * 64, False
        elif self.fault == 'bad_result':
            uploaded_result, result_ok = result + b' ', False
        elif self.fault == 'garbage_result':
            result = uploaded_result = b'{"this is": "not a result"}'
            result_ok = False
        upload = list(files)
        if upload and self.fault == 'corrupt_object':
            index = w.rng.randrange(len(upload))
            name, role, data = upload[index]
            upload[index] = (name, role, data[:-1] + bytes([data[-1] ^ 1]))
            state_ok = False
        elif upload and self.fault == 'truncated_object':
            # A partial write: the object is there, but it is cut short.
            index = w.rng.randrange(len(upload))
            name, role, data = upload[index]
            upload[index] = (name, role, data[:w.rng.randrange(len(data))])
            state_ok = False
        elif upload and self.fault == 'missing_object':
            upload.pop(w.rng.randrange(len(upload)))
            state_ok = False
        complete = canonical(complete_doc(
            batch_id=batch, fence=claimed_fence, base_state=claimed_base,
            result_sha=sha256_hex(result), result_size=len(result), manifest=manifest,
        ))

        def record() -> None:
            w.truth.packages[fence] = Package(
                batch, fence, base, verdict, sha256_hex(result), result_ok,
                manifest.state_id if manifest is not None else None, state_ok, self.unstable,
            )

        if self.fault == 'early_complete':
            # COMPLETE first, and then the box dies: nothing else arrives.
            result_ok = state_ok = False
            self._put(lay.staging_complete(batch, fence), complete)
            record()
            self.box.alive = False
            return
        for name, _role, data in upload:
            self._put(lay.staging_object(batch, fence, name), data)
            beat('upload')
            yield
        self._put(lay.staging_result(batch, fence), uploaded_result)
        yield
        self._put(lay.staging_complete(batch, fence), complete)
        record()
        yield


class FakeSource:
    def __init__(self, world: 'World'):
        self.w = world
        self.released: List[BatchRef] = []

    def list(self) -> List[BatchRef]:
        self.w.point('source.list')
        if self.w.faults.roll('source.error'):
            raise SourceError('injected: listing failed')
        refs = list(self.released)
        if self.w.sc.source_forgets:
            # The contract lets the source drop a batch once the controller
            # has admitted it (durably).
            state, _doomed = self.w.audit.durable()

            def admitted(ref: BatchRef) -> bool:
                batch = state.batch(ref.batch_id)
                return batch is not None and (batch.miner, batch.ref) == (ref.miner, ref.ref)

            refs = [ref for ref in refs if not admitted(ref)]
        return refs

    def expected(self) -> Tuple[Dict[str, Tuple[str, str]], Set[Tuple[str, str, str]]]:
        """What a correct controller admits, and what it must record as
        refused. The first reference that was released with an id owns it."""
        admitted: Dict[str, Tuple[str, str]] = {}
        refused: Set[Tuple[str, str, str]] = set()
        for ref in self.released:
            ok = is_safe_batch_id(ref.batch_id) and 0 < len(ref.miner) <= 128 and len(ref.ref) <= 1024
            if ok and admitted.setdefault(ref.batch_id, (ref.miner, ref.ref)) == (ref.miner, ref.ref):
                continue
            refused.add((ref.batch_id, ref.miner, ref.ref))
        return admitted, refused


class SimJudge:
    """Reads the verdict out of a result. It can have a bug.

    ``transient``: the first calls fail with an error that nobody expects.
    ``poison``: every call for one batch fails, until an operator fixes it.
    """

    POISON_BATCH = 'b001'
    TRANSIENT_CALLS = 3

    def __init__(self, bug: Optional[str], bugs: Dict[str, int]):
        self.bug = bug
        self.bugs = bugs

    @property
    def raised(self) -> int:
        return self.bugs['settle']

    def verdict(self, result: bytes, *, batch_id: str, base_state: str) -> str:
        if (
            (self.bug == JUDGE_BUG_TRANSIENT and self.raised < self.TRANSIENT_CALLS)
            or (self.bug == JUDGE_BUG_POISON and batch_id == self.POISON_BATCH)
        ):
            self.bugs['settle'] += 1
            raise RuntimeError('injected: a bug in the judge')
        try:
            doc = json.loads(result)
        except ValueError:
            raise JudgeInvalid('not JSON') from None
        if (
            not isinstance(doc, dict) or doc.get('batch_id') != batch_id
            or doc.get('base_state') != base_state
            or doc.get('verdict') not in (VERDICT_GREEN, VERDICT_RED)
        ):
            raise JudgeInvalid('not a result for this batch and base')
        return doc['verdict']


class Truth:
    """What is really true in the fake world."""

    def __init__(self, world: 'World'):
        self.w = world
        self.packages: Dict[int, Package] = {}

    def verdict(self, batch_id: str, base_state: str) -> str:
        text = '%d|%s|%s' % (self.w.seed, batch_id, base_state)
        roll = int(sha256_hex(text.encode())[:8], 16) % 100
        return VERDICT_GREEN if roll < self.w.sc.green_pct else VERDICT_RED

    def result_bytes(self, batch_id: str, base_state: str, verdict: str, fence: int) -> bytes:
        return canonical({
            'batch_id': batch_id, 'base_state': base_state, 'verdict': verdict, 'fence': fence,
        })

    def state_files(self, batch_id: str, base_state: Optional[str], fence: int) -> List[Tuple[str, str, bytes]]:
        tag = '%s|%s' % (batch_id, base_state)
        if not self.w.sc.deterministic_training:
            tag += '|%d' % fence
        return [
            ('master.fp32.bin', ROLE_FP32_MASTER, ('fp32 master of %s' % tag).encode() * 3),
            ('weights.bf16.bin', ROLE_BF16_WEIGHTS, ('bf16 weights of %s' % tag).encode() * 2),
            ('tokenizer.json', ROLE_AUX, ('{"aux": "%s"}' % tag).encode()),
        ]

    def manifest(self, batch_id: str, base_state: Optional[str],
                 files: List[Tuple[str, str, bytes]]) -> StateManifest:
        master = next(data for _name, role, data in files if role == ROLE_FP32_MASTER)
        doc = {
            'schema': SCHEMA_STATE,
            'parent': base_state,
            'batch_id': batch_id,
            'objects': [
                {'name': name, 'role': role, 'sha256': sha256_hex(data), 'size': len(data)}
                for name, role, data in files
            ],
            'meta': {META_FP32_MASTER_DIGEST: sha256_hex(b'digest of ' + master)},
        }
        return parse_manifest(doc, batch_id=batch_id, parent=base_state)

    def manifest_of(self, state_id: str) -> StateManifest:
        doc = json.loads(self.w.raw.objs[self.w.layout.state_manifest(state_id)])
        return parse_manifest(doc, batch_id=doc['batch_id'], parent=doc['parent'])

    def state_problem(self, state_id: str) -> Optional[str]:
        """None if R2 holds the full state ``state_id`` right now."""
        raw, lay = self.w.raw, self.w.layout
        body = raw.objs.get(lay.state_manifest(state_id))
        if body is None:
            return 'no manifest'
        if sha256_hex(body) != state_id:
            return 'manifest bytes do not hash to the state id'
        try:
            doc = json.loads(body)
            objects = doc['objects']
            roles = {item['role'] for item in objects}
        except (ValueError, KeyError, TypeError):
            return 'manifest is not readable'
        if ROLE_FP32_MASTER not in roles or ROLE_BF16_WEIGHTS not in roles:
            return 'not a full state'
        if not is_hex64(doc.get('meta', {}).get(META_FP32_MASTER_DIGEST)):
            return 'no fp32 master digest'
        for item in objects:
            data = raw.objs.get(lay.state_object(state_id, item['name']))
            if data is None:
                return 'file %s is missing' % item['name']
            if len(data) != item['size'] or sha256_hex(data) != item['sha256']:
                return 'file %s has other bytes' % item['name']
        return None


# ----------------------------------------------------------------------------
# The auditor
# ----------------------------------------------------------------------------

class Auditor:
    def __init__(self, world: 'World'):
        self.w = world
        self.first: Optional[str] = None
        self.pointers: List[Tuple[str, bytes]] = []
        self.applied: List[str] = []
        # batch id -> the base of the first job that was sent for the batch.
        self.first_base: Dict[str, Tuple[str, int]] = {}
        self.foreign = False
        self.durable_floor = b''
        self.kinds: Dict[str, int] = {}
        self._cache: Optional[Tuple[MemFile, int, C.State, Set[str]]] = None
        world.raw.listeners.append(self._on_change)

    def fail(self, message: str) -> None:
        self.first = self.first or message
        raise AuditFailure(self.first)

    def need(self, ok: Any, message: str) -> None:
        if not ok:
            self.fail(message)

    # -- what the journal has made durable ----------------------------------

    def durable(self) -> Tuple[C.State, Set[str]]:
        """The state of the fsynced journal, and every box it ever doomed."""
        files = self.w.files
        target = files.files.get(self.w.journal_path)
        if target is None or any(item[0] == self.w.journal_path for item in files.renames):
            return C.State(), set()
        cache = self._cache
        if cache is None or cache[0] is not target or cache[1] > target.synced:
            cache = (target, 0, C.State(), set())
        _file, done, state, doomed = cache
        if target.synced > done:
            for line in bytes(target.data[done:target.synced]).split(b'\n'):
                if line:
                    rec = json.loads(line)
                    C.evolve(state, rec['t'], rec['b'])
                    doomed.update(state.doomed)
        self._cache = (target, target.synced, state, doomed)
        return state, doomed

    # -- checks at the moment the controller acts ---------------------------

    def _rel(self, key: str) -> str:
        self.need(key.startswith(self.w.prefix), 'the controller touched a key outside its shadow prefix')
        return key[len(self.w.prefix):]

    def controller_put(self, key: str, data: bytes) -> None:
        rel = self._rel(key)
        if rel == 'HEAD.json':
            pointer = parse_head(data)
            pending = self.durable()[0].pending
            self.need(
                pending is not None and pending.link_id == pointer.link
                and pending.seq == pointer.seq and pending.state_id == pointer.state_id,
                'pointer write without a durable matching apply intent',
            )
            # The controller must have verified the full state and the batch
            # result in R2 before it sends this request.
            problem = self.w.truth.state_problem(pointer.state_id)
            self.need(problem is None, 'pointer write for a state that R2 does not hold in full: %s' % problem)
            if pending.result_sha is not None:
                result = self.w.raw.objs.get(self.w.layout.result(pending.result_sha))
                self.need(
                    result is not None and sha256_hex(result) == pending.result_sha,
                    'pointer write without the batch result in R2',
                )
            return
        if rel == 'RUN.json':
            return
        parts = rel.split('/')
        if parts[0] in ('links', 'results') and len(parts) == 2:
            self.need(parts[1] == sha256_hex(data) + '.json', 'a content-addressed key got other bytes')
        elif parts[0] == 'states' and len(parts) == 3 and parts[2] == 'manifest.json':
            self.need(parts[1] == sha256_hex(data), 'a state manifest key got other bytes')
            # The manifest is the last object of a state. Every file that it
            # names must be in R2, with the right bytes, before it is sent.
            for item in json.loads(data)['objects']:
                have = self.w.raw.objs.get(self.w.layout.state_object(parts[1], item['name']))
                self.need(
                    have is not None and len(have) == item['size'] and sha256_hex(have) == item['sha256'],
                    'state manifest written before file %s was verified in R2' % item['name'],
                )
        elif parts[0] == 'outcomes' and len(parts) == 2:
            batch = self.durable()[0].done.get(parts[1][:-len('.json')])
            self.need(batch is not None, 'outcome written for a batch that is not durably terminal')
            self.need(data == outcome_body(C.outcome_fields(batch)), 'outcome differs from the journal')
        else:
            self.fail('the controller wrote an unexpected key: %s' % rel)

    def controller_copy(self, src: str, dst: str) -> None:
        src_rel, dst_rel = self._rel(src), self._rel(dst)
        parsed = self.w.layout.parse_staging_key(src)
        self.need(src_rel.startswith('staging/') and parsed is not None, 'copy from a key outside staging')
        self.need(dst_rel.startswith('states/') and '/obj/' in dst_rel, 'copy to a key that is not a state file')
        batch = self.durable()[0].leased(parsed[1])
        self.need(
            batch is not None and batch.phase is C.Phase.SETTLING and batch.batch_id == parsed[0],
            'copy from the staging of a fence that is not settling',
        )
        # The bytes must be the ones that the manifest of this fence names.
        name = dst_rel.split('/')[-1]
        true = {
            item[0]: item[2]
            for item in self.w.truth.state_files(batch.batch_id, batch.base_state, parsed[1])
        }
        self.need(
            self.w.raw.objs.get(src) == true.get(name),
            'the controller copied bytes that the manifest does not name',
        )

    def controller_delete(self, key: str) -> None:
        rel = self._rel(key)
        parsed = self.w.layout.parse_staging_key(key)
        self.need(rel.startswith('staging/') and parsed is not None, 'the controller deleted a key outside staging')
        state = self.durable()[0]
        self.need(parsed[1] not in state.by_fence, 'the controller deleted staging of a live fence')
        self.need(parsed[1] < state.next_fence, 'the controller deleted staging of a fence it never issued')

    def controller_rent(self, provider: str, name: str) -> None:
        parsed = parse_box_name(self.w.run_id, name)
        self.need(parsed is not None, 'rental with a name that is not ours')
        batch = self.durable()[0].leased(parsed[0])
        self.need(
            batch is not None and batch.phase is C.Phase.RENTING and batch.lease.try_no == parsed[1],
            'rental without a durable open rent try',
        )

    def controller_destroy(self, provider: FakeProvider, box_id: str) -> None:
        box = provider.boxes.get(box_id)
        self.need(box is None or box.ours, 'the controller destroyed a box that is not its own')
        self.need(box_id in self.durable()[1], 'destroy without a durable record that dooms the box')

    def controller_launch(self, box_id: str, spec: JobSpec) -> None:
        batch = self.durable()[0].leased(spec.fence)
        base = (spec.base_state, spec.base_seq)
        self.need(
            batch is not None and batch.phase is C.Phase.RUNNING and batch.lease.box_id == box_id
            and (batch.base_state, batch.base_seq) == base and batch.batch_id == spec.batch_id
            and batch.attempts == spec.attempt,
            'launch without a durable binding of this box to this fence',
        )
        # The base rule, as the boxes see it: every box of a batch gets the
        # base of its first box, also when the HEAD moved in the meantime.
        self.need(
            self.first_base.setdefault(spec.batch_id, base) == base,
            'a fresh box was sent to another base than the first box of its batch',
        )

    # -- checks at the moment R2 changes ------------------------------------

    def _on_change(self, who: str, key: str, before: Optional[bytes], after: Optional[bytes]) -> None:
        lay = self.w.layout
        if key == lay.head:
            self.pointers.append((who, after))
            if who == 'foreign':
                self.foreign = True
            else:
                self._pointer_moved(before, after)
            return
        rel = key[len(self.w.prefix):] if key.startswith(self.w.prefix) else key
        if who == 'box':
            self.need(rel.startswith('staging/'), 'a box wrote outside staging')
        elif who == 'controller' and not rel.startswith('staging/'):
            # Everything but the pointer and staging is immutable. One case
            # is not in the hands of the controller: a copy request that
            # lands late, on a key that someone else damaged in between.
            self.need(after is not None, 'the controller deleted an immutable object')
            self.need(before is None or before == after or self.w.tampered, 'an immutable object changed')

    def _pointer_moved(self, before: Optional[bytes], after: Optional[bytes]) -> None:
        w, lay = self.w, self.w.layout
        self.need(after is not None, 'the pointer was deleted')
        self.need(not self.foreign, 'the controller moved the pointer after a foreign write')
        new = parse_head(after)
        old = parse_head(before) if before is not None else None
        self.need(new.seq == (old.seq + 1 if old is not None else 0), 'HEAD chain is not linear')
        body = w.raw.objs.get(lay.link(new.link))
        self.need(body is not None and sha256_hex(body) == new.link, 'the pointer names a link that R2 does not hold')
        link = json.loads(body)
        self.need(
            link['seq'] == new.seq and link['state'] == new.state_id
            and link['parent'] == (old.link if old is not None else None),
            'the new link does not extend the old HEAD',
        )
        # A request can land long after it was sent. If someone damaged the
        # state in between, no controller can see that: the check that the
        # state was whole is made when the request is sent.
        problem = w.truth.state_problem(new.state_id)
        self.need(problem is None or w.tampered, 'applied a state that R2 does not hold in full: %s' % problem)
        manifest = w.truth.manifest_of(new.state_id)
        self.need(
            manifest.parent == (old.state_id if old is not None else None),
            'the applied state was not trained on the HEAD it replaced',
        )
        if link['batch_id'] is None:
            self.need(old is None and manifest.batch_id == GENESIS_BATCH, 'genesis link after genesis')
            return
        self.need(manifest.batch_id == link['batch_id'], 'the applied state belongs to another batch')
        self.need(link['batch_id'] not in self.applied, 'a batch was applied twice')
        self.applied.append(link['batch_id'])
        result = w.raw.objs.get(lay.result(link['result']))
        self.need(
            result is not None and sha256_hex(result) == link['result'],
            'applied without the batch result in R2',
        )

    # -- checks at the end ---------------------------------------------------

    def final(self, harness: 'Harness') -> None:
        w, lay, raw = self.w, self.w.layout, self.w.raw
        ctl = harness.ctl
        data = bytes(w.files.files[w.journal_path].data)
        records, valid = scan(data)
        self.need(valid == len(data) and records, 'the journal has a torn tail after a clean stop')
        if isinstance(w.files, DiskOps):
            w.files.check_disk()
        self.need(records[0]['t'] == 'init', 'the journal does not start with init')
        cfg = harness.config.core

        state = C.State()
        failures: Dict[str, List[int]] = collections.defaultdict(list)
        bound: Dict[int, str] = {}
        bases: Dict[int, Tuple[str, int]] = {}
        observed: Set[int] = set()
        decided: Dict[int, Dict[str, Any]] = {}
        decided_batches: Set[str] = set()
        leases: Dict[str, int] = collections.Counter()
        first_bound: Dict[str, Tuple[str, int]] = {}
        invalid: Set[int] = set()
        greens: Dict[int, List[Tuple[int, str]]] = collections.defaultdict(list)
        intents: Dict[int, str] = {}
        winners: Dict[int, str] = {}
        seqs: Dict[str, int] = {}
        journal_applied: List[str] = []
        kinds: Dict[str, int] = collections.Counter()
        self.kinds = kinds
        for rec in records:
            kind, body = rec['t'], rec['b']
            if kind == 'lease_issued':
                queued = [b for b in state.active.values() if b.phase is C.Phase.QUEUED]
                first = min(queued, key=lambda b: b.seq)
                self.need(first.batch_id == body['batch_id'], 'a lease went to a batch that is not first in the queue')
                # A batch runs again for one reason only: its last box was
                # lost before a decision. The count is made here, from the
                # records, and not taken from the state.
                leases[body['batch_id']] += 1
                self.need(
                    body['batch_id'] not in decided_batches
                    and leases[body['batch_id']] == len(failures[body['batch_id']]) + 1 == body['attempt'],
                    'a batch got a new box, but its last box was not lost before a decision',
                )
            elif kind == 'box_bound':
                base = (body['base_state'], body['base_seq'])
                if body['batch_id'] not in first_bound:
                    self.need(base == state.base(), 'the first box of a batch did not get the HEAD of that moment')
                    first_bound[body['batch_id']] = base
                self.need(
                    base == first_bound[body['batch_id']],
                    'a fresh box got another base than the first box of its batch',
                )
            elif kind == 'credited':
                # A GREEN is paid without an apply for one of three reasons.
                decision = state.active[body['batch_id']].decision
                self.need(decision.verdict == C.VERDICT_GREEN, 'credit without a GREEN verdict')
                self.need(
                    (body['reason'] == 'no_verified_state') == (decision.state_id is None),
                    'credit reason does not match the decision',
                )
                if body['reason'] == 'stale_base':
                    self.need(decision.base_seq < state.head.seq, 'credited as stale, but its base is the HEAD')
                elif body['reason'] == 'not_first':
                    first = min(greens[decision.base_seq])
                    self.need(
                        first[0] < decision.order and intents.get(decision.base_seq) == first[1],
                        'credited as not first, but no earlier GREEN on that HEAD has the apply',
                    )
            elif kind == 'apply_intent' and body['batch_id'] is not None:
                intents[body['seq'] - 1] = body['batch_id']
                on_head = greens.get(body['seq'] - 1)
                self.need(
                    on_head and min(on_head)[1] == body['batch_id'],
                    'the apply did not go to the first GREEN on its HEAD',
                )
            elif kind == 'halted':
                # A halt needs a cause that the world really gave.
                if body['reason'].startswith('infra_breaker'):
                    self.need(harness.sc.breaker_failures is not None, 'the breaker tripped, but it is off')
                elif body['reason'].startswith('effect_error:'):
                    effect = body['reason'].split(':', 1)[1]
                    self.need(
                        w.bugs[effect] >= harness.config.effect_error_limit,
                        'the controller halted on a bug that the world did not have',
                    )
                else:
                    self.need(
                        self.foreign or w.tampered,
                        'the controller halted without a cause: %s' % body['reason'],
                    )
            kinds[kind] += 1
            C.evolve(state, kind, body)
            self.need(state.slots_in_use() <= cfg.max_boxes, 'more boxes in use than max_boxes')
            if kind == 'batch_admitted':
                seqs[body['batch_id']] = body['seq']
            elif kind == 'box_bound':
                self.need(body['box_id'] not in bound.values(), 'one box was bound to two fences')
                bound[body['fence']] = body['box_id']
                bases[body['fence']] = (body['base_state'], body['base_seq'])
                box = w.providers[body['provider']].boxes.get(body['box_id'])
                rung = cfg.ladder[state.leased(body['fence']).lease.rung]
                self.need(
                    box is not None and box.name == box_name(w.run_id, body['fence'], body['try'])
                    and (box.provider, box.gpu, box.tier) == (rung.provider, rung.gpu, rung.tier)
                    and box.gpu_count == harness.config.gpu_count
                    and 0 <= box.price <= harness.config.price_ceiling_usd_h,
                    'a box that breaks the rental rules was bound',
                )
            elif kind == 'upload_observed':
                observed.add(body['fence'])
            elif kind == 'attempt_failed':
                failures[body['batch_id']].append(body['fence'])
                self.need(body['fence'] in bound, 'an attempt without a box counted as an infra failure')
                self.need(
                    body['reason'] in ('box_gone', 'lease_expired') or body['reason'].startswith('invalid_upload:'),
                    'an attempt ended for a reason that is not an infra loss: %s' % body['reason'],
                )
                if body['reason'].startswith('invalid_upload'):
                    invalid.add(body['fence'])
            elif kind == 'decided':
                self.need(body['batch_id'] not in decided_batches, 'a batch was decided twice')
                decided_batches.add(body['batch_id'])
                decided[body['fence']] = body
                if body['verdict'] == C.VERDICT_GREEN and body['state_id'] is not None:
                    greens[body['base_seq']].append((body['order'], body['batch_id']))
            elif kind == 'applied' and body['batch_id'] is not None:
                self.need(
                    first_bound[body['batch_id']][1] == body['seq'] - 1,
                    'a batch was applied on a HEAD that is not its base',
                )
                winners[body['seq'] - 1] = body['batch_id']
                journal_applied.append(body['batch_id'])
            for batch in list(state.active.values()) + list(state.done.values()):
                self.need(batch.seq == seqs[batch.batch_id], 'a batch lost its queue position')
        self.need(state.key() == ctl.state.key(), 'the state in memory is not the journal')
        for batch_id, base in self.first_base.items():
            self.need(first_bound.get(batch_id) == base, 'a box was sent to a base that the journal does not name')

        # Ground truth of every decision.
        for fence, body in decided.items():
            package = w.truth.packages.get(fence)
            self.need(package is not None, 'a decision for a fence whose box never finished an upload')
            self.need(
                package.batch_id == body['batch_id'] and package.base_state == body['base_state']
                and bases[fence] == (body['base_state'], body['base_seq']),
                'a decision used the upload of another fence or base',
            )
            self.need(package.result_ok, 'a decision was made from an invalid package')
            self.need(body['verdict'] == package.verdict, 'the verdict differs from the truth')
            self.need(
                bases[fence] == first_bound[body['batch_id']]
                and body['verdict'] == w.truth.verdict(body['batch_id'], first_bound[body['batch_id']][0]),
                'the verdict is not the verdict against the first base of the batch',
            )
            self.need(body['result_sha'] == package.result_sha, 'the decision names another result')
            self.need(
                raw.objs.get(lay.result(body['result_sha'])) is not None,
                'a decided result is not durable in R2',
            )
            if body['verdict'] == C.VERDICT_GREEN:
                if body['state_id'] is not None:
                    self.need(body['state_id'] == package.state_id, 'the decision names a state the box did not make')
                    if not w.tampered:
                        problem = w.truth.state_problem(body['state_id'])
                        self.need(problem is None, 'a verified state is not in R2: %s' % problem)
                else:
                    self.need(
                        not package.state_ok or package.unstable or w.tampered,
                        'a valid uploaded state was not recognized',
                    )
        for fence in observed:
            package = w.truth.packages.get(fence)
            self.need(package is not None, 'an upload was observed that no box completed')
            if package.result_ok and not package.unstable:
                self.need(fence in decided, 'a package with a valid result was rejected')
        for fence in invalid:
            package = w.truth.packages[fence]
            self.need(not package.result_ok or package.unstable, 'a valid upload was called invalid')

        # Outcomes.
        every = dict(state.active)
        every.update(state.done)
        for batch_id, batch in every.items():
            fields = None
            if batch.phase in C.TERMINAL:
                fields = C.outcome_fields(batch)
            green = batch.decision is not None and batch.decision.verdict == C.VERDICT_GREEN
            if green and batch.phase in C.TERMINAL:
                self.need(
                    batch.phase in (C.Phase.APPLIED, C.Phase.CREDITED)
                    and fields['payout_list'] and fields['credit'] == C.CREDIT_GREEN,
                    'a GREEN was lost',
                )
            if batch.phase is C.Phase.RED:
                self.need(batch.decision.verdict == C.VERDICT_RED, 'RED without a RED verdict')
                self.need(not fields['payout_list'], 'RED on the payout list')
            if batch.phase is C.Phase.LOST_INFRA:
                self.need(
                    len(failures[batch_id]) == cfg.max_infra_failures and fields['credit'] == C.CREDIT_LOST_INFRA,
                    'lost_infra without exactly N lost attempts, or without credit',
                )
            else:
                self.need(len(failures[batch_id]) < cfg.max_infra_failures, 'a batch went on after N lost attempts')
            if batch.phase is C.Phase.CREDITED:
                self.need(green, 'credited without a GREEN verdict')
            if batch.phase is C.Phase.APPLIED:
                self.need(batch_id in self.applied, 'journal says applied, R2 never moved')
            if fields is not None:
                base = first_bound.get(batch_id, (None, None))
                self.need(
                    (fields['base_state'], fields['base_seq']) == base,
                    'the outcome names another base than the first box of the batch got',
                )
                # Paid: GREEN against its own base, or lost by the infra.
                paid = batch.phase is C.Phase.LOST_INFRA or (
                    batch.decision is not None and w.truth.verdict(batch_id, base[0]) == C.VERDICT_GREEN
                )
                self.need(
                    fields['payout_list'] == paid and (batch.phase is C.Phase.APPLIED) == fields['applied'],
                    'the payout does not follow the verdict against the base of the batch',
                )
            if fields is not None and batch_id not in state.unpublished:
                self.need(
                    raw.objs.get(lay.outcome(batch_id)) == outcome_body(fields),
                    'the published outcome differs from the journal',
                )
        # First GREEN wins, and it is the only one that is applied.
        for base_seq, winner in winners.items():
            first = min(greens[base_seq])
            self.need(
                first[1] == winner and intents.get(base_seq) == winner,
                'the applied batch was not the first GREEN on its HEAD',
            )
        self.need(len(set(winners.values())) == len(winners), 'a batch was applied on two HEADs')

        # Batch conservation.
        admitted, refused = w.source.expected()
        self.need(set(seqs) == set(admitted), 'a batch was dropped or invented')
        for batch_id, owner in admitted.items():
            batch = state.batch(batch_id)
            self.need((batch.miner, batch.ref) == owner, 'a batch runs for the miner or the reference of another')
        self.need(len(state.rejected) == len(refused), 'a refused batch reference was not recorded')
        halted = state.halted is not None
        if not halted:
            self.need(not state.active and len(state.done) == len(admitted), 'a batch is not terminal')

        # The chain in R2 and in the journal.
        ours = [body for who, body in self.pointers if who != 'foreign']
        for index, body in enumerate(ours):
            self.need(parse_head(body).seq == index, 'the pointer history has a gap or a fork')
        if not self.foreign and state.pending is None:
            self.need(state.head is not None, 'no HEAD at the end')
            now = parse_head(raw.objs[lay.head])
            self.need(
                (now.seq, now.state_id, now.link) == tuple(state.head),
                'the journal HEAD is not the R2 pointer',
            )
            self.need(len(ours) == state.head.seq + 1, 'the pointer moved more often than the journal says')
        # Every apply in the journal is a pointer move that R2 saw, in the
        # same order. R2 can be one move ahead: the move of the open intent.
        known = len(journal_applied)
        self.need(self.applied[:known] == journal_applied, 'the journal names an apply that R2 never saw')
        ahead = self.applied[known:]
        self.need(
            not ahead or (state.pending is not None and ahead == [state.pending.batch_id]),
            'R2 saw an apply that the journal does not know',
        )

        # Boxes: ours are all gone, others were never touched.
        for provider in w.providers.values():
            for box in provider.boxes.values():
                if box.ours:
                    self.need(not box.alive, 'a box of this run is still alive')
                else:
                    self.need(box.alive, 'a box of someone else was destroyed')
        self.need(not raw.list(lay.staging_root), 'staging is not empty at the end')
        if not harness.sc.chaos and not w.faults.used and not w.tampered:
            # Every part of this world worked and got its turn. So no box was
            # lost, whatever the HEAD did in the meantime.
            self.need(
                not kinds['attempt_failed'],
                'an attempt ended without a decision in a world that had no fault',
            )
        status = ctl.status()
        self.need(status['cost']['boxes_alive'] == 0 and status['boxes']['slots_in_use'] == 0, 'status counts boxes at the end')
        self.need(status['cost']['usd_spent_estimate'] >= 0, 'status has no cost')


# ----------------------------------------------------------------------------
# World and harness
# ----------------------------------------------------------------------------

class World:
    def __init__(self, seed: int, sc: Scenario, run_id: str, journal_path: str):
        self.seed = seed
        self.sc = sc
        self.run_id = run_id
        self.rng = random.Random(seed)
        self.prefix = shadow_prefix(run_id)
        self.layout = Layout(self.prefix)
        self.journal_path = journal_path
        self.clock = SimClock()
        self.faults = Faults(self.rng, sc.fault_rate, sc.fault_budget)
        self.raw = RawStore()
        self.files: MemOps = DiskOps(self) if sc.real_disk else MemOps(self)
        self.late: List[Callable[[], Any]] = []
        self.points = 0
        self.crash_at = sc.crash_at
        self.crash_again = sc.crash_again
        self.crash_budget = sc.crash_budget
        self.crash_labels: List[str] = []
        self.crashing = False
        self.tampered = False
        self.dry: Set[Tuple[str, str, str]] = set()
        self.truth = Truth(self)
        self.audit = Auditor(self)
        self.providers = {name: FakeProvider(self, name) for name in ('p0', 'p1')}
        self.agent = FakeAgent(self)
        self.source = FakeSource(self)
        # Unexpected errors that the world raised, by the kind of effect.
        self.bugs: Dict[str, int] = collections.Counter()
        self.judge = SimJudge(sc.judge_bug, self.bugs)

    def hit(self, label: str) -> bool:
        """Count one crash point. True when the controller dies here."""
        self.points += 1
        dies = False
        if self.crash_at is not None and self.points == self.crash_at:
            dies = True
            if self.crash_again is not None:
                self.crash_at, self.crash_again = self.points + self.crash_again, None
        elif self.crashing and self.crash_budget > 0 and self.rng.random() < self.sc.crash_rate:
            self.crash_budget -= 1
            dies = True
        if dies:
            self.crash_labels.append(label)
        return dies

    def point(self, label: str) -> None:
        if self.hit(label):
            raise SimCrash(label)

    def later(self, fn: Callable[[], Any]) -> None:
        self.late.append(fn)

    def foreign_pointer_write(self) -> None:
        """A writer that is not the controller moves (or makes) the pointer."""
        current = self.raw.objs.get(self.layout.head)
        seq = parse_head(current).seq + 1 if current is not None else 0
        body = canonical({'schema': SCHEMA_HEAD, 'seq': seq, 'link': 'e' * 64, 'state': 'd' * 64})
        self.raw.put(self.layout.head, body, who='foreign')

    def land(self, index: int) -> None:
        fn = self.late.pop(index)
        try:
            fn()
        except StoreError:
            pass


@dataclass
class Report:
    seed: int
    points: int = 0
    crashes: int = 0
    rounds: int = 0
    records: int = 0
    head_seq: int = -1
    halted: Optional[str] = None
    outcomes: Dict[str, int] = field(default_factory=dict)
    faults: Dict[str, int] = field(default_factory=dict)
    # How many journal records of each kind the run wrote.
    kinds: Dict[str, int] = field(default_factory=dict)
    # The crash points at which the controller died.
    crash_labels: List[str] = field(default_factory=list)


class Harness:
    def __init__(self, base: Path, seed: int, sc: Scenario, tag: str = ''):
        self.seed = seed
        self.sc = sc
        self.run_id = 'sim%06d' % (seed % 1000000)
        self.root_path = Path(base) / ('root-%d%s' % (seed, tag))
        root = StateRoot.create(self.root_path, self.run_id)
        self.world = World(seed, sc, self.run_id, str(root.file(JOURNAL)))
        w = self.world
        ladder = (
            C.Rung('p0', 'RTX5090', 'secure'), C.Rung('p0', 'RTX5090', 'spot'),
            C.Rung('p1', 'H100', 'secure'), C.Rung('p1', 'B300', 'spot'),
        )
        for rung in ladder[:sc.dry_rungs]:
            w.dry.add((rung.provider, rung.gpu, rung.tier))
        files = w.truth.state_files(GENESIS_BATCH, None, 0)
        genesis = w.truth.manifest(GENESIS_BATCH, None, files)
        # The operator puts the first HEAD state in R2 before the first start.
        for name, _role, data in files:
            w.raw.put(w.layout.state_object(genesis.state_id, name), data, who='operator')
        w.raw.put(w.layout.state_manifest(genesis.state_id), genesis.body(), who='operator')
        self.config = Config(
            run_id=self.run_id,
            genesis_state=genesis.state_id,
            core=C.CoreConfig(sc.max_boxes, sc.max_infra_failures, ladder),
            allow_sharded_gpus=True,
            rent_timeout_s=30.0, rent_backoff_s=10.0, box_ttl_s=1200.0,
            lease_ttl_s=600.0, heartbeat_grace_s=60.0, heartbeat_every_s=5.0, renew_every_s=20.0,
            absent_polls=2, retry_s=0.0, poll_every_s=0.0, ingest_every_s=0.0, sweep_every_s=0.0,
            max_cost_usd=sc.max_cost_usd, breaker_failures=sc.breaker_failures,
            apply_verify=sc.apply_verify,
        )
        # Boxes of other owners. The controller must never touch them.
        w.providers['p0'].make('affine-pe-pit-9-a1', 'B300', 'secure')
        w.providers['p0'].make(own_prefix('other0001') + 'f1-t0', 'RTX5090', 'secure')
        w.providers['p1'].make(own_prefix(self.run_id) + 'f1-t0-x', 'H100', 'secure')
        w.providers['p1'].make(own_prefix(self.run_id)[:-1] + 'x-f2-t0', 'H100', 'spot')
        self.todo: List[BatchRef] = [
            BatchRef('b%03d' % i, 'miner%d' % (i % 3), 'ref-%d' % i) for i in range(sc.batches)
        ]
        bad = [
            BatchRef('bad id!', 'm'),
            # The id of the first batch again, for another miner.
            BatchRef('b000', 'miner-other', 'ref-of-another-batch'),
            BatchRef(GENESIS_BATCH, 'm'), BatchRef('../up', 'm'), BatchRef('ok', ''),
        ]
        for i in range(sc.bad_refs):
            self.todo.insert(min(len(self.todo), 1 + 2 * i), bad[i % len(bad)])
        self.mailbox: Dict[Tuple, List[Any]] = {}
        self.ctl: Optional[Controller] = None
        self.report = Report(seed)
        self.step_no = 0
        # A test can put a layer around each fake that the controller gets.
        self.wrap: Callable[[str, Any], Any] = lambda role, part: part

    # -- controller life cycle ----------------------------------------------

    def _boot(self) -> None:
        w = self.world
        for _ in range(200):
            wrap = self.wrap
            ctl = Controller(
                root=StateRoot.open(self.root_path, self.run_id),
                config=self.config,
                store=wrap('store', ControllerStore(w)),
                providers={name: wrap('provider', part) for name, part in w.providers.items()},
                agent=wrap('agent', w.agent),
                source=wrap('source', w.source),
                judge=wrap('judge', w.judge),
                clock=wrap('clock', w.clock),
                file_ops=wrap('files', w.files),
            )
            try:
                self.ctl = ctl.open()
                break
            except SimCrash:
                self._power(ctl)
            except StoreError:
                # The process exits with an error; its supervisor starts it again.
                continue
        else:
            raise AuditFailure('the controller cannot start')
        data = bytes(w.files.files[w.journal_path].data)
        if not data.startswith(w.audit.durable_floor):
            w.audit.fail('a restart lost journal records that were fsynced')

    def _power(self, ctl: Optional[Controller]) -> None:
        """The controller process is dead. Apply the crash to its files."""
        w = self.world
        self.report.crashes += 1
        floor = w.files.durable(w.journal_path)
        if len(floor) >= len(w.audit.durable_floor):
            w.audit.durable_floor = floor
        power_loss = self.sc.power_loss
        if power_loss is None:
            power_loss = w.rng.random() < 0.5
        w.files.crash(w.rng, power_loss)
        if ctl is not None:
            # The operating system releases the lock of a dead process.
            ctl.close()
        self.mailbox.clear()

    def _restart(self) -> None:
        ctl, self.ctl = self.ctl, None
        self._power(ctl)
        self._boot()

    # -- actions -------------------------------------------------------------

    def _arrive(self) -> None:
        if self.todo:
            self.world.source.released.append(self.todo.pop(0))

    def _boxes(self, only_ours: bool = True) -> List[SimBox]:
        return [
            box for name in sorted(self.world.providers)
            for box in self.world.providers[name].boxes.values()
            if box.alive and (box.ours or not only_ours)
        ]

    def _deliver(self, key: Tuple, twice: bool = False) -> None:
        events = self.mailbox.pop(key)
        for _ in range(2 if twice else 1):
            for event in events:
                self.ctl.commit(event)

    def _controller_action(self) -> None:
        w, ctl = self.world, self.ctl
        if self.mailbox and w.rng.random() < 0.5:
            key = w.rng.choice(sorted(self.mailbox))
            self._deliver(key, twice=w.faults.roll('result.delivered_twice'))
            ctl.tick()
            return
        ctl.tick()
        ops = [op for op in ctl.due_ops() if op_key(op) not in self.mailbox]
        if not ops:
            return
        op = w.rng.choice(ops)
        events = ctl.attempt(op)
        self._sticky()
        if events and not w.faults.roll('result.lost'):
            self.mailbox[op_key(op)] = events

    def _sticky(self) -> None:
        """An audit failure inside an effect must end the run at once."""
        if self.world.audit.first is not None:
            raise AuditFailure(self.world.audit.first)

    def _tamper_target(self) -> Optional[str]:
        w, lay, state = self.world, self.world.layout, self.ctl.state
        if self.sc.tamper == TAMPER_PENDING:
            pending = state.pending
            if pending is None or pending.batch_id is None:
                return None
            pointer = w.raw.objs.get(lay.head)
            if pointer is None or parse_head(pointer).link == pending.link_id:
                return None
            return lay.state_object(pending.state_id, 'master.fp32.bin')
        if self.sc.tamper in (TAMPER_HEAD, TAMPER_MANIFEST):
            if state.head is None or self.step_no < self.sc.chaos_steps // 4:
                return None
            if self.sc.tamper == TAMPER_MANIFEST:
                return lay.state_manifest(state.head.state_id)
            return lay.state_object(state.head.state_id, 'weights.bf16.bin')
        every = list(state.active.values()) + list(state.done.values())
        decided = {b.decision.state_id for b in every if b.decision is not None}
        decided.add(self.config.genesis_state)
        for key in sorted(w.raw.objs):
            parts = key[len(w.prefix):].split('/')
            if len(parts) == 4 and parts[0] == 'states' and parts[2] == 'obj' and parts[1] not in decided:
                return key
        return None

    def _tamper(self) -> None:
        """Someone who is not the controller damages one state file in R2."""
        w = self.world
        if w.tampered or self.ctl is None:
            return
        key = self._tamper_target()
        if key is not None and key in w.raw.objs:
            w.tampered = True
            w.raw.put(key, w.raw.objs[key] + b'!', who='foreign')

    def _operator(self) -> None:
        halted = self.ctl.state.halted
        if halted is None:
            return
        if self.sc.operator_resumes and halted.startswith('infra_breaker'):
            self.ctl.resume('operator checked the pipeline')
        elif self.sc.judge_bug == JUDGE_BUG_POISON and halted.startswith('effect_error'):
            self.world.judge.bug = None
            self.ctl.resume('operator fixed the judge')

    def _chaos_step(self) -> None:
        w, sc = self.world, self.sc
        if sc.foreign_write_at == self.step_no:
            w.foreign_pointer_write()
        if sc.tamper is not None:
            self._tamper()
        roll = w.rng.random()
        if roll < 0.46:
            self._controller_action()
        elif roll < 0.76:
            jobs = [box.job for box in self._boxes() if box.job is not None and not box.job.done]
            if jobs:
                w.rng.choice(jobs).step()
        elif roll < 0.84:
            if w.late:
                w.land(w.rng.randrange(len(w.late)))
        elif roll < 0.90:
            w.clock.advance(w.rng.uniform(0.0, sc.max_jump_s))
        elif roll < 0.96:
            self._arrive()
        else:
            kind = w.faults.pick('chaos', ('kill_box', 'kill_box', 'crash'))
            if kind == 'kill_box':
                boxes = self._boxes()
                if boxes:
                    w.rng.choice(boxes).alive = False
            elif kind == 'crash':
                raise SimCrash('between two steps')
        w.clock.advance(1.0)

    def _fair_round(self) -> None:
        """One round of a world that works: every part gets its turn."""
        w = self.world
        while self.todo:
            self._arrive()
        while w.late:
            w.land(0)
        for box in self._boxes():
            if box.job is not None:
                box.job.step()
        w.clock.advance(5.0)
        for key in sorted(self.mailbox):
            self._deliver(key)
        self.ctl.step()
        self._sticky()

    def _quiescent(self) -> bool:
        w, state = self.world, self.ctl.state
        if self.todo or self.mailbox or w.late or self._boxes():
            return False
        if state.doomed or state.dirty or state.unpublished or w.raw.list(w.layout.staging_root):
            return False
        if state.halted is not None:
            return not state.by_fence
        if state.head is None or state.pending is not None or state.active:
            return False
        return len(state.done) == len(w.source.expected()[0])

    # -- run -----------------------------------------------------------------

    def run(self) -> Report:
        self._boot()
        if self.sc.chaos:
            self.chaos()
        return self.finish()

    def chaos(self) -> None:
        """The hostile part: a random scheduler, faults and crashes."""
        w = self.world
        w.crashing = self.sc.crash_budget > 0
        for self.step_no in range(self.sc.chaos_steps):
            try:
                self._chaos_step()
            except SimCrash:
                self._restart()
        w.crashing = False

    def fair(self, rounds: int) -> None:
        """Run some rounds of a world that works."""
        for _ in range(rounds):
            try:
                self._fair_round()
            except SimCrash:
                self._restart()

    def finish(self) -> Report:
        """Fair rounds until nothing is left to do, and then the audit."""
        w, sc = self.world, self.sc
        # From here on the world is fair: no new faults, every part runs.
        w.faults.quiet = True
        w.crashing = False
        rounds = 0
        while True:
            try:
                self._operator()
                if self._quiescent():
                    break
                self._fair_round()
            except SimCrash:
                self._restart()
            rounds += 1
            if rounds > sc.max_rounds:
                w.audit.fail('not every batch became terminal in %d fair rounds: %s' % (
                    sc.max_rounds, json.dumps(self.ctl.status()['batches'], sort_keys=True),
                ))
        if w.audit.first is not None:
            raise AuditFailure(w.audit.first)
        w.audit.final(self)
        state = self.ctl.state
        report = self.report
        report.points = w.points
        report.rounds = rounds
        report.records = self.ctl.journal.count
        report.head_seq = state.head.seq if state.head is not None else -1
        report.halted = state.halted
        outcomes: Dict[str, int] = collections.Counter()
        for batch in list(state.done.values()) + list(state.active.values()):
            outcomes[batch.phase.value] += 1
        report.outcomes = dict(outcomes)
        report.faults = dict(w.faults.used)
        report.kinds = dict(w.audit.kinds)
        report.crash_labels = list(w.crash_labels)
        self.ctl.close()
        return report


def run_one(base: Path, seed: int, sc: Scenario, tag: str = '') -> Report:
    return Harness(base, seed, sc, tag).run()


__all__ = ['AuditFailure', 'Harness', 'Report', 'Scenario', 'SimCrash', 'run_one']
