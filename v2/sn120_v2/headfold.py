"""The fold, the checkpoint and the rebuild of a head whose weight files stay on the box (``chain`` mode).

    python -m sn120_v2.headfold rebuild --chain SOURCE --work DIR [--links SOURCE | --store] [--head VERSION_ID] [--device cpu|cuda:N]

``headchain.py`` has the public record. This module has the work on files:

* ``ChainFolder`` is the ``Folder`` seam of the box. It folds from the local
  fp32 master, keeps the new master and the new base on the local disk, and
  uploads only the small files of the base and the fold record.
* ``HeadKeeper`` proves the local head when ``run`` starts, and rebuilds it
  from the store when the disk lost it. It also makes the checkpoints: hard
  links of one head, then the upload of that head in a background thread.
* ``rebuild`` makes a head from the newest checkpoint and the LoRAs of the
  steps after it. Every step is compared with the hashes of the chain record.

A kernel name (``headchain.KERNELS``) is the rule of the arithmetic. An
implementation is code that follows a rule:

``reference``
    ``fold_kernel.replay_fold``: one tensor at a time, about 1 GB of memory.
    It gives the bytes of the rule on every machine.
``blas``
    ``gpu/folder.py``: torch's matrix product, one whole weight file in
    memory. It is the rule ``torch-matmul-v1``. For a ``fixed-order-v1`` step
    its bytes are right only on a machine whose product equals the rule, so
    the hashes must decide: on the box the second fold, for a reader the record.

A fold of the box runs in a child process, never in the ``run`` process: a
CUDA context there would stay for the life of the loop.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from . import fold_kernel, layout
from .afterking import Beside
from .canonical import file_sha256, manifest_digest
from .errors import ConfigError, ContractError, HeadLost, InfraError, V2Error, VerifyError
from .gpu.folder import fold_files, read_lora
from .gpu.plan import BoxConfig, check_pins, log_phase
from .gpu.resident import Resident, hash_tree
from .headchain import FIXED_ORDER, FOLD_RECORD, FOLD_SCHEMA, KERNELS, MATMUL, ChainWriter, check_chain, checkpoints, heads_of, marker, route, small_files
from .params import Params
from .r2 import R2Store
from .seams import ObjectStore
from .state import State
from .store import log_time
from .types import FoldArtifact, HeadManifest, HeadVersion

REFERENCE, BLAS = "reference", "blas"
IMPLEMENTATION_ENV, TWICE_ENV, DEVICES_ENV, THREADS_ENV = "SN120_FOLD_IMPLEMENTATION", "SN120_FOLD_TWICE", "SN120_FOLD_DEVICES", "SN120_FOLD_THREADS"
CPU = "cpu"
BASE, MASTER, LORA = "base", "master", "lora"
UPLOADS = 2  # files of a checkpoint that go to the store at one time
READ_BACK, READ_BACK_THREADS = "sn120-v2-read-back", 4  # the name of the threads of ``ReadBack``, and how many files of a new head they hash at one time
BLOCK = 1 << 20
# The public site answers 403 to the default agent of urllib for the objects under public/, so every request names this one.
AGENT = {"User-Agent": "sn120-v2-replicate/1"}
_KEY_OF = {BASE: layout.public_base_file, MASTER: layout.public_master_file, LORA: layout.public_lora_file}
_IDS = (("base", "base_hash"), ("master", "base_fp32_master_digest"))


def say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


# ---- one fold ----


@dataclasses.dataclass(frozen=True)
class FoldSettings:
    """How this machine folds. ``from_env`` reads the four variables; a wrong value is an error."""

    implementation: str = REFERENCE  # the code of the first fold: ``reference`` or ``blas``
    twice: bool = True  # the box folds a second time with ``reference`` and compares the hashes
    devices: tuple[str, ...] = (CPU,)  # where the product of ``reference`` runs; the second fold takes the second device
    threads: int | None = None  # threads of torch on the CPU for one fold (``None``: ``in_children`` gives each fold its share); the time changes, never the bytes

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ, box: bool = True) -> FoldSettings:
        implementation = environ.get(IMPLEMENTATION_ENV, "").strip() or REFERENCE
        if implementation not in (REFERENCE, BLAS):
            raise ConfigError(f"{IMPLEMENTATION_ENV} must be {REFERENCE!r} or {BLAS!r}")
        twice = box and environ.get(TWICE_ENV, "").strip() != "0"
        if box and implementation == BLAS and not twice:
            raise ConfigError(f"{IMPLEMENTATION_ENV}={BLAS} needs the second fold: without it nothing proves that its bytes are those of {FIXED_ORDER}. Remove {TWICE_ENV}=0")
        devices = tuple(part.strip() for part in environ.get(DEVICES_ENV, "").split(",") if part.strip()) or (CPU,)
        if any(device != CPU and not re.fullmatch(r"cuda:\d+", device) for device in devices):
            raise ConfigError(f"{DEVICES_ENV} must be a list of cpu or cuda:N, for example cuda:0,cuda:1")
        threads = environ.get(THREADS_ENV, "").strip()
        if threads and (not threads.isdigit() or int(threads) < 1):
            raise ConfigError(f"{THREADS_ENV} must be a whole number, 1 or more")
        return cls(implementation, twice, devices, int(threads) if threads else None)


@dataclasses.dataclass(frozen=True)
class FoldJob:
    """One fold. An output directory must not exist. ``None``: that output is hashed and not written."""

    master_dir: Path
    base_dir: Path  # read for the file names, the tensor order, the dtypes and the small files
    loras: tuple[tuple[Path, int], ...]  # (directory, rank) of each LoRA, in the order of the head
    out_master: Path | None
    out_base: Path | None
    kernel: str = FIXED_ORDER
    implementation: str = REFERENCE
    device: str = CPU
    threads: int | None = None


@dataclasses.dataclass(frozen=True)
class FoldResult:
    base_files: dict[str, str]
    master_files: dict[str, str]
    device: str  # where the product ran
    seconds: float
    note: str = ""  # why the device is not the one of the job

    @property
    def ids(self) -> tuple[str, str]:
        """(base_hash, base_fp32_master_digest) of the folded files."""
        return manifest_digest(self.base_files), manifest_digest(self.master_files)


Runner = Callable[[Sequence[FoldJob]], list[FoldResult]]


def run_fold(job: FoldJob) -> FoldResult:
    """Do one fold in this process. A device that fails gives the fold to the CPU: the same bytes, more time."""
    started = time.monotonic()
    if job.kernel not in KERNELS:
        raise VerifyError(f"this code does not know the kernel {job.kernel!r} (it knows {', '.join(KERNELS)}); a newer version of this code is needed")
    try:
        if job.implementation == BLAS:
            if job.out_master is None or job.out_base is None:
                raise ContractError(f"the implementation {BLAS} writes its files; it cannot hash only")
            fold_files(job.master_dir, job.base_dir, [read_lora(path, rank) for path, rank in job.loras], job.out_master, job.out_base)
            return FoldResult(hash_tree(job.out_base), hash_tree(job.out_master), CPU, time.monotonic() - started)
        loras = [fold_kernel.read_pairs(path, rank) for path, rank in job.loras]
        device, note = CPU if job.kernel == MATMUL else job.device, ""

        def fold(on: str) -> fold_kernel.Folded:
            return fold_kernel.replay_fold(job.master_dir, job.base_dir, loras, job.out_master, job.out_base, kernel=job.kernel, device=on, threads=job.threads)

        if device != CPU:
            try:
                fold_kernel.check_arithmetic(device, job.threads)
            except Exception as exc:  # noqa: BLE001 - no driver, no room, wrong answers: each one means "not this device"
                device, note = CPU, f"{job.device} did not pass its check ({type(exc).__name__}), so the CPU did this fold"
        try:
            folded = fold(device)
        except RuntimeError as exc:  # torch's error for a GPU without room, or one that fails in the middle
            if device == CPU:
                raise
            for out in (job.out_master, job.out_base):
                if out is not None:
                    shutil.rmtree(out, ignore_errors=True)
            device, note = CPU, f"{job.device} failed in the fold ({type(exc).__name__}), so the CPU did this fold again"
            folded = fold(device)
        return FoldResult(folded.base_files, folded.master_files, device, time.monotonic() - started, note)
    except V2Error:
        raise
    except Exception as exc:  # noqa: BLE001 - the controller must see a failed fold, not a traceback
        raise InfraError(f"the fold stopped by {type(exc).__name__}: {exc}") from None


def inline(jobs: Sequence[FoldJob]) -> list[FoldResult]:
    """Each fold in this process, one after the other. For a reader's tool, which is its own process."""
    return [run_fold(job) for job in jobs]


def _fold_in_child(job: FoldJob, threads: int) -> FoldResult:
    torch.set_num_threads(threads)
    return run_fold(job)


def in_children(jobs: Sequence[FoldJob]) -> list[FoldResult]:
    """Each fold in a new process (start method ``spawn``), all at the same time. The first error is raised after all ended.

    The folds share the threads of torch that this process has. Without that, each fold has a full team
    of threads on the same cores, and the teams wait on each other: two folds then take many times the time of one.
    """
    share = max(1, torch.get_num_threads() // len(jobs))
    results, errors = [], []
    with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in [pool.submit(_fold_in_child, job, job.threads or share) for job in jobs]:
            try:
                results.append(future.result())
            except V2Error as exc:
                errors.append(exc)
            except Exception as exc:  # noqa: BLE001 - a child that the system killed
                errors.append(InfraError(f"a fold process stopped: {type(exc).__name__}: {exc}"))
    if errors:
        raise errors[0]
    return results


def fold_checked(job: FoldJob, settings: FoldSettings, run: Runner, label: str) -> list[FoldResult]:
    """Fold by the settings: one fold that writes, and with ``twice`` a ``reference`` fold that only hashes. Equal hashes, or ``VerifyError``."""
    devices = settings.devices
    jobs = [dataclasses.replace(job, implementation=settings.implementation, device=devices[0], threads=settings.threads)]
    if settings.twice:
        jobs.append(dataclasses.replace(job, out_master=None, out_base=None, implementation=REFERENCE, device=devices[1 % len(devices)], threads=settings.threads))
    results = run(jobs)
    for number, (one, result) in enumerate(zip(jobs, results), 1):
        note = f" ({result.note})" if result.note else ""
        log_time(f"{label}: fold {number} of {len(jobs)}, {one.implementation} on {result.device}{note}", time.monotonic() - result.seconds)
    if len({result.ids for result in results}) != 1:
        raise VerifyError(f"{label}: the two folds gave different files; nothing was kept")
    return results


def tree_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


class ReadBack:
    """The second sha256 of a head that a fold just wrote, in threads beside the next steps of ``run``.

    ``Resident.adopt`` makes this read before it takes the files. Here the files are in the cache already and
    the next round uses them while the read runs. So the caller must ask ``check`` before it trusts anything
    that was made with these files. A set that fails leaves the list of what is verified.
    """

    def __init__(self, resident: Resident, sets: Sequence[tuple[Path, str]], label: str, threads: int = READ_BACK_THREADS) -> None:
        self.resident, self.sets, self.label, self.threads = resident, tuple(sets), label, threads
        self._work = Beside(self._read, READ_BACK)

    def _read(self) -> None:
        started = time.monotonic()
        with ThreadPoolExecutor(self.threads, thread_name_prefix=READ_BACK) as pool:
            for path, identity in self.sets:
                try:
                    files = sorted(file for file in path.rglob("*") if file.is_file())
                    found = dict(zip((file.relative_to(path).as_posix() for file in files), pool.map(file_sha256, files)))
                    same = manifest_digest(found) == identity
                except OSError:
                    same = False
                if not same:
                    with self.resident._lock:  # noqa: SLF001 - ``gpu/resident.py`` has no call for this, and a change there changes the code digest
                        self.resident._verified.discard(identity)  # noqa: SLF001
                    say(f"{self.label}: the local files of {identity[:16]} do not read back as the fold wrote them")
                    raise VerifyError(f"the local files of {identity[:16]} do not read back as the fold wrote them")
        log_time(f"{self.label}: the second sha256 of the new head, beside the next steps", started)

    def wait(self) -> None:
        self._work.wait()

    def check(self, wait: bool) -> None:
        """Raises ``HeadLost`` when the read failed. ``wait``: first wait for its end; without it, a read that still runs passes."""
        if not wait and not self._work.done():
            return
        try:
            self._work.result()
        except VerifyError as exc:
            raise HeadLost(f"{exc}. Nothing that was made on these files is kept. Start run again: the start rebuilds the head from the chain") from None


class ChainFolder:
    """The ``Folder`` seam in ``chain`` mode: the folded weight files stay on this box.

    The store gets the small files of the new base and the fold record, under
    ``out_prefix``. ``headchain.promote_fold`` publishes them. The sha256 of
    every folded file goes into the head manifest, as in ``full`` mode.
    """

    def __init__(self, store: ObjectStore, config: BoxConfig, settings: FoldSettings | None = None, run: Runner = in_children) -> None:
        self.store, self.config, self.settings, self.run = store, config, settings or FoldSettings(), run
        self.resident = Resident(store, config.work_dir)
        self._read_back: ReadBack | None = None

    def fold(self, head: HeadVersion, params: Params, out_prefix: str) -> FoldArtifact:
        return self._fold(head, params, out_prefix, {}, ahead=False)

    def fold_ahead(self, head: HeadVersion, params: Params, out_prefix: str, lora_prefixes: Mapping[str, str]) -> FoldArtifact:
        """The fold for the order ``overlap`` (``afterking.py``). Two differences from ``fold``.

        ``head`` is not in the journal yet, so a LoRA of ``lora_prefixes`` (``{lora_hash: prefix}``) is read
        from that prefix and not from ``public/``. Its hash proves the bytes in both places.

        The new files go to the cache at once, and their second sha256 runs in threads (``ReadBack``).
        ``proven`` gives the result.
        """
        return self._fold(head, params, out_prefix, lora_prefixes, ahead=True)

    def proven(self, wait: bool) -> None:
        """``HeadPublish.proven``: raises ``HeadLost`` when the files of the last ``fold_ahead`` did not read back as written."""
        if self._read_back is not None:
            self._read_back.check(wait)

    def drop(self, head: HeadVersion) -> None:
        """The head did not move to the result of the last ``fold_ahead``: keep only the files of ``head`` in the cache."""
        if self._read_back is not None:
            self._read_back.wait()  # its files go now, so its result is of no use
            self._read_back = None
        self.resident.prune({head.base_hash, head.base_fp32_master_digest, *(ref.lora_hash for ref in head.loras)})

    def _rest(self) -> None:
        """Before the cache changes: wait for a second sha256 that still runs. One that failed raises ``HeadLost`` here, and at each later ask."""
        if self._read_back is not None:
            self._read_back.check(wait=True)
            self._read_back = None

    def _place(self, path: Path, identity: str) -> None:
        """Put a directory that the fold wrote in the cache, as ``Resident.adopt`` does, but before its second sha256 (``ReadBack``)."""
        with self.resident._lock:  # noqa: SLF001 - ``gpu/resident.py`` has no call for this, and a change there changes the code digest
            shutil.rmtree(self.resident.root / identity, ignore_errors=True)
            path.rename(self.resident.root / identity)
            self.resident._verified.add(identity)  # noqa: SLF001

    def _fold(self, head: HeadVersion, params: Params, out_prefix: str, lora_prefixes: Mapping[str, str], ahead: bool) -> FoldArtifact:
        check_pins(self.config.image_digest, params)
        started = time.monotonic()
        if not head.loras:
            raise ContractError("the head has no LoRA to fold")
        self._rest()
        self.resident.root.mkdir(parents=True, exist_ok=True)  # a box whose disk was replaced has no cache directory
        self.resident.prune({head.base_hash, head.base_fp32_master_digest, *(ref.lora_hash for ref in head.loras)})
        try:
            master_dir = self.resident.tree(layout.public_master_prefix(head.base_fp32_master_digest), head.base_fp32_master_digest)
            base_dir = self.resident.tree(layout.public_base_prefix(head.base_hash), head.base_hash)
        except (KeyError, VerifyError) as exc:
            raise VerifyError(f"the local files of the head are lost or changed ({exc}). Stop run and start it again: the start rebuilds the head from the chain") from None
        loras = tuple((self.resident.tree(lora_prefixes.get(ref.lora_hash) or layout.public_lora_prefix(ref.lora_hash), ref.lora_hash), ref.rank) for ref in head.loras)
        # A new directory for each try, and nothing of an older try: a fold process that outlived its ``run`` can then write into no file of this one.
        shutil.rmtree(Path(self.config.work_dir) / "fold", ignore_errors=True)
        out = Path(self.config.work_dir) / "fold" / f"{head.version_id}.{os.getpid()}.{time.monotonic_ns()}"
        need, free = tree_bytes(master_dir) + tree_bytes(base_dir), shutil.disk_usage(self.config.work_dir).free
        if free < need:
            raise InfraError(f"not enough free disk for the fold: {free} bytes are free, the new head needs {need}")
        try:
            fetched = time.monotonic()
            label = f"fold of {head.version_id[:12]}"
            results = fold_checked(FoldJob(master_dir, base_dir, loras, out / MASTER, out / BASE), self.settings, self.run, label)
            summed = time.monotonic()
            base_files, master_files = results[0].base_files, results[0].master_files
            base_hash, master_digest = results[0].ids
            for name, sha in sorted(small_files(base_files).items()):
                self.store.put_file(f"{out_prefix}/{BASE}/{name}", out / BASE / name)
                if self.store.sha256(f"{out_prefix}/{BASE}/{name}") != sha:
                    raise InfraError(f"the stored copy of {out_prefix}/{BASE}/{name} differs from the local file")
            record = {"schema": FOLD_SCHEMA, "stacked_version_id": head.version_id, "base_hash": base_hash, "base_fp32_master_digest": master_digest}
            record |= {"kernel": FIXED_ORDER, "torch": torch.__version__, "image_digest": self.config.image_digest}
            record |= {"folds": [{"implementation": REFERENCE if number else self.settings.implementation, "device": result.device} for number, result in enumerate(results)]}
            self.store.put(f"{out_prefix}/{FOLD_RECORD}", (json.dumps(record, indent=1, sort_keys=True) + "\n").encode("utf-8"))
            artifact = FoldArtifact(
                parent_version_id=head.version_id,
                base_hash=base_hash,
                base_files=base_files,
                base_fp32_master_digest=master_digest,
                master_files=master_files,
                prefix=out_prefix,
                image_digest=self.config.image_digest,
            )
            if ahead:
                self._place(out / BASE, base_hash)
                self._place(out / MASTER, master_digest)
                self._read_back = ReadBack(self.resident, [(self.resident.root / base_hash, base_hash), (self.resident.root / master_digest, master_digest)], label)
            else:
                self.resident.adopt(out / BASE, base_hash)  # the hash of the written files is checked here; the next round downloads nothing
                self.resident.adopt(out / MASTER, master_digest)
            times = {"fetch_s": fetched - started, "sum_s": summed - fetched, "upload_s": time.monotonic() - summed, "wall_s": time.monotonic() - started}
            log_phase(self.config.work_dir, "fold", version_id=head.version_id, loras=len(loras), mode="chain", **times)
            return artifact
        finally:
            shutil.rmtree(out, ignore_errors=True)


# ---- the rebuild ----


def read(source: str) -> bytes:
    """The bytes of a file or of an address. An error never shows the query of an address: a signed link holds a key."""
    if "://" not in source:
        try:
            return Path(source).read_bytes()
        except OSError as exc:
            raise ConfigError(f"cannot read {source}: {exc.strerror}") from None
    try:
        with urllib.request.urlopen(urllib.request.Request(source, headers=AGENT), timeout=60) as reply:  # noqa: S310 - the address comes from the user or from mining.json
            return reply.read()
    except OSError as exc:
        raise InfraError(f"cannot fetch {source.split('?')[0]}: {type(exc).__name__} {getattr(exc, 'code', '')}".strip()) from None


def read_json(source: str, what: str) -> dict[str, Any]:
    try:
        doc = json.loads(read(source))
    except ValueError:
        raise ConfigError(f"{what} is not JSON") from None
    if not isinstance(doc, dict):
        raise ConfigError(f"{what} is not a JSON object")
    return doc


def download(url: str, path: Path, sha256: str, size: int) -> None:
    """GET one file and keep it only when its size and its sha256 are the expected ones."""
    part, hasher, got = path.with_name(path.name + ".part"), hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=60) as reply, part.open("wb") as out:  # noqa: S310 - a link of read_links.json
            while block := reply.read(BLOCK):
                out.write(block)
                hasher.update(block)
                got += len(block)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise InfraError(f"the download of {path.name} failed after {got} bytes: {type(exc).__name__} {getattr(exc, 'code', '')}".strip()) from None
    if got != size or hasher.hexdigest() != sha256:
        part.unlink()
        raise VerifyError(f"the download of {path.name} has {got} bytes and is not the file of the head manifest ({size} bytes)")
    part.replace(path)


class Source(Protocol):
    """Where a rebuild gets files. ``fetched`` counts the bytes."""

    fetched: int

    def checkpoint(self, head: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
        """(base files, master files) of a checkpoint head, each ``{file name: sha256}``."""
        ...

    def get(self, kind: str, set_id: str, name: str, sha256: str, path: Path) -> None:
        """Put one file at ``path``, only when it has this sha256."""
        ...


class StoreSource:
    """The object store itself. For the box, and for a machine that has a key of the store."""

    def __init__(self, store: ObjectStore) -> None:
        self.store, self.fetched = store, 0

    def checkpoint(self, head: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
        try:
            manifest = HeadManifest.from_json(self.store.get(layout.head_manifest(head["version_id"])))
        except KeyError:
            raise VerifyError(f"the store has no manifest of head {head['version_id'][:12]}") from None
        return dict(manifest.base_files), dict(manifest.master_files)

    def get(self, kind: str, set_id: str, name: str, sha256: str, path: Path) -> None:
        try:
            self.store.get_file(_KEY_OF[kind](set_id, name), path)
        except KeyError:
            raise VerifyError(f"{name} of the {kind} {set_id[:12]} is not in the store") from None
        if file_sha256(path) != sha256:
            path.unlink()
            raise VerifyError(f"{name} of the {kind} {set_id[:12]} in the store does not match its sha256")
        self.fetched += path.stat().st_size


class LinkSource:
    """Signed read links: the links file of the site, or the result of ``headchain.chain_links``."""

    def __init__(self, links: Mapping[str, Any], now: Callable[[], float] = time.time) -> None:
        if now() >= int(links.get("expires_at") or 0):
            raise ConfigError("the links have expired; fetch the links file again")
        self.fetched, self.sets = 0, {}
        try:
            points = [part for point in links.get("checkpoints") or [] for part in (point[BASE], point[MASTER])]
            for group in [*points, links.get(BASE), links.get(MASTER), *(links.get("chain_loras") or []), *(links.get("loras") or [])]:
                if group:
                    self.sets.setdefault(str(group["id"]), {}).update({str(row["name"]): row for row in group["files"]})
        except (KeyError, TypeError, AttributeError):
            raise ConfigError("the links file is not complete; fetch it again") from None

    def _rows(self, set_id: str) -> Mapping[str, Mapping[str, Any]]:
        if set_id not in self.sets:
            raise ConfigError(f"the links file has no links for the file set {set_id[:12]}; fetch the links file and mining.json again")
        return self.sets[set_id]

    def checkpoint(self, head: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
        base, master = ({name: str(row["sha256"]) for name, row in self._rows(head[key]).items()} for _, key in _IDS)
        return base, master

    def get(self, kind: str, set_id: str, name: str, sha256: str, path: Path) -> None:
        row = self._rows(set_id).get(name)
        if row is None or row.get("sha256") != sha256:
            raise VerifyError(f"the links file does not list {name} of the {kind} {set_id[:12]} with the sha256 of the record")
        say(f"fetch {name} ({row['size']} bytes)")
        download(str(row["url"]), path, sha256, int(row["size"]))
        self.fetched += int(row["size"])


def fetch_set(source: Source, kind: str, set_id: str, files: Mapping[str, str], work: Path) -> Path:
    """``<work>/cache/<set_id>/`` with exactly ``files``, each checked. The files are collected in ``<work>/fetch/<set_id>/``; one that is there stays."""
    if any(not name or name.startswith("/") or ".." in name.split("/") for name in files):
        raise VerifyError(f"the {kind} {set_id[:12]} has an unsafe file name")
    cache, stage = work / "cache" / set_id, work / "fetch" / set_id
    if cache.is_dir():
        if hash_tree(cache) == dict(files):
            return cache
        shutil.rmtree(cache)
    for name, sha in sorted(files.items()):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (path.is_file() and file_sha256(path) == sha):
            path.unlink(missing_ok=True)
            source.get(kind, set_id, name, sha, path)
    for file in [file for file in stage.rglob("*") if file.is_file() and file.relative_to(stage).as_posix() not in files]:
        file.unlink()
    cache.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(cache)
    return cache


def cast_base(master_dir: Path, small: Sequence[Path], out_base: Path, scratch: Path) -> dict[str, str]:
    """Write the base of a master whose every weight is bf16 in the base. Returns ``{file name: sha256}``.

    A fold reads the base only for the names, the dtypes and the small files.
    So a stand-in base does: weight files that are a header and no values.
    The caller compares the result with the record; a base with another
    dtype gives other hashes there.
    """
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        stems: dict[str, dict[str, fold_kernel.Entry]] = {}
        for part in sorted(master_dir.glob("*.part*.safetensors")):
            stems.setdefault(part.name.rsplit(".part", 1)[0], {}).update(fold_kernel.read_index(part))
        for stem, index in stems.items():
            (scratch / f"{stem}.safetensors").write_bytes(fold_kernel.header_bytes([(name, "BF16", entry.shape, entry.nbytes // 2) for name, entry in index.items()]))
        for file in small:
            shutil.copy2(file, scratch / file.name)
        return fold_kernel.replay_fold(master_dir, scratch, [], None, out_base, kernel=MATMUL).base_files  # no LoRA: no product, so the kernel does not matter
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _checkpoint_files(source: Source, head: Mapping[str, Any], work: Path) -> None:
    """Bring a checkpoint head to ``<work>/cache``: the master is fetched, the base is the cast of the master and the small files."""
    base_files, master_files = source.checkpoint(head)
    if (manifest_digest(base_files), manifest_digest(master_files)) != (head["base_hash"], head["base_fp32_master_digest"]):
        raise VerifyError(f"the file list of checkpoint {head['version_id'][:12]} does not hash to the ids of the chain record")
    master_dir = fetch_set(source, MASTER, head["base_fp32_master_digest"], master_files, work)
    cache, stage = work / "cache" / head["base_hash"], work / "fetch" / head["base_hash"]
    if cache.is_dir() and hash_tree(cache) == base_files:
        return
    small = small_files(base_files)
    for name, sha in sorted(small.items()):
        if not ((stage / name).is_file() and file_sha256(stage / name) == sha):
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            source.get(BASE, head["base_hash"], name, sha, stage / name)
    out = work / "fold" / "cast"
    shutil.rmtree(out, ignore_errors=True)
    try:
        cast = cast_base(master_dir, [stage / name for name in sorted(small)], out, work / "fold" / "stand-in")
    except V2Error as exc:
        cast = {}
        say(f"the cast of the master stopped: {exc}")
    if cast == base_files:
        shutil.rmtree(cache, ignore_errors=True)
        out.rename(cache)
        shutil.rmtree(stage)
        return
    shutil.rmtree(out, ignore_errors=True)
    say("the cast of the master is not the base of the record, so the base weight files are fetched")
    fetch_set(source, BASE, head["base_hash"], base_files, work)


def _other_hashes(step: Mapping[str, Any], implementation: str) -> str:
    text = f"the step to head {step['version_id'][:12]} (kernel {step['kernel']}, implementation {implementation}) gave files with other hashes than the chain record; they were deleted. "
    if implementation == BLAS:
        return text + f"The product of torch on this machine is not the one of the record. Set {IMPLEMENTATION_ENV}={REFERENCE} (or remove the variable) and run again"
    if step["kernel"] == MATMUL:
        return text + f"The kernel {MATMUL} depends on the CPU and on the torch build, so only a machine like the box repeats it. Start from a checkpoint after this step"
    return text + "The reference code gives the same bytes on every machine, so this machine or a file on it has a fault. Run: python -m sn120_v2.fold_kernel marks"


def rebuild(
    doc: Mapping[str, Any], work: Path, source: Source, target: str | None = None, *, settings: FoldSettings | None = None, run: Runner = inline
) -> dict[str, Any]:
    """Make the folded head ``target`` (default: the last one of ``doc``) in ``<work>/cache``. Returns where it is.

    The start is the newest head before ``target`` that this machine holds,
    else the newest checkpoint. Each step is one fold, and its files must
    have the hashes of the chain record: a step that differs is deleted and
    the rebuild stops. The head before a step is deleted after the step.
    """
    check_chain(doc)
    settings = dataclasses.replace(settings or FoldSettings(), twice=False)  # the record decides here, not a second fold
    target = target or heads_of(doc)[-1]["version_id"]
    work = Path(work).resolve()
    cache, known = work / "cache", {}

    def held(head: Mapping[str, Any]) -> bool:
        if head["version_id"] not in known:
            known[head["version_id"]] = all((cache / head[key]).is_dir() and manifest_digest(hash_tree(cache / head[key])) == head[key] for _, key in _IDS)
        return known[head["version_id"]]

    at, steps = route(doc, target, held)
    start = at
    if not held(start):
        say(f"start: checkpoint {start['version_id'][:12]}, then {len(steps)} step(s)")
        _checkpoint_files(source, start, work)
    for step in steps:
        loras = tuple((fetch_set(source, LORA, lora["lora_hash"], lora["files"], work), int(lora["rank"])) for lora in step["loras"])
        out = work / "fold" / step["version_id"]
        shutil.rmtree(out, ignore_errors=True)
        job = FoldJob(cache / at["base_fp32_master_digest"], cache / at["base_hash"], loras, out / MASTER, out / BASE, kernel=step["kernel"])
        try:
            result = fold_checked(job, settings, run, f"step to head {step['version_id'][:12]}")[0]
            if result.ids != (step["base_hash"], step["base_fp32_master_digest"]):
                raise VerifyError(_other_hashes(step, settings.implementation))
            for kind, key in _IDS:
                shutil.rmtree(cache / step[key], ignore_errors=True)
                (out / kind).rename(cache / step[key])
        finally:
            shutil.rmtree(out, ignore_errors=True)
        for stale in {at["base_hash"], at["base_fp32_master_digest"], *(path.name for path, _ in loras)} - {step["base_hash"], step["base_fp32_master_digest"]}:
            shutil.rmtree(cache / stale, ignore_errors=True)
        at = step
    result = {"version_id": at["version_id"], "base_hash": at["base_hash"], "base_dir": str(cache / at["base_hash"])}
    result |= {"base_fp32_master_digest": at["base_fp32_master_digest"], "master_dir": str(cache / at["base_fp32_master_digest"])}
    return result | {"started_from": start["version_id"], "steps": len(steps), "fetched_bytes": source.fetched}


# ---- the local head of the box, and its checkpoints ----


def last_folded(state: State) -> HeadVersion:
    return next(head for head in reversed(state.heads) if head.folded)


class HeadKeeper:
    """The box's own copy of the head in ``chain`` mode.

    ``prepare`` runs when ``run`` starts: the local base and master must hash
    to the head, else they are rebuilt from the store. ``after_round`` makes
    a checkpoint when one is due: hard links of the head's files in
    ``<work_dir>/checkpoint/<version_id>/`` (so the next round's cleanup
    cannot remove them), then the upload in a thread. The marker is written
    last, after each file in the store was read back and hashed.
    """

    def __init__(
        self,
        store: ObjectStore,
        work_dir: Path,
        writer: ChainWriter,
        every: int,
        settings: FoldSettings | None = None,
        run: Runner = in_children,
        proven: Callable[[bool], None] | None = None,
    ) -> None:
        self.store, self.work, self.writer, self.every, self.settings, self.run = store, Path(work_dir), writer, every, settings or FoldSettings(), run
        self.proven = proven  # ``ChainFolder.proven``: the upload thread waits for the second sha256 of a head before it sends a file of it
        self.resident = Resident(store, work_dir)
        self.links = self.work / "checkpoint"
        self._lock = threading.Lock()
        self._jobs: dict[str, HeadManifest] = {}
        self._working = False
        self._thread: threading.Thread | None = None
        self._settled, self._problem = "", ("", "")

    def _proof(self, head: HeadVersion) -> None:
        self.resident.root.mkdir(parents=True, exist_ok=True)
        self.resident.tree(layout.public_master_prefix(head.base_fp32_master_digest), head.base_fp32_master_digest)
        self.resident.tree(layout.public_base_prefix(head.base_hash), head.base_hash)

    def prepare(self, state: State) -> None:
        head = last_folded(state)
        try:
            self._proof(head)
        except (KeyError, VerifyError) as exc:
            say(f"the local files of head {head.version_id[:12]} are not there ({exc}); the head is rebuilt from the store")
            rebuild(self.writer.doc(state), self.work, StoreSource(self.store), head.version_id, settings=self.settings, run=self.run)
            self._proof(head)
        self.after_round(state)

    def _linked(self) -> set[str]:
        return {path.name for path in self.links.iterdir() if path.is_dir() and not path.name.endswith(".part")} if self.links.is_dir() else set()

    def due(self, state: State) -> bool:
        """True when the last folded head must become a checkpoint now.

        That is after ``every`` folds with no checkpoint. A head that ``full``
        mode published has its files in the store already, so it is due at
        once: its files are hashed there and it gets its marker.
        """
        doc = self.writer.doc(state)
        heads, covered = heads_of(doc), self._linked()
        since = next(count for count, head in enumerate(reversed(heads)) if head["checkpoint"] or head["version_id"] in covered)
        return since > 0 and (heads[-1]["kernel"] == MATMUL or 0 < self.every <= since)

    def link(self, manifest: HeadManifest) -> None:
        """Hard links of the local files of one head. They cost no disk now; they keep the head when the cache drops it."""
        head, target = manifest.head, self.links / manifest.head.version_id
        if target.is_dir():
            return
        sets = ((BASE, head.base_hash, manifest.base_files), (MASTER, head.base_fp32_master_digest, manifest.master_files))
        part = target.with_name(target.name + ".part")
        shutil.rmtree(part, ignore_errors=True)
        try:
            size = sum((self.resident.root / identity / name).stat().st_size for _, identity, files in sets for name in files)
            free = shutil.disk_usage(self.work).free
            if free < 2 * size:  # the two next folds each write a new head while this one waits for its upload
                raise InfraError(f"{free} bytes of disk are free; a checkpoint needs room for two more heads ({2 * size} bytes)")
            for kind, identity, files in sets:
                for name in files:
                    (part / kind / name).parent.mkdir(parents=True, exist_ok=True)
                    os.link(self.resident.root / identity / name, part / kind / name)
            part.rename(target)
        except OSError as exc:
            shutil.rmtree(part, ignore_errors=True)
            raise InfraError(f"the local files of head {head.version_id[:12]} cannot be linked: {type(exc).__name__}: {exc}") from None

    def upload(self, manifest: HeadManifest) -> None:
        """Put the linked files of one head in the store, read each one back, then write the marker. A file that is there already is not sent."""
        head, started = manifest.head, time.monotonic()
        root = self.links / head.version_id
        items = [(BASE, name, sha, layout.public_base_file(head.base_hash, name)) for name, sha in sorted(manifest.base_files.items())]
        items += [(MASTER, name, sha, layout.public_master_file(head.base_fp32_master_digest, name)) for name, sha in sorted(manifest.master_files.items())]

        def one(item: tuple[str, str, str, str]) -> tuple[str, str, int]:
            kind, name, sha, key = item
            path = root / kind / name
            try:
                there = self.store.sha256(key) == sha
            except KeyError:
                there = False
            if not there:
                self.store.put_file(key, path)
                if self.store.sha256(key) != sha:
                    raise InfraError(f"the stored copy of {key} differs from the file of the fold")
            return kind, name, self.store.size(key)

        with ThreadPoolExecutor(UPLOADS) as pool:
            sizes = list(pool.map(one, items))
        base_sizes, master_sizes = ({name: size for kind, name, size in sizes if kind == wanted} for wanted in (BASE, MASTER))
        key = layout.head_checkpoint(head.version_id)
        data = (json.dumps(marker(manifest, base_sizes, master_sizes), indent=1, sort_keys=True) + "\n").encode("utf-8")
        self.store.put(key, data)
        if self.store.get(key) != data:
            raise InfraError(f"{key} does not read back as written")
        self.writer.mark(head.version_id)
        shutil.rmtree(root, ignore_errors=True)
        log_time(f"checkpoint of {head.version_id[:12]}: {len(items)} files in the store, each one read back, and the marker", started)

    def after_round(self, state: State) -> None:
        """Link the head when a checkpoint is due, and start the upload thread when a linked head waits. An error here stops no round.

        ``run`` calls it after each pass of its loop, so an upload that stopped starts again there.
        """
        head = last_folded(state).version_id
        try:
            if head != self._settled and self.due(state):
                self.link(state.manifests[head])
            self._settled = head  # the answer for this head stays: the next pass does not read the store for it
        except V2Error as exc:
            if self._problem != (head, type(exc).__name__):  # one line for a problem that stays, not one for each pass
                say(f"checkpoint: not started: {type(exc).__name__}: {exc}")
            self._problem = (head, type(exc).__name__)
        linked = self._linked()
        for version_id in linked - set(state.manifests):  # not a head of this journal
            shutil.rmtree(self.links / version_id, ignore_errors=True)
        with self._lock:
            for head in state.heads:  # oldest first
                if head.version_id in linked:
                    self._jobs.setdefault(head.version_id, state.manifests[head.version_id])
            if self._jobs and not self._working:
                self._working = True
                self._thread = threading.Thread(target=self._work, name="head-checkpoint", daemon=True)
                self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the upload thread. True when no upload runs after the wait."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def _work(self) -> None:
        while True:
            with self._lock:
                if not self._jobs:
                    self._working = False
                    return
                version_id, manifest = next(iter(self._jobs.items()))
            try:
                if self.proven is not None:
                    self.proven(True)
                self.upload(manifest)
            except Exception as exc:  # noqa: BLE001 - a thread that dies must say why; the next round starts it again
                say(f"checkpoint of {version_id[:12]}: stopped by {type(exc).__name__}: {exc}; it starts again at the next pass of run")
                with self._lock:
                    self._working = False
                return
            with self._lock:
                self._jobs.pop(version_id, None)

    def checkpoint(self, state: State) -> dict[str, Any]:
        """The command by hand: make the last folded head a checkpoint now, in this process."""
        head = last_folded(state)
        if head.version_id in checkpoints(self.store):
            return {"version_id": head.version_id, "checkpoint": True, "uploaded": False}
        manifest = state.manifests[head.version_id]
        self.link(manifest)
        try:
            self.upload(manifest)
        except (OSError, KeyError, InfraError):
            if head.version_id not in checkpoints(self.store):  # ``run`` can end the same upload first, and it then removes the links
                raise
        return {"version_id": head.version_id, "checkpoint": True, "uploaded": True}


# ---- the command ----


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sn120_v2.headfold", description="Rebuild a head from the head chain: the newest checkpoint and the LoRAs of the steps after it.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("rebuild")
    build.add_argument("--chain", required=True, metavar="SOURCE", help="mining.json or head_chain.json: a file or an address")
    build.add_argument("--work", required=True, type=Path, help="the work directory; the head goes to <work>/cache/")
    where = build.add_mutually_exclusive_group()
    where.add_argument("--links", metavar="SOURCE", help="the links file (read_links.json): a file or an address (default: the one that mining.json names)")
    where.add_argument("--store", action="store_true", help="read the object store with the SN120_R2_* variables of this machine")
    build.add_argument("--head", metavar="VERSION_ID", help="the folded head to make (default: the last one of the chain)")
    build.add_argument("--device", help=f"where the product runs: cpu or cuda:N (default: {DEVICES_ENV}, else cpu)")
    args = parser.parse_args(argv)
    try:
        settings = FoldSettings.from_env({**os.environ, DEVICES_ENV: args.device} if args.device else os.environ, box=False)
        doc = read_json(args.chain, "the chain file")
        if args.store:
            source: Source = StoreSource(R2Store.from_env())
        else:
            contract = doc.get("contract") or {}
            if not args.links and not (contract.get("public_url") and contract.get("read_links")):
                raise ConfigError("the chain file names no links file; pass --links or --store")
            source = LinkSource(read_json(args.links or f"{str(contract['public_url']).rstrip('/')}/{contract['read_links']}", "the links file"))
        result = rebuild(doc.get("head_chain") or doc, args.work, source, args.head, settings=settings, run=inline)
    except V2Error as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
