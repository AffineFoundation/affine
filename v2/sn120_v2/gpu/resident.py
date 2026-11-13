"""Verified local copies of the head, the fp32 master and LoRAs.

The object store is the record. The local disk is a cache that can be lost.
A file set is used only after every file was hashed in this process and the
manifest digest equals the expected identity. A copy that does not match is
deleted, never repaired and never used.
"""

from __future__ import annotations

import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from .. import layout
from ..canonical import file_sha256, manifest_digest
from ..errors import InfraError, VerifyError
from ..seams import ObjectStore
from ..types import HeadVersion


def hash_tree(path: Path) -> dict[str, str]:
    """``{file name: sha256}`` of every file below ``path``."""
    return {file.relative_to(path).as_posix(): file_sha256(file) for file in sorted(path.rglob("*")) if file.is_file()}


def upload_tree(store: ObjectStore, path: Path, prefix: str, workers: int = 4) -> tuple[dict[str, str], dict[str, int]]:
    """Upload every file to ``<prefix>/<file name>`` and check what the store holds.

    Returns the hashes and the sizes. The hash comes from the local file and
    must equal the hash the store computes: an upload is not trusted blind.
    """
    files = hash_tree(path)

    def one(name: str) -> None:
        store.put_file(f"{prefix}/{name}", path / name)
        if store.sha256(f"{prefix}/{name}") != files[name]:
            raise InfraError(f"the stored copy of {prefix}/{name} differs from the local file")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files))
    return files, {name: (path / name).stat().st_size for name in files}


# One lock and one set of verified identities per cache directory, shared by every Resident of this process.
_SHARED: dict[Path, tuple[threading.RLock, set[str]]] = {}
_SHARED_LOCK = threading.Lock()


class Resident:
    """The cache directory of one box: ``<work_dir>/cache/<identity>/<file name>``.

    The trainer, the evaluator and the folder each hold a Resident of the
    same directory. They share the lock and the list of what is verified, so
    a 56 GB head is hashed one time in a process, not one time per part.
    """

    def __init__(self, store: ObjectStore, work_dir: Path):
        self.store = store
        self.root = Path(work_dir) / "cache"
        self.root.mkdir(parents=True, exist_ok=True)
        with _SHARED_LOCK:  # training threads of one wave share this cache too
            self._lock, self._verified = _SHARED.setdefault(self.root.resolve(), (threading.RLock(), set()))

    def tree(self, prefix: str, identity: str) -> Path:
        """The local directory of the file set stored under ``prefix``, verified against ``identity``."""
        path = self.root / identity
        with self._lock:
            if identity in self._verified and path.is_dir():
                return path
            self._verified.discard(identity)
            for fresh in (False, True):
                if fresh or not path.is_dir():
                    self._download(prefix, path)
                files = hash_tree(path)
                if files and manifest_digest(files) == identity:
                    self._verified.add(identity)
                    return path
                shutil.rmtree(path)
        raise VerifyError(f"the files under {prefix} do not hash to {identity[:16]}")

    def adopt(self, path: Path, identity: str) -> None:
        """Take a directory this box just built and uploaded, so it is not downloaded again."""
        if manifest_digest(hash_tree(path)) != identity:
            raise VerifyError(f"the directory does not hash to {identity[:16]}")
        with self._lock:
            shutil.rmtree(self.root / identity, ignore_errors=True)
            path.rename(self.root / identity)
            self._verified.add(identity)

    def prune(self, keep: Iterable[str]) -> None:
        """Delete every cached file set that is not named in ``keep``."""
        keep = set(keep)
        with self._lock:
            for path in self.root.iterdir():
                if path.name not in keep:
                    self._verified.discard(path.name)
                    shutil.rmtree(path, ignore_errors=True)

    def _download(self, prefix: str, path: Path) -> None:
        keys = self.store.list(prefix)
        if not keys:
            raise VerifyError(f"nothing is stored under {prefix}")
        if sum(self.store.size(key) for key in keys) > shutil.disk_usage(self.root).free:
            raise InfraError(f"not enough free disk for {prefix}")
        part = path.with_name(path.name + ".part")
        shutil.rmtree(part, ignore_errors=True)
        for key in keys:
            self.store.get_file(key, part / key[len(prefix) + 1 :])
        shutil.rmtree(path, ignore_errors=True)
        part.rename(path)


def resident_head(resident: Resident, head: HeadVersion) -> Path:
    """The verified local bf16 base of ``head``. Fails closed on any difference.

    The fp32 master is verified too: a head without a complete master cannot
    be folded, so nothing may be trained or scored on it.
    """
    if head.loras:
        raise InfraError("the head has LoRAs that are not folded; fold it before a round")
    resident.tree(layout.public_master_prefix(head.base_fp32_master_digest), head.base_fp32_master_digest)
    return resident.tree(layout.public_base_prefix(head.base_hash), head.base_hash)
