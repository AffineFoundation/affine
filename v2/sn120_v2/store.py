"""A local object store, and verify-then-promote on top of any ``ObjectStore``.

Promotion rule: a box uploads to ``pending/``. The controller checks every
file against its sha256, copies it to its content-addressed key under
``public/``, and checks the copy. A missing file or a wrong hash raises
``VerifyError`` and nothing moves. A pointer moves only after that.

Only keys under ``pending/`` are ever deleted here. So cleanup can never
remove the live head, a master or a miner's file.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, TypeVar

from . import layout
from .canonical import Record, file_sha256
from .errors import ContractError, InfraError, VerifyError
from .seams import ObjectStore
from .types import FoldArtifact, HeadManifest, LoRAArtifact

R = TypeVar("R", bound=Record)
TMP = ".part"


class LocalStore:
    """An ``ObjectStore`` on local disk: one file per key under a root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        parts = key.split("/")
        if not key or key.startswith("/") or any(part in ("", ".", "..") for part in parts) or key.endswith(TMP):
            raise ContractError(f"unsafe object key {key!r}")
        return self.root.joinpath(*parts)

    @contextmanager
    def _io(self, key: str) -> Iterator[None]:
        try:
            yield
        except FileNotFoundError:
            raise KeyError(key) from None
        except OSError as exc:
            raise InfraError(f"local store failed on {key!r}: {exc}") from exc

    def _write(self, key: str, fill) -> None:
        path = self._path(key)
        with self._io(key):
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + TMP)
            fill(partial)
            os.replace(partial, path)

    def put(self, key: str, data: bytes) -> None:
        self._write(key, lambda partial: partial.write_bytes(data))

    def put_file(self, key: str, path: Path) -> None:
        self._write(key, lambda partial: shutil.copyfile(path, partial))

    def get(self, key: str) -> bytes:
        with self._io(key):
            return self._path(key).read_bytes()

    def get_file(self, key: str, path: Path) -> None:
        source, target = self._path(key), Path(path)
        with self._io(key):
            if not source.is_file():
                raise FileNotFoundError(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + TMP)
            shutil.copyfile(source, partial)
            os.replace(partial, target)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int:
        with self._io(key):
            return self._path(key).stat().st_size

    def sha256(self, key: str) -> str:
        with self._io(key):
            return file_sha256(self._path(key))

    def copy(self, src: str, dst: str) -> None:
        source = self._path(src)
        if not source.is_file():
            raise KeyError(src)
        if src != dst:
            self._write(dst, lambda partial: shutil.copyfile(source, partial))

    def delete(self, key: str) -> None:
        with self._io(key):
            self._path(key).unlink(missing_ok=True)

    def list(self, prefix: str) -> list[str]:
        base = self.root.joinpath(*[part for part in prefix.split("/") if part])
        if prefix and not prefix.endswith("/") and base.is_file():
            return [prefix]
        if not base.is_dir():
            return []
        found = (path for path in base.rglob("*") if path.is_file() and not path.name.endswith(TMP))
        return sorted(path.relative_to(self.root).as_posix() for path in found)


def put_record(store: ObjectStore, key: str, record: Record) -> None:
    store.put(key, record.to_json())


def get_record(store: ObjectStore, key: str, cls: type[R]) -> R:
    return cls.from_json(store.get(key))


def verify_files(store: ObjectStore, prefix: str, files: Mapping[str, str], sizes: Mapping[str, int] | None = None) -> None:
    """Fail closed unless every file is present under ``prefix`` with its size and sha256."""
    for name, expected in sorted(files.items()):
        key = f"{prefix}/{name}"
        try:
            if sizes is not None and store.size(key) != sizes[name]:
                raise VerifyError(f"{key} has the wrong size")
            actual = store.sha256(key)
        except KeyError:
            raise VerifyError(f"{key} is missing") from None
        if actual != expected:
            raise VerifyError(f"{key} does not match its sha256")


def promote_files(
    store: ObjectStore, src_prefix: str, dst_prefix: str, files: Mapping[str, str], sizes: Mapping[str, int] | None = None
) -> None:
    """Verify in ``pending/``, copy to the public prefix, verify the copy. Doing it again changes nothing."""
    try:
        verify_files(store, dst_prefix, files, sizes)
        return
    except VerifyError:
        pass
    verify_files(store, src_prefix, files, sizes)
    for name in sorted(files):
        store.copy(f"{src_prefix}/{name}", f"{dst_prefix}/{name}")
    verify_files(store, dst_prefix, files, sizes)


def promote_lora(store: ObjectStore, artifact: LoRAArtifact) -> None:
    promote_files(store, artifact.prefix, layout.public_lora_prefix(artifact.lora_hash), artifact.files, artifact.sizes)


def promote_fold(store: ObjectStore, fold: FoldArtifact) -> None:
    """Publish a folded base and its fp32 master. The master goes to its own prefix, never beside the base."""
    promote_files(store, f"{fold.prefix}/base", layout.public_base_prefix(fold.base_hash), fold.base_files)
    master_prefix = layout.public_master_prefix(fold.base_fp32_master_digest)
    promote_files(store, f"{fold.prefix}/master", master_prefix, fold.master_files)


def publish_head(store: ObjectStore, manifest: HeadManifest) -> None:
    """Write the head manifest, then move the ``current`` pointer. The pointer moves last."""
    for ref in manifest.head.loras:
        verify_files(store, layout.public_lora_prefix(ref.lora_hash), manifest.lora_files[ref.lora_hash])
    key = layout.head_manifest(manifest.head.version_id)
    put_record(store, key, manifest)
    if get_record(store, key, HeadManifest) != manifest:
        raise VerifyError("the head manifest did not read back as written")
    store.put(layout.CURRENT, manifest.to_json())


def delete_pending(store: ObjectStore, prefix: str) -> int:
    """Delete the staged objects of one round or one fold: ``pending/<kind>/<id>`` or deeper.

    Any other prefix is refused. That includes ``pending/`` itself: no call can empty the zone.
    """
    parts = prefix.split("/")
    if parts[0] != layout.PENDING or len(parts) < 3 or not all(parts):
        raise ContractError(f"refusing to delete outside one round or fold under pending/: {prefix!r}")
    keys = store.list(prefix)
    for key in keys:
        store.delete(key)
    return len(keys)
