"""Shared control-plane I/O. No module here performs a chain write."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..canonical import canonical_json, is_sha256
from ..errors import ConfigError, ContractError, VerifyError
from ..links import store_token

UTC = dt.timezone.utc


def timestamp(value: float | int | dt.datetime) -> float:
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            raise ValueError("a timestamp needs a timezone")
        return value.timestamp()
    if type(value) not in (float, int):
        raise ValueError("a timestamp must be a number or datetime")
    return float(value)


def utc_text(value: float | int | dt.datetime) -> str:
    return dt.datetime.fromtimestamp(timestamp(value), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_bytes(path: str | Path, raw: bytes, *, private: bool = False) -> None:
    """Write a complete file in its destination directory, then rename it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    if private:
        os.fchmod(handle, 0o600)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: str | Path, value: Any, *, private: bool = False) -> None:
    atomic_bytes(path, json.dumps(value, sort_keys=True, indent=1, allow_nan=False).encode() + b"\n", private=private)


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=1, allow_nan=False).encode() + b"\n"


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise VerifyError(f"cannot read Generator JSON {path}: {type(exc).__name__}") from exc


def read_jsonl(raw: bytes | str, where: str = "JSONL") -> list[dict[str, Any]]:
    if isinstance(raw, str):
        raw = raw.encode()
    rows = []
    try:
        for number, line in enumerate(raw.splitlines(), 1):
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {number} is not an object")
            rows.append(row)
    except (UnicodeDecodeError, ValueError) as exc:
        raise VerifyError(f"invalid Generator {where}: {type(exc).__name__}: {exc}") from exc
    return rows


def append_once(path: str | Path, previous: bytes, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append only to the byte prefix previously read; retries add no duplicate."""
    addition = jsonl_bytes(rows)
    target = Path(path)
    try:
        current = target.read_bytes()
    except FileNotFoundError:
        current = b""
    if current == previous + addition:
        return
    if current != previous:
        raise VerifyError(f"refusing to overwrite a changed append-only file: {path}")
    atomic_bytes(target, previous + addition)


def open_store(credentials_file: str = ""):
    """Construct the existing R2 adapter; credentials stay in its memory."""
    try:
        from ..r2 import R2Store
    except ImportError as exc:
        raise ConfigError("Generator needs boto3: install v2's r2 extra") from exc
    environ = dict(os.environ)
    if credentials_file:
        environ.update(store_token(credentials_file))
    return R2Store.from_env(environ)


def object_metadata(store: Any, key: str) -> dict[str, Any]:
    """Store-authenticated metadata, without treating an ETag as a SHA-256."""
    if hasattr(store, "describe"):
        row = store.describe(key)
    elif hasattr(store, "_head"):
        row = store._head(key)
    elif hasattr(store, "_path"):
        stat = store._path(key).stat()
        row = {"ContentLength": stat.st_size, "LastModified": stat.st_mtime}
    else:
        raise ConfigError("the Generator snapshot store must expose describe(key) or R2 _head(key)")
    size = row.get("ContentLength", row.get("size"))
    modified = row.get("LastModified", row.get("last_modified"))
    if type(size) is not int or size < 0:
        raise VerifyError(f"Generator object {key} has no valid size")
    try:
        uploaded = timestamp(modified)
    except ValueError as exc:
        raise VerifyError(f"Generator object {key} has no valid LastModified") from exc
    return {"size": size, "uploaded_at": uploaded}


class PublicSink:
    """Atomic website files mirrored byte-for-byte to public/generator in R2."""

    def __init__(self, site_dir: str | Path, store: Any | None = None, public_url: str = "https://affine.io", *, mirror: bool = True):
        self.root = Path(site_dir)
        if not self.root.is_dir():
            raise ConfigError(f"the public website root {self.root} must already exist")
        self.store = store
        self.mirror = mirror
        self.public_url = public_url.rstrip("/")
        if not self.public_url.startswith(("https://", "http://")):
            raise ConfigError("the Generator public URL must be an explicit HTTP(S) URL")

    def path(self, relative: str) -> Path:
        if relative.startswith("/") or any(part in ("", ".", "..") for part in relative.split("/")):
            raise ContractError("unsafe Generator public path")
        return self.root / "generator" / relative

    def url(self, relative: str) -> str:
        return f"{self.public_url}/generator/{relative}"

    def put(self, relative: str, raw: bytes) -> None:
        if self.store is not None:
            self.put_mirror(relative, raw)
        else:
            atomic_bytes(self.path(relative), raw)

    def put_mirror(self, relative: str, raw: bytes) -> None:
        """Write local bytes first, then mirror. Callers may run mirrors concurrently."""
        atomic_bytes(self.path(relative), raw)
        if self.mirror and self.store is not None:
            self.store.put("public/generator/" + relative, raw)

    def json(self, relative: str, value: Any) -> None:
        self.put(relative, json_bytes(value))

    def exists(self, relative: str) -> bool:
        return self.path(relative).is_file()

    def read(self, relative: str) -> bytes:
        return self.path(relative).read_bytes()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(value: Any, where: str) -> str:
    if not is_sha256(value):
        raise VerifyError(f"{where} needs a lowercase SHA-256")
    return value
