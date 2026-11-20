"""Pinned adapter for the existing ``sn120-math-prior-v1`` release.

The source is the eight public parquet shards, not the separate production
reserve feed. Original 20-hex ``id`` values are returned verbatim. This module
never renames, allocates, deduplicates or ingests IDs into another pool.

The manifest's raw bytes are pinned by SHA-256. Every downloaded shard must
match its manifest SHA before parquet parsing. A local JSONL index caches only
published ID -> shard/row locations; task text stays in the release shards.
Reading rows needs the optional ``pyarrow`` dependency. Manifest-only
verification and submission membership checks remain standard-library code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol

from ..canonical import canonical_json, file_sha256, sha256_hex
from ..errors import ConfigError, ContractError, VerifyError
from .constants import (
    MAX_POOL_MANIFEST_BYTES,
    POOL_CACHE_ENV,
    POOL_COLUMNS,
    POOL_ID,
    POOL_MANIFEST_ENV,
    POOL_MANIFEST_SHA256,
    POOL_MANIFEST_URL,
    POOL_NAME,
    POOL_ROWS,
    POOL_VERSION,
)
from .protocol import valid_task_id

_SHARD = re.compile(r"prior-[0-9]{2}-of-08\.parquet")
_INDEX_VERSION = "affine-generator-pool-index-v1"
_DOWNLOAD_CHUNK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PoolShard:
    file: str
    rows: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PoolManifest:
    pool: str
    manifest_url: str
    manifest_sha256: str
    shards: tuple[PoolShard, ...]
    rows: int = POOL_ROWS
    columns: tuple[str, ...] = POOL_COLUMNS

    def to_dict(self) -> dict[str, Any]:
        """The small pointer document for ``public/generator/pool``."""
        return {
            "pool": self.pool,
            "manifest_url": self.manifest_url,
            "manifest_sha256": self.manifest_sha256,
            "rows": self.rows,
            "columns": list(self.columns),
            "shards": [{"file": shard.file, "rows": shard.rows, "sha256": shard.sha256} for shard in self.shards],
        }


@dataclass(frozen=True, slots=True)
class PoolLocation:
    shard: int
    row: int


class ParquetReader(Protocol):
    """Small injectable reader seam; tests need not install a parquet engine."""

    def __call__(self, path: Path, shard: PoolShard) -> Iterable[tuple[int, Mapping[str, Any]]]: ...


def _validate_location(location: str) -> str:
    if not isinstance(location, str) or not location:
        raise ConfigError(f"set {POOL_MANIFEST_ENV} to the published prior MANIFEST.json URL")
    parsed = urllib.parse.urlparse(location)
    if parsed.scheme not in ("https", "http", "file"):
        raise ConfigError("the prior manifest location must be an explicit https, http or file URL")
    if parsed.scheme == "file" and (not parsed.path or parsed.netloc not in ("", "localhost")):
        raise ConfigError("a local prior manifest URL must have the form file:///absolute/path")
    if parsed.scheme != "file" and not parsed.netloc:
        raise ConfigError("the prior manifest URL must include its host")
    return location


def manifest_url(location: str | None = None) -> str:
    """Explicit URL, defaulting to the verified public release (never guessed)."""
    return _validate_location(location if location is not None else os.environ.get(POOL_MANIFEST_ENV, POOL_MANIFEST_URL))


def _read_limited(stream: Any, maximum: int) -> bytes:
    raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise VerifyError(f"Generator pool manifest exceeds {maximum} bytes")
    return raw


def read_url(url: str) -> bytes:
    """Read only the small manifest, with a bounded body and no credential handling."""
    _validate_location(url)
    try:
        if urllib.parse.urlparse(url).scheme == "file":
            with Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).open("rb") as stream:
                return _read_limited(stream, MAX_POOL_MANIFEST_BYTES)
        request = urllib.request.Request(url, headers={"User-Agent": "affine-generator-pool/1"})
        with urllib.request.urlopen(request, timeout=60) as stream:  # noqa: S310 - explicit operator/public release URL
            return _read_limited(stream, MAX_POOL_MANIFEST_BYTES)
    except VerifyError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read prior manifest {url}: {type(exc).__name__}: {exc}") from exc


def parse_manifest(raw: bytes, *, location: str = POOL_MANIFEST_URL) -> PoolManifest:
    """Validate the exact release schema, size and raw manifest hash."""
    if not isinstance(raw, bytes) or sha256_hex(raw) != POOL_MANIFEST_SHA256:
        raise VerifyError(f"Generator pool manifest must have raw SHA-256 {POOL_MANIFEST_SHA256}")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise VerifyError(f"invalid prior manifest JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("name") != POOL_NAME or data.get("version") != POOL_VERSION:
        raise VerifyError("Generator pool must be the published sn120-math-prior v1")
    if data.get("rows") != POOL_ROWS or data.get("columns") != list(POOL_COLUMNS):
        raise VerifyError("Generator pool must have 811,609 rows and the exact published columns")
    rows = data.get("shards")
    if not isinstance(rows, list) or not rows:
        raise VerifyError("prior manifest must list its parquet shards")
    shards, files, total = [], set(), 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"file", "rows", "sha256"}:
            raise VerifyError("invalid Generator pool shard schema")
        file, count, digest = row["file"], row["rows"], row["sha256"]
        if not isinstance(file, str) or not _SHARD.fullmatch(file) or file in files:
            raise VerifyError("invalid or duplicate Generator pool shard filename")
        if type(count) is not int or count < 0:
            raise VerifyError("Generator pool shard rows must be nonnegative integers")
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise VerifyError("Generator pool shard needs a lowercase SHA-256")
        files.add(file)
        total += count
        shards.append(PoolShard(file, count, digest))
    if total != POOL_ROWS:
        raise VerifyError("Generator pool shard counts do not sum to 811,609")
    return PoolManifest(POOL_ID, manifest_url(location), POOL_MANIFEST_SHA256, tuple(shards))


def load_manifest(*, location: str | None = None) -> PoolManifest:
    url = manifest_url(location)
    return parse_manifest(read_url(url), location=url)


def _pyarrow_rows(path: Path, shard: PoolShard) -> Iterable[tuple[int, Mapping[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ConfigError("reading Generator pool shards needs pyarrow; install v2's 'generator' extra") from exc
    try:
        table = pq.ParquetFile(path)
        if table.schema_arrow.names != list(POOL_COLUMNS) or table.metadata.num_rows != shard.rows:
            raise VerifyError(f"Generator shard {shard.file} has an unexpected schema or row count")
        row = 0
        for batch in table.iter_batches(batch_size=4096, columns=list(POOL_COLUMNS)):
            for record in batch.to_pylist():
                yield row, record
                row += 1
        if row != shard.rows:
            raise VerifyError(f"Generator shard {shard.file} did not yield its manifest row count")
    except VerifyError:
        raise
    except Exception as exc:
        raise VerifyError(f"cannot parse Generator shard {shard.file}: {type(exc).__name__}: {exc}") from exc


def _verify_shard(path: Path, shard: PoolShard) -> None:
    try:
        actual = file_sha256(path)
    except OSError as exc:
        raise VerifyError(f"cannot read Generator pool shard {path}: {exc}") from exc
    if actual != shard.sha256:
        raise VerifyError(f"Generator shard {shard.file} SHA-256 {actual} does not match manifest")


def _download_shard(url: str, path: Path, expected_sha256: str) -> None:
    """Atomically cache a shard only after its complete raw-byte hash matches."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "affine-generator-pool/1"})
        with urllib.request.urlopen(request, timeout=60) as stream, tempfile.NamedTemporaryFile(  # noqa: S310
            dir=path.parent, prefix=".download-", delete=False
        ) as temporary:
            target = Path(temporary.name)
            digest = hashlib.sha256()
            while chunk := stream.read(_DOWNLOAD_CHUNK):
                digest.update(chunk)
                temporary.write(chunk)
            if digest.hexdigest() != expected_sha256:
                raise VerifyError(f"downloaded Generator shard {path.name} has the wrong SHA-256")
        target.replace(path)
    except Exception:
        if "target" in locals():
            target.unlink(missing_ok=True)
        raise


def _local_manifest(root: Path, manifest: PoolManifest) -> None:
    pointer = root / "MANIFEST.json"
    if pointer.is_file() and sha256_hex(pointer.read_bytes()) == POOL_MANIFEST_SHA256:
        return
    # Fetch the same pinned raw bytes rather than reconstructing the manifest.
    raw = read_url(manifest.manifest_url)
    parse_manifest(raw, location=manifest.manifest_url)
    temporary = pointer.with_suffix(".tmp")
    temporary.write_bytes(raw)
    temporary.replace(pointer)


def cache_directory(root: str | Path | None = None) -> Path:
    """Explicit cache directory or environment setting; no host path is invented."""
    chosen = root if root is not None else os.environ.get(POOL_CACHE_ENV)
    if not chosen:
        raise ConfigError(f"set {POOL_CACHE_ENV} or pass a cache directory for Generator pool shards")
    return Path(chosen)


def prepare_shards(
    root: str | Path,
    manifest: PoolManifest,
    *,
    fetch: bool = False,
) -> tuple[Path, ...]:
    """Verify local shards; optionally fetch only missing ones from the release."""
    directory = cache_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "shards").mkdir(exist_ok=True)
    _local_manifest(directory, manifest)
    base = manifest.manifest_url.rsplit("/", 1)[0]
    paths = []
    for shard in manifest.shards:
        path = directory / "shards" / shard.file
        if not path.is_file():
            if not fetch:
                raise VerifyError(f"Generator pool shard {shard.file} is missing from the explicit cache")
            _download_shard(f"{base}/{shard.file}", path, shard.sha256)
        _verify_shard(path, shard)
        paths.append(path)
    return tuple(paths)


class PriorPool:
    """Read-only ID registry over a verified copy of the published prior."""

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: PoolManifest | None = None,
        reader: ParquetReader | None = None,
        build_index: bool = False,
    ) -> None:
        self.root = cache_directory(root)
        self.manifest = manifest or load_manifest(location=(self.root / "MANIFEST.json").as_uri())
        if self.manifest.pool != POOL_ID:
            raise VerifyError("Generator registry cannot switch to another pool pin")
        self.paths = prepare_shards(self.root, self.manifest)
        self.reader = reader or _pyarrow_rows
        self.index_path = self.root / "id-index.jsonl"
        self._locations: dict[str, PoolLocation] = {}
        if self.index_path.is_file():
            try:
                self._load_index()
            except VerifyError as exc:
                if "does not cover all" in str(exc):
                    raise ConfigError("Generator pool ID index is incomplete; rebuild it explicitly with build_index=True") from exc
                raise
        elif build_index:
            self.build_index()
        else:
            raise ConfigError("Generator pool ID index is missing; build it explicitly with build_index=True before snapshot admission")
        if len(self._locations) != POOL_ROWS:
            raise ConfigError(f"Generator pool ID index is incomplete: {len(self._locations)} rows, expected {POOL_ROWS}; rebuild it explicitly")

    def _load_index(self) -> None:
        try:
            lines = self.index_path.read_bytes().splitlines()
            if not lines or json.loads(lines[-1]) != {
                "index_version": _INDEX_VERSION,
                "pool": POOL_ID,
                "rows": POOL_ROWS,
            }:
                raise VerifyError("Generator pool index lacks its exact pool pin and row-count trailer")
            locations: dict[str, PoolLocation] = {}
            for raw in lines[:-1]:
                row = json.loads(raw)
                if not isinstance(row, dict) or set(row) != {"id", "shard", "row"}:
                    raise VerifyError("invalid Generator pool index row")
                task_id = row["id"]
                if not valid_task_id(task_id) or task_id in locations:
                    raise VerifyError("invalid or duplicate published task ID in index")
                if type(row["shard"]) is not int or type(row["row"]) is not int:
                    raise VerifyError("pool index shard and row must be integers")
                if not 0 <= row["shard"] < len(self.paths) or not 0 <= row["row"] < self.manifest.shards[row["shard"]].rows:
                    raise VerifyError("pool index location is outside its published shard")
                if canonical_json(row) != raw:
                    raise VerifyError("pool index must be canonical JSONL")
                locations[task_id] = PoolLocation(row["shard"], row["row"])
            if len(locations) != POOL_ROWS:
                raise VerifyError("pool index does not cover all 811,609 published tasks")
            self._locations = locations
        except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
            raise VerifyError(f"cannot read Generator pool index: {exc}") from exc

    def build_index(self) -> None:
        """Build an ID/location cache without changing or storing a new dataset."""
        temporary = self.index_path.with_suffix(".tmp")
        count = 0
        try:
            with temporary.open("xb") as stream:
                for shard_number, (path, shard) in enumerate(zip(self.paths, self.manifest.shards)):
                    for row_number, record in self.reader(path, shard):
                        task_id = record.get("id")
                        if not valid_task_id(task_id) or task_id in self._locations:
                            raise VerifyError(f"invalid or duplicate published task ID {task_id!r}")
                        self._locations[task_id] = PoolLocation(shard_number, row_number)
                        line = {"id": task_id, "shard": shard_number, "row": row_number}
                        stream.write(canonical_json(line) + b"\n")
                        count += 1
                if count != POOL_ROWS:
                    raise VerifyError(f"published pool has {count} rows, expected {POOL_ROWS}")
                trailer = {"index_version": _INDEX_VERSION, "pool": POOL_ID, "rows": POOL_ROWS}
                stream.write(canonical_json(trailer) + b"\n")
            temporary.replace(self.index_path)
        except Exception:
            self._locations = {}
            temporary.unlink(missing_ok=True)
            raise

    def __contains__(self, task_id: object) -> bool:
        return isinstance(task_id, str) and task_id in self._locations

    def require_ids(self, task_ids: Iterable[str]) -> tuple[str, ...]:
        """Accept only exact known IDs, preserving input order and rejecting duplicates."""
        selected = tuple(task_ids)
        if not selected or len(set(selected)) != len(selected) or any(task_id not in self for task_id in selected):
            raise ContractError("Generator submissions need distinct existing published prior IDs, unchanged")
        return selected

    def iter_rows(self, shard_number: int) -> Iterator[Mapping[str, Any]]:
        for row_number, row in self.reader(self.paths[shard_number], self.manifest.shards[shard_number]):
            if not valid_task_id(row.get("id")):
                raise VerifyError(f"invalid task ID at Generator shard {shard_number}, row {row_number}")
            yield row

    def row(self, task_id: str) -> dict[str, Any]:
        """Return the original published row, looked up through the ID index."""
        if not self._locations:
            raise ConfigError("build or load the Generator ID index before task lookup")
        try:
            location = self._locations[task_id]
        except KeyError:
            raise ContractError(f"task ID {task_id!r} is not in the published prior") from None
        for row_number, row in self.reader(self.paths[location.shard], self.manifest.shards[location.shard]):
            if row_number == location.row:
                if row.get("id") != task_id:
                    raise VerifyError("Generator pool index does not resolve to the published ID")
                return dict(row)
        raise VerifyError(f"Generator pool row {task_id} is missing")

    def tasks(self, task_ids: Iterable[str]) -> tuple[Any, ...]:
        """Render the unchanged published rows through v2's production Task type."""
        from ..envs.math_pool import ENV_ID, Task

        selected = self.require_ids(task_ids)
        grouped: dict[int, dict[int, list[str]]] = {}
        for task_id in selected:
            location = self._locations[task_id]
            grouped.setdefault(location.shard, {}).setdefault(location.row, []).append(task_id)
        found: dict[str, dict[str, Any]] = {}
        for shard, locations in grouped.items():
            remaining = set(locations)
            for row_number, row in self.reader(self.paths[shard], self.manifest.shards[shard]):
                if row_number not in remaining:
                    continue
                for task_id in locations[row_number]:
                    if row.get("id") != task_id:
                        raise VerifyError("Generator pool index does not resolve to the published ID")
                    found[task_id] = dict(row)
                remaining.remove(row_number)
                if not remaining:
                    break
            if remaining:
                raise VerifyError("Generator pool index points beyond its published shard")
        result = []
        for task_id in selected:
            row = found[task_id]
            if not isinstance(row.get("problem"), str) or not isinstance(row.get("answer"), str):
                raise VerifyError(f"Generator prior row {task_id} has no valid problem and answer")
            result.append(Task(task_id=task_id, slice=ENV_ID, problem=row["problem"], answer=row["answer"]))
        return tuple(result)

    def messages(self, task_id: str) -> list[dict[str, str]]:
        """Exactly the production v2 math system/user prompt for this task."""
        return self.tasks((task_id,))[0].messages()

    def grade(self, task_id: str, output: str, capped: bool) -> Any:
        """Delegate directly to the pinned v2 math grader, without another rubric."""
        from ..envs.math_grader import grade

        return grade(self.tasks((task_id,))[0], output, capped)
