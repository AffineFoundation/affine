"""Append-only public task reserve. Standard library only in this package.

The CLI is intentionally the only writer. ``load_reserve`` verifies every
listed byte and row before anything may enter an active evaluation set.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator

from ...canonical import canonical_json, file_sha256, sha256_hex
from ...errors import ConfigError, VerifyError
from ...rotation import Reserve, ReserveShard
from ..math_pool import MathPool, POOL_V2_0_FILE, POOL_V2_0_SHA256, Task, load_pool

RESERVE_ENV = "SN120_RESERVE_DIR"
MANIFEST = "MANIFEST.jsonl"
WITHDRAWN = "withdrawn.jsonl"
INGEST_LOCK = ".ingest.lock"
PACKAGED_RESERVE = Path(__file__).parent
ID = re.compile(r"([a-z][a-z0-9_]*):([0-9]+)\Z")
DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
SHARD = re.compile(r"[0-9]{4}-[a-z0-9][a-z0-9-]*\.jsonl\Z")
MIN_PROBLEM_CHARS, MAX_PROBLEM_CHARS = 1, 16_384
MIN_ANSWER_CHARS, MAX_ANSWER_CHARS = 1, 2_048


def reserve_dir() -> Path:
    return Path(os.environ.get(RESERVE_ENV) or PACKAGED_RESERVE)


@contextlib.contextmanager
def ingest_lock(root: str | Path, *, exclusive: bool) -> Iterator[None]:
    """Serialize ingestion and reload/publication on the live reserve."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / INGEST_LOCK).open("a+b") as stream:
        os.set_inheritable(stream.fileno(), False)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_live_reserve(root: str | Path | None, seed: str | Path | None = None) -> Path:
    """Seed the selected live directory; never change the packaged checkout.

    Without SN120_RESERVE_DIR the original packaged directory is returned
    unchanged. A live directory may extend its packaged prefix; a new packaged
    suffix is copied forward when the live history is its strict prefix.
    """
    seed = PACKAGED_RESERVE if seed is None else Path(seed)
    root = reserve_dir() if root is None else Path(root)
    if root == seed or not os.environ.get(RESERVE_ENV) and seed == PACKAGED_RESERVE:
        return root
    root.mkdir(parents=True, exist_ok=True)
    with ingest_lock(root, exclusive=True):
        if not (root / MANIFEST).exists():
            (root / "shards").mkdir(exist_ok=True)
            for source in (seed / MANIFEST, seed / WITHDRAWN,
                           *(seed / "shards").glob("*.jsonl"),
                           *(path for name in ("retired.json", "retired_tasks.json") if (path := seed / name).exists())):
                if source.exists():
                    target = root / source.relative_to(seed)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open("rb") as src, target.open("xb") as dst:
                        dst.write(src.read())
        _, _, live, live_withdrawals = read_raw(root)
        _, _, packaged, packaged_withdrawals = read_raw(seed)
        if live[:-1][:len(packaged) - 1] == packaged[:-1]:
            # Live extends packaged. Future deploys cannot retract live rows.
            if live_withdrawals[:len(packaged_withdrawals)] != packaged_withdrawals:
                raise VerifyError("live reserve withdrawals diverge from the packaged reserve")
            return root
        if live[:-1] == packaged[:-1][:len(live) - 1] and len(live) < len(packaged):
            if live_withdrawals != packaged_withdrawals[:len(live_withdrawals)]:
                raise VerifyError("live reserve withdrawals diverge from the packaged extension")
            for row in packaged[len(live) - 1:-1]:
                name = row["shard"]
                source, target = seed / "shards" / name, root / "shards" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != source.read_bytes():
                    raise VerifyError(f"live reserve shard {name} differs from its packaged extension")
                if not target.exists():
                    with source.open("rb") as src, target.open("xb") as dst:
                        dst.write(src.read())
            atomic_write(root / MANIFEST, (seed / MANIFEST).read_bytes())
            atomic_write(root / WITHDRAWN, (seed / WITHDRAWN).read_bytes())
            return root
        raise VerifyError("live reserve manifest diverges from the packaged reserve; expected an append-only extension")


def extends_reserve(previous: Any, current: Any) -> None:
    """New shards append in position; withdrawals only grow."""
    if current.shards[:len(previous.shards)] != previous.shards:
        raise VerifyError("reserve reload is not an append-only extension of the loaded manifest")
    old = [(row["id"], row["reason"], row["added_at"]) for row in previous.withdrawn]
    new = [(row["id"], row["reason"], row["added_at"]) for row in current.withdrawn]
    if new[:len(old)] != old:
        raise VerifyError("reserve reload changes or removes an earlier withdrawal")


def normalise(problem: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", problem).casefold().split())


def _rows(raw: bytes, where: str) -> list[dict[str, Any]]:
    try:
        return [dict(__import__("json").loads(line)) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise VerifyError(f"{where}: invalid canonical JSONL: {exc}") from exc


def task_row(task: Task, **extra: Any) -> dict[str, Any]:
    return {"id": task.task_id, "env": task.slice, "problem": task.problem, "answer": task.answer, **extra}


class Catalog(MathPool):
    """Task lookup over the base, archive and reserve; never restrict ``task``.

    ``active`` optionally supplies a rotation-scoped admission lookup. Normal
    ``task`` remains catalog-wide so old and queued draws resolve unchanged.
    ``sha256`` stays the pinned base-pool hash, not a changing env pin.
    """
    def __init__(self, base: MathPool, extra: MathPool, active: MathPool | None = None, excluded: frozenset[str] = frozenset()) -> None:
        by_id = {task: base.task(task) for task in base.task_ids}
        by_id.update({task_id: extra.task(task_id) for task_id in extra.task_ids})
        super().__init__(tuple(by_id.values()), base.sha256)
        self._by_problem.update(extra._by_problem)
        self.active = active
        self.excluded = excluded

    def task_for_prompt(self, prompt: str) -> Task | None:
        if self.active is not None:
            task = self.active.task_for_prompt(prompt)
            if task is not None:
                return task
        task = super().task_for_prompt(prompt)
        return task if task is None or task.task_id not in self.excluded else None


def read_raw(root: str | Path) -> tuple[bytes, bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(root)
    try:
        manifest = (root / MANIFEST).read_bytes()
        withdrawn = (root / WITHDRAWN).read_bytes() if (root / WITHDRAWN).exists() else b""
    except OSError as exc:
        raise ConfigError(f"cannot read reserve {root}: {exc}") from exc
    shards = _rows(manifest, MANIFEST)
    withdrawals = _rows(withdrawn, WITHDRAWN) if withdrawn else []
    if not shards or "withdrawn" not in shards[-1]:
        raise VerifyError("reserve manifest lacks its final withdrawn pointer")
    return manifest, withdrawn, shards, withdrawals


def load_reserve(root: str | Path | None = None, *, with_catalog: bool = False, allow_retired: bool = False) -> tuple[Reserve, Catalog | None]:
    """Verify manifest, all shards, schema, uniqueness and canonical bytes."""
    try:
        return _load_reserve(root, with_catalog=with_catalog, allow_retired=allow_retired)
    except ConfigError:
        raise
    except (VerifyError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ConfigError(f"reserve verification failed: {type(exc).__name__}: {exc}") from exc


def _load_reserve(root: str | Path | None = None, *, with_catalog: bool = False, allow_retired: bool = False) -> tuple[Reserve, Catalog | None]:
    root = reserve_dir() if root is None else Path(root)
    manifest, withdrawn_raw, lines, withdrawn = read_raw(root)
    history, seen_shards = [], set()
    for row in lines[:-1]:
        if set(row) != {"shard", "sha256", "count", "added_at"} or not isinstance(row["shard"], str) or not SHARD.fullmatch(row["shard"]):
            raise VerifyError("invalid reserve shard manifest row")
        if row["shard"] in seen_shards:
            raise VerifyError("a reserve shard is listed twice")
        seen_shards.add(row["shard"])
        if not isinstance(row["count"], int) or row["count"] < 0 or not isinstance(row["added_at"], str) or not DATE.fullmatch(row["added_at"]):
            raise VerifyError("invalid reserve shard count or added_at")
        if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
            raise VerifyError("invalid reserve shard sha256")
        history.append(ReserveShard(row["shard"], row["sha256"], row["count"], row["added_at"]))
    pointer = lines[-1]
    if set(pointer) != {"withdrawn"} or not isinstance(pointer["withdrawn"], dict) or set(pointer["withdrawn"]) != {"sha256", "count"}:
        raise VerifyError("invalid reserve withdrawn pointer")
    if pointer["withdrawn"] != {"sha256": sha256_hex(withdrawn_raw), "count": len(withdrawn)}:
        raise VerifyError("reserve withdrawn pointer does not match withdrawn.jsonl")
    seen_withdrawn: set[str] = set()
    for row in withdrawn:
        if set(row) != {"id", "reason", "added_at"} or not ID.fullmatch(str(row.get("id", ""))) or row["id"] in seen_withdrawn:
            raise VerifyError("invalid or repeated withdrawal")
        if not isinstance(row["reason"], str) or not row["reason"].strip() or not DATE.fullmatch(str(row["added_at"])):
            raise VerifyError("invalid withdrawal reason or time")
        seen_withdrawn.add(row["id"])

    base = load_pool()
    archive = load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    # Archive and base overlap by design. The reserve itself must be unique.
    # Readmissions are loaded below after the retired high index is known.
    by_id = {task: base.task(task) for task in base.task_ids if task not in seen_withdrawn}
    for task_id in archive.task_ids:
        if task_id not in seen_withdrawn:
            by_id.setdefault(task_id, archive.task(task_id))
    normalised = {normalise(task.problem): task.task_id for task in by_id.values()}
    retired_path = root / "retired.json"
    retired_rows = []
    if retired_path.exists():
        retired_rows = json.loads(retired_path.read_bytes())
    high_retired = {row["id"] for row in retired_rows if row.get("reason") == "high"}
    low_retired = {row["id"] for row in retired_rows if row.get("reason") == "low"}
    retired_path = root / "retired_tasks.json"
    retired_tasks = {row["id"]: Task(row["id"], row["env"], row["problem"], row["answer"])
                     for row in json.loads(retired_path.read_bytes())} if retired_path.exists() else {}
    for task_id in high_retired & retired_tasks.keys() - seen_withdrawn:
        if not allow_retired:
            continue
        by_id[task_id] = retired_tasks[task_id]
        normalised[normalise(retired_tasks[task_id].problem)] = task_id
    # Withdrawn rows never enter any task lookup. The public retired-task
    # archive lets high-retired keys and explicitly readmitted keys coexist;
    # low-retired tasks stay excluded from admission forever.
    extra: dict[str, Task] = {}
    queue: list[str] = []
    if allow_retired:
        for task_id, task in retired_tasks.items():
            if task_id not in seen_withdrawn:
                by_id[task_id] = task
                normalised.setdefault(normalise(task.problem), task_id)
    retired_ids = {str(row.get("id", "")) for row in retired_rows}
    for shard in history:
        try:
            raw = (root / "shards" / shard.shard).read_bytes()
        except OSError as exc:
            raise VerifyError(f"reserve shard {shard.shard} cannot be read: {exc}") from exc
        if sha256_hex(raw) != shard.sha256:
            raise VerifyError(f"reserve shard {shard.shard} does not match its manifest sha256")
        rows = _rows(raw, shard.shard)
        if len(rows) != shard.count:
            raise VerifyError(f"reserve shard {shard.shard} has the wrong line count")
        ordered = []
        for row in rows:
            expected = {"id", "env", "problem", "answer", "source", "license", "added_at", "audit"} | ({"prior_p"} if "prior_p" in row else set())
            if set(row) != expected or not ID.fullmatch(str(row["id"])) or row["id"].split(":")[0] in ("math", "dapo"):
                raise VerifyError("reserve row has a bad schema or id")
            if any(not isinstance(row[key], str) for key in expected - {"audit", "prior_p"}) or not DATE.fullmatch(row["added_at"]):
                raise VerifyError("reserve row has a mistyped field or added_at")
            if "prior_p" in row and (type(row["prior_p"]) not in (int, float) or not math.isfinite(row["prior_p"]) or not 0.0 <= row["prior_p"] <= 1.0):
                raise VerifyError("reserve prior_p must be a finite number in [0, 1]")
            if not isinstance(row["audit"], dict) or row["audit"].get("verdict") != "keep" or not {"method", "ref"} <= set(row["audit"]):
                raise VerifyError("only an audited keep row may be in the reserve")
            if set(row["audit"]) - {"verdict", "method", "ref", "readmit_of"}:
                raise VerifyError("unknown reserve audit field")
            if not row["problem"].strip() or not row["answer"].strip():
                raise VerifyError(f"reserve task {row['id']} has an empty problem or answer")
            if row["id"] in extra:
                raise VerifyError(f"reserve task {row['id']} repeats a task id")
            if "readmit_of" in row["audit"]:
                original = retired_tasks.get(row["id"])
                if row["id"] not in high_retired or original is None or (row["problem"], row["answer"]) != (original.problem, original.answer):
                    raise VerifyError("an explicit readmit must exactly match its high-retired original")
            elif row["id"] in by_id or normalise(row["problem"]) in normalised:
                if allow_retired and row["id"] in retired_ids:
                    # Ranked rotation returns through the retired archive; its
                    # original shard row is historical evidence, not a new
                    # admission candidate. Keep the queue's complete history.
                    pass
                else:
                    raise VerifyError(f"reserve task {row['id']} repeats a task or problem")
            if canonical_json(row) + b"\n" != raw.splitlines(keepends=True)[len(ordered)] or not raw.endswith(b"\n"):
                raise VerifyError("reserve rows must be canonical JSONL, one newline-terminated row each")
            task = Task(row["id"], row["env"], row["problem"], row["answer"])
            if not (allow_retired and task.task_id in retired_ids):
                extra[task.task_id], normalised[task.problem and normalise(task.problem)] = task, task.task_id
                by_id[task.task_id] = task
            ordered.append(task.task_id)
        ordered.sort(key=lambda task_id: sha256_hex(f"{shard.sha256}|reserve-order|{task_id}".encode()))
        queue.extend(ordered)
    for task_id in seen_withdrawn:
        by_id.pop(task_id, None)
        extra.pop(task_id, None)
    combined = MathPool(tuple(by_id.values()), base.sha256)
    admitted = MathPool(tuple(task for task_id, task in by_id.items() if allow_retired or task_id not in low_retired), base.sha256)
    catalog = Catalog(base, combined, admitted, frozenset() if allow_retired else frozenset(low_retired)) if with_catalog else None
    reserve = Reserve(sha256_hex(manifest), tuple(task for task in queue if task not in seen_withdrawn),
                      tuple(history), withdrawn, sha256_hex(withdrawn_raw), len(withdrawn))
    return reserve, catalog


def candidate_metadata(root: str | Path | None, reserve: Reserve, catalog: Catalog | None = None) -> dict[str, dict[str, Any]]:
    """IDs and small public ranking fields, keyed by shard bytes not task text.

    This reads shards once at runtime construction, never once per round.
    Returning retired tasks keep their source metadata in the v2 archive.
    """
    root = reserve_dir() if root is None else Path(root)
    base = load_pool()
    archive = load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    meta: dict[str, dict[str, Any]] = {}
    for pool in (archive, base):
        for task_id in pool.task_ids:
            task = pool.task(task_id)
            meta[task_id] = {"source_prefix": task_id.split(":", 1)[0], "env": task.slice}
    for shard in reserve.shards:
        raw = (root / "shards" / shard.shard).read_bytes()
        if sha256_hex(raw) != shard.sha256:
            raise ConfigError(f"reserve shard {shard.shard} does not match its manifest sha256")
        for line in raw.splitlines():
            row = json.loads(line)
            meta[row["id"]] = {"source_prefix": row["id"].split(":", 1)[0], "env": row["env"]}
            if "prior_p" in row:
                meta[row["id"]]["prior_p"] = row["prior_p"]
    path = root / "retired_tasks.json"
    if path.exists():
        for row in json.loads(path.read_bytes()):
            meta[row["id"]] = {"source_prefix": row["id"].split(":", 1)[0], "env": row["env"]}
    if catalog is not None:
        for task_id in catalog.task_ids:
            task = catalog.task(task_id)
            meta.setdefault(task_id, {"source_prefix": task_id.split(":", 1)[0], "env": task.slice})
    return meta


    """Strict reconstruction from public mirrors, without local reserve I/O."""
    # Temporary files are confined to the caller-controlled object-store cache
    # in production; this helper is used for pure verification in tests/tools.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="sn120-reserve-verify-") as directory:
        root = Path(directory)
        (root / "shards").mkdir()
        (root / MANIFEST).write_bytes(manifest)
        (root / WITHDRAWN).write_bytes(withdrawn)
        for name, raw in shards.items():
            (root / "shards" / name).write_bytes(raw)
        return load_reserve(root)[0]


def empty_manifest() -> bytes:
    return canonical_json({"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}) + b"\n"
