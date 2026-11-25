"""Store adapter for deterministic pool rotation. Off mode builds none of it.

Only the active-set artifact id is added to State. The larger ledger and
retired archive live in content-addressed, public rotation artifacts. Each
published rotation carries every public input needed to replay it, including
the complete reserve queue and ledger at the previous round boundary.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import layout
from .canonical import canonical_json, sha256_hex
from .envs.math_pool import MathPool, Task, load_pool
from .envs.reserve import (
    Catalog, atomic_write, candidate_metadata, extends_reserve, ingest_lock, load_reserve, prepare_live_reserve, read_raw, reserve_dir,
)
from .errors import ConfigError, InfraError, VerifyError
from .irt import IRTSampler
from .params import Params
from .rotation import INFO_RULE_VERSION, Reserve, Rotation, fold_ledger, rotate, rotate_info
from .seams import ObjectStore
from .state import ABANDONED, NO_DECISION, State


def load_active(store: ObjectStore, sha: str) -> tuple[str, ...]:
    raw = store.get(layout.active_pool(sha))
    if sha256_hex(raw) != sha:
        raise VerifyError("active pool object does not match its content id")
    ids = json.loads(raw)
    if not isinstance(ids, list) or any(not isinstance(task, str) for task in ids) or ids != sorted(ids) or len(set(ids)) != len(ids):
        raise VerifyError("active pool object is not a sorted unique id list")
    return tuple(ids)


def load_rotation(store: ObjectStore, state_sha: str) -> Rotation:
    raw = store.get(layout.pool_rotation_state(state_sha))
    if sha256_hex(raw) != state_sha:
        raise VerifyError("rotation state object does not match its content id")
    rotation = Rotation.from_json(raw)
    # The first live v1 state (r000066) was addressed by its byte hash;
    # r000069 moved to the canonical digest. Both are strict, public, verified
    # forms. A v2 lineage must use the canonical form, and a mixed v1->v2
    # boundary is allowed to read the older state once.
    if rotation.digest() != state_sha:
        if rotation.rule_version != INFO_RULE_VERSION:
            return rotation
        raise VerifyError("rotation state canonical bytes do not match its id")
    return rotation


def mirror_reserve(store: ObjectStore, root: str | Path | None = None) -> None:
    """Mirror verified reserve bytes; shards go before the manifest pointer."""
    root = reserve_dir() if root is None else Path(root)
    manifest, withdrawn, rows, _ = read_raw(root)
    for row in rows[:-1]:
        sidecar = root / "shards" / (row["shard"] + ".rejected.json")
        if sidecar.exists():
            store.put(layout.reserve_shard(row["shard"] + ".rejected.json", sha256_hex(sidecar.read_bytes())), sidecar.read_bytes())
        key = layout.reserve_shard(row["shard"], row["sha256"])
        raw = (root / "shards" / row["shard"]).read_bytes()
        if sha256_hex(raw) != row["sha256"]:
            raise VerifyError(f"reserve shard {row['shard']} does not match its manifest sha256")
        if not store.exists(key) or store.get(key) != raw:
            store.put(key, raw)
        if store.get(key) != raw:
            raise VerifyError(f"published reserve shard {row['shard']} failed readback")
    if not store.exists(layout.RESERVE_WITHDRAWN) or store.get(layout.RESERVE_WITHDRAWN) != withdrawn:
        store.put(layout.RESERVE_WITHDRAWN, withdrawn)
    if store.get(layout.RESERVE_WITHDRAWN) != withdrawn:
        raise VerifyError("published reserve withdrawals failed readback")
    if not store.exists(layout.RESERVE_MANIFEST) or store.get(layout.RESERVE_MANIFEST) != manifest:
        store.put(layout.RESERVE_MANIFEST, manifest)
    if store.get(layout.RESERVE_MANIFEST) != manifest:
        raise VerifyError("published reserve manifest failed readback")


def append_only(previous: Rotation | None, current: Rotation) -> None:
    """Manifest shards must extend in position; withdrawn rows may only grow."""
    if previous is None:
        return
    old, new = previous.reserve_history, current.reserve_history
    before, after = old.get("shards") or [], new.get("shards") or []
    if after[:len(before)] != before:
        raise ConfigError("reserve manifest does not extend the previous rotated round: a shard changed, moved or disappeared")
    old_withdrawn, new_withdrawn = old.get("withdrawn") or [], new.get("withdrawn") or []
    if new_withdrawn[:len(old_withdrawn)] != old_withdrawn:
        raise ConfigError("reserve withdrawn.jsonl shrank or changed an earlier row")


def checked_put(store: ObjectStore, key: str, data: bytes, *, missing_only: bool = False) -> None:
    """Upload when needed, then require exact stored bytes before a pointer can name the object.

    Content-addressed active sets are never needlessly rewritten. An already-correct active file
    is only read; a missing one is uploaded from its public rotation decision. Other public pool
    artifacts remain deterministic and byte-idempotent. Every read-back failure fails closed.
    """
    try:
        stored = store.get(key)
    except KeyError:
        store.put(key, data)
    else:
        if missing_only and stored == data:
            return
        if not missing_only and stored != data:
            store.put(key, data)
    try:
        stored = store.get(key)
    except (KeyError, ValueError, InfraError) as exc:
        raise VerifyError(f"public pool object {key} failed readback") from exc
    if stored != data or sha256_hex(stored) != sha256_hex(data):
        raise VerifyError(f"public pool object {key} failed readback")


def _ranking_bytes(rotation: Rotation) -> bytes:
    return canonical_json({
        "schema": "affine-pool-ranking-v1", "round_id": rotation.round_id,
        "rule_version": rotation.rule_version, "active_sha": rotation.active_sha,
        "source_snapshot_id": rotation.source_snapshot_id, "reserve_manifest_sha": rotation.reserve_manifest_sha,
        "rule_params": rotation.rule_params, "candidates": list(rotation.candidate_ranking)})


def _summary_recent(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"round_id": row["round_id"], "opened_at": row.get("opened_at", 0),
            "active_size": len(row["active_ids"]), "admitted": len(row["admitted"]),
            "retired": len(row["retired"]), "retire_reason_counts": row["retire_reason_counts"],
            "rotation": layout.pool_rotation(row["round_id"])}


def _public_summary(store: ObjectStore, rotation: Rotation) -> bytes:
    summary = pool_summary(rotation, rotation.opened_at)
    history = [json.loads(store.get(key)) for key in store.list("public/pool/rotation/")]
    recent = [row for row in history if row.get("rule_version") == INFO_RULE_VERSION][-20:]
    summary["recent_rounds"] = [_summary_recent(row) for row in recent]
    return canonical_json(summary)


def _json_lines(raw: bytes) -> list[dict[str, Any]]:
    rows = json.loads("[" + b",".join(raw.splitlines()).decode("utf-8") + "]") if raw else []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise VerifyError("public reserve is not a JSON-lines object sequence")
    return rows


def _reserve_bytes(rotation: Rotation) -> tuple[bytes, bytes] | None:
    """Reconstruct public reserve bytes when the rotation retained the full manifest history."""
    history = rotation.reserve_history
    if "shards" not in history or "withdrawn" not in history:
        return None
    shards = history["shards"]
    withdrawn = history["withdrawn"]
    if not isinstance(shards, list) or not isinstance(withdrawn, list):
        raise VerifyError("rotation reserve history is malformed")
    manifest_rows = [{"shard": row["shard"], "sha256": row["sha256"], "count": row["count"], "added_at": row["added_at"]}
                     for row in shards]
    manifest_rows.append({"withdrawn": {"sha256": sha256_hex(b"".join(canonical_json(row) + b"\n" for row in withdrawn)),
                                        "count": len(withdrawn)}})
    manifest = b"".join(canonical_json(row) + b"\n" for row in manifest_rows)
    withdrawn_raw = b"".join(canonical_json(row) + b"\n" for row in withdrawn)
    if history.get("withdrawn_sha") not in (None, "", sha256_hex(withdrawn_raw)):
        raise VerifyError("rotation reserve withdrawals do not match their public hash")
    if history.get("withdrawn_count", len(withdrawn)) != len(withdrawn):
        raise VerifyError("rotation reserve withdrawals do not match their public count")
    if sha256_hex(manifest) != rotation.reserve_manifest_sha:
        raise VerifyError("rotation reserve history does not match its public manifest hash")
    return manifest, withdrawn_raw


def _reference(pool: Mapping[str, Any], name: str, expected: str) -> None:
    if pool.get(name, expected) != expected:
        raise VerifyError(f"mining snapshot names a nonstandard pool {name} object")


def verify_public_pool_snapshot(store: ObjectStore, snapshot: Mapping[str, Any]) -> None:
    """Resolve every public pool reference before ``mining.json`` may name it.

    Only public rotations and their public reserve inputs are used. Missing deterministic artifacts are
    restored and read back; correct content-addressed active bytes are never rewritten. Current global
    objects belong to the newest rotation, while older rotations require append-only prefixes. Missing
    source bytes that cannot be derived from public data fail closed rather than being invented.
    """
    records = [record for record in (snapshot.get("round"), *(snapshot.get("history") or ())) if record]
    referenced: list[tuple[Mapping[str, Any], Rotation]] = []
    try:
        for record in records:
            pool = record.get("pool")
            if not isinstance(pool, Mapping):
                continue
            active_sha = pool["active_sha"]
            if not isinstance(active_sha, str) or len(active_sha) != 64 or any(c not in "0123456789abcdef" for c in active_sha):
                raise VerifyError("snapshot active pool reference is not a lowercase sha256")
            rotation_key = layout.pool_rotation(record["round_id"])
            _reference(pool, "rotation", rotation_key)
            _reference(pool, "active", layout.active_pool(active_sha))
            _reference(pool, "reserve_manifest", layout.RESERVE_MANIFEST)
            _reference(pool, "retired_archive", layout.RETIRED_POOL)
            rotation = Rotation.from_json(store.get(rotation_key))
            if rotation.round_id != record["round_id"] or rotation.active_sha != active_sha:
                raise VerifyError("public pool rotation does not match its mining snapshot")
            if rotation.reserve_manifest_sha != pool["reserve_manifest_sha"]:
                raise VerifyError("public reserve reference does not match its mining snapshot")
            if pool.get("rule_version", rotation.rule_version) != rotation.rule_version:
                raise VerifyError("public pool rule version does not match its rotation")
            if rotation.rule_version == INFO_RULE_VERSION:
                _reference(pool, "ranking", layout.pool_ranking(rotation.round_id))
                _reference(pool, "summary", layout.POOL_SUMMARY)
            referenced.append((pool, rotation))
        if not referenced:
            return

        # Resolve public reserve dependencies before any active object can be repaired or referenced.
        source = next((rotation for _, rotation in reversed(referenced) if _reserve_bytes(rotation) is not None), None)
        if source is not None:
            rebuilt = _reserve_bytes(source)
            assert rebuilt is not None
            reserve_bytes, withdrawn_bytes = rebuilt
            for row in source.reserve_history["shards"]:
                shard = store.get(layout.reserve_shard(row["shard"], row["sha256"]))
                if sha256_hex(shard) != row["sha256"]:
                    raise VerifyError("public reserve shard does not match its content id")
            for key, data in ((layout.RESERVE_WITHDRAWN, withdrawn_bytes), (layout.RESERVE_MANIFEST, reserve_bytes)):
                if not store.exists(key):
                    checked_put(store, key, data)
        elif not store.exists(layout.RESERVE_MANIFEST):
            raise VerifyError("public reserve manifest is missing and cannot be reconstructed")
        reserve = store.get(layout.RESERVE_MANIFEST)
        reserve_rows = _json_lines(reserve)
        if not reserve_rows or "withdrawn" not in reserve_rows[-1]:
            raise VerifyError("public reserve manifest lacks its withdrawn pointer")
        withdrawn = store.get(layout.RESERVE_WITHDRAWN)
        withdrawn_rows = _json_lines(withdrawn)
        pointer = reserve_rows[-1]["withdrawn"]
        if pointer != {"sha256": sha256_hex(withdrawn), "count": len(withdrawn_rows)}:
            raise VerifyError("public reserve withdrawals do not match their manifest")

        # Public history is chronological; opened_at breaks any nonstandard ordering without using local time.
        latest = max(enumerate(referenced), key=lambda item: (item[1][1].opened_at, item[0]))[1][1]
        for _, rotation in referenced:
            rebuilt = _reserve_bytes(rotation)
            if rebuilt is None:
                if rotation.round_id == latest.round_id and sha256_hex(reserve) != rotation.reserve_manifest_sha:
                    raise VerifyError("current public reserve does not match its mining snapshot")
            else:
                old_manifest, old_withdrawn = rebuilt
                old_rows, old_withdrawn_rows = _json_lines(old_manifest), _json_lines(old_withdrawn)
                if reserve_rows[:len(old_rows)] != old_rows or withdrawn_rows[:len(old_withdrawn_rows)] != old_withdrawn_rows:
                    raise VerifyError("public reserve does not extend a referenced rotation")
            checked_put(store, layout.active_pool(rotation.active_sha), canonical_json(list(rotation.active_ids)), missing_only=True)
            if load_active(store, rotation.active_sha) != rotation.active_ids:
                raise VerifyError("snapshot active pool differs from its public rotation")
            if rotation.rule_version == INFO_RULE_VERSION:
                checked_put(store, layout.pool_ranking(rotation.round_id), _ranking_bytes(rotation))
        if store.exists(layout.RETIRED_POOL):
            current_retired = json.loads(store.get(layout.RETIRED_POOL))
            if not isinstance(current_retired, list):
                raise VerifyError("public retired pool is not a list")
            for _, rotation in referenced:
                old = list(rotation.retired_archive)
                if current_retired[:len(old)] != old:
                    raise VerifyError("public retired pool does not extend a referenced rotation")
        else:
            checked_put(store, layout.RETIRED_POOL, canonical_json(list(latest.retired_archive)))
        if latest.rule_version == INFO_RULE_VERSION:
            checked_put(store, layout.POOL_SUMMARY, _public_summary(store, latest))
    except VerifyError:
        raise
    except (KeyError, ValueError, TypeError, InfraError) as exc:
        raise VerifyError("a public pool reference failed readback") from exc


def active_file(store: ObjectStore, rotation: Rotation, catalog: MathPool | None = None, *, decision_record: Rotation | None = None) -> None:
    """Verify every referenced public pool object before publishing a rotation or summary.

    Callers mirror and read back the reserve before this function. No reference is allowed to
    repair or outrun a missing dependency; a failed active read-back prevents all pointer writes.
    """
    opened_at = rotation.opened_at
    active = layout.active_pool(rotation.active_sha)
    active_bytes = canonical_json(list(rotation.active_ids))
    checked_put(store, active, active_bytes, missing_only=True)
    try:
        published = load_active(store, rotation.active_sha)
    except (KeyError, ValueError, InfraError, VerifyError) as exc:
        raise VerifyError("published active pool failed readback") from exc
    if published != rotation.active_ids:
        raise VerifyError("published active pool ids differ from the rotation decision")
    if rotation.initial_active_sha:
        # The production open path uploads this before the decision. It is public and content-addressed.
        try:
            load_active(store, rotation.initial_active_sha)
        except (KeyError, ValueError, InfraError, VerifyError) as exc:
            raise VerifyError("initial active pool artifact is missing or failed readback") from exc
    if rotation.rule_version == INFO_RULE_VERSION:
        checked_put(store, layout.pool_ranking(rotation.round_id), _ranking_bytes(rotation))
    checked_put(store, layout.pool_rotation(rotation.round_id), (decision_record or rotation).to_json())
    checked_put(store, layout.pool_rotation_state(rotation.digest()), rotation.to_json())
    if store.exists(layout.RETIRED_POOL):
        previous = json.loads(store.get(layout.RETIRED_POOL))
        current = list(rotation.retired_archive)
        if current[:len(previous)] != previous:
            raise VerifyError("retired archive would change an earlier row")
    checked_put(store, layout.RETIRED_POOL, canonical_json(list(rotation.retired_archive)))
    if catalog is not None:
        rows = []
        for row in rotation.retired_archive:
            try:
                task = catalog.task(row["id"])
            except Exception:
                continue
            rows.append({"id": task.task_id, "env": task.slice, "problem": task.problem, "answer": task.answer})
        checked_put(store, layout.RETIRED_TASKS, canonical_json(rows))
    if rotation.rule_version == INFO_RULE_VERSION:
        checked_put(store, layout.POOL_SUMMARY, canonical_json(pool_summary(rotation, opened_at)))
        if decision_record is None:
            history = [json.loads(store.get(key)) for key in store.list("public/pool/rotation/")]
            recent = [row for row in history if row.get("rule_version") == INFO_RULE_VERSION][-20:]
            summary = pool_summary(rotation, opened_at)
            summary["recent_rounds"] = [_summary_recent(row) for row in recent]
            checked_put(store, layout.POOL_SUMMARY, canonical_json(summary))


def unscored_rotation(previous: Rotation | None, snapshot: Mapping[str, Any], record: Any = None) -> bool:
    """An unscored rotation keeps measured rows and treats new admits as fresh.

    Its previous snapshot must be the source of that rotation. Live open gets
    the close outcome from replayed State; public replay gets its stored report.
    """
    if previous is None or not previous.source_snapshot_id or snapshot.get("snapshot_id") != previous.source_snapshot_id:
        return False
    if record is None:
        raise VerifyError("unscored rotation replay needs its round close record")
    outcome = record.get("outcome") if isinstance(record, Mapping) else getattr(record, "outcome", None)
    round_id = record.get("round_id") if isinstance(record, Mapping) else getattr(record, "round_id", None)
    return round_id == previous.round_id and outcome in (NO_DECISION, ABANDONED)


def pool_summary(rotation: Rotation, opened_at: int = 0) -> dict[str, Any]:
    """Small public aggregate of the full ranking, for the static pool page."""
    if rotation.rule_version != INFO_RULE_VERSION:
        raise ConfigError("a ranked-pool summary requires rotation_rule=info")
    by_state: dict[str, dict[str, Any]] = {}
    active = set(rotation.active_ids)
    remaining = {row["id"] for row in rotation.candidate_ranking if row["state"] == "reserve" and not row["admitted"] and not row["retired"]}
    for state, ids in (("active", active), ("reserve", remaining)):
        rows = [row for row in rotation.candidate_ranking if row["id"] in ids]
        by_source: dict[str, int] = {}
        by_env: dict[str, int] = {}
        histogram = [{"min": index / 20, "max": (index + 1) / 20, "count": 0} for index in range(20)]
        for row in rows:
            by_source[row["source_prefix"]] = by_source.get(row["source_prefix"], 0) + 1
            by_env[row["env"]] = by_env.get(row["env"], 0) + 1
            histogram[min(19, int(row["score"] * 20))]["count"] += 1
        by_state[state] = {"size": len(rows), "by_source_prefix": dict(sorted(by_source.items())),
                           "by_env": dict(sorted(by_env.items())), "score_histogram": histogram}
    retired_rows = [row for row in rotation.candidate_ranking if row["state"] == "retired" and not row["admitted"]]
    return {"schema": "affine-pool-summary-v1", "round_id": rotation.round_id, "opened_at": opened_at,
            "rule_version": rotation.rule_version, "active_sha": rotation.active_sha, "target": len(rotation.active_ids),
            "reserve_size": rotation.queue_remaining, "retired_size": len(rotation.retired_archive),
            "retired_candidates": len(retired_rows), "active_size": len(rotation.active_ids), "admitted": len(rotation.admitted),
            "retired": len(rotation.retired), "retire_reason_counts": rotation.retire_reason_counts,
            "active": by_state["active"], "reserve": by_state["reserve"],
            "next_up": list(rotation.next_up), "ranking": layout.pool_ranking(rotation.round_id),
            "rotation": layout.pool_rotation(rotation.round_id)}


class PoolRuntime:
    """A MathPool-compatible catalog plus the reserve rotation lifecycle."""
    def __init__(self, store: ObjectStore, params: Params, state: State, *, root: str | Path | None = None,
                 base: MathPool | None = None, initial_ids: Sequence[str] | None = None,
                 seed: str | Path | None = None) -> None:
        self.store, self.params, self.state = store, params, state
        self.base = base or load_pool()
        self.initial_ids = tuple(sorted(initial_ids or self.base.task_ids))
        self.root = Path(root) if root is not None else reserve_dir()
        self.seed = Path(seed) if seed is not None else None
        prepare_live_reserve(self.root, self.seed)
        self.previous: Rotation | None = None
        if state.rotation_state_sha:
            self.previous = load_rotation(store, state.rotation_state_sha)
        self.admitted_counts: dict[str, int] = {}
        current = self.previous
        while current is not None:
            for task_id in current.admitted:
                self.admitted_counts[task_id] = self.admitted_counts.get(task_id, 0) + 1
            current = load_rotation(self.store, current.prev_state_sha) if current.prev_state_sha else None
        with ingest_lock(self.root, exclusive=False):
            self._restore_public_cache()
            self.reserve, self.catalog = load_reserve(self.root, with_catalog=True, allow_retired=self.params.rotation_rule == "info")
            self.candidate_meta = candidate_metadata(self.root, self.reserve, self.catalog) if self.params.rotation_rule == "info" else {}
        self.active_rotation: Rotation | None = None
        if state.round is not None and state.round.rotation is not None:
            current = load_rotation(store, state.round.rotation["state_sha"])
            expected = {key: value for key, value in state.round.rotation.items() if key not in ("state_sha", "size")}
            if current.round_id != state.round.round_id or current.journal_block() != expected:
                raise VerifyError("open round rotation journal block does not match its stored decision")
            if load_active(store, current.active_sha) != current.active_ids:
                raise VerifyError("open round active ids differ from the active pool artifact")
            self.active_rotation = current
        self._active_ids = self.active_rotation.active_ids if self.active_rotation else (
            self.previous.active_ids if self.previous else self.initial_ids
        )
        self.admission_pool = self
        self.task_ids, self.sha256 = self.catalog.task_ids, self.catalog.sha256
        self._loaded_manifest = (self.root / "MANIFEST.jsonl").read_bytes()

    @property
    def active_ids(self) -> tuple[str, ...]:
        return self._active_ids

    def _restore_public_cache(self, root: str | Path | None = None) -> None:
        """Restore consumed reserve and retired rows from durable mirrors.

        A fresh checkout has only the empty packaged reserve. A newer local
        ingest may extend the last public mirror, but it may not replace it.
        """
        root = self.root if root is None else Path(root)
        local = root / "shards"
        if not local.is_dir() or not (root / "MANIFEST.jsonl").is_file():
            return
        if self.store.exists(layout.RETIRED_POOL):
            index = json.loads(self.store.get(layout.RETIRED_POOL))
            tasks = json.loads(self.store.get(layout.RETIRED_TASKS))
            if not isinstance(index, list) or not isinstance(tasks, list):
                raise VerifyError("public retired archive is not a list")
            for name, raw in (("retired.json", canonical_json(index)), ("retired_tasks.json", canonical_json(tasks))):
                path = root / name
                if not path.exists() or path.read_bytes() != raw:
                    atomic_write(path, raw)
        if not self.store.exists(layout.RESERVE_MANIFEST):
            return
        manifest = self.store.get(layout.RESERVE_MANIFEST)
        withdrawn = self.store.get(layout.RESERVE_WITHDRAWN)
        local_manifest = (root / "MANIFEST.jsonl").read_bytes()
        local_withdrawn = (root / "withdrawn.jsonl").read_bytes()
        remote = [json.loads(line) for line in manifest.splitlines()]
        here = [json.loads(line) for line in local_manifest.splitlines()]
        cache_miss = len(here) == 1 and set(here[0]) == {"withdrawn"}
        restore = cache_miss
        remote_withdrawn = [json.loads(line) for line in withdrawn.splitlines()]
        local_withdrawals = [json.loads(line) for line in local_withdrawn.splitlines()]
        if not cache_miss:
            local_history, remote_history = here[:-1], remote[:-1]
            equal = local_history == remote_history
            extends = len(local_history) > len(remote_history) and local_history[:len(remote_history)] == remote_history
            older = len(local_history) < len(remote_history) and local_history == remote_history[:len(local_history)]
            if not (equal or extends or older):
                raise VerifyError("local reserve does not extend its public mirror")
            if (equal or extends) and local_withdrawals[:len(remote_withdrawn)] != remote_withdrawn:
                # Packaged extensions, including withdrawals, stay local; a
                # shorter or changed withdrawal history is never overwritten.
                raise VerifyError("local reserve withdrawals do not extend its public mirror")
            if older:
                # An older checkout may be a strict prefix of the mirror. Only
                # restore it when its withdrawals are also a remote prefix.
                if local_withdrawals != remote_withdrawn[:len(local_withdrawals)]:
                    raise VerifyError("local reserve withdrawals do not extend its public mirror")
                restore = True
        if restore and manifest != local_manifest:
            (root / "MANIFEST.jsonl").write_bytes(manifest)
            (root / "withdrawn.jsonl").write_bytes(withdrawn)
        prefix = "public/pool/reserve/shards/"
        mirrored: dict[str, bytes] = {}
        for key in self.store.list(prefix):
            # <name>.jsonl.<sha256>.jsonl, or the optional
            # <name>.jsonl.rejected.json.<sha256>.jsonl.
            name = key[len(prefix):].rsplit(".", 2)[0]
            mirrored[name] = self.store.get(key)
        for name, raw in mirrored.items():
            target = local / name
            if target.exists() and target.read_bytes() != raw:
                raise VerifyError(f"local reserve shard {name} differs from its public mirror")
        for name, raw in mirrored.items():
            target = local / name
            if not target.exists() or target.read_bytes() != raw:
                atomic_write(target, raw)

    def _reload_at_open(self, *, locked: bool = False) -> None:
        root = self.root
        assert root is not None
        manifest_path = root / "MANIFEST.jsonl"
        if not manifest_path.is_file() or (manifest_path.exists() and manifest_path.read_bytes() == self._loaded_manifest):
            return
        with contextlib.nullcontext() if locked else ingest_lock(root, exclusive=False):
            if manifest_path.read_bytes() == self._loaded_manifest:
                return
            candidate = root / f".reload-candidate-{os.getpid()}"
            if candidate.exists():
                shutil.rmtree(candidate)
            shutil.copytree(root, candidate, ignore=shutil.ignore_patterns(".ingest.lock", ".reload-candidate-*", ".generator-stage-*"))
            try:
                self._restore_public_cache(candidate)
                reserve, catalog = load_reserve(candidate, with_catalog=True, allow_retired=self.params.rotation_rule == "info")
                extends_reserve(self.reserve, reserve)
                if reserve.manifest_sha == self.reserve.manifest_sha:
                    raise VerifyError("reserve manifest changed without a new shard or withdrawal")
                # rotate_info itself requires the effective queue to be disjoint
                # from active and retired; raw shards intentionally retain history.
                meta = candidate_metadata(candidate, reserve, catalog) if self.params.rotation_rule == "info" else {}
                for source in candidate.rglob("*"):
                    if not source.is_file() or source.name == ".ingest.lock":
                        continue
                    target = root / source.relative_to(candidate)
                    raw = source.read_bytes()
                    if not target.exists() or target.read_bytes() != raw:
                        atomic_write(target, raw)
                self.reserve, self.catalog, self.candidate_meta = reserve, catalog, meta
                self._loaded_manifest = manifest_path.read_bytes()
                self.task_ids = catalog.task_ids
                print(f"reserve reloaded: {len(reserve.shards)} shards, {len(reserve.queue)} queued, {reserve.manifest_sha}", file=sys.stderr, flush=True)
            except (ConfigError, VerifyError, OSError, ValueError) as exc:
                print(f"reserve reload refused; keeping loaded {self.reserve.manifest_sha}: {exc}", file=sys.stderr, flush=True)
            finally:
                shutil.rmtree(candidate, ignore_errors=True)

    @contextlib.contextmanager
    def _publication(self):
        """One exclusive lock from reload through decision and publication."""
        root = self.root
        assert root is not None
        with ingest_lock(root, exclusive=True):
            try:
                self._reload_at_open(locked=True)
            except (ConfigError, VerifyError, OSError, ValueError) as exc:
                print(f"reserve reload/restore failed; keeping loaded {self.reserve.manifest_sha}: {exc}", file=sys.stderr, flush=True)
            yield

    def task(self, task_id: str) -> Task:
        return self.catalog.task(task_id)

    def task_for_prompt(self, prompt: str) -> Task | None:
        return self.catalog.task_for_prompt(prompt)

    def _rule_reserve(self) -> Reserve:
        """Consume queue entries by historical admission count, including readmits."""
        remaining = dict(self.admitted_counts)
        history = self.reserve.history()
        queue = []
        for task in self.reserve.queue:
            if remaining.get(task, 0):
                remaining[task] -= 1
            else:
                queue.append(task)
        history["queue"] = queue
        result = Reserve.from_dict(history)
        if self.params.rotation_rule != "info":
            return result
        # A ranked active set consumes its original queue entry; a later return
        # comes from the retired archive, never by consuming a second shard row.
        retired_ids = {str(row.get("id")) for row in self.previous.retired_archive} if self.previous else set()
        history["queue"] = [task for task in queue if task not in retired_ids]
        history["candidate_meta"] = self.candidate_meta
        return Reserve.from_dict(history)

    def _retired_inputs(self, previous: Rotation | None) -> Sequence[Mapping[str, Any]]:
        # The committed public state, not a cache left by a crashed open. A
        # v1->v2 transition carries the complete archive of the previous state.
        return previous.retired_archive if previous else ()

    def open_and_publish(self, round_id: str, snapshot: Mapping[str, Any], state: State | None = None) -> tuple[Rotation, IRTSampler, dict[str, Any]]:
        """Controller boundary: one lock for reload, decision and publication."""
        with self._publication():
            decision, sampler = self.open(round_id, snapshot, state, reload_reserve=False)
            return decision, sampler, self.publish_open(decision)

    def open(self, round_id: str, snapshot: Mapping[str, Any], state: State | None = None, *, reload_reserve: bool = True) -> tuple[Rotation, IRTSampler]:
        # startup State by replay. Always take the controller's current state.
        state = self.state if state is None else state
        if self.active_rotation is not None:
            raise ConfigError("cannot rotate while a rotated round is open")
        if snapshot.get("snapshot_id") and state.snapshot_id and snapshot.get("snapshot_id") != state.snapshot_id:
            raise VerifyError("rotation source snapshot is not the last closed round's snapshot")
        previous = self.previous
        a_prev = previous.active_ids if previous else self.initial_ids
        unscored = unscored_rotation(previous, snapshot, state.rounds[-1] if state.rounds else None)
        retired_archive = self._retired_inputs(previous)
        if self.params.rotation_rule == "info":
            previous_ledger = dict(previous.ledger if previous else {})
            previous_ledger.update(previous.retired_ledger if previous else {})
            rotate_rule = rotate_info
        else:
            previous_ledger = previous.ledger if previous else {}
            rotate_rule = rotate
        if reload_reserve:
            self._reload_at_open()
        reserve = self._rule_reserve()
        decision = rotate_rule(round_id, a_prev, snapshot, previous_ledger, reserve, self.params,
                               initial_size=len(self.initial_ids), retired_archive=retired_archive,
                               source_snapshot_id=snapshot.get("snapshot_id", ""),
                               prev_state_sha=state.rotation_state_sha if previous and previous.rule_version != INFO_RULE_VERSION else previous.digest() if previous else "",
                               **({"allow_unscored_previous": unscored} if self.params.rotation_rule == "info" else {}),
                               **({"initial_candidate_meta": self.candidate_meta, "initial_retired_ledger": previous.retired_ledger}
                                  if self.params.rotation_rule == "info" and previous and previous.rule_version != INFO_RULE_VERSION else {}))
        if self.params.rotation_rule == "info":
            archived_ledger = {task: row for task, row in (previous.retired_ledger if previous else {}).items()
                               if task not in decision.admitted}
            archived_ledger.update({row.id: (row.head_rollouts, row.head_passes) for row in decision.retired})
            decision = Rotation.from_dict(decision.to_dict() | {
                "retired_ledger": {task: tuple(row) for task, row in sorted(archived_ledger.items())},
                **({"initial_retired_archive": list(retired_archive), "initial_candidate_meta": self.candidate_meta,
                    "initial_retired_ledger": previous.retired_ledger if previous else {}}
                   if previous is None or previous.rule_version != INFO_RULE_VERSION else {}),
            })
        if not previous:
            checked_put(self.store, layout.active_pool(decision.prev_active_sha), canonical_json(list(a_prev)), missing_only=True)
        append_only(previous, decision)
        meta = {"active_sha": decision.active_sha, "reserve_manifest_sha": decision.reserve_manifest_sha, "round_id": round_id}
        ability = float(snapshot.get("head_ability", snapshot["heads"].get(snapshot["head_version_id"], {}).get("ability", 0.0)))
        initial = {task: ability for task in decision.admitted if task not in decision.restored_irt_rows}
        initial.update(decision.restored_irt_rows)
        sampler = IRTSampler(decision.active_ids, self.params, snapshot=snapshot, initial=initial, rotation=meta)
        return decision, sampler

    def publish_open(self, decision: Rotation, *, opened_at: int = 0) -> dict[str, Any]:
        """Artifacts strictly before ROUND_OPENED; repeat is byte-idempotent."""
        if decision.rule_version == INFO_RULE_VERSION:
            decision = Rotation.from_dict(decision.to_dict() | {"opened_at": opened_at})
        root = Path(self.root or reserve_dir())
        if decision.reserve_manifest_sha != sha256_hex((root / "MANIFEST.jsonl").read_bytes()):
            raise VerifyError("reserve changed between rotation decision and publication")
        mirror_reserve(self.store, self.root)
        active_file(self.store, decision, self.catalog)
        self._publish_retired_local(decision)
        block = decision.journal_block()
        block["state_sha"], block["size"] = decision.digest(), len(decision.active_ids)
        return block

    def _publish_retired_local(self, decision: Rotation) -> None:
        root = Path(self.root or reserve_dir())
        rows, tasks = [], []
        for row in decision.retired_archive:
            try:
                task = self.catalog.task(row["id"])
            except Exception:
                continue
            tasks.append({"id": task.task_id, "env": task.slice, "problem": task.problem, "answer": task.answer})
        (root / "retired.json").write_bytes(canonical_json(list(decision.retired_archive)))
        (root / "retired_tasks.json").write_bytes(canonical_json(tasks))
        checked_put(self.store, layout.RETIRED_TASKS, canonical_json(tasks))

    def commit(self, decision: Rotation) -> None:
        """Advance the in-process boundary only after the close is computed."""
        self.previous = decision
        self.active_rotation = None
        self._active_ids = decision.active_ids
        for task_id in decision.admitted:
            self.admitted_counts[task_id] = self.admitted_counts.get(task_id, 0) + 1

    def close(self, decision: Rotation, draw_ids: Sequence[str], null_outcomes: Sequence[str], *, publish: bool = True) -> Rotation:
        counts = fold_ledger(decision.ledger, decision.active_ids, draw_ids, null_outcomes)
        updated = Rotation.from_dict(decision.to_dict() | {"ledger": counts})
        if publish:
            active_file(self.store, updated, self.catalog, decision_record=decision)
        self.commit(updated)
        return updated


def replay_rotation(store: ObjectStore, round_id: str, snapshot: Mapping[str, Any] | None = None,
                    *, initial_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Recompute from the round artifact and its named public IRT snapshot."""
    expected = Rotation.from_json(store.get(layout.pool_rotation(round_id)))
    previous = load_rotation(store, expected.prev_state_sha) if expected.prev_state_sha else None
    history = expected.reserve_history
    reserve = Reserve.from_dict(history)
    source = snapshot or (json.loads(store.get(layout.irt_snapshot(expected.source_snapshot_id))) if expected.source_snapshot_id else None)
    if source is None:
        raise VerifyError("rotation replay needs the source IRT snapshot")
    rotate_rule = rotate_info if expected.rule_version == INFO_RULE_VERSION else rotate
    params = Params.from_dict(expected.rule_params)
    previous_ledger = dict(previous.ledger if previous else {})
    archive = previous.retired_archive if previous else expected.initial_retired_archive
    if expected.rule_version == INFO_RULE_VERSION:
        previous_ledger.update(previous.retired_ledger if previous else {})
        previous_ledger.update(expected.initial_retired_ledger)
        if not reserve.candidate_meta and expected.initial_candidate_meta:
            reserve = Reserve.from_dict(history | {"candidate_meta": expected.initial_candidate_meta})
    unscored = False
    if expected.rule_version == INFO_RULE_VERSION and previous and set(source["tasks"]) != set(previous.active_ids):
        unscored = unscored_rotation(previous, source, json.loads(store.get(layout.round_file(previous.round_id))))
    actual = rotate_rule(round_id, previous.active_ids if previous else load_active(store, expected.initial_active_sha),
                         source, previous_ledger, reserve, params,
                         initial_size=len(load_active(store, expected.initial_active_sha)) if previous is None else len(previous.active_ids),
                         retired_archive=archive, source_snapshot_id=expected.source_snapshot_id, prev_state_sha=expected.prev_state_sha,
                         **({"allow_unscored_previous": unscored} if expected.rule_version == INFO_RULE_VERSION else {}),
                         **({"initial_candidate_meta": expected.initial_candidate_meta or None,
                             "initial_retired_ledger": expected.initial_retired_ledger or None}
                            if expected.rule_version == INFO_RULE_VERSION else {}))
    if params.rotation_rule == "info":
        retired_ledger = {task: row for task, row in (previous.retired_ledger if previous else {}).items()
                          if task not in actual.admitted}
        retired_ledger.update({row.id: (row.head_rollouts, row.head_passes) for row in actual.retired})
        actual = Rotation.from_dict(actual.to_dict() | {"retired_ledger": retired_ledger})
    if params.rotation_rule == "info":
        ranking = json.loads(store.get(layout.pool_ranking(round_id)))
        if ranking["candidates"] != list(actual.candidate_ranking) or ranking["round_id"] != round_id:
            raise VerifyError("public v2 ranking does not match the recomputed decision")
    return {"round_id": round_id, "matches": actual == expected, "expected": expected.to_dict(), "actual": actual.to_dict()}
