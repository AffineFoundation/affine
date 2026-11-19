"""Store adapter for deterministic pool rotation. Off mode builds none of it.

Only the active-set artifact id is added to State. The larger ledger and
retired archive live in content-addressed, public rotation artifacts. Each
published rotation carries every public input needed to replay it, including
the complete reserve queue and ledger at the previous round boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import layout
from .canonical import canonical_json, sha256_hex
from .envs.math_pool import MathPool, Task, load_pool
from .envs.reserve import Catalog, candidate_metadata, load_reserve, read_raw, reserve_dir
from .errors import ConfigError, VerifyError
from .irt import IRTSampler
from .params import Params
from .rotation import INFO_RULE_VERSION, RANKING_OUTSIDE_LIMIT, Reserve, Rotation, fold_ledger, rotate, rotate_info
from .seams import ObjectStore
from .state import State


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
        store.put(layout.reserve_shard(row["shard"], row["sha256"]), (root / "shards" / row["shard"]).read_bytes())
    store.put(layout.RESERVE_WITHDRAWN, withdrawn)
    store.put(layout.RESERVE_MANIFEST, manifest)


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


def active_file(store: ObjectStore, rotation: Rotation, catalog: MathPool | None = None, *, decision_record: Rotation | None = None) -> None:
    opened_at = rotation.opened_at
    store.put(layout.active_pool(rotation.active_sha), canonical_json(list(rotation.active_ids)))
    if rotation.initial_active_sha:
        # A pure Rotation stores the initial active list in its previous-round
        # artifact when there is no preceding decision; the production open
        # path writes it explicitly before this function.
        if not store.exists(layout.active_pool(rotation.initial_active_sha)):
            raise VerifyError("initial active pool artifact is missing")
    store.put(layout.pool_rotation(rotation.round_id), (decision_record or rotation).to_json())
    store.put(layout.pool_rotation_state(rotation.digest()), rotation.to_json())
    if rotation.rule_version == INFO_RULE_VERSION:
        ranking = {"schema": "affine-pool-ranking-v1", "round_id": rotation.round_id,
                   "rule_version": rotation.rule_version, "active_sha": rotation.active_sha,
                   "source_snapshot_id": rotation.source_snapshot_id, "reserve_manifest_sha": rotation.reserve_manifest_sha,
                   "rule_params": rotation.rule_params, "candidates": list(rotation.candidate_ranking),
                   "ranking_complete": rotation.ranking_complete}
        store.put(layout.pool_ranking(rotation.round_id), canonical_json(ranking))
        store.put(layout.POOL_SUMMARY, canonical_json(pool_summary(rotation, opened_at)))
        if decision_record is None:
            history = [json.loads(store.get(key)) for key in store.list("public/pool/rotation/")]
            recent = [row for row in history if row.get("rule_version") == INFO_RULE_VERSION][-20:]
            summary = pool_summary(rotation, opened_at)
            summary["recent_rounds"] = [{"round_id": row["round_id"], "opened_at": row.get("opened_at", 0),
                                          "active_size": len(row["active_ids"]), "admitted": len(row["admitted"]),
                                          "retired": len(row["retired"]), "retire_reason_counts": row["retire_reason_counts"],
                                          "rotation": layout.pool_rotation(row["round_id"])} for row in recent]
            store.put(layout.POOL_SUMMARY, canonical_json(summary))
    if store.exists(layout.RETIRED_POOL):
        previous = json.loads(store.get(layout.RETIRED_POOL))
        current = list(rotation.retired_archive)
        if current[:len(previous)] != previous:
            raise VerifyError("retired archive would change an earlier row")
    store.put(layout.RETIRED_POOL, canonical_json(list(rotation.retired_archive)))
    if catalog is not None:
        rows = []
        for row in rotation.retired_archive:
            try:
                task = catalog.task(row["id"])
            except Exception:
                continue
            rows.append({"id": task.task_id, "env": task.slice, "problem": task.problem, "answer": task.answer})
        store.put(layout.RETIRED_TASKS, canonical_json(rows))


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
                 base: MathPool | None = None, initial_ids: Sequence[str] | None = None) -> None:
        self.store, self.params, self.state = store, params, state
        self.base = base or load_pool()
        self.initial_ids = tuple(sorted(initial_ids or self.base.task_ids))
        self.root = root
        self.previous: Rotation | None = None
        if state.rotation_state_sha:
            self.previous = load_rotation(store, state.rotation_state_sha)
        self.admitted_counts: dict[str, int] = {}
        current = self.previous
        while current is not None:
            for task_id in current.admitted:
                self.admitted_counts[task_id] = self.admitted_counts.get(task_id, 0) + 1
            current = load_rotation(self.store, current.prev_state_sha) if current.prev_state_sha else None
        self._restore_public_cache()
        self.reserve, self.catalog = load_reserve(root, with_catalog=True, allow_retired=self.params.rotation_rule == "info")
        self.candidate_meta = candidate_metadata(root, self.reserve, self.catalog) if self.params.rotation_rule == "info" else {}
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
        self.admission_pool = self.catalog
        self.task_ids, self.sha256 = self.catalog.task_ids, self.catalog.sha256

    @property
    def active_ids(self) -> tuple[str, ...]:
        return self._active_ids

    def _restore_public_cache(self) -> None:
        """Restore consumed reserve and retired rows from durable mirrors.

        A fresh checkout has only the empty packaged reserve. A newer local
        ingest may extend the last public mirror, but it may not replace it.
        """
        root = Path(self.root or reserve_dir())
        local = root / "shards"
        if not local.is_dir() or not (root / "MANIFEST.jsonl").is_file():
            return
        if self.store.exists(layout.RETIRED_POOL):
            index = json.loads(self.store.get(layout.RETIRED_POOL))
            tasks = json.loads(self.store.get(layout.RETIRED_TASKS))
            if not isinstance(index, list) or not isinstance(tasks, list):
                raise VerifyError("public retired archive is not a list")
            (root / "retired.json").write_bytes(canonical_json(index))
            (root / "retired_tasks.json").write_bytes(canonical_json(tasks))
        if not self.store.exists(layout.RESERVE_MANIFEST):
            return
        manifest = self.store.get(layout.RESERVE_MANIFEST)
        withdrawn = self.store.get(layout.RESERVE_WITHDRAWN)
        remote = [json.loads(line) for line in manifest.splitlines()]
        here = [json.loads(line) for line in (root / "MANIFEST.jsonl").read_bytes().splitlines()]
        if len(here) == 1 and set(here[0]) == {"withdrawn"}:
            # The shipped empty manifest is a cache miss, not evidence of
            # deleted history. Its existing withdrawal pointer is checked by
            # the normal loader after restoration.
            here = remote
        if here[:-1][:len(remote) - 1] != remote[:-1]:
            raise VerifyError("local reserve does not extend its public mirror")
        if manifest != (root / "MANIFEST.jsonl").read_bytes():
            (root / "MANIFEST.jsonl").write_bytes(manifest)
            (root / "withdrawn.jsonl").write_bytes(withdrawn)
        prefix = "public/pool/reserve/shards/"
        mirrored: dict[str, bytes] = {}
        for key in self.store.list(prefix):
            # Keys are <name>.jsonl.<sha256>.jsonl; the original name
            # already includes its .jsonl suffix.
            name = key[len(prefix):].rsplit(".", 2)[0]
            mirrored[name] = self.store.get(key)
        for name, raw in mirrored.items():
            target = local / name
            if target.exists() and target.read_bytes() != raw:
                raise VerifyError(f"local reserve shard {name} differs from its public mirror")
        for name, raw in mirrored.items():
            (local / name).write_bytes(raw)

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
        history["audit_verdicts"] = {task: verdict for task, verdict in self.reserve.audit_verdicts.items() if task in queue}
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

    def open(self, round_id: str, snapshot: Mapping[str, Any]) -> tuple[Rotation, IRTSampler]:
        if self.active_rotation is not None:
            raise ConfigError("cannot rotate while a rotated round is open")
        if snapshot.get("snapshot_id") and self.state.snapshot_id and snapshot.get("snapshot_id") != self.state.snapshot_id:
            raise VerifyError("rotation source snapshot is not the last closed round's snapshot")
        previous = self.previous
        a_prev = previous.active_ids if previous else self.initial_ids
        reserve = self._rule_reserve()
        retired_archive = self._retired_inputs(previous)
        if self.params.rotation_rule == "info":
            previous_ledger = dict(previous.ledger if previous else {})
            previous_ledger.update(previous.retired_ledger if previous else {})
            rotate_rule = rotate_info
        else:
            previous_ledger = previous.ledger if previous else {}
            rotate_rule = rotate
        decision = rotate_rule(round_id, a_prev, snapshot, previous_ledger, reserve, self.params,
                               initial_size=len(self.initial_ids), retired_archive=retired_archive,
                               source_snapshot_id=snapshot.get("snapshot_id", ""),
                               prev_state_sha=self.state.rotation_state_sha if previous and previous.rule_version != INFO_RULE_VERSION else previous.digest() if previous else "",
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
            self.store.put(layout.active_pool(decision.prev_active_sha), canonical_json(list(a_prev)))
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
        active_file(self.store, decision, self.catalog)
        mirror_reserve(self.store, self.root)
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
        self.store.put(layout.RETIRED_TASKS, canonical_json(tasks))

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
    actual = rotate_rule(round_id, previous.active_ids if previous else load_active(store, expected.initial_active_sha),
                         source, previous_ledger, reserve, params,
                         initial_size=len(load_active(store, expected.initial_active_sha)) if previous is None else len(previous.active_ids),
                         retired_archive=archive, source_snapshot_id=expected.source_snapshot_id, prev_state_sha=expected.prev_state_sha,
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
        candidates = list(actual.candidate_ranking)
        if not actual.ranking_complete:
            candidates = [row for row in candidates if row["state"] in ("active", "retired") or row["retired"] or row["id"] in actual.active_ids]
            outside = [row for row in actual.candidate_ranking if row["id"] not in {item["id"] for item in candidates}]
            candidates.extend(outside[:RANKING_OUTSIDE_LIMIT])
            order = {row["id"]: index for index, row in enumerate(actual.candidate_ranking)}
            candidates.sort(key=lambda row: order[row["id"]])
        if ranking["candidates"] != candidates or ranking["round_id"] != round_id:
            raise VerifyError("public v2 ranking does not match the recomputed decision")
    return {"round_id": round_id, "matches": actual == expected, "expected": expected.to_dict(), "actual": actual.to_dict()}
