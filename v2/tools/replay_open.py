#!/usr/bin/env python3
"""Offline replay of a ranked-pool open after an unscored rotation.

Uses production journal replay, PoolRuntime and Controller.open_round with a
LocalStore. This is the non-GPU pool/sampler part of assembly.stack and the
production controller open/close path. No GPU, chain, network or prod writes.
The scored null close is synthetic and labelled as such in the scratch store.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, manifest_digest, sha256_hex
from sn120_v2.controller import Controller, StackContext, stored_irt_snapshot
from sn120_v2.envs.reserve import reserve_dir
from sn120_v2.irt import IRTSampler
from sn120_v2.journal import Journal, Mirror, pull
from sn120_v2.params import Params
from sn120_v2.pool_runtime import PoolRuntime
from sn120_v2.state import replay
from sn120_v2.store import LocalStore
from sn120_v2.types import BoxResult, EntryResult, FrozenBatch, LoRAArtifact, Registration, TrainReport, Verdict

PROD_PARAMS_HASH = "94f931588114f335814231409938f39f2056893708bc798e90f1f7676920a35c"


class OfflineAdmission:
    """Deterministic valid seats; no production miner files are needed."""

    def __init__(self, store: LocalStore) -> None:
        self.store = store

    def verdicts(self, hotkeys: Sequence[str]) -> Mapping[str, Verdict]:
        return {hotkey: Verdict(hotkey, True, batch_sha256=sha256_hex(hotkey.encode())) for hotkey in hotkeys}

    def freeze(self, reg: Registration, round_id: str) -> FrozenBatch:
        raw = canonical_json({"offline": True, "hotkey": reg.hotkey})
        key = layout.frozen_batch(round_id, reg.hotkey)
        self.store.put(key, raw)
        return FrozenBatch(reg.hotkey, sha256_hex(raw), key, "offline-replay", 1, 1)


def show(stage: str, **fields: Any) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def empty_reserve(work: Path) -> Path:
    root = work / "reserve"
    shutil.copytree(reserve_dir(), root, dirs_exist_ok=True)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    return root


def reserve_sources(store: LocalStore) -> tuple[dict[str, str], dict[str, int]]:
    rows = [json.loads(line) for line in store.get(layout.RESERVE_MANIFEST).splitlines()]
    sources, sizes = {}, {}
    for row in rows[:-1]:
        name = row["shard"]
        raw = store.get(layout.reserve_shard(name, row["sha256"]))
        if sha256_hex(raw) != row["sha256"]:
            raise ValueError(f"reserve shard hash mismatch: {name}")
        lines = [json.loads(line) for line in raw.splitlines()]
        if len(lines) != row["count"]:
            raise ValueError(f"reserve shard count mismatch: {name}")
        sources.update({line["id"]: name for line in lines})
        sizes[name] = len(lines)
    return sources, sizes


def controller_at(store: LocalStore, work: Path, restart: str) -> tuple[Controller, PoolRuntime]:
    # StackContext.load and Journal/Mirror are exactly open_controller's wiring;
    # PoolRuntime/IRTSampler match assembly's non-GPU construction verbatim.
    context = StackContext(work / "state", Params(), command="run")
    state = context.load(store)
    pool = PoolRuntime(store, context.params, state, root=empty_reserve(work))
    sampler = IRTSampler(pool.active_ids, context.params, snapshot=stored_irt_snapshot(store, state))
    journal = Journal(work / "state" / "journal.jsonl", restart, 1_800_000_000, Mirror(store))
    return Controller(journal, store, OfflineAdmission(store), sampler, context.params, pool=pool), pool


def open_round(controller: Controller, store: LocalStore, round_id: str, entropy: str) -> dict[str, Any]:
    before = controller.state.snapshot_id
    now = max(controller.state.last_failure_at + controller.params.apply_retry_s + 1, 1_800_000_001)
    tickets = controller.open_round(now, entropy)
    if not tickets:
        raise RuntimeError(f"{round_id}: no seats opened")
    rotation = controller._rotation
    if rotation is None or len(tickets) != 1:
        raise RuntimeError(f"{round_id}: expected one rotated lease")
    draw = tickets[0].draw
    sources, shard_sizes = reserve_sources(store)
    r2_shards = {name for name in shard_sizes if 11 <= int(name[:4]) <= 26}
    refill_size = sum(shard_sizes[name] for name in sorted(r2_shards))
    return {
        "round_id": round_id,
        "result": "PASS",
        "active_size": len(rotation.active_ids),
        "active_sha": rotation.active_sha,
        "retired": len(rotation.retired),
        "admitted": len(rotation.admitted),
        "r2_reserve_admits": sum(sources.get(task) in r2_shards for task in rotation.admitted),
        "r2_refill_shard_tasks": refill_size,
        "r2_refill_shards": sorted(r2_shards),
        "reserve_total_tasks": sum(shard_sizes.values()),
        "reserve_manifest_sha": rotation.reserve_manifest_sha,
        "params_hash": controller.params.params_hash,
        "source_snapshot_id": before,
        "snapshot_id": draw.snapshot_id,
        "draw_sha": sha256_hex(canonical_json(draw.to_dict())),
        "draw_size": len(draw.task_ids),
    }


def synthetic_null_close(controller: Controller) -> dict[str, Any]:
    """Exercise result validation, judging and the real IRT update close path."""
    ticket = controller.tickets()[0]
    n = len(ticket.draw.task_ids)
    outcomes = tuple("P" * n for _ in range(ticket.params.samples_per_task))
    entries = []
    for entry in ticket.entries:
        recipe = ticket.params.recipe.for_miner(entry.batch.train)
        raw = canonical_json({"synthetic_offline_lora": entry.queue_pos})
        controller.store.put(f"offline/loras/{entry.queue_pos}/weight", raw)
        files = {"weight": sha256_hex(raw)}
        artifact = LoRAArtifact(manifest_digest(files), files, {"weight": len(raw)},
                                f"offline/loras/{entry.queue_pos}", ticket.head.version_id,
                                entry.batch.batch_sha256, ticket.params.recipe_id, recipe.rank,
                                recipe.alpha, recipe.target_modules,
                                TrainReport(1, 1, 0.0, 0.0, 1, 1, "offline", ticket.params.recipe.image_digests[0]))
        # Exactly null M's outcomes: scored and valid, but never a winner.
        entries.append(EntryResult(entry.queue_pos, "scored", lora=artifact, outcomes=outcomes))
    result = BoxResult(ticket.round_id, ticket.lease_id, ticket.box_id, ticket.digest(),
                       ticket.head.version_id, ticket.params.recipe.image_digests[0], outcomes, tuple(entries),
                       ticket.issued_at + 1, ticket.issued_at + 2)
    before = controller.state.snapshot_id
    controller.submit_result(result, ticket.issued_at + 3)
    closed = controller.state.rounds[-1]
    if closed.outcome != "null" or controller.state.snapshot_id == before:
        raise RuntimeError(f"synthetic close was not a scored null with an IRT update: {closed.outcome}")
    return {"round_id": closed.round_id, "outcome": closed.outcome,
            "snapshot_id": controller.state.snapshot_id, "previous_snapshot_id": before}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="read-only LocalStore-layout production copy")
    parser.add_argument("--work", type=Path, required=True, help="new empty scratch directory for replay writes")
    args = parser.parse_args(argv)
    source = LocalStore(args.store)
    if args.work.exists() and any(args.work.iterdir()):
        parser.error("--work must be empty; refusing to overwrite an existing replay")
    args.work.mkdir(parents=True, exist_ok=True)
    state = replay(pull(source, args.work / "import" / "journal.jsonl"))
    params = state.params
    if params is None:
        raise RuntimeError("the supplied journal has no pinned params")
    show("input", params_hash=params.params_hash, params_verified=params.params_hash == PROD_PARAMS_HASH,
         last_round=state.rounds[-1].round_id, last_outcome=state.rounds[-1].outcome,
         last_snapshot_id=state.snapshot_id)
    shutil.copytree(args.store, args.work / "store")
    store = LocalStore(args.work / "store")
    controller, _ = controller_at(store, args.work, "initial")
    next_id = f"r{controller.state.round_seq + 1:06d}"
    try:
        first = open_round(controller, store, next_id, "offline-r000080")
    except Exception as exc:
        show(next_id, result="REFUSE", refused=f"{type(exc).__name__}: {exc}", params_hash=params.params_hash)
        return 1
    show(next_id, **first)
    # Fresh state rebuilt from the journal and store between the two opens.
    show("synthetic_close", **synthetic_null_close(controller))
    restarted, _ = controller_at(store, args.work / "restarted", "after-scored-close")
    second_id = f"r{restarted.state.round_seq + 1:06d}"
    second = open_round(restarted, store, second_id, "offline-r000081")
    show(second_id, **second)
    # Independently replay the r000081 open again, using its actual journal
    # seed and the IRT snapshot stored by the open. No rotation is recomputed.
    final = replay(pull(store, args.work / "restart-check" / "journal.jsonl"))
    pool = PoolRuntime(store, params, final, root=empty_reserve(args.work / "restart-check"))
    if pool.active_rotation is None or final.round is None:
        raise RuntimeError("restart did not recover the open rotation")
    sampler = IRTSampler(pool.active_ids, params,
                         snapshot=json.loads(store.get(layout.irt_snapshot(final.round.snapshot_id))))
    draw = sampler.draw(final.head, final.round.round_seed, params.n_tasks)
    draw_sha = sha256_hex(canonical_json(draw.to_dict()))
    same = pool.active_rotation.active_sha == second["active_sha"] and draw_sha == second["draw_sha"]
    show(f"{second_id}_restart", result="PASS" if same else "FAIL",
         active_sha=pool.active_rotation.active_sha, draw_sha=draw_sha,
         same_active_sha=pool.active_rotation.active_sha == second["active_sha"], same_draw=draw_sha == second["draw_sha"])
    if not same or first["active_size"] != 10_000 or second["active_size"] != 10_000:
        raise RuntimeError("restart equality or 10,000-task active-size check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
