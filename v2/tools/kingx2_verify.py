#!/usr/bin/env python3
"""Offline king-x2 / sqrt-Fisher qualification, never a production operation.

Inputs are the supplied journal copy and a LocalStore of public artifacts.
Historical rounds use their own params and public outcome strings. New-params
rounds are explicitly synthetic and use fake outcomes only. No GPU, chain,
R2, secrets or production paths are used.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Mapping

# python -I ignores PYTHONPATH and an editable install may point at another
# checkout. Resolve this checkout's package as a namespace without sys.path.
_package = Path(__file__).resolve().parents[1] / "sn120_v2"
_spec = importlib.util.spec_from_file_location("sn120_v2", _package / "__init__.py", submodule_search_locations=[str(_package)])
assert _spec is not None and _spec.loader is not None
sys.modules[__package__ or "sn120_v2"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sys.modules[__package__ or "sn120_v2"])

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, manifest_digest, sha256_hex, to_jsonable
from sn120_v2.controller import Controller, stored_irt_snapshot
from sn120_v2.envs.math_pool import load_pool
from sn120_v2.envs.reserve import reserve_dir
from sn120_v2.irt import IRTSampler, draw_from_snapshot, predicted_pass, verify_snapshot
from sn120_v2.journal import Journal, JournalRecord, read_journal
from sn120_v2.judging import check_result, decide, score_entry
from sn120_v2.params import Params
from sn120_v2.pool_runtime import PoolRuntime, load_rotation, replay_rotation
from sn120_v2.public import round_view, entry_view as public_entry_view
from sn120_v2.rotation import Rotation, fold_ledger
from sn120_v2.state import ROUND_CLOSED, ROUND_OPENED, PARAMS_SET, load
from sn120_v2.store import LocalStore
from sn120_v2.types import BoxResult, Draw, EntryResult, FrozenBatch, LoRAArtifact, RoundTicket, TrainReport, Verdict
from sn120_v2.types import NULL_MODEL

OLD_COMMIT = "28c56f439c0c0fbf3554c8546cc86dbd88119125"
OLD_PARAMS = "94f931588114f335814231409938f39f2056893708bc798e90f1f7676920a35c"
OLD_DIGEST = "64e7bd9c10bba7eeb139ca70a68c41711da882dbee42763890f85be59a4ea3f9"


def get(store: LocalStore, key: str, url: str | None = None) -> bytes:
    path = store.root / key
    if path.exists():
        return path.read_bytes()
    if url is None:
        raise FileNotFoundError(key)
    request = urllib.request.Request(url.rstrip("/") + "/" + key, headers={"Accept-Encoding": "identity", "Cache-Control": "no-cache", "User-Agent": "curl/8.0"})
    with urllib.request.urlopen(request, timeout=90) as reply:
        raw = reply.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def params_history(journal: Path) -> tuple[list[Any], dict[str, Params]]:
    rows = read_journal(journal)
    current = None
    result = {}
    for row in rows:
        if row.kind == PARAMS_SET:
            current = Params.from_dict(row.data["params"])
        if row.kind == ROUND_OPENED:
            record = row.data["round"]
            assert current is not None and current.params_hash == record["params_hash"]
            result[record["round_id"]] = current
    return rows, result


def missing_score_artifacts(store: LocalStore, rounds: list[str], public: str | None) -> list[str]:
    missing = []
    for rid in rounds:
        key = layout.round_scores(rid)
        if store.exists(key):
            continue
        if public:
            try:
                get(store, key, public)
                continue
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
        missing.append(rid)
    return missing


def materialize(store: LocalStore, journal: Path, public: str | None) -> dict[str, int]:
    """Fetch only public objects referenced by the supplied journal. No R2 API."""
    state = load(journal)
    records = state.rounds + ([state.round] if state.round else [])
    pending = {layout.RESERVE_MANIFEST, layout.RESERVE_WITHDRAWN, layout.RETIRED_POOL, layout.RETIRED_TASKS}
    for record in records:
        if record in state.rounds:
            pending.add(layout.round_file(record.round_id))
        pending.add(layout.irt_snapshot(record.snapshot_id))
        if not record.rotation:
            continue
        pending.update({layout.pool_rotation(record.round_id), layout.pool_rotation_state(record.rotation["state_sha"]),
                        layout.active_pool(record.rotation["active_sha"]), layout.active_pool(record.rotation["prev_active_sha"])})
        if record.rotation.get("prev_state_sha"):
            pending.add(layout.pool_rotation_state(record.rotation["prev_state_sha"]))
        if record.rotation.get("rule_version") == "affine-pool-rotation-v2":
            pending.add(layout.pool_ranking(record.round_id))
    if state.snapshot_id:
        pending.add(layout.irt_snapshot(state.snapshot_id))
    if state.rotation_state_sha:
        pending.add(layout.pool_rotation_state(state.rotation_state_sha))
    seen = set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        try:
            raw = get(store, key, public)
        except FileNotFoundError:
            if public is not None:
                raise
            # This optional aggregate was not present in any historical round;
            # the verifier compares its fields in each public round.json.
            if not key.startswith("public/rounds/") or not key.endswith("/scores.json"):
                raise
            continue
        except urllib.error.HTTPError as error:
            if error.code != 404 or not key.startswith("public/pool/state/"):
                raise
            expected = key.rsplit("/", 1)[-1][:-5]
            raw = None
            for number in range(1, 83):
                rid = f"r{number:06d}"
                try:
                    rotation_raw = get(store, layout.pool_rotation(rid), public)
                except urllib.error.HTTPError as missing:
                    if missing.code != 404:
                        raise
                    continue
                rotation = Rotation.from_json(rotation_raw)
                candidates = [rotation_raw, canonical_json(rotation.to_dict())]
                # PoolRuntime.close publishes the open decision under its round
                # name but addresses the folded ledger as the closed state.
                # Both the draw and every king outcome are public.
                try:
                    report = json.loads(get(store, layout.round_file(rid), public))
                    if report.get("draw"):
                        draw_ids = report["draw"]["task_ids"]
                        ledger = rotation.ledger
                        for null in report["null_outcomes"].values():
                            ledger = fold_ledger(ledger, rotation.active_ids, draw_ids, null)
                        closed = Rotation.from_dict(rotation.to_dict() | {"ledger": ledger})
                        candidates.append(canonical_json(closed.to_dict()))
                except (KeyError, urllib.error.HTTPError):
                    pass
                for candidate in candidates:
                    if sha256_hex(candidate) == expected:
                        raw = candidate
                        break
                if raw is not None:
                    break
            if raw is None:
                raise ValueError(f"cannot reconstruct public rotation state {expected}") from error
            target = store.root / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        if key.startswith("public/pool/state/"):
            rotation = json.loads(raw)
            for name in ("prev_state_sha", "initial_active_sha", "prev_active_sha", "active_sha"):
                value = rotation.get(name, "")
                if value:
                    pending.add(layout.pool_rotation_state(value) if name == "prev_state_sha" else layout.active_pool(value))
            if rotation.get("source_snapshot_id"):
                pending.add(layout.irt_snapshot(rotation["source_snapshot_id"]))
            for shard in rotation.get("reserve_history", {}).get("shards", []):
                pending.add(layout.reserve_shard(shard["shard"], shard["sha256"]))
        elif key.startswith("public/rounds/") and key.endswith("/round.json"):
            record = next(row for row in records if layout.round_file(row.round_id) == key)
            published = json.loads(raw)
            for ticket in record.leases:
                if published["null_outcomes"].get(ticket.box_id):
                    pending.add(layout.irt_snapshot(record.snapshot_id))
    return {"public_objects": len(seen), "public_bytes": sum((store.root / key).stat().st_size for key in seen)}


def verified_rotation(store: LocalStore, rid: str, public: str | None, journal_rotation: Mapping[str, Any] | None = None,
                      journal_snapshot_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replay one coherent generation of a round's mutable rotation artifacts.

    ``ranking``, ``rotation`` and the open round report are independently
    mutable public paths. Fetch all three together and compare their shared
    active/source/reserve identities to the journal before replaying. Cached
    bytes are restored in ``finally`` and their hashes are retained; a changed
    cache is never silently treated as the current public generation.
    """
    keys = ((layout.pool_ranking(rid),) if journal_rotation and journal_rotation.get("rule_version") == "affine-pool-rotation-v2" else ()) + (layout.pool_rotation(rid), layout.round_file(rid))
    cached = {key: (store.root / key).read_bytes() if (store.root / key).exists() else None for key in keys}
    fetched: dict[str, bytes] = {}
    try:
        if public is not None:
            # Bypass all three mutable-path caches; otherwise a fresh ranking
            # can accidentally be replayed against an older rotation (r081).
            for key in keys:
                request = urllib.request.Request(public.rstrip("/") + "/" + key,
                                                 headers={"Accept-Encoding": "identity", "Cache-Control": "no-cache", "User-Agent": "curl/8.0"})
                try:
                    with urllib.request.urlopen(request, timeout=90) as reply:
                        fetched[key] = reply.read()
                except urllib.error.HTTPError as error:
                    if error.code != 404 or any(cached[key] is not None for key in keys):
                        raise ValueError(f"{rid}: public generation is incomplete at {key}") from error
                (store.root / key).write_bytes(fetched[key])
            if not fetched:
                return replay_rotation(store, rid), None
        elif all(cached[key] is not None for key in keys):
            fetched = {key: cached[key] for key in keys if cached[key] is not None}
        else:
            return replay_rotation(store, rid), None
        ranking = json.loads(fetched[layout.pool_ranking(rid)]) if layout.pool_ranking(rid) in fetched else None
        rotation = Rotation.from_json(fetched[layout.pool_rotation(rid)])
        report = json.loads(fetched[layout.round_file(rid)])
        if ranking is not None:
            assert ranking["active_sha"] == rotation.active_sha
            assert ranking["source_snapshot_id"] == rotation.source_snapshot_id
            assert ranking["reserve_manifest_sha"] == rotation.reserve_manifest_sha
        assert report["round_id"] == rid
        if journal_snapshot_id is not None:
            assert report["irt_snapshot_id"] == journal_snapshot_id
        # Rotation may deliberately consume the previous close snapshot (v1,
        # or an unscored previous round); it is not necessarily the new draw's
        # snapshot. The journal's own rotation block is the authoritative join.
        if journal_rotation is not None:
            assert rotation.journal_block() == {key: value for key, value in journal_rotation.items() if key not in ("state_sha", "size")}
        result = replay_rotation(store, rid)
        assert result["matches"] and canonical_json(result["actual"]) == canonical_json(result["expected"])
        # Mutable-path bytes are restored in finally. Carry the exact coherent
        # bytes used for replay back to the caller so it cannot accidentally
        # reconstruct closed state from a different cached generation.
        result["verified_bytes"] = fetched
        old_ranking = cached.get(layout.pool_ranking(rid))
        discrepancy = None
        if ranking is not None and old_ranking is not None and old_ranking != fetched[layout.pool_ranking(rid)]:
            old_rotation = cached[layout.pool_rotation(rid)]
            discrepancy = {"sha256": sha256_hex(old_ranking),
                           "rotation_sha256": sha256_hex(old_rotation) if old_rotation is not None else None,
                           "active_sha": json.loads(old_ranking).get("active_sha"),
                           "source_snapshot_id": json.loads(old_ranking).get("source_snapshot_id"),
                           "reserve_manifest_sha": json.loads(old_ranking).get("reserve_manifest_sha"),
                           "gen_rows": sum(row["id"].startswith("gen:") for row in json.loads(old_ranking)["candidates"]),
                           "matches_journal_rotation": False,
                           "note": "older cached mutable generation preserved; current fetched generation verified against the journal"}
        return result, discrepancy
    finally:
        for key, raw in cached.items():
            if raw is not None and fetched.get(key) != raw:
                (store.root / key).write_bytes(raw)
            elif raw is None and key in fetched:
                (store.root / key).unlink(missing_ok=True)


def verify_historical(store: LocalStore, journal: Path, public: str | None, report_path: Path) -> dict[str, Any]:
    rows, historical = params_history(journal)
    state = load(journal)
    rounds = state.rounds
    assert [row.round_id for row in rounds] == [f"r{n:06d}" for n in range(1, 82)]
    assert state.round is not None and state.round.round_id == "r000082"
    assert state.params is not None and state.params.params_hash == OLD_PARAMS
    report: dict[str, Any] = {"journal_sha256": sha256_hex(journal.read_bytes()), "rounds": {}, "scores_json_missing": [], "base_commit": OLD_COMMIT,
                              "verification_scope": "journal hash chain, all public score fields, public winner outcomes, round/draw/IRT/rotation artifacts and hashes; historical reports may omit losing outcomes"}
    report["materialized"] = materialize(store, journal, public)
    timings = []
    for record in rounds:
        rid = record.round_id
        params = historical[rid]
        raw = get(store, layout.round_file(rid), public)
        published = json.loads(raw)
        assert published["params_hash"] == params.params_hash
        assert published["head_version_id"] == record.head_version_id
        assert published["irt_snapshot_id"] == record.snapshot_id
        assert published["round_seed"] == record.round_seed
        assert published["n_tasks"] == record.n_tasks == params.n_tasks
        assert published["outcome"] == record.outcome
        assert published["closed_at"] == record.closed_at and published["opened_at"] == record.opened_at
        head = next(head for head in state.heads if head.version_id == record.head_version_id)
        snapshot = json.loads(get(store, layout.irt_snapshot(record.snapshot_id), public))
        # Older public round reports omitted the draw; it remains fully
        # reconstructible from their named public snapshot and round seed.
        draw = Draw.from_dict(published["draw"]) if "draw" in published else draw_from_snapshot(snapshot, record.round_seed, record.n_tasks)
        leases = []
        public_outcomes: dict[str, tuple[str, ...]] = {}
        for lease in record.leases:
            ticket = RoundTicket(rid, lease.lease_id, lease.box_id, head, params, draw, lease.entries, lease.issued_at, lease.expires_at)
            assert ticket.digest() == lease.ticket_digest
            null = published["null_outcomes"].get(lease.box_id)
            if not null:
                continue
            for seat in lease.entries:
                decision = record.decision and record.decision.entries[seat.queue_pos]
                if decision and decision.status == "winner":
                    outcomes = tuple(published["winner_outcomes"])
                    public_outcomes[decision.lora_hash] = outcomes
                    score = score_entry(draw.weights, tuple(null), outcomes, params, n_total=len(draw.task_ids))
                    if decision.score.n_tasks < len(draw.task_ids) and published["stopped_at_limit"]:
                        score = score_entry(draw.weights, tuple(null), outcomes, params, n_total=decision.score.n_tasks)
                    assert score == decision.score, (rid, seat.queue_pos, score, decision.score)
            # Public outcomes are sufficient for the winner's score and cap
            # rates, but not for its sealed LoRA artifact or any losing result.
            # Do not manufacture a LoRAArtifact: its file hash cannot be checked
            # from round.json, and check_result/decide must remain production-only
            # validation of real sealed results.
            leases.append((ticket, tuple(null)))
            timings.append({"round": rid, **published["wall_s"], "open_to_settle": (record.decision and next(row.at for row in rows if row.kind == "round_settled" and row.data["round_id"] == rid) - record.opened_at)})
        item = {"round_sha256": sha256_hex(raw), "params_hash": params.params_hash, "outcome": record.outcome,
                "n_tasks": record.n_tasks, "null_samples": sorted({len(strings) for strings in published["null_outcomes"].values()}),
                "public_winner_decision_identical": bool(public_outcomes),
                "public_scores_identical": True, "round_metadata_identical": True}
        if draw is not None:
            rebuilt = draw_from_snapshot(snapshot, record.round_seed, record.n_tasks)
            assert canonical_json(rebuilt.to_dict()) == canonical_json(draw.to_dict()), rid
            item["draw_identical"] = True
        item["public_entry_fields_identical"] = True
        for view in published["entries"]:
            old = record.decision and next(row for row in record.decision.entries if row.queue_pos == view["queue_pos"])
            if old is None:
                expected = next(row for row in round_view(record)["entries"] if row["queue_pos"] == view["queue_pos"])
            else:
                expected = public_entry_view(old)
            # The round report adds only training fields and a safe public reason.
            assert all(view.get(key) == value for key, value in expected.items()), (rid, view["queue_pos"])
        closing_id = next((row.data.get("snapshot_id") for row in reversed(rows) if row.kind == ROUND_CLOSED and row.data["round_id"] == rid), "")
        if closing_id and leases:
            rotation = record.rotation
            pool = tuple(json.loads(get(store, layout.pool_rotation(rid), public))["active_ids"]) if rotation else tuple(snapshot["tasks"])
            # Historical snapshots may retain their prior mass configuration
            # after a params change (notably the band snapshots in r024-r032).
            # Only an actual new opening rebuilds for a changed draw rule.
            # ``mass_rule`` is prose and band's text deliberately omits its
            # values; restore the exact configuration sealed in the body.
            saved_config = snapshot["config"]
            draw_rule = saved_config.get("draw_rule", "band")
            restored = dataclasses.replace(params, irt_draw_rule=draw_rule)
            if draw_rule == "band":
                restored = dataclasses.replace(
                    restored,
                    irt_middle_band=tuple(saved_config["middle_band"]),
                    irt_explore_frac=saved_config["explore_frac"],
                )
            closing = IRTSampler(pool, restored, snapshot=snapshot, rotation=snapshot.get("rotation"))
            outcomes = {NULL_MODEL: null for _, null in leases}
            outcomes.update(public_outcomes)
            decision = record.decision
            winner = decision.winner if record.outcome == "applied" and decision else None
            private_loras = [entry.queue_pos for entry in decision.entries if entry.lora_hash and entry.status != "winner"] if decision else []
            if not private_loras:
                # head_applied is the training fold manifest. The next round's
                # head is the promoted head whose ability this update records.
                new_id = next((later.head_version_id for later in rounds if later.round_id > rid), "") if winner else ""
                boundary = closing.update(head, draw, outcomes, winner_lora_hash=winner.lora_hash if winner else "", new_head_version_id=new_id)
                assert boundary == closing_id, (rid, boundary, closing_id)
                saved = json.loads(get(store, layout.irt_snapshot(boundary), public))
                # ``mass_rule`` is explanatory text added after some historical
                # snapshots; the sealed IRT body itself must still be identical.
                body_keys = ("schema", "model", "head_version_id", "rounds", "updates", "config", "heads", "loras", "tasks")
                assert all(canonical_json(closing.public_snapshot()[key]) == canonical_json(saved[key]) for key in body_keys), rid
                item["irt_update_identical"] = True
                item["irt_body_identical"] = True
            else:
                # Recomputing IRT without losing outcomes would produce a
                # different model and must never be labelled byte identity.
                item["irt_update_identical"] = False
                item["irt_unavailable_private_scored_loras"] = private_loras
                saved = json.loads(get(store, layout.irt_snapshot(closing_id), public))
                verify_snapshot(saved)
                item["closing_irt_artifact"] = {"snapshot_id": closing_id, "sha256": sha256_hex(get(store, layout.irt_snapshot(closing_id), public)),
                                                "verified": saved["snapshot_id"] == closing_id}
        elif not closing_id:
            # The controller may close a failed/no-result round without an IRT
            # update. Assert the successor opens at the same snapshot rather
            # than manufacturing king-only outcomes for private dropped LoRAs.
            later = next((row for row in rounds if row.round_id > rid), None)
            if later is not None:
                assert later.snapshot_id == record.snapshot_id, rid
            item["irt_update_expected"] = False
            item["irt_no_update_boundary_identical"] = later is not None
        else:
            saved = json.loads(get(store, layout.irt_snapshot(closing_id), public))
            verify_snapshot(saved)
            item["irt_update_identical"] = False
            item["irt_public_inputs_incomplete"] = True
            item["closing_irt_artifact"] = {"snapshot_id": closing_id, "sha256": sha256_hex(get(store, layout.irt_snapshot(closing_id), public)),
                                            "verified": True}
        if record.rotation is not None:
            rotation, alternate_ranking = verified_rotation(store, rid, public, record.rotation, record.snapshot_id)
            assert rotation["matches"] and canonical_json(rotation["actual"]) == canonical_json(rotation["expected"])
            actual = rotation["verified_bytes"][layout.pool_rotation(rid)]
            assert Rotation.from_json(actual).to_dict() == Rotation.from_dict(rotation["expected"]).to_dict(), rid
            item["rotation"] = {"sha256": sha256_hex(actual), "matches": True, "active_size": len(rotation["expected"]["active_ids"])}
            if alternate_ranking is not None:
                item["rotation"]["alternate_cached_ranking"] = alternate_ranking
            expected_state = next((row.data.get("rotation_state_sha") for row in reversed(rows) if row.kind == ROUND_CLOSED and row.data["round_id"] == rid), "")
            if expected_state:
                opened = Rotation.from_json(actual)
                ledger = opened.ledger
                for _, null in leases:
                    ledger = fold_ledger(ledger, opened.active_ids, draw.task_ids, null)
                closed = Rotation.from_dict(opened.to_dict() | {"ledger": ledger})
                expected_raw = get(store, layout.pool_rotation_state(expected_state), public)
                # Early v1 states are addressed by their original bytes;
                # current snapshots use Rotation.digest(). Check both the
                # authenticated stored bytes and the reconstructed fields.
                assert sha256_hex(expected_raw) == expected_state or closed.digest() == expected_state, rid
                assert closed.to_dict() == Rotation.from_json(expected_raw).to_dict(), rid
                item["rotation"]["state_hash_identity"] = "raw_sha256" if sha256_hex(expected_raw) == expected_state else "canonical_digest"
                item["rotation"]["closed_state_sha256"] = expected_state
                item["rotation"]["closed_state_identical"] = True
        report["rounds"][rid] = item
        print(json.dumps({"historical": rid, **item}, sort_keys=True), flush=True)
    report["scores_json_missing"] = missing_score_artifacts(store, [row.round_id for row in rounds], public)
    report["timings"] = timings
    report["params_hash"] = state.params.params_hash
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"historical_summary": {"rounds": len(rounds), "decision_verified": sum(row["public_winner_decision_identical"] for row in report["rounds"].values()),
                                            "draw_verified": sum(row.get("draw_identical", False) for row in report["rounds"].values()),
                                            "rotation_verified": sum("rotation" in row for row in report["rounds"].values()),
                                            "irt_update_verified": sum(row.get("irt_update_identical", False) for row in report["rounds"].values()),
                                            "irt_private_inputs": sum(bool(row.get("irt_unavailable_private_scored_loras")) for row in report["rounds"].values()),
                                            "scores_json_missing": len(report["scores_json_missing"])}}, sort_keys=True), flush=True)
    return report


class Admission:
    params = None
    def verdicts(self, hotkeys):
        return {key: Verdict(key, True, batch_sha256=sha256_hex(key.encode()), checked_at=1795202000) for key in hotkeys}
    def freeze(self, reg, round_id):
        raw = canonical_json({"offline": True, "hotkey": reg.hotkey})
        key = layout.frozen_batch(round_id, reg.hotkey)
        self.store.put(key, raw)
        return FrozenBatch(reg.hotkey, sha256_hex(raw), key, "offline-replay", 1, 1)


def controller(store: LocalStore, state_dir: Path, reserve: Path, owner: str) -> Controller:
    state = load(state_dir / "journal.jsonl")
    pool = PoolRuntime(store, state.params, state, root=reserve, base=load_pool())
    sampler = IRTSampler(pool.active_ids, state.params, snapshot=stored_irt_snapshot(store, state))
    admission = Admission()
    admission.store = store
    return Controller(Journal(state_dir / "journal.jsonl", owner, 1795202000), store, admission, sampler, state.params, pool=pool)


def compare_next_open(store: LocalStore, journal: Path, reserve_source: Path, work: Path, new_params: Params) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=False)
    # Never mutate the materialized public input cache. Each run gets a
    # private store; no earlier synthetic r083/r084 result can leak into it.
    private_store = LocalStore(work / "store")
    # LocalStore creates its root in the constructor; it is empty here.
    shutil.copytree(store.root, private_store.root, dirs_exist_ok=True)
    for number in (82, 83, 84):
        key = layout.sealed_result(f"r{number:06d}", f"r{number:06d}-l0")
        if private_store.exists(key):
            previous = BoxResult.from_json(private_store.get(key))
            assert previous.entries and all(entry.lora and entry.lora.prefix.startswith("offline/loras/") for entry in previous.entries)
            private_store.delete(key)
    store = private_store
    state_dir = work / "state"
    state_dir.mkdir()
    shutil.copy2(journal, state_dir / "journal.jsonl")
    # This pointer may have advanced beyond the supplied journal snapshot in
    # the public cache. Rewind ONLY the private copy to the authenticated
    # r082 rotation; reject a genuine prefix mismatch. Never mutate the input.
    state = load(journal)
    rotation = load_rotation(store, state.rotation_state_sha)
    current_archive = list(rotation.retired_archive)
    published_archive = json.loads(store.get(layout.RETIRED_POOL))
    assert published_archive[:len(current_archive)] == current_archive
    store.put(layout.RETIRED_POOL, canonical_json(current_archive))
    reserve = work / "reserve"
    shutil.copytree(reserve_source, reserve)
    current = controller(store, state_dir, reserve, "kingx2-r082-close")
    # A prior interrupted offline replay may have left its synthetic result in
    # the shared copy of the public cache. Never overwrite a real result.
    synthetic_key = layout.sealed_result("r000082", "r000082-l0")
    if store.exists(synthetic_key):
        previous = BoxResult.from_json(store.get(synthetic_key))
        assert previous.entries and all(entry.lora and entry.lora.prefix.startswith("offline/loras/") for entry in previous.entries)
        store.delete(synthetic_key)
    assert current.state.round is not None and current.state.round.round_id == "r000082"
    # The current open is not scored by this simulation. Preserve its rotation
    # and sampler while making a local, clearly synthetic scored-null close.
    # The supplied private ticket belongs to an older state copy, so derive
    # the actual open ticket from its authenticated journal, public draw and
    # per-round params; never trust the stale ticket bytes.
    record = current.state.round
    lease = record.leases[0]
    head = next(row for row in current.state.heads if row.version_id == record.head_version_id)
    snapshot = json.loads(store.get(layout.irt_snapshot(record.snapshot_id)))
    draw = draw_from_snapshot(snapshot, record.round_seed, record.n_tasks)
    ticket = RoundTicket(record.round_id, lease.lease_id, lease.box_id, head, current.params, draw, lease.entries, lease.issued_at, lease.expires_at)
    assert ticket.digest() == lease.ticket_digest
    store.put(layout.ticket(record.round_id, lease.lease_id), ticket.to_json())
    n = len(ticket.draw.task_ids)
    null = tuple("P" * n for _ in range(ticket.params.null_samples))
    entries = []
    for seat in ticket.entries:
        recipe = ticket.params.recipe.for_miner(seat.batch.train)
        raw = canonical_json({"synthetic": seat.queue_pos})
        store.put(f"offline/loras/{seat.queue_pos}/weight", raw)
        files = {"weight": sha256_hex(raw)}
        artifact = LoRAArtifact(manifest_digest(files), files, {"weight": len(raw)}, f"offline/loras/{seat.queue_pos}", ticket.head.version_id, seat.batch.batch_sha256, ticket.params.recipe_id, recipe.rank, recipe.alpha, recipe.target_modules, TrainReport(1, 1, 0.0, 0.0, 1, 1, "offline", ticket.params.recipe.image_digests[0]))
        lora_outcomes = tuple("P" * n for _ in range(ticket.params.samples_per_task))
        entries.append(EntryResult(seat.queue_pos, "scored", lora=artifact, outcomes=lora_outcomes))
    result = BoxResult(ticket.round_id, ticket.lease_id, ticket.box_id, ticket.digest(), ticket.head.version_id, ticket.params.recipe.image_digests[0], null, tuple(entries), ticket.issued_at + 1, ticket.issued_at + 2)
    current.submit_result(result, ticket.issued_at + 3)
    assert current.state.round is None and current.state.rounds[-1].outcome == "null"
    # Two independent journal copies make the old and new open comparable.
    answer = {}
    for label, params in (("old", None), ("new", new_params)):
        branch = work / label
        branch.mkdir()
        shutil.copytree(state_dir, branch / "journal")
        shutil.copytree(reserve, branch / "reserve")
        # Build the rotation under the same params for both branches. Changing
        # only the sampling rule after rotation keeps its inputs byte-identical.
        rotated = controller(store, branch / "journal", branch / "reserve", f"kingx2-{label}-rotate")
        rotated.open_round(ticket.issued_at + 10, "kingx2-r083")
        # Exercise the real restart seam: startup reconstructs the sampler from
        # the authenticated open ticket and its public snapshot, then set-params
        # must replace it before the next opening (old params keep Fisher).
        previous_records = read_journal(branch / "journal/journal.jsonl")
        previous = previous_records[-1]
        failed = JournalRecord(len(previous_records), previous.fence, ticket.issued_at + 11, "lease_failed",
                               {"round_id": "r000083", "lease_id": "r000083-l0", "reason": "offline next-open rehearsal"}, previous.checksum)
        with (branch / "journal/journal.jsonl").open("ab") as journal_file:
            journal_file.write(failed.line() + b"\n")
        started = controller(store, branch / "journal", branch / "reserve", f"kingx2-{label}")
        # A failed lease remains an open RoundRecord until the normal settle
        # path closes it with no decision. Exercise that seam before set-params
        # or the next open; merely advancing the clock cannot close the round.
        started._settle(ticket.issued_at + 12)
        assert started.state.round is None and started.state.rounds[-1].round_id == "r000083"
        assert started.state.rounds[-1].outcome == "no_decision"
        # No scored null close runs the pool's close/commit path. On a real
        # restart PoolRuntime clears active_rotation from the replayed State;
        # mirror that constructor seam in this already-started fixture.
        started.pool.active_rotation = None
        started._rotation = None
        if params is not None:
            started.set_params(params, ticket.issued_at + 13)
        else:
            # Keep the same event-time boundary on both branches so their
            # rotation is an input-identical comparison, not a timestamp diff.
            started.set_params(started.params, ticket.issued_at + 13)
        # A lease failure starts the normal apply-retry backoff. The synthetic
        # next-open clock must advance past it, or no tickets are returned.
        reopen_at = ticket.issued_at + 12 + started.params.apply_retry_s
        tickets = started.open_round(reopen_at, "kingx2-r084")
        assert len(tickets) == 1
        rotation = started._rotation
        assert len(rotation.active_ids) == 10000
        snap = json.loads(store.get(layout.irt_snapshot(tickets[0].draw.snapshot_id)))
        answer[label] = {"round": tickets[0].round_id, "params_hash": started.params.params_hash,
                         "active_sha": rotation.active_sha, "rotation_sha": rotation.digest(), "active_size": len(rotation.active_ids),
                         "draw_sha": sha256_hex(canonical_json(tickets[0].draw.to_dict())), "snapshot_rule": snap["config"]["draw_rule"],
                         "snapshot_sha": snap["snapshot_id"], "first_draw": list(tickets[0].draw.task_ids[:5])}
    assert answer["new"]["snapshot_rule"] == "sqrt_fisher"
    assert answer["new"]["snapshot_sha"] != answer["old"]["snapshot_sha"]
    answer["same_active_sha"] = answer["old"]["active_sha"] == answer["new"]["active_sha"]
    answer["same_rotation"] = answer["old"]["rotation_sha"] == answer["new"]["rotation_sha"]
    return answer


def synthetic_round(store: LocalStore, journal: Path, output: Path, new_params: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=False)
    state = load(journal)
    # The exact deploy-kit JSON is the source of the selected-params evidence,
    # including its recipe code digest. Do not derive a different hash here.
    params = Params.from_dict(json.loads(new_params.read_text()))
    assert params.null_samples == 2 and params.samples_per_task == 1
    assert params.irt_draw_rule == "sqrt_fisher"
    assert params.recipe.code_digest != state.params.recipe.code_digest
    assert dataclasses.replace(params, recipe=state.params.recipe, null_samples_per_task=None, irt_draw_rule="fisher") == state.params
    rotation = replay_rotation(store, "r000082")
    active = tuple(rotation["expected"]["active_ids"])
    source = json.loads(store.get(layout.irt_snapshot(rotation["expected"]["source_snapshot_id"])))
    old_sampler = IRTSampler(active, state.params, snapshot=source, rotation=source.get("rotation"))
    new_sampler = IRTSampler(active, params, snapshot=source, rotation=source.get("rotation"))
    head = state.heads[-1]
    old = old_sampler.draw(head, "synthetic-kingx2", 6)
    new = new_sampler.draw(head, "synthetic-kingx2", 6)
    ability = float(source.get("head_ability", 0.0))
    rows = []
    for task in sorted(set(old.task_ids) | set(new.task_ids))[:6]:
        difficulty = source["tasks"][task][0]
        p = predicted_pass(ability, difficulty)
        rows.append({"task": task, "p": p, "old_mass": old_sampler.public_snapshot()["tasks"][task][3],
                     "new_mass": new_sampler.public_snapshot()["tasks"][task][3],
                     "weight_if_drawn": next((weight for other, weight in zip(new.task_ids, new.weights) if other == task), None)})
    # Four simple tasks make every arithmetic result directly auditable.
    weights = (1.0,) * 4
    null = ("PFPF", "FFFF")
    lora = ("PPPP",)
    small = dataclasses.replace(params, n_tasks=4)
    score = score_entry(weights, null, lora, small, n_total=4)
    d = [1.0 - ((a == "P") + (b == "P")) / 2 for a, b in zip(null[0], null[1])]
    delta = math.fsum(d) / len(d)
    se = math.sqrt(math.fsum((x - delta) ** 2 for x in d) / (len(d) - 1) / len(d))
    assert score.delta == delta and score.se == se
    rng = random.Random(119)
    one_se, two_se = [], []
    for _ in range(400):
        null0, null1, trained = [], [], []
        for _ in range(500):
            null0.append("P" if rng.random() < 0.5 else "F")
            null1.append("P" if rng.random() < 0.5 else "F")
            trained.append("P" if rng.random() < 0.6 else "F")
        one_se.append(score_entry((1.0,) * 500, ("".join(null0),), ("".join(trained),), dataclasses.replace(params, n_tasks=500)).se)
        two_se.append(score_entry((1.0,) * 500, ("".join(null0), "".join(null1)), ("".join(trained),), dataclasses.replace(params, n_tasks=500)).se)
    ratio = math.fsum(two_se) / math.fsum(one_se)
    result = {"new_params": params.to_dict(), "new_params_hash": params.params_hash,
              "old_params_hash": state.params.params_hash,
              "old_recipe_code_digest": state.params.recipe.code_digest,
              "new_recipe_code_digest": params.recipe.code_digest,
              "old_rule": source["config"].get("draw_rule", "band"), "new_rule": new_sampler.public_snapshot()["config"]["draw_rule"],
              "old_snapshot_sha": old_sampler.snapshot_id(), "new_snapshot_sha": new_sampler.snapshot_id(), "masses": rows,
              "old_draw": old.to_dict(), "new_draw": new.to_dict(), "synthetic_outcomes": {"null": null, "lora": lora, "d": d},
              "hand_score": score.to_dict(), "hand_delta": delta, "hand_se": se,
              "cap_denominators": {"null": 4 * 2, "lora": 4 * 1}, "fake_evaluator_se_ratio": ratio,
              "expected_independent_noise_ratio": math.sqrt(0.75)}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("history", "synthetic", "next-open", "digest"))
    parser.add_argument("--journal", type=Path, default=Path("/home/bb/scratch/kingx2-r082-live/sn120-run/state/journal.jsonl"))
    parser.add_argument("--store", type=Path, default=Path("/home/bb/scratch/kingx2-qualification/store"))
    parser.add_argument("--reserve", type=Path, default=Path("/home/bb/scratch/kingx2-r082-live/sn120/v2/sn120_v2/envs/reserve"))
    parser.add_argument("--work", type=Path, default=Path("/home/bb/scratch/kingx2-qualification"))
    parser.add_argument("--public", type=lambda value: value or None, default="https://affine.io")
    parser.add_argument("--new-params", type=Path)
    args = parser.parse_args()
    store = LocalStore(args.store)
    if args.phase == "history":
        verify_historical(store, args.journal, args.public, args.work / "history.json")
    elif args.phase == "synthetic":
        assert args.new_params is not None
        print(json.dumps(synthetic_round(store, args.journal, args.work / "synthetic.json", args.new_params), indent=2, sort_keys=True))
    elif args.phase == "next-open":
        assert args.new_params is not None
        params = Params.from_dict(json.loads(args.new_params.read_text()))
        answer = compare_next_open(store, args.journal, args.reserve, args.work / "next-open", params)
        (args.work / "next-open.json").write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n")
        print(json.dumps(answer, indent=2, sort_keys=True))
    else:
        from sn120_v2.gpu.plan import code_digest
        print(json.dumps({"code_digest": code_digest()}))


if __name__ == "__main__":
    main()
