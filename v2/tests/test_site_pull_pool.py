"""Public reserve and rotation artifacts mirrored by v2/site/pull.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json
from sn120_v2.store import LocalStore


def pull_module():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "site"))
    import pull
    return pull


def test_layout_key_accepts_exact_pool_keys_and_refuses_unknowns():
    pull = pull_module()
    fixed = (layout.RESERVE_MANIFEST, layout.RESERVE_WITHDRAWN, layout.RETIRED_POOL, layout.RETIRED_TASKS, layout.POOL_SUMMARY)
    dynamic = (layout.reserve_shard("0001-set.jsonl", "a" * 64), layout.active_pool("b" * 64),
               layout.pool_rotation("r000066"), layout.pool_rotation_state("c" * 64), layout.pool_ranking("r000066"))
    assert all(pull.layout_key(key) and pull.mine(key) for key in (*fixed, *dynamic))
    bad = ("public/pool/other.json", "public/pool/reserve/other.json", "public/pool/reserve/shards/deep/x.jsonl",
           "public/pool/reserve/shards/0001-set.jsonl.jsonl", "public/pool/reserve/shards/0001-set.bad.jsonl",
           "public/pool/active/not-sha/deep.json", "public/pool/rotation/r000066/extra.json",
           "public/pool/state/bad.json", "public/pool/active/../x.json", "public/pool/ranking/r1.json", "public/pool/ranking/other.json")
    assert not any(pull.layout_key(key) for key in bad)


def make_pool(store: LocalStore, site: Path) -> set[str]:
    shards = []
    for number in (1, 2):
        name = f"000{number}-set.jsonl"
        raw = canonical_json({"id": f"omni:{number}", "problem": f"problem {number}"}) + b"\n"
        sha = hashlib.sha256(raw).hexdigest()
        key = layout.reserve_shard(name, sha)
        store.put(key, raw)
        shards.append({"shard": name, "sha256": sha, "count": 1, "added_at": "2026-11-19"})
    manifest = b"".join(canonical_json(row) + b"\n" for row in [*shards, {"withdrawn": {"sha256": "0" * 64, "count": 0}}])
    store.put(layout.RESERVE_MANIFEST, manifest)
    store.put(layout.RESERVE_WITHDRAWN, b"")
    active, previous, initial, state, prev_state = (f"{letter}" * 64 for letter in "abcde")
    rotation = {"active_sha": active, "prev_active_sha": previous, "initial_active_sha": initial,
                "state_sha": state, "prev_state_sha": prev_state, "round_id": "r000066"}
    round_id = "r000066"
    store.put(layout.pool_rotation(round_id), canonical_json(rotation))
    store.put(layout.round_file(round_id), canonical_json({"round_id": round_id, "pool": {"active_sha": active, "state_sha": state}}))
    for sha in (active, previous, initial):
        store.put(layout.active_pool(sha), canonical_json(["omni:1"]))
    for sha in (state, prev_state):
        store.put(layout.pool_rotation_state(sha), canonical_json(rotation))
    for key in (layout.RETIRED_POOL, layout.RETIRED_TASKS):
        store.put(key, canonical_json([]))
    doc = {"history": [{"round_id": round_id, "irt_snapshot_id": "irt-one"}], "round": None, "kings": []}
    (site / layout.MINING_JSON).parent.mkdir(parents=True)
    (site / layout.MINING_JSON).write_bytes(canonical_json(doc))
    keys = {layout.RESERVE_MANIFEST, layout.RESERVE_WITHDRAWN, layout.RETIRED_POOL, layout.RETIRED_TASKS,
            layout.pool_rotation(round_id), layout.active_pool(active), layout.active_pool(previous), layout.active_pool(initial),
            layout.pool_rotation_state(state), layout.pool_rotation_state(prev_state)}
    keys |= {layout.reserve_shard(row["shard"], row["sha256"]) for row in shards}
    return keys


def test_open_round_pool_uses_mining_view_and_mirrors_ranking_and_summary(tmp_path):
    store = LocalStore(tmp_path / "store")
    site = tmp_path / "site"
    rid = "r000070"
    active, state = "a" * 64, "b" * 64
    store.put(layout.pool_rotation(rid), canonical_json({"round_id": rid, "active_sha": active,
        "state_sha": state, "rule_version": "affine-pool-rotation-v2"}))
    store.put(layout.active_pool(active), canonical_json(["new:1"]))
    store.put(layout.pool_rotation_state(state), canonical_json({"round_id": rid}))
    store.put(layout.pool_ranking(rid), canonical_json({"round_id": rid, "candidates": []}))
    store.put(layout.POOL_SUMMARY, canonical_json({"round_id": rid, "active_size": 1}))
    (site / layout.MINING_JSON).parent.mkdir(parents=True)
    (site / layout.MINING_JSON).write_bytes(canonical_json({"history": [], "kings": [], "round": {
        "round_id": rid, "irt_snapshot_id": "irt-one", "pool": {"active_sha": active, "state_sha": state}}}))
    pull = pull_module().Pull(store, site, 123)
    pull.pool()
    assert not store.exists(layout.round_file(rid))
    assert pull.problems == []
    for key in (layout.pool_rotation(rid), layout.pool_ranking(rid), layout.POOL_SUMMARY,
                layout.active_pool(active), layout.pool_rotation_state(state)):
        assert (site / key).read_bytes() == store.get(key)
    second = pull_module().Pull(store, tmp_path / "missing-round", 124)
    (tmp_path / "missing-round" / layout.MINING_JSON).parent.mkdir(parents=True)
    (tmp_path / "missing-round" / layout.MINING_JSON).write_bytes(canonical_json({"round": {"round_id": "r000071"}, "history": [], "kings": []}))
    second.pool()
    assert second.problems == [] and not (tmp_path / "missing-round" / layout.pool_ranking(rid)).exists()


def test_old_rotation_decision_trims_but_content_addressed_artifacts_stay(tmp_path):
    store = LocalStore(tmp_path / "store")
    site = tmp_path / "site"
    keys = make_pool(store, site)
    first = pull_module().Pull(store, site, 123, keep_rounds=2)
    first.pool()
    (site / layout.MINING_JSON).write_bytes(canonical_json({"history": [], "round": None, "kings": []}))
    second = pull_module().Pull(store, site, 124, keep_rounds=2)
    second.pool()
    assert not (site / layout.pool_rotation("r000066")).exists()
    assert layout.pool_rotation("r000066") not in second.owned
    for key in keys - {layout.pool_rotation("r000066")}:
        assert (site / key).read_bytes() == store.get(key)


def test_pool_mirror_copies_every_artifact_and_index_and_skips_tampered_shard(tmp_path):
    store = LocalStore(tmp_path / "store")
    site = tmp_path / "site"
    keys = make_pool(store, site)
    pull = pull_module().Pull(store, site, 123)
    pull.pool()
    files = {key: (site / key).read_bytes() for key in keys}
    assert files == {key: store.get(key) for key in keys}
    assert keys <= set(json.loads((site / pull_module().INDEX).read_bytes())["files"])
    again = pull_module().Pull(store, site, 124)
    again.pool()
    assert again.problems == [] and {key: (site / key).read_bytes() for key in keys} == files

    reserve_only = tmp_path / "reserve-only"
    reserve_doc = {"history": [], "round": None, "kings": []}
    (reserve_only / layout.MINING_JSON).parent.mkdir(parents=True)
    (reserve_only / layout.MINING_JSON).write_bytes(canonical_json(reserve_doc))
    reserved = pull_module().Pull(store, reserve_only, 125)
    reserved.pool()
    reserve_keys = {key for key in keys if key.startswith("public/pool/reserve/")}
    assert {key: (reserve_only / key).read_bytes() for key in reserve_keys} == {key: store.get(key) for key in reserve_keys}
    assert not (reserve_only / layout.pool_rotation("r000066")).exists()
    shard = next(key for key in keys if key.startswith("public/pool/reserve/shards/"))
    store.put(shard, b"tampered")
    second = tmp_path / "second"
    (second / layout.MINING_JSON).parent.mkdir(parents=True)
    (second / layout.MINING_JSON).write_bytes((site / layout.MINING_JSON).read_bytes())
    other = pull_module().Pull(store, second, 124)
    other.pool()
    assert not (second / shard).exists()
    assert any("does not match the reserve manifest sha256" in note for note in other.notes)


def test_full_run_mirrors_real_rotated_round_artifacts(tmp_path):
    from test_rotation_controller import rotated_world
    from sn120_v2.publisher import render

    world, _ = rotated_world(tmp_path / "world")
    record = world.round()
    world.register(2, 0.0)
    record = world.round()
    round_key = layout.round_file(record.round_id)
    view = json.loads(world.store.get(round_key))
    view["pool"]["state_sha"] = record.rotation["state_sha"]
    world.store.put(round_key, canonical_json(view))
    class SigningStore(LocalStore):
        def presign_get(self, key, ttl):
            return f"https://site.example/{key}?expires={ttl}"

    store = SigningStore(tmp_path / "store")
    for key in world.store.list(""):
        store.put(key, world.store.get(key))
    mining, llms = render(world.publisher.snapshots[-1], "https://site.example")
    store.put(layout.MINING_JSON, mining)
    store.put(layout.LLMS_TXT, llms)
    manifest = json.loads(store.get(layout.CURRENT))
    for name in manifest["base_files"]:
        store.put(layout.public_base_file(manifest["head"]["base_hash"], name), b"sim base")
    for name in manifest["master_files"]:
        store.put(layout.public_master_file(manifest["head"]["base_fp32_master_digest"], name), b"sim master")
    site = tmp_path / "site"
    site.mkdir()
    code = pull_module().run(store, site, "https://site.example", 600, 200, 128 << 20, None, 123)
    assert code == 0
    rotation = json.loads(store.get(layout.pool_rotation(record.round_id)))
    keys = {layout.RESERVE_MANIFEST, layout.RESERVE_WITHDRAWN, layout.RETIRED_POOL, layout.RETIRED_TASKS,
            layout.pool_rotation(record.round_id), layout.active_pool(rotation["active_sha"]),
            layout.active_pool(rotation["prev_active_sha"]), layout.pool_rotation_state(record.rotation["state_sha"]),
            layout.pool_rotation_state(rotation["prev_state_sha"])}
    keys |= {key for key in store.list("public/pool/reserve/shards/")}
    assert all((tmp_path / "site" / key).read_bytes() == store.get(key) for key in keys)


def test_pool_step_without_pool_objects_has_no_file_or_index_effect(tmp_path):
    store = LocalStore(tmp_path / "store")
    site = tmp_path / "site"
    doc = {"history": [{"round_id": "r000001", "irt_snapshot_id": "irt-one"}], "round": None, "kings": []}
    (site / layout.MINING_JSON).parent.mkdir(parents=True)
    (site / layout.MINING_JSON).write_bytes(canonical_json(doc))
    before = {str(path.relative_to(site)): path.read_bytes() for path in site.rglob("*") if path.is_file()}
    pull = pull_module().Pull(store, site, 123)
    pull.pool()
    after = {str(path.relative_to(site)): path.read_bytes() for path in site.rglob("*") if path.is_file()}
    assert after == before and pull.owned == set() and pull.problems == []
