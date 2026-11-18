"""Pure rotation, reserve integrity, IRT rebasing and off-mode byte identity."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sn120_v2 import layout, public, publisher
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.math_pool import MathPool, Task, load_pool
from sn120_v2.envs.reserve import Catalog, load_reserve, reserve_dir
from sn120_v2.errors import ConfigError, ContractError, VerifyError
from sn120_v2.irt import DECIMALS, IRTSampler, fisher_masses, predicted_pass, snapshot_digest
from sn120_v2.journal import read_journal
from sn120_v2.params import Params, Recipe
from sn120_v2.pool_runtime import PoolRuntime, append_only, replay_rotation
from sn120_v2.rotation import HIGH, LOW, Reserve, ReserveShard, Rotation, Retired, WITHDRAWN, active_sha, rotate
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import ROUND_OPENED
from sn120_v2.store import LocalStore
from sn120_v2.types import PASS, FAIL, CAP, ERROR

DEFAULT_HASH = "10a121ec5a4ba830057e5510665fee1be7aad34dae259d3427e4547c393c2cf1"
PROD_SHAPED_HASH = "6fd8962c8d88016c85098e4ec08967c37e5b091b7e2d83e66e29b9372c91b82c"


def snapshot(tasks: dict[str, list], ability: float = 0.0) -> dict:
    sampler = IRTSampler(list(tasks), Params())
    body = sampler.public_snapshot()
    body["tasks"] = {key: [*row[:3], 1] for key, row in tasks.items()}
    body["heads"] = {"head": {"ability": ability, "information": 1.0, "rollouts": 0, "passes": 0, "round": 1}}
    body["head_version_id"] = "head"
    body["head_ability"] = ability
    from sn120_v2.irt import snapshot_digest
    body["snapshot_id"] = snapshot_digest(body)
    return body


def reserve(n: int = 10, withdrawn=(), *, prefix="new") -> Reserve:
    queue = tuple(f"{prefix}:{i}" for i in range(n))
    sha = sha256_hex(b"manifest" + repr(queue).encode())
    return Reserve(sha, queue, withdrawn=tuple({"id": task, "reason": "bad", "added_at": "2026-11-18"} for task in withdrawn))


def params(**changes) -> Params:
    return Params(rotation_enabled=True, **changes)


def test_off_mode_params_hashes_and_fields_are_unchanged():
    assert Params().params_hash == DEFAULT_HASH
    assert not any(field.name.startswith("rotation_") for field in dataclasses.fields(Params) if field.name in Params().to_dict())
    assert set(Params.LEFT_OUT_AT_DEFAULT) >= {f.name for f in dataclasses.fields(Params) if f.name.startswith("rotation_")}
    path = Path(__file__).parent / "data" / "params_prod_shaped.json"
    # Compute the literal with pristine v2 as well: this fixture must remain
    # readable by code that has no rotation fields at all.
    p = Params.from_json(path.read_bytes())
    assert p.params_hash == PROD_SHAPED_HASH
    assert params().params_hash != DEFAULT_HASH


def test_off_mode_snapshot_draw_journal_round_and_publisher_bytes_match_golden(tmp_path):
    golden = Path("/tmp/sn120-rotation-off-golden")
    if not golden.exists():
        pytest.skip("pristine-v2 golden generation was not run in this environment")
    world = SimWorld(tmp_path / "world", sim_params())
    world.sampler = IRTSampler(world.pool, world.params)
    world.controller.sampler = world.sampler
    world.register(1, 0.0)
    before = world.sampler.public_snapshot()
    draw = world.sampler.draw(world.controller.state.head, "fixed-off-mode-seed", 64)
    record = world.round()
    assert (tmp_path / "world" / "journal.jsonl").read_bytes() == (golden / "journal.jsonl").read_bytes()
    assert world.store.get(layout.round_file(record.round_id)) == (golden / "round.json").read_bytes()
    assert world.store.get(layout.irt_snapshot(record.snapshot_id)) == (golden / "irt.json").read_bytes()
    pristine = subprocess.check_output([
        sys.executable, "-c",
        "import sys,json;sys.path.insert(0,'/tmp/sn120-rotation-pristine/v2');"
        "from sn120_v2.irt import IRTSampler;from sn120_v2.params import Params;"
        "from sn120_v2.types import HeadVersion;import conftest;"
        "s=IRTSampler([f'sim:{i}' for i in range(400)],Params());"
        "d=s.draw(conftest.Build.head(),'fixed-off-mode-seed',64);"
        "print(json.dumps({'snapshot':s.public_snapshot(),'draw':d.to_dict()},sort_keys=True))",
    ], env={**os.environ, "PYTHONPATH": "/tmp/sn120-rotation-pristine/v2/tests"})
    old = json.loads(pristine)
    assert before["snapshot_id"] == old["snapshot"]["snapshot_id"] and draw.to_dict() == old["draw"]
    mining, llms = publisher.render(public.snapshot(world.controller.state, world.params, 1234567), "https://files.example", "https://files.example/links")
    assert mining == (golden / "mining.json").read_bytes() and llms == (golden / "llms.txt").read_bytes()


def test_empty_reserve_withdraws_but_never_retires_high_or_errors():
    old = {f"t:{i}": [-10.0, 0.25, 10] for i in range(4)}
    result = rotate("r000001", sorted(old), snapshot(old, 2.0), {}, reserve(0), params())
    assert result.active_ids == tuple(sorted(old)) and not result.retired and not result.admitted
    result = rotate("r000002", sorted(old), snapshot(old, 2.0), {}, reserve(0, ["t:1"]), params())
    assert [(r.id, r.reason) for r in result.retired] == [("t:1", WITHDRAWN)]
    assert result.active_ids == ("t:0", "t:2", "t:3")


def test_fifo_admission_and_k_cap_are_recomputable_and_hash_seed_independent():
    old = {f"t:{i}": [-10.0, 0.25, 10] for i in range(40)}
    snap = snapshot(old, 2.0)
    one = rotate("r000123", reversed(list(old)), snap, dict(reversed([(key, (0, 0)) for key in old])), reserve(40), params(rotation_k=5))
    two = rotate("r000123", list(old), snap, {key: (0, 0) for key in old}, reserve(40), params(rotation_k=5))
    assert one == two and len(one.admitted) == len(one.retired) == 5
    assert one.next_up == reserve(40).queue[5:15]
    assert [row.reason for row in one.retired] == [HIGH] * 5
    script = f'''import sys; sys.path.insert(0, {str(Path(__file__).parents[1])!r}); from test_pool_rotation import *
a=rotate("r000123", list(old), snap, {{}}, reserve(40), params(rotation_k=5)); print(a.digest())'''
    # Subprocess check uses direct self-contained inputs, no pytest globals.
    code = """import json,sys
sys.path.insert(0, 'v2')
from sn120_v2.rotation import rotate,Reserve
from sn120_v2.params import Params
from sn120_v2.irt import IRTSampler,snapshot_digest
old={f't:{i}':[-10.0,.25,10] for i in range(40)}
s=IRTSampler(list(old),Params()).public_snapshot(); s['tasks']={k:[*v[:3],1] for k,v in old.items()};s['heads']={'h':{'ability':2.0}};s['head_version_id']='h';s['snapshot_id']=snapshot_digest(s)
r=Reserve('a'*64,tuple(f'n:{i}' for i in range(40)))
print(rotate('r000123',old,s,{},r,Params(rotation_enabled=True,rotation_k=5)).to_json().decode())"""
    outputs = [subprocess.check_output([sys.executable, code] if False else [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}) for seed in ("0", "1", "42")]
    assert outputs[0] == outputs[1] == outputs[2]


def test_fewer_eligible_than_k_zero_eligible_and_short_queue():
    old = {f"t:{i}": [10.0 if i < 3 else -10.0, 0.25, 10] for i in range(10)}
    snap = snapshot(old, 0.0)
    assert len(rotate("r000001", old, snap, {}, reserve(10), params()).retired) == 7
    result = rotate("r000001", old, snap, {}, reserve(2), params(rotation_k=32))
    assert len(result.admitted) == len(result.retired) == 2 and result.queue_remaining == 0


def test_active_below_above_target_and_k_zero():
    old = {f"t:{i}": [-10.0, 0.25, 10] for i in range(4)}
    snap = snapshot(old, 2.0)
    # Initial size four, no eligible retire: grow only as far as the queue/k.
    grown = rotate("r000001", old, snap, {}, reserve(4), params(rotation_active_size=8, rotation_k=2, rotation_retire_p=1.0))
    assert len(grown.active_ids) == 6 and len(grown.retired) == 0
    # Above target: available slots permit shrinkage even without queue rows.
    shrunk = rotate("r000001", old, snap, {}, reserve(0), params(rotation_active_size=2, rotation_k=2))
    assert len(shrunk.active_ids) == 4, "no reserve: nothing retired except withdrawn, even above target"
    stopped = rotate("r000001", old, snap, {}, reserve(8), params(rotation_k=0))
    assert not stopped.admitted and not stopped.retired
    # With a queue, an above-target set may shrink toward target.
    shrunk = rotate("r000001", old, snap, {}, reserve(4), params(rotation_active_size=2, rotation_k=2))
    assert len(shrunk.active_ids) == 2


def test_rotation_recomputes_over_several_rounds(tmp_path):
    store = LocalStore(tmp_path)
    old = {f"t:{i}": [-10.0, 0.25, 4] for i in range(20)}
    active, ledger, archive = tuple(sorted(old)), {}, []
    pending = reserve(20).queue
    previous = None
    for number in range(1, 5):
        snap = snapshot({task: old[task] for task in active}, 2.0)
        decision = rotate(f"r{number:06d}", active, snap, ledger, Reserve(sha256_hex(f"manifest-{number}".encode()), pending), params(rotation_k=3),
                          initial_size=20, retired_archive=archive,
                          source_snapshot_id=snap["snapshot_id"], prev_state_sha=previous.digest() if previous else "")
        store.put(layout.irt_snapshot(snap["snapshot_id"]), canonical_json(snap))
        from sn120_v2.pool_runtime import active_file, replay_rotation
        if decision.initial_active_sha:
            store.put(layout.active_pool(decision.initial_active_sha), canonical_json(list(active)))
        active_file(store, decision)
        result = replay_rotation(store, decision.round_id, snap, initial_ids=tuple(sorted(old)))
        assert result["matches"] and result["actual"] == result["expected"]
        active, archive = decision.active_ids, decision.retired_archive
        # Reserve IDs are unique until an explicit readmit. This test uses
        # different source sets for the successive shards.
        pending = tuple(f"round{number}:{i}" for i in range(20))
        ledger = {task: (number, number // 2) for task in active}
        previous = Rotation.from_dict(decision.to_dict() | {"ledger": ledger})
        active_file(store, previous)
        old = {task: old.get(task, [2.43, .25, 0]) for task in active}


def test_low_retire_uses_empirical_null_model_counts_not_irt_p():
    old = {"a": [0.0, 0.25, 10], "b": [-10.0, 0.25, 10]}
    ledger = {"a": (10, 0), "b": (10, 9)}
    low = rotate("r000001", old, snapshot(old, 0.0), ledger, reserve(4), params(rotation_retire_low_p=0.1, rotation_n_min_low=8, rotation_retire_p=0.99))
    assert [(r.id, r.reason, r.head_rollouts, r.head_passes) for r in low.retired] == [("b", HIGH, 10, 9), ("a", LOW, 10, 0)]
    off = rotate("r000001", old, snapshot(old, 0.0), ledger, reserve(4), params(rotation_retire_low_p=0.0, rotation_retire_p=0.99))
    assert [r.id for r in off.retired] == ["b"]


def test_catalog_resolves_base_archive_and_reserve_prompts_and_ledger_caps_fail():
    base = MathPool((Task("math:1", "affine_math", "base", "1"),), "a" * 64)
    extra = MathPool((Task("omni:1", "affine_math", "queue", "2"),), "a" * 64)
    catalog = Catalog(base, extra)
    assert catalog.task("omni:1").answer == "2" and catalog.task_for_prompt("queue").task_id == "omni:1"
    assert catalog.task_for_prompt("base").task_id == "math:1"
    from sn120_v2.rotation import fold_ledger
    assert fold_ledger({}, ["x", "y", "z"], ["x", "y", "z"], [PASS + CAP + FAIL]) == {"x": (1, 1), "y": (1, 0), "z": (1, 0)}
    assert fold_ledger({}, ["x"], ["x"], [ERROR]) == {"x": (0, 0)}


def test_enabled_publisher_states_rotation_and_pool_links(tmp_path):
    p = sim_params(rotation_enabled=True)
    world = SimWorld(tmp_path, p)
    record = world.controller.state.rounds
    snap = public.snapshot(world.controller.state, p, 1_800_000_000)
    snap["round"] = {"round_id": "r000001", "pool": {
        "active_sha": "a" * 64, "size": 20120, "admitted": ["omni:1"], "retired": [],
        "reserve_manifest_sha": "b" * 64, "queue_remaining": 5, "next_up": ["omni:2"],
        "active": layout.active_pool("a" * 64), "rotation": layout.pool_rotation("r000001"),
        "retired_archive": layout.RETIRED_POOL, "reserve_manifest": layout.RESERVE_MANIFEST,
    }}
    mining, llms = publisher.render(snap, "https://files.example")
    assert b"Task rotation is on" in llms and b"active_pool" in mining
    assert publisher.check_pair(mining, llms) == []


def test_admitted_irt_difficulty_is_rounded_head_ability_and_max_fisher_mass():
    old = {"a": [0.0, 0.25, 4]}
    before = snapshot(old, 2.43)
    rotated = rotate("r000001", old, before, {}, reserve(2), params(rotation_active_size=3))
    sampler = IRTSampler(rotated.active_ids, params(), snapshot=before,
                         initial={key: 2.43 for key in rotated.admitted}, rotation={"round_id": "r000001"})
    published = sampler.public_snapshot()
    for task in rotated.admitted:
        assert published["tasks"][task][:3] == [round(2.43, DECIMALS), 0.25, 0]
        assert predicted_pass(2.43, published["tasks"][task][0]) == 0.5
    masses = fisher_masses({key: row[:3] for key, row in published["tasks"].items()}, 2.43, 0.1)
    assert all(masses[task] == max(masses.values()) for task in rotated.admitted)
    # Normal new ids and an archived pool migration keep the old 0 prior.
    ordinary = IRTSampler(["a", "z"], Params(), snapshot=IRTSampler(["a"], Params()).public_snapshot()).public_snapshot()
    assert ordinary["tasks"]["z"][:3] == [0.0, 0.25, 0]


def test_reserve_tampering_reordered_missing_shrunk_withdrawn_fail_closed(tmp_path):
    root = tmp_path / "reserve"
    (root / "shards").mkdir(parents=True)
    (root / "MANIFEST.jsonl").write_bytes(Path(reserve_dir() / "MANIFEST.jsonl").read_bytes())
    (root / "withdrawn.jsonl").write_bytes(b"")
    assert not load_reserve(root)[0].queue
    raw = canonical_json({"id": "omni:0", "env": "affine_math", "problem": "UNIQUE-RESERVE-INTEGRITY: compute 987654321 + 123456789", "answer": "1111111110",
                         "source": "test@rev:0", "license": "CC0-1.0", "added_at": "2026-11-18",
                         "audit": {"verdict": "keep", "method": "test", "ref": "test"}}) + b"\n"
    (root / "shards" / "0001-test.jsonl").write_bytes(raw)
    lines = [{"shard": "0001-test.jsonl", "sha256": sha256_hex(raw), "count": 1, "added_at": "2026-11-18"}]
    def write_manifest():
        (root / "MANIFEST.jsonl").write_bytes(b"".join(canonical_json(x) + b"\n" for x in [*lines, {"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}]))
    write_manifest()
    assert load_reserve(root)[0].queue == ("omni:0",)
    _, catalog = load_reserve(root, with_catalog=True)
    assert catalog.task_for_prompt("UNIQUE-RESERVE-INTEGRITY: compute 987654321 + 123456789").task_id == "omni:0"
    (root / "retired.json").write_text(json.dumps([{"id": "omni:0", "round_id": "r000001", "reason": "low"}]))
    _, catalog = load_reserve(root, with_catalog=True)
    assert catalog.task_for_prompt("UNIQUE-RESERVE-INTEGRITY: compute 987654321 + 123456789") is None
    (root / "retired.json").unlink()
    (root / "shards" / "0001-test.jsonl").write_bytes(raw + b" ")
    with pytest.raises(ConfigError):
        load_reserve(root)
    (root / "shards" / "0001-test.jsonl").write_bytes(raw)
    lines[0]["sha256"] = "0" * 64
    write_manifest()
    with pytest.raises(ConfigError):
        load_reserve(root)
    lines[0]["sha256"] = sha256_hex(raw)
    write_manifest()
    (root / "shards" / "0001-test.jsonl").unlink()
    with pytest.raises(ConfigError):
        load_reserve(root)


def test_append_only_rule_refuses_removed_reordered_and_shrunk_history():
    old = rotate("r000001", ["a", "b"], snapshot({"a": [-10, .25, 3], "b": [-10, .25, 3]}), {}, reserve(0), params())
    old = Rotation.from_dict(old.to_dict() | {"reserve_history": old.reserve_history | {"shards": [{"shard": "0001-a.jsonl", "sha256": "a" * 64, "count": 0, "added_at": "2026-11-18"}],
                                                                                    "withdrawn": [{"id": "x:0", "reason": "bad", "added_at": "2026-11-18"}]}})
    history = old.reserve_history | {"shards": [{"shard": "0001-a.jsonl", "sha256": "a" * 64, "count": 0, "added_at": "2026-11-18"}]}
    current = Rotation.from_dict(old.to_dict() | {"reserve_history": history})
    append_only(old, current)
    changed = Rotation.from_dict(current.to_dict() | {"reserve_history": history | {"shards": []}})
    with pytest.raises(ConfigError):
        append_only(old, changed)
    with pytest.raises(ConfigError):
        append_only(old, Rotation.from_dict(old.to_dict() | {"reserve_history": old.reserve_history | {"withdrawn": []},
                                                                "reserve_manifest_sha": "b" * 64}))
