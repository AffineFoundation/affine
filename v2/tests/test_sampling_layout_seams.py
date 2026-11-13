from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from sn120_v2 import errors, layout, seams
from sn120_v2.errors import ContractError, V2Error
from sn120_v2.sampling import UniformSampler, make_round_seed, rollout_seed, train_seed, uniform_draw

POOL = [f"math:{i}" for i in range(40)]


def test_rollout_seed_is_sha256_of_seed_task_and_sample_and_of_nothing_else():
    base = rollout_seed("seed-1", "math:7", 0)
    assert base == int(hashlib.sha256(b"seed-1|math:7|0").hexdigest(), 16) >> 193 and 0 <= base < 2**63
    assert len({base, rollout_seed("seed-2", "math:7", 0), rollout_seed("seed-1", "math:8", 0), rollout_seed("seed-1", "math:7", 1)}) == 4
    assert train_seed("seed-1", "a" * 64) == train_seed("seed-1", "a" * 64) != train_seed("seed-1", "b" * 64)


def test_round_seed_changes_with_round_head_and_entropy():
    base = make_round_seed("r000001", "head-a", "block-hash")
    others = {make_round_seed("r000002", "head-a", "block-hash"), make_round_seed("r000001", "head-b", "block-hash"), make_round_seed("r000001", "head-a", "x")}
    assert make_round_seed("r000001", "head-a", "block-hash") == base and base not in others and len(others) == 3
    with pytest.raises(ContractError):
        make_round_seed("r000001", "head-a", "")


def test_the_draw_can_be_rebuilt_from_the_public_seed_with_sha256_alone():
    draw = uniform_draw(POOL, "snap", "seed-1", 10)
    rebuilt = sorted(POOL, key=lambda task: hashlib.sha256(f"seed-1|draw|{task}".encode()).hexdigest())[:10]
    assert list(draw.task_ids) == rebuilt and draw.weights == (4.0,) * 10, "weight is pool size over draw size"
    assert uniform_draw(list(reversed(POOL)), "snap", "seed-1", 10) == draw
    assert uniform_draw(POOL, "snap", "seed-2", 10).task_ids != draw.task_ids
    assert sorted(uniform_draw(POOL, "snap", "seed-1", 1000).task_ids) == sorted(POOL)
    for pool, size in (([], 5), (POOL, 0)):
        with pytest.raises(ContractError):
            uniform_draw(pool, "snap", "seed-1", size)


def test_every_task_is_drawn_about_equally_often():
    counts = Counter(task for seed in range(400) for task in uniform_draw(POOL, "snap", f"seed-{seed}", 10).task_ids)
    assert set(counts) == set(POOL) and all(60 <= counts[task] <= 140 for task in POOL), "expected about 100 draws per task"


def test_uniform_sampler_follows_the_task_sampler_seam(build):
    sampler = UniformSampler(POOL)
    assert isinstance(sampler, seams.TaskSampler)
    draw = sampler.draw(build.head(), "seed-1", 10)
    assert draw == uniform_draw(POOL, sampler.snapshot_id(), "seed-1", 10)
    assert sampler.update(build.head(), draw, {"null": ("P" * 10,)}) == sampler.snapshot_id() == sampler.public_snapshot()["snapshot_id"]
    assert UniformSampler(list(reversed(POOL))).snapshot_id() == sampler.snapshot_id() != UniformSampler(POOL[:-1]).snapshot_id()


def test_store_keys_keep_their_zones(build):
    assert layout.miner_object("5Hotkey") == "private/ff/v1/miners/5Hotkey/batch.json"
    pending = [layout.pending_result("r000001", "r000001-l0"), layout.pending_lora_prefix("r000001", "r000001-l0", 3), layout.pending_fold_prefix("abc")]
    assert all(key.startswith("pending/") for key in pending) and pending[1].endswith("/loras/003")
    private = [layout.frozen_batch("r000001", "5Hotkey"), layout.ticket("r000001", "r000001-l0"), layout.sealed_result("r000001", "r000001-l0")]
    assert all(key.startswith("private/") for key in private)
    base, master = layout.public_base_prefix("a" * 64), layout.public_master_prefix("a" * 64)
    assert not master.startswith(base) and not base.startswith(master), "the fp32 master never shares a prefix with the base"
    assert layout.public_base_file("a" * 64, "model.safetensors") == f"public/bases/{'a' * 64}/model.safetensors"
    assert (layout.CURRENT, layout.BENCHMARKS) == ("public/current.json", "public/benchmarks.json")
    for bad in ("", ".", "..", "../x", "a/b", "a b", ".hidden", "x\ny"):
        with pytest.raises(ContractError):
            layout.frozen_batch("r000001", bad)


def test_every_seam_is_a_runtime_checkable_protocol_and_every_error_shares_one_base():
    names = ("ObjectStore", "Presigner", "Admission", "TaskSampler", "Trainer", "Evaluator", "Folder", "RoundRunner", "Publisher", "ChainReader", "WeightSetter")
    for name in names:
        assert getattr(getattr(seams, name), "_is_protocol", False) and not isinstance(object(), getattr(seams, name)), name

    class OneBox:
        def run(self, ticket):
            raise NotImplementedError

    assert isinstance(OneBox(), seams.RoundRunner) and not isinstance(OneBox(), seams.WeightSetter)
    for name in ("ConfigError", "InfraError", "BatchRejected", "VerifyError", "HeadMismatchError", "LeaseError", "ContractError", "FencedError", "JournalCorrupt"):
        assert issubclass(getattr(errors, name), V2Error), name
