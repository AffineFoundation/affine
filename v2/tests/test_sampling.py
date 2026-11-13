from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from sn120_v2.errors import ContractError
from sn120_v2.sampling import UniformSampler, make_round_seed, rollout_seed, train_seed, uniform_draw
from sn120_v2.seams import TaskSampler

POOL = [f"math:{i}" for i in range(40)]


def test_rollout_seed_is_sha256_of_seed_task_and_sample():
    expected = int(hashlib.sha256(b"seed-1|math:7|0").hexdigest(), 16) >> 193
    assert rollout_seed("seed-1", "math:7", 0) == expected
    assert 0 <= expected < 2**63


def test_rollout_seed_depends_on_each_input_and_on_nothing_else():
    base = rollout_seed("seed-1", "math:7", 0)
    assert rollout_seed("seed-1", "math:7", 0) == base
    assert len({base, rollout_seed("seed-2", "math:7", 0), rollout_seed("seed-1", "math:8", 0), rollout_seed("seed-1", "math:7", 1)}) == 4


def test_train_seed_is_fixed_by_the_round_seed_and_the_batch():
    assert train_seed("seed-1", "a" * 64) == train_seed("seed-1", "a" * 64)
    assert train_seed("seed-1", "a" * 64) != train_seed("seed-1", "b" * 64)


def test_round_seed_changes_with_round_head_and_entropy():
    base = make_round_seed("r000001", "head-a", "block-hash")
    assert make_round_seed("r000001", "head-a", "block-hash") == base
    others = {make_round_seed("r000002", "head-a", "block-hash"), make_round_seed("r000001", "head-b", "block-hash"), make_round_seed("r000001", "head-a", "other")}
    assert base not in others and len(others) == 3
    with pytest.raises(ContractError):
        make_round_seed("r000001", "head-a", "")


def test_the_draw_can_be_rebuilt_from_the_public_seed_with_sha256_alone():
    draw = uniform_draw(POOL, "snap", "seed-1", 10)
    rebuilt = sorted(POOL, key=lambda task: hashlib.sha256(f"seed-1|draw|{task}".encode()).hexdigest())[:10]
    assert list(draw.task_ids) == rebuilt
    assert draw.round_seed == "seed-1" and draw.snapshot_id == "snap"


def test_the_draw_does_not_depend_on_the_order_of_the_pool():
    assert uniform_draw(POOL, "snap", "seed-1", 10) == uniform_draw(list(reversed(POOL)), "snap", "seed-1", 10)


def test_another_seed_gives_another_draw():
    assert uniform_draw(POOL, "snap", "seed-1", 10).task_ids != uniform_draw(POOL, "snap", "seed-2", 10).task_ids


def test_sampling_weight_is_pool_size_over_draw_size():
    draw = uniform_draw(POOL, "snap", "seed-1", 10)
    assert draw.weights == (4.0,) * 10


def test_a_draw_larger_than_the_pool_takes_the_whole_pool():
    draw = uniform_draw(POOL, "snap", "seed-1", 1000)
    assert sorted(draw.task_ids) == sorted(POOL) and draw.weights == (1.0,) * 40


def test_a_draw_needs_a_pool_and_a_size():
    with pytest.raises(ContractError):
        uniform_draw([], "snap", "seed-1", 5)
    with pytest.raises(ContractError):
        uniform_draw(POOL, "snap", "seed-1", 0)


def test_every_task_is_drawn_about_equally_often():
    counts = Counter(task for seed in range(400) for task in uniform_draw(POOL, "snap", f"seed-{seed}", 10).task_ids)
    assert set(counts) == set(POOL)
    assert all(60 <= counts[task] <= 140 for task in POOL), "expected about 100 draws per task"


def test_the_first_tasks_of_a_draw_are_also_a_fair_sample():
    firsts = Counter(uniform_draw(POOL, "snap", f"seed-{seed}", 10).task_ids[0] for seed in range(800))
    assert len(firsts) == len(POOL)


def test_uniform_sampler_follows_the_task_sampler_seam(build):
    sampler = UniformSampler(POOL)
    assert isinstance(sampler, TaskSampler)
    draw = sampler.draw(build.head(), "seed-1", 10)
    assert draw == uniform_draw(POOL, sampler.snapshot_id(), "seed-1", 10)
    assert sampler.update(build.head(), draw, {"null": ("P" * 10,)}) == sampler.snapshot_id()
    assert sampler.public_snapshot()["snapshot_id"] == sampler.snapshot_id()


def test_uniform_sampler_snapshot_names_the_pool():
    assert UniformSampler(POOL).snapshot_id() == UniformSampler(list(reversed(POOL))).snapshot_id()
    assert UniformSampler(POOL).snapshot_id() != UniformSampler(POOL[:-1]).snapshot_id()
    with pytest.raises(ContractError):
        UniformSampler(["a", "a"])
    with pytest.raises(ContractError):
        UniformSampler([])
