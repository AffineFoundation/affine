from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from fractions import Fraction

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json
from sn120_v2.errors import ContractError, VerifyError
from sn120_v2.irt import LIMIT, PRIOR_PRECISION, IRTSampler, draw_from_masses, draw_from_snapshot, predicted_pass
from sn120_v2.params import Params
from sn120_v2.seams import TaskSampler
from sn120_v2.types import HeadVersion

POOL = [f"math:{i}" for i in range(60)]
TRUTH = {task: (-3.0, 0.0, 3.0)[i % 3] for i, task in enumerate(POOL)}  # easy, middle, hard
NEW_TASK = [0.0, PRIOR_PRECISION, 0]


def head_after(parent: HeadVersion, label: str) -> HeadVersion:
    return HeadVersion.make(
        parent_version_id=parent.version_id, base_hash=parent.base_hash, base_fp32_master_digest=parent.base_fp32_master_digest,
        loras=(), round_id=label, applied_at=parent.applied_at + 1,
    )


def play(draw, ability: float, rng: random.Random) -> str:
    return "".join("P" if rng.random() < predicted_pass(ability, TRUTH[task]) else "F" for task in draw.task_ids)


def trained(head: HeadVersion, pool=POOL, rounds: int = 14, n_tasks: int = 40) -> IRTSampler:
    """A sampler that saw ``rounds`` simulated rounds: null M at ability 0 and three LoRAs near it."""
    sampler, rng = IRTSampler(pool), random.Random(7)
    for number in range(rounds):
        draw = sampler.draw(head, f"seed-{number}", n_tasks)
        outcomes = {"null": (play(draw, 0.0, rng),)} | {f"lora-{k}": (play(draw, 0.1 * k, rng),) for k in range(3)}
        sampler.update(head, draw, outcomes)
    return sampler


def published(sampler: IRTSampler) -> dict:
    """The snapshot as a reader gets it: through JSON."""
    return json.loads(canonical_json(sampler.public_snapshot()))


def test_the_irt_sampler_follows_the_seam_and_starts_as_an_equal_draw(build):
    sampler = IRTSampler(POOL)
    assert isinstance(sampler, TaskSampler)
    draw = sampler.draw(build.head(), "seed-1", 15)
    assert len(set(draw.task_ids)) == 15 and draw.weights == (4.0,) * 15
    assert draw.snapshot_id == sampler.snapshot_id() == sampler.public_snapshot()["snapshot_id"]
    assert layout.irt_snapshot(sampler.snapshot_id()).startswith("public/irt/irt-")
    with pytest.raises(ContractError):
        IRTSampler(["a", "a"])


def test_the_draw_is_fixed_by_the_snapshot_and_the_round_seed(build):
    head = build.head()
    first, second = trained(head, rounds=3), trained(head, list(reversed(POOL)), rounds=3)
    before = first.snapshot_id()
    draw = first.draw(head, "seed-x", 20)
    assert draw == second.draw(head, "seed-x", 20) == first.draw(head_after(head, "other"), "seed-x", 20)
    assert draw.task_ids != first.draw(head, "seed-y", 20).task_ids
    assert first.snapshot_id() == before, "a draw never changes the table"


def test_the_draw_can_be_rebuilt_from_the_published_snapshot_with_sha256_and_whole_numbers(build):
    sampler = trained(build.head())
    snapshot, seed, n = published(sampler), "seed-public", 20
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    mass = {task: row[3] for task, row in snapshot["tasks"].items()}
    total = sum(mass.values())
    assert len(set(mass.values())) > 10 and n * max(mass.values()) < total, "the masses are unequal and no task is certain"
    points = [int(sha(f"{seed}|irt-start"), 16) % total + k * total for k in range(n)]
    chosen, edge = [], 0
    for task in sorted(mass, key=lambda task: sha(f"{seed}|irt-order|{task}")):
        if any(edge <= point < edge + n * mass[task] for point in points):
            chosen.append(task)
        edge += n * mass[task]
    chosen.sort(key=lambda task: sha(f"{seed}|draw|{task}"))
    draw = sampler.draw(build.head(), seed, n)
    assert list(draw.task_ids) == chosen and len(chosen) == n
    assert list(draw.weights) == [float(Fraction(total, n * mass[task])) for task in chosen]
    assert draw_from_snapshot(snapshot, seed, n) == draw
    assert snapshot["draw"] is None and published(sampler)["draw"]["round_seed"] == seed and published(sampler)["draw"]["n_tasks"] == n


def test_each_task_is_drawn_as_often_as_its_published_weight_says():
    masses = {f"t{i:02d}": i for i in range(1, 13)}  # total 78; with 4 slots, pi = 4 * mass / 78
    counts: Counter[str] = Counter()
    for seed in range(3000):
        draw = draw_from_masses(masses, "snap", f"seed-{seed}", 4)
        assert len(draw.task_ids) == 4
        assert all(weight == 78 / (4 * masses[task]) for task, weight in zip(draw.task_ids, draw.weights))
        counts.update(draw.task_ids)
    for task, mass in masses.items():
        assert abs(counts[task] / 3000 - 4 * mass / 78) < 0.04, task


def test_a_task_with_a_large_mass_is_certain_and_the_first_tasks_stay_a_fair_sample():
    masses = {"big": 1000} | {f"t{i}": 1 for i in range(9)}
    firsts = set()
    for seed in range(300):
        draw = draw_from_masses(masses, "snap", f"seed-{seed}", 3)
        weights = dict(zip(draw.task_ids, draw.weights))
        assert weights.pop("big") == 1.0 and list(weights.values()) == [4.5, 4.5]
        firsts.add(draw.task_ids[0])
    assert firsts == set(masses), "the evaluation order is random, so an early look is a fair part of the draw"
    whole = draw_from_masses(masses, "snap", "seed", 99)
    assert sorted(whole.task_ids) == sorted(masses) and whole.weights == (1.0,) * 10
    for bad_masses, n in (({}, 3), (masses, 0), ({"t": 0}, 1)):
        with pytest.raises(ContractError):
            draw_from_masses(bad_masses, "snap", "seed", n)


def test_online_updates_learn_difficulty_and_favour_the_middle_band(build):
    snapshot = trained(build.head()).public_snapshot()
    rows = snapshot["tasks"]
    assert all(row[2] >= 3 for row in rows.values()), "every task was seen often enough to leave the exploration share"
    mean = lambda values: sum(values) / len(values)
    easy, middle, hard = ([rows[task] for task in POOL if TRUTH[task] == level] for level in (-3.0, 0.0, 3.0))
    assert mean([r[0] for r in easy]) < mean([r[0] for r in middle]) - 1 < mean([r[0] for r in hard]) - 2
    assert abs(snapshot["head_ability"]) < 0.5
    assert mean([r[3] for r in middle]) > 1.25 * max(mean([r[3] for r in easy]), mean([r[3] for r in hard]))
    assert min(r[3] for r in rows.values()) > 0, "no task is ever excluded"


def test_new_or_barely_seen_tasks_get_the_exploration_share(build):
    old_pool, new_tasks = POOL[:50], POOL[50:]
    snapshot = published(trained(build.head(), old_pool))
    assert all(row[2] >= 3 for row in snapshot["tasks"].values())

    def new_share(explore_frac: float) -> float:
        grown = IRTSampler(POOL, Params(irt_explore_frac=explore_frac), snapshot=snapshot).public_snapshot()
        assert all(grown["tasks"][task][:3] == NEW_TASK for task in new_tasks)
        assert all(grown["tasks"][task][:3] == snapshot["tasks"][task][:3] for task in old_pool), "a new pool keeps what was learned"
        return sum(grown["tasks"][task][3] for task in new_tasks) / grown["mass_total"]

    assert new_share(0.3) > new_share(0.0) + 0.2 and new_share(0.3) > 0.3
    smaller = IRTSampler(POOL[:40], snapshot=snapshot)
    assert sorted(smaller.public_snapshot()["tasks"]) == sorted(POOL[:40]) and smaller.snapshot_id() != snapshot["snapshot_id"]


def test_a_head_change_keeps_difficulties_and_takes_the_new_heads_ability_from_its_own_rollouts(build):
    sampler, head = IRTSampler(POOL), build.head()
    second, folded = head_after(head, "r1"), head_after(head_after(head, "r1"), "fold")
    draw = sampler.draw(head, "seed-1", 30)
    outcomes = {"null": ("PF" * 15,), "lora-good": ("PPPF" * 7 + "PP",), "lora-bad": ("FFFP" * 7 + "FF",)}
    sampler.update(head, draw, outcomes, winner_lora_hash="lora-good", new_head_version_id=second.version_id)
    first = sampler.public_snapshot()
    heads, loras = first["heads"], first["loras"]
    assert first["head_version_id"] == second.version_id and first["rounds"] == 1
    assert heads[second.version_id]["ability"] == loras["lora-good"]["ability"] == first["head_ability"]
    assert loras["lora-good"]["ability"] > heads[head.version_id]["ability"] > loras["lora-bad"]["ability"]
    assert (heads[second.version_id]["rollouts"], heads[second.version_id]["passes"]) == (30, 23)
    assert all(first["tasks"][task][:3] == NEW_TASK for task in set(POOL) - set(draw.task_ids))

    draw_2 = sampler.draw(second, "seed-2", 30)
    sampler.update(second, draw_2, {"null": ("P" * 30,)})
    later = sampler.public_snapshot()
    kept = set(draw.task_ids) - set(draw_2.task_ids)
    assert kept and all(later["tasks"][task][:3] == first["tasks"][task][:3] for task in kept), "difficulties outlive the old head"
    assert later["heads"][head.version_id] == heads[head.version_id]
    assert later["heads"][second.version_id]["rollouts"] == 60 and later["head_ability"] > first["head_ability"]

    sampler.update(folded, sampler.draw(folded, "seed-3", 30), {"null": ("P" * 30,)})
    last = sampler.public_snapshot()
    assert last["head_version_id"] == folded.version_id
    assert last["heads"][folded.version_id]["rollouts"] == 90, "a folded head is the same model: it keeps its record"


def test_a_cap_hit_is_not_a_pass_and_an_infrastructure_error_teaches_nothing(build):
    head = build.head()

    def after(code: str, rounds: int = 1) -> dict:
        sampler = IRTSampler(POOL)
        for number in range(rounds):
            draw = sampler.draw(head, f"seed-{number}", 20)
            sampler.update(head, draw, {"null": (code * 20,), "lora": (code * 20,)})
        return sampler.public_snapshot()

    capped, failed, passed = after("C"), after("F"), after("P")
    assert capped["tasks"] == failed["tasks"] and capped["heads"] == failed["heads"]
    assert capped["head_ability"] < 0 < passed["head_ability"]
    for code in "E-":
        unknown = after(code)
        assert unknown["heads"] == {} and unknown["loras"] == {} and unknown["rounds"] == 1
        assert all(row[:3] == NEW_TASK for row in unknown["tasks"].values())
    extreme = after("P", rounds=40)
    values = [extreme["head_ability"]] + [row[0] for row in extreme["tasks"].values()]
    assert all(abs(value) <= LIMIT for value in values) and canonical_json(extreme), "always finite, always valid JSON"


def test_a_replayed_or_malformed_update_does_not_change_the_table(build):
    sampler, head = IRTSampler(POOL), build.head()
    draw = sampler.draw(head, "seed-1", 20)
    outcomes = {"null": ("PF" * 10,), "lora": ("PPFF" * 5,)}
    once = sampler.update(head, draw, outcomes)
    assert sampler.update(head, draw, outcomes) == once and sampler.public_snapshot()["rounds"] == 1
    bad_calls = [
        dict(outcomes={"null": ("PF" * 9,)}),
        dict(outcomes={"null": ("PX" * 10,)}),
        dict(outcomes=outcomes, winner_lora_hash="lora"),
        dict(outcomes=outcomes, winner_lora_hash="missing", new_head_version_id="head-2"),
    ]
    for call in bad_calls:
        with pytest.raises(ContractError):
            sampler.update(head, draw, **call)
    assert sampler.snapshot_id() == once


def test_the_published_snapshot_is_the_whole_state(build):
    head = build.head()
    sampler = trained(head, rounds=3)
    snapshot = published(sampler)
    twin = IRTSampler(POOL, snapshot=snapshot)
    assert twin.snapshot_id() == sampler.snapshot_id() == snapshot["snapshot_id"]
    draw = sampler.draw(head, "seed-next", 20)
    assert twin.draw(head, "seed-next", 20) == draw
    outcomes = {"null": ("PFFP" * 5,), "lora": ("PPFP" * 5,)}
    assert twin.update(head, draw, outcomes) == sampler.update(head, draw, outcomes) != snapshot["snapshot_id"]
    assert snapshot["head_version_id"] == head.version_id and snapshot["task_columns"][3] == "sampling_mass"
    assert set(snapshot["heads"][head.version_id]) == {"ability", "information", "rollouts", "passes", "round"}

    forged = {**snapshot, "tasks": {**snapshot["tasks"], POOL[0]: [0.0, 1.0, 9, 10**11]}}
    for bad in (forged, {**snapshot, "schema": "other"}, {"snapshot_id": snapshot["snapshot_id"]}):
        with pytest.raises(VerifyError):
            IRTSampler(POOL, snapshot=bad)
        with pytest.raises(VerifyError):
            draw_from_snapshot(bad, "seed", 5)
