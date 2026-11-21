"""Active-rollout admission and terminal null-update judging boundaries."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sn120_v2.admission import MIN_ROLLOUTS, NO_SIGNAL, NULL_UPDATE, Invalid, check_batch, signal_counts
from sn120_v2.canonical import manifest_digest
from sn120_v2.judging import decide
from sn120_v2.params import Recipe
from sn120_v2.types import BoxResult, Draw, EntryResult, HeadVersion, LoRAArtifact, RoundTicket, SCORED, TrainReport, WINNER

HOTKEY = "5MinerSignal"


def batch_file(active: int, *, train: dict | None = None, equal_rewards: bool = False) -> bytes:
    # Each [1, 0] prompt contributes two active rollouts. For an odd target,
    # [0, 1, 2, 5] contributes three: only the reward 2 equals the mean.
    pair_count = active // 2 - (1 if active % 2 else 0)
    rewards_by_group = [[1, 0] for _ in range(pair_count)]
    if active % 2:
        rewards_by_group.append([0, 1, 2, 5])
    if not rewards_by_group:
        rewards_by_group = [[1, 0]]
    groups = [{"prompt": f"signal prompt {group}",
               "rollouts": [{"output": f"reasoning {group}-{index}",
                             "reward": 1 if equal_rewards else reward}
                            for index, reward in enumerate(rewards)]}
              for group, rewards in enumerate(rewards_by_group)]
    body = {
        "version": "affine-batch-v3",
        "hotkey": HOTKEY,
        "groups": groups,
    }
    if train is not None:
        body["train"] = dict(train)
    if active == 1:
        body["train"] = {**body.get("train", {}), "group_size": 2}
    return json.dumps(body).encode()


def rejection(raw: bytes, params) -> Invalid:
    with pytest.raises(Invalid) as caught:
        check_batch(raw, HOTKEY, params)
    return caught.value


def test_63_active_rollouts_are_rejected_and_64_are_admitted(build):
    params = build.params(min_signal_rollouts=64)
    below = rejection(batch_file(63), params)
    assert below.rule == MIN_ROLLOUTS
    assert "only 63 rollouts" in str(below) and "floor is 64" in str(below)
    from sn120_v2.admission import parse_groups
    assert signal_counts(parse_groups(json.loads(batch_file(63)), HOTKEY, 4096, 65536), params.recipe)[0] == 63
    accepted = check_batch(batch_file(64), HOTKEY, params)
    assert accepted.n_rollouts == 64 and accepted.train == {}


def test_minibatches_raise_the_active_rollout_floor_above_the_static_minimum(build):
    params = build.params(min_signal_rollouts=64)
    # 70 active rollouts clear the static floor of 64, not 71 optimizer updates.
    below = rejection(batch_file(70, train={"minibatches": 71}), params)
    assert below.rule == MIN_ROLLOUTS
    assert "only 70 rollouts" in str(below) and "floor is 71" in str(below)
    assert check_batch(batch_file(71, train={"minibatches": 71}), HOTKEY, params).n_rollouts == 72


def test_active_rollout_upper_bound_matches_the_largest_minibatch_setting(build):
    params = build.params(min_signal_rollouts=64)
    groups = [[f"upper-bound prompt {i}", [[f"different output {i}-{j}", float(j)] for j in range(4)]]
          for i in range(2000)]
    raw = json.dumps({"version": "affine-batch-v3", "hotkey": HOTKEY, "train": {"minibatches": 8000},
                      "groups": [{"prompt": prompt, "rollouts": [{"output": output, "reward": reward} for output, reward in rows]}
                                 for prompt, rows in groups]}).encode()
    accepted = check_batch(raw, HOTKEY, params)
    assert accepted.n_rollouts == accepted.n_prompts * 4 == 8000
    assert signal_counts(accepted.groups, params.recipe.for_miner(accepted.train))[0] == 8000


def test_equal_reward_group_has_no_signal_before_minimum_rollout_check(build):
    error = rejection(batch_file(64, equal_rewards=True), build.params(min_signal_rollouts=64))
    assert error.rule == NO_SIGNAL


def test_static_norm_cap_below_judging_floor_is_rejected_at_admission(build):
    params = build.params(min_signal_rollouts=1, min_step_norm=0.02)
    below = rejection(batch_file(1, train={"global_norm_cap": 0.0199}), params)
    assert below.rule == NULL_UPDATE
    assert "0.0199" in str(below) and "0.02" in str(below)
    on_boundary = check_batch(batch_file(1, train={"global_norm_cap": 0.02}), HOTKEY, params)
    assert on_boundary.train == {"global_norm_cap": 0.02, "group_size": 2}
    assert check_batch(batch_file(1), HOTKEY, params).train == {"group_size": 2}, "the default cap remains above the floor"


def test_default_recipe_does_not_change_active_rollout_count(build):
    # A miner that leaves train unset gets the published default recipe. Its
    # single optimizer update must not silently multiply or divide activity.
    params = build.params(min_signal_rollouts=64)
    assert params.recipe.minibatches == 1 and params.recipe.group_size == 0
    explicit = build.params(min_signal_rollouts=64, recipe=Recipe(image_digests=(build.image,), minibatches=1, group_size=0))
    accepted_default = check_batch(batch_file(64), HOTKEY, params)
    accepted_explicit = check_batch(batch_file(64), HOTKEY, explicit)
    assert accepted_default.n_rollouts == accepted_explicit.n_rollouts == 64
    assert signal_counts(accepted_default.groups, params.recipe)[0] == 64
    assert params.recipe.for_miner(accepted_default.train) is params.recipe
    assert rejection(batch_file(63), explicit).rule == MIN_ROLLOUTS


# ---------------------------------------------------------------- Pure judging


def pure_ticket(build, count: int, params):
    # The judging path does not read head content. Pin a lightweight real head for these pure-rule checks.
    ticket = RoundTicket(
        round_id="r000001", lease_id="r000001-l0", box_id="box-0",
        head=build.head(), params=params,
        draw=Draw("snap-0", "seed-0", tuple(f"t{i:03d}" for i in range(params.n_tasks)), (1.0,) * params.n_tasks),
        entries=tuple(build.entry(position) for position in range(count)),
        issued_at=1000, expires_at=5000,
    )
    object.__setattr__(ticket, "_sealed", "pure-ticket-digest")
    return ticket


def pure_result(build, ticket, null: str, outcomes: dict[int, str]):
    entries = []
    for seat in ticket.entries:
        files = {"adapter_config.json": build.h("config"), "adapter_model.safetensors": build.h(f"weights-{seat.queue_pos}")}
        artifact = LoRAArtifact(
            lora_hash=manifest_digest(files),
            files=files, sizes={name: 1 for name in files}, prefix=f"pure/{seat.queue_pos:03d}",
            head_version_id=ticket.head.version_id, batch_sha256=seat.batch.batch_sha256,
            recipe_id=ticket.params.recipe_id, rank=ticket.params.recipe.rank, alpha=ticket.params.recipe.alpha,
            target_modules=ticket.params.recipe.target_modules,
            report=TrainReport(seed=artifact_seed(ticket, seat), steps=1, loss=0.5, step_norm=0.1,
                               tokens=100, wall_s=1.0, gpu_class="cpu", image_digest=build.image),
        )
        entries.append(EntryResult(seat.queue_pos, SCORED, lora=artifact, outcomes=(outcomes[seat.queue_pos],)))
    return BoxResult(
        round_id=ticket.round_id, lease_id=ticket.lease_id, box_id=ticket.box_id,
        ticket_digest=ticket.digest(), head_version_id=ticket.head.version_id, image_digest=build.image,
        null_outcomes=(null,), entries=tuple(entries), started_at=1001, finished_at=1500,
    )


def artifact_seed(ticket, seat):
    from sn120_v2.sampling import train_seed
    return train_seed(ticket.draw.round_seed, seat.batch.batch_sha256, ticket.params.train_seed_scope)


def judged_with_norm(build, norm: float):
    null = "P" * 10 + "F" * 90
    strong = "P" * 30 + "F" * 70
    params = build.params(n_tasks=100, min_scored_tasks=2, min_step_norm=0.02)
    ticket = pure_ticket(build, 1, params)
    result = pure_result(build, ticket, null, {0: strong})
    scored = result.entries[0]
    artifact = replace(scored.lora, report=replace(scored.lora.report, step_norm=norm))
    result = replace(result, entries=(replace(scored, lora=artifact),))
    return decide([(ticket, result)])


def test_judging_null_update_is_terminal_at_0199_and_boundary_is_judged(build):
    below = judged_with_norm(build, 0.0199)
    assert below.winner_queue_pos is None
    assert below.entries[0].status == NULL_UPDATE
    assert below.entries[0].charged and below.entries[0].score is None
    assert "0.0199" in below.entries[0].reason

    on_boundary = judged_with_norm(build, 0.02)
    assert on_boundary.winner_queue_pos == 0
    assert on_boundary.entries[0].status == WINNER
    assert on_boundary.entries[0].score is not None and on_boundary.entries[0].score.eligible


def test_judging_nan_update_is_terminal_not_a_score(build):
    decision = judged_with_norm(build, float("nan"))
    assert decision.winner_queue_pos is None
    assert decision.entries[0].status == NULL_UPDATE
    assert decision.entries[0].charged and decision.entries[0].score is None
    assert "non-finite" in decision.entries[0].reason or "nan" in decision.entries[0].reason


def test_best_scoring_null_update_cannot_beat_next_real_candidate(build):
    null = "P" * 10 + "F" * 90
    best = "P" * 40 + "F" * 60
    next_best = "P" * 30 + "F" * 70
    params = build.params(n_tasks=100, min_scored_tasks=2, min_step_norm=0.02)
    ticket = pure_ticket(build, 2, params)
    result = pure_result(build, ticket, null, {0: best, 1: next_best})
    nulled, real = result.entries
    nulled = replace(nulled, lora=replace(nulled.lora, report=replace(nulled.lora.report, step_norm=0.0)))
    result = replace(result, entries=(nulled, real))
    decision = decide([(ticket, result)])
    assert decision.winner_queue_pos == 1
    assert [row.status for row in decision.entries] == [NULL_UPDATE, WINNER]
    assert decision.entries[0].score is None and decision.entries[1].score is not None
    assert decision.entries[0].charged and decision.entries[1].charged
