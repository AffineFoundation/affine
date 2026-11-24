"""Paired crown confirmation: matched sample means, deferred payout, replay and public evidence."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.cli import changed_params
from sn120_v2.controller import Controller
from sn120_v2.envs.math_pool import MathPool, Task
from sn120_v2.errors import ContractError, InfraError
from sn120_v2.gpu.plan import code_digest
from sn120_v2.gpu.solo import SoloRunner, SoloScheduler, SoloSettings
from sn120_v2.journal import JOURNAL_FILE, JournalRecord, read_journal
from sn120_v2.judging import TIE_TOLERANCE, check_result, duplicate_decision, score_confirmation
from sn120_v2.params import Params, load_params
from sn120_v2.payout import paid_set, weights
from sn120_v2.public import round_report, snapshot
from sn120_v2.publisher import check_pair, contract, render
from sn120_v2.sampling import UniformSampler, rollout_seed
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import load
from sn120_v2.state import CONFIRMATION_RESOLVED, HEAD_APPLIED, HEAD_FOLDED, ROUND_CLOSED, ROUND_OPENED, ROUND_SETTLED
from sn120_v2.store import get_record, put_record
from sn120_v2.types import DUPLICATE, FAIL, NOT_RUN, NULL_MODEL, PASS, BoxResult, King, RoundDecision, RoundTicket


def confirming(tmp_path, **changes) -> SimWorld:    return SimWorld(tmp_path, sim_params(confirm_next_round=True, **changes))


def crown(world: SimWorld, uid: int = 1, quality: float = 0.3):
    world.register(uid, quality)
    record = world.round()
    assert record.outcome == "applied" and world.controller.state.pending_king is not None
    assert world.controller.state.pending_king.new_version_id in {head.version_id for head in world.controller.state.heads}
    assert world.controller.state.head.parent_version_id == world.controller.state.pending_king.new_version_id
    assert world.controller.state.head.folded and world.controller.state.head.loras == ()
    assert [king.uid for king in world.controller.state.kings] == []
    return record, world.controller.state.pending_king


def test_confirm_production_params_exact_identity_and_effective_diffs():
    ops = Path(__file__).parents[1] / "ops"
    live = load_params(ops / "train64k_params_t3600_prune_open.json")
    kl8k = load_params(ops / "train64k_params_t3600_prune_open_kl8k.json")
    new_path = ops / "train64k_params_t3600_prune_open_kl8k_confirm.json"
    new = load_params(new_path)
    assert live.params_hash == "073ef680cf338de5e085100704e60f80ba96657d6d4e19d12995e4cb971ef6f8"
    assert kl8k.params_hash == "dcab97d47d2f4b02ef7a14d5d8786a0f5a14885ddbf2634845435612fb9b4b1e"
    assert new.params_hash == "46791df11c9b8b55a83bfed44f9c579900aa2253b3ae6716c667e2f01582f876"
    assert new.recipe_id == "3631cb3accac35c785e946b95870245ef54b2fd37429572782490ad92e12558e"
    assert Params.from_dict(new.to_dict()) == new
    assert json.loads(new_path.read_text())["kl_rollouts"] == 128  # explicit in the file; omitted at the canonical default
    assert new.confirm_next_round is True and new.min_kl_seq == 22
    assert (new.kl_max_tokens, new.kl_rollouts, new.kl_time_limit_s) == (8192, 128, 1200)
    assert new.batch_cap == 6 and new.n_gpus == 8
    assert new.recipe.code_digest == "ab2922c88558afd37e95938773a7e148e88c1a6ec543f96c1ce05483fd1b7941"
    assert code_digest() == "cd207930a14ac2f653744dcd48ba96ee263bc6803ca4e573902786411a2b70e4"
    assert new.recipe.image_digests == live.recipe.image_digests == kl8k.recipe.image_digests == (
        "9a4b16ff282b21baf44350e54c852ed59586b9efb6fe09e5cd2f1e288f4c108a",
    )
    assert changed_params(live, new) == {
        "confirm_next_round": {"before": False, "after": True},
        "min_kl_seq": {"before": 0.0, "after": 22.0},
        "kl_max_tokens": {"before": 0, "after": 8192},
        "recipe.code_digest": {
            "before": "22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1",
            "after": new.recipe.code_digest,
        },
    }
    assert changed_params(kl8k, new) == {
        "confirm_next_round": {"before": False, "after": True},
        "min_kl_seq": {"before": 0.000225, "after": 22.0},
        "recipe.code_digest": {
            "before": "5e3aef5849ef6903c9fc1ab67f89eddbd2ea2cac9748621fc42b75224bf58419",
            "after": new.recipe.code_digest,
        },
    }


def score_examples():
    return (1.0, 2.0, 3.0, 4.0), ("PPPP", "PPFF"), ("PPPF", "PPFF")


def test_score_confirmation_uses_matched_sample_means_and_paired_se():
    # Per-task differences: (0, 0, 0.5, 0). delta = 2/10. The existing weighted
    # estimator gives spread sum((w * (d - delta))**2) = 2.0.
    scored = score_confirmation(*score_examples())
    assert scored["confirmed"] and scored["delta"] == pytest.approx(0.2)
    assert scored["new_pass_rate"] == pytest.approx(0.65)
    assert scored["old_pass_rate"] == pytest.approx(0.45)
    assert scored["se"] == pytest.approx((4 / 3 * 2.0) ** 0.5 / 10)
    assert scored["z"] == pytest.approx(scored["delta"] / scored["se"])


def test_score_confirmation_exact_tie_and_loss_fail_without_a_z_threshold():
    for outcomes in ((PASS * 4, PASS * 4), (FAIL * 4, PASS * 4)):
        rejected = score_confirmation((1.0,) * 4, outcomes[:1], outcomes[1:])
        assert not rejected["confirmed"] and rejected["delta"] <= TIE_TOLERANCE
        assert "TIE_TOLERANCE" in rejected["reason"]
    tied = score_confirmation((1.0,) * 4, ("PPFF", "FFPP"), ("PFPF", "FPFP"))
    assert tied["delta"] == 0 and tied["se"] == 0 and not tied["confirmed"]
    with pytest.raises(ContractError, match="same nonzero number"):
        score_confirmation((1.0,), (PASS, PASS), (FAIL,))
    with pytest.raises(ContractError, match="cover different numbers|full fresh draw"):
        score_confirmation((1.0,) * 4, (PASS * 3 + NOT_RUN,), (FAIL * 4,))
    assert TIE_TOLERANCE == 1e-12
    # Sample 0 ties, but the later matched samples make the head win.
    assert score_confirmation((1.0,) * 4, ("PPPP", "PPFF"), ("PPPP", "FFFF"))["delta"] == 0.25


def test_clear_win_confirms_later_without_backpaying_the_confirming_round(tmp_path):
    world = confirming(tmp_path)
    first, pending = crown(world)
    before = paid_set(world.controller.state.kings, world.controller.state.null_rounds, world.params)
    world.evaluator.parent_quality = -0.4
    world.register(2, -1.0)  # no second crown: isolates the first crown's settlement
    second = world.round()
    confirmation = second.confirmation
    assert confirmation.confirmed and confirmation.pending_round_id == first.round_id
    assert confirmation.parent_version_id == pending.parent_version_id
    assert confirmation.new_version_id == pending.new_version_id
    assert world.controller.state.pending_king is None
    assert [king.head_version_id for king in world.controller.state.kings] == [pending.new_version_id]
    assert world.controller.state.kings[0].applied_at == pending.applied_at < second.opened_at
    assert paid_set(world.controller.state.kings, world.controller.state.null_rounds, world.params)[0].round_id == first.round_id
    assert load(world.root / JOURNAL_FILE).kings == world.controller.state.kings
    report = json.loads(world.store.get(layout.round_file(second.round_id)))
    assert report["confirmation"]["confirmed"] and report["parent_outcomes"]
    assert len(report["parent_outcomes"]) == world.params.null_samples
    assert all(len(text) == second.n_tasks for text in report["parent_outcomes"])
    assert set(report["confirmation"]) == {
        "parent_version_id", "pending_round_id", "pending_uid", "new_version_id", "delta", "se", "z",
        "new_pass_rate", "old_pass_rate", "new_head_pass_rate", "old_head_pass_rate", "n_tasks", "excluded_training_prompts", "confirmed", "paid", "payout_status", "reason",
    }
    assert report["confirmation"]["paid"] is True and report["confirmation"]["payout_status"] == "paid"
    public_doc = snapshot(world.controller.state, world.params, world.clock())
    assert public_doc["round"] is None
    assert public_doc["history"][-1]["confirmation"] == report["confirmation"]


def test_loss_tie_and_missing_parent_leave_installed_head_unpaid(tmp_path):
    for parent_quality, remove_manifest in ((0.5, False), (None, False), (0.0, True)):
        world = confirming(tmp_path / f"{parent_quality}-{remove_manifest}")
        first, pending = crown(world)
        if parent_quality is not None:
            world.evaluator.parent_quality = parent_quality
        if remove_manifest:
            world.controller.state.manifests.pop(pending.parent_version_id)
        world.register(2, -1.0)
        second = world.round()
        assert second.confirmation is not None and not second.confirmation.confirmed
        assert world.controller.state.pending_king is None and not world.controller.state.kings
        assert world.controller.state.head.parent_version_id == pending.new_version_id
        if remove_manifest:
            assert "parent manifest is unavailable" in second.confirmation.reason
        report = json.loads(world.store.get(layout.round_file(second.round_id)))
        assert report["confirmation"]["paid"] is False and report["confirmation"]["payout_status"] == "unpaid"
        if remove_manifest:
            assert report["confirmation"]["n_tasks"] == 0
            assert report["confirmation"]["new_head_pass_rate"] == report["confirmation"]["old_head_pass_rate"] == 0
            assert report["confirmation"]["delta"] == 0
        else:
            assert report["confirmation"]["n_tasks"] == second.n_tasks
        doc = snapshot(world.controller.state, world.params, world.clock())
        assert doc["history"][-1]["confirmation"] == report["confirmation"]
        replayed = load(world.root / JOURNAL_FILE)
        assert replayed.head == world.controller.state.head and replayed.kings == []


def test_back_to_back_resolution_and_new_pending_survive_replay(tmp_path):
    world = confirming(tmp_path)
    _, old = crown(world, 1)
    world.evaluator.parent_quality = -0.5
    world.register(2, 0.3)
    second = world.round()
    assert second.confirmation.confirmed and world.controller.state.pending_king.uid == 2
    new = world.controller.state.pending_king
    assert new.crown_round_id == second.round_id and new.parent_version_id == second.head_version_id
    assert [king.uid for king in world.controller.state.kings] == [1]
    replayed = load(world.root / JOURNAL_FILE)
    assert replayed.pending_king == new and replayed.rounds[-1].confirmation == second.confirmation
    world.evaluator.parent_quality = -0.5
    world.register(3, -1.0)
    third = world.round()
    assert third.confirmation.confirmed and [king.uid for king in world.controller.state.kings] == [1, 2]


def test_disabled_mode_is_exactly_immediate_apply_and_pay(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.3)
    record = world.round()
    assert record.parent is None and record.confirmation is None
    assert world.controller.state.pending_king is None and [king.uid for king in world.controller.state.kings] == [1]
    assert world.evaluator.closed == 1
    world.reopen()
    world.register(2, 0.3)
    assert world.round().parent is None


def test_nonconfirming_rollouts_and_outcomes_are_identical_with_confirmation_enabled(tmp_path):
    worlds = [SimWorld(tmp_path / name, sim_params(confirm_next_round=enabled, n_tasks=8, min_scored_tasks=2,
                                                  looks=(1.0,), null_samples_per_task=2))
              for name, enabled in (("off", False), ("on", True))]
    requests, outcomes = [], []
    for world in worlds:
        world.register(1, 0.0)
        seen, chunks = [], []
        original = world.evaluator.evaluate
        def evaluate(request, *, original=original, seen=seen, chunks=chunks):
            seen.append(request)
            chunk = original(request)
            chunks.append(chunk)
            return chunk
        world.evaluator.evaluate = evaluate
        world.evaluator.score_parent = lambda *args, **kwargs: pytest.fail("a non-confirming round scored a parent")
        record = world.round()
        sealed = json.loads(world.store.get(layout.sealed_result(record.round_id, record.leases[0].lease_id)))
        requests.append((seen, chunks))
        outcomes.append(sealed | {"ticket_digest": "the pinned params differ by design"})
    for request1, request2 in zip(requests[0][0], requests[1][0]):
        assert request2 == dataclasses.replace(request1, params=dataclasses.replace(
            request1.params, confirm_next_round=True, _legacy_fields=request2.params._legacy_fields))
    assert requests[0][1] == requests[1][1]
    assert outcomes[0] == outcomes[1]
    assert [sum(len(text) for sample in chunk.outcomes.values() for text in sample) for chunk in requests[0][1]] == [24]


def test_disabling_while_pending_expires_old_crown_before_immediate_new_king(tmp_path):
    world = confirming(tmp_path)
    _, old = crown(world, 1)
    world.controller.set_params(dataclasses.replace(world.params, confirm_next_round=False), world.clock())
    world.reopen()
    world.register(2, 0.3)
    second = world.round()
    assert second.parent is None and not second.confirmation.confirmed
    assert second.confirmation.reason.startswith("no_decision")
    assert world.controller.state.pending_king is None
    assert [king.uid for king in world.controller.state.kings] == [2]
    assert all(king.head_version_id != old.new_version_id for king in world.controller.state.kings)
    assert load(world.root / JOURNAL_FILE).kings == world.controller.state.kings


def test_parent_infrastructure_failure_is_not_a_round_failure(tmp_path):
    world = confirming(tmp_path)
    crown(world)
    def broken(ticket, deadline=0):
        raise InfraError("parent base missing")
    world.evaluator.score_parent = broken
    world.register(2, -1.0)
    record = world.round()
    assert record.outcome == "null" and not record.confirmation.confirmed
    assert "parent scoring failed" in record.confirmation.reason
    assert not world.controller.state.kings
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    assert report["parent_outcomes"] is None and "parent base missing" in report["parent_error"]


def test_no_result_pending_expiry_still_publishes_parent_evidence(tmp_path):
    world = confirming(tmp_path)
    crown(world)
    world.register(2, 0.0)
    world.trainer.fail_hotkeys = {"hk2"}
    record = world.round()
    assert record.outcome == "no_decision" and record.confirmation is not None
    assert not record.confirmation.confirmed and not world.controller.state.kings
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    assert report["parent_outcomes"] is None and report["parent_error"]


def test_pending_and_failed_crowns_do_not_change_the_ten_uid_table(tmp_path):
    params = sim_params(null_prune_from_round="")
    registered = {uid: f"hk{uid}" for uid in range(1, 12)}
    old = [King(uid, f"hk{uid}", f"r{uid:06d}", f"head-{uid}", uid) for uid in range(1, 11)]
    expected = weights(old, registered, params)
    pending_world = confirming(tmp_path)
    crown(pending_world, 11)
    assert weights(old, registered, params) == expected
    assert not paid_set(pending_world.controller.state.kings, pending_world.controller.state.null_rounds, pending_world.params)
    pending_world.evaluator.parent_quality = 0.5
    pending_world.register(12, -1.0)
    assert not pending_world.round().confirmation.confirmed
    assert weights(old, registered, params) == expected
    pending_world.evaluator.parent_quality = -0.5
    # A fresh crown followed by confirmation enters only then; there is no pending-table shortcut.
    other = confirming(tmp_path / "confirmed")
    crown(other, 11)
    other.evaluator.parent_quality = -0.5
    other.register(12, -1.0)
    other.round()
    assert [king.uid for king in other.controller.state.kings] == [11]
    assert 11 in weights(old + other.controller.state.kings, registered, params)
    assert 1 not in weights(old + other.controller.state.kings, registered, params)


def test_prompt_exclusion_is_shared_and_unmapped_prompts_do_not_block(tmp_path):
    world = confirming(tmp_path)
    first, pending = crown(world)
    tasks = tuple(Task(f"sim:{i}", "affine_math", f"problem {i}", "1") for i in range(400))
    prompts = [task.batch_prompt() for task in tasks[:12]]
    body = {"version": "affine-batch-v3", "hotkey": "hk1", "groups": [{"prompt": prompt} for prompt in prompts] + [{"prompt": "unmapped"}]}
    raw = canonical_json(body)
    world.store.put(layout.winner_batch(first.round_id), raw)
    pending = dataclasses.replace(pending, batch_sha256=sha256_hex(raw))
    world.controller.state.pending_king = pending  # test-only setup; journal events are exercised by the other tests
    world.controller.prompt_pool = MathPool(tasks, "test")
    calls = []
    real = world.controller.sampler.draw
    def observed(head, seed, n, exclude=frozenset()):
        calls.append((head, seed, n, exclude))
        return real(head, seed, n, exclude)
    world.controller.sampler.draw = observed
    world.evaluator.parent_quality = -0.5
    world.register(2, 0.3)
    second = world.round()
    assert len(calls) == 1 and calls[0][3] == frozenset(task.task_id for task in tasks[:12])
    assert second.excluded_training_prompts == 12
    assert not set(second.confirmation.to_dict()) & {"prompt", "groups"}
    ticket = json.loads(world.store.get(layout.ticket(second.round_id, second.leases[0].lease_id)))
    assert not set(ticket["draw"]["task_ids"]) & calls[0][3]
    assert second.confirmation.excluded_training_prompts == 12
    assert world.controller.state.pending_king.crown_round_id == second.round_id
    report = world.store.get(layout.round_file(second.round_id)).decode()
    assert "problem 0" not in report and "private/" not in report


def test_no_pending_draw_is_byte_identical_and_parent_slot_is_conditional(tmp_path):
    off = SimWorld(tmp_path / "off", sim_params())
    off.register(1, 0.3)
    first = off.round()
    assert first.parent is None
    assert "parent" not in json.loads(off.store.get(layout.ticket(first.round_id, first.leases[0].lease_id)))
    head, seed = off.controller.state.head, "seed"
    assert off.controller.sampler.draw(head, seed, 8).to_json() == UniformSampler(off.pool).draw(head, seed, 8).to_json()
    on = confirming(tmp_path / "on")
    assert on.controller._parent_slot() is None
    crown(on)
    assert on.controller._parent_slot().pending == on.controller.state.pending_king


def test_parent_result_sealing_requires_null_samples_and_a_full_fresh_null(build):
    params = build.params(confirm_next_round=True, null_samples_per_task=2)
    ticket = dataclasses.replace(build.ticket(n_entries=1, params=params), parent=build.parent(build.head()))
    result = build.result(ticket, (PASS * 8, FAIL * 8), {0: PASS * 8}, parent_outcomes=(FAIL * 8, FAIL * 8))
    check_result(ticket, result)
    scored = score_confirmation(ticket.draw.weights, result.null_outcomes, result.parent_outcomes)
    assert scored["delta"] == 0.5 and scored["confirmed"]
    for bad in (
        dataclasses.replace(result, parent_outcomes=(FAIL * 8,)),
        dataclasses.replace(result, parent_outcomes=(FAIL * 8, FAIL * 8, FAIL * 8)),
        dataclasses.replace(result, parent_outcomes=(FAIL * 7 + NOT_RUN, FAIL * 8)),
        dataclasses.replace(result, null_outcomes=(PASS * 7 + NOT_RUN, FAIL * 8)),
    ):
        with pytest.raises(ContractError):
            check_result(ticket, bad)
    with pytest.raises(ContractError, match="without a pinned parent"):
        without_parent = dataclasses.replace(ticket, parent=None)
        check_result(without_parent, dataclasses.replace(result, ticket_digest=without_parent.digest()))


@pytest.mark.parametrize("old_digest", [
    "e1a7b6af03451847c349d7b4db459d15d6efec1e7316eacc234c9a384d7ce7fd",
    "ab2922c88558afd37e95938773a7e148e88c1a6ec543f96c1ce05483fd1b7941",
])
def test_historical_one_sample_parent_revalidates_and_resettles_with_original_scoring(build, old_digest):
    params = build.params(confirm_next_round=True, null_samples_per_task=2,
                          recipe=dataclasses.replace(build.params().recipe, code_digest=old_digest))
    parent = build.parent(build.head())
    ticket = dataclasses.replace(build.ticket(n_entries=1, params=params), parent=parent)
    old_result = build.result(ticket, (PASS * 8, FAIL * 8), {0: PASS * 8}, parent_outcomes=(FAIL * 8,))
    check_result(ticket, old_result)
    controller = object.__new__(Controller)
    controller.state = SimpleNamespace(pending_king=parent.pending, round=SimpleNamespace(
        round_id=ticket.round_id, confirmation=None, parent=parent, excluded_training_prompts=0))
    events = []
    controller._log = lambda kind, data, now: events.append((kind, data))
    controller._resolve_confirmation([(ticket, old_result)], 1000)
    confirmation = events[0][1]["confirmation"]
    assert confirmation["delta"] == 1 and confirmation["confirmed"]
    with pytest.raises(ContractError, match="null_samples fresh samples"):
        check_result(ticket, dataclasses.replace(old_result, parent_outcomes=(FAIL * 8, PASS * 8, FAIL * 8)))
    unpinned = dataclasses.replace(ticket, params=build.params(confirm_next_round=True, null_samples_per_task=2))
    with pytest.raises(ContractError, match="null_samples fresh samples"):
        check_result(unpinned, dataclasses.replace(old_result, ticket_digest=unpinned.digest()))


def test_20f5ee33_paired_parent_result_revalidates_and_resettles_without_legacy_exception(build):
    params = build.params(confirm_next_round=True, null_samples_per_task=2,
                          recipe=dataclasses.replace(build.params().recipe,
                                                     code_digest="20f5ee3347b40340b459bd8321b08a67e195d3f760fc376cb60f61536ea5dff3"))
    parent = build.parent(build.head())
    ticket = dataclasses.replace(build.ticket(n_entries=1, params=params), parent=parent)
    result = build.result(ticket, (PASS * 8, FAIL * 8), {0: PASS * 8},
                          parent_outcomes=(FAIL * 8, FAIL * 8))
    check_result(ticket, result)
    controller = object.__new__(Controller)
    controller.state = SimpleNamespace(pending_king=parent.pending, round=SimpleNamespace(
        round_id=ticket.round_id, confirmation=None, parent=parent, excluded_training_prompts=0))
    events = []
    controller._log = lambda kind, data, now: events.append((kind, data))
    controller._resolve_confirmation([(ticket, result)], 1000)
    confirmation = events[0][1]["confirmation"]
    assert confirmation["delta"] == 0.5 and confirmation["confirmed"]
    with pytest.raises(ContractError, match="null_samples fresh samples"):
        check_result(ticket, dataclasses.replace(result, parent_outcomes=(FAIL * 8,)))


def test_historical_one_sample_result_survives_journal_replay_and_resettlement(tmp_path):
    world = confirming(tmp_path, n_tasks=8, min_scored_tasks=2, looks=(1.0,), null_samples_per_task=2)
    crown(world, quality=1.0)
    world.register(2, -1.0)
    tickets = world.controller.open_round(world.clock(), "legacy-parent")
    assert world.controller.state.pending_king is not None
    # Construct a synthetic old-format ticket, stored bytes and matching lease
    # seal in this test's own mirrored journal; no production record is changed.
    raw_ticket = canonical_json(tickets[0].to_dict() | {
        "params": tickets[0].params.to_dict() | {"recipe": tickets[0].params.recipe.to_dict() | {
            "code_digest": "ab2922c88558afd37e95938773a7e148e88c1a6ec543f96c1ce05483fd1b7941",
        }},
    })
    store = world.controller.store
    store.put(layout.ticket(tickets[0].round_id, tickets[0].lease_id), raw_ticket)
    old_ticket = get_record(store, layout.ticket(tickets[0].round_id, tickets[0].lease_id), RoundTicket)
    assert old_ticket.params.recipe.code_digest == "ab2922c88558afd37e95938773a7e148e88c1a6ec543f96c1ce05483fd1b7941"
    lease = dataclasses.replace(world.controller.state.round.leases[0], ticket_digest=old_ticket.digest())
    record = dataclasses.replace(world.controller.state.round, leases=(lease,))
    records = read_journal(world.root / JOURNAL_FILE)
    opened = next(item for item in records if item.kind == ROUND_OPENED and item.data["round"]["round_id"] == record.round_id)
    replacement = JournalRecord(opened.seq, opened.fence, opened.at, opened.kind, {"round": record.to_dict()}, opened.prev)
    records = [replacement if item.seq == opened.seq else item for item in records]
    lines = []
    previous = ""
    for item in records:
        item = dataclasses.replace(item, prev=previous)
        lines.append(item.line())
        previous = item.checksum
    for key, line in zip(sorted(store.list(layout.JOURNAL)), lines):
        store.put(key, line)
    (world.root / JOURNAL_FILE).write_bytes(b"".join(line + b"\n" for line in lines))
    old_result = BoxResult.from_json(canonical_json(world.runner.run(old_ticket).to_dict() | {"parent_outcomes": [FAIL * 8]}))
    check_result(old_ticket, old_result)
    put_record(store, layout.sealed_result(old_result.round_id, old_result.lease_id), old_result)
    world.reopen()
    world.evaluator.score_parent = lambda *args, **kwargs: pytest.fail("stored parent evidence must not run again")
    world.controller.tick(world.clock())
    assert world.controller.state.round is None and world.controller.state.pending_king is None
    confirmation = world.controller.state.rounds[-1].confirmation
    assert confirmation is not None and confirmation.confirmed
    assert load(world.root / JOURNAL_FILE).kings == world.controller.state.kings
    assert [king.uid for king in world.controller.state.kings] == [1]


def test_public_pair_and_check_pair_describe_confirmation_and_kl_publication(tmp_path):
    world = confirming(tmp_path, min_kl_seq=22.0, kl_max_tokens=8192)
    doc = snapshot(world.controller.state, world.params, world.clock())
    mining, llms = render(doc, "https://example.org")
    assert check_pair(mining, llms) == []
    text = " ".join(llms.decode().split())
    assert "22 nats per sequence" in text and "8192 completion tokens" in text
    assert "kl_seq, kl_seq_se, kl_tok, kl_k and kl_tokens" in text
    assert "There is no second confirmation round" not in text
    assert "Exact ties and losses" in text and "diagnostic z" in text
    for phrase in ("new_head_pass_rate", "old_head_pass_rate", "n_tasks, delta, se", "paid and", "payout_status", "round.json and mining.json"):
        assert phrase in text
    published = contract(world.params, "https://example.org")
    confirmation = published["crown_confirmation"]
    assert confirmation["enabled"]
    assert confirmation["new_samples"] == confirmation["parent_samples"] == confirmation["new_head_null_samples"] == world.params.null_samples
    assert "sample mean" in confirmation["comparison"] and "paired differences of sample means" in confirmation["comparison"]
    assert "same seed and sampling configuration" in text and "differences of sample means" in text
    assert confirmation["prior_winner_outcomes_used"] is False
    assert published["crown_confirmation"]["enabled"]
    assert published["duplicate_policy"]["kl_public_round_entry_fields"] == ["kl_seq", "kl_seq_se", "kl_tok", "kl_k", "kl_tokens"]


def test_solo_keeps_full_null_after_every_candidate_drops_when_parent_is_pinned(build):
    params = build.params(n_tasks=8, null_samples_per_task=2, confirm_next_round=True, looks=(0.5, 1.0),
                          min_scored_tasks=2, eval_chunk_tasks=4, z_bar=1.0)
    ticket = dataclasses.replace(build.ticket(n_entries=1, params=params), parent=build.parent(build.head()))
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=2), evaluator=SimpleNamespace(http=None))
    scheduler = SoloScheduler(runner, ticket, 10000)
    lora = build.artifact(ticket.head, ticket.entries[0], params)
    scheduler.add_lora(0, lora, gpu=0)
    for index in range(4):
        scheduler.result(NULL_MODEL, 0, index, PASS if index % 2 == 0 else FAIL)
        scheduler.result(NULL_MODEL, 1, index, PASS if index % 2 == 0 else FAIL)
        scheduler.result(lora.lora_hash, 0, index, FAIL)
    assert scheduler.dropped == {0: 0}
    scheduler.training_finished()
    assert scheduler.steal(0, NULL_MODEL) is not None
    for sample in range(2):
        for index in range(4, 8):
            scheduler.result(NULL_MODEL, sample, index, PASS)
    null, scored, dropped, stopped = scheduler.finish()
    assert null == ((PASS + FAIL) * 2 + PASS * 4, (PASS + FAIL) * 2 + PASS * 4) and scored[0] == (FAIL * 4 + NOT_RUN * 4,)
    assert dropped == {0: 0} and not stopped


def test_solo_eight_gpus_six_candidates_parent_does_not_change_batch_or_lora_table(build):
    params = build.params(n_tasks=8, n_gpus=8, batch_cap=6, null_samples_per_task=2,
                          confirm_next_round=True, looks=(1.0,), eval_chunk_tasks=2)
    ticket = dataclasses.replace(build.ticket(n_entries=6, params=params), parent=build.parent(build.head()))
    artifacts = {entry.queue_pos: build.artifact(ticket.head, entry, params) for entry in ticket.entries}
    events = []

    class Evaluator:
        http = None
        def clients_for(self, params):
            return 2
        def begin_round(self, ticket):
            events.append(("begin", ticket.parent.pending.uid))
        def _serve_one(self, head, params, table, gpu, **kwargs):
            events.append(("serve", head.version_id, table, gpu, kwargs))
            return f"http://gpu{gpu}"
        def solo_worker(self, ticket, trained, deadline, gpu):
            scheduler = self._solo_scheduler
            class Worker:
                def run(self, *, steal, result, **kwargs):
                    while (claim := steal(gpu, NULL_MODEL)) is not None:
                        model, start, stop = claim
                        for index in range(start, stop):
                            for sample in range(2 if model == NULL_MODEL else 1):
                                result(model, sample, index, PASS)
            return Worker()
        def score_parent(self, ticket, deadline):
            assert len(ticket.entries) == 6
            return (PASS * 8,) * ticket.params.null_samples
        def close(self):
            pass

    evaluator = Evaluator()
    trainer = SimpleNamespace(train=lambda request: artifacts[request.entry.queue_pos], close=lambda: None)
    result = SoloRunner(trainer, evaluator, lambda: 1000, build.image, settings=SoloSettings(steal_block_tasks=2), poll_s=0.01).run(ticket)
    assert ticket.params.batch_cap == 6 and params.n_gpus == 8
    assert result.null_outcomes == (PASS * 8, PASS * 8) and result.parent_outcomes == (PASS * 8, PASS * 8)
    check_result(ticket, result)
    assert all(event[2] == 6 for event in events if event[0] == "serve")


def test_parent_scoring_uses_the_same_fresh_seed_as_each_new_head_sample(tmp_path):
    world = confirming(tmp_path, null_samples_per_task=2)
    crown(world)
    world.register(2, -1.0)
    seen = []
    original = world.evaluator.score_parent
    def checked(ticket, deadline=0):
        seen.extend((sample, task, rollout_seed(ticket.draw.round_seed, task, sample))
                    for sample in range(ticket.params.null_samples) for task in ticket.draw.task_ids)
        return original(ticket, deadline)
    world.evaluator.score_parent = checked
    second = world.round()
    task_ids = json.loads(world.store.get(layout.round_file(second.round_id)))["draw"]["task_ids"]
    assert seen == [(sample, task, rollout_seed(second.round_seed, task, sample))
                    for sample in range(world.params.null_samples) for task in task_ids]
    assert seen and second.confirmation.n_tasks == len(task_ids)


def test_duplicate_only_close_expires_pending_without_a_lease_or_king(tmp_path):
    world = confirming(tmp_path)
    crown(world)
    old_head = world.controller.state.head
    world.register(2, 0.3)
    duplicate = dataclasses.replace(world.controller.state.rounds[-1].decision.entries[0], status=DUPLICATE,
                                    queue_pos=0, uid=2, hotkey="hk2", registration_block=1002)
    # Sim dedup is off by default: pin stage A to the real controller's all-duplicate, no-lease branch.
    world.controller._stage_a = lambda entries, now: ((), (duplicate,), (), {})
    world.controller.open_round(world.clock(), "duplicate-only")
    record = world.controller.state.rounds[-1]
    assert not record.leases and record.parent is not None and not world.controller.state.kings
    assert world.controller.state.pending_king is None and world.controller.state.head == old_head
    assert record.confirmation.reason.startswith("no_decision")
    assert load(world.root / JOURNAL_FILE).pending_king is None


class ControllerCrash(BaseException):
    pass


def test_crown_and_confirmation_replay_at_each_journal_boundary(tmp_path):
    for kind in (HEAD_APPLIED, HEAD_FOLDED, ROUND_CLOSED):
        for before in (False, True):
            world = confirming(tmp_path / f"crown-{kind}-{before}")
            world.register(1, 0.3)
            journal, real = world.controller.journal, world.controller.journal.append
            fired = []
            def crash(record_kind, data, at, *, real=real, kind=kind, before=before, fired=fired):
                if record_kind == kind and (before or fired):
                    raise ControllerCrash(kind)
                answer = real(record_kind, data, at)
                if record_kind == kind:
                    fired.append(True)
                    if not before:
                        raise ControllerCrash(kind)
                return answer
            journal.append = crash
            with pytest.raises(ControllerCrash):
                world.round()
            world.reopen()
            if world.controller.state.pending_king is None:
                world.round()
            assert world.controller.state.pending_king.uid == 1 and not world.controller.state.kings
            assert sum(json.loads(line)["kind"] == HEAD_APPLIED for line in (world.root / JOURNAL_FILE).read_text().splitlines()) == 1
    for kind in (ROUND_SETTLED, CONFIRMATION_RESOLVED, ROUND_CLOSED):
        for before in (False, True):
            world = confirming(tmp_path / f"confirm-{kind}-{before}")
            crown(world)
            world.evaluator.parent_quality = -0.5
            world.register(2, -1.0)
            journal, real = world.controller.journal, world.controller.journal.append
            fired = []
            def crash(record_kind, data, at, *, real=real, kind=kind, before=before, fired=fired):
                if record_kind == kind and (before or fired):
                    raise ControllerCrash(kind)
                answer = real(record_kind, data, at)
                if record_kind == kind:
                    fired.append(True)
                    if not before:
                        raise ControllerCrash(kind)
                return answer
            journal.append = crash
            with pytest.raises(ControllerCrash):
                world.round()
            world.reopen()
            if world.controller.state.pending_king is not None:
                world.round()
            assert [king.uid for king in world.controller.state.kings] == [1]
            assert sum(json.loads(line)["kind"] == CONFIRMATION_RESOLVED for line in (world.root / JOURNAL_FILE).read_text().splitlines()) == 1
            assert load(world.root / JOURNAL_FILE).kings == world.controller.state.kings


def test_multi_box_parent_evidence_is_sealed_and_scored_from_a_usable_box(build):
    params = build.params(confirm_next_round=True, null_samples_per_task=2, n_gpus=8, batch_cap=6)
    parent = build.parent(build.head())
    tickets = [dataclasses.replace(build.ticket(n_entries=3, first_pos=3 * index, lease=index, params=params), parent=parent)
               for index in range(2)]
    results = [build.result(ticket, (PASS * 8, FAIL * 8), {entry.queue_pos: PASS * 8 for entry in ticket.entries},
                            parent_outcomes=(FAIL * 8, FAIL * 8)) for ticket in tickets]
    controller = object.__new__(Controller)
    controller.state = SimpleNamespace(pending_king=parent.pending, round=SimpleNamespace(
        round_id="r000001", confirmation=None, parent=parent, excluded_training_prompts=7))
    events = []
    controller._log = lambda kind, data, now: events.append((kind, data))
    controller._resolve_confirmation(list(zip(tickets, results)), 1000)
    confirmation = events[0][1]["confirmation"]
    assert events[0][0] == CONFIRMATION_RESOLVED and confirmation["confirmed"]
    assert confirmation["delta"] == 0.5 and confirmation["n_tasks"] == 8 and confirmation["excluded_training_prompts"] == 7


def test_production_parent_uses_exact_merged_manifest_matched_samples_and_no_lora_table_growth(build, tmp_path, monkeypatch):
    from sn120_v2.gpu import evaluator as production
    from sn120_v2.gpu.resident import Resident
    from sn120_v2.store import LocalStore

    params = build.params(confirm_next_round=True, eval_merged=True, n_gpus=8, batch_cap=6,
                          null_samples_per_task=2, max_num_seqs=2)
    ticket = dataclasses.replace(build.ticket(n_entries=6, params=params), parent=build.parent(build.head()))
    store = LocalStore(tmp_path / "store")
    parent = ticket.parent.manifest
    for name, contents in {"model.bin": b"parent-base"}.items():
        store.put(layout.public_base_file(parent.head.base_hash, name), contents)
    # The old fold can prune both sets. Confirmation needs only the bf16 base.
    assert parent.master_files and not store.list(layout.public_master_prefix(parent.head.base_fp32_master_digest))
    parent_path = tmp_path / "work" / "cache" / parent.head.base_hash
    calls = []
    evaluator = production.VllmEvaluator.__new__(production.VllmEvaluator)
    evaluator.resident = Resident(store, tmp_path / "work")
    evaluator.capture = SimpleNamespace(close=lambda: None)
    evaluator._solo_clients = 2
    evaluator._solo_servers = {0: SimpleNamespace(alive=lambda: True)}
    evaluator.settings = production.ServeSettings(clients_per_server=2)
    evaluator.pool = SimpleNamespace(task=lambda task: SimpleNamespace(messages=lambda: []))
    evaluator.http = lambda url, body, timeout: {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}, "request": body}
    monkeypatch.setattr(production.ThreadPoolExecutor, "map", lambda self, fn, jobs: [fn(job) for job in jobs])
    evaluator._serve_one = lambda head, params, table, gpu, **kwargs: calls.append((head, table, gpu, kwargs)) or "http://parent"
    def rollout(request, url, job, written):
        assert job[0] == "parent" and 0 <= job[1] < ticket.params.null_samples and request.head == ticket.parent.manifest.head
        assert request.batches == 6 and request.round_seed == ticket.draw.round_seed
        answer = production.VllmEvaluator._rollout(evaluator, request, url, job, written)
        assert answer is not None, "the real rollout hook must accept its synthetic HTTP response"
        return answer
    jobs = []
    seeds = []
    original_http = evaluator.http
    def checked_http(url, body, timeout):
        if url.endswith("/v1/chat/completions"):
            model, sample, index = jobs[len(seeds)]
            task = ticket.draw.task_ids[index]
            null_body = production.chat_body(NULL_MODEL, [], rollout_seed(ticket.draw.round_seed, task, sample), params)
            assert body == production.chat_body(model, [], null_body["seed"], params)
            assert body["seed"] == null_body["seed"]
            assert {key: value for key, value in body.items() if key != "model"} == {
                key: value for key, value in null_body.items() if key != "model"
            }
            assert body["seed"] == rollout_seed(ticket.draw.round_seed, task, sample)
            assert body["model"] == "parent"
            seeds.append(body["seed"])
        return original_http(url, body, timeout)
    evaluator.http = checked_http
    def record_rollout(request, url, job, written):
        jobs.append(job)
        return rollout(request, url, job, written)
    evaluator._rollout = record_rollout
    evaluator._grade = lambda tasks, jobs, replies, request: {job: PASS for job in jobs}
    evaluator.begin_round(ticket)
    monkeypatch.setattr(production.VllmEvaluator, "clock_expired", staticmethod(lambda deadline: False))
    assert evaluator.score_parent(ticket, 0) == (PASS * 8, PASS * 8)
    assert evaluator.capture is not None
    assert len(calls) == 1 and calls[0][0] == ticket.parent.manifest.head
    assert calls[0][1:3] == (6, 0) and calls[0][3] == {"model": "parent", "base_dir": parent_path}
    assert jobs == [("parent", sample, index) for sample in range(params.null_samples) for index in range(8)]
    assert ticket.params.batch_cap == 6 and len(ticket.entries) == 6 and len(seeds) == 16
    assert evaluator._confirming_parent == ticket.parent.manifest.head
    ticket_without_parent = dataclasses.replace(ticket, parent=None)
    with pytest.raises(InfraError, match="manifest is unavailable"):
        evaluator.score_parent(ticket_without_parent, ticket.expires_at)
    with pytest.raises(InfraError, match="manifest is unavailable"):
        evaluator.score_parent(dataclasses.replace(ticket, parent=build.parent(build.head(), manifest=None)), ticket.expires_at)


def test_parent_close_retains_current_and_confirming_base_identities(build, tmp_path):
    from sn120_v2.gpu import evaluator as production

    ticket = dataclasses.replace(build.ticket(n_entries=1), parent=build.parent(build.head()))
    evaluator = production.VllmEvaluator.__new__(production.VllmEvaluator)
    evaluator._servers, evaluator._urls, evaluator._loaded = [], [], set()
    evaluator._solo_servers, evaluator._solo_server_clients, evaluator._solo_urls = {}, {}, {}
    evaluator._solo_loaded, evaluator._solo_models, evaluator._solo_base_dirs = {}, {}, {}
    evaluator._solo_start_locks, evaluator._solo_clients, evaluator._solo_scheduler = {}, None, None
    evaluator.capture = None
    evaluator._head = ticket.head
    evaluator._round_head = ticket.head
    evaluator._confirming_parent = ticket.parent.manifest.head
    kept = []
    evaluator.resident = SimpleNamespace(prune=lambda identities: kept.append(identities))
    evaluator.grader = SimpleNamespace(reset=lambda: None)
    evaluator._grader_failure = False
    evaluator.close()
    assert kept == [{head.base_hash for head in (ticket.head, ticket.parent.manifest.head)} |
                    {head.base_fp32_master_digest for head in (ticket.head, ticket.parent.manifest.head)}]


def test_parent_base_recovers_from_its_checkpoint_without_the_fp32_master(build, tmp_path):
    from sn120_v2.gpu.resident import Resident, resident_parent_base
    from sn120_v2.store import LocalStore

    manifest = build.parent(build.head()).manifest
    store = LocalStore(tmp_path / "store")
    store.put(layout.public_base_file(manifest.head.base_hash, "model.bin"), b"small metadata only")
    link = tmp_path / "work" / "checkpoint" / manifest.head.version_id / "base"
    link.mkdir(parents=True)
    (link / "model.bin").write_bytes(b"parent-base")
    resident = Resident(store, tmp_path / "work")
    base = resident_parent_base(resident, manifest)
    assert (base / "model.bin").read_bytes() == b"parent-base"
    assert not store.list(layout.public_master_prefix(manifest.head.base_fp32_master_digest))
