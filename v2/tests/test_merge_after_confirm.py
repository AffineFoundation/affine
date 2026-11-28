"""merge_after_confirm: a crown reaches the head only through its next-round paired confirmation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from sn120_v2 import layout
from sn120_v2.errors import ConfigError
from sn120_v2.journal import JOURNAL_FILE, read_journal
from sn120_v2.params import Params, load_params
from sn120_v2.payout import paid_set
from sn120_v2.public import confirmation_stats, snapshot, winner_stats
from sn120_v2.publisher import MERGE_AFTER_CONFIRM_CHANGELOG, StorePublisher, check_pair, contract, head_now_preview, render
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import CONFIRMATION_RESOLVED, CROWN_DEFERRED, HEAD_APPLIED, HEAD_FOLDED, HEAD_REVERTED, load
from sn120_v2.types import PendingKing

LIVE = Path(__file__).parent / "fixtures" / "live_params_92676f30.json"
CHANGELOG = ("A crown (z > z_crown) is not merged into the head until its LoRA passes a fresh paired re-test in the next round "
             "(z >= z_bar). Until then miners train on and are judged against the unchanged head. A crown that fails confirmation "
             "is discarded and never touches the head; a legacy unconfirmed fold that fails is reverted. Only confirmed crowns are merged and paid.")


def attach(world: SimWorld) -> SimWorld:
    """The real store publisher, so head_now.json, mining.json and llms.txt are written as in production."""
    world.controller.publisher = StorePublisher(world.store, "https://files.example")
    return world


def merging(tmp_path, **changes) -> SimWorld:
    return attach(SimWorld(tmp_path, sim_params(confirm_next_round=True, merge_after_confirm=True, **changes)))


def kinds(world: SimWorld) -> list[str]:
    return [record.kind for record in read_journal(world.root / JOURNAL_FILE)]


def public(world: SimWorld, key: str) -> dict:
    return json.loads(world.store.get(key))


def head_ids(world: SimWorld) -> set[str]:
    return {head.version_id for head in world.controller.state.heads}


def assert_replays(world: SimWorld) -> None:
    replayed = load(world.root / JOURNAL_FILE)
    state = world.controller.state
    assert (replayed.head, replayed.kings, replayed.pending_king, replayed.heads) == (state.head, state.kings, state.pending_king, state.heads)


def crown(world: SimWorld, uid: int = 1, quality: float = 0.3):
    head = world.controller.state.head
    world.register(uid, quality)
    record = world.round()
    state = world.controller.state
    assert record.outcome == "applied" and state.head == head
    pending = state.pending_king
    assert isinstance(pending, PendingKing) and pending.deferred and pending.parent_version_id == head.version_id
    assert pending.new_version_id not in head_ids(world) and pending.candidate.head.version_id not in head_ids(world)
    assert pending.candidate.head.folded and pending.candidate.head.parent_version_id == pending.new_version_id
    assert CROWN_DEFERRED in kinds(world) and HEAD_APPLIED not in kinds(world)
    return record, pending


# ---- params ----


def test_default_and_live_params_hash_are_unchanged_and_the_flag_round_trips():
    assert "merge_after_confirm" not in Params().to_dict()
    live = load_params(LIVE)
    assert live.params_hash == "92676f30aff26748285f49a4c37638938fb7ac2bc346217142c76d62c67b4c80"
    on = Params.from_dict({**live.to_dict(), "merge_after_confirm": True})
    assert on.merge_after_confirm and on.to_dict()["merge_after_confirm"] is True
    assert on.params_hash == "7838317829e36ebbf769b0b150fbbd45e18984da4d7fe9b97d7ebca1bf7ff6f7"
    assert Params.from_dict(on.to_dict()) == on
    assert Params.from_dict(json.loads(json.dumps(on.to_dict()))).params_hash == on.params_hash


def test_the_flag_needs_next_round_confirmation_and_one_fold_per_crown():
    with pytest.raises(ConfigError, match="confirm_next_round"):
        sim_params(merge_after_confirm=True)
    with pytest.raises(ConfigError, match="fold_every_k"):
        sim_params(confirm_next_round=True, merge_after_confirm=True, fold_every_k=2)
    with pytest.raises(ConfigError):
        dataclasses.replace(sim_params(confirm_next_round=True), merge_after_confirm=1)


# ---- the rules, end to end in SimWorld ----


def test_crown_is_deferred_and_the_next_round_trains_on_the_unchanged_head(tmp_path):
    world = merging(tmp_path)
    head = world.controller.state.head
    first, pending = crown(world)
    assert public(world, layout.HEAD_NOW)["version_id"] == head.version_id
    assert public(world, layout.MINING_JSON)["head"]["head"]["version_id"] == head.version_id
    mining = public(world, layout.MINING_JSON)
    row = mining["confirmations"]["rows"][0]
    assert row["status"] == "pending_confirmation" and row["merged"] is False and row["paid"] is False
    assert mining["confirmations"]["total"]["pending"] == 1
    assert all(king["round_id"] != first.round_id for king in mining["kings"]) and mining["paid_kings"] == []
    stats = winner_stats(world.controller.state, world.clock())
    assert stats["winners"][0]["head"] is None and stats["winners"][0]["confirmed"] is None
    world.evaluator.candidate_quality = -0.5
    world.register(2, -1.0)
    second = world.round()
    assert second.head_version_id == head.version_id
    assert second.parent.manifest == pending.candidate
    assert_replays(world)


def test_confirm_pass_merges_the_candidate_and_pays_from_the_confirmation(tmp_path):
    world = merging(tmp_path)
    head = world.controller.state.head
    first, pending = crown(world)
    world.register(2, -1.0)
    second = world.round()
    state = world.controller.state
    confirmation = second.confirmation
    assert confirmation.confirmed and confirmation.merged and confirmation.parent_version_id == head.version_id
    assert state.head == pending.candidate.head and state.head.parent_version_id == pending.new_version_id
    assert [h.version_id for h in state.heads[-2:]] == [pending.new_version_id, pending.candidate.head.version_id]
    assert state.pending_king is None
    king = state.kings[-1]
    assert (king.uid, king.round_id, king.head_version_id) == (pending.uid, first.round_id, pending.new_version_id)
    assert king.applied_at > pending.applied_at and king.applied_at >= second.opened_at  # paid from the confirmation on
    assert [k.round_id for k in paid_set(state.kings, state.null_rounds, world.params)] == [first.round_id]
    assert public(world, layout.HEAD_NOW)["version_id"] == state.head.version_id
    mining = public(world, layout.MINING_JSON)
    assert mining["head"]["head"]["version_id"] == state.head.version_id
    assert mining["heads"][-1]["version_id"] == state.head.version_id
    assert world.store.exists(layout.head_manifest(pending.new_version_id))
    report = public(world, layout.round_file(second.round_id))
    assert report["confirmation"]["merged"] is True and report["new_head_version_id"] == state.head.version_id
    row = mining["confirmations"]["rows"][0]
    assert row["status"] == "paid" and row["merged"] is True
    assert [k.kind for k in read_journal(world.root / JOURNAL_FILE)].count(CONFIRMATION_RESOLVED) == 1
    assert_replays(world)


@pytest.mark.parametrize("why", ["fails", "no_decision"])
def test_confirm_fail_or_no_decision_discards_the_crown(tmp_path, why):
    world = merging(tmp_path)
    head = world.controller.state.head
    first, pending = crown(world)
    if why == "fails":
        world.evaluator.candidate_quality = -0.5
    else:
        def broken(ticket, deadline=0):
            from sn120_v2.errors import InfraError
            raise InfraError("simulated parent failure")
        world.evaluator.score_parent = broken
    world.register(2, -1.0)
    second = world.round()
    state = world.controller.state
    assert not second.confirmation.confirmed and not second.confirmation.merged
    assert (why == "no_decision") == second.confirmation.reason.startswith("no_decision")
    assert state.head == head and state.pending_king is None and state.kings == []
    assert pending.new_version_id not in head_ids(world)
    assert all(ref.lora_hash != pending.stacked.head.loras[0].lora_hash for h in state.heads for ref in h.loras)
    assert public(world, layout.HEAD_NOW)["version_id"] == head.version_id
    assert HEAD_REVERTED not in kinds(world)
    world.register(3, -1.0)
    third = world.round()  # never re-tested
    assert third.parent is None and third.confirmation is None
    assert_replays(world)


def test_same_round_confirmation_and_new_winner(tmp_path):
    world = merging(tmp_path)
    _, old = crown(world, 1)
    world.register(2, 0.4)
    second = world.round()
    state = world.controller.state
    assert second.outcome == "applied" and second.confirmation.merged
    assert state.head == old.candidate.head and [k.uid for k in state.kings] == [1]
    new = state.pending_king
    assert new.deferred and new.uid == 2 and new.crown_round_id == second.round_id
    assert new.parent_version_id == state.head.version_id  # tested next round on the head that is current then
    assert new.candidate.head.version_id not in head_ids(world)
    world.register(3, -1.0)
    third = world.round()
    assert third.confirmation.merged and [k.uid for k in world.controller.state.kings] == [1, 2]
    assert world.controller.state.head == new.candidate.head
    assert_replays(world)


@pytest.mark.parametrize("passes", [False, True])
def test_legacy_pending_after_the_flip_is_reverted_on_failure_and_kept_on_success(tmp_path, passes):
    world = attach(SimWorld(tmp_path, sim_params(confirm_next_round=True)))
    root = world.controller.state.head
    world.register(1, 0.3)
    world.round()
    legacy = world.controller.state.pending_king
    tip = world.controller.state.head
    assert not legacy.deferred and tip.parent_version_id == legacy.new_version_id
    world.controller.set_params(dataclasses.replace(world.params, merge_after_confirm=True), world.clock())
    world.reopen()
    attach(world)
    world.evaluator.parent_quality = -0.5 if passes else 0.5
    world.register(2, -1.0)
    second = world.round()
    state = world.controller.state
    assert second.confirmation.confirmed is passes and not second.confirmation.merged
    if passes:
        assert state.head == tip and [k.uid for k in state.kings] == [1] and state.reverts == []
    else:
        assert state.head == root and state.kings == []
        assert state.reverts[-1]["round_id"] == second.round_id and state.reverts[-1]["to_version_id"] == root.version_id
        assert HEAD_REVERTED in kinds(world)
        assert public(world, layout.HEAD_NOW)["version_id"] == root.version_id
        mining = public(world, layout.MINING_JSON)
        assert mining["head"]["head"]["version_id"] == root.version_id
        assert mining["heads"][-1]["version_id"] == root.version_id
        assert tip.version_id not in {row["version_id"] for row in mining["heads"]}
    assert_replays(world)


def test_disabled_mode_keeps_the_old_install_at_crown(tmp_path):
    world = SimWorld(tmp_path, sim_params(confirm_next_round=True))
    world.register(1, 0.3)
    world.round()
    assert HEAD_APPLIED in kinds(world) and HEAD_FOLDED in kinds(world) and CROWN_DEFERRED not in kinds(world)
    assert not world.controller.state.pending_king.deferred
    assert "merged" not in json.dumps(confirmation_stats(world.controller.state))


# ---- the public contract ----


def test_contract_llms_and_changelog_state_the_rule_only_when_on(tmp_path):
    off = sim_params(confirm_next_round=True)
    on = dataclasses.replace(off, merge_after_confirm=True)
    assert "merge_rule" not in contract(off, "https://example.org")["crown_confirmation"]
    block = contract(on, "https://example.org")["crown_confirmation"]
    assert block["merge_rule"] == "after_confirmation" and "merged" in block["confirmation_fields"]
    assert CHANGELOG in MERGE_AFTER_CONFIRM_CHANGELOG[0] and MERGE_AFTER_CONFIRM_CHANGELOG[0].startswith("- 2026-11-28 /")
    world = merging(tmp_path)
    crown(world)
    doc = snapshot(world.controller.state, world.params, world.clock())
    mining, llms = render(doc, "https://example.org")
    assert check_pair(mining, llms) == []
    text = " ".join(llms.decode().split())
    assert " ".join(MERGE_AFTER_CONFIRM_CHANGELOG[0].split()) in text and "| `merge_after_confirm` | yes |" in text
    assert "is not merged into the head" in text
    assert json.loads(mining)["contract"]["crown_confirmation"]["merge_rule"] == "after_confirmation"
    plain = SimWorld(tmp_path / "off", off)
    mining_off, llms_off = render(snapshot(plain.controller.state, off, plain.clock()), "https://example.org")
    assert MERGE_AFTER_CONFIRM_CHANGELOG[0] not in llms_off.decode() and "merge_after_confirm" not in llms_off.decode()


def test_head_now_preview_drops_rows_after_a_revert_target():
    from types import SimpleNamespace

    rows = [{"version_id": v, "round_id": "r", "applied_at": i, "folded": True} for i, v in enumerate("abc")]
    head = SimpleNamespace(head=SimpleNamespace(version_id="a", round_id="r", applied_at=9, folded=True))
    assert [row["version_id"] for row in head_now_preview({"heads": rows}, head)] == ["a"]
    head.head.version_id = "d"
    assert [row["version_id"] for row in head_now_preview({"heads": rows}, head)] == ["a", "b", "c", "d"]


def test_public_loader_used_by_replicate_and_miner_tools_reproduces_the_hash():
    from sn120_v2.public import visible_params

    on = Params.from_dict({**load_params(LIVE).to_dict(), "merge_after_confirm": True})
    shown = json.loads(json.dumps(visible_params(on)))
    assert shown["merge_after_confirm"] is True
    assert Params.from_public_dict(shown).params_hash == Params.from_dict(shown).params_hash == on.params_hash
    off = load_params(LIVE)
    assert "merge_after_confirm" not in visible_params(off)
    assert Params.from_public_dict(json.loads(json.dumps(visible_params(off)))).params_hash == off.params_hash
