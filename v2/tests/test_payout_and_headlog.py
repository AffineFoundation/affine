from __future__ import annotations

import pytest

from sn120_v2.canonical import manifest_digest
from sn120_v2.errors import ContractError, HeadMismatchError
from sn120_v2.headlog import apply_lora, fold_due, fold_head, genesis_head, lora_ref
from sn120_v2.payout import check_weights, paid_payees, payees, weights
from sn120_v2.types import FoldArtifact, King


def king(uid: int, order: int, hotkey: str | None = None) -> King:
    return King(uid, hotkey or f"hk{uid}", f"r{order:06d}", f"head{order}", 1000 + order)


def chain(*uids: int, **swapped: str) -> dict[int, str]:
    table = {uid: f"hk{uid}" for uid in uids}
    table.update({int(uid[1:]): hotkey for uid, hotkey in swapped.items()})
    return table


def test_payees_are_the_last_ten_distinct_winner_uids_newest_first(build):
    kings = [king(uid, order) for order, uid in enumerate(range(1, 13))]
    assert [k.uid for k in payees(kings, 10)] == [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]


def test_an_eleventh_distinct_king_pushes_out_the_oldest(build):
    kings = [king(uid, order) for order, uid in enumerate(range(1, 11))]
    assert 1 in [k.uid for k in payees(kings, 10)]
    kings.append(king(11, 10))
    assert [k.uid for k in payees(kings, 10)][-1] == 2


def test_a_uid_that_wins_again_keeps_one_place_with_its_newest_win(build):
    kings = [king(1, 0), king(2, 1), king(1, 2)]
    chosen = payees(kings, 10)
    assert [k.uid for k in chosen] == [1, 2]
    assert chosen[0].round_id == "r000002"


def test_each_paid_payee_gets_an_equal_share(build):
    kings = [king(uid, order) for order, uid in enumerate((4, 5, 6))]
    table = weights(kings, chain(4, 5, 6), build.params())
    assert table == {4: pytest.approx(1 / 3), 5: pytest.approx(1 / 3), 6: pytest.approx(1 / 3)}
    assert sum(table.values()) == pytest.approx(1.0)


def test_ten_payees_get_a_tenth_each(build):
    kings = [king(uid, order) for order, uid in enumerate(range(1, 11))]
    table = weights(kings, chain(*range(1, 11)), build.params())
    assert len(table) == 10 and all(value == pytest.approx(0.1) for value in table.values())


def test_a_deregistered_payee_drops_and_the_rest_split_the_weight(build):
    kings = [king(uid, order) for order, uid in enumerate((4, 5, 6))]
    table = weights(kings, chain(4, 6), build.params())
    assert table == {4: 0.5, 6: 0.5}


def test_a_uid_now_held_by_another_hotkey_is_not_paid(build):
    kings = [king(4, 0), king(5, 1)]
    table = weights(kings, chain(4, u5="newcomer"), build.params())
    assert table == {4: 1.0}
    assert [k.uid for k in paid_payees(kings, chain(4, u5="newcomer"), 10)] == [4]


def test_a_dropped_payee_still_holds_its_place_in_the_window(build):
    kings = [king(uid, order) for order, uid in enumerate(range(1, 12))]
    registered = chain(*range(1, 11))
    table = weights(kings, registered, build.params())
    assert 1 not in table, "uid 1 left the window when uid 11 won, even though uid 11 is not paid"
    assert len(table) == 9


def test_with_no_king_all_weight_goes_to_the_burn_uid(build):
    assert weights([], chain(1, 2), build.params()) == {0: 1.0}


def test_with_no_paid_payee_all_weight_goes_to_the_burn_uid(build):
    assert weights([king(4, 0)], chain(1, 2), build.params()) == {0: 1.0}


def test_weights_are_never_all_zero(build):
    for kings, registered in (([], {}), ([king(4, 0)], {}), ([king(4, 0)], chain(4))):
        table = weights(kings, registered, build.params())
        assert table and sum(table.values()) == pytest.approx(1.0) and max(table.values()) > 0


def test_the_burn_uid_is_never_paid_as_a_king(build):
    assert weights([King(0, "owner", "r000001", "head", 1)], {0: "owner"}, build.params()) == {0: 1.0}


def test_the_window_size_is_a_parameter(build):
    kings = [king(uid, order) for order, uid in enumerate((4, 5, 6))]
    assert set(weights(kings, chain(4, 5, 6), build.params(kings_window=2))) == {5, 6}


@pytest.mark.parametrize(
    "table",
    [{}, {1: 0.0}, {1: -0.5, 2: 1.5}, {1: float("nan")}, {1: 0.6, 2: 0.6}, {1: 0.2}, {-1: 1.0}],
)
def test_check_weights_refuses_a_bad_table(table):
    with pytest.raises(ContractError):
        check_weights(table)


def test_genesis_head_comes_from_the_genesis_spec(build):
    spec = build.genesis_spec()
    head = genesis_head(spec, at=1000)
    assert (head.base_hash, head.base_fp32_master_digest) == (spec.base_hash, spec.base_fp32_master_digest)
    assert head.parent_version_id is None and head.folded and head.loras == ()
    assert genesis_head(spec, at=1000).version_id == head.version_id
    assert genesis_head(spec, at=1001).version_id != head.version_id


def applied(build, head):
    params = build.params()
    entry = build.entry(0)
    artifact = build.artifact(head, entry, params)
    return apply_lora(head, artifact, lora_ref("r000001", entry, artifact), at=2000), artifact, entry


def test_apply_stacks_the_lora_and_links_the_parent(build):
    head = build.head()
    child, artifact, entry = applied(build, head)
    assert child.parent_version_id == head.version_id and child.version_id != head.version_id
    assert [ref.lora_hash for ref in child.loras] == [artifact.lora_hash]
    assert (child.loras[0].miner_uid, child.loras[0].miner_hotkey) == (entry.reg.uid, entry.reg.hotkey)
    assert child.base_hash == head.base_hash and not child.folded


def test_a_lora_trained_on_another_head_can_never_be_applied(build):
    head = build.head()
    child, artifact, entry = applied(build, head)
    with pytest.raises(HeadMismatchError):
        apply_lora(child, artifact, lora_ref("r000002", entry, artifact), at=3000)


def fold_of(head) -> FoldArtifact:
    base_files = {"model.safetensors": "5" * 64}
    return FoldArtifact(head.version_id, manifest_digest(base_files), base_files, "6" * 64, {"master.safetensors": "7" * 64}, "pending/folds/x", "sim")


def test_fold_replaces_the_stack_with_a_new_base(build):
    child, _, _ = applied(build, build.head())
    folded = fold_head(child, fold_of(child), at=3000)
    assert folded.folded and folded.loras == ()
    assert folded.base_hash == fold_of(child).base_hash and folded.base_fp32_master_digest == "6" * 64
    assert folded.parent_version_id == child.version_id and folded.round_id == child.round_id


def test_a_fold_made_from_another_head_is_refused(build):
    head = build.head()
    child, _, _ = applied(build, head)
    with pytest.raises(HeadMismatchError):
        fold_head(child, fold_of(head), at=3000)
    with pytest.raises(HeadMismatchError):
        fold_head(head, fold_of(head), at=3000)


def test_fold_is_due_after_fold_every_k_applies(build):
    head = build.head()
    child, _, _ = applied(build, head)
    assert not fold_due(head, build.params())
    assert fold_due(child, build.params(fold_every_k=1))
    assert not fold_due(child, build.params(fold_every_k=2))


def test_the_same_transitions_always_give_the_same_version_ids(build):
    def walk():
        head = build.head()
        child, _, _ = applied(build, head)
        folded = fold_head(child, fold_of(child), at=3000)
        return [head.version_id, child.version_id, folded.version_id]

    assert walk() == walk() and len(set(walk())) == 3
