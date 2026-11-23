from __future__ import annotations

import pytest

from sn120_v2.canonical import manifest_digest
from sn120_v2.errors import ContractError, HeadMismatchError
from sn120_v2.headlog import apply_lora, fold_due, fold_head, genesis_head, lora_ref
from sn120_v2.payout import check_weights, paid_payees, payees, weights
from sn120_v2.types import FoldArtifact, King


def king(uid: int, order: int, hotkey: str | None = None) -> King:
    return King(uid, hotkey or f"hk{uid}", f"r{order:06d}", f"head{order}", 1000 + order)


def chain(*uids: int) -> dict[int, str]:
    return {uid: f"hk{uid}" for uid in uids}


def test_payees_are_the_last_ten_distinct_winner_uids_newest_first():
    kings = [king(uid, order) for order, uid in enumerate(range(1, 13))]
    assert [k.uid for k in payees(kings, 10)] == [12, 11, 10, 9, 8, 7, 6, 5, 4, 3], "the eleventh and twelfth pushed out the oldest"
    again = payees([king(1, 0), king(2, 1), king(1, 2)], 10)
    assert [k.uid for k in again] == [1, 2] and again[0].round_id == "r000002", "a uid that wins again keeps one place"


def test_each_paid_payee_gets_an_equal_share(build):
    kings = [king(uid, order) for order, uid in enumerate((4, 5, 6))]
    assert weights(kings, chain(4, 5, 6), build.params()) == {uid: pytest.approx(1 / 3) for uid in (4, 5, 6)}
    ten = [king(uid, order) for order, uid in enumerate(range(1, 11))]
    assert weights(ten, chain(*range(1, 11)), build.params()) == {uid: pytest.approx(0.1) for uid in range(1, 11)}
    assert set(weights(kings, chain(4, 5, 6), build.params(kings_window=2))) == {5, 6}


def test_a_payee_is_paid_only_while_its_earning_hotkey_stays_on_its_uid(build):
    kings = [king(uid, order) for order, uid in enumerate((4, 5, 6))]
    assert weights(kings, chain(4, 6), build.params()) == {4: 0.5, 6: 0.5}, "uid 5 deregistered: the rest split the weight"
    assert weights(kings, chain(4, 6) | {5: "newcomer"}, build.params()) == {4: 0.5, 6: 0.5}
    assert [k.uid for k in paid_payees(kings, chain(4), 10)] == [4]
    eleven = [king(uid, order) for order, uid in enumerate(range(1, 12))]
    table = weights(eleven, chain(*range(1, 11)), build.params())
    assert 1 not in table and len(table) == 9, "uid 11 took a place in the window even though it is not paid"


def test_weights_are_never_all_zero(build):
    assert weights([], chain(1, 2), build.params()) == {0: 1.0}, "no king: all weight to the burn uid"
    assert weights([king(4, 0)], chain(1, 2), build.params()) == {0: 1.0}, "no paid payee: all weight to the burn uid"
    assert weights([King(0, "owner", "r000001", "head", 1)], {0: "owner"}, build.params()) == {0: 1.0}
    for table in ({}, {1: 0.0}, {1: -0.5, 2: 1.5}, {1: float("nan")}, {1: 0.6, 2: 0.6}, {1: 0.2}, {-1: 1.0}):
        with pytest.raises(ContractError):
            check_weights(table)


def applied(build, head):
    entry = build.entry(0)
    artifact = build.artifact(head, entry, build.params())
    return apply_lora(head, artifact, lora_ref("r000001", entry, artifact), at=2000), artifact, entry


def fold_of(head) -> FoldArtifact:
    base_files = {"model.safetensors": "5" * 64}
    return FoldArtifact(head.version_id, manifest_digest(base_files), base_files, "6" * 64, {"master.safetensors": "7" * 64}, "pending/folds/x", "sim")


def test_genesis_apply_and_fold_give_a_chain_of_heads(build):
    spec = build.genesis_spec()
    genesis = genesis_head(spec, at=1000)
    assert (genesis.base_hash, genesis.base_fp32_master_digest, genesis.parent_version_id, genesis.folded) == (spec.base_hash, spec.base_fp32_master_digest, None, True)
    child, artifact, entry = applied(build, genesis)
    assert child.parent_version_id == genesis.version_id and [ref.lora_hash for ref in child.loras] == [artifact.lora_hash]
    assert (child.loras[0].miner_uid, child.loras[0].miner_hotkey, child.base_hash, child.folded) == (entry.reg.uid, entry.reg.hotkey, genesis.base_hash, False)
    folded = fold_head(child, fold_of(child), at=3000)
    assert folded.folded and folded.loras == () and folded.parent_version_id == child.version_id
    assert (folded.base_hash, folded.base_fp32_master_digest, folded.round_id) == (fold_of(child).base_hash, "6" * 64, child.round_id)
    assert not fold_due(genesis, build.params()) and fold_due(child, build.params()) and not fold_due(child, build.params(fold_every_k=2))


def test_a_lora_or_a_fold_made_from_another_head_is_refused(build):
    head = build.head()
    child, artifact, entry = applied(build, head)
    for wrong in (
        lambda: apply_lora(child, artifact, lora_ref("r000002", entry, artifact), at=3000),
        lambda: fold_head(child, fold_of(head), at=3000),
        lambda: fold_head(head, fold_of(head), at=3000),
    ):
        with pytest.raises(HeadMismatchError):
            wrong()


def test_the_same_transitions_always_give_the_same_version_ids(build):
    def walk():
        head = build.head()
        child, _, _ = applied(build, head)
        return [head.version_id, child.version_id, fold_head(child, fold_of(child), at=3000).version_id]

    assert walk() == walk() and len(set(walk())) == 3
    assert genesis_head(build.genesis_spec(), at=1001).version_id != genesis_head(build.genesis_spec(), at=1000).version_id
