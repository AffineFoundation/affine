from __future__ import annotations

import dataclasses

import pytest

from sn120_v2.canonical import digest, manifest_digest
from sn120_v2.errors import ContractError, VerifyError
from sn120_v2.params import Params
from sn120_v2.types import (
    LOST,
    NO_DECISION,
    NULL_MODEL,
    SCORED,
    SKIP,
    WINNER,
    BoxResult,
    Draw,
    Entry,
    EntryDecision,
    EntryResult,
    EvalChunk,
    FoldArtifact,
    FrozenBatch,
    HeadManifest,
    HeadVersion,
    LoRAArtifact,
    LoRARef,
    Registration,
    RoundDecision,
    RoundTicket,
    TrainReport,
)

H1, H2, H3, H4 = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)
LORA_FILES = {"adapter_config.json": H1, "adapter_model.safetensors": H2}
BASE_FILES = {"config.json": H3, "model.safetensors": H4}


def genesis() -> HeadVersion:
    return HeadVersion.make(
        parent_version_id=None,
        base_hash=manifest_digest(BASE_FILES),
        base_fp32_master_digest=H1,
        loras=(),
        round_id="genesis",
        applied_at=1000,
    )


def lora_ref(lora_hash: str | None = None) -> LoRARef:
    return LoRARef(
        lora_hash=lora_hash or manifest_digest(LORA_FILES),
        round_id="r000001",
        miner_hotkey="hk7",
        miner_uid=7,
        batch_sha256=H3,
        recipe_id=Params().recipe_id,
        rank=16,
    )


def stacked(parent: HeadVersion) -> HeadVersion:
    return HeadVersion.make(
        parent_version_id=parent.version_id,
        base_hash=parent.base_hash,
        base_fp32_master_digest=parent.base_fp32_master_digest,
        loras=(lora_ref(),),
        round_id="r000001",
        applied_at=2000,
    )


def entry(position: int = 0, uid: int = 7) -> Entry:
    hotkey = f"hk{uid}"
    batch = FrozenBatch(hotkey, H3, f"private/rounds/r000001/batches/{hotkey}.json", "affine-batch-v2", 5, 10)
    return Entry(position, Registration(uid, hotkey, 100 + uid, 900), batch)


def artifact() -> LoRAArtifact:
    return LoRAArtifact(
        lora_hash=manifest_digest(LORA_FILES),
        files=LORA_FILES,
        sizes={"adapter_config.json": 300, "adapter_model.safetensors": 4000},
        prefix="pending/rounds/r000001/r000001-l0/loras/000",
        head_version_id=genesis().version_id,
        batch_sha256=H3,
        recipe_id=Params().recipe_id,
        rank=16,
        alpha=32.0,
        target_modules=("q_proj",),
        report=TrainReport(seed=1, steps=1, loss=0.5, step_norm=0.01, tokens=1000, wall_s=2.0, gpu_class="cpu", image_digest="sim"),
    )


def ticket() -> RoundTicket:
    return RoundTicket(
        round_id="r000001",
        lease_id="r000001-l0",
        box_id="box-a",
        head=genesis(),
        params=Params(),
        draw=Draw("snap-0", "seed", ("t1", "t2", "t3"), (1.0, 2.0, 1.5)),
        entries=(entry(0, 7), entry(1, 8)),
        issued_at=1000,
        expires_at=2000,
    )


def test_registration_key_is_uid_and_registration_block():
    assert Registration(7, "hk7", 123).key == (7, 123)
    with pytest.raises(ContractError):
        Registration(7, "", 123)


def test_head_version_id_is_the_hash_of_every_other_field():
    head = genesis()
    body = head.to_dict()
    del body["version_id"]
    assert head.version_id == digest(body)
    assert head.folded and head.parent_version_id is None


def test_head_version_id_changes_when_a_lora_is_stacked():
    head = genesis()
    child = stacked(head)
    assert child.version_id != head.version_id
    assert child.parent_version_id == head.version_id
    assert not child.folded


def test_head_version_with_a_wrong_id_cannot_be_built():
    data = genesis().to_dict()
    data["base_hash"] = H2
    with pytest.raises(VerifyError):
        HeadVersion.from_dict(data)


def test_head_version_folded_flag_must_match_the_lora_list():
    head = stacked(genesis())
    data = head.to_dict()
    data["folded"] = True
    body = dict(data)
    del body["version_id"]
    data["version_id"] = digest(body)
    with pytest.raises(ContractError):
        HeadVersion.from_dict(data)


def test_head_version_survives_a_json_round_trip():
    head = stacked(genesis())
    assert HeadVersion.from_json(head.to_json()) == head


def test_head_manifest_checks_every_hash_it_carries():
    head = stacked(genesis())
    good = dict(head=head, base_files=BASE_FILES, master_files={"master.safetensors": H1}, lora_files={manifest_digest(LORA_FILES): LORA_FILES})
    assert HeadManifest(**good).head == head
    wrong_base = good | {"base_files": {"config.json": H1}}
    missing_lora = good | {"lora_files": {}}
    wrong_lora = good | {"lora_files": {manifest_digest(LORA_FILES): {"adapter_model.safetensors": H4}}}
    for bad in (wrong_base, missing_lora, wrong_lora):
        with pytest.raises(VerifyError):
            HeadManifest(**bad)
    with pytest.raises(ContractError):
        HeadManifest(**(good | {"master_files": {}}))


@pytest.mark.parametrize(
    "task_ids,weights",
    [((), ()), (("a", "b"), (1.0,)), (("a", "a"), (1.0, 1.0)), (("a",), (0.0,)), (("a",), (float("inf"),))],
)
def test_draw_rejects_a_malformed_draw(task_ids, weights):
    with pytest.raises(ContractError):
        Draw("snap", "seed", task_ids, weights)


def test_lora_artifact_hash_must_match_its_files():
    assert artifact().lora_hash == manifest_digest(LORA_FILES)
    with pytest.raises(VerifyError):
        dataclasses.replace(artifact(), files={"adapter_model.safetensors": H4})
    with pytest.raises(ContractError):
        dataclasses.replace(artifact(), sizes={"adapter_config.json": 1})


def test_eval_chunk_must_score_null_and_every_task():
    good = EvalChunk(0, ("t1", "t2"), {NULL_MODEL: ("PF",), "lora": ("PC",)}, "sim")
    assert good.outcomes[NULL_MODEL] == ("PF",)
    for outcomes in ({"lora": ("PF",)}, {NULL_MODEL: ("P",)}, {NULL_MODEL: ("PX",)}, {NULL_MODEL: ("P-",)}):
        with pytest.raises(ContractError):
            EvalChunk(0, ("t1", "t2"), outcomes, "sim")


def test_fold_artifact_base_hash_must_match_its_files():
    good = dict(
        parent_version_id=genesis().version_id,
        base_hash=manifest_digest(BASE_FILES),
        base_files=BASE_FILES,
        base_fp32_master_digest=H2,
        master_files={"master.safetensors": H2},
        prefix="pending/folds/x",
        image_digest="sim",
    )
    assert FoldArtifact(**good).base_hash == manifest_digest(BASE_FILES)
    with pytest.raises(VerifyError):
        FoldArtifact(**(good | {"base_hash": H1}))
    with pytest.raises(ContractError):
        FoldArtifact(**(good | {"master_files": {}}))


def test_entry_batch_must_belong_to_the_entry_hotkey():
    batch = FrozenBatch("other", H3, "uri", "affine-batch-v1", 1, 1)
    with pytest.raises(ContractError):
        Entry(0, Registration(7, "hk7", 107), batch)


def test_ticket_entries_must_be_in_queue_order():
    with pytest.raises(ContractError):
        dataclasses.replace(ticket(), entries=(entry(1, 8), entry(0, 7)))
    with pytest.raises(ContractError):
        dataclasses.replace(ticket(), entries=())
    with pytest.raises(ContractError):
        dataclasses.replace(ticket(), expires_at=1000)


def test_ticket_digest_pins_head_params_draw_and_entries():
    base = ticket()
    assert RoundTicket.from_json(base.to_json()) == base
    assert RoundTicket.from_json(base.to_json()).digest() == base.digest()
    variants = [
        dataclasses.replace(base, params=dataclasses.replace(Params(), n_tasks=3)),
        dataclasses.replace(base, draw=Draw("snap-0", "other-seed", ("t1", "t2", "t3"), (1.0, 2.0, 1.5))),
        dataclasses.replace(base, entries=(entry(0, 7),)),
        dataclasses.replace(base, head=stacked(genesis())),
    ]
    assert len({v.digest() for v in variants} | {base.digest()}) == 5


def test_entry_result_status_rules():
    assert EntryResult(0, SCORED, lora=artifact(), outcomes=("PFP",)).status == SCORED
    assert EntryResult(0, NO_DECISION, reason="trainer out of memory").lora is None
    assert EntryResult(0, SKIP, reason="frozen file unreadable").status == SKIP
    for bad in (
        dict(status=SCORED),
        dict(status=NO_DECISION, reason="x", lora=artifact()),
        dict(status=NO_DECISION),
        dict(status="green", reason="x"),
    ):
        with pytest.raises(ContractError):
            EntryResult(0, **bad)


def test_box_result_round_trips_and_its_digest_is_the_seal():
    result = BoxResult(
        round_id="r000001",
        lease_id="r000001-l0",
        box_id="box-a",
        ticket_digest=ticket().digest(),
        head_version_id=genesis().version_id,
        image_digest="sim",
        null_outcomes=("PFP",),
        entries=(EntryResult(0, SCORED, lora=artifact(), outcomes=("PPP",)), EntryResult(1, NO_DECISION, reason="oom")),
        started_at=1001,
        finished_at=1500,
    )
    again = BoxResult.from_json(result.to_json())
    assert again == result and again.digest() == result.digest()
    assert dataclasses.replace(result, null_outcomes=("PPP",)).digest() != result.digest()


def decision(status: str, position: int = 0) -> EntryDecision:
    return EntryDecision(position, 7 + position, f"hk{7 + position}", 107 + position, H3, status)


def test_only_winners_and_losers_are_charged():
    assert decision(WINNER).charged and decision(LOST).charged
    assert not decision(NO_DECISION).charged and not decision(SKIP).charged
    with pytest.raises(ContractError):
        decision("green")


def test_round_decision_has_at_most_one_named_winner():
    null_round = RoundDecision("r000001", "head", None, (decision(LOST, 0), decision(LOST, 1)), 3, 0.5, 0.1)
    assert null_round.winner is None
    applied = RoundDecision("r000001", "head", 1, (decision(LOST, 0), decision(WINNER, 1)), 3, 0.5, 0.1)
    assert applied.winner.queue_pos == 1
    for winner_pos, entries in (
        (None, (decision(WINNER, 0),)),
        (0, (decision(LOST, 0),)),
        (0, (decision(WINNER, 0), decision(WINNER, 1))),
    ):
        with pytest.raises(ContractError):
            RoundDecision("r000001", "head", winner_pos, entries, 3, 0.5, 0.1)
