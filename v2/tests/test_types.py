from __future__ import annotations

import pytest

from sn120_v2.canonical import digest, manifest_digest
from sn120_v2.errors import ContractError, VerifyError
from sn120_v2.headlog import apply_lora, lora_ref
from sn120_v2.types import (
    LOST,
    NO_DECISION,
    NULL_MODEL,
    SCORED,
    SKIP,
    WINNER,
    Draw,
    Entry,
    EntryDecision,
    EntryResult,
    EvalChunk,
    HeadManifest,
    HeadVersion,
    Registration,
    RoundDecision,
    RoundTicket,
)


def stacked(build) -> tuple[HeadVersion, dict[str, str]]:
    head, entry = build.head(), build.entry(0)
    artifact = build.artifact(head, entry, build.params())
    return apply_lora(head, artifact, lora_ref("r000001", entry, artifact), at=2000), artifact.files


def test_head_version_id_is_the_hash_of_every_other_field_and_cannot_be_wrong(build):
    head = build.head()
    body = head.to_dict()
    del body["version_id"]
    assert head.version_id == digest(body) and head.folded and head.parent_version_id is None
    child, _ = stacked(build)
    assert child.version_id != head.version_id and child.parent_version_id == head.version_id and not child.folded
    assert HeadVersion.from_json(child.to_json()) == child
    with pytest.raises(VerifyError):
        HeadVersion.from_dict(head.to_dict() | {"base_hash": "2" * 64})
    lying = child.to_dict() | {"folded": True}
    lying["version_id"] = digest({k: v for k, v in lying.items() if k != "version_id"})
    with pytest.raises(ContractError):
        HeadVersion.from_dict(lying)


def test_head_manifest_checks_every_hash_it_carries(build):
    child, lora_files = stacked(build)
    base_files = {"model.bin": build.h("x")}
    with pytest.raises(VerifyError):
        HeadManifest(child, base_files, {"master.bin": build.h("m")}, {child.loras[0].lora_hash: lora_files})
    genesis_like = HeadVersion.make(
        parent_version_id=None, base_hash=manifest_digest(base_files), base_fp32_master_digest=build.h("d"), loras=(), round_id="genesis", applied_at=1
    )
    good = dict(head=genesis_like, base_files=base_files, master_files={"master.bin": build.h("m")}, lora_files={})
    assert HeadManifest.from_json(HeadManifest(**good).to_json()).head == genesis_like
    with pytest.raises(VerifyError):
        HeadManifest(**(good | {"lora_files": {build.h("stray"): lora_files}}))
    with pytest.raises(ContractError):
        HeadManifest(**(good | {"master_files": {}}))


def test_records_with_a_hash_or_a_shape_rule_refuse_bad_content(build):
    assert Registration(7, "hk7", 123).key == (7, 123)
    for bad in (lambda: Registration(7, "", 123), lambda: Entry(0, Registration(7, "hk7", 107), build.batch(build.reg(8)))):
        with pytest.raises(ContractError):
            bad()
    for task_ids, weights in (((), ()), (("a", "b"), (1.0,)), (("a", "a"), (1.0, 1.0)), (("a",), (0.0,)), (("a",), (float("inf"),))):
        with pytest.raises(ContractError):
            Draw("snap", "seed", task_ids, weights)
    artifact = build.artifact(build.head(), build.entry(0), build.params())
    with pytest.raises(VerifyError):
        build.replace(artifact, files={"adapter_model.safetensors": build.h("other")})
    assert EvalChunk(0, ("t1", "t2"), {NULL_MODEL: ("PF",), "lora": ("PC",)}, "sim").outcomes[NULL_MODEL] == ("PF",)
    for outcomes in ({"lora": ("PF",)}, {NULL_MODEL: ("P",)}, {NULL_MODEL: ("PX",)}, {NULL_MODEL: ("P-",)}):
        with pytest.raises(ContractError):
            EvalChunk(0, ("t1", "t2"), outcomes, "sim")


def test_ticket_digest_pins_head_params_draw_and_entries(build):
    base = build.ticket(2, 3)
    again = RoundTicket.from_json(base.to_json())
    assert again == base and again.digest() == base.digest()
    variants = [
        build.replace(base, params=build.params(n_tasks=3, z_bar=2.0)),
        build.replace(base, draw=Draw("snap-0", "other-seed", base.draw.task_ids, base.draw.weights)),
        build.replace(base, entries=base.entries[:1]),
        build.replace(base, head=stacked(build)[0]),
    ]
    assert len({v.digest() for v in variants} | {base.digest()}) == 5
    for bad in (dict(entries=tuple(reversed(base.entries))), dict(entries=()), dict(expires_at=base.issued_at)):
        with pytest.raises(ContractError):
            build.replace(base, **bad)


def test_box_result_round_trips_and_its_digest_is_the_seal(build):
    ticket = build.ticket(2, 3)
    result = build.result(ticket, "PFP", {0: "PPP", 1: NO_DECISION})
    again = type(result).from_json(result.to_json())
    assert again == result and again.digest() == result.digest()
    assert build.replace(result, null_outcomes=("PPP",)).digest() != result.digest()
    artifact = result.entries[0].lora
    assert EntryResult(0, SKIP, reason="frozen file unreadable").status == SKIP
    for bad in (dict(status=SCORED), dict(status=NO_DECISION, reason="x", lora=artifact), dict(status=NO_DECISION), dict(status="green", reason="x")):
        with pytest.raises(ContractError):
            EntryResult(0, **bad)


def test_a_round_decision_has_at_most_one_named_winner_and_only_scored_entries_are_charged(build):
    decision = lambda status, pos=0: EntryDecision(pos, 7 + pos, f"hk{7 + pos}", 107 + pos, build.h("b"), status)
    assert decision(WINNER).charged and decision(LOST).charged
    assert not decision(NO_DECISION).charged and not decision(SKIP).charged
    assert RoundDecision("r1", "head", None, (decision(LOST, 0), decision(LOST, 1)), 3, 0.5, 0.1).winner is None
    assert RoundDecision("r1", "head", 1, (decision(LOST, 0), decision(WINNER, 1)), 3, 0.5, 0.1).winner.queue_pos == 1
    for winner_pos, entries in ((None, (decision(WINNER, 0),)), (0, (decision(LOST, 0),)), (0, (decision(WINNER, 0), decision(WINNER, 1)))):
        with pytest.raises(ContractError):
            RoundDecision("r1", "head", winner_pos, entries, 3, 0.5, 0.1)
    with pytest.raises(ContractError):
        decision("green")
