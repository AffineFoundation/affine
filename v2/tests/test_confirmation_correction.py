"""One approved offline correction of an infrastructure-only crown no-decision."""

from __future__ import annotations

import dataclasses
import json
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from sn120_v2 import layout, recheck
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.chain import FakeChain, Reader
from sn120_v2.cli import main
from sn120_v2.controller import Controller
from sn120_v2.correction import check_correction, read_evidence
from sn120_v2.errors import ConfigError, ContractError, InfraError, JournalCorrupt, V2Error, VerifyError
from sn120_v2.gpu.plan import code_digest
from sn120_v2.journal import JOURNAL_FILE, Journal, read_journal
from sn120_v2.judging import check_result, score_confirmation
from sn120_v2.params import Params, load_params
from sn120_v2.payout import paid_set, payees
from sn120_v2.public import snapshot
from sn120_v2.publisher import check_pair, render
from sn120_v2.sim import SimEvaluator, SimWorld, sim_params
from sn120_v2.state import CONFIRMATION_CORRECTED, apply, load, replay
from sn120_v2.types import BoxResult, PASS, FAIL, RoundTicket
from sn120_v2.weights_writer import dry_run, read_view


def cli(world: SimWorld, verdict, approved_by="operator approved offline re-check #1"):
    return main([
        "correct-confirmation", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack",
        "--verdict", str(verdict), "--approved-by", approved_by,
    ])


def no_decision(tmp_path):
    from sn120_v2.envs.math_pool import MathPool, Task

    world = SimWorld(tmp_path, sim_params(confirm_next_round=True, n_tasks=8, min_scored_tasks=2, looks=(1.0,), batch_cap=1,
                                         null_samples_per_task=2, null_prune_from_round="", z_bar=0.0))
    world.register(1, 0.3)
    # Preserve the original winner file and its hash. The real recheck writer
    # verifies those exact training bytes against the exclusion evidence.
    world.admission.upload("hk1", 0.3)
    original = json.loads(world.store.get(layout.miner_object("hk1")))
    training = {"version": "affine-batch-v3", "hotkey": "hk1", "groups": [
        {"prompt": f"problem {i}", "rollouts": [{"output": f"trained answer {i}", "reward": 1.0}]}
        for i in range(12)
    ]}
    world.store.put(layout.miner_object("hk1"), canonical_json(original | training))
    first = world.round()
    pending = world.controller.state.pending_king
    assert first is not None and pending is not None
    real_exclusions = world.controller._training_task_exclusions

    def exclusions(parent):
        if parent is not None:
            return frozenset(f"sim:{i}" for i in range(12))
        return real_exclusions(parent)

    world.controller._training_task_exclusions = exclusions
    world.controller.prompt_pool = MathPool(
        tuple(Task(f"sim:{i}", "affine_math", f"problem {i}", "1") for i in range(400)), "test")
    world.evaluator.score_parent = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated parent crash"))
    world.register(2, -1.0)
    second = world.round()
    assert second is not None and second.confirmation is not None
    assert second.confirmation.reason.startswith("no_decision: parent scoring failed:")
    assert second.excluded_training_prompts == 12
    assert not world.controller.state.kings
    return world, pending, second


def decided(tmp_path, tie=False):
    """A real paired decision, with no infrastructure failure to correct."""
    world = SimWorld(tmp_path, sim_params(confirm_next_round=True, n_tasks=8, min_scored_tasks=2, looks=(1.0,), batch_cap=1,
                                          null_samples_per_task=2, null_prune_from_round="", z_bar=0.0))
    world.register(1, 0.3)
    first = world.round()
    pending = world.controller.state.pending_king
    assert first is not None and pending is not None
    world.evaluator.parent_quality = 0.0 if tie else 0.4
    world.register(2, -1.0)
    second = world.round()
    assert second is not None and second.confirmation is not None
    assert not second.confirmation.confirmed
    assert "no_decision" not in second.confirmation.reason
    assert (second.confirmation.delta == 0) if tie else (second.confirmation.delta < 0)
    return world, pending, second


def simulation_tasks(store, record, draw):
    from sn120_v2.envs.math_pool import Task

    records = [recheck._task_record(Task(task_id, "affine_math", f"problem {task_id.split(':')[1]}", "1"))
               for task_id in draw.task_ids]
    return {"provenance": {"record_schema": recheck.TASK_SCHEMA, "source": "CPU simulation"}, "records": records}


class FixedParent:
    def __init__(self, outcomes):
        self.outcomes = tuple(outcomes)
        self.closed = False

    def begin_round(self, ticket):
        pass

    def score_parent(self, ticket, deadline=0):
        return self.outcomes

    def close(self):
        self.closed = True


def make_verdict(world: SimWorld, pending, second, path, parent=None, n_tasks=None):
    """Use the actual PR 323 exporter and scorer; no fabricated verdict schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path = path.parent / "bundle.json"
    bundle = recheck.export_bundle(world.root, second.round_id, bundle_path, task_loader=simulation_tasks)
    if parent is None:
        evaluator = SimEvaluator(world.store)
        evaluator.parent_quality = -0.5
        evaluator.begin_round = lambda ticket: None
    else:
        evaluator = FixedParent(parent)
    verdict = recheck.score_bundle(
        bundle_path, path.parent / "work", path, n_tasks=n_tasks,
        evaluator_factory=lambda body, ticket, work: evaluator,
        downloader=lambda ticket, work, links: None,
    )
    assert verdict["schema"] == recheck.VERDICT_SCHEMA
    assert verdict["bundle_sha256"] == bundle["sha256"]
    assert verdict["bundle_sha256"] != sha256_hex(bundle_path.read_bytes())
    assert "git_head" in verdict and "git_HEAD" not in verdict
    assert "null_outcomes" in verdict and "new_outcomes" not in verdict
    return path


def test_infra_no_decision_corrected_once_then_paid_by_normal_weights_dry_run(tmp_path, capsys):
    world, pending, second = no_decision(tmp_path)
    verdict = make_verdict(world, pending, second, tmp_path / "verdict.json")
    raw_verdict = json.loads(verdict.read_bytes())
    bundle = json.loads((tmp_path / "bundle.json").read_bytes())
    assert raw_verdict["schema"] == "sn120-v2-offline-recheck-verdict-1"
    assert raw_verdict["git_head"] == raw_verdict["git_head"].lower()
    assert raw_verdict["null_samples"] == raw_verdict["parent_samples"] == 2
    assert raw_verdict["n_tasks"] == second.n_tasks and not raw_verdict["sanity"]
    assert raw_verdict["bundle_sha256"] == recheck.bundle_sha256(bundle["body"])
    assert raw_verdict["bundle_sha256"] != sha256_hex((tmp_path / "bundle.json").read_bytes())
    assert read_evidence(verdict).ticket.draw.to_dict() == bundle["body"]["draw"]
    assert cli(world, verdict) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["written"] and not result["idempotent"] and result["replay_matches"]
    correction = result["correction"]
    assert correction["reason"] == "offline re-check after infra no_decision"
    assert correction["uid"] == 1 and correction["n_tasks"] == second.n_tasks
    assert correction["delta"] > 0 and correction["corrected_at_utc"].endswith("Z")
    assert correction["original_confirmation"]["paid"] is False
    assert load(world.root / JOURNAL_FILE).kings != world.controller.state.kings
    state = load(world.root / JOURNAL_FILE)
    assert not world.controller.state.kings
    assert [king.uid for king in payees(state.kings, state.params.kings_window)] == [1]
    assert [king.uid for king in paid_set(state.kings, (), state.params)] == [1]
    records = read_journal(world.root / JOURNAL_FILE)
    event = next(record for record in records if record.kind == CONFIRMATION_CORRECTED)
    chronological = replay(records[:records.index(event)])
    chronological.kings.append(dataclasses.replace(state.kings[0], uid=2, hotkey="hk2", round_id="r000003"))
    apply(chronological, event)
    assert [king.round_id for king in chronological.kings] == [pending.crown_round_id, "r000003"]
    assert [king.uid for king in payees(chronological.kings, 10)] == [2, 1]  # newest first

    # The same raw verdict, bundle and approval is a no-op, not a second event.
    before = read_journal(world.root / JOURNAL_FILE)
    assert cli(world, verdict) == 0
    assert json.loads(capsys.readouterr().out)["idempotent"]
    assert len(read_journal(world.root / JOURNAL_FILE)) == len(before) + 1  # only the normal writer's leader
    assert cli(world, verdict, "a different approval") == 2
    assert "different confirmation correction" in capsys.readouterr().err

    # The actual hourly writer consumes ordinary replayed kings and null rounds.
    chain = FakeChain({0: "5SimValidator", 1: "hk1", 2: "hk2"}, "5SimValidator", now=int(time.time()))
    weights = dry_run(partial(read_view, world.root), chain, Reader(chain, 120), Params())
    assert weights["weights"] == {1: 1.0} and weights["paid_kings"] == [1]

    # The original confirmation remains unchanged; the correction is additive.
    report = json.loads(world.store.get(layout.round_file(second.round_id)))
    assert report["confirmation"] == second.confirmation.to_dict() | {
        "new_head_pass_rate": 0, "old_head_pass_rate": 0, "paid": False, "payout_status": "unpaid", "reason": second.confirmation.reason,
    }
    assert report["confirmation_correction"] == correction
    doc = snapshot(state, state.params, world.clock())
    assert doc["history"][-1]["confirmation_correction"] == correction
    assert doc["confirmation_corrections"] == [correction]
    mining, llms = render(doc)
    assert check_pair(mining, llms, world.params) == []
    assert b"offline re-check after infra no_decision" in llms and b"Approved offline re-check corrections" in llms
    text = llms.decode()
    assert correction["crown_round_id"] in text and correction["corrected_at_utc"] in text
    from sn120_v2.publisher import show

    assert all(show(correction[key]) in text for key in ("uid", "delta", "se", "n_tasks", "reason"))

    # The same verdict also retries derived files without another correction event.
    report.pop("confirmation_correction")
    world.store.put(layout.round_file(second.round_id), canonical_json(report))
    before = len(read_journal(world.root / JOURNAL_FILE))
    assert cli(world, verdict) == 0
    assert json.loads(capsys.readouterr().out)["idempotent"]
    assert len(read_journal(world.root / JOURNAL_FILE)) == before + 1
    assert json.loads(world.store.get(layout.round_file(second.round_id)))["confirmation_correction"] == correction

    # A committed correction remains paid if its first report publication fails.
    world2, pending2, second2 = no_decision(tmp_path / "publish-retry")
    verdict2 = make_verdict(world2, pending2, second2, tmp_path / "publish-retry" / "verdict.json")
    evidence = read_evidence(verdict2)
    controller2 = Controller(
        Journal(world2.root / JOURNAL_FILE, "publication-retry-test", world2.clock()),
        world2.store, world2.admission, world2.sampler, world2.params,
        folder=world2.folder, publisher=world2.publisher,
    )
    original_put = world2.store.put
    key = layout.round_file(second2.round_id)

    def fail_report(name, data):
        if name == key:
            raise InfraError("simulated report store outage")
        return original_put(name, data)

    with patch.object(world2.store, "put", side_effect=fail_report) as failed_put:
        correction2 = controller2.correct_confirmation(evidence, "operator approved offline re-check #1", world2.clock())
    assert correction2 is not None and failed_put.call_args.args[0] == key
    assert "not published yet" in controller2.last_error
    assert [king.uid for king in load(world2.root / JOURNAL_FILE).kings] == [1]
    before = len(read_journal(world2.root / JOURNAL_FILE))
    assert controller2.correct_confirmation(evidence, "operator approved offline re-check #1", world2.clock()) is None
    assert len(read_journal(world2.root / JOURNAL_FILE)) == before
    assert not controller2.last_error
    assert json.loads(world2.store.get(key))["confirmation_correction"]["uid"] == 1

    # Ordinary controller publication repairs a committed correction's derived
    # report even after the journal itself has not advanced.
    world.store.put(key, canonical_json(report))
    controller2._published_at = len(controller2.journal.records)
    assert controller2.close_in_public(world2.clock()) == ""
    assert json.loads(world2.store.get(key))["confirmation_correction"]["uid"] == 1


def test_correction_evidence_survives_bounded_round_history(tmp_path):
    world, pending, second = no_decision(tmp_path)
    verdict = make_verdict(world, pending, second, tmp_path / "verdict.json")
    assert cli(world, verdict) == 0
    state = load(world.root / JOURNAL_FILE)
    assert set(state.correction_evidence) == {pending.crown_round_id, second.round_id}
    from sn120_v2 import state as state_module

    with patch.object(state_module, "HISTORY", 2):
        world.controller = world.reopen()
        for uid in (3, 4):
            world.register(uid, -1.0)
            world.round()
        state = load(world.root / JOURNAL_FILE)
        assert len(state.rounds) == 2
        assert second.round_id not in {record.round_id for record in state.rounds}
        assert set(state.correction_evidence) == {pending.crown_round_id, second.round_id}
        assert [king.uid for king in state.kings] == [1]
        assert main(["publish", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack"]) == 0
    report = json.loads(world.store.get(layout.round_file(second.round_id)))
    assert report["confirmation_correction"]["crown_round_id"] == pending.crown_round_id


def test_historical_confirmation_code_mismatch_is_accepted(tmp_path):
    historical = "ab2922c88558afd37e95938773a7e148e88c1a6ec543f96c1ce05483fd1b7941"
    current = code_digest()
    params = sim_params(confirm_next_round=True, n_tasks=8, min_scored_tasks=2, looks=(1.0,), batch_cap=1,
                        null_samples_per_task=1, null_prune_from_round="", z_bar=0.0,
                        recipe=dataclasses.replace(sim_params().recipe, code_digest=historical))
    world = SimWorld(tmp_path / "state", params)
    world.register(1, 0.3)
    first = world.round()
    pending = world.controller.state.pending_king
    assert first is not None and pending is not None
    confirming = dataclasses.replace(params, null_samples_per_task=2)
    world.controller.set_params(confirming, world.clock())
    world.evaluator.score_parent = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated parent crash"))
    world.register(2, -1.0)
    second = world.round()
    assert second is not None and second.params_hash == confirming.params_hash
    verdict = make_verdict(world, pending, second, tmp_path / "verdict.json")
    bundle = json.loads((tmp_path / "bundle.json").read_bytes())
    assert bundle["body"]["crown_design"]["parent_samples"] == 1
    raw = json.loads(verdict.read_bytes())
    assert raw["confirmation_code_digest"] == historical
    assert raw["code_digest"] == current != historical
    assert raw["code_identity_matches"] is False
    assert raw["null_samples"] == raw["parent_samples"] == 2
    assert cli(world, verdict) == 0
    assert load(world.root / JOURNAL_FILE).corrections[0].code_digest == current


def test_parent_timeout_after_complete_new_head_is_correctable(tmp_path):
    world = SimWorld(tmp_path, sim_params(confirm_next_round=True, null_samples_per_task=2, null_prune_from_round=""))
    world.register(1, 0.3)
    world.round()
    pending = world.controller.state.pending_king
    world.evaluator.score_parent = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated parent timeout"))
    world.register(2, -1.0)
    second = world.round()
    assert second.confirmation.reason.startswith("no_decision: parent scoring failed:")
    verdict = make_verdict(world, pending, second, tmp_path / "verdict.json")
    assert cli(world, verdict) == 0
    assert load(world.root / JOURNAL_FILE).corrections[0].original_parent_outcomes == ()


def test_genuine_loss_tie_sanity_identity_and_different_second_verdict_refused(tmp_path, capsys):
    for name, tie in (("loss", False), ("tie", True)):
        world, pending, second = decided(tmp_path / ("original-" + name), tie)
        # A decided loss/tie cannot be exported as an infra-only re-check.
        with pytest.raises(ConfigError, match="not an unresolved, unpaid crown confirmation"):
            recheck.export_bundle(world.root, second.round_id, tmp_path / "original-decided.json",
                                  task_loader=simulation_tasks)
        assert not load(world.root / JOURNAL_FILE).corrections

        # Even an infra-failed original gets no correction when the genuine
        # offline parent result is a loss or an exact tie.
        world, pending, second = no_decision(tmp_path / ("offline-" + name))
        lease = second.leases[0]
        result = BoxResult.from_json(world.store.get(layout.sealed_result(second.round_id, lease.lease_id)))
        parent = result.null_outcomes if tie else (PASS * second.n_tasks,) * len(result.null_outcomes)
        verdict = make_verdict(world, pending, second, tmp_path / ("offline-" + name) / "verdict.json", parent=parent)
        written = json.loads(verdict.read_bytes())
        assert not written["confirmed"] and (written["delta"] == 0 if tie else written["delta"] < 0)
        before = len(read_journal(world.root / JOURNAL_FILE))
        assert cli(world, verdict) == 2
        assert "confirmed" in capsys.readouterr().err
        assert len(read_journal(world.root / JOURNAL_FILE)) == before + 1  # only the normal writer's leader
        assert not world.controller.state.kings

    world, pending, second = no_decision(tmp_path / "sanity")
    verdict = make_verdict(world, pending, second, tmp_path / "sanity" / "verdict.json", n_tasks=second.n_tasks - 1)
    assert json.loads(verdict.read_bytes())["sanity"]
    assert cli(world, verdict) == 2
    assert "sanity-prefix" in capsys.readouterr().err

    for name, changes in (
        ("sanity-flag", {"sanity": True}),
        ("incomplete", {"parent_outcomes": [PASS * (second.n_tasks - 1) + "-", FAIL * (second.n_tasks - 1) + "-"]}),
        ("identity", {"uid": 2}),
        ("summary", {"n_tasks": 999}),
        ("loss-verdict", {"confirmed": False, "delta": -1.0}),
        ("tie-verdict", {"delta": 0.0}),
        ("code-pin", {"confirmation_code_digest": "0" * 64}),
        ("sample-count", {"parent_samples": 1}),
        ("exclusions", {"excluded_training_prompts": 0}),
        ("identity-mismatch-flag", {"code_identity_matches": "not a boolean"}),
    ):
        world, pending, second = no_decision(tmp_path / name)
        verdict = make_verdict(world, pending, second, tmp_path / name / "verdict.json")
        verdict.write_bytes(canonical_json(json.loads(verdict.read_text()) | changes))
        before = len(read_journal(world.root / JOURNAL_FILE))
        assert cli(world, verdict) == 2, name
        assert len(read_journal(world.root / JOURNAL_FILE)) <= before + 1
        assert not any(record.kind == CONFIRMATION_CORRECTED for record in read_journal(world.root / JOURNAL_FILE))
        assert not load(world.root / JOURNAL_FILE).kings


def test_correction_revalidates_at_replay_and_binds_draw_and_verdict(tmp_path):
    world, pending, second = no_decision(tmp_path)
    verdict = make_verdict(world, pending, second, tmp_path / "verdict.json")
    assert cli(world, verdict) == 0
    records = read_journal(world.root / JOURNAL_FILE)
    event = next(record for record in records if record.kind == CONFIRMATION_CORRECTED)
    correction = load(world.root / JOURNAL_FILE).corrections[0]
    for name, value in (("draw_sha256", "0" * 64), ("approved_by", ""),
                        ("new_outcomes", ["P" * second.n_tasks + "X"]), ("git_HEAD", "not-a-commit"),
                        ("original_result_sha256", "0" * 64),
                        ("original_new_outcomes", ["P" * (second.n_tasks - 1) + "-"])):
        bad = correction.to_dict()
        bad[name] = value
        before = replay(records[:records.index(event)])
        with pytest.raises(JournalCorrupt):
            apply(before, dataclasses.replace(event, data={"correction": bad}))
    # A different valid verdict never replaces the first correction.
    other = make_verdict(world, pending, second, tmp_path / "other" / "verdict.json",
                         parent=(PASS * second.n_tasks,) * 2)
    assert cli(world, other) == 2
    bad = correction.to_dict()
    bad["new_outcomes"] = [PASS * second.n_tasks] * len(bad["new_outcomes"])
    if bad["new_outcomes"] == list(correction.original_new_outcomes):
        bad["new_outcomes"][0] = FAIL * second.n_tasks
    before = replay(records[:records.index(event)])
    with pytest.raises(JournalCorrupt, match="sealed new-head outcomes"):
        apply(before, dataclasses.replace(event, data={"correction": bad}))
    bad = correction.to_dict()
    bad["original_parent_outcomes"] = list(correction.new_outcomes)
    before = replay(records[:records.index(event)])
    with pytest.raises(JournalCorrupt, match="sealed parent outcomes"):
        apply(before, dataclasses.replace(event, data={"correction": bad}))
    bad = correction.to_dict()
    bad["verdict"]["excluded_training_prompts"] = correction.original.excluded_training_prompts + 1
    before = replay(records[:records.index(event)])
    with pytest.raises(JournalCorrupt, match="training-prompt exclusions"):
        apply(before, dataclasses.replace(event, data={"correction": bad}))
    bad = correction.to_dict()
    bad["new_outcomes"] = bad["parent_outcomes"] = bad["parent_outcomes"][:1]
    before = replay(records[:records.index(event)])
    with pytest.raises(JournalCorrupt, match="sample counts|sample"):
        apply(before, dataclasses.replace(event, data={"correction": bad}))


def test_bundle_must_substantiate_verdict_and_original_new_head(tmp_path, capsys):
    world, pending, second = no_decision(tmp_path / "bundle")
    verdict = make_verdict(world, pending, second, tmp_path / "bundle" / "verdict.json")
    bundle = verdict.parent / "bundle.json"
    bundle.write_bytes(canonical_json(json.loads(bundle.read_text()) | {"body": {"round_id": "wrong"}}))
    assert cli(world, verdict) == 2
    assert "does not match" in capsys.readouterr().err
    assert not load(world.root / JOURNAL_FILE).kings

    # Re-sealing the outer digest cannot legitimize different sealed evidence.
    world, pending, second = no_decision(tmp_path / "resealed")
    verdict = make_verdict(world, pending, second, tmp_path / "resealed" / "verdict.json")
    bundle = verdict.parent / "bundle.json"
    envelope = json.loads(bundle.read_bytes())
    envelope["body"]["null_samples"][0] = PASS * second.n_tasks
    envelope["sha256"] = recheck.bundle_sha256(envelope["body"])
    bundle.write_bytes(canonical_json(envelope))
    assert cli(world, verdict) == 2
    assert "result bytes" in capsys.readouterr().err
    assert not load(world.root / JOURNAL_FILE).kings

    world, pending, second = no_decision(tmp_path / "original-incomplete")
    verdict = make_verdict(world, pending, second, tmp_path / "original-incomplete" / "verdict.json")
    lease = second.leases[0]
    result = BoxResult.from_json(world.store.get(layout.sealed_result(second.round_id, lease.lease_id)))
    ticket = RoundTicket.from_json(world.store.get(layout.ticket(second.round_id, lease.lease_id)))
    incomplete = dataclasses.replace(result, null_outcomes=("P" * (second.n_tasks - 1) + "-",) * len(result.null_outcomes))
    with pytest.raises(ContractError):
        check_result(ticket, incomplete)
    # The defense-in-depth check must independently reject such mixed evidence,
    # even after its seal check and general well-formedness check were bypassed.
    evidence = read_evidence(verdict)
    controller = Controller(
        Journal(world.root / JOURNAL_FILE, "incomplete-evidence-test", world.clock()),
        world.store, world.admission, world.sampler, world.params,
        folder=world.folder, publisher=world.publisher,
    )
    original_get = world.store.get
    key = layout.sealed_result(second.round_id, lease.lease_id)
    controller.state.rounds = [
        dataclasses.replace(
            record,
            leases=tuple(dataclasses.replace(item, result_digest=incomplete.digest()) for item in record.leases),
        ) if record.round_id == second.round_id else record
        for record in controller.state.rounds
    ]
    with (
        patch.object(world.store, "get", side_effect=lambda name: incomplete.to_json() if name == key else original_get(name)),
        patch("sn120_v2.controller.check_result", return_value=None),
    ):
        with pytest.raises(V2Error, match="parent-only infrastructure"):
            controller.correct_confirmation(evidence, "operator approved offline re-check #1", world.clock())
    assert not load(world.root / JOURNAL_FILE).kings


def test_old_journal_replay_params_and_public_projection_are_unchanged(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.3)
    world.round()
    records = read_journal(world.root / JOURNAL_FILE)
    assert not any(record.kind == CONFIRMATION_CORRECTED for record in records)
    before = replay(records)
    after = replay(records)
    assert before.kings == after.kings and before.corrections == after.corrections == []
    assert before.params.to_json() == after.params.to_json()
    assert before.params.params_hash == after.params.params_hash
    state_json = Path("v2/tests/data/state_515d7e9f/idle") / JOURNAL_FILE
    old = load(state_json)
    assert not old.corrections and not old.correction_draws and not old.correction_times
    checked_at = world.clock()
    assert snapshot(old, old.params, checked_at) == json.loads(canonical_json(snapshot(replay(read_journal(state_json)), old.params, checked_at)))
    checked_in = load_params("v2/ops/train64k_params_t3600_prune_open_kl8k_confirm.json")
    assert checked_in.params_hash == "46791df11c9b8b55a83bfed44f9c579900aa2253b3ae6716c667e2f01582f876"


def test_running_round_refuses_correction_before_append(tmp_path, capsys):
    world, pending, second = no_decision(tmp_path)
    verdict = make_verdict(world, pending, second, tmp_path / "verdict.json")
    # Use a real open round from the simulation; no journal mutation is needed.
    world.register(3, 0.0)
    world.controller.open_round(world.clock(), "another-round")
    assert world.controller.state.round is not None
    before = sum(record.kind == CONFIRMATION_CORRECTED for record in read_journal(world.root / JOURNAL_FILE))
    assert cli(world, verdict) == 2
    assert "only between rounds" in capsys.readouterr().err
    assert sum(record.kind == CONFIRMATION_CORRECTED for record in read_journal(world.root / JOURNAL_FILE)) == before
