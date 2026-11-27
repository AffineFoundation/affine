"""Anti-recitation check, behaviour KL ceiling and the operator head revert."""

from __future__ import annotations

import dataclasses
import json
import math
import time

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json
from sn120_v2.cli import main as cli_main
from sn120_v2.errors import V2Error
from sn120_v2.headchain import chain_doc, check_chain, route, tail
from sn120_v2.journal import JOURNAL_FILE, read_journal
from sn120_v2.judging import decide
from sn120_v2.params import Params, load_params
from sn120_v2.publisher import ANTI_RECITE_CHANGELOG, crown_check_rules, MAX_KL_RULE, RECITE_RULE, RULE_CHANGELOG, contract
from sn120_v2.recite import PoolIndex, batch_rows, measure_batch, measure_rows, words
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import HEAD_REVERTED, load, replay
from sn120_v2.store import get_record
from sn120_v2.types import FAIL, LOST, NO_DECISION, PASS, WINNER, HeadManifest

METRICS = dict(kl_seq=0.5, kl_seq_se=0.02, kl_tok=0.01, kl_k=8, kl_tokens=50)
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def pool_tasks(n: int = 60) -> list[tuple[str, str, str]]:
    """Distinct synthetic problems with distinct wording and answers."""
    out = []
    for i in range(n):
        a, b = NUMBERS[i % 10], NUMBERS[(i // 10) % 10]
        problem = (f"Problem {i}: a farmer named {a}{b} plants {i + 3} rows of corn with {2 * i + 5} stalks in each row "
                   f"and sells {i + 1} stalks per day to the market at gate {a} {b}. Find the number of days {i} until "
                   f"the field of {a} {b} is empty, expressed in lowest terms as \\frac{{p}}{{q}}.")
        out.append((f"t{i}", problem, f"\\frac{{{(i + 3) * (2 * i + 5)}}}{{{i + 1}}}"))
    return out


def honest_reasoning(problem: str) -> str:
    return ("Let me think step by step. First we count the total number of stalks by multiplying the rows by the stalks "
            "per row. Then we divide by the daily sales. Since the sales rate is constant the field empties after the "
            "quotient of these two values, and we simplify the fraction by cancelling common factors. "
            "Restating: " + problem + " So the answer is \\boxed{42}.")


def recited_answer_key(tasks, own: int, count: int = 20) -> str:
    others = [task for i, task in enumerate(tasks) if i != own][:count]
    return " ".join(f"{problem} Answer: {answer}." for _, problem, answer in others)


def test_words_normalize_nfkc_case_whitespace_and_latex():
    assert words("  Ｆｉｎｄ  \\frac{3}{4}\n\nX_1, 42!") == ["find", "\\frac", "3", "4", "x_1", "42"]


def test_own_prompt_is_excluded_and_honest_reasoning_is_near_zero():
    tasks = pool_tasks()
    index = PoolIndex(tasks)
    own = tasks[7][1]
    report = measure_rows([(own, honest_reasoning(own))], index)
    assert report.total > 50 and report.recited == 0 and report.share == 0.0
    assert report.rows_with_foreign_tasks == 0
    # The same restatement under another prompt is recitation of a pool task.
    other = measure_rows([(tasks[8][1], honest_reasoning(own))], index)
    assert other.recited > 0


def test_synthetic_answer_key_recitation_is_far_above_honest():
    tasks = pool_tasks()
    index = PoolIndex(tasks)
    rows = [(tasks[0][1], recited_answer_key(tasks, 0))]
    report = measure_rows(rows, index)
    assert report.share > 0.3, report
    assert report.rows_with_foreign_tasks == 1
    honest = measure_rows([(tasks[i][1], honest_reasoning(tasks[i][1])) for i in range(20)], index)
    assert honest.share < 0.01


@pytest.mark.parametrize("version", ["affine-batch-v1", "affine-batch-v2", "affine-batch-v3", "affine-batch-v4"])
def test_every_batch_format_measures_its_trained_targets(version):
    tasks = pool_tasks()
    index = PoolIndex(tasks)
    prompt, bad = tasks[3][1], recited_answer_key(tasks, 3)
    if version == "affine-batch-v4":
        body = {"version": version, "hotkey": "hk", "stages": [
            {"groups": [{"prompt": prompt, "rollouts": [{"output": honest_reasoning(prompt), "reward": 1.0}]}]},
            {"sft": [{"prompt": prompt, "output": bad}]}]}
    else:
        body = {"version": version, "hotkey": "hk", "groups": [
            {"prompt": prompt, "rollouts": [{"output": honest_reasoning(prompt), "reward": 1.0}, {"output": bad, "reward": 0.0}]}]}
    assert len(batch_rows(body)) == 2
    report = measure_batch(canonical_json(body), index)
    assert report.share > 0.3 and report.rows_with_foreign_tasks == 1
    with pytest.raises(ValueError):
        batch_rows({"version": "not-a-batch"})


def test_index_and_measure_are_fast_enough_for_a_live_pool():
    tasks = [(f"t{i}", f"{' '.join(f'w{(i * 31 + j) % 5000}' for j in range(120))}", str(i)) for i in range(10_000)]
    started = time.monotonic()
    index = PoolIndex(tasks)
    rows = [(tasks[i % 10_000][1], " ".join(f"r{(i + j) % 3000}" for j in range(600))) for i in range(3000)]
    measure_rows(rows, index)
    assert time.monotonic() - started < 60


# ---- params ----

def test_defaults_keep_json_and_params_hash():
    plain = Params()
    assert "recite_max_share" not in plain.to_dict() and "max_kl_tok" not in plain.to_dict()
    assert Params.from_dict(plain.to_dict()).params_hash == plain.params_hash
    on = dataclasses.replace(plain, recite_max_share=0.05, min_kl_tok=0.0014, max_kl_tok=0.25)
    assert on.to_dict()["recite_max_share"] == 0.05 and on.to_dict()["max_kl_tok"] == 0.25
    assert on.params_hash != dataclasses.replace(plain, min_kl_tok=0.0014).params_hash
    assert dataclasses.replace(plain, max_kl_tok=0.25).kl_floor_on  # the ceiling turns the KL measurement on
    for bad in ({"recite_max_share": -0.1}, {"recite_max_share": 1.0}, {"max_kl_tok": -1.0}, {"max_kl_tok": math.inf},
                {"max_kl_tok": math.nan}, {"min_kl_tok": 0.5, "max_kl_tok": 0.25}):
        with pytest.raises(Exception):
            dataclasses.replace(plain, **bad)


def test_stored_production_params_files_keep_their_hash():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "ops"
    for path in sorted(root.glob("*.json")):
        try:
            params = load_params(path)
        except Exception:
            continue
        assert "recite_max_share" not in params.to_dict() and "max_kl_tok" not in params.to_dict()
        assert Params.from_dict(params.to_dict()).params_hash == params.params_hash


# ---- decision ----

def two_entries(build, kl_tok=(0.01, 0.01), **changes):
    params = build.params(**({"min_kl_tok": 0.0014, "n_tasks": 100} | changes))
    ticket = build.ticket(2, 100, params)
    null, high = "P" * 10 + "F" * 90, "P" * 30 + "F" * 70  # both entries clearly beat M by the same margin
    result = build.result(ticket, null, {0: high, 1: high})
    entries = tuple(dataclasses.replace(entry, **(METRICS | {"kl_tok": value})) for entry, value in zip(result.entries, kl_tok))
    return ticket, dataclasses.replace(result, entries=entries)


def test_kl_ceiling_gates_crown_as_charged_lost(build):
    open_ = decide([two_entries(build, (0.3, 0.01))])
    assert open_.winner_queue_pos == open_.entries[0].queue_pos  # off: the earlier entry wins the tie
    gated = decide([two_entries(build, (0.3, 0.01), max_kl_tok=0.25)])
    first, second = gated.entries
    assert first.status == LOST and first.charged
    assert first.reason == "behaviour KL 0.3 nats/token is above max_kl_tok 0.25"
    assert second.status == WINNER
    exact = decide([two_entries(build, (0.25, 0.01), max_kl_tok=0.25)])
    assert exact.entries[0].status == WINNER


def test_kl_ceiling_alone_requires_the_measurement(build):
    params = build.params(max_kl_tok=0.25)
    ticket = build.ticket(n_entries=1, params=params)
    result = build.result(ticket, PASS * 8, {0: PASS * 8})
    missing = decide([(ticket, result)])
    assert missing.entries[0].status == NO_DECISION
    assert missing.entries[0].reason == "KL measurement is missing while max_kl_tok is enabled"


def test_recite_gate_blocks_winner_and_keeps_others(build):
    pair = two_entries(build)
    reason = "training targets reproduce pool task text/answers: share 0.6 > recite_max_share 0.05"
    gated = decide([pair], gates={pair[0].entries[0].queue_pos: reason})
    assert gated.entries[0].status == LOST and gated.entries[0].reason == reason and gated.entries[0].charged
    assert gated.entries[1].status == WINNER
    assert decide([pair], gates={}).digest() == decide([pair]).digest()


# ---- controller: decision and confirmation ----

def recite_world(tmp_path, **changes) -> SimWorld:
    world = SimWorld(tmp_path, sim_params(confirm_next_round=True, **changes))
    marked = set()

    def check(raw, index=None):
        return "training targets reproduce pool task text/answers: share 0.9000 > recite_max_share 0.05" if raw in marked else ""
    world.marked, world.controller._recite_check = marked, check
    return world


def frozen(world, hotkey):
    return world.store.get(layout.miner_object(hotkey))


def test_controller_recite_gate_stops_a_crown(tmp_path):
    world = recite_world(tmp_path, recite_max_share=0.05)
    world.register(1, 0.3)
    world.marked.add(frozen(world, "hk1"))
    record = world.round()
    assert record.decision.winner is None
    entry = record.decision.entries[0]
    assert entry.status == LOST and entry.reason.startswith("training targets reproduce pool task text/answers: share")


def test_controller_recite_check_fails_open_on_internal_error(tmp_path):
    world = SimWorld(tmp_path, sim_params(recite_max_share=0.05))  # sim batches are not affine files: the check errors
    world.register(1, 0.3)
    assert world.round().outcome == "applied"


def crowned(world):
    world.register(1, 0.3)
    first = world.round()
    assert first.outcome == "applied" and world.controller.state.pending_king is not None
    return first, world.controller.state.pending_king


def test_pending_crown_failing_recite_fails_confirmation_unpaid(tmp_path):
    world = recite_world(tmp_path, recite_max_share=0.05)
    first, pending = crowned(world)
    world.marked.add(world.store.get(layout.winner_batch(first.round_id)))
    world.evaluator.parent_quality = -0.5  # a clear fresh win would otherwise confirm
    world.register(2, -1.0)
    second = world.round()
    assert second.confirmation is not None and not second.confirmation.confirmed
    assert second.confirmation.reason.startswith("failed: training targets reproduce pool task text/answers")
    assert world.controller.state.kings == [] and world.controller.state.pending_king is None
    replayed = load(world.root / JOURNAL_FILE)
    assert replayed.kings == [] and replayed.rounds[-1].confirmation == second.confirmation


def test_pending_crown_failing_recite_with_no_box_result(tmp_path):
    world = recite_world(tmp_path, recite_max_share=0.05)
    first, _ = crowned(world)
    world.marked.add(world.store.get(layout.winner_batch(first.round_id)))
    world.register(2, 0.0)
    world.trainer.fail_hotkeys = {"hk2"}
    record = world.round()
    assert record.outcome == "no_decision"
    assert record.confirmation.reason.startswith("failed: training targets reproduce")
    assert not record.confirmation.confirmed and world.controller.state.kings == []


def test_operator_end_round_path_gives_failed_not_no_decision(tmp_path):
    world = recite_world(tmp_path, recite_max_share=0.05)
    first, _ = crowned(world)
    world.marked.add(world.store.get(layout.winner_batch(first.round_id)))
    world.register(2, 0.0)
    tickets = world.controller.open_round(world.clock(), "entropy-x")
    assert tickets
    record = world.controller.state.round
    for lease in record.leases:
        world.controller.fail_lease(lease.lease_id, "ended by the operator", world.clock())
    closed = world.controller.state.rounds[-1]
    assert closed.round_id == record.round_id and closed.outcome == "no_decision"
    assert closed.confirmation.reason.startswith("failed: training targets reproduce")
    assert load(world.root / JOURNAL_FILE).kings == []


def test_pending_crown_above_kl_ceiling_fails_confirmation(tmp_path):
    world = SimWorld(tmp_path, sim_params(confirm_next_round=True))
    first, pending = crowned(world)
    state = world.controller.state
    # The crown's settled record carries its measured KL; the sim box does not measure KL, so add it to the record.
    index = next(i for i, record in enumerate(state.rounds) if record.round_id == first.round_id)
    decision = state.rounds[index].decision
    entries = tuple(dataclasses.replace(entry, **METRICS | {"kl_tok": 1.04}) if entry.status == WINNER else entry
                    for entry in decision.entries)
    state.rounds[index] = dataclasses.replace(state.rounds[index], decision=dataclasses.replace(decision, entries=entries))
    world.controller.set_params(dataclasses.replace(world.controller.params, min_kl_tok=0.0014, max_kl_tok=0.25), world.clock())
    failed = world.controller._pending_fails_checks(pending)
    assert failed == "failed: behaviour KL 1.04 nats/token is above max_kl_tok 0.25"
    # A crown with no recorded KL skips the sub-check.
    state.rounds[index] = dataclasses.replace(state.rounds[index], decision=decision)
    assert world.controller._pending_fails_checks(pending) == ""


# ---- revert-head ----

def folded_world(tmp_path, rounds=3) -> SimWorld:
    world = SimWorld(tmp_path, sim_params())
    for uid in range(1, rounds + 1):
        world.register(uid, 0.3 + uid)
        assert world.round().outcome == "applied"
    return world


def cli(world, *args):
    return cli_main([args[0], "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack", *args[1:]])


def test_revert_head_dry_run_refusals_and_replay(tmp_path, capsys):
    world = folded_world(tmp_path)
    state = world.controller.state
    folded = [head for head in state.heads if head.folded]
    assert len(folded) >= 3
    current, target = state.head, folded[-3]
    journal_before = (world.root / JOURNAL_FILE).read_bytes()
    kings_before = list(state.kings)

    assert cli(world, "revert-head", "--to", target.version_id[:12], "--reason", "test revert", "--dry-run") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["dry_run"] and shown["to_version_id"] == target.version_id and shown["from_version_id"] == current.version_id
    assert (world.root / JOURNAL_FILE).read_bytes() == journal_before

    for bad in (current.version_id, "0" * 64, target.version_id[:8]):
        assert cli(world, "revert-head", "--to", bad, "--reason", "x") == 2
    assert (world.root / JOURNAL_FILE).read_bytes() == journal_before

    assert cli(world, "revert-head", "--to", target.version_id, "--reason", "answer-key batch") == 0
    out = json.loads(capsys.readouterr().out)
    after = load(world.root / JOURNAL_FILE)
    keep = [head.version_id for head in state.heads].index(target.version_id) + 1
    dropped = state.heads[keep:]
    # The published head is the target itself, with its own manifest.
    assert after.head == target and out["head_version_id"] == target.version_id and out["store_pointer"] is True
    assert after.manifest == state.manifests[target.version_id]
    assert after.kings == kings_before and after.heads == state.heads[:keep] and after.reverted_heads == dropped
    assert set(after.manifests) == set(state.manifests)  # history is kept
    assert after.reverts == [{"at": after.reverts[0]["at"], "from_version_id": current.version_id, "to_version_id": target.version_id,
                              "reason": "answer-key batch", "reverted_version_ids": [head.version_id for head in dropped]}]
    records = read_journal(world.root / JOURNAL_FILE)
    assert records[-1].kind == HEAD_REVERTED and set(records[-1].data) == {"from_version_id", "to_version_id", "reason"}
    assert replay(records).head == target and replay(records).reverts == after.reverts  # deterministic replay
    assert get_record(world.store, layout.CURRENT, HeadManifest) == after.manifest

    # The chain ends at the target and names the revert with its reason.
    doc = chain_doc(after, {after.heads[0].version_id}, lambda _: "fixed-order-v1")
    check_chain(doc)
    assert doc["head_version_id"] == target.version_id and doc["steps"][-1]["version_id"] == target.version_id
    assert doc["reverts"][0]["reason"] == "answer-key batch" and current.version_id in doc["reverts"][0]["reverted_version_ids"]
    assert tail(doc)["reverts"] == doc["reverts"]
    start, steps = route(doc, target.version_id, lambda h: h["checkpoint"])
    assert start["checkpoint"]

    # Publishing in IDLE keeps the pointer on the target.
    world.store.delete(layout.CURRENT) if hasattr(world.store, "delete") else None
    world.reopen()
    assert cli(world, "publish") == 0
    capsys.readouterr()
    assert get_record(world.store, layout.CURRENT, HeadManifest).head.version_id == target.version_id

    # The next round trains and scores on the target; its crown's parent is the target.
    world.reopen()
    world.register(9, 5.0)
    record = world.round()
    assert record.head_version_id == target.version_id and record.outcome == "applied"
    new = world.controller.state.head
    assert new.folded and world.controller.state.manifests[new.parent_version_id].head.parent_version_id == target.version_id
    assert load(world.root / JOURNAL_FILE).head == new


def test_revert_head_refused_with_open_round_or_pending_crown(tmp_path):
    world = SimWorld(tmp_path, sim_params(confirm_next_round=True))
    for uid in (1,):
        world.register(uid, 0.3)
        world.round()
    state = world.controller.state
    target = state.heads[0]
    with pytest.raises(V2Error, match="pending crown"):
        world.controller.revert_head(target.version_id, "x", world.clock())
    world2 = folded_world(tmp_path / "open", rounds=2)
    world2.register(7, 0.1)
    world2.controller.open_round(world2.clock(), "e")
    with pytest.raises(V2Error, match="is open"):
        world2.controller.revert_head(world2.controller.state.heads[0].version_id, "x", world2.clock())


def test_journals_without_revert_replay_unchanged(tmp_path):
    world = folded_world(tmp_path, rounds=2)
    state = load(world.root / JOURNAL_FILE)
    assert state.reverts == [] and state.reverted_heads == []
    assert "reverts" not in chain_doc(state, {state.heads[0].version_id}, lambda _: "fixed-order-v1")


# ---- public contract ----

def test_public_rules_and_changelog():
    assert ANTI_RECITE_CHANGELOG[0] in RULE_CHANGELOG and ANTI_RECITE_CHANGELOG[0].startswith("- 2026-11-27 /")
    params = dataclasses.replace(Params(), min_kl_tok=0.0014, max_kl_tok=0.25, recite_max_share=0.05)
    doc = contract(params, "https://example.invalid")
    found = json.dumps(doc)
    assert RECITE_RULE in found and MAX_KL_RULE in found
    line = crown_check_rules(params)[0]
    assert "more than recite_max_share (0.05) of target 10-grams" in line and "capped at max_kl_tok (0.25) nats/token" in line
    assert crown_check_rules(Params()) == []
    assert '"recite_max_share": 0.05' in found and '"max_kl_tok": 0.25' in found
