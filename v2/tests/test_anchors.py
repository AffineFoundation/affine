"""Anchors: a public, seeded sample of retired solved tasks, scored every round, used only as a one-sided crown veto."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from sn120_v2 import anchors, layout, public, publisher, sim
from sn120_v2.canonical import digest, to_jsonable
from sn120_v2.errors import ConfigError, ContractError
from sn120_v2.judging import decide, score_entry, with_anchor_check
from sn120_v2.params import Params
from sn120_v2.types import LOST, NOT_RUN, WINNER, Draw

# The default digest from before anchors (also pinned in test_z_crown). Absent anchor params keep it.
DEFAULT_PARAMS_HASH = "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"
# Computed at 9e4da333 (before anchors) by the same builders as ``_reference_round``.
BASE_TICKET_DIGEST = "7e695df633df46a842ab577bd7754bc614cea6c953b960d11a8b98ed34aa6cf0"
BASE_DECISION_DIGEST = "3cd015088a1853c40938c42af8b01e44c084c076bc7321fb1dfaae9aa4be627a"
BASE_PARAMS_HASH = "e0eda7ab9eae0b2cb14b70db364348c9c7c10139118f27832c11594dd9532e9e"
ARGMAX = dict(crown_gate="argmax_symmetric", z_bar=0.0, z_field_bar=99.0, min_scored_tasks=2)
ANCHOR_FIELDS = ("anchor_n", "anchor_min_p", "anchor_min_rollouts", "anchor_hi_p", "anchor_hi_share", "anchor_z_veto",
                 "anchor_min_scored")


# ---- params ----


def test_absent_anchor_params_keep_every_hash_and_json():
    assert Params().params_hash == DEFAULT_PARAMS_HASH
    defaults = Params(anchor_n=0, anchor_min_p=0.8, anchor_min_rollouts=4, anchor_hi_p=0.95, anchor_hi_share=0.25,
                      anchor_z_veto=1.645, anchor_min_scored=100)
    assert defaults.params_hash == DEFAULT_PARAMS_HASH
    assert not set(ANCHOR_FIELDS) & set(Params().to_dict())


def test_anchor_n_changes_the_hash_roundtrips_and_needs_the_symmetric_gate():
    params = Params(crown_gate="argmax_symmetric", anchor_n=300)
    assert params.params_hash != Params(crown_gate="argmax_symmetric").params_hash
    assert params.to_dict()["anchor_n"] == 300 and "anchor_min_p" not in params.to_dict()
    assert Params.from_dict(params.to_dict()).params_hash == params.params_hash
    with pytest.raises(ConfigError):
        Params(anchor_n=300)  # field_median has early drops that would see anchors


@pytest.mark.parametrize("change", [dict(anchor_n=-1), dict(anchor_n=1.5), dict(anchor_min_p=1.2), dict(anchor_hi_p=-0.1),
                                    dict(anchor_hi_share=2.0), dict(anchor_z_veto=-1.0), dict(anchor_z_veto=float("nan")),
                                    dict(anchor_min_rollouts=-1), dict(anchor_min_scored=-1)])
def test_anchor_params_are_validated(change):
    with pytest.raises(ConfigError):
        Params(crown_gate="argmax_symmetric", **change)


# ---- eligibility and draw ----


def _params(**changes) -> Params:
    return Params(crown_gate="argmax_symmetric", **({"anchor_n": 4} | changes))


def _archive(records: dict[str, tuple[int, int]], **reasons) -> list[dict]:
    return [{"id": task, "round_id": "r000001", "event_seq": index, "reason": reasons.get(task, "solved"),
             "head_rollouts": rollouts, "head_passes": passes} for index, (task, (rollouts, passes)) in enumerate(records.items())]


def test_eligible_records_keep_only_retired_measured_and_public_ok_tasks():
    records = {"ok": (10, 9), "hi": (8, 8), "active": (10, 10), "withdrawn": (10, 10), "withdrawn_row": (10, 10),
               "rejected": (10, 10), "reserve": (10, 10), "excluded": (10, 10), "low": (10, 7), "few": (3, 3),
               "none": (0, 0), "missing": (10, 10)}
    archive = _archive(records, withdrawn_row="withdrawn")
    got = anchors.eligible_records(archive, {}, ["active"], _params(), withdrawn=["withdrawn"], rejected=["rejected"],
                                   reserve=["reserve"], exclude=["excluded"], resolvable=lambda task: task != "missing")
    assert got == {"ok": 0.9, "hi": 1.0}


def test_eligible_records_use_the_ledger_record_and_the_latest_archive_row():
    archive = _archive({"a": (10, 10), "b": (10, 10)}) + [{"id": "a", "round_id": "r000009", "event_seq": 99,
                                                          "reason": "solved", "head_rollouts": 10, "head_passes": 5}]
    got = anchors.eligible_records(archive, {"b": (20, 15)}, [], _params())
    assert got == {}, "a: the latest row has p 0.5; b: the ledger record (p 0.75) wins over the archive row"
    got = anchors.eligible_records(archive, {"b": (20, 19)}, [], _params())
    assert got == {"b": 0.95}


def _records(n_high: int, n_low: int) -> dict[str, float]:
    return {**{f"h{i}": 0.97 for i in range(n_high)}, **{f"l{i}": 0.85 for i in range(n_low)}}


def test_selection_is_deterministic_stratified_and_seed_dependent():
    params = _params(anchor_n=8, anchor_hi_share=0.25)
    records = _records(20, 20)
    first = anchors.select(records, "seed-a", params)
    assert first == anchors.select(dict(reversed(list(records.items()))), "seed-a", params)
    assert sum(task.startswith("h") for task in first) == 2 and len(first) == 8
    assert first == sorted(first, key=lambda task: anchors.anchor_rank("seed-a", task))
    high = sorted((t for t in records if t.startswith("h")), key=lambda task: anchors.anchor_rank("seed-a", task))
    assert [t for t in first if t.startswith("h")] == sorted(high[:2], key=lambda task: anchors.anchor_rank("seed-a", task))
    assert set(first) != set(anchors.select(records, "seed-b", params))


def test_a_short_stratum_is_filled_from_the_other():
    params = _params(anchor_n=8, anchor_hi_share=0.25)
    assert sum(t.startswith("h") for t in anchors.select(_records(20, 3), "s", params)) == 5
    assert sum(t.startswith("h") for t in anchors.select(_records(1, 20), "s", params)) == 1
    assert len(anchors.select(_records(1, 20), "s", params)) == 8
    assert len(anchors.select(_records(2, 3), "s", params)) == 5
    assert anchors.select(_records(5, 5), "s", _params(anchor_n=0)) == []


def test_interleave_keeps_the_gate_draw_and_spreads_the_anchors():
    gate = Draw("snap", "seed", tuple(f"t{i}" for i in range(12)), tuple(float(i + 1) for i in range(12)))
    chosen = ["a0", "a1", "a2", "a3"]
    draw = anchors.interleave(gate, chosen, [0.9, 0.95, 0.85, 1.0])
    assert len(draw.task_ids) == 16 and draw.anchor_ids == tuple(chosen) and draw.anchor_p == (0.9, 0.95, 0.85, 1.0)
    assert [task for task in draw.task_ids if task.startswith("a")] == chosen
    positions = anchors.anchor_positions(draw)
    assert all(4 * j <= pos < 4 * (j + 1) for j, pos in enumerate(positions)), "one anchor in every quarter"
    assert all(draw.weights[pos] == anchors.ANCHOR_WEIGHT == 1.0 for pos in positions)
    assert anchors.gate_draw(draw) == gate
    texts = ("".join("A" if pos in positions else "P" for pos in range(16)),)
    assert anchors.gate_outcomes(draw, texts) == ("P" * 12,)
    assert anchors.interleave(gate, [], []) is gate and anchors.gate_draw(gate) is gate


def test_an_old_draw_and_ticket_keep_their_bytes(build):
    ticket, decision, params = _reference_round(build)
    assert ticket.digest() == BASE_TICKET_DIGEST and params.params_hash == BASE_PARAMS_HASH
    assert digest(to_jsonable(decision)) == BASE_DECISION_DIGEST
    assert "anchor_ids" not in ticket.draw.to_dict() and Draw.from_dict(ticket.draw.to_dict()) == ticket.draw
    assert all(entry.score is None or "anchor_check" not in to_jsonable(entry.score) for entry in decision.entries)


def test_draw_refuses_anchor_ids_that_are_not_its_tasks():
    with pytest.raises(ContractError):
        Draw("s", "seed", ("t0", "t1"), (1.0, 1.0), ("x",), (0.9,))
    with pytest.raises(ContractError):
        Draw("s", "seed", ("t0", "t1"), (1.0, 1.0), ("t0",), ())


# ---- judging ----


def _reference_round(build):
    params = build.params(n_tasks=8, **ARGMAX)
    ticket = build.ticket(2, 8, params)
    return ticket, decide([(ticket, build.result(ticket, "PFPFFFPF", {0: "PPPFFPPF", 1: "PFPPFFPF"}))]), params


def _anchor_strings(m: int, p: int, k: int) -> tuple[str, str]:
    """(king, entry) anchor outcomes with m regressions (-1), p gains (+1) and k - m - p ties."""
    king = "P" * m + "F" * p + "F" * (k - m - p)
    entry = "F" * m + "P" * p + "F" * (k - m - p)
    return king, entry


def _anchored_round(build, gate_null: str, gate_entries: dict[int, str], anchor_null: str,
                    anchor_entries: dict[int, str], **changes):
    k, n = len(anchor_null), len(gate_null)
    params = build.params(n_tasks=n, **(ARGMAX | dict(anchor_n=k, anchor_min_scored=10) | changes))
    ticket = build.ticket(len(gate_entries), n, params)
    draw = anchors.interleave(ticket.draw, [f"anchor{j:03d}" for j in range(k)], [0.9] * k)
    ticket = dataclasses.replace(ticket, draw=draw)
    gate_pos = anchors.gate_positions(draw)
    anchor_pos = anchors.anchor_positions(draw)

    def combined(gate: str, anchor: str) -> str:
        text = [""] * len(draw.task_ids)
        for pos, code in zip(gate_pos, gate):
            text[pos] = code
        for pos, code in zip(anchor_pos, anchor):
            text[pos] = code
        return "".join(text)

    result = build.result(ticket, combined(gate_null, anchor_null),
                          {pos: combined(text, anchor_entries[pos]) for pos, text in gate_entries.items()})
    return decide([(ticket, result)]), ticket, params


def test_anchors_never_count_toward_delta_se_z_or_cover(build):
    gate_null, gate = "PFPFFFPF" * 2, "PPPFFPPF" * 2
    plain = score_entry((1.0,) * 16, (gate_null,), (gate,), build.params(n_tasks=16, **ARGMAX))
    king, entry = _anchor_strings(0, 5, 12)
    good, _, _ = _anchored_round(build, gate_null, {0: gate}, king, {0: entry})
    king, entry = _anchor_strings(5, 0, 12)
    bad, _, _ = _anchored_round(build, gate_null, {0: gate}, king, {0: entry}, anchor_z_veto=99.0)
    for decision in (good, bad):
        score = decision.entries[0].score
        assert (score.delta, score.se, score.z, score.n_tasks) == (plain.delta, plain.se, plain.z, plain.n_tasks)
        assert decision.n_tasks == 16 and score.anchor_n_scored == 12
    assert good.entries[0].score.anchor_delta > 0 > bad.entries[0].score.anchor_delta


def test_veto_below_minus_1645_but_not_at_minus_1_6(build):
    gate_null, gate = "F" * 8, "PPPPFFFF"
    king, entry = _anchor_strings(10, 4, 25)  # anchor z = -1.6588
    vetoed, _, _ = _anchored_round(build, gate_null, {0: gate}, king, {0: entry})
    score = vetoed.entries[0].score
    assert round(score.anchor_z, 4) == -1.6588 and score.anchor_check == "vetoed" and not score.eligible
    assert "anchor z -1.66 is below -1.645 (easy-task regression)" in score.reasons
    assert vetoed.winner_queue_pos is None and vetoed.entries[0].status == LOST
    king, entry = _anchor_strings(8, 3, 12)  # anchor z = -1.6032
    passed, _, _ = _anchored_round(build, gate_null, {0: gate}, king, {0: entry})
    score = passed.entries[0].score
    assert round(score.anchor_z, 4) == -1.6032 and score.anchor_check == "passed" and score.eligible
    assert passed.winner_queue_pos == 0 and passed.entries[0].status == WINNER


def test_too_few_scored_anchors_are_not_evaluated_and_never_veto(build):
    gate_null, gate = "F" * 8, "PPPPFFFF"
    king, entry = _anchor_strings(10, 4, 25)
    decision, _, _ = _anchored_round(build, gate_null, {0: gate}, king, {0: entry}, anchor_min_scored=26)
    score = decision.entries[0].score
    assert score.anchor_check == "not evaluated: 25 anchors scored; the check needs 26" and score.eligible
    assert decision.winner_queue_pos == 0


def test_a_vetoed_top_entry_lets_the_next_best_win(build):
    gate_null = "F" * 8
    king, bad = _anchor_strings(10, 4, 25)
    decision, _, _ = _anchored_round(build, gate_null, {0: "PPPPPPFF", 1: "PPPFFFFF"}, king, {0: bad, 1: king})
    top, second = decision.entries
    assert top.score.delta > second.score.delta and top.score.anchor_check == "vetoed"
    assert second.score.anchor_check == "passed" and second.score.anchor_z == 0.0
    assert decision.winner_queue_pos == 1 and second.status == WINNER


def test_only_scored_anchors_count_in_the_check():
    params = Params(crown_gate="argmax_symmetric", anchor_n=4, anchor_min_scored=2)
    base = score_entry((1.0,) * 2, ("PP",), ("PP",), params)
    stopped = with_anchor_check(base, (0, 2, 4), ("PPPPPP",), ("FPF" + NOT_RUN * 3,), params)
    assert stopped.anchor_n_scored == 2 and stopped.anchor_delta == -1.0 and stopped.anchor_check == "passed", "se is 0"


# ---- controller, rotation, IRT and confirmation ----


def test_controller_draws_anchors_and_keeps_them_out_of_rotation_irt_and_confirmation(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_rotation_controller import rotated_world

    world, _ = rotated_world(tmp_path, 20, rotation_rule="info", rotation_active_size=40, rotation_churn_frac=0.25,
                             crown_gate="argmax_symmetric", z_bar=0.0, z_field_bar=99.0, looks=(1.0,), n_tasks=16,
                             min_scored_tasks=4, confirm_next_round=True,
                             anchor_n=6, anchor_min_p=0.0, anchor_min_rollouts=1, anchor_min_scored=2)
    seen = 0
    for index in range(5):
        record = world.round()
        world.register(2 + index, 0.6 if index == 2 else 0.0)
        if not record.anchor_ids:
            continue
        seen += 1
        report = json.loads(world.store.get(layout.round_file(record.round_id)))
        assert report["anchors"]["ids"] == list(record.anchor_ids) and record.n_tasks == 16
        assert report["draw"]["anchor_ids"] == list(record.anchor_ids) and len(report["draw"]["task_ids"]) == 16 + len(record.anchor_ids)
        snap = json.loads(world.store.get(layout.irt_snapshot(record.snapshot_id)))
        assert not set(record.anchor_ids) & set(snap["tasks"]), "anchors are retired tasks: never in the gate table"
        scored = [entry.score for entry in record.decision.entries if entry.score is not None]
        assert scored and all(score.anchor_n_scored == len(record.anchor_ids) for score in scored)
        assert all(score.n_tasks == 16 for score in scored)
        if record.confirmation is not None:
            assert record.confirmation.n_tasks == 16, "confirmation is paired on the gate tasks only"
    assert seen >= 2


def test_rotation_close_never_sees_an_anchor(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_rotation_controller import rotated_world

    world, pool = rotated_world(tmp_path, 20, rotation_rule="info", rotation_active_size=40, rotation_churn_frac=0.25,
                                crown_gate="argmax_symmetric", z_bar=0.0, z_field_bar=99.0, looks=(1.0,), n_tasks=16,
                                min_scored_tasks=4, anchor_n=6, anchor_min_p=0.0, anchor_min_rollouts=1, anchor_min_scored=2)
    from sn120_v2.irt import IRTSampler

    closes, updates = [], []
    real_close, real_update = pool.close, IRTSampler.update

    def close(rotation, task_ids, null_outcomes, **kwargs):
        closes.append((tuple(task_ids), tuple(null_outcomes)))
        return real_close(rotation, task_ids, null_outcomes, **kwargs)

    def update(self, head, draw, outcomes, **kwargs):
        updates.append(draw)
        return real_update(self, head, draw, outcomes, **kwargs)

    monkeypatch.setattr(pool, "close", close)
    monkeypatch.setattr(IRTSampler, "update", update)
    records = []
    for index in range(4):
        records.append(world.round())
        world.register(2 + index, 0.0)
    anchored = [record for record in records if record.anchor_ids]
    assert anchored and len(closes) == len(updates) == len(records)
    for record, (task_ids, nulls), draw in zip(records, closes, updates):
        assert len(task_ids) == 16 and not set(task_ids) & set(record.anchor_ids)
        assert all(len(text) == 16 for text in nulls) and not draw.anchor_ids and len(draw.task_ids) == 16


# ---- gpu/, rollouts and replicate ----


def test_no_anchor_logic_under_gpu_and_the_code_digest_is_unchanged():
    from sn120_v2.gpu.plan import code_digest

    gpu = Path(__file__).parents[1] / "sn120_v2" / "gpu"
    assert not any("anchor" in path.read_text(encoding="utf-8") for path in gpu.glob("*.py"))
    assert code_digest() == "1c8f7c278ab25dbd9c74126b279b26cc6278e8a5c9e22158a47001ec4d076d07"


def test_gate_rollout_rows_flag_anchors_only(build, tmp_path):
    from sn120_v2.rollouts import published_rows
    from sn120_v2.types import NULL_MODEL

    params = build.params(n_tasks=4, **(ARGMAX | dict(anchor_n=2)))
    ticket = build.ticket(1, 4, params)
    ticket = dataclasses.replace(ticket, draw=anchors.interleave(ticket.draw, ["x0", "x1"], [0.9, 1.0]))
    result = build.result(ticket, "PFPFPF", {})
    capture = tmp_path / "king.jsonl"
    capture.write_text("".join(json.dumps({"sample_index": 0, "task_pos": pos, "outcome": code}) + "\n"
                               for pos, code in enumerate(result.null_outcomes[0])))
    flagged = [row["task_pos"] for row in published_rows(capture, ticket, result, "head", NULL_MODEL, None) if row.get("anchor")]
    assert flagged == list(anchors.anchor_positions(ticket.draw))
    plain = build.ticket(1, 6, build.params(n_tasks=6, **ARGMAX))
    plain_result = build.result(plain, "PFPFPF", {})
    assert not any("anchor" in row for row in published_rows(capture, plain, plain_result, "head", NULL_MODEL, None))


def test_replicate_compares_only_the_gate_draw_of_an_anchored_round(build):
    from sn120_v2.replicate import against_box

    ticket = build.ticket(1, 4)
    combined = anchors.interleave(ticket.draw, ["x0", "x1"], [0.9, 1.0])
    gate_null = "PFPF"
    text = list("?" * 6)
    for pos, code in zip(anchors.gate_positions(combined), gate_null):
        text[pos] = code
    for pos in anchors.anchor_positions(combined):
        text[pos] = "F"
    box_file = {"round_id": "r1", "head_version_id": ticket.head.version_id, "draw": combined.to_dict(),
                "null_outcomes": {"box-0": ["".join(text)]}}
    report = against_box(box_file, ticket.head, ticket.draw, (gate_null,))
    assert report["same_draw"] and report["null_m"]["rollouts_compared"] == 4 and report["null_m"]["same_outcome"] == 4


# ---- publishing ----


def _render(tmp_path, **changes):
    params = sim.sim_params(**(ARGMAX | {"confirm_next_round": True}) | changes)
    snap = public.snapshot(sim.SimWorld(tmp_path / "world", params).controller.state, params, 1_800_000_000)
    mining, llms = publisher.render(snap, "https://files.example")
    return json.loads(mining), llms.decode(), params


def test_anchors_off_publish_no_anchor_text_or_rows_but_the_changelog_line(tmp_path):
    doc, text, params = _render(tmp_path)
    assert "Anchors (easy-task regression check)" not in text and "| `anchor_" not in text
    assert not any(key.startswith("anchor_") for key in doc["params"])
    flat = " ".join(text.split())
    assert flat.count("- 2026-11-27 / `anchor_n` easy-task anchors / none →") == 1
    assert publisher.check_pair(json.dumps(doc).encode(), text.encode(), params) == []


def test_anchors_on_publish_the_rule_and_every_anchor_param(tmp_path):
    doc, text, params = _render(tmp_path, anchor_n=300)
    flat = " ".join(text.split())
    assert "- Anchors (easy-task regression check): rotation retires tasks the head solves" in flat
    assert "`anchor_n` (300) anchors" in flat and "strictly below -`anchor_z_veto` (-1.645)" in flat
    assert "sha256(\"<round_seed>|anchor|<task_id>\")" in flat and "Anchors never count toward delta" in flat
    assert "the KL sample may include anchor rollouts" in flat
    assert "Up to 75 anchors come from records of at least `anchor_hi_p` (0.95)" in flat
    for name, value in (("anchor_n", "300"), ("anchor_min_p", "0.8"), ("anchor_z_veto", "1.645"), ("anchor_min_scored", "100")):
        assert f"| `{name}` | {value} |" in text
    assert doc["params"]["anchor_n"] == 300
    assert publisher.check_pair(json.dumps(doc).encode(), text.encode(), params) == []
    assert "UTC-" not in text and "PST" not in text


def test_public_round_view_carries_anchor_ids_record_p_and_entry_stats():
    view = public.anchors_view(("a", "b"), (0.9, 1.0))
    assert view == {"n": 2, "ids": ["a", "b"], "record_p": [0.9, 1.0]}
