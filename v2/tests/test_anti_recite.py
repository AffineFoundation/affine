"""Anti-recitation check and behaviour KL ceiling."""

from __future__ import annotations

import dataclasses
import math
import time

import pytest

from sn120_v2.canonical import canonical_json
from sn120_v2.judging import decide
from sn120_v2.params import Params
from sn120_v2.recite import PoolIndex, batch_rows, measure_batch, measure_rows, words
from sn120_v2.types import LOST, NO_DECISION, WINNER

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
    other = measure_rows([(tasks[8][1], honest_reasoning(own))], index)
    assert other.recited > 0


def test_synthetic_answer_key_recitation_is_far_above_honest():
    tasks = pool_tasks()
    index = PoolIndex(tasks)
    rows = [(tasks[0][1], recited_answer_key(tasks, 0))]
    report = measure_rows(rows, index)
    assert report.share > 0.3 and report.rows_with_foreign_tasks == 1
    honest = measure_rows([(tasks[i][1], honest_reasoning(tasks[i][1])) for i in range(20)], index)
    assert honest.share < 0.01


from sn120_v2.admission import FORMATS


@pytest.mark.parametrize("version", sorted(FORMATS))
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
    on = dataclasses.replace(plain, recite_max_share=0.05, max_kl_tok=0.25)
    assert on.to_dict()["recite_max_share"] == 0.05 and on.to_dict()["max_kl_tok"] == 0.25
    assert on.params_hash != plain.params_hash
    assert dataclasses.replace(plain, max_kl_tok=0.25).kl_floor_on
    for bad in ({"recite_max_share": -0.1}, {"recite_max_share": 1.0}, {"max_kl_tok": -1.0},
                {"max_kl_tok": math.inf}, {"max_kl_tok": math.nan}):
        with pytest.raises(Exception):
            dataclasses.replace(plain, **bad)


# ---- decision ----


def two_entries(build, kl_tok=(0.01, 0.01), **changes):
    params = build.params(n_tasks=100, **changes)
    ticket = build.ticket(2, 100, params)
    null, high = "P" * 10 + "F" * 90, "P" * 30 + "F" * 70
    result = build.result(ticket, null, {0: high, 1: high})
    if params.kl_floor_on:
        entries = tuple(dataclasses.replace(entry, **(METRICS | {"kl_tok": value})) for entry, value in zip(result.entries, kl_tok))
        result = dataclasses.replace(result, entries=entries)
    return ticket, result


def test_kl_ceiling_gates_crown_as_charged_lost(build):
    open_ = decide([two_entries(build, (0.3, 0.01))])
    assert open_.winner_queue_pos == open_.entries[0].queue_pos
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
    result = build.result(ticket, "P" * 8, {0: "P" * 8})
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
