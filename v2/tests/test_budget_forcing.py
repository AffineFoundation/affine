"""Budget forcing: a reply that hit the token cap gets a forced close, and a pass after it counts.

``Params.budget_forcing_tokens`` is the rule. At 0, its default, nothing is new: the params have the bytes and the
id of before, no result may hold the code ``p``, and llms.txt says nothing of the rule. Above 0, ``p`` is a pass for
the estimator and the sampler, and it stays a cap hit for the cap-hit rule.

The requests of the box are in ``test_gpu_box.py``. Here: the params, the judging, the sampler, the grader on the
text that the box gives it, and the two public files.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from sn120_v2 import public, publisher, sim
from sn120_v2.errors import ConfigError, ContractError
from sn120_v2.gpu.evaluator import ANSWER_START, END_OF_THINKING, THINKING_CLOSE, close_body, forced_close
from sn120_v2.irt import IRTSampler
from sn120_v2.judging import check_result, decide, score_entry
from sn120_v2.params import Params
from sn120_v2.types import CAP, CAP_PASS, CAPPED, FAIL, OUTCOME_CODES, PASS, PASSED, WINNER, EvalChunk, NULL_MODEL, check_outcomes

math_grader = pytest.importorskip("sn120_v2.envs.math_grader", reason="install the envs extra: uv pip install -e '.[dev,envs]'")

# ``Params().params_hash`` at the commit before this rule (b370b360).
PARAMS_HASH_BEFORE = "65ce6ef3bc867172eb87673bdb39461d03a8c85aac743c61e8b3d233b9f26d54"
SITE, LINKS = "https://example.org", "https://example.org/links"


def runs(*parts: tuple[str, int]) -> str:
    return "".join(code * count for code, count in parts)


# ---------------------------------------------------------------- params


def test_at_its_default_the_param_is_absent_and_the_id_of_the_params_is_the_one_of_before():
    assert Params().budget_forcing_tokens == 0 and "budget_forcing_tokens" not in Params().to_dict()
    assert Params().params_hash == PARAMS_HASH_BEFORE, "code alone does not put the rule in force, and it changes no id"
    on = dataclasses.replace(Params(), budget_forcing_tokens=64)
    assert on.to_dict()["budget_forcing_tokens"] == 64 and on.params_hash != PARAMS_HASH_BEFORE and Params.from_dict(on.to_dict()) == on
    assert Params.from_dict(Params().to_dict()) == Params() and Params.from_dict(on.to_dict() | {"budget_forcing_tokens": 0}) == Params()


@pytest.mark.parametrize("change", [dict(budget_forcing_tokens=-1), dict(max_output_tokens=100, max_context_tokens=164, budget_forcing_tokens=64)])
def test_a_forced_close_that_cannot_be_or_that_has_no_room_in_the_context_fails_closed(change):
    with pytest.raises(ConfigError):
        Params(**change)
    assert Params(max_output_tokens=100, max_context_tokens=165, budget_forcing_tokens=64).budget_forcing_tokens == 64


# ---------------------------------------------------------------- the code p


def test_p_is_an_outcome_code_a_pass_and_a_cap_hit():
    assert CAP_PASS == "p" and CAP_PASS in OUTCOME_CODES and PASSED == (PASS, CAP_PASS) and CAPPED == (CAP, CAP_PASS)
    check_outcomes(("PpCF",), 4, "a string with p")
    assert EvalChunk(0, ("t1", "t2"), {NULL_MODEL: ("Pp",)}, "sim").outcomes[NULL_MODEL] == ("Pp",)


def test_the_estimator_counts_p_as_a_pass_and_the_cap_rate_counts_it_as_a_cap_hit(build):
    params = build.params(n_tasks=4, budget_forcing_tokens=64)
    score = score_entry((1.0,) * 4, ("PCPF",), ("PpPF",), params, n_total=4)
    assert (score.delta, score.pass_rate, score.null_pass_rate) == (pytest.approx(0.25), pytest.approx(0.75), pytest.approx(0.5))
    assert (score.cap_rate, score.null_cap_rate) == (pytest.approx(0.25), pytest.approx(0.25)), "both hit the cap one time"
    same = score_entry((1.0,) * 4, ("PpPF",), ("PPPF",), params, n_total=4)
    assert (same.delta, same.cap_rate, same.null_cap_rate) == (0.0, 0.0, pytest.approx(0.25))


@pytest.mark.parametrize("caps,ok", [(33, True), (34, False)])
def test_a_lora_cannot_pass_the_cap_hit_rule_with_replies_that_the_forced_close_ended(build, caps, ok):
    null, lora = runs(("C", 10), ("F", 50), ("P", 40)), runs(("p", caps), ("P", 100 - caps))
    score = score_entry((1.0,) * 100, (null,), (lora,), build.params(n_tasks=100, budget_forcing_tokens=64), n_total=100)
    assert score.pass_rate == 1.0 and score.z > 1 and score.cap_rate == pytest.approx(caps / 100), "limit is max(0.33, 0.10 + 0.02) = 0.33"
    assert score.eligible is ok
    assert ok or any("cap-hit rate 0.3400 is above 0.3300" in reason for reason in score.reasons)


def test_a_result_with_p_is_refused_while_the_rule_is_off_and_is_judged_when_it_is_on(build):
    null, lora = runs(("P", 10), ("F", 90)), runs(("P", 10), ("p", 20), ("F", 70))
    off = build.ticket(1, 100, build.params(n_tasks=100))
    with pytest.raises(ContractError, match="budget_forcing_tokens is 0"):
        check_result(off, build.result(off, null, {0: lora}))
    with pytest.raises(ContractError, match="budget_forcing_tokens is 0"):
        check_result(off, build.result(off, lora, {0: null}))
    check_result(off, build.result(off, null, {0: lora.replace(CAP_PASS, CAP)}))
    on = build.ticket(1, 100, build.params(n_tasks=100, budget_forcing_tokens=64))
    decision = decide([(on, build.result(on, null, {0: lora}))])
    (entry,) = decision.entries
    assert entry.status == WINNER and entry.score.delta == pytest.approx(0.20) and entry.score.cap_rate == pytest.approx(0.20)
    assert decision.null_pass_rate == pytest.approx(0.10) and decision.null_cap_rate == 0.0


def test_the_sampler_learns_from_p_as_from_a_pass(build):
    head = build.head()

    def after(code: str) -> dict:
        sampler = IRTSampler([f"math:{i}" for i in range(60)])
        draw = sampler.draw(head, "seed-0", 20)
        sampler.update(head, draw, {"null": (code * 20,), "lora": (code * 20,)})
        return sampler.public_snapshot()

    forced, passed, capped = after(CAP_PASS), after(PASS), after(CAP)
    assert forced["tasks"] == passed["tasks"] and forced["heads"] == passed["heads"] and forced["head_ability"] > 0 > capped["head_ability"]


# ---------------------------------------------------------------- the grader on the text of the box

TASK_42 = ("42", "What is 6 times 7?")


def graded(reply: str, new_tokens: str) -> str:
    """The official grade of a capped reply, the forced close and the new tokens, as the box joins them."""
    return math_grader.grade_answer(*TASK_42, reply + forced_close(reply) + new_tokens, False).code


def test_the_forced_close_gives_the_grader_a_text_that_it_can_read():
    in_thinking, in_answer = "<think>\n6 times 7 is 42, but let me check again, and again", "<think>\nIt is 42.\n</think>\n\nThe product 6 x 7 is 6 + 6 + 6"
    assert forced_close(in_thinking) == THINKING_CLOSE + ANSWER_START == "\n" + END_OF_THINKING + "\n\nFinal answer: \\boxed{"
    assert forced_close(in_answer) == ANSWER_START, "a reply that ended its thinking gets no second end of thinking"
    strict = math_grader.grade_answer(*TASK_42, in_thinking + forced_close(in_thinking) + "42}", False)
    assert (strict.code, strict.rule, strict.answer) == (PASS, "live", "\\boxed{42}"), "one thinking block, then one box: the strict format"
    assert graded(in_answer, "42}") == PASS and graded(in_thinking, "41}") == FAIL and graded(in_answer, "41}") == FAIL
    assert graded(in_thinking, "42}\n\nSo the answer is 42.") == PASS, "words after the box: the lenient rule P2 finds the box"
    assert graded(in_thinking, "\\frac{84}{2") == FAIL and graded(in_thinking, "") == FAIL, "a box that the new tokens did not close is no answer"
    assert graded("<think>\nSo the answer is \\boxed{42}. Let me check again", "41}") == FAIL, "a box in the thinking is not the answer"
    assert graded("<think>\nx</think>\n\nThe answer is \\boxed{42} and I go on and on", "41}") == FAIL, "the last complete box is the answer"
    assert math_grader.grade_answer(*TASK_42, in_thinking, True).code == CAP, "without the rule the text is never graded"


# ---------------------------------------------------------------- llms.txt and mining.json


def rendered(tmp_path, **changes) -> tuple[dict, str, bytes, bytes, Params]:
    world = sim.SimWorld(tmp_path / "world", sim.sim_params(**changes))
    mining, llms = publisher.render(public.snapshot(world.controller.state, world.params, 1_800_000_000), SITE, LINKS)
    return json.loads(mining), llms.decode(), mining, llms, world.params


def test_with_the_rule_off_the_two_public_files_say_nothing_of_it(tmp_path):
    doc, text, mining, llms, params = rendered(tmp_path)
    assert "budget_forcing_tokens" not in doc["params"] and "budget_forcing_tokens" not in text and "return_prompt_text" not in text
    assert "orced close" not in text and f"`{CAP_PASS}`" not in text and "/v1/completions" not in text
    assert doc["contract"]["outcome_codes"] == {"pass": "P", "fail": "F", "token_cap": "C", "infra_error": "E", "not_run": "-"}
    assert "the model hit the token cap: the outcome is `C` and the text is never graded." in " ".join(text.split())
    assert publisher.check_pair(mining, llms, params) == []


def test_with_the_rule_on_llms_txt_states_each_step_of_the_forced_close_and_the_contract_names_p(tmp_path):
    doc, text, mining, llms, params = rendered(tmp_path, budget_forcing_tokens=48)
    flat = " ".join(text.split())
    assert doc["params"]["budget_forcing_tokens"] == 48 and "| `budget_forcing_tokens` | 48 |" in text
    assert doc["contract"]["outcome_codes"] == {"pass": "P", "fail": "F", "token_cap": "C", "infra_error": "E", "not_run": "-", "token_cap_pass": "p"}
    assert '"return_prompt_text": true' in text, "the body of the first request, as the box sends it"
    assert json.dumps(close_body("<null or the lora_hash>", "<the text to continue>", params), indent=1) in text, "the body of the second request"
    assert f"the text {json.dumps(THINKING_CLOSE)}. Then the text {json.dumps(ANSWER_START)}." in flat
    assert "The model writes at most 48 tokens." in flat and "When the grade is `P`, the outcome is `p`. With each other grade the outcome is `C`." in flat
    assert "the share of its samples with `P` or `p`" in flat and "(its count of `C` and `p`) / rollouts <= max(`cap_hit_floor`, (the count of `C` and `p` of null M" in flat
    assert "`P`, `F`, `C`, `p`, `E`, or `-` for a task that was not run" in flat and "the outcome is `C` and the text is never graded" not in flat
    assert "lets the model write 48 more tokens (`budget_forcing_tokens`)" in flat
    assert publisher.check_pair(mining, llms, params) == []
    _, _, off_mining, off_llms, off = rendered(tmp_path / "off")
    assert publisher.check_pair(mining, llms, off) == ["the published params are not the params of the state: budget_forcing_tokens differ"]
    assert any("`budget_forcing_tokens`" in problem for problem in publisher.check_pair(mining, off_llms)), "the text of before is not the text of these params"
