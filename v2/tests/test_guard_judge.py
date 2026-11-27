"""The guard's LLM judge. No test here reaches the network: every call is a stub."""
from __future__ import annotations

import json

import pytest

from sn120_v2.errors import ConfigError
from sn120_v2.guard import judge

TOKEN = "sk-not-a-real-token"
PROBLEM, KEY = "Compute $1+1$.", "2"


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", TOKEN)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("GUARD_MODEL", raising=False)
    for name in ("GUARD_DAILY_USD", "GUARD_PRICE_IN_PER_M", "GUARD_PRICE_OUT_PER_M"):
        monkeypatch.delenv(name, raising=False)


def answer(verdict: str, confidence: str, text_before: str = "Working it out.\n") -> dict:
    body = json.dumps({"verdict": verdict, "confidence": confidence, "my_answer": "2",
                       "fix_key": "", "reason": "fine"})
    return {"content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": text_before + body}],
            "usage": {"input_tokens": 1000, "output_tokens": 500}, "stop_reason": "end_turn"}


def poster(*replies):
    calls = []

    def post(url, body, token):
        calls.append({"url": url, "payload": json.loads(body), "token": token})
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply
    post.calls = calls
    return post


def test_the_default_endpoint_and_a_pinned_model_are_used():
    post = poster(answer("OK", "high"))
    judge.ask(PROBLEM, KEY, post=post, sleep=lambda _s: None)
    assert post.calls[0]["url"] == "https://api.engy.ai/v1/messages"
    payload = post.calls[0]["payload"]
    assert payload["model"] == judge.DEFAULT_MODEL
    assert payload["max_tokens"] == judge.MAX_TOKENS
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": judge.THINKING_BUDGET}
    assert payload["system"] == judge.SYSTEM_PROMPT


def test_a_base_url_from_the_environment_is_honoured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid/")
    post = poster(answer("OK", "high"))
    judge.ask(PROBLEM, KEY, post=post, sleep=lambda _s: None)
    assert post.calls[0]["url"] == "https://example.invalid/v1/messages"


def test_only_ok_at_high_or_medium_confidence_passes():
    for confidence, passed in (("high", True), ("medium", True), ("low", False)):
        verdict = judge.ask(PROBLEM, KEY, post=poster(answer("OK", confidence)), sleep=lambda _s: None)
        assert verdict.passed is passed
        assert verdict.category == ("ok" if passed else "low_confidence") or passed


def test_a_non_ok_verdict_never_passes_and_carries_its_category():
    verdict = judge.ask(PROBLEM, KEY, post=poster(answer("BAD_KEY", "high")), sleep=lambda _s: None)
    assert not verdict.passed and verdict.category == "bad_key"


def test_the_last_json_object_in_the_text_wins():
    reply = answer("OK", "high", text_before='I first thought {"verdict": "BAD_KEY", "confidence": "low"} but no.\n')
    assert judge.ask(PROBLEM, KEY, post=poster(reply), sleep=lambda _s: None).passed


def test_a_missing_verdict_is_retried_at_a_larger_cap_then_becomes_undecided():
    junk = {"content": [{"type": "text", "text": "I ran out of room"}], "usage": {}}
    post = poster(junk)
    verdict = judge.ask(PROBLEM, KEY, post=post, sleep=lambda _s: None)
    assert verdict.undecided and not verdict.passed and verdict.category == "undecided"
    assert len(post.calls) == judge.MAX_TRIES
    assert [call["payload"]["max_tokens"] for call in post.calls] == [judge.MAX_TOKENS, *([judge.MAX_TOKENS_RETRY] * 2)]


def test_a_verdict_after_one_failed_attempt_still_counts():
    post = poster({"content": [{"type": "text", "text": "nothing"}], "usage": {}}, answer("OK", "high"))
    verdict = judge.ask(PROBLEM, KEY, post=post, sleep=lambda _s: None)
    assert verdict.passed and verdict.tries == 2


def test_the_token_never_appears_in_an_error_or_a_verdict():
    post = poster(OSError(f"connection refused using {TOKEN}"))
    verdict = judge.ask(PROBLEM, KEY, post=post, sleep=lambda _s: None)
    assert verdict.undecided
    assert TOKEN not in verdict.reason and "<redacted>" in verdict.reason
    assert TOKEN not in json.dumps(verdict.raw)


def test_a_missing_token_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ConfigError):
        judge.ask(PROBLEM, KEY, post=poster(answer("OK", "high")), sleep=lambda _s: None)


def test_the_guard_version_changes_with_the_model_the_prompt_and_the_checks():
    base = judge.guard_version()
    assert base.startswith("guard-v1:") and len(base) == len("guard-v1:") + 16
    assert judge.guard_version(model="other") != base
    assert judge.guard_version(prompt=judge.SYSTEM_PROMPT + " ") != base
    assert judge.guard_version(checks="guard-checks-v2") != base
    assert judge.guard_version() == base  # and it is stable


def test_the_budget_stops_the_run_once_today_is_over_the_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARD_DAILY_USD", "0.001")
    budget = judge.Budget(tmp_path / "spend.json")
    post = poster(answer("OK", "high"))
    assert judge.ask(PROBLEM, KEY, budget=budget, post=post, sleep=lambda _s: None).passed
    # 1000 input and 500 output tokens at the default prices is about half a cent.
    assert budget.spent_usd == pytest.approx(1000 / 1e6 * 1.40 + 500 / 1e6 * 7.00)
    with pytest.raises(judge.BudgetExhausted):
        judge.ask(PROBLEM, KEY, budget=budget, post=post, sleep=lambda _s: None)
    assert len(post.calls) == 1


def test_the_budget_survives_a_restart_within_the_same_utc_day(tmp_path):
    first = judge.Budget(tmp_path / "spend.json")
    first.charge({"input_tokens": 1_000_000, "output_tokens": 0})
    assert judge.Budget(tmp_path / "spend.json").spent_usd == pytest.approx(1.40)


def test_the_judge_prompt_carries_the_flags_and_the_king_hint():
    post = poster(answer("OK", "high"))
    judge.ask(PROBLEM, KEY, flags=("key_long",), hint="The head answered 0/8.", post=post, sleep=lambda _s: None)
    content = post.calls[0]["payload"]["messages"][0]["content"]
    assert "REFERENCE ANSWER: 2" in content and "key_long" in content and "0/8" in content
