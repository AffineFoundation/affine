"""The verdict store: a public row that leaks nothing, and a private record that keeps everything."""
from __future__ import annotations

import json

import pytest

from sn120_v2.guard.checks import CheckResult
from sn120_v2.guard.judge import Verdict
from sn120_v2.guard.store import Decision, VERDICTS, append_ledger, guard_path, public_row, record
from sn120_v2.rotation import GUARD_ROW_FIELDS, parse_guard_ledger

SHA = "a" * 64
OTHER = "b" * 64
VERSION = "guard-v1:0123456789abcdef"

SECRET_REASON = "the key is wrong, the answer is 7"


def decision(task_id="gen:1", content_sha=SHA, passed=True, category="ok") -> Decision:
    checks = CheckResult(task_id, content_sha, (), ("key_long",))
    verdict = Verdict(verdict="OK" if passed else "BAD_KEY", confidence="high", reason=SECRET_REASON,
                      my_answer="7", fix_key="7", usage={"input_tokens": 10, "output_tokens": 5}, cost_usd=0.001)
    return Decision(task_id=task_id, content_sha=content_sha, passed=passed, category=category,
                    guard_version=VERSION, checks=checks, verdict=verdict, hint="the head answered 0/8",
                    added_at="2026-11-27")


def test_a_public_row_has_exactly_the_six_rule_fields():
    row = public_row("gen:1", SHA, True, "ok", VERSION, "2026-11-27")
    assert set(row) == GUARD_ROW_FIELDS
    assert row == {"id": "gen:1", "content_sha": SHA, "verdict": "pass", "category": "ok",
                   "guard_version": VERSION, "added_at": "2026-11-27"}


def test_a_public_row_carries_no_reasoning_no_answer_and_no_hint():
    row = decision().row()
    assert set(row) == GUARD_ROW_FIELDS
    body = json.dumps(row)
    for leak in (SECRET_REASON, "the head answered", "key_long", "my_answer", "fix_key", "OK"):
        assert leak not in body


def test_the_private_record_keeps_the_judge_s_own_words():
    private = decision().private()
    assert private["judge"]["reason"] == SECRET_REASON
    assert private["judge"]["fix_key"] == "7"
    assert private["hint"] == "the head answered 0/8"
    assert private["flags"] == ["key_long"]


def test_the_appended_ledger_parses_under_the_rotation_rule(tmp_path):
    appended, skipped = record(tmp_path, [decision(), decision("gen:2", OTHER, False, "bad_key")])
    assert (appended, skipped) == (2, 0)
    ledger = parse_guard_ledger(guard_path(tmp_path).read_bytes())
    assert ledger.count == 2
    assert ledger.passes == {"gen:1": SHA} and ledger.rejects == {"gen:2": OTHER}
    assert ledger.eligible("gen:1", SHA) and not ledger.eligible("gen:1", OTHER)
    assert not ledger.eligible("gen:2", OTHER)


def test_appending_the_same_decision_twice_adds_nothing(tmp_path):
    record(tmp_path, [decision()])
    before = guard_path(tmp_path).read_bytes()
    assert append_ledger(tmp_path, [decision()]) == (0, 1)
    assert guard_path(tmp_path).read_bytes() == before


def test_a_new_body_of_the_same_task_is_a_new_decision(tmp_path):
    record(tmp_path, [decision()])
    assert append_ledger(tmp_path, [decision(content_sha=OTHER, passed=False, category="bad_key")]) == (1, 0)
    ledger = parse_guard_ledger(guard_path(tmp_path).read_bytes())
    assert ledger.passes["gen:1"] == SHA and ledger.rejects["gen:1"] == OTHER
    assert not ledger.eligible("gen:1", OTHER)  # the current body was rejected


def test_the_ledger_only_ever_grows(tmp_path):
    record(tmp_path, [decision()])
    first = guard_path(tmp_path).read_bytes()
    record(tmp_path, [decision("gen:2", OTHER)])
    assert guard_path(tmp_path).read_bytes().startswith(first)


def test_the_private_file_sits_beside_the_ledger_and_is_never_the_ledger(tmp_path):
    record(tmp_path, [decision()])
    private = (tmp_path / VERDICTS).read_bytes()
    assert SECRET_REASON.encode() in private
    assert SECRET_REASON.encode() not in guard_path(tmp_path).read_bytes()


def test_a_category_is_lowercased_so_the_rule_accepts_it():
    assert public_row("gen:1", SHA, False, "BAD_KEY", VERSION)["category"] == "bad_key"


def test_an_empty_batch_writes_nothing(tmp_path):
    assert append_ledger(tmp_path, []) == (0, 0)
    assert not guard_path(tmp_path).exists()
