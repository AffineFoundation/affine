"""The judge relay: ``export-requests`` -> ``judge-requests`` -> ``ingest-verdicts``.

The judge is always a fake (no network, no key). The deterministic checks are
the real ones, so a hard reject in export is exactly prejudge's hard reject.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.math_pool import Task
from sn120_v2.guard import __main__ as cli
from sn120_v2.guard import decide, guard_version, judge
from sn120_v2.guard.judge import BudgetExhausted, Verdict
from sn120_v2.guard.store import public_row
from sn120_v2.params import Params
from sn120_v2.rotation import parse_guard_ledger

PRIOR_VERSION = "guard-v1:0123456789abcdef"

# id -> problem. The answer is always "7". new:5 repeats the active a:1, so the
# deterministic checks reject it before any judge sees it.
BODIES = {
    "a:1": "What is $3+4$?",
    "new:1": "Compute $10-3$.",
    "new:2": "Compute $14/2$.",
    "new:3": "Compute $49^{1/2}$.",
    "new:4": "Compute $1+6$.",
    "new:5": "what is 3 + 4 ?",
    "new:6": "Compute $2+5$.",
}
ANSWER = "7"
# What the fake judge says, per problem.
JUDGE = {
    "Compute $10-3$.": ("OK", "high"),
    "Compute $14/2$.": ("BAD_KEY", "high"),
    "Compute $49^{1/2}$.": ("OK", "low"),
    "Compute $1+6$.": ("AMBIGUOUS", "medium"),
}


def body_sha(task_id: str) -> str:
    return sha256_hex(canonical_json({"problem": BODIES[task_id], "answer": ANSWER}))


def fake_ask(problem, answer, *, flags=(), hint="", budget=None, **_kwargs):
    verdict, confidence = JUDGE[problem]
    return Verdict(verdict=verdict, confidence=confidence, reason="SECRET-REASON", my_answer="SECRET-ANSWER",
                   usage={"input_tokens": 10, "output_tokens": 5})


def make_pool():
    """Active a:1; new:6 already has a ledger PASS for its current body."""
    rows = [public_row("new:6", body_sha("new:6"), True, "ok", PRIOR_VERSION, "2026-11-26")]
    guard = parse_guard_ledger(b"".join(canonical_json(row) + b"\n" for row in rows))
    tasks = {task_id: Task(task_id=task_id, slice="math", problem=problem, answer=ANSWER)
             for task_id, problem in BODIES.items()}
    return types.SimpleNamespace(
        active_ids=("a:1",),
        guard=guard,
        candidate_meta={task_id: {"content_sha": body_sha(task_id)} for task_id in BODIES},
        catalog=types.SimpleNamespace(task_ids=tuple(BODIES), task=tasks.__getitem__),
        reserve=types.SimpleNamespace(withdrawn_ids=frozenset()),
    )


def new_reserve(base: Path, name: str) -> Path:
    root = base / name
    root.mkdir()
    prior = public_row("new:6", body_sha("new:6"), True, "ok", PRIOR_VERSION, "2026-11-26")
    (root / "guard.jsonl").write_bytes(canonical_json(prior) + b"\n")
    return root


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("GUARD_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-not-a-real-token")
    pool = make_pool()
    monkeypatch.setattr(cli, "_pool", lambda _args, _command: (pool, Params(rotation_enabled=True,
                                                                            rotation_rule="info-guard")))
    monkeypatch.setattr(judge, "ask", fake_ask)

    def use_reserve(root: Path) -> Path:
        monkeypatch.setattr(cli, "reserve_dir", lambda: root)
        return root
    return types.SimpleNamespace(pool=pool, tmp=tmp_path, use=use_reserve)


def state_args(command, **changes):
    base = dict(command=command, state_dir="/nonexistent", stack="x:y", params=None, budget_state=None,
                dry_run=False, limit=200, ids=[], epoch_results=None, gate_rollouts=None)
    return types.SimpleNamespace(**(base | changes))


def judge_args(requests, out, **changes):
    base = dict(command="judge-requests", requests=str(requests), out=str(out), parallel=4, budget_state=None,
                dry_run=False)
    return types.SimpleNamespace(**(base | changes))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def relay(env, root: Path, **export_changes) -> tuple[Path, Path, dict]:
    requests, verdicts = env.tmp / f"{root.name}-R.jsonl", env.tmp / f"{root.name}-V.jsonl"
    cli.cmd_export_requests(state_args("export-requests", out=str(requests), **export_changes))
    cli.cmd_judge_requests(judge_args(requests, verdicts))
    report = cli.cmd_ingest_verdicts(state_args("ingest-verdicts", verdicts=str(verdicts)))
    return requests, verdicts, report


def test_export_writes_hard_rejects_to_the_ledger_and_requests_for_the_rest(env):
    root = env.use(new_reserve(env.tmp, "reserve"))
    out = env.tmp / "R.jsonl"
    report = cli.cmd_export_requests(state_args("export-requests", out=str(out)))
    assert report["candidates"] == 5 and report["hard_rejected"] == 1 and report["requests"] == 4
    assert report["ledger_appended"] == 1 and report["categories"] == ["duplicate"]
    requests = rows(out)
    assert [row["id"] for row in requests] == ["new:1", "new:2", "new:3", "new:4"]
    assert all(set(row) == set(cli.REQUEST_FIELDS) for row in requests)
    assert requests[0]["content_sha"] == body_sha("new:1") and requests[0]["answer"] == ANSWER
    ledger = parse_guard_ledger((root / "guard.jsonl").read_bytes())
    assert ledger.rejects == {"new:5": body_sha("new:5")}


def test_an_export_dry_run_writes_requests_but_no_ledger_rows(env):
    root = env.use(new_reserve(env.tmp, "reserve"))
    before = (root / "guard.jsonl").read_bytes()
    report = cli.cmd_export_requests(state_args("export-requests", out=str(env.tmp / "R.jsonl"), dry_run=True))
    assert report["dry_run"] and report["hard_rejected"] == 1 and report["requests"] == 4
    assert (root / "guard.jsonl").read_bytes() == before
    assert not (root / "verdicts.jsonl").exists()


def test_judge_requests_keeps_the_judges_words_private(env):
    env.use(new_reserve(env.tmp, "reserve"))
    requests, verdicts = env.tmp / "R.jsonl", env.tmp / "V.jsonl"
    cli.cmd_export_requests(state_args("export-requests", out=str(requests)))
    report = cli.cmd_judge_requests(judge_args(requests, verdicts))
    assert report["judged"] == 4 and report["passed"] == 1
    public = rows(verdicts)
    assert all(set(row) == cli.VERDICT_FIELDS for row in public)
    by_id = {row["id"]: row for row in public}
    assert by_id["new:1"] == {"id": "new:1", "content_sha": body_sha("new:1"), "verdict": "pass",
                                   "category": "ok", "guard_version": guard_version(), "confidence": "high"}
    assert by_id["new:2"]["category"] == "bad_key" and by_id["new:3"]["category"] == "low_confidence"
    assert b"SECRET" not in verdicts.read_bytes()
    private = cli.private_path(verdicts)
    assert private.name == "V.private.jsonl"
    secret = rows(private)
    assert len(secret) == 4 and all(row["judge"]["reason"] == "SECRET-REASON" for row in secret)


def test_judge_requests_resumes_without_judging_twice(env, monkeypatch):
    env.use(new_reserve(env.tmp, "reserve"))
    requests, verdicts = env.tmp / "R.jsonl", env.tmp / "V.jsonl"
    cli.cmd_export_requests(state_args("export-requests", out=str(requests)))
    # A previous run judged new:1 and then crashed mid-append.
    first = rows(requests)[0]
    verdicts.write_bytes(json.dumps({"id": first["id"], "content_sha": first["content_sha"], "verdict": "pass",
                                     "category": "ok", "guard_version": guard_version(),
                                     "confidence": "high"}).encode() + b"\n{\"id\": \"new:")
    asked = []

    def counting_ask(problem, answer, **kwargs):
        asked.append(problem)
        return fake_ask(problem, answer, **kwargs)
    monkeypatch.setattr(judge, "ask", counting_ask)
    report = cli.cmd_judge_requests(judge_args(requests, verdicts))
    assert report["already_judged"] == 1 and report["judged"] == 3
    assert BODIES["new:1"] not in asked
    again = cli.cmd_judge_requests(judge_args(requests, verdicts))
    assert again["already_judged"] == 4 and again["judged"] == 0 and len(asked) == 3
    assert len(rows(verdicts)) == 4 and verdicts.read_bytes().endswith(b"\n")  # the torn tail was cut


def test_judge_requests_stops_cleanly_when_the_budget_is_spent(env, monkeypatch):
    env.use(new_reserve(env.tmp, "reserve"))
    requests, verdicts = env.tmp / "R.jsonl", env.tmp / "V.jsonl"
    cli.cmd_export_requests(state_args("export-requests", out=str(requests)))
    calls = []

    def capped_ask(problem, answer, **kwargs):
        calls.append(problem)
        if len(calls) > 1:
            raise BudgetExhausted("over")
        return fake_ask(problem, answer, **kwargs)
    monkeypatch.setattr(judge, "ask", capped_ask)
    report = cli.cmd_judge_requests(judge_args(requests, verdicts, parallel=1))
    assert report["judged"] == 1 and report["stopped"] == "over"
    assert len(rows(verdicts)) == 1


def test_ingest_appends_only_rows_that_still_match_the_current_body(env):
    root = env.use(new_reserve(env.tmp, "reserve"))
    verdicts = env.tmp / "V.jsonl"
    version = guard_version()

    def verdict(task_id, sha=None, **changes):
        return {"id": task_id, "content_sha": sha or body_sha(task_id), "verdict": "pass", "category": "ok",
                "guard_version": version, "confidence": "high"} | changes
    verdicts.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in (
        verdict("new:1"),
        verdict("new:2", sha="0" * 64),                                    # the body changed since export
        verdict("gone:9", sha="1" * 64),                                   # no longer in the catalog
        verdict("new:6"),                                               # already decided
        verdict("new:3", confidence="low"),                                # a pass the judge was unsure of
        verdict("new:4", verdict="reject", category="ambiguous"),
        verdict("new:4", verdict="reject", category="ambiguous"),          # repeated row
    )))
    report = cli.cmd_ingest_verdicts(state_args("ingest-verdicts", verdicts=str(verdicts)))
    assert report["eligible"] == 2 and report["ledger_appended"] == 2 and report["passed"] == 1
    assert report["skipped"] == {"invalid": 1, "unknown_or_withdrawn": 1, "stale_content_sha": 1,
                                 "already_in_ledger": 1, "duplicate_rows": 1}
    assert report["invalid_reasons"] == {"pass_inconsistent": 1}
    ledger = parse_guard_ledger((root / "guard.jsonl").read_bytes())
    assert ledger.eligible("new:1", body_sha("new:1"))
    assert ledger.rejects["new:4"] == body_sha("new:4")
    again = cli.cmd_ingest_verdicts(state_args("ingest-verdicts", verdicts=str(verdicts)))
    assert again["ledger_appended"] == 0 and again["skipped"]["already_in_ledger"] == 3
    dry = (root / "guard.jsonl").read_bytes()
    cli.cmd_ingest_verdicts(state_args("ingest-verdicts", verdicts=str(verdicts), dry_run=True))
    assert (root / "guard.jsonl").read_bytes() == dry


def test_the_relay_writes_the_same_ledger_rows_as_prejudge(env, monkeypatch):
    direct = env.use(new_reserve(env.tmp, "direct"))
    monkeypatch.setattr(cli, "decide", functools.partial(decide, ask_judge=fake_ask))
    cli.cmd_prejudge(state_args("prejudge"))
    expected = (direct / "guard.jsonl").read_bytes().splitlines()

    relayed = env.use(new_reserve(env.tmp, "relayed"))
    _requests, _verdicts, report = relay(env, relayed)
    got = (relayed / "guard.jsonl").read_bytes().splitlines()

    assert report["ledger_appended"] == 4
    # Same rows, byte for byte. Only their order differs: the relay appends the
    # hard rejects at export and the judged rows at ingest.
    assert sorted(got) == sorted(expected)
    assert len(got) == 1 + 5  # the prior pass plus every candidate
    assert parse_guard_ledger(b"\n".join(got) + b"\n").passes == parse_guard_ledger(
        b"\n".join(expected) + b"\n").passes
    # The private records carry the category apply uses for the withdrawal reason.
    private = {row["id"]: row["category"] for row in rows(relayed / "verdicts.jsonl")}
    assert private["new:2"] == "bad_key" and private["new:5"] == "duplicate"


def test_judge_requests_needs_only_the_standard_library(tmp_path):
    """No math grader, no torch, no cloud SDK: the key host has only this repo and stdlib."""
    requests = tmp_path / "R.jsonl"
    requests.write_bytes(b"")
    blocked = ("math_verify", "sympy", "latex2sympy2_extended", "torch", "transformers", "boto3", "nacl",
               "pyarrow", "numpy")
    script = f"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {blocked!r}:
            raise ImportError('blocked ' + name)
        return None
sys.meta_path.insert(0, Block())
from sn120_v2.guard.__main__ import main
sys.exit(main(['judge-requests', '--requests', {str(requests)!r}, '--out', {str(tmp_path / 'V.jsonl')!r}]))
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True,
                            env=os.environ | {"PYTHONPATH": str(root)})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["requests"] == 0
