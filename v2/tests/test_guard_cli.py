"""``python -m sn120_v2.guard``: what ``apply`` refuses to do, and when ``monitor`` says so.

No test here reaches the network or the journal. The pool is a stub with the
few attributes the commands read, and the judge is never called.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.errors import ConfigError
from sn120_v2.guard import __main__ as cli
from sn120_v2.guard.store import public_row
from sn120_v2.params import Params
from sn120_v2.rotation import parse_guard_ledger

VERSION = "guard-v1:0123456789abcdef"
OPEN = "2026-11-27T00:00:00Z open: the public files of round {rid}\n"


def body_sha(task_id: str) -> str:
    return sha256_hex(canonical_json({"problem": f"problem {task_id}", "answer": "1"}))


def stub_pool(active, outside, passes=(), rejects=()):
    rows = [public_row(t, body_sha(t), True, "ok", VERSION, "2026-11-27") for t in passes]
    rows += [public_row(t, body_sha(t), False, "bad_key", VERSION, "2026-11-27") for t in rejects]
    guard = parse_guard_ledger(b"".join(canonical_json(row) + b"\n" for row in rows))
    ids = tuple(active) + tuple(outside)
    return types.SimpleNamespace(
        active_ids=tuple(active),
        guard=guard,
        candidate_meta={task: {"content_sha": body_sha(task)} for task in ids},
        catalog=types.SimpleNamespace(task_ids=ids),
        reserve=types.SimpleNamespace(withdrawn_ids=frozenset()),
    )


@pytest.fixture
def reserve(tmp_path, monkeypatch):
    root = tmp_path / "reserve"
    root.mkdir()
    (root / "MANIFEST.jsonl").write_bytes(canonical_json({"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}) + b"\n")
    (root / "withdrawn.jsonl").write_bytes(b"")
    monkeypatch.setattr(cli, "reserve_dir", lambda: root)
    return root


def args(**changes):
    base = dict(command="apply", state_dir="/nonexistent", stack="x:y", params=None, budget_state=None,
                dry_run=False, reserve=None, run_log=None, max=None, force=False)
    return types.SimpleNamespace(**(base | changes))


def patch_pool(monkeypatch, pool, rule="info-guard"):
    monkeypatch.setattr(cli, "_pool", lambda _args, _command: (pool, Params(rotation_enabled=True,
                                                                            rotation_rule=rule)))


def test_the_last_completed_open_names_the_current_round(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000160") + "some other line\n" + OPEN.format(rid="r000161"))
    assert cli.last_opened_round(log) == "r000161"


def test_apply_refuses_when_the_run_log_shows_no_completed_open(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("round r000161 started\n")
    with pytest.raises(ConfigError, match="no completed open"):
        cli.last_opened_round(log)


def test_apply_withdraws_the_rejected_active_tasks(tmp_path, reserve, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000161"))
    pool = stub_pool(["a:1", "a:2", "a:3"], [f"new:{i}" for i in range(80)],
                     passes=[f"new:{i}" for i in range(80)], rejects=["a:1", "a:2"])
    patch_pool(monkeypatch, pool)
    report = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve)))
    assert report["withdrawn"] == 2 and report["ids"] == ["a:1", "a:2"]
    rows = [json.loads(line) for line in (reserve / "withdrawn.jsonl").read_bytes().splitlines()]
    assert [row["id"] for row in rows] == ["a:1", "a:2"]
    assert all(row["reason"].startswith("quality:") for row in rows)
    assert json.loads((reserve / "MANIFEST.jsonl").read_bytes().splitlines()[-1])["withdrawn"]["count"] == 2


def test_apply_is_idempotent_within_one_round(tmp_path, reserve, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000161"))
    pool = stub_pool(["a:1", "a:2"], [f"new:{i}" for i in range(80)],
                     passes=[f"new:{i}" for i in range(80)], rejects=["a:1"])
    patch_pool(monkeypatch, pool)
    first = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve)))
    assert first["withdrawn"] == 1
    second = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve)))
    assert second["withdrawn"] == 0 and "already applied" in second["skipped"]


def test_a_new_round_may_apply_again(tmp_path, reserve, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000161"))
    pool = stub_pool(["a:1", "a:2"], [f"new:{i}" for i in range(80)],
                     passes=[f"new:{i}" for i in range(80)], rejects=["a:1", "a:2"])
    patch_pool(monkeypatch, pool)
    cli.cmd_apply(args(run_log=str(log), reserve=str(reserve), max=1))
    log.write_text(log.read_text() + OPEN.format(rid="r000162"))
    second = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve)))
    assert second["round_id"] == "r000162" and second["withdrawn"] == 1 and second["ids"] == ["a:2"]


def test_a_dry_run_writes_nothing(tmp_path, reserve, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000161"))
    pool = stub_pool(["a:1"], [f"new:{i}" for i in range(80)],
                     passes=[f"new:{i}" for i in range(80)], rejects=["a:1"])
    patch_pool(monkeypatch, pool)
    report = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve), dry_run=True))
    assert report["dry_run"] and report["withdrawing"] == 1 and report["withdrawn"] == 0
    assert (reserve / "withdrawn.jsonl").read_bytes() == b""
    assert not cli.marker_path(reserve, "r000161").exists()


def test_the_round_cap_is_three_hundred(tmp_path, reserve, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000161"))
    bad = [f"a:{i:04d}" for i in range(400)]
    good = [f"new:{i}" for i in range(500)]
    pool = stub_pool(bad, good, passes=good, rejects=bad)
    patch_pool(monkeypatch, pool)
    report = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve), dry_run=True))
    assert report["cap"] == cli.MAX_WITHDRAW == 300 and report["withdrawing"] == 300
    assert report["limited_by"] == "round cap"


def test_a_thin_eligible_supply_holds_the_batch_back(tmp_path, reserve, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000161"))
    bad = [f"a:{i}" for i in range(100)]
    good = [f"new:{i}" for i in range(60)]  # 60 eligible, 50 must remain
    pool = stub_pool(bad, good, passes=good, rejects=bad)
    patch_pool(monkeypatch, pool)
    report = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve), dry_run=True))
    assert report["cap"] == 10 and report["limited_by"] == "eligible replacement supply"


def test_an_empty_ledger_withdraws_nothing_at_all(tmp_path, reserve, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000161"))
    pool = stub_pool(["a:1"], ["new:1"])
    patch_pool(monkeypatch, pool)
    report = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve)))
    assert report["withdrawn"] == 0 and report["rejected_active"] == 0


def test_under_the_v2_rule_the_supply_rule_does_not_apply(tmp_path, reserve, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(OPEN.format(rid="r000161"))
    pool = stub_pool(["a:1", "a:2"], ["new:1"], rejects=["a:1", "a:2"])
    patch_pool(monkeypatch, pool, rule="info")
    report = cli.cmd_apply(args(run_log=str(log), reserve=str(reserve), dry_run=True))
    assert report["cap"] == cli.MAX_WITHDRAW and report["withdrawing"] == 2


def test_monitor_reports_the_ledger_and_the_eligible_supply(reserve, monkeypatch):
    pool = stub_pool(["a:1", "a:2"], ["new:1", "new:2", "new:3"], passes=["new:1"], rejects=["a:1", "new:2"])
    patch_pool(monkeypatch, pool)
    report = cli.cmd_monitor(args(command="monitor"))
    assert report["rotation_rule"] == "info-guard"
    assert report["guard_rows"] == 3 and report["guard_passes"] == 1 and report["guard_rejects"] == 2
    assert report["active_size"] == 2 and report["active_rejected"] == 1
    assert report["outside_candidates"] == 3 and report["outside_eligible"] == 1
    assert report["undecided_outside"] == 1  # new:3
    assert report["withdraw_headroom"] == 0  # one eligible pass is far below the buffer
    assert report["daily_limit_usd"] == 55.0


def prejudge_args(**changes):
    return args(command="prejudge", limit=10, ids=[], epoch_results=None, gate_rollouts=None, **changes)


def catalog_pool(pool, bodies):
    from sn120_v2.envs.math_pool import Task
    tasks = {task_id: Task(task_id=task_id, slice="math", problem=problem, answer="1")
             for task_id, problem in bodies.items()}
    pool.catalog = types.SimpleNamespace(task_ids=tuple(bodies), task=tasks.__getitem__)
    return pool


def test_prejudge_only_judges_what_the_ledger_has_not_decided(reserve, monkeypatch):
    pool = catalog_pool(stub_pool(["a:1"], ["new:1", "new:2"], passes=["new:1"]),
                        {"a:1": "problem a:1", "new:1": "problem new:1", "new:2": "problem new:2"})
    patch_pool(monkeypatch, pool)
    seen = []

    def fake_decide(task, _universe, **_kwargs):
        from sn120_v2.guard.store import Decision
        seen.append(task.task_id)
        return Decision(task_id=task.task_id, content_sha=body_sha(task.task_id), passed=True,
                        category="ok", guard_version=VERSION, added_at="2026-11-27")

    monkeypatch.setattr(cli, "decide", fake_decide)
    report = cli.cmd_prejudge(prejudge_args())
    assert seen == ["new:2"]  # new:1 is already decided for this body; a:1 is active
    assert report["judged"] == 1 and report["passed"] == 1 and report["ledger_appended"] == 1
    ledger = parse_guard_ledger((reserve / "guard.jsonl").read_bytes())
    assert ledger.eligible("new:2", body_sha("new:2"))


def test_a_prejudge_dry_run_appends_nothing(reserve, monkeypatch):
    pool = catalog_pool(stub_pool(["a:1"], ["new:1"]), {"a:1": "problem a:1", "new:1": "problem new:1"})
    patch_pool(monkeypatch, pool)

    def fake_decide(task, _universe, **_kwargs):
        from sn120_v2.guard.store import Decision
        return Decision(task_id=task.task_id, content_sha=body_sha(task.task_id), passed=False,
                        category="bad_key", guard_version=VERSION, added_at="2026-11-27")

    monkeypatch.setattr(cli, "decide", fake_decide)
    report = cli.cmd_prejudge(prejudge_args(dry_run=True))
    assert report["dry_run"] and report["judged"] == 1 and report["categories"] == ["bad_key"]
    assert not (reserve / "guard.jsonl").exists()


def test_the_universe_sees_withdrawn_bodies_so_a_repeat_is_still_a_duplicate(reserve, monkeypatch):
    shards = reserve / "shards"
    shards.mkdir()
    (shards / "0001-x.jsonl").write_bytes(
        canonical_json({"id": "old:1", "problem": "What is $1+1$?", "answer": "2"}) + b"\n")
    pool = catalog_pool(stub_pool(["a:1"], ["new:1"]), {"a:1": "problem a:1", "new:1": "what is 1 + 1 ?"})
    pool.reserve = types.SimpleNamespace(withdrawn_ids=frozenset({"old:1"}))
    _bodies, universe = cli._universe(pool)
    assert universe.duplicate_of("new:1", "what is 1 + 1 ?") == "old:1"


def test_an_active_task_owns_a_repeated_problem_before_a_candidate_does(reserve, monkeypatch):
    pool = catalog_pool(stub_pool(["a:1"], ["new:1"]), {"new:1": "What is $1+1$?", "a:1": "what is 1 + 1 ?"})
    pool.catalog.task_ids = ("new:1", "a:1")  # catalog order puts the candidate first
    _bodies, universe = cli._universe(pool)
    assert universe.duplicate_of("new:1", "What is $1+1$?") == "a:1"


def test_the_run_log_path_needs_a_source(monkeypatch):
    monkeypatch.delenv("GUARD_RUN_LOG", raising=False)
    monkeypatch.delenv("SN120_WORK_DIR", raising=False)
    with pytest.raises(ConfigError):
        cli.run_log_path(args())
    monkeypatch.setenv("SN120_WORK_DIR", "/srv/sn120")
    assert cli.run_log_path(args()) == Path("/srv/sn120/logs/run.log")
