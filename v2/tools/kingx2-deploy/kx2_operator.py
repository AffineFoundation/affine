#!/usr/bin/env python3
"""King-x2 deploy operator. No generator/ingestion behavior.

Only an operator may use this against a live checkout. Inspection never prints
secrets. set-params, start-loop, the bounded evidence precheck and the
work-directory lookup use the existing operator loop environment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

DRY = os.environ.get("KINGX2_DRY") == "1"
V2 = Path(os.environ.get("KINGX2_V2", "/root/sn120/v2"))
HERE = Path(os.environ.get("KINGX2_HERE", "/root/sn120-run/kingx2-deploy"))
STATE = Path(os.environ.get("KINGX2_STATE", "/root/sn120-run/state"))
PUBLIC = os.environ.get("KINGX2_PUBLIC", "https://affine.io").rstrip("/")
PY = os.environ.get("KINGX2_PY", "/root/sn120/v2/.venv/bin/python")
STACK = os.environ.get("KINGX2_STACK", "sn120_v2.assembly:stack")
LOOP = os.environ.get("KINGX2_LOOP_SCRIPT", "systemd/sn120-v2-run-loop.sh")
ENV_FILE = Path(os.environ.get("KINGX2_ENV_FILE", "/root/sn120-run/secrets/loop-env-903093.environ"))
WORK = Path(os.environ["KINGX2_WORK_DIR"]) if os.environ.get("KINGX2_WORK_DIR") else None
JOURNAL = STATE / "journal.jsonl"
TARGET = HERE / "rollback-target.json"
EXPECT = json.loads((HERE / "expect.json").read_text())
_package = V2 / "sn120_v2"
_spec = importlib.util.spec_from_file_location("sn120_v2", _package / "__init__.py", submodule_search_locations=[str(_package)])
assert _spec is not None and _spec.loader is not None
sys.modules["sn120_v2"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sys.modules["sn120_v2"])
from sn120_v2 import layout  # noqa: E402
from sn120_v2.canonical import canonical_json, sha256_hex  # noqa: E402
from sn120_v2.cli import summary  # noqa: E402
from sn120_v2.irt import verify_snapshot  # noqa: E402
from sn120_v2.journal import read_journal  # noqa: E402
from sn120_v2.params import load_params  # noqa: E402
from sn120_v2.rotation import Rotation  # noqa: E402
from sn120_v2.state import load  # noqa: E402

ENV = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "GIT_TERMINAL_PROMPT": "0"}
# A post-checkout child is a fresh interpreter, not a fresh environment.
# Preserve explicit dry paths (and other intentional operator overrides).
for _name, _value in os.environ.items():
    if _name.startswith("KINGX2_") or _name == "PYTHONPATH":
        ENV[_name] = _value
UA = {"User-Agent": "sn120-kingx2-deploy/1", "Accept-Encoding": "identity", "Cache-Control": "no-cache"}


def log(message: str) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), message, flush=True)


def work_dir() -> Path:
    """The loop's authoritative local path, configured once by the operator."""
    return WORK if WORK is not None else Path(load_env()["SN120_WORK_DIR"])


def sh(cmd, *, cwd=V2, check=True, timeout=900, env=None):
    result = subprocess.run([str(x) for x in cmd], cwd=cwd, env=env or ENV,
                            capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        detail = (result.stderr.strip() or result.stdout.strip()).splitlines()[-4:]
        raise RuntimeError(f"command {cmd[:4]} exited {result.returncode}: {detail}")
    return result


def head() -> str:
    return sh(["git", "rev-parse", "HEAD"]).stdout.strip()


def digest() -> str:
    # This pure local pin must not import or use an installed GPU environment.
    from sn120_v2.canonical import file_sha256, manifest_digest
    return manifest_digest({path.name: file_sha256(path) for path in sorted((V2 / "sn120_v2/gpu").glob("*.py"))})


def status() -> dict:
    state = load(JOURNAL)
    return summary(state, state.params)


def params_hash(path: Path) -> str:
    return load_params(path).params_hash


def pidfile() -> Path:
    return Path(os.environ.get("KINGX2_LOOP_PIDFILE", str(work_dir() / "logs/run-loop.pid")))


def alive() -> list[str]:
    """The run-loop pidfile names one loop; its run child is not another loop."""
    path = pidfile()
    if not path.is_file():
        return []
    value = path.read_text().strip()
    if DRY:
        return ["dry loop marker"] if value else [f"empty dry loop marker {path}"]
    if not value.isdigit():
        return [f"invalid run-loop pidfile {path}"]
    try:
        raw = Path(f"/proc/{value}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return []
    if Path(LOOP).name not in raw:
        return [f"pidfile {path} names an unexpected process: {raw[:180]}"]
    return [f"{value} {raw}"]


def loop_count() -> None:
    print(len(alive()))


def checkout(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("refusing a non-SHA checkout")
    reserve = V2 / "sn120_v2/envs/reserve"
    saved = HERE / f"reserve-before-{commit[:12]}"
    if saved.exists():
        raise RuntimeError(f"reserve preservation directory already exists: {saved}")
    shutil.copytree(reserve, saved)
    from sn120_v2.envs.reserve import load_reserve
    load_reserve(saved, allow_retired=True)
    try:
        sh(["git", "checkout", "-q", "--detach", commit], cwd=V2.parent)
        # Preserve the complete external mutable cache, including untracked
        # append-only shards, withdrawn rows and retired task mirrors.
        shutil.copytree(saved, reserve, dirs_exist_ok=True)
        load_reserve(reserve, allow_retired=True)
    except BaseException:
        shutil.copytree(saved, reserve, dirs_exist_ok=True)
        raise
    if head() != commit:
        raise RuntimeError("checkout did not reach the requested commit")


def load_env() -> dict[str, str]:
    """Read NUL-delimited loop settings only for set-params or start-loop."""
    env = {}
    for item in ENV_FILE.read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode()
    for key in ("APPLY", "C", "CAPS", "MB", "OLDPWD"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def set_params(filename: str) -> None:
    if not DRY and alive():
        raise RuntimeError(f"set-params refused: loop alive {alive()}")
    path = HERE / filename
    result = None
    if DRY:
        command = os.environ.get("KINGX2_DRY_SET_PARAMS", "")
        if not command:
            raise RuntimeError("dry set-params needs KINGX2_DRY_SET_PARAMS")
        result = sh(["bash", "-c", command + " " + shlex.quote(str(path))], timeout=3600)
    else:
        env = load_env()
        if not env.get("SN120_WORK_DIR"):
            raise RuntimeError("loop environment has no SN120_WORK_DIR")
        result = sh([PY, "-I", "-m", "sn120_v2", "set-params", "--state-dir", STATE,
                     "--stack", STACK, "--params", path], timeout=3600, env=env)
    (HERE / f"set-params-{Path(filename).stem}.json").write_text(result.stdout)
    if result.stdout.strip():
        log(f"{filename}: {result.stdout.strip()[:1400]}")


def cli(command: str) -> subprocess.CompletedProcess:
    """Run a controller command under the currently checked-out code."""
    if DRY:
        override = os.environ.get(f"KINGX2_DRY_{command.upper().replace('-', '_')}", "")
        if not override:
            raise RuntimeError(f"dry {command} needs its isolated CLI override")
        return sh(["bash", "-c", override], timeout=3600)
    env = load_env()
    if not env.get("SN120_WORK_DIR"):
        raise RuntimeError("loop environment has no SN120_WORK_DIR")
    return sh([PY, "-I", "-m", "sn120_v2", command, "--state-dir", STATE,
               "--stack", STACK], timeout=3600, env=env)


def sealed_result_leases() -> list[str]:
    """Check every lease; end-round itself checks only leases still marked open."""
    if DRY:
        from sn120_v2.store import LocalStore
        root = os.environ.get("KINGX2_PUBLIC_ROOT")
        if not root:
            raise RuntimeError("dry sealed-result check needs its isolated object-store root")
        store = LocalStore(root)
    else:
        # This child uses the operator's environment without printing secrets.
        # An explicit timeout bounds the complete evidence precheck on rollback.
        code = (
            "import json;"
            "from sn120_v2 import layout;"
            "from sn120_v2.r2 import R2Store;"
            "from sn120_v2.state import load;"
            "record=load(" + repr(str(JOURNAL)) + ").round;store=R2Store.from_env();"
            "print(json.dumps([] if record is None else "
            "[lease.lease_id for lease in record.leases if store.exists(layout.sealed_result(record.round_id,lease.lease_id))]))"
        )
        result = sh([PY, "-I", "-c", code], cwd=HERE, env=load_env(), timeout=180)
        found = json.loads(result.stdout)
        if not isinstance(found, list) or any(not isinstance(value, str) for value in found):
            raise RuntimeError("sealed-result precheck returned an invalid lease list")
        return found
    state = load(JOURNAL)
    if state.round is None:
        return []
    return [lease.lease_id for lease in state.round.leases
            if store.exists(layout.sealed_result(state.round.round_id, lease.lease_id))]


def close_undecided_for_rollback() -> None:
    """Use the supported operator path; never discard a result to force recovery."""
    before = load(JOURNAL)
    record = before.round
    if record is None:
        return
    if record.decision is not None:
        raise RuntimeError(f"rollback blocked: {record.round_id} is decided and must be finished by run")
    stored = sealed_result_leases()
    if stored:
        raise RuntimeError(f"rollback blocked: {record.round_id} has sealed result(s) on lease(s) {stored}; judge normally, never discard them")
    # The CLI repeats its mirror check before taking the journal fence. A race
    # or a differently mirrored result still produces its exit-3 refusal.
    result = cli("end-round")
    if result.returncode == 3:
        raise RuntimeError(f"rollback blocked: {record.round_id} has a sealed result; judge it normally, never discard it")
    if result.returncode:
        raise RuntimeError(f"rollback blocked: end-round for {record.round_id} exited {result.returncode}")
    after = load(JOURNAL)
    if after.round is not None:
        raise RuntimeError(f"rollback blocked: {record.round_id} is still open; stored results must be judged first")
    closed = after.rounds[-1]
    if closed.round_id != record.round_id or closed.outcome != "no_decision":
        raise RuntimeError(f"rollback close did not end {record.round_id} with no_decision")
    log(f"rollback: ended undecided {closed.round_id} with no_decision; no stored result was discarded")


def ensure_params(filename: str, expected: str) -> None:
    if params_hash(HERE / filename) != expected:
        raise RuntimeError(f"{filename} does not have the expected params hash")
    if status()["params_hash"] != expected:
        set_params(filename)
    if status()["params_hash"] != expected:
        raise RuntimeError(f"set-params did not pin {expected}")


def verify_before() -> dict:
    report = status()
    if report.get("phase") != "IDLE":
        raise RuntimeError(f"journal phase is {report.get('phase')}, not IDLE")
    if head() != EXPECT["old_commit"]:
        raise RuntimeError(f"HEAD {head()} != old commit {EXPECT['old_commit']}")
    if digest() != EXPECT["old_digest"]:
        raise RuntimeError("old code digest mismatch")
    if report["params_hash"] != EXPECT["old_params_hash"]:
        raise RuntimeError("old pinned params hash mismatch")
    if load(JOURNAL).params.recipe.code_digest != EXPECT["old_digest"]:
        raise RuntimeError("old pinned code digest mismatch")
    # Old code cannot parse null_samples_per_task. Verify the new file only
    # after checkout; ensure_params then both validates and pins it.
    actual = params_hash(HERE / "old-params.json")
    if actual != EXPECT["old_params_hash"]:
        raise RuntimeError(f"old-params.json hashes to {actual}, expected {EXPECT['old_params_hash']}")
    return report


def preflight() -> None:
    if alive():
        raise RuntimeError(f"loop/run alive: {alive()}")
    dirty = sh(["git", "status", "--porcelain", "--untracked-files=all"]).stdout
    allowed = ("sn120_v2/envs/reserve/retired.json", "sn120_v2/envs/reserve/retired_tasks.json")
    other = [line for line in dirty.splitlines() if not line.endswith(allowed)]
    if other:
        raise RuntimeError(f"checkout not clean: {other[:10]}")
    report = verify_before()
    log(f"preflight: HEAD {head()} digest {digest()} params {report['params_hash']} phase {report['phase']}")


def apply() -> None:
    report = verify_before()
    if alive():
        raise RuntimeError("apply refused: loop alive")
    new = EXPECT["new_commit"]
    if new == "PENDING_IMPLEMENTATION_COMMIT":
        raise RuntimeError("expect.json new_commit is still PENDING_IMPLEMENTATION_COMMIT")
    if sh(["git", "cat-file", "-t", new], check=False).stdout.strip() != "commit":
        # Offline kit: the implementation commit must already be in this repo.
        raise RuntimeError("new commit is not present locally; no fetch is attempted")
    TARGET.write_text(json.dumps({"old_commit": EXPECT["old_commit"], "old_digest": EXPECT["old_digest"],
                                  "old_params_hash": EXPECT["old_params_hash"],
                                  "journal_records_before": report["journal_records"],
                                  "recorded_at": time.strftime("%FT%TZ", time.gmtime())}, indent=2) + "\n")
    checkout(new)
    try:
        # Params and State were imported from the old checkout in this process.
        # A fresh interpreter must parse the new schema and replay new params.
        sh([sys.executable, "-I", Path(__file__).resolve(), "apply-checked-out"], timeout=3600)
    except BaseException as error:
        # This child also starts fresh under new code, so params-first rollback
        # can read the new journal before old code is checked out.
        recovered = sh([sys.executable, "-I", Path(__file__).resolve(), "rollback"], check=False, timeout=3600)
        if recovered.returncode:
            raise RuntimeError(f"apply failed ({error}); rollback also failed: {recovered.stderr.strip()[-1000:]}") from error
        raise


def apply_checked_out() -> None:
    """Run only in a fresh interpreter after the new commit is checked out."""
    new = EXPECT["new_commit"]
    if head() != new:
        raise RuntimeError("post-checkout HEAD is not the new commit")
    try:
        if digest() != EXPECT["new_digest"]:
            raise RuntimeError("new code digest mismatch")
        ensure_params("new-params.json", EXPECT["new_params_hash"])
        state = load(JOURNAL)
        if state.params.recipe.code_digest != EXPECT["new_digest"] or state.params.irt_draw_rule != EXPECT["irt_draw_rule"]:
            raise RuntimeError("new pinned code digest or sqrt_fisher rule mismatch")
        if state.params.null_samples != EXPECT["null_samples"] or state.params.n_tasks != EXPECT["n_tasks"]:
            raise RuntimeError("new null sample count or task count mismatch")
        if state.params.rotation_active_size != EXPECT["active_size"]:
            raise RuntimeError("new active size mismatch")
        log(f"APPLY DONE: HEAD {head()} digest {digest()} params {state.params.params_hash}")
    except BaseException:
        # New code can still parse both parameter sets. Never let old code try
        # to replay null_samples_per_task or sqrt_fisher first.
        rollback()
        raise


def start_loop() -> None:
    if alive():
        raise RuntimeError(f"start-loop refused: already alive {alive()}")
    if DRY:
        path = pidfile()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()) + "\n")
        log("dry: exactly one simulated loop started; no real service")
        return
    env = load_env()
    stop = work_dir() / "logs/run-loop.stop-after-round"
    if stop.exists():
        raise RuntimeError("stop-after-round file present")
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    with open("/dev/null", "rb") as null:
        subprocess.Popen(["setsid", "nohup", "bash", LOOP, PY, "--state-dir", STATE, "--stack", STACK],
                         cwd=V2, env=env, stdin=null, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(float(os.environ.get("KINGX2_START_WAIT", "5")))
    path = pidfile()
    if not path.is_file() or not path.read_text().strip().isdigit() or not Path(f"/proc/{path.read_text().strip()}").exists():
        raise RuntimeError("run loop did not leave a live pidfile")
    if len(alive()) != 1:
        raise RuntimeError(f"start-loop did not yield exactly one loop: {alive()}")
    log(f"start-loop: one live loop, pidfile {path}")


def round_id(row) -> str | None:
    data = row.data or {}
    return data.get("round_id") or (data.get("round") or {}).get("round_id")


def opened_row(rows, rid: str):
    found = [row for row in rows if row.kind == "round_opened" and round_id(row) == rid]
    return found[-1] if found else None


def open_check(rid: str) -> dict:
    rows = read_journal(JOURNAL)
    opened = opened_row(rows, rid)
    out = {"round": rid, "ok": False, "why": []}
    if opened is None:
        out["why"].append("round_opened not found")
        return out
    record = opened.data["round"]
    rotation = opened.data.get("rotation") or record.get("rotation")
    out.update(params_hash=record.get("params_hash"), active_size=rotation and rotation.get("size"),
               snapshot_id=record.get("snapshot_id"), n_tasks=record.get("n_tasks"))
    if record.get("params_hash") != EXPECT["new_params_hash"]:
        out["why"].append("round params hash mismatch")
    if record.get("n_tasks") != EXPECT["n_tasks"]:
        out["why"].append("round task count mismatch")
    if not rotation or rotation.get("size") != EXPECT["active_size"]:
        out["why"].append("journal active size is not 10000")
    try:
        state = pub_get(layout.pool_rotation_state(rotation["state_sha"]), object_json=True)
        if sha256_hex(canonical_json(state)) != rotation["state_sha"]:
            out["why"].append("published rotation state content id mismatch")
        actual = Rotation.from_dict(state)
        expected = dict(rotation)
        if expected.pop("state_sha") != actual.digest() or expected.pop("size") != len(actual.active_ids) or expected != actual.journal_block():
            out["why"].append("published rotation state differs from the journal block")
        active_raw = pub_get(layout.active_pool(actual.active_sha))
        active_ids = json.loads(active_raw)
        if (sha256_hex(active_raw) != actual.active_sha or active_ids != list(actual.active_ids)
                or active_ids != sorted(active_ids) or len(set(active_ids)) != len(active_ids)
                or any(not isinstance(task, str) or not task for task in active_ids)):
            out["why"].append("content-addressed active bytes/ids differ from sorted unique rotation ids")
        snapshot = pub_get(layout.irt_snapshot(record["snapshot_id"]), object_json=True)
        verify_snapshot(snapshot)
        if snapshot["snapshot_id"] != record["snapshot_id"] or set(snapshot["tasks"]) != set(actual.active_ids):
            out["why"].append("IRT snapshot identity or active task set mismatch")
        config = snapshot.get("config") or {}
        out["snapshot_config"] = config
        if config.get("draw_rule") != EXPECT["irt_draw_rule"]:
            out["why"].append("IRT snapshot draw_rule is not sqrt_fisher")
        if float(config.get("uniform_frac")) != float(load_params(HERE / "new-params.json").irt_explore_frac):
            out["why"].append("IRT snapshot uniform_frac mismatch")
    except Exception as error:
        out["why"].append(f"open artifact verification failed: {type(error).__name__}: {error}")
    out["ok"] = not out["why"]
    return out


def expected_open(rid: str) -> dict:
    """Bind the watched open to the journal, not to an unrelated run-log line."""
    if not re.fullmatch(r"r[0-9]{6}", rid) or rid == "r000000":
        raise ValueError(f"invalid expected round id {rid!r}")
    state = load(JOURNAL)
    previous = f"r{int(rid[1:]) - 1:06d}"
    out = {"expected": rid, "round": state.round.round_id if state.round else None,
           "last_closed": state.rounds[-1].round_id if state.rounds else None, "ok": False, "why": []}
    if state.round is not None:
        if state.round.round_id != rid:
            out["why"].append(f"open {state.round.round_id} is not expected {rid}")
        elif state.round.params_hash != EXPECT["new_params_hash"]:
            out["why"].append("open params hash mismatch")
        elif state.rounds[-1].round_id != previous:
            out["why"].append(f"previous round {previous} is not the last close")
        else:
            out["ok"] = True
    elif any(record.round_id >= rid for record in state.rounds):
        out["why"].append(f"journal passed expected round {rid}")
    elif state.rounds[-1].round_id == previous:
        # Starting the loop and publishing its next open are separate events.
        # The caller's bounded timeout handles an open that never arrives.
        out["why"].append("prior round closed; waiting for expected next open")
    return out


def journal_close_summary(rid: str, *, next_open_ok=False) -> dict:
    rows = read_journal(JOURNAL)
    opened = opened_row(rows, rid)
    settled = [row for row in rows if row.kind == "round_settled" and round_id(row) == rid]
    closed = [row for row in rows if row.kind == "round_closed" and round_id(row) == rid]
    out = {"round": rid, "journal_records": len(rows),
           "opened_seq": opened.seq if opened else None,
           "settled_seq": settled[-1].seq if settled else None,
           "closed_seq": closed[-1].seq if closed else None}
    why = [f"no {kind}" for kind, found in (("round_opened", opened), ("round_settled", settled), ("round_closed", closed)) if not found]
    if opened:
        failed = [row for row in rows if row.kind == "lease_failed" and row.seq > opened.seq and round_id(row) == rid]
        out["lease_failed"] = [{"seq": row.seq, "lease_id": row.data.get("lease_id"), "reason": row.data.get("reason", "")[:160]} for row in failed]
        if failed:
            why.append(f"lease_failed since open: {len(failed)}")
        record = opened.data["round"]
        out["boxes"] = [lease["box_id"] for lease in record.get("leases", [])]
        out["round_params_hash"] = record.get("params_hash")
        out["round_n_tasks"] = record.get("n_tasks")
    if closed:
        out.update(outcome=closed[-1].data.get("outcome"), close_at=closed[-1].at,
                   snapshot_id=closed[-1].data.get("snapshot_id"), rotation_state_sha=closed[-1].data.get("rotation_state_sha"))
        if closed[-1].data.get("outcome") not in ("null", "applied"):
            why.append(f"outcome {closed[-1].data.get('outcome')!r}")
        after = [round_id(row) for row in rows if row.kind == "round_opened" and row.seq > closed[-1].seq]
        out["opened_after_close"] = after
        if after and not next_open_ok:
            why.append(f"round(s) opened after close: {after}")
    if settled:
        decision = settled[-1].data.get("decision") or {}
        out.update(n_tasks=decision.get("n_tasks"), null_pass_rate=decision.get("null_pass_rate"))
        if not decision.get("n_tasks"):
            why.append("decision n_tasks 0")
    out.update(clean=not why, why=why)
    return out


def public_round(rid: str) -> dict:
    raw = pub_get(layout.round_file(rid))
    report = json.loads(raw)
    return {"raw": raw, "report": report}


def health(rid: str) -> dict:
    """A failed health check is a report, not an exception, for shell rollback."""
    out = {"round": rid, "ok": False, "why": []}
    try:
        summary = journal_close_summary(rid, next_open_ok=True)
        out["journal"] = summary
        if not summary.get("clean"):
            out["why"].extend(summary.get("why") or [])
        if summary.get("outcome") not in ("null", "applied"):
            out["why"].append("first new-code round did not close null/applied")
        if summary.get("n_tasks") != EXPECT["n_tasks"] or summary.get("round_n_tasks") != EXPECT["n_tasks"]:
            out["why"].append("first new-code round did not have n_tasks=2000")
        if summary.get("round_params_hash") != EXPECT["new_params_hash"]:
            out["why"].append("first new-code round did not use new params")
        published = public_round(rid)
        report, raw = published["report"], published["raw"]
        out["round_sha256"] = hashlib.sha256(raw).hexdigest()
        if report.get("round_id") != rid or report.get("params_hash") != EXPECT["new_params_hash"]:
            out["why"].append("public round id/params mismatch")
        if report.get("outcome") not in ("null", "applied") or report.get("n_tasks") != EXPECT["n_tasks"]:
            out["why"].append("public round outcome/task count mismatch")
        boxes = summary.get("boxes") or []
        null = report.get("null_outcomes")
        if not isinstance(null, dict) or set(null) != set(boxes) or not boxes:
            out["why"].append("public null outcomes do not cover every leased box exactly once")
            null = null if isinstance(null, dict) else {}
        lengths = {}
        for box in boxes:
            strings = null.get(box)
            valid = isinstance(strings, list) and len(strings) == EXPECT["null_samples"]
            valid = valid and all(isinstance(text, str) and len(text) == EXPECT["n_tasks"] for text in strings)
            lengths[box] = [len(text) for text in strings] if isinstance(strings, list) and all(isinstance(text, str) for text in strings) else None
            if not valid:
                out["why"].append(f"box {box}: need exactly 2 null strings x 2000 chars")
        out["null_string_lengths"] = lengths
        wall = report.get("wall_s") or {}
        eval_wall = wall.get("evaluation")
        if type(eval_wall) not in (int, float) or not math.isfinite(eval_wall) or eval_wall <= 0:
            out["why"].append("public round evaluation wall time is missing or not positive")
        out["eval_wall_s"] = eval_wall
        out["wall_s"] = wall
        phases = []
        phase_path = Path(os.environ["KINGX2_PHASES"]) if os.environ.get("KINGX2_PHASES") else work_dir() / "logs/phases.jsonl"
        if phase_path.is_file():
            for line in phase_path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                    if (row.get("round_id") == rid and row.get("phase") in ("eval", "solo_eval", "eval_stream", "eval_cut")
                            and type(row.get("wall_s")) in (int, float) and math.isfinite(row["wall_s"]) and row["wall_s"] >= 0):
                        phases.append({"at": row.get("at"), "phase": row["phase"], "look": row.get("look"), "wall_s": row["wall_s"]})
                except (ValueError, TypeError):
                    continue
        if not phases:
            out["why"].append("no local eval/solo_eval/eval_stream/eval_cut phase timing for this round")
        out["eval_phase_wall_s"] = phases
        limits = load_params(HERE / "new-params.json")
        scoring_deadline = limits.round_time_limit_s - limits.promote_reserve_s
        out["timing_limits_s"] = {"round": limits.round_time_limit_s, "scoring_before_reserve": scoring_deadline,
                                  "per_request_eval": limits.eval_time_budget_s, "lease": limits.lease_timeout_s}
        if isinstance(eval_wall, (int, float)) and not isinstance(eval_wall, bool) and eval_wall > scoring_deadline:
            out["why"].append(f"evaluation wall time exceeds scoring-before-reserve bound {scoring_deadline}")
        out["ok"] = not out["why"]
    except Exception as error:
        out["why"].append(f"{type(error).__name__}: {error}")
    return out


def pub_get(key: str, *, object_json=False, deadline: float | None = None):
    root = os.environ.get("KINGX2_PUBLIC_ROOT")
    if root:
        raw = (Path(root) / key).read_bytes()
    else:
        timeout = 90.0 if deadline is None else max(0.1, min(90.0, deadline - time.monotonic()))
        url = PUBLIC + "/" + key + "?kingx2=" + str(time.time_ns())
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as reply:
            raw = reply.read()
    return json.loads(raw) if object_json else raw


def public_check(rid: str) -> dict:
    out = {"round": rid, "ok": False, "why": []}
    try:
        rows = read_journal(JOURNAL)
        opened = opened_row(rows, rid)
        if opened is None:
            raise ValueError("round_opened not found")
        record = opened.data["round"]
        rotation = json.loads(pub_get(layout.pool_rotation(record["round_id"])))
        actual = Rotation.from_dict(rotation)
        journal_rotation = dict(opened.data.get("rotation") or record.get("rotation") or {})
        state_sha, size = journal_rotation.pop("state_sha", None), journal_rotation.pop("size", None)
        if (actual.journal_block() != journal_rotation or actual.digest() != state_sha
                or len(actual.active_ids) != size or size != EXPECT["active_size"] or actual.round_id != rid):
            out["why"].append("public rotation differs from journal identity/block")
        out["active_size"] = len(actual.active_ids)
        out["rotation_state_sha"] = state_sha
        active_raw = pub_get(layout.active_pool(actual.active_sha))
        if sha256_hex(active_raw) != actual.active_sha:
            out["why"].append("active pool raw bytes do not match active_sha")
        active = json.loads(active_raw)
        if (active != list(actual.active_ids) or active != sorted(active)
                or len(set(active)) != len(active) or any(not isinstance(task, str) or not task for task in active)):
            out["why"].append("content-addressed active pool is not the rotation's sorted unique ids")
        state = pub_get(layout.pool_rotation_state(state_sha), object_json=True)
        if sha256_hex(canonical_json(state)) != state_sha or Rotation.from_dict(state).journal_block() != journal_rotation:
            out["why"].append("published rotation state differs from its content id/journal")
        snapshot = json.loads(pub_get(layout.irt_snapshot(record["snapshot_id"])))
        verify_snapshot(snapshot)
        if snapshot["snapshot_id"] != record["snapshot_id"] or set(snapshot["tasks"]) != set(actual.active_ids):
            out["why"].append("public IRT snapshot identity/active task set mismatch")
        out["snapshot_config"] = snapshot.get("config") or {}
        if out["snapshot_config"].get("draw_rule") != EXPECT["irt_draw_rule"]:
            out["why"].append("public IRT snapshot is not sqrt_fisher")
        if float(out["snapshot_config"].get("uniform_frac")) != float(load_params(HERE / "new-params.json").irt_explore_frac):
            out["why"].append("public IRT snapshot exploration share mismatch")
        report = json.loads(pub_get(layout.round_file(rid)))
        out["public_params_hash"] = report.get("params_hash")
        if report.get("params_hash") != EXPECT["new_params_hash"]:
            out["why"].append("public round params hash mismatch")
        out["ok"] = not out["why"]
    except Exception as error:
        out["why"].append(f"{type(error).__name__}: {error}")
    return out


def rollback() -> None:
    if alive():
        raise RuntimeError(f"rollback refused: loop alive {alive()}")
    target = json.loads(TARGET.read_text()) if TARGET.exists() else EXPECT
    old_commit = target.get("old_commit", EXPECT["old_commit"])
    # Params first: old code strictly rejects the newly introduced fields.
    # If a failure followed a new-code open, only an undecided round without a
    # stored result can be ended by the supported no-decision operator command.
    close_undecided_for_rollback()
    ensure_params("old-params.json", EXPECT["old_params_hash"])
    checkout(old_commit)
    if head() != old_commit or digest() != EXPECT["old_digest"] or status()["params_hash"] != EXPECT["old_params_hash"]:
        raise RuntimeError("rollback verification failed")
    if load(JOURNAL).params.recipe.code_digest != EXPECT["old_digest"]:
        raise RuntimeError("rollback pinned digest mismatch")
    log(f"ROLLBACK DONE -> {old_commit}; old params restored")


def facts() -> None:
    log(f"facts: HEAD {head()} code_digest {digest()} phase {status().get('phase')} params {status().get('params_hash')}")


def emit(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    if len(sys.argv) < 2:
        raise RuntimeError("missing operator command")
    command = sys.argv[1]
    if command not in ("preflight", "apply", "apply-checked-out", "start-loop", "loop-count", "work-dir", "rollback", "facts") and len(sys.argv) < 3:
        raise RuntimeError(f"{command} needs a round id")
    if command == "preflight": preflight()
    elif command == "apply": apply()
    elif command == "apply-checked-out": apply_checked_out()
    elif command == "start-loop": start_loop()
    elif command == "loop-count": loop_count()
    elif command == "journal-close": emit(journal_close_summary(sys.argv[2], next_open_ok="--next-open-ok" in sys.argv))
    elif command == "work-dir": print(work_dir())
    elif command == "journal-open": emit(expected_open(sys.argv[2]))
    elif command == "open-check":
        override = os.environ.get("KINGX2_DRY_OPEN_CHECK", "") if DRY else ""
        if override:
            completed = sh(["bash", "-c", override + " " + shlex.quote(sys.argv[2])])
            sys.stdout.write(completed.stdout)
        else:
            emit(open_check(sys.argv[2]))
    elif command == "public-check":
        override = os.environ.get("KINGX2_DRY_PUBLIC_CHECK", "") if DRY else ""
        if override:
            completed = sh(["bash", "-c", override + " " + shlex.quote(sys.argv[2])], timeout=float(EXPECT["public_retry_seconds"]) + 30)
            sys.stdout.write(completed.stdout)
        else:
            emit(public_check(sys.argv[2]))
    elif command == "health": emit(health(sys.argv[2]))
    elif command == "rollback": rollback()
    elif command == "facts": facts()
    else: raise RuntimeError(f"unknown command {command}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
