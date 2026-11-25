#!/usr/bin/env python3
"""Dedup deploy operator preserving the live Generator feed.

Only an operator may use this against a live checkout. Inspection never prints
secrets. set-params, start-loop, the bounded evidence precheck and the
work-directory lookup use the existing operator loop environment. The external
reserve, ingestd and generator inbox are never stopped, moved or reseeded.
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
import subprocess
import sys
import time
import urllib.error
import urllib.request

DRY = os.environ.get("DEDUP_DRY") == "1"
V2 = Path(os.environ.get("DEDUP_V2", "/root/sn120/v2"))
HERE = Path(os.environ.get("DEDUP_HERE", "/root/sn120-run/dedup-deploy"))
STATE = Path(os.environ.get("DEDUP_STATE", "/root/sn120-run/state"))
PUBLIC = os.environ.get("DEDUP_PUBLIC", "https://affine.io").rstrip("/")
PY = os.environ.get("DEDUP_PY", "/root/sn120/v2/.venv/bin/python")
STACK = os.environ.get("DEDUP_STACK", "sn120_v2.assembly:stack")
LOOP = os.environ.get("DEDUP_LOOP_SCRIPT", "systemd/sn120-v2-run-loop.sh")
ENV_FILE = Path(os.environ.get("DEDUP_ENV_FILE", "/root/sn120-run/secrets/loop-env-903093.environ"))
WORK = Path(os.environ["DEDUP_WORK_DIR"]) if os.environ.get("DEDUP_WORK_DIR") else None
JOURNAL = STATE / "journal.jsonl"
TARGET = HERE / "rollback-target.json"
EXPECT = json.loads((HERE / "expect.json").read_text())
RESERVE_PATH = Path(os.environ.get("DEDUP_RESERVE_PATH_FILE", EXPECT["reserve_path_file"]))
LIVE = Path(os.environ.get("DEDUP_RESERVE_DIR", EXPECT["reserve_dir"]))
INBOX = Path(os.environ.get("DEDUP_GENERATOR_INBOX", EXPECT["generator_inbox"]))
INGESTD_PID = Path(os.environ.get("DEDUP_GENERATOR_PID", EXPECT["generator_ingestd_pid"]))
PRIOR_INDEX = Path(os.environ.get("DEDUP_PRIOR_INDEX", EXPECT["prior_index"]))
_package = V2 / "sn120_v2"
_loaded = sys.modules.get("sn120_v2")
if _loaded is None or Path(_loaded.__file__).resolve().parent != _package.resolve():
    _spec = importlib.util.spec_from_file_location("sn120_v2", _package / "__init__.py", submodule_search_locations=[str(_package)])
    assert _spec is not None and _spec.loader is not None
    sys.modules["sn120_v2"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["sn120_v2"])
from sn120_v2 import layout  # noqa: E402
from sn120_v2.canonical import sha256_hex  # noqa: E402
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
    if _name.startswith("DEDUP_") or _name == "PYTHONPATH":
        ENV[_name] = _value
UA = {"User-Agent": "sn120-dedup-deploy/1", "Accept-Encoding": "identity", "Cache-Control": "no-cache"}


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
    return Path(os.environ.get("DEDUP_LOOP_PIDFILE", str(work_dir() / "logs/run-loop.pid")))


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
    """Switch code only. The external reserve belongs to the running ingestd."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("refusing a non-SHA checkout")
    sh(["git", "checkout", "-q", "--detach", commit], cwd=V2.parent)
    if head() != commit:
        raise RuntimeError("checkout did not reach the requested commit")


def ingestd_identity() -> dict[str, str | int | bool | None]:
    """Bind the daemon identity before checkout and recheck it after rollback.

    A live daemon may append to LIVE concurrently. Never hash, copy, move,
    delete or reseed that mutable directory as a checkout preservation step.
    """
    result: dict[str, str | int | bool | None] = {"pidfile": str(INGESTD_PID), "running": False,
                                                   "pid": None, "command": None, "start_ticks": None}
    if not INGESTD_PID.is_file():
        raise RuntimeError(f"ingestd pidfile is missing: {INGESTD_PID}")
    value = INGESTD_PID.read_text().strip()
    if not value.isdigit():
        raise RuntimeError("ingestd pidfile does not contain one numeric pid")
    pid = int(value)
    result["pid"] = pid
    # The dry fixture runs the real disposable ingestd, so it gets the same
    # process identity checks as production; a numeric pidfile alone is not OK.
    try:
        proc = Path(f"/proc/{pid}")
        args = [part.decode(errors="replace") for part in proc.joinpath("cmdline").read_bytes().split(b"\0") if part]
        command = " ".join(args)
        stat = proc.joinpath("stat").read_text().rsplit(") ", 1)[1].split()
        cwd = Path(os.readlink(proc / "cwd")).resolve()
    except FileNotFoundError:
        raise RuntimeError(f"ingestd pid {pid} from {INGESTD_PID} is not running") from None
    script = V2 / "systemd/generator-ingestd.sh"
    scripts = [Path(arg) for arg in args if Path(arg).name == "generator-ingestd.sh"]
    if len(scripts) != 1 or scripts[0].resolve() != script.resolve():
        raise RuntimeError(f"pidfile names an unexpected process: {command[:240]}")
    if len(stat) <= 19 or stat[0] == "Z":
        raise RuntimeError("ingestd process is a zombie or has an invalid proc stat")
    if cwd != V2.resolve():
        raise RuntimeError("ingestd is not running from the selected v2 checkout")
    if str(PRIOR_INDEX) not in args:
        raise RuntimeError("ingestd is not using the expected generator prior index")
    result.update(running=True, command=command, start_ticks=int(stat[19]))
    return result


def verify_ingestd_preserved(before: dict | None = None) -> dict:
    after = ingestd_identity()
    fields = ("pid", "command", "start_ticks")
    if before is not None and any(before.get(field) != after[field] for field in fields):
        raise RuntimeError("ingestd identity changed; refusing to replace or kill the live daemon")
    return after


def other_deploys() -> list[str]:
    """Exclude this shell ancestry, including Python's own [python3] <defunct>."""
    ancestry = {os.getpid()}
    current = os.getpid()
    for _ in range(64):
        try:
            fields = Path(f"/proc/{current}/stat").read_text().split()
            current = int(fields[3])
        except (FileNotFoundError, ValueError, IndexError):
            break
        if current <= 1 or current in ancestry:
            break
        ancestry.add(current)
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestry:
            continue
        try:
            raw = entry.joinpath("cmdline").read_bytes()
            args = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
            state = entry.joinpath("stat").read_text().split()[2]
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
            continue
        if state == "Z":
            continue
        for arg in args:
            path = Path(arg)
            if ((path.name in ("autodeploy.sh", "ol_operator.py") and "genfeed-deploy" in path.parts)
                    or (path.name in ("autodeploy", "kx2_operator.py") and "kingx2-deploy" in path.parts)
                    or (path.name in ("autodeploy", "dedup_operator.py") and "dedup-deploy" in path.parts)):
                found.append(f"{entry.name} {' '.join(args)[:360]}")
                break
    return sorted(found)


def verify_live_reserve(*, ingestd_before: dict | None = None) -> dict:
    """Resolve and verify the existing selector; no unlink, reseed or rewrite."""
    if not RESERVE_PATH.is_file() or not LIVE.is_absolute():
        raise RuntimeError("reserve-live.path or its absolute live directory is missing")
    if RESERVE_PATH.read_text() != str(LIVE) + "\n":
        raise RuntimeError("reserve-live.path does not select the expected existing live reserve")
    if not LIVE.is_dir() or not INBOX.is_dir() or not PRIOR_INDEX.is_file():
        raise RuntimeError("external live reserve, prior index or generator inbox is missing")
    from sn120_v2.envs.reserve import ingest_lock, load_reserve
    # Shared-lock verification is read-only. The verify CLI creates a lockfile
    # for a missing directory; never use that on the production selector.
    with ingest_lock(LIVE, exclusive=False):
        reserve, _ = load_reserve(LIVE)
    return {"path": str(RESERVE_PATH), "live": str(LIVE), "inbox": str(INBOX),
            "verification": {"shards": len(reserve.shards), "manifest_sha256": reserve.manifest_sha},
            "ingestd": verify_ingestd_preserved(ingestd_before)}


def start_ingestd() -> None:
    """Preserve the genfeed daemon. Never turn preservation into a restart."""
    verify_ingestd_preserved()
    log(f"ingestd preserved: live pid {INGESTD_PID.read_text().strip()}; no restart needed")


def generator_check() -> None:
    out = verify_live_reserve()
    out["ok"] = True
    emit(out)


def deployment_guard() -> None:
    found = other_deploys()
    if found:
        raise RuntimeError(f"another genfeed/dedup deploy is alive: {found}")
    log("no competing genfeed/dedup deployment is alive")


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
    if filename == "old-params.json" and head() != EXPECT["old_commit"]:
        raise RuntimeError("legacy king-x2 params require the reviewed old-code checkout and compatibility loader")
    if filename == "new-params.json" and head() != EXPECT["new_commit"]:
        raise RuntimeError("new params require the reviewed new-code checkout")
    result = None
    if DRY:
        command = os.environ.get("DEDUP_DRY_SET_PARAMS", "")
        if not command:
            raise RuntimeError("dry set-params needs DEDUP_DRY_SET_PARAMS")
        result = sh(["bash", "-c", command + " " + shlex.quote(str(path))], timeout=3600)
    else:
        env = load_env()
        if not env.get("SN120_WORK_DIR"):
            raise RuntimeError("loop environment has no SN120_WORK_DIR")
        command = [PY, "-I", "-m", "sn120_v2"] if filename == "new-params.json" else [str(HERE / "rollback-python")]
        result = sh([*command, "set-params", "--state-dir", STATE,
                     "--stack", STACK, "--params", path], timeout=3600, env=env | {"DEDUP_OLD_V2": str(V2)})
    (HERE / f"set-params-{Path(filename).stem}.json").write_text(result.stdout)
    if result.stdout.strip():
        log(f"{filename}: {result.stdout.strip()[:1400]}")


def cli(command: str, *, stop_after_round: bool = False) -> subprocess.CompletedProcess:
    """Run a controller command under the currently checked-out code."""
    if stop_after_round and command != "run":
        raise ValueError("stop-after-round applies only to run")
    if DRY:
        override = os.environ.get(f"DEDUP_DRY_{command.upper().replace('-', '_')}", "")
        if not override:
            raise RuntimeError(f"dry {command} needs its isolated CLI override")
        if stop_after_round:
            if alive():
                raise RuntimeError(f"bounded recovery run refused: loop alive {alive()}")
            stop = work_dir() / "logs/run-loop.stop-after-round"
            if stop.exists():
                raise RuntimeError("bounded recovery run refused: a stop-after-round request already exists")
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.touch()
            try:
                # Keep the operator's explicit stop path, not a sanitized
                # environment in which StopRequest.path is None.
                return sh(["bash", "-c", override], timeout=3600, check=False,
                          env=ENV | {"SN120_WORK_DIR": str(work_dir())})
            finally:
                if stop.exists():
                    raise RuntimeError("bounded recovery run did not remove its stop-after-round request; remove it only after checking why")
        return sh(["bash", "-c", override], timeout=3600, env=ENV)
    env = load_env()
    if not env.get("SN120_WORK_DIR"):
        raise RuntimeError("loop environment has no SN120_WORK_DIR")
    stop = work_dir() / "logs/run-loop.stop-after-round"
    if stop_after_round:
        if alive():
            raise RuntimeError(f"bounded recovery run refused: loop alive {alive()}")
        if stop.exists():
            raise RuntimeError("bounded recovery run refused: a stop-after-round request already exists")
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.touch()
        env.pop("SN120_LOOP_MARK", None)
    try:
        return sh([PY, "-I", "-m", "sn120_v2", command, "--state-dir", STATE,
                   "--stack", STACK], timeout=3600, env=env, check=not stop_after_round)
    finally:
        if stop_after_round and stop.exists():
            raise RuntimeError("bounded recovery run did not remove its stop-after-round request; remove it only after checking why")


def sealed_result_leases() -> list[str]:
    """Check every lease; end-round itself checks only leases still marked open."""
    if DRY:
        from sn120_v2.store import LocalStore
        root = os.environ.get("DEDUP_OBJECT_ROOT") or os.environ.get("DEDUP_PUBLIC_ROOT")
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


def finish_open_for_rollback() -> None:
    """Finish every open dedup round with reviewed new code before rollback.

    A sealed result, including one on a failed lease, is judged normally. Never
    use ``end-round --discard-result`` to make an open dedup round disappear.
    Old code cannot finish even a decided round: stripping its historical
    fields would change the stored ticket's Params digest.
    """
    if head() != EXPECT["new_commit"]:
        raise RuntimeError("an open dedup round must be finished with the reviewed new-code checkout")
    record = load(JOURNAL).round
    if record is None:
        return
    if record.params_hash != EXPECT["new_params_hash"]:
        raise RuntimeError(f"rollback blocked: {record.round_id} does not use the reviewed dedup params")
    rid = record.round_id
    stage = "decided" if record.decision is not None else "undecided"
    stored = sealed_result_leases()
    if stored:
        log(f"rollback: {rid} has sealed evidence on {stored}; reviewed new-code run will judge it normally")
    result = cli("run", stop_after_round=True)
    # ``autodeploy`` appends this real bounded-run stderr to its audit log.
    # Do not replace the stop contract with a dry fixture's synthetic marker.
    sys.stderr.write(result.stderr)
    sys.stderr.flush()
    if result.returncode:
        raise RuntimeError(f"rollback blocked: new-code run for {rid} exited {result.returncode}; sealed evidence was preserved")
    if not any(line.startswith("stop after the round: obeyed.") for line in result.stderr.splitlines()):
        raise RuntimeError(f"rollback blocked: new-code run did not confirm its stop-after-round request for {rid}")
    after = load(JOURNAL)
    if after.round is not None or not after.rounds or after.rounds[-1].round_id != rid:
        raise RuntimeError(f"rollback blocked: new-code run did not close {rid}")
    closed = after.rounds[-1]
    if closed.outcome not in ("null", "applied", "no_decision"):
        raise RuntimeError(f"rollback blocked: {rid} closed as {closed.outcome}; inspect it before changing code")
    log(f"rollback: reviewed new-code run finished {stage} {rid} as {closed.outcome}; no stored result was discarded")



def ensure_params(filename: str, expected: str) -> None:
    old_file = filename == "old-params.json"
    hash_file = old_params_hash if old_file else params_hash
    current = old_status if old_file else status
    if hash_file(HERE / filename) != expected:
        raise RuntimeError(f"{filename} does not have the expected params hash")
    if current()["params_hash"] != expected:
        set_params(filename)
    if current()["params_hash"] != expected:
        raise RuntimeError(f"set-params did not pin {expected}")


def old_status() -> dict:
    """Replay through the same hashed compatibility loader as the old loop."""
    if head() != EXPECT["old_commit"]:
        raise RuntimeError("old-code compatibility status requires the reviewed old-code checkout")
    if DRY:
        override = os.environ.get("DEDUP_DRY_ROLLBACK_STATUS", "")
        if not override:
            raise RuntimeError("dry old-code status needs its compatibility CLI override")
        report = json.loads(sh(["bash", "-c", override], timeout=180).stdout)
    else:
        # The same compatibility loader used by the recovered loop must also
        # inspect old execution state. The long-lived new-code process cannot
        # safely reinterpret stripped historical Params through its new schema.
        code = (
            "import json,runpy,sys;"
            "sys.path.insert(0," + repr(str(HERE)) + ");"
            "import rollback_cli;"
            "rollback_cli.install(" + repr(str(HERE)) + ");"
            "from sn120_v2.cli import summary;"
            "from sn120_v2.state import load;"
            "state=load(" + repr(str(JOURNAL)) + ");"
            "report=summary(state,state.params);"
            "report['batch_cap']=state.params.batch_cap;"
            "report['recipe_code_digest']=state.params.recipe.code_digest;"
            "print(json.dumps(report))"
        )
        result = sh([PY, "-I", "-c", code], cwd=HERE,
                    env=ENV | {"DEDUP_OLD_V2": str(V2)}, timeout=180)
        report = json.loads(result.stdout)
    return report


def old_params_hash(path: Path) -> str:
    """Hash a params file under old code without importing any new defaults."""
    if head() != EXPECT["old_commit"]:
        raise RuntimeError("legacy params hashing requires the reviewed old-code checkout")
    if DRY:
        override = os.environ.get("DEDUP_DRY_ROLLBACK_PARAMS_HASH", "")
        if not override:
            raise RuntimeError("dry legacy params hashing needs its compatibility CLI override")
        return sh(["bash", "-c", override + " " + shlex.quote(str(path))], timeout=180).stdout.strip()
    code = (
        "import runpy,sys;sys.argv=sys.argv[1:];"
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    result = sh([PY, "-I", "-c", code, str(HERE / "rollback_cli.py"), "--kit", str(HERE),
                 "verify-params", str(path)],
                cwd=HERE, env=ENV | {"DEDUP_OLD_V2": str(V2)}, timeout=180)
    return result.stdout.strip()


def verify_before() -> dict:
    report = old_status()
    if report.get("phase") != "IDLE":
        raise RuntimeError(f"journal phase is {report.get('phase')}, not IDLE")
    if head() != EXPECT["old_commit"]:
        raise RuntimeError(f"HEAD {head()} != old commit {EXPECT['old_commit']}")
    if digest() != EXPECT["old_digest"]:
        raise RuntimeError("old code digest mismatch")
    if report.get("batch_cap") != 6:
        raise RuntimeError("live genfeed batch_cap=6 differs from old params")
    if report["params_hash"] != EXPECT["old_params_hash"]:
        raise RuntimeError("old pinned params hash mismatch")
    if report.get("recipe_code_digest") != EXPECT["old_digest"]:
        raise RuntimeError("old pinned code digest mismatch")
    # Rollback parses the complete king-x2 file only after the old-code checkout.
    actual = old_params_hash(HERE / "old-params.json")
    if actual != EXPECT["old_params_hash"]:
        raise RuntimeError(f"old-params.json hashes to {actual}, expected {EXPECT['old_params_hash']}")
    return report


def preflight() -> None:
    if other_deploys():
        raise RuntimeError(f"another genfeed/dedup deploy is alive: {other_deploys()}")
    verify_live_reserve()
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
    if other_deploys():
        raise RuntimeError(f"another genfeed/dedup deploy is alive: {other_deploys()}")
    ingestd_before = verify_live_reserve()["ingestd"]
    # The new-code child must prove that this exact daemon survived checkout.
    (HERE / "ingestd-before.json").write_text(json.dumps(ingestd_before, sort_keys=True, indent=2) + "\n")
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
                                  "ingestd": ingestd_before,
                                  "recorded_at": time.strftime("%FT%TZ", time.gmtime())}, indent=2) + "\n")
    checkout(new)
    try:
        # Params and State were imported from the old checkout in this process.
        # A fresh interpreter must parse the new schema and replay new params.
        sh([sys.executable, "-I", Path(__file__).resolve(), "apply-checked-out"], timeout=3600)
    except BaseException as error:
        # This child also starts fresh under new code, so it can finish any
        # open dedup round before checking out old code and its params.
        recovered = sh([sys.executable, "-I", Path(__file__).resolve(), "rollback"], check=False, timeout=3600)
        if recovered.returncode:
            raise RuntimeError(f"apply failed ({error}); rollback also failed: {recovered.stderr.strip()[-1000:]}") from error
        raise


def apply_checked_out() -> None:
    """Run only in a fresh interpreter after the new commit is checked out."""
    new = EXPECT["new_commit"]
    if head() != new:
        raise RuntimeError("post-checkout HEAD is not the new commit")
    target = json.loads(TARGET.read_text())
    saved_ingestd = json.loads((HERE / "ingestd-before.json").read_text())
    if target.get("ingestd") != saved_ingestd:
        raise RuntimeError("apply ingestd identity target is absent or changed")
    try:
        if digest() != EXPECT["new_digest"]:
            raise RuntimeError("new code digest mismatch")
        ensure_params("new-params.json", EXPECT["new_params_hash"])
        state = load(JOURNAL)
        if state.params.recipe.code_digest != EXPECT["new_digest"] or state.params.irt_draw_rule != EXPECT["irt_draw_rule"]:
            raise RuntimeError("new params did not pin the combined worker recipe or sqrt_fisher rule")
        if state.params.null_samples != EXPECT["null_samples"] or state.params.n_tasks != EXPECT["n_tasks"] or state.params.batch_cap != EXPECT["batch_cap"]:
            raise RuntimeError("new null sample count or task count mismatch")
        if state.params.rotation_active_size != EXPECT["active_size"]:
            raise RuntimeError("new active size mismatch")
        if not state.params.dedup_enabled or state.params.train_seed_scope != "round" or not state.params.eval_merged:
            raise RuntimeError("new params did not enable round-seeded dedup and merged evaluation")
        if (state.params.dedup_rollout_overlap, state.params.dedup_cos_shared_seed,
                state.params.dedup_cos_own_seed, state.params.min_signal_rollouts, state.params.min_step_norm,
                state.params.crown_gate, state.params.z_field_bar, state.params.z_bar) != (
                    0.5, 0.135, 0.002, 1, 0.0, "argmax_symmetric", 0.0, 0.0):
            raise RuntimeError("new params do not match the reviewed field-gate thresholds")
        if state.params.batch_cap != EXPECT["batch_cap"]:
            raise RuntimeError("new params did not preserve genfeed batch_cap=6")
        verify_live_reserve(ingestd_before=saved_ingestd)
        log(f"APPLY DONE: HEAD {head()} digest {digest()} params {state.params.params_hash}; ingestd/reserve preserved")
    except BaseException:
        # The historical dedup records stay intact. Old code must be checked
        # out before its complete king-x2 file can be applied through the
        # reviewed compatibility loader; new code must never read that file.
        rollback()
        raise


def start_loop() -> None:
    if alive():
        raise RuntimeError(f"start-loop refused: already alive {alive()}")
    if DRY:
        path = pidfile()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not os.environ.get("DEDUP_DRY_RUN"):
            raise RuntimeError("dry start-loop needs the real isolated run command")
        path.write_text(str(os.getpid()) + "\n")
        log("dry: one real isolated CLI run requested; no production service")
        return
    env = load_env()
    stop = work_dir() / "logs/run-loop.stop-after-round"
    if stop.exists():
        raise RuntimeError("stop-after-round file present")
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    command = ["setsid", "nohup", "bash", LOOP, PY, "--state-dir", STATE, "--stack", STACK]
    if head() == EXPECT["old_commit"]:
        env["DEDUP_OLD_V2"] = str(V2)
        command = ["setsid", "nohup", "bash", str(HERE / "rollback-loop.sh"),
                   "--state-dir", STATE, "--stack", STACK]
    elif head() != EXPECT["new_commit"]:
        raise RuntimeError("start-loop requires either reviewed old or new HEAD")
    with open("/dev/null", "rb") as null:
        subprocess.Popen(command,
                         cwd=V2, env=env, stdin=null, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(float(os.environ.get("DEDUP_START_WAIT", "5")))
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
        # The site deliberately serves the named rotation, not pool/state.
        deadline = time.monotonic() + float(EXPECT["public_retry_seconds"])
        actual = public_rotation(rid, rotation, deadline=deadline)
        active_raw = public_poll(lambda left: pub_get(layout.active_pool(actual.active_sha), deadline=left), deadline=deadline)
        active_ids = json.loads(active_raw)
        if (sha256_hex(active_raw) != actual.active_sha or active_ids != list(actual.active_ids)
                or active_ids != sorted(active_ids) or len(set(active_ids)) != len(active_ids)
                or any(not isinstance(task, str) or not task for task in active_ids)):
            out["why"].append("content-addressed active bytes/ids differ from sorted unique rotation ids")
        snapshot = public_poll(lambda left: pub_get(layout.irt_snapshot(record["snapshot_id"]), deadline=left), deadline=deadline)
        snapshot = json.loads(snapshot)
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
    if settled or (opened and isinstance(opened.data["round"].get("decision"), dict)):
        decision = settled[-1].data.get("decision") if settled else opened.data["round"]["decision"]
        decision = decision or {}
        entries = decision.get("entries") or []
        out.update(n_tasks=decision.get("n_tasks"), null_pass_rate=decision.get("null_pass_rate"))
        out["cut_counts"] = {
            "duplicate": sum(entry.get("status") == "duplicate" for entry in entries),
            "null_update": sum(entry.get("status") == "null_update" for entry in entries),
        }
        selected = len(entries)
        duplicate_cuts = sum(entry.get("status") == "duplicate" for entry in entries)
        null_cuts = sum(entry.get("status") == "null_update" for entry in entries)
        out["selected_batches"] = selected
        out["surviving_batches"] = selected - duplicate_cuts - null_cuts
        out["cut_counts"] = {"duplicate": duplicate_cuts, "null_update": null_cuts}
        if selected and duplicate_cuts + null_cuts == selected:
            why.append("dedup/null-update cuts removed every selected batch")
        out["duplicate_clusters"] = len(decision.get("duplicate_clusters") or []) + len(opened.data["round"].get("duplicate_clusters") or [])
        if out["duplicate_clusters"] and not duplicate_cuts:
            why.append("duplicate clusters have no corresponding duplicate decisions")
        if not decision.get("n_tasks"):
            why.append("decision n_tasks 0")
    out.update(clean=not why, why=why)
    return out


def public_round(rid: str) -> dict:
    raw = pub_get(layout.round_file(rid))
    report = json.loads(raw)
    return {"raw": raw, "report": report}


def run_log_eval_wall(rid: str, path: Path) -> int | None:
    """Read this round's eval field from the actual ``round_text`` run-log block.

    The GPU phase JSONL is optional in production. ``run`` instead prints the
    public round report's text, whose ``_span`` fields are whole seconds.
    """
    in_round = False
    found = None
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        header = re.match(r"^[0-9:.TZ -]*SN120 v2 round (r[0-9]{6}):", line)
        if header:
            in_round = header.group(1) == rid
            continue
        if not in_round:
            continue
        summary = re.search(r"(?:^|\s)wall time: (.*)$", line)
        if summary:
            match = re.search(r"(?:^|\| )eval ([0-9]+)m([0-9]{2})s(?: \||$)", summary.group(1))
            if match:
                found = int(match.group(1)) * 60 + int(match.group(2))
                in_round = False
    return found


def run_log_cut_counts(rid: str, path: Path) -> dict[str, int]:
    """Return only the file-check block belonging to this round's open.

    Repeated checks and later rounds are not a per-round cut count. The
    verdict-log format omits rules with zero invalid seats; a complete matching
    file-check block therefore proves zero for both omitted rules. Without a
    matching file-check block, the evidence is missing rather than zero.
    """
    if not path.is_file():
        return {}
    pattern = re.compile(
        r"^files: check at .*: [0-9]+ waiting seats, [0-9]+ valid, "
        r"[0-9]+ not valid(?: \(([^)]*)\))?, [0-9]+ with no file, [0-9]+ changed$"
    )
    counts: dict[str, int] = {}
    current: dict[str, int] = {}
    observed: dict[str, bool] = {}
    saw_check = False
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            current = {}
            observed = {name: False for name in ("min_rollouts", "null_update")}
            saw_check = True
            for item in (match.group(1) or "").split(", "):
                rule, _, number = item.rpartition(" ")
                if rule in observed and number.isdigit():
                    current[rule] = int(number)
                    observed[rule] = True
            continue
        if re.search(rf"open: round {re.escape(rid)} is open(?:,|$)", line):
            counts = {name: current.get(name, 0) for name in observed} if saw_check else {}
            current = {}
    return counts

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
        cuts = summary.get("cut_counts") or {}
        out["cut_counts"] = cuts
        out["selected_batches"] = summary.get("selected_batches")
        out["surviving_batches"] = summary.get("surviving_batches")
        if set(cuts) != {"duplicate", "null_update"}:
            out["why"].append("journal duplicate/null-update cut counts are missing")
        run_log = Path(os.environ.get("DEDUP_RUN_LOG", str(work_dir() / "logs/run.log")))
        out["admission_cut_counts"] = run_log_cut_counts(rid, run_log)
        # Disabled floors normally produce no admission cuts. The required
        # evidence is the journal's actual duplicate/null-update decision
        # counts, which the autodeploy summary logs. Admission counters remain
        # optional diagnostics; an absent file-check block must not fail an
        # otherwise healthy no-op-floor round.
        if not all(key in out["admission_cut_counts"] for key in ("min_rollouts", "null_update")):
            out["admission_cut_counts_note"] = "no round-bound file-check block; floors are no-ops"
        wall = report.get("wall_s") or {}
        eval_wall = wall.get("evaluation")
        if type(eval_wall) not in (int, float) or not math.isfinite(eval_wall) or eval_wall <= 0:
            out["why"].append("public round evaluation wall time is missing or not positive")
        out["eval_wall_s"] = eval_wall
        out["wall_s"] = wall
        phases = []
        phase_path = Path(os.environ["DEDUP_PHASES"]) if os.environ.get("DEDUP_PHASES") else work_dir() / "logs/phases.jsonl"
        if phase_path.is_file():
            for line in phase_path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                    if (row.get("round_id") == rid and row.get("phase") in ("eval", "solo_eval", "eval_stream", "eval_cut")
                            and type(row.get("wall_s")) in (int, float) and math.isfinite(row["wall_s"]) and row["wall_s"] >= 0):
                        phases.append({"at": row.get("at"), "phase": row["phase"], "look": row.get("look"), "wall_s": row["wall_s"]})
                except (ValueError, TypeError):
                    continue
        out["eval_phase_wall_s"] = phases
        out["run_log_eval_wall_s"] = run_log_eval_wall(rid, run_log)
        out["eval_timing_evidence"] = "phases" if phases else ("run_log" if out["run_log_eval_wall_s"] is not None else "public_eval_wall_s")
        if (type(out["run_log_eval_wall_s"]) is int and type(eval_wall) in (int, float)
                and abs(out["run_log_eval_wall_s"] - eval_wall) > 1):
            out["why"].append("run-log evaluation wall time differs from the public round report")
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
    """One bounded public fetch. Callers choose the shared retry deadline."""
    root = os.environ.get("DEDUP_PUBLIC_ROOT")
    if root:
        raw = (Path(root) / key).read_bytes()
    else:
        timeout = 90.0 if deadline is None else max(0.1, min(90.0, deadline - time.monotonic()))
        url = PUBLIC + "/" + key + "?dedup=" + str(time.time_ns())
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as reply:
            raw = reply.read()
    return json.loads(raw) if object_json else raw


def public_poll(fetch, *, deadline: float | None = None):
    """Retry transport/publication delays; never retry an integrity failure."""
    deadline = time.monotonic() + float(EXPECT["public_retry_seconds"]) if deadline is None else deadline
    while True:
        try:
            return fetch(deadline)
        except (OSError, urllib.error.HTTPError) as error:
            # HTTPError is also URLError/OSError. Only 404 and 5xx represent a
            # possibly unfinished public mirror; schema/hash failures propagate.
            if isinstance(error, urllib.error.HTTPError) and error.code != 404 and error.code < 500:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))


def public_rotation(rid: str, journal_rotation: dict, *, deadline: float | None = None) -> Rotation:
    """Authenticate the site's named rotation against the local journal."""
    rotation = json.loads(public_poll(
        lambda left: pub_get(layout.pool_rotation(rid), deadline=left), deadline=deadline))
    actual = Rotation.from_dict(rotation)
    expected = dict(journal_rotation)
    state_sha, size = expected.pop("state_sha", None), expected.pop("size", None)
    if actual.digest() != state_sha or len(actual.active_ids) != size or actual.journal_block() != expected:
        raise ValueError("public rotation differs from journal state_sha/size/block")
    return actual


def public_check(rid: str) -> dict:
    out = {"round": rid, "ok": False, "why": []}
    try:
        rows = read_journal(JOURNAL)
        opened = opened_row(rows, rid)
        if opened is None:
            raise ValueError("round_opened not found")
        record = opened.data["round"]
        journal_rotation = dict(opened.data.get("rotation") or record.get("rotation") or {})
        deadline = time.monotonic() + float(EXPECT["public_retry_seconds"])
        actual = public_rotation(rid, journal_rotation, deadline=deadline)
        state_sha = journal_rotation["state_sha"]
        if len(actual.active_ids) != EXPECT["active_size"] or actual.round_id != rid:
            out["why"].append("public rotation round/active size differs from the expected deployment")
        out["active_size"] = len(actual.active_ids)
        out["rotation_state_sha"] = state_sha
        active_raw = public_poll(lambda left: pub_get(layout.active_pool(actual.active_sha), deadline=left), deadline=deadline)
        if sha256_hex(active_raw) != actual.active_sha:
            out["why"].append("active pool raw bytes do not match active_sha")
        active = json.loads(active_raw)
        if (active != list(actual.active_ids) or active != sorted(active)
                or len(set(active)) != len(active) or any(not isinstance(task, str) or not task for task in active)):
            out["why"].append("content-addressed active pool is not the rotation's sorted unique ids")
        snapshot = json.loads(public_poll(
            lambda left: pub_get(layout.irt_snapshot(record["snapshot_id"]), deadline=left), deadline=deadline))
        verify_snapshot(snapshot)
        if snapshot["snapshot_id"] != record["snapshot_id"] or set(snapshot["tasks"]) != set(actual.active_ids):
            out["why"].append("public IRT snapshot identity/active task set mismatch")
        out["snapshot_config"] = snapshot.get("config") or {}
        if out["snapshot_config"].get("draw_rule") != EXPECT["irt_draw_rule"]:
            out["why"].append("public IRT snapshot is not sqrt_fisher")
        if float(out["snapshot_config"].get("uniform_frac")) != float(load_params(HERE / "new-params.json").irt_explore_frac):
            out["why"].append("public IRT snapshot exploration share mismatch")
        report = json.loads(public_poll(lambda left: pub_get(layout.round_file(rid), deadline=left), deadline=deadline))
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
    # Code first: new code must finish every open dedup round. It must never
    # apply the legacy king-x2 file, whose omitted dedup fields would activate
    # new defaults. Only the checked-out old compatibility CLI may re-pin it.
    finish_open_for_rollback()
    checkout(old_commit)
    ensure_params("old-params.json", EXPECT["old_params_hash"])
    if head() != old_commit or digest() != EXPECT["old_digest"]:
        raise RuntimeError("rollback old-code commit/GPU digest verification failed")
    # Status must use the same compatibility loader as the recovered loop. The
    # bare old interpreter intentionally rejects historical dedup records.
    report = old_status()
    if report.get("params_hash") != EXPECT["old_params_hash"]:
        raise RuntimeError("rollback old params hash verification failed")
    if report.get("recipe_code_digest") != EXPECT["old_digest"]:
        raise RuntimeError("rollback pinned digest mismatch")
    if report.get("batch_cap") != 6:
        raise RuntimeError("rollback did not preserve genfeed batch_cap=6")
    verify_live_reserve(ingestd_before=target.get("ingestd"))
    log(f"ROLLBACK DONE -> {old_commit}; old params restored; ingestd/reserve preserved")


def facts() -> None:
    log(f"facts: HEAD {head()} code_digest {digest()} phase {status().get('phase')} params {status().get('params_hash')}")


def emit(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    if len(sys.argv) < 2:
        raise RuntimeError("missing operator command")
    command = sys.argv[1]
    if command not in ("preflight", "apply", "apply-checked-out", "start-loop", "start-ingestd", "generator-check", "deployment-guard", "loop-count", "work-dir", "rollback", "facts") and len(sys.argv) < 3:
        raise RuntimeError(f"{command} needs a round id")
    if command == "preflight": preflight()
    elif command == "apply": apply()
    elif command == "apply-checked-out": apply_checked_out()
    elif command == "start-loop": start_loop()
    elif command == "start-ingestd": start_ingestd()
    elif command == "generator-check": generator_check()
    elif command == "deployment-guard": deployment_guard()
    elif command == "loop-count": loop_count()
    elif command == "journal-close": emit(journal_close_summary(sys.argv[2], next_open_ok="--next-open-ok" in sys.argv))
    elif command == "work-dir": print(work_dir())
    elif command == "journal-open": emit(expected_open(sys.argv[2]))
    elif command == "open-check":
        override = os.environ.get("DEDUP_DRY_OPEN_CHECK", "") if DRY else ""
        if override:
            completed = sh(["bash", "-c", override + " " + shlex.quote(sys.argv[2])], timeout=float(EXPECT["public_retry_seconds"]) + 30)
            sys.stdout.write(completed.stdout)
        else:
            emit(open_check(sys.argv[2]))
    elif command == "public-check":
        override = os.environ.get("DEDUP_DRY_PUBLIC_CHECK", "") if DRY else ""
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
