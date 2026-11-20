#!/usr/bin/env python3
"""Generator-feed deploy operator. Never reads or prints secret values.

Production commands (run only by the operator):
  preflight | stage | apply | ingest-ready | start-loop | start-ingestd |
  stop-ingestd | journal-close RID [--next-open-ok] | open-check RID | public-check RID | rollback | facts
Dry run uses R2_DRY=1 plus R2_V2/R2_STATE/R2_HERE/R2_PUBLIC/R2_PY/R2_CUT.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

DRY = os.environ.get("R2_DRY") == "1"
V2 = Path(os.environ.get("R2_V2", "/root/sn120/v2"))
HERE = Path(os.environ.get("R2_HERE", "/root/sn120-run/generator/genfeed-deploy"))
STATE = Path(os.environ.get("R2_STATE", "/root/sn120-run/state"))
CUT = Path(os.environ.get("R2_CUT", "/root/sn120-run/cutover"))
PUBLIC = os.environ.get("R2_PUBLIC", "https://affine.io").rstrip("/")
PY = os.environ.get("R2_PY", "/root/sn120/v2/.venv/bin/python")
STACK = "sn120_v2.assembly:stack"
LOOP = "systemd/sn120-v2-run-loop.sh"
EXPECT = json.loads((HERE / "expect.json").read_text())
JOURNAL = STATE / "journal.jsonl"
TARGET = HERE / "rollback-target.json"
MANIFEST_REL = "sn120_v2/envs/reserve/MANIFEST.jsonl"
MANIFEST_GIT = "v2/" + MANIFEST_REL
RESERVE = V2 / "sn120_v2/envs/reserve"
PATH_FILE = Path(os.environ.get("R2_RESERVE_PATH_FILE", EXPECT["reserve_path_file"]))
LIVE = Path(os.environ.get("R2_RESERVE_DIR", EXPECT["reserve_dir"]))
INDEX = Path(os.environ.get("R2_PRIOR_INDEX", EXPECT["prior_index"]))
INBOX = Path(os.environ.get("R2_GENERATOR_INBOX", EXPECT["generator_inbox"]))
INGESTD_PID = Path(os.environ.get("R2_GENERATOR_PID", EXPECT["generator_ingestd_pid"]))
UA = {"User-Agent": "sn120-genfeed-deploy/1", "Accept-Encoding": "identity", "Cache-Control": "no-cache"}


def log(message):
    print(time.strftime("%H:%M:%SZ", time.gmtime()), message, flush=True)


def load_env():
    # Only start_loop calls this. Inspection/stage/apply/ingest never read secrets.
    path = Path("/root/sn120-run/secrets/loop-env-903093.environ")
    env = {}
    for item in path.read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode()
    for key in ("APPLY", "C", "CAPS", "MB", "OLDPWD"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def sh(cmd, cwd=V2, check=True, timeout=900, env=None):
    result = subprocess.run([str(x) for x in cmd], cwd=cwd, env=env or ENV, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise SystemExit(f"FAIL {cmd[:4]} rc={result.returncode}: {result.stderr.strip().splitlines()[-3:]}")
    return result


ENV = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "GIT_TERMINAL_PROMPT": "0"}


def head():
    return sh(["git", "rev-parse", "HEAD"]).stdout.strip()


def digest():
    return json.loads(sh([PY, "-m", "sn120_v2.gpu.plan"]).stdout)["code_digest"]


def status():
    return json.loads(sh([PY, "-m", "sn120_v2", "status", "--state-dir", STATE]).stdout)


def pinned_digest():
    return sh([PY, "-c", "import sys; from pathlib import Path; from sn120_v2.state import load; print(load(Path(sys.argv[1]) / 'journal.jsonl').params.recipe.code_digest)", STATE]).stdout.strip()


def pool_ok():
    code = sh([PY, "-c", "from sn120_v2.envs.math_pool import load_pool, POOL_SHA256; p=load_pool(); print(POOL_SHA256, len(p))"]).stdout.split()
    return code == [EXPECT["pool_sha"], str(EXPECT["pool_n"])], f"code {code}"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def params_hash(path):
    return sh([PY, "-c", "import sys; from sn120_v2.params import load_params; print(load_params(sys.argv[1]).params_hash)", path]).stdout.strip()


def alive():
    if DRY:
        return []
    result = subprocess.run(["pgrep", "-af", r"sn120-v2-run-loo[p]|sn120_v2 ru[n]"], capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if "_operator" not in line]


def reserve_verify():
    result = sh([PY, "-m", "sn120_v2.envs.reserve", "verify"], check=False)
    return result.returncode == 0, (result.stdout + result.stderr).strip().splitlines()[-3:]


def preflight():
    if alive():
        raise SystemExit(f"loop/run alive: {alive()}")
    dirty = sh(["git", "status", "--porcelain", "--untracked-files=all"]).stdout
    # The two supplied retired archives are intentionally untracked on old prod.
    other = [line for line in dirty.splitlines() if not line.endswith(("retired.json", "retired_tasks.json"))]
    if other:
        raise SystemExit(f"checkout not clean: {other[:10]}")
    state = status()
    if state.get("phase") != "IDLE" and not DRY:
        raise SystemExit(f"phase is {state.get('phase')}, not IDLE")
    log(f"preflight: HEAD {head()} code_digest {digest()} pinned {pinned_digest()} phase {state.get('phase')} params {state.get('params_hash')}")
    return state


def pub_get(key, timeout=90):
    request = urllib.request.Request(PUBLIC + "/" + key + "?genfeed=" + str(time.time_ns()), headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return reply.read()


def reserve_status():
    env = dict(ENV, SN120_RESERVE_DIR=str(LIVE))
    result = sh([PY, "-m", "sn120_v2.envs.reserve", "status"], check=False, env=env)
    try:
        return json.loads(result.stdout)
    except ValueError:
        return {"error": (result.stdout + result.stderr)[-400:]}


def manifest_rows(raw):
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or "withdrawn" not in rows[-1]:
        raise ValueError("reserve manifest has no final withdrawn pointer")
    return rows[:-1], rows[-1]


def append_only_public(packaged_raw, public_raw):
    """The public mirror may extend the packaged reserve, not replace its history."""
    packaged, _ = manifest_rows(packaged_raw)
    public, _ = manifest_rows(public_raw)
    return len(public) >= len(packaged) and public[:len(packaged)] == packaged


def checkout(commit):
    # Only the old packaged manifest is restored; the live reserve is external.
    sh(["git", "checkout", "-q", "--", MANIFEST_REL], check=False)
    sh(["git", "checkout", "-q", "--detach", commit])


def record_target(state):
    target = {"commit": head(), "params_hash": state["params_hash"], "code_digest": digest(), "pinned": pinned_digest(),
              "recorded_at": time.strftime("%FT%TZ", time.gmtime())}
    TARGET.write_text(json.dumps(target, indent=2, sort_keys=True) + "\n")
    if not DRY:
        shutil.copy2(CUT / "params.json", HERE / "params-before.json")
        shutil.copy2(CUT / "plan.json", HERE / "plan-before.json")
    log(f"rollback target recorded: {target['commit']} params {target['params_hash']} digest {target['code_digest']}")
    return target


def verify_prior_index(path=INDEX):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"prior index missing or unreadable; run stage: {exc}") from None
    try:
        document = json.loads(raw)
    except ValueError:
        raise SystemExit("prior index is not valid JSON; run stage") from None
    expected_pool = "sn120-math-prior-v1:" + EXPECT["prior_pool_sha"]
    if document.get("pool") != expected_pool or document.get("schema") != "sn120-math-prior-index-v1":
        raise SystemExit("prior index pool/schema mismatch")
    if document.get("rows") != EXPECT["prior_index_rows"] or document.get("source_rows") != EXPECT["prior_index_rows"]:
        raise SystemExit("prior index row counts differ from expect.json")
    if len(document.get("problems", {})) != EXPECT["prior_index_rows"]:
        raise SystemExit("prior index problem count differs from expect.json")
    if len(raw) != EXPECT["prior_index_bytes"] or hashlib.sha256(raw).hexdigest() != EXPECT["prior_index_sha256"]:
        raise SystemExit("prior index bytes/SHA differ from expect.json")
    log(f"prior index: {len(document['problems']):,} problems, {len(raw)} bytes, sha256 {hashlib.sha256(raw).hexdigest()}")


def download(url, target, sha):
    """Download an immutable public asset without trusting its filename."""
    if target.exists() and file_hash(target) == sha:
        return
    temporary = target.with_name(target.name + ".tmp")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as reply, temporary.open("wb") as stream:
        shutil.copyfileobj(reply, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if file_hash(temporary) != sha:
        raise SystemExit(f"public asset {target.name} SHA mismatch")
    os.replace(temporary, target)


def stage():
    """Build from the public release in scratch, never from the live checkout."""
    if not DRY and (not HERE.is_relative_to(INBOX.parent) or not INBOX.parent.is_relative_to(Path("/root/sn120-run/generator"))):
        raise SystemExit("stage work directory must stay under /root/sn120-run/generator/")
    work = HERE / "stage-work"
    work.mkdir(parents=True, exist_ok=True)
    release = work / "prior-release"
    release.mkdir(exist_ok=True)
    base = os.environ.get("R2_PRIOR_RELEASE", "https://github.com/AffineFoundation/affine/releases/download/sn120-math-prior-v1")
    download(base + "/MANIFEST.json", release / "MANIFEST.json", EXPECT["prior_pool_sha"])
    manifest = json.loads((release / "MANIFEST.json").read_bytes())
    for shard in manifest["shards"]:
        download(base + "/" + shard["file"], release / shard["file"], shard["sha256"])
    scratch = Path(tempfile.mkdtemp(prefix="build-", dir=work))
    try:
        archive = scratch / "package.tar"
        sh(["git", "archive", "-o", archive, EXPECT["new_commit"], "sn120_v2"])
        sh(["tar", "-xf", archive], cwd=scratch)
        output = scratch / "prior-index.json"
        sh([PY, "-m", "sn120_v2.envs.tools.prior_index", release, output], cwd=scratch, timeout=3600)
        verify_prior_index(output)
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        temporary = INDEX.with_name(INDEX.name + ".tmp")
        shutil.copyfile(output, temporary)
        os.replace(temporary, INDEX)
        os.chmod(INDEX, 0o644)
        verify_prior_index()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    log("STAGE DONE: prior index built from public release and atomically installed")


def apply():
    state = preflight()
    if head() != EXPECT["old_commit"] or state["params_hash"] != EXPECT["old_hash"] or digest() != EXPECT["digest"]:
        raise SystemExit(f"unexpected before state: HEAD {head()} hash {state.get('params_hash')}")
    if EXPECT["new_hash"] != EXPECT["old_hash"] or EXPECT["new_digest"] != EXPECT["digest"]:
        raise SystemExit("this is a code-only deploy; params/digest changes are refused")
    verify_prior_index()
    live_manifest = (V2 / MANIFEST_REL).read_bytes()
    if file_hash(V2 / MANIFEST_REL) != EXPECT["new_manifest_sha256"]:
        raise SystemExit("old packaged reserve manifest differs from expected")
    if sh(["git", "cat-file", "-t", EXPECT["new_commit"]], check=False).stdout.strip() != "commit":
        sh(["git", "fetch", "--no-tags", "origin", EXPECT["new_commit"]], timeout=300)
    if sh(["git", "cat-file", "-t", EXPECT["new_commit"]], check=False).stdout.strip() != "commit":
        raise SystemExit("new commit unavailable; nothing changed")
    public = pub_get("public/pool/reserve/MANIFEST.jsonl")
    if not append_only_public(live_manifest, public):
        raise SystemExit("public reserve is not an append-only extension of the packaged reserve; refusing apply")
    target = record_target(state)
    try:
        checkout(EXPECT["new_commit"])
        if head() != EXPECT["new_commit"] or digest() != EXPECT["new_digest"] or pinned_digest() != EXPECT["digest"]:
            raise SystemExit("new HEAD/digest/pinned digest mismatch")
        if file_hash(V2 / MANIFEST_REL) != EXPECT["new_manifest_sha256"]:
            raise SystemExit("new packaged reserve manifest mismatch")
        ok, why = pool_ok()
        if not ok:
            raise SystemExit(why)
        if params_hash(CUT / "params.json") != EXPECT["old_hash"] and not DRY:
            raise SystemExit("params file hash changed")
        PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = PATH_FILE.with_name(PATH_FILE.name + ".tmp")
        temporary.write_text(str(LIVE) + "\n")
        os.replace(temporary, PATH_FILE)
        if PATH_FILE.read_text() != str(LIVE) + "\n" or not LIVE.is_absolute():
            raise SystemExit("reserve-live.path invalid")
        # Reserve seeding is done before ingest, without starting the run loop.
        env = dict(ENV, SN120_RESERVE_DIR=str(LIVE))
        sh([PY, "-c", "from sn120_v2.envs.reserve import prepare_live_reserve; prepare_live_reserve(__import__('os').environ['SN120_RESERVE_DIR'])"], env=env, cwd=V2)
        sh([PY, "-m", "sn120_v2.envs.reserve", "verify"], env=env)
        ok, why = reserve_verify()
        if not ok:
            raise SystemExit(f"external reserve verification failed: {why}")
        log("APPLY DONE: HEAD/digest/params/pool/packaged reserve verified; external live path installed")
    except BaseException:
        checkout(target["commit"])
        PATH_FILE.unlink(missing_ok=True)
        raise


def ingest_ready():
    """Ingest only the oldest complete epoch; ingestd handles later ones after health PASS."""
    if not LIVE.is_dir():
        raise SystemExit("live reserve has not been seeded; run apply first")
    env = dict(ENV, SN120_RESERVE_DIR=str(LIVE))
    ledger_path = LIVE / "generator-ingested.jsonl"
    completed = {json.loads(line)["epoch"] for line in ledger_path.read_bytes().splitlines()} if ledger_path.exists() else set()
    for path in sorted(INBOX.glob("accepted-*.jsonl")):
        epoch = path.stem.removeprefix("accepted-")
        if not re.fullmatch(r"[0-9]{8}T[0-9]{4}Z", epoch):
            continue
        if epoch in completed:
            log(f"skip {epoch}: already in the reserve ingestion ledger")
            continue
        done = INBOX / f"accepted-{epoch}.done"
        results = INBOX / f"results-{epoch}.json"
        if not done.is_file() and not results.is_file():
            try:
                raw = pub_get(f"generator/epochs/{epoch}/results.json")
            except Exception as error:
                log(f"skip {epoch}: results.json not public yet ({type(error).__name__}); ingestd will retry")
                continue
            temporary = results.with_name(results.name + ".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, results)
        if not done.is_file() and not results.is_file():
            log(f"skip {epoch}: no .done/results.json yet; ingestd will retry")
            continue
        args = [PY, "-m", "sn120_v2.envs.reserve", "ingest-generator", path, "--epoch", epoch,
                "--prior-index", INDEX]
        if results.is_file():
            args += ["--results", results]
        result = sh(args, env=env, check=False, timeout=3600)
        if result.returncode:
            if result.returncode == 3:
                log(f"skip {epoch}: incomplete input or results not public yet; ingestd will retry")
                continue
            raise SystemExit(f"ingest {epoch} failed exit {result.returncode}: {result.stderr[-1000:]}")
        log(f"ingest {epoch}: {result.stdout[-1800:]}")
        # Backpressure at the deploy boundary: only the oldest complete epoch
        # goes into the first open. The daemon starts after open-check PASS.
        break
    log("INGEST READY DONE: oldest complete epoch ingested; later epochs left for ingestd")


def start_loop():
    if DRY:
        log("dry: start-loop suppressed")
        return
    env = dict(load_env())
    stopf = Path(env["SN120_WORK_DIR"]) / "logs/run-loop.stop-after-round"
    if stopf.exists():
        raise SystemExit("stop-after-round file present")
    if alive():
        raise SystemExit("loop already alive")
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    with open("/dev/null", "wb") as null:
        subprocess.Popen(["setsid", "nohup", "bash", LOOP, PY, "--state-dir", STATE, "--stack", STACK], cwd=V2, env=env, stdin=null, stdout=null, stderr=null, start_new_session=True)
    time.sleep(5)
    pidfile = stopf.parent / "run-loop.pid"
    if not pidfile.is_file() or not Path(f"/proc/{pidfile.read_text().strip()}").exists():
        raise SystemExit("run loop did not leave a live pidfile after start")
    log(f"start-loop live: pidfile {pidfile}; exactly one loop requested")


def start_ingestd():
    if DRY:
        log("dry: ingestd start suppressed")
        return
    if INGESTD_PID.exists() and Path(f"/proc/{INGESTD_PID.read_text().strip()}").exists():
        raise SystemExit("ingestd already running")
    logfile = INBOX.parent / "ingestd.log"
    command = f"exec setsid nohup bash {shlex.quote(str(V2 / 'systemd/generator-ingestd.sh'))} {shlex.quote(PY)} {shlex.quote(str(INDEX))} >> {shlex.quote(str(logfile))} 2>&1 < /dev/null & echo $! > {shlex.quote(str(INGESTD_PID))}"
    with open("/dev/null", "rb") as null:
        subprocess.Popen(["bash", "-c", command], cwd=V2, env=ENV, stdin=null, start_new_session=True).wait(timeout=15)
    if not INGESTD_PID.is_file():
        raise SystemExit("ingestd pidfile was not created")
    time.sleep(2)
    pid = INGESTD_PID.read_text().strip()
    if not pid.isdigit() or not Path(f"/proc/{pid}").exists():
        raise SystemExit("ingestd did not stay running")
    log(f"ingestd live: pidfile {INGESTD_PID}")


def stop_ingestd():
    if INGESTD_PID.exists():
        pid = INGESTD_PID.read_text().strip()
        if pid.isdigit():
            subprocess.run(["kill", "-TERM", pid], check=False)
        INGESTD_PID.unlink(missing_ok=True)
        log("ingestd stopped and pidfile removed")


def rotation(rid):
    for line in JOURNAL.read_text().splitlines():
        if '"round_opened"' in line and f'"{rid}"' in line:
            record = json.loads(line)
            return record["data"]["rotation"]
    return None


def open_check():
    rid = sys.argv[2]
    rot = rotation(rid)
    out = {"round": rid}
    if rot is None:
        out.update(ok=False, error="round_opened not found")
    else:
        admitted = rot.get("admitted", [])
        gen = [task for task in admitted if task.startswith("gen:")]
        out.update(ok=(rot.get("size") == 10000 and bool(gen)),
                   admitted=len(admitted), gen_admitted=len(gen), retired=len(rot.get("retired", [])),
                   size=rot.get("size"), reserve_manifest_sha=rot.get("reserve_manifest_sha"),
                   active_sha=rot.get("active_sha"), state_sha=rot.get("state_sha"),
                   params_hash=status().get("params_hash"))
        if out["params_hash"] != EXPECT["new_hash"]:
            out["ok"] = False
        if not gen:
            out["error"] = "no Generator task admitted"
    print(json.dumps(out, indent=2, sort_keys=True))


def journal_records():
    """Read the journal without taking its lock or writing any state."""
    rows = []
    for line in JOURNAL.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def journal_close_summary(rid):
    def round_id(row):
        data = row.get("data") or {}
        return data.get("round_id") or (data.get("round") or {}).get("round_id")

    rows = journal_records()
    opened = [row for row in rows if row.get("kind") == "round_opened" and round_id(row) == rid]
    settled = [row for row in rows if row.get("kind") == "round_settled" and round_id(row) == rid]
    closed = [row for row in rows if row.get("kind") == "round_closed" and round_id(row) == rid]
    failed = [row for row in rows if row.get("kind") == "lease_failed" and round_id(row) == rid]
    out = {"round": rid, "journal_records": len(rows),
           "opened_seq": opened[-1]["seq"] if opened else None,
           "settled_seq": settled[-1]["seq"] if settled else None,
           "closed_seq": closed[-1]["seq"] if closed else None}
    why = [f"no {kind}" for kind, records in (("round_opened", opened), ("round_settled", settled), ("round_closed", closed)) if not records]
    if closed:
        data = closed[-1]["data"]
        out.update(outcome=data.get("outcome"), close_at=time.strftime("%FT%TZ", time.gmtime(closed[-1].get("at", 0))),
                   snapshot_id=data.get("snapshot_id"), rotation_state_sha=data.get("rotation_state_sha"))
        if data.get("outcome") not in ("null", "applied"):
            why.append(f"outcome {data.get('outcome')!r}")
        out["opened_after_close"] = [round_id(row) for row in rows if row.get("kind") == "round_opened" and row.get("seq", -1) > closed[-1].get("seq", 0)]
    if settled:
        decision = (settled[-1]["data"] or {}).get("decision") or {}
        out.update(n_tasks=decision.get("n_tasks"), null_pass_rate=decision.get("null_pass_rate"),
                   winner_queue_pos=decision.get("winner_queue_pos"),
                   miner_cuts=[{"uid": row.get("uid"), "status": row.get("status"), "reason": (row.get("reason") or "")[:120]}
                               for row in decision.get("entries") or [] if row.get("status") == "no_decision"])
        if not decision.get("n_tasks"):
            why.append("decision n_tasks 0")
    if failed:
        why.append(f"lease_failed: {[(row['data'] or {}).get('reason', '')[:120] for row in failed]}")
    out.update(clean=not why, why=why)
    return out


def journal_close():
    summary = journal_close_summary(sys.argv[2])
    if "--next-open-ok" not in sys.argv and summary.get("opened_after_close"):
        summary["why"].append(f"round(s) opened after the close: {summary['opened_after_close']}")
        summary["clean"] = False
    print(json.dumps(summary, indent=2, sort_keys=True))


def public_check():
    rid = sys.argv[2]
    out = {"round": rid}
    why = []
    try:
        manifest = pub_get("public/pool/reserve/MANIFEST.jsonl")
        rows = [json.loads(line) for line in manifest.splitlines()]
        shards = [row for row in rows if "shard" in row]
        gen = [row for row in shards if "sn120-generator-" in row["shard"]]
        for row in gen:
            raw = pub_get(f"public/pool/reserve/shards/{row['shard']}.{row['sha256']}.jsonl")
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                why.append(f"shard {row['shard']} sha mismatch")
        out.update(manifest_sha256=hashlib.sha256(manifest).hexdigest(), manifest_shards=len(shards), generator_shards=len(gen))
        if not gen:
            why.append("no Generator shards mirrored")
        rotation = json.loads(pub_get(f"public/pool/rotation/{rid}.json"))
        if rotation.get("round_id") != rid:
            why.append(f"public rotation round {rotation.get('round_id')} != {rid}")
        admitted = rotation.get("admitted", [])
        if not any(task.startswith("gen:") for task in admitted):
            why.append("no Generator ids in public rotation")
        out.update(admitted=len(admitted), gen_admitted=sum(task.startswith("gen:") for task in admitted),
                   active_size=len(rotation.get("active_ids", [])), retired=len(rotation.get("retired", [])))
    except Exception as error:
        why.append(f"{type(error).__name__}: {error}")
    out.update(ok=not why, why=why)
    (HERE / f"public-{rid}.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


def rollback():
    if alive():
        raise SystemExit(f"loop alive: {alive()}")
    target = json.loads(TARGET.read_text())
    checkout(target["commit"])
    PATH_FILE.unlink(missing_ok=True)
    if head() != target["commit"] or digest() != target["code_digest"] or status().get("params_hash") != target["params_hash"] or pinned_digest() != target["pinned"]:
        raise SystemExit("rollback verification failed")
    log(f"ROLLBACK DONE -> {target['commit']}; reserve-live.path removed")


def facts():
    state = status()
    log(f"facts: HEAD {head()} code_digest {digest()} pinned {pinned_digest()} phase {state.get('phase')} params {state.get('params_hash')}")


if __name__ == "__main__":
    {"preflight": preflight, "stage": stage, "apply": apply, "ingest-ready": ingest_ready,
     "start-loop": start_loop, "start-ingestd": start_ingestd, "stop-ingestd": stop_ingestd,
     "journal-close": journal_close,
     "open-check": open_check, "public-check": public_check, "rollback": rollback, "facts": facts}[sys.argv[1]]()
