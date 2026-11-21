#!/usr/bin/env python3
"""One-link deploy 6c3d5b64 -> rotfix + reserve r2 + retired-reserve reload fix (one commit) on the SN120 v2 GPU box. Code-only: no set-params.
preflight | apply | rollback | start-loop | facts | round-check <rid> | open-check <rid> | public-check <rid>. Never prints secret values.
apply/rollback only with the loop stopped after a round (phase IDLE).
Live reserve MANIFEST.jsonl (tracked, rewritten at loop start by _restore_public_cache) is the only dirty path allowed.
apply: verifies the live MANIFEST is a strict shard-prefix of the new commit's packaged MANIFEST, saves it, checks out the new
commit (the packaged long MANIFEST replaces the live short one; the fix keeps it at loop start).
rollback: checks out the old commit and sets the live MANIFEST to whatever the public mirror holds now (short = nothing was
published; long = a round already rotated with the new shards and old code must see local == public or it refuses)."""
from __future__ import annotations
import hashlib, json, os, pathlib, re, shutil, subprocess, sys, time, urllib.request, urllib.error

ENVIRON = pathlib.Path("/root/sn120-run/secrets/loop-env-903093.environ")
CUT = pathlib.Path(os.environ.get("R2_CUT", "/root/sn120-run/cutover"))
V2 = os.environ.get("R2_V2", "/root/sn120/v2")  # dry runs only: a scratch clone
PY = "/root/sn120/v2/.venv/bin/python"
STATE = os.environ.get("R2_STATE", "/root/sn120-run/state")  # dry runs only: a journal copy
STACK = "sn120_v2.assembly:stack"
LOOP = "systemd/sn120-v2-run-loop.sh"
HERE = pathlib.Path(os.environ.get("R2_HERE", "/root/sn120-run/one-deploy"))
RUNLOG = pathlib.Path("/root/sn120-run/work/logs/run.log")
PHASES = pathlib.Path("/root/sn120-run/work/logs/phases.jsonl")
JOURNAL = pathlib.Path(STATE) / "journal.jsonl"
TARGET = HERE / "rollback-target.json"
EXPECT = json.loads((HERE / "expect.json").read_text())
# {old_commit, old_hash, digest, new_commit, new_hash(=old), new_digest(=digest), pool_sha, pool_n,
#  new_manifest_sha256 (file bytes), reserve_manifest_sha (loader), reserve_total, new_shards: [names], new_ids_n}
BUNDLE = HERE / "one.bundle"
MANIFEST_GIT = "v2/sn120_v2/envs/reserve/MANIFEST.jsonl"
MANIFEST_REL = "sn120_v2/envs/reserve/MANIFEST.jsonl"
WITHDRAWN_REL = "sn120_v2/envs/reserve/withdrawn.jsonl"
MANIFEST_PATH = pathlib.Path(V2) / MANIFEST_REL
MANIFEST_SAVE = HERE / "live-MANIFEST-before.jsonl"
UA = {"User-Agent": "sn120-onelink-watch/1"}

def log(m): print(time.strftime("%H:%M:%SZ", time.gmtime()), m, flush=True)

def _porcelain_path(line):
    rest = line[3:] if len(line) >= 4 else line
    if " -> " in rest:
        rest = rest.split(" -> ", 1)[1]
    if len(rest) >= 2 and rest[0] == rest[-1] == '"':
        rest = rest[1:-1].encode("utf-8").decode("unicode_escape")
    return rest

def dirty_except_manifest():
    raw = sh(["git", "status", "--porcelain", "--untracked-files=no"]).stdout
    other, manifest = [], []
    for line in raw.splitlines():
        if not line.strip():
            continue
        (manifest if _porcelain_path(line) == MANIFEST_GIT else other).append(line)
    return other, manifest

def save_live_manifest():
    if not MANIFEST_PATH.is_file():
        raise SystemExit("live reserve MANIFEST.jsonl is missing; refusing to checkout")
    data = MANIFEST_PATH.read_bytes()
    MANIFEST_SAVE.write_bytes(data)
    if MANIFEST_SAVE.read_bytes() != data:
        raise SystemExit("failed to save live MANIFEST.jsonl (bytes mismatch)")
    log(f"saved live MANIFEST.jsonl sha256={hashlib.sha256(data).hexdigest()} bytes={len(data)}")
    return data

def restore_live_manifest(data):
    MANIFEST_PATH.write_bytes(data)
    got = MANIFEST_PATH.read_bytes()
    if got != data:
        raise SystemExit("MANIFEST restore did not verify identical bytes")
    log(f"restored live MANIFEST.jsonl sha256={hashlib.sha256(got).hexdigest()} bytes={len(got)}")

def checkout_detach(commit, saved):
    """Checkout a commit without discarding live reserve MANIFEST bytes."""
    # Clean just this path so git cannot refuse the checkout or reset the file.
    if sh(["git", "status", "--porcelain", "--untracked-files=no", "--", MANIFEST_REL]).stdout.strip():
        sh(["git", "checkout", "-q", "--", MANIFEST_REL])
    sh(["git", "checkout", "-q", "--detach", commit])
    restore_live_manifest(saved)

def load_env():
    env = {}
    for item in ENVIRON.read_bytes().split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1); env[k.decode()] = v.decode()
    for k in ("APPLY", "C", "CAPS", "MB", "OLDPWD"): env.pop(k, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"; env["GIT_TERMINAL_PROMPT"] = "0"
    return env

ENV = load_env()

def sh(cmd, cwd=V2, check=True, timeout=600):
    p = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise SystemExit(f"FAIL {cmd[:5]} rc={p.returncode}: {p.stderr.strip().splitlines()[-3:]}")
    return p

def alive():
    if os.environ.get("R2_DRY") == "1": return []
    p = subprocess.run(["pgrep", "-af", r"sn120-v2-run-loo[p]|sn120_v2 ru[n]"], capture_output=True, text=True)
    return [l for l in p.stdout.splitlines() if "_operator" not in l]

def head(): return sh(["git", "rev-parse", "HEAD"]).stdout.strip()
def digest(): return json.loads(sh([PY, "-m", "sn120_v2.gpu.plan"]).stdout)["code_digest"]
def status(): return json.loads(sh([PY, "-m", "sn120_v2", "status", "--state-dir", STATE]).stdout)
def file_hash(path): return sh([PY, "-c", "import sys; from sn120_v2.params import load_params; print(load_params(sys.argv[1]).params_hash)", str(path)]).stdout.strip()
def pinned_digest():
    return sh([PY, "-c", "import sys; from pathlib import Path; from sn120_v2.journal import JOURNAL_FILE; from sn120_v2.state import load; print(load(Path(sys.argv[1]) / JOURNAL_FILE).params.recipe.code_digest)", STATE]).stdout.strip()
def pool_ok():
    code = sh([PY, "-c", "from sn120_v2.envs.math_pool import load_pool, POOL_SHA256; p=load_pool(); print(POOL_SHA256, len(p))"]).stdout.split()
    return code == [EXPECT["pool_sha"], str(EXPECT["pool_n"])], f"code {code}"
def reserve_verify():
    p = sh([PY, "-m", "sn120_v2.envs.reserve", "verify"], check=False)
    return p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-3:]

def facts(tag):
    s = status()
    log(f"{tag}: HEAD {head()} code_digest(box) {digest()} pinned {pinned_digest()} phase {s.get('phase')} params_hash {s.get('params_hash')}")
    return s

def preflight():
    a = alive()
    if a: raise SystemExit(f"loop/run alive: {a}")
    other, manifest = dirty_except_manifest()
    if other: raise SystemExit(f"checkout not clean: {chr(10).join(other)[:300]}")
    if manifest: log(f"ignoring live reserve manifest dirty state: {manifest}")
    s = facts("preflight")
    if s.get("phase") != "IDLE" and os.environ.get("R2_DRY") != "1": raise SystemExit(f"phase is {s.get('phase')}, not IDLE")
    return s

def set_params(path):
    p = sh([PY, "-m", "sn120_v2", "set-params", "--state-dir", STATE, "--stack", STACK, "--params", str(path)], check=False, timeout=900)
    sys.stderr.write(p.stderr); sys.stdout.write(p.stdout)
    return p.returncode


def pub_url(): return ENV["SN120_PUBLIC_URL"].rstrip("/")
def pub_get(key, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(pub_url() + "/" + key + "?r2=" + str(int(time.time())), headers=UA), timeout=timeout).read()

def shard_rows(raw):
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    return [r for r in rows if "shard" in r], [r for r in rows if "withdrawn" in r]

def is_prefix(short_raw, long_raw):
    s, sw = shard_rows(short_raw); l, lw = shard_rows(long_raw)
    return len(s) <= len(l) and l[:len(s)] == s and sw == lw

def record_target(s):
    t = {"commit": head(), "params_hash": s["params_hash"], "code_digest": digest(), "pinned": pinned_digest(),
         "live_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
         "recorded_at": time.strftime("%FT%TZ", time.gmtime())}
    shutil.copy2(CUT / "params.json", HERE / "params-before.json"); shutil.copy2(CUT / "plan.json", HERE / "plan-before.json")
    TARGET.write_text(json.dumps(t, indent=1)); log(f"rollback target recorded: code {t['commit'][:8]} params {t['params_hash'][:8]} digest {t['code_digest'][:8]}")
    return t

def new_manifest_bytes(commit):
    return subprocess.run(["git", "show", f"{commit}:{MANIFEST_GIT}"], cwd=V2, env=ENV, capture_output=True, check=True).stdout

def fetch_new_commit(nc):
    if sh(["git", "cat-file", "-t", nc], check=False).stdout.strip() == "commit": return
    if BUNDLE.exists():
        refs = [l.split()[1] for l in sh(["git", "bundle", "list-heads", str(BUNDLE)]).stdout.splitlines() if len(l.split()) == 2]
        f = sh(["git", "fetch", "--no-tags", str(BUNDLE)] + [f"{r}:refs/onelink/{i}" for i, r in enumerate(refs)], check=False, timeout=300)
        if f.returncode != 0: log(f"bundle fetch warning: {f.stderr.strip().splitlines()[-1:]}")
    if sh(["git", "cat-file", "-t", nc], check=False).stdout.strip() != "commit":
        f = sh(["git", "fetch", "--no-tags", "origin", nc], check=False, timeout=300)
        if f.returncode != 0: log(f"origin fetch warning: {f.stderr.strip().splitlines()[-1:]}")
    if sh(["git", "cat-file", "-t", nc], check=False).stdout.strip() != "commit":
        raise SystemExit(f"new commit {nc[:8]} not available after fetch; nothing changed")

def checkout_clean(commit):
    if sh(["git", "status", "--porcelain", "--untracked-files=no", "--", MANIFEST_REL]).stdout.strip():
        sh(["git", "checkout", "-q", "--", MANIFEST_REL])
    sh(["git", "checkout", "-q", "--detach", commit])

def reserve_status():
    p = sh([PY, "-m", "sn120_v2.envs.reserve", "status"], check=False)
    try: return json.loads(p.stdout)
    except ValueError: return {"error": (p.stdout + p.stderr)[-300:]}

def apply():
    s = preflight()
    if head() != EXPECT["old_commit"] or s["params_hash"] != EXPECT["old_hash"] or digest() != EXPECT["digest"]:
        raise SystemExit(f"not at the expected before state: HEAD {head()} hash {s.get('params_hash')} want {EXPECT['old_commit'][:8]} {EXPECT['old_hash'][:8]}")
    if EXPECT["new_hash"] != EXPECT["old_hash"] or EXPECT["new_digest"] != EXPECT["digest"]: raise SystemExit("expect.json is not a code-only change")
    nc = EXPECT["new_commit"]; fetch_new_commit(nc)
    live = MANIFEST_PATH.read_bytes(); MANIFEST_SAVE.write_bytes(live)
    if MANIFEST_SAVE.read_bytes() != live: raise SystemExit("could not save live MANIFEST")
    new = new_manifest_bytes(nc)
    if hashlib.sha256(new).hexdigest() != EXPECT["new_manifest_sha256"]: raise SystemExit("new commit MANIFEST bytes differ from expect.json")
    if not is_prefix(live, new): raise SystemExit("live MANIFEST is not a shard-prefix of the new packaged MANIFEST; nothing changed")
    try: public = pub_get("public/pool/reserve/MANIFEST.jsonl")
    except Exception as e: raise SystemExit(f"public MANIFEST not readable ({e}); nothing changed")
    if not is_prefix(public, new): raise SystemExit("public MANIFEST is not a shard-prefix of the new packaged MANIFEST; nothing changed")
    log(f"live MANIFEST sha256 {hashlib.sha256(live).hexdigest()[:16]} ({len(shard_rows(live)[0])} shards), public {len(shard_rows(public)[0])} shards, new {len(shard_rows(new)[0])} shards")
    t = record_target(s)
    def back(why):
        try: checkout_clean(t["commit"])
        finally: MANIFEST_PATH.write_bytes(live)
        TARGET.unlink(); raise SystemExit(f"verify failed ({why}); checked out old again with the saved live MANIFEST, nothing pinned")
    try: checkout_clean(nc)
    except SystemExit as e: back(f"checkout: {e}")
    if head() != nc: back("head")
    if MANIFEST_PATH.read_bytes() != new: back("MANIFEST after checkout != packaged")
    other, _ = dirty_except_manifest()
    if other or sh(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.strip(): back("tree dirty after checkout")
    if digest() != EXPECT["new_digest"]: back(f"code_digest {digest()} != expected {EXPECT['new_digest']}")
    ok, why = pool_ok()
    if not ok: back(why)
    if file_hash(CUT / "params.json") != EXPECT["old_hash"]: back("live params_hash differs under new code")
    if pinned_digest() != EXPECT["digest"]: back("pinned code_digest changed")
    rok, rwhy = reserve_verify()
    if not rok: back(f"reserve verify: {rwhy}")
    st = reserve_status()
    log(f"reserve status: { {k: st.get(k) for k in ('reserve_manifest_sha', 'shards', 'tasks', 'queued', 'withdrawn', 'error') if k in st} }")
    if st.get("tasks") != EXPECT["reserve_total"]: back(f"reserve tasks {st.get('tasks')} != {EXPECT['reserve_total']}")
    if st.get("reserve_manifest_sha") != EXPECT["reserve_manifest_sha"]: back(f"reserve_manifest_sha {st.get('reserve_manifest_sha')} != {EXPECT['reserve_manifest_sha']}")
    s = facts("after checkout (no set-params)")
    if s["params_hash"] != EXPECT["old_hash"]: back("status params_hash changed")
    log(f"pool {why}; reserve verify {rwhy}")
    log("APPLY DONE")

def rollback():
    a = alive()
    if a: raise SystemExit(f"loop/run alive: {a}")
    s = status()
    if s.get("phase") != "IDLE" and os.environ.get("R2_DRY") != "1": raise SystemExit(f"phase {s.get('phase')}")
    t = json.loads(TARGET.read_text())
    saved = MANIFEST_SAVE.read_bytes()
    try: public = pub_get("public/pool/reserve/MANIFEST.jsonl")
    except Exception as e: raise SystemExit(f"public MANIFEST not readable ({e}); rollback NOT done (old code needs local == public)")
    new = new_manifest_bytes(EXPECT["new_commit"])
    if public == saved or is_prefix(public, saved): live = saved; why = "public still the pre-deploy mirror -> saved live MANIFEST"
    elif is_prefix(saved, public) and is_prefix(public, new): live = public; why = "public already extended -> public MANIFEST bytes"
    else: raise SystemExit("public MANIFEST is neither the saved live nor a prefix of the new one; rollback NOT done")
    checkout_clean(t["commit"])
    MANIFEST_PATH.write_bytes(live)
    if MANIFEST_PATH.read_bytes() != live: raise SystemExit("MANIFEST write did not verify")
    log(f"live MANIFEST set: {why} ({len(shard_rows(live)[0])} shards, sha256 {hashlib.sha256(live).hexdigest()[:16]})")
    s = facts("after rollback")
    if head() != t["commit"] or digest() != t["code_digest"] or s["params_hash"] != t["params_hash"] or pinned_digest() != t["pinned"]:
        raise SystemExit("rollback incomplete")
    log(f"ROLLBACK DONE -> {t['commit'][:8]}")

def find_key(o, k):
    if isinstance(o, dict):
        if k in o and isinstance(o[k], dict): return o[k]
        for v in o.values():
            r = find_key(v, k)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = find_key(v, k)
            if r is not None: return r
    return None

def open_check():
    """Raw journal read (no lock): the ROUND_OPENED rotation block of rid uses the new reserve and admits new-shard ids."""
    rid = sys.argv[2]; out = {"round": rid}
    line = None
    for l in JOURNAL.read_text().splitlines():
        if '"round_opened"' in l and f'"{rid}"' in l: line = l
    if line is None: out.update(ok=False, final=False, error="round_opened not in journal"); print(json.dumps(out, indent=1)); return
    d = json.loads(line); rot = find_key(d, "rotation") or {}; r0 = find_key(d, "round") or {}
    ph = r0.get("params_hash")
    if ph is None:
        st = [d]
        while st:
            x = st.pop()
            if isinstance(x, dict):
                if isinstance(x.get("params_hash"), str): ph = x["params_hash"]
                st.extend(x.values())
            elif isinstance(x, list): st.extend(x)
    new_ids = set()
    for name in EXPECT["new_shards"]:
        p = pathlib.Path(V2) / "sn120_v2/envs/reserve/shards" / name
        new_ids |= {json.loads(l)["id"] for l in p.read_text().splitlines() if l.strip()}
    adm = rot.get("admitted") or []
    out.update(params_hash=ph, reserve_manifest_sha=rot.get("reserve_manifest_sha"), admitted=len(adm),
               admitted_new=sum(1 for i in adm if i in new_ids), size=rot.get("size"), new_ids_known=len(new_ids))
    why = []
    if ph != EXPECT["new_hash"]: why.append(f"params_hash {ph}")
    if rot.get("reserve_manifest_sha") != EXPECT["reserve_manifest_sha"]: why.append(f"reserve_manifest_sha {rot.get('reserve_manifest_sha')}")
    if rot.get("size") not in (None, 10000): why.append(f"size {rot.get('size')}")
    if not out["admitted_new"]: why.append("no admitted id from the new shards")
    out.update(ok=not why, final=True, why=why); print(json.dumps(out, indent=1))

def public_check():
    """Public mirror (what miners see): reserve MANIFEST == packaged bytes, every new shard object reachable, summary + rotation for rid."""
    rid = sys.argv[2]; out = {"round": rid}; why = []
    try:
        m = pub_get("public/pool/reserve/MANIFEST.jsonl")
        out["manifest_sha256"] = hashlib.sha256(m).hexdigest(); out["manifest_shards"] = len(shard_rows(m)[0])
        if out["manifest_sha256"] != EXPECT["new_manifest_sha256"]: why.append("public MANIFEST != packaged")
        rows = {r["shard"]: r for r in shard_rows(m)[0]}
        miss = []
        for name in EXPECT["new_shards"]:
            r = rows.get(name)
            if not r: miss.append(name); continue
            try:
                raw = pub_get(f"public/pool/reserve/shards/{name}.{r['sha256']}.jsonl")
                if hashlib.sha256(raw).hexdigest() != r["sha256"]: miss.append(name + ":sha")
            except Exception as e: miss.append(f"{name}:{str(e)[:40]}")
        out["new_shards_missing"] = miss
        if miss: why.append(f"{len(miss)} new shard objects not public")
        s = json.loads(pub_get("public/pool/summary.json"))
        out.update(summary_round=s.get("round_id"), reserve_size=s.get("reserve_size"), active_size=s.get("active_size"), admitted=s.get("admitted"))
        if s.get("round_id") != rid: why.append(f"summary round {s.get('round_id')}")
        # reserve_size may legitimately be 0 again: the first r2 round can admit the whole refill (offline: 941 of 941).
        if s.get("active_size") != 10000: why.append(f"summary active {s.get('active_size')}")
        pub_get(f"public/pool/rotation/{rid}.json")
    except urllib.error.HTTPError as e: why.append(f"HTTP {e.code} {e.url[:120]}")
    except Exception as e: why.append(f"{type(e).__name__}: {str(e)[:160]}")
    out.update(ok=not why, why=why); print(json.dumps(out, indent=1))

def start_loop():
    stopf = pathlib.Path(ENV["SN120_WORK_DIR"]) / "logs" / "run-loop.stop-after-round"
    if stopf.exists(): raise SystemExit("stop file present")
    if alive(): raise SystemExit("already alive")
    env = dict(ENV); env.pop("PYTHONDONTWRITEBYTECODE", None)
    with open("/dev/null", "wb") as n:
        subprocess.Popen(["setsid", "nohup", "bash", LOOP, PY, "--state-dir", STATE, "--stack", STACK], cwd=V2, env=env, stdin=n, stdout=n, stderr=n, start_new_session=True)
    time.sleep(5)
    log(f"loop pid {(pathlib.Path(ENV['SN120_WORK_DIR']) / 'logs' / 'run-loop.pid').read_text().strip()} alive: {alive()}")

def round_summary(rid):
    lines = RUNLOG.read_text(errors="replace").splitlines()
    idx = [i for i, l in enumerate(lines) if l.startswith(f"SN120 v2 round {rid}:")]
    if not idx: return None
    blk = lines[idx[-1]: idx[-1] + 20]
    opened = [i for i, l in enumerate(lines[: idx[-1]]) if f"round {rid} is open" in l]
    prev = opened[-1] if opened else max(0, idx[-1] - 50)
    m = next((re.search(r"tasks scored: (\d+) of (\d+)", l) for l in blk if "tasks scored:" in l), None)
    wt = next((l for l in blk if l.startswith("wall time:")), "")
    rows = [l for l in blk if re.match(r"^\s+\d+\s+\d+\s+5", l)]
    cut_notes = [l.strip()[:160] for l in rows if re.search(r"whole training time|training failed|not trained|not scored", l)]
    cut_notes += [l for l in blk if l.startswith(("not trained:", "not scored:"))]
    train = []
    for l in PHASES.read_text(errors="replace").splitlines():
        try: d = json.loads(l)
        except ValueError: continue
        if d.get("round_id") == rid and d.get("phase") == "train": train.append(d)
    per_tok = [d["wall_s"] / d["forward_tokens"] for d in train if d.get("forward_tokens")]
    return {"round": rid, "header": blk[0], "scored": int(m.group(1)) if m else None, "n_tasks": int(m.group(2)) if m else None,
            "wall_time": wt, "cuts": cut_notes, "train_walls": {d["queue_pos"]: round(d["wall_s"], 1) for d in train},
            "s_per_mtok": (sum(per_tok) / len(per_tok) * 1e6) if per_tok else None,
            "errors": [l[:200] for l in lines[prev + 1: idx[-1] + 20] if re.search(r"Too many open files|EMFILE|handle is closed|Traceback|InfraError|ConfigError|VerifyError", l)]}

def round_check():
    print(json.dumps(round_summary(sys.argv[2]), indent=1))



def journal_records():
    """Raw read of the journal file. Takes no lock and writes nothing (safe while the loop runs)."""
    out = []
    for l in JOURNAL.read_text(errors="replace").splitlines():
        try: out.append(json.loads(l))
        except ValueError: continue
    return out

def _rid_of(d):
    x = d.get("data") or {}
    return x.get("round_id") or (x.get("round") or {}).get("round_id")

def journal_close_summary(rid):
    """Is round rid closed cleanly, judged from the journal itself (not from log markers)?
    clean = round_settled(rid) and round_closed(rid) present, outcome null|applied, settled decision scored n_tasks > 0,
    no lease_failed for rid (infra failure), and no round opened after the close (the loop stopped before the next open).
    Miner-side cuts (an entry with status no_decision, e.g. a batch over the training time) are normal round outcomes:
    they are reported, not gating."""
    recs = journal_records(); why = []
    opened = [d for d in recs if d.get("kind") == "round_opened" and _rid_of(d) == rid]
    settled = [d for d in recs if d.get("kind") == "round_settled" and _rid_of(d) == rid]
    closed = [d for d in recs if d.get("kind") == "round_closed" and _rid_of(d) == rid]
    lf = [d for d in recs if d.get("kind") == "lease_failed" and _rid_of(d) == rid]
    out = {"round": rid, "opened_seq": opened[-1]["seq"] if opened else None, "settled_seq": settled[-1]["seq"] if settled else None,
           "closed_seq": closed[-1]["seq"] if closed else None, "journal_records": len(recs), "last_kind": recs[-1].get("kind") if recs else None}
    if not opened: why.append("no round_opened")
    if not settled: why.append("no round_settled")
    if not closed: why.append("no round_closed")
    if closed:
        c = closed[-1]["data"]; out.update(outcome=c.get("outcome"), close_at=time.strftime("%FT%TZ", time.gmtime(closed[-1].get("at", 0))),
                                           fronted=c.get("front"), snapshot_id=c.get("snapshot_id"), rotation_state_sha=c.get("rotation_state_sha"))
        if c.get("outcome") not in ("null", "applied"): why.append(f"outcome {c.get('outcome')!r}")
        later = [d for d in recs if d.get("kind") == "round_opened" and d.get("seq", -1) > closed[-1].get("seq", 0)]
        out["opened_after_close"] = [_rid_of(d) for d in later]
    if settled:
        dec = (settled[-1]["data"] or {}).get("decision") or {}
        ents = dec.get("entries") or []
        out.update(n_tasks=dec.get("n_tasks"), null_pass_rate=dec.get("null_pass_rate"), winner_queue_pos=dec.get("winner_queue_pos"),
                   miner_cuts=[{"uid": e.get("uid"), "status": e.get("status"), "reason": (e.get("reason") or "")[:120]} for e in ents if e.get("status") == "no_decision"])
        if not dec.get("n_tasks"): why.append("decision n_tasks 0")
    if lf: why.append(f"lease_failed: {[(d['data'] or {}).get('reason', '')[:120] for d in lf]}")
    out.update(clean=not why, why=why)
    return out

def journal_close():
    """journal-close <rid> [--next-open-ok]: by default the close must be the last round event (gate use, loop stopped)."""
    rid = sys.argv[2]; s = journal_close_summary(rid)
    if "--next-open-ok" not in sys.argv and s.get("opened_after_close"):
        s["why"].append(f"round(s) opened after the close: {s['opened_after_close']}"); s["clean"] = False
    print(json.dumps(s, indent=1))

def rollouts_check():
    """The public head rollouts of rid: reachable on the site, rows = scored head outcomes, outcomes equal, text present."""
    import gzip, io, urllib.request, urllib.error
    rid = sys.argv[2]; pub = ENV["SN120_PUBLIC_URL"].rstrip("/")
    out = {"round": rid}
    def get(path):
        req = urllib.request.Request(pub + "/" + path, headers={"User-Agent": "sn120-rollouts-watch/1"})
        return urllib.request.urlopen(req, timeout=120).read()
    try:
        rnd = json.loads(get(f"public/rounds/{rid}/round.json"))
        out["round_params_hash"] = rnd.get("params_hash")
        null = rnd.get("null_outcomes") or {}
        s0 = next(iter(null.values())) if null else []
        strings = [s0] if isinstance(s0, str) else list(s0)
        want = {(si, i): c for si, st in enumerate(strings) for i, c in enumerate(st) if c != "-"}
        out["want_rows"] = len(want)
        raw = get(f"public/rollouts/gate/{rid}/head.jsonl.gz")
        rows = [json.loads(l) for l in gzip.GzipFile(fileobj=io.BytesIO(raw)) if l.strip()]
        got = {(r["sample_index"], r["task_pos"]): r for r in rows}
        out.update(rows=len(rows), bytes=len(raw), roles=sorted({r.get("role") for r in rows}),
                   mismatched=sum(1 for k, c in want.items() if k not in got or got[k].get("outcome") != c),
                   extra=len(set(got) - set(want)),
                   text_rows=sum(1 for r in rows if r.get("completion_text")),
                   resolved_rows=sum(1 for r in rows if r.get("outcome") in ("P", "F", "C")),
                   sample_row={k: (str(v)[:80] if k in ("completion_text", "prompt") else v) for k, v in rows[0].items()} if rows else None)
        out["ok"] = bool(rows) and out["mismatched"] == 0 and out["extra"] == 0 and out["roles"] == ["head"] and out["text_rows"] >= 0.99 * out["resolved_rows"]
    except urllib.error.HTTPError as e:
        out.update(ok=False, error=f"HTTP {e.code} {e.url}")
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"preflight": preflight, "apply": apply, "rollback": rollback, "start-loop": start_loop, "facts": lambda: facts("facts"),
     "round-check": round_check, "open-check": open_check, "public-check": public_check,
     "journal-close": journal_close, "rollouts-check": rollouts_check}[sys.argv[1]]()
