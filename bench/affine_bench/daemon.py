"""Host side: the single queue-driven daemon (systemd user unit affine-bench.service on affinevalidator) + CLI helpers.

Loop (every 30 s): read the king list (affine.io/performance.json), build the queue, keep exactly one model running on the
node, collect finished benchmarks into state/bench-results/, publish affine.io/benchmarks.json. Idempotent and resumable:
a result file per (model, benchmark, version) is the only "done" marker; a crashed run is resumed in place
(vf-eval --resume keeps finished rollouts); nothing is published per task.

Queue (Jacob 2026-11-02 10:08 CR: bench only kings going forward, no backlog): manual requests (`affine-bench run`) first, then
every GREEN king with epoch >= MIN_EPOCH (544 = the first king after the last benched one, E540), newest first, then backfill
requests (`affine-bench run --backfill`, e.g. a new benchmark on older kings / Genesis / references; Jacob pt1159u arxivmath). Older kings,
Genesis and the reference models are never queued automatically (their existing results stay published).

Pod (affine_bench/pod.py): rented on demand when the queue is non-empty and no pod exists, released after IDLE_RELEASE_S with an
empty queue; a pod Lium reclaims is detected (node-gone) and replaced the same way. A king's weights are deleted from the pod
as soon as all of its results are collected (results on the host are never deleted)."""
import glob, json, math, os, re, shlex, subprocess, sys, time, traceback, urllib.request
from . import config as C, registry as R
from .node import Node
from .fetch import fetch as fetch_king, descriptor
from . import pod as P

ST = C.STATE; RES = ST + "/results"; PRIV = ST + "/private"; REQ = ST + "/requests"
for d in (ST, RES, PRIV, REQ):
    os.makedirs(d, exist_ok=True)
os.chmod(PRIV, 0o700)
SF = ST + "/daemon.json"
MAX_FAIL = 3; STALL_S = 2400
MIN_EPOCH = int(os.environ.get("BENCH_MIN_EPOCH", "544"))  # kings before this epoch are not benched (backlog dropped 2026-11-02)
NODE = Node()


def now(): return time.time()
def cr(t): return time.strftime("%Y-%m-%d %H:%M CR", time.gmtime(t - 6 * 3600)) if t else None
def rj(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
def wj(p, o, mode=0o644):
    t = p + ".tmp"; json.dump(o, open(t, "w"), indent=1); os.chmod(t, mode); os.replace(t, p)
def log(**k):
    k["at_cr"] = cr(now()); print(json.dumps(k), flush=True)


S = rj(SF, {}) or {}
S.setdefault("fail", {}); S.setdefault("unavailable", []); S.setdefault("history", [])
def save(): wj(SF, S)


# ------------------------------------------------------------------------------------------------ models
def kings():
    """King rows from performance.json (local copy on this host first; the public URL otherwise)."""
    d = rj(os.path.join(C.SITE, "performance.json"))
    if not d:
        with urllib.request.urlopen(C.PERFORMANCE_JSON, timeout=30) as r:
            d = json.load(r)
    return sorted(d.get("kings") or [], key=lambda k: (k.get("epoch") or 0))


def model_rows():
    """Every model the table knows: registry (genesis + refs) + every king. model_id is stable and file-safe."""
    ks = kings(); reg = R.models()
    gen = next((m for m in reg if m["kind"] == "genesis"), None)
    rows = []
    for m in reg:
        r = dict(model_id=m["id"], label=m["label"], kind=m["kind"], reasoning_parser=m.get("reasoning_parser", "qwen3"),
                 priority=m.get("priority", 999), max_model_len=m.get("max_model_len"), chat_template=m.get("chat_template"))
        if m.get("king"):
            r["king"] = m["king"]; k = next((k for k in ks if k["king"] == m["king"]), None)
            if k: r.update(epoch=k.get("epoch"), crowned_at=k.get("crowned_at"), era=k.get("era"))
        if m.get("hf"):
            r["hf"] = m["hf"]
        rows.append(r)
    for k in ks:
        if gen and k["king"] == gen.get("king"):
            continue
        rows.append(dict(model_id="king-" + k["king"][:16], king=k["king"], label="E%s %s" % (k.get("epoch"), k["king"][:8]), kind="king",
                         epoch=k.get("epoch"), crowned_at=k.get("crowned_at"), crowned_at_cr=k.get("crowned_at_cr"), era=k.get("era"),
                         green=k.get("era") == "gate", reasoning_parser=(gen or {}).get("reasoning_parser", "qwen3")))
    return rows


def result_path(mid, bid): return f"{RES}/{mid}/{bid}.json"


def missing(m, bs):
    out = []
    for b in bs:
        r = rj(result_path(m["model_id"], b["id"]))
        if not r or r.get("bench_version") != R.bench_version(b):
            out.append(b["id"])
    return out


def queue(rows=None):
    rows = rows or model_rows(); bs = R.benchmarks(enabled_only=True, runnable_only=True)
    ok = lambda m: m["model_id"] not in S["unavailable"] and S["fail"].get(m["model_id"], 0) < MAX_FAIL and missing(m, bs)
    q, back = [], []
    for f in sorted(glob.glob(REQ + "/*.json")):
        r = rj(f)
        if not r:
            continue
        if r.get("backfill"):  # `run --backfill`: after the new kings, and dropped once the model has failed MAX_FAIL times
            if r["model_id"] not in S["unavailable"] and S["fail"].get(r["model_id"], 0) < MAX_FAIL:
                back.append(dict(r, request=f))
        else:
            q.append(dict(r, request=f))
    kings_ = sorted((m for m in rows if m["kind"] == "king" and m.get("green") and (m.get("epoch") or 0) >= MIN_EPOCH),
                    key=lambda m: -(m.get("epoch") or 0))
    q += [dict(m, why="new-king") for m in kings_ if ok(m)] + back
    seen, out = set(), []
    for m in q:
        k = m["model_id"] + (m.get("tag") or "")
        if k not in seen:
            seen.add(k); out.append(m)
    return out


# ------------------------------------------------------------------------------------------------ node ops
def model_dir(m):
    if m.get("king"):
        return f"{C.NODE_KINGS}/{m['king']}"
    repo = m["hf"].split("@")[0]
    return f"{C.NODE_REFS}/{repo.replace('/', '__')}"


def have_weights(m):
    d = model_dir(m); mark = ".verified" if m.get("king") else ".revision"
    return "yes" in NODE.out(f"test -f {d}/{mark} && echo yes")


def ensure_weights(m):
    if have_weights(m):
        return True
    if m.get("king"):
        if descriptor(m["king"]) is None:
            return None
        r = fetch_king(m["king"], NODE); log(event="fetch", model=m["model_id"], status=r.get("status"), out=r.get("out", "")[-160:])
    else:
        repo, rev = m["hf"].split("@"); d = model_dir(m)
        p = NODE.run(f". {C.NODE_ROOT}/env.sh && {C.NODE_ROOT}/venv-vllm/bin/hf download {repo} --revision {rev} --local-dir {d} "
                     f"--max-workers 16 >> {C.NODE_ROOT}/logs/hfdl.log 2>&1 && echo {rev} > {d}/.revision", timeout=3600)
        log(event="hf-download", model=m["model_id"], rc=p.returncode)
    return have_weights(m)


def prune_king(m):
    """Delete one king's weights from the pod once all its results are in (fetch is a full checkpoint, so no later king
    needs them). Only {NODE_KINGS}/<64-hex ckpt>[.partial|.descriptor.json|.fetch.log]; never run dirs, never host results."""
    ck = m.get("king") or ""
    if not re.fullmatch(r"[0-9a-f]{64}", ck):
        return
    D = C.NODE_KINGS
    NODE.run(f"rm -rf {D}/{ck} {D}/{ck}.partial {D}/{ck}.descriptor.json {D}/{ck}.fetch.log", timeout=300)
    log(event="pruned", path=f"{D}/{ck}")


def run_key(m):
    return m["model_id"] + (("." + m["tag"]) if m.get("tag") else "")


def launch(m, bids):
    rk = run_key(m)
    args = ["affine-bench", "node-run", "--model-id", rk, "--model-dir", model_dir(m), "--reasoning-parser", m.get("reasoning_parser", "qwen3"),
            "--benchmarks", ",".join(bids)]
    if m.get("max_model_len"): args += ["--max-model-len", str(m["max_model_len"])]
    if m.get("chat_template"): args += ["--chat-template", f"{C.NODE_REPO}/bench/config/{m['chat_template']}"]
    if m.get("rollouts"): args += ["--rollouts", m["rollouts"]]
    # only the runner is backgrounded (all its fds redirected), so the ssh session returns at once
    cmd = (f"cd {C.NODE_ROOT} && . ./env.sh && rm -f runs/{rk}/DONE && mkdir -p runs/{rk} || exit 1; "
           f"nohup setsid {' '.join(shlex.quote(a) for a in args)} >> runs/{rk}/node-run.log 2>&1 < /dev/null & echo $!")
    try:
        out = NODE.run(cmd, timeout=90).stdout.strip()
    except subprocess.TimeoutExpired:
        out = ""  # the runner may still have started; liveness is read from the node's runner lock, not this pid
    S["running"] = dict(model=m, run_key=rk, bids=bids, started=now(), pid=out.split()[-1] if out else None, last_progress=now(), progress=0,
                        pod_id=C_POD.get("id"))
    save(); log(event="launched", model=rk, bids=bids, pid=S["running"]["pid"])


def node_status(rk, pid):
    o = NODE.out(f"cd {C.NODE_ROOT}/runs/{rk} 2>/dev/null && echo done=$(test -f DONE && echo 1); echo alive=$(flock -n {C.NODE_ROOT}/runs/.node-run.lock true 2>/dev/null || echo 1); "
                 f"echo lines=$(cat */vf/*/traces.jsonl 2>/dev/null | wc -l); echo ok=1", timeout=90)
    d = {}
    for line in o.splitlines():
        if "=" in line: a, b = line.split("=", 1); d[a] = b.strip()
    return d if d.get("ok") == "1" else None  # None = node unreachable or overloaded: decide nothing this round


def abort(run):
    """Exact PIDs only: the node-run process group (setsid) -> its vf-eval children. The model server is left up."""
    pid = run.get("pid")
    if pid and pid.isdigit():
        NODE.run(f"C=$(pgrep -P {pid}); kill -TERM {pid} $C 2>/dev/null; sleep 15; kill -KILL {pid} $C 2>/dev/null; true", timeout=60)
    log(event="aborted", model=run["run_key"], pid=pid)


def collect(run):
    m, rk = run["model"], run["run_key"]; got = {}
    for bid in run["bids"]:
        s = NODE.out(f"cat {C.NODE_ROOT}/runs/{rk}/{bid}/summary.json 2>/dev/null", timeout=60)
        if not s.strip():
            continue
        r = json.loads(s)
        r.update(model_id=m["model_id"], model_kind=m["kind"], tag=m.get("tag"), node=dict(pod=C_POD.get("name"), lium_id=C_POD.get("id"), gpu=C_POD.get("gpu")),
                 collected_at=now(), serve_s=None)
        pt = NODE.out(f"cat {C.NODE_ROOT}/runs/{rk}/{bid}/per_task.jsonl 2>/dev/null", timeout=120)
        os.makedirs(f"{PRIV}/{rk}", exist_ok=True)
        with open(f"{PRIV}/{rk}/{bid}.per_task.jsonl", "w") as f: f.write(pt)
        os.chmod(f"{PRIV}/{rk}/{bid}.per_task.jsonl", 0o600)
        if m.get("tag"):
            os.makedirs(f"{ST}/tagged/{rk}", exist_ok=True); wj(f"{ST}/tagged/{rk}/{bid}.json", r)
        else:
            os.makedirs(f"{RES}/{m['model_id']}", exist_ok=True); wj(result_path(m["model_id"], bid), r)
        got[bid] = round(r["score"], 4) if r.get("score") is not None else None
    done = rj_node_done(rk)
    S["history"].append(dict(model=rk, started=run["started"], finished=now(), wall_s=now() - run["started"], node_wall_s=(done or {}).get("wall_seconds"),
                             serve_s=(done or {}).get("serve_seconds"), scores=got, gpu=C_POD.get("gpu"), usd_h=C_POD.get("cost"), pod=C_POD.get("huid")))
    S["history"] = S["history"][-400:]
    log(event="collected", model=rk, scores=got)
    if m.get("request"):
        try: os.replace(m["request"], m["request"] + ".done")
        except FileNotFoundError: pass
    return got


def rj_node_done(rk):
    try: return json.loads(NODE.out(f"cat {C.NODE_ROOT}/runs/{rk}/DONE", timeout=30))
    except Exception: return None


C_POD = P.load()  # mutable: replaced in place when the pod is rented, released or reclaimed


POD_CHECK_AFTER_S = 600; POD_CHECK_EVERY_S = 600


def node_down(run):
    """Unreachable node: decide nothing about the run (an overloaded node is not a dead runner), but after 10 min ask Lium
    whether the pod still exists. A reclaimed pod never comes back (2026-11-02: noble-comet-40 was reclaimed at 06:00 CR and
    the old daemon polled it for 3.5 h), so drop it: the run is requeued without a failure and the next step rents a new pod."""
    d = S.setdefault("node_down", dict(since=now(), checked=0, state=None))
    if now() - d["since"] < POD_CHECK_AFTER_S or now() - d["checked"] < POD_CHECK_EVERY_S:
        log(event="node-unreachable", model=run["run_key"]); return
    d["checked"] = now(); st = P.state(C_POD.get("id")); d["state"] = st
    if st == "gone":
        log(event="node-gone", model=run["run_key"], pod=C_POD.get("huid"), lium_id=C_POD.get("id"), down_since_cr=cr(d["since"]))
        P.save(dict(gone=now(), last=dict(C_POD))); C_POD.clear(); S["running"] = None; S.pop("node_down", None)
    else:
        log(event="node-unreachable", model=run["run_key"], lium=st, down_min=round((now() - d["since"]) / 60))
    save()


def pod_ready(q):
    """Make sure a bootstrapped pod exists iff there is work. Returns True when the node can be used this step."""
    if C_POD and C_POD.get("bootstrapped") is False:  # a crash mid-bootstrap: start over on a fresh pod
        if P.release(log, why="not bootstrapped"): C_POD.clear()
        return False
    if C_POD:
        return True
    if not q:
        return False
    rec = P.rent(S, log); save()
    if rec:
        C_POD.clear(); C_POD.update(rec); S.pop("idle_since", None); save()
    return bool(rec)


def maybe_release():
    if not C_POD or S.get("running"):
        S.pop("idle_since", None); return
    S.setdefault("idle_since", now())
    if now() - S["idle_since"] >= P.IDLE_RELEASE_S:
        if P.release(log, why="queue empty for %d min" % round((now() - S["idle_since"]) / 60)):
            C_POD.clear(); S.pop("idle_since", None)
    save()


# ------------------------------------------------------------------------------------------------ main loop
def step():
    from . import publish
    rows = model_rows(); bs = R.benchmarks(enabled_only=True, runnable_only=True)
    run = S.get("running")
    if not run and not pod_ready(queue(rows)):
        S["queue"] = [m["model_id"] for m in queue(rows)]; publish.publish(rows, S); return
    if run and run.setdefault("pod_id", C_POD.get("id")) != C_POD.get("id"):
        # The run belonged to a pod that has since been replaced (deploy/pod.json); its run dir went with that pod.
        # Requeue it from scratch on the new pod; this is not counted as a failure of the model.
        log(event="pod-changed", model=run["run_key"], old_pod=run["pod_id"], new_pod=C_POD.get("id")); S["running"] = run = None
        S.pop("node_down", None); save()
    if run:
        st = node_status(run["run_key"], run.get("pid"))
        if st is None:
            node_down(run); publish.publish(rows, S); return
        if S.pop("node_down", None):
            log(event="node-back", model=run["run_key"]); save()
        lines = int(st.get("lines") or 0)
        if lines > run.get("progress", 0):
            run["progress"] = lines; run["last_progress"] = now(); save()
        if st.get("done") == "1":
            collect(run); S["running"] = None; save()
            if run["model"].get("king") and not run["model"].get("tag") and not missing(run["model"], bs):
                prune_king(run["model"])
        elif st.get("alive") != "1":
            k = run["model"]["model_id"]; S["fail"][k] = S["fail"].get(k, 0) + 1
            log(event="run-died", model=run["run_key"], fails=S["fail"][k]); S["running"] = None; save()
        elif now() - run["last_progress"] > STALL_S:
            k = run["model"]["model_id"]; S["fail"][k] = S["fail"].get(k, 0) + 1
            log(event="stalled", model=run["run_key"]); abort(run); S["running"] = None; save()
    if not S.get("running"):
        q = queue(rows)
        for m in q:
            bids = m.get("benchmarks") or missing(m, bs)
            if not bids:
                continue
            ok = ensure_weights(m)
            if ok is None:
                S["unavailable"].append(m["model_id"]); save(); log(event="unavailable", model=m["model_id"]); continue
            if not ok:
                S["fail"][m["model_id"]] = S["fail"].get(m["model_id"], 0) + 1; save(); continue
            nxt = next((x for x in q if x["model_id"] != m["model_id"]), None)
            launch(m, bids)
            if nxt:
                try: ensure_weights(nxt); S["next"] = nxt["model_id"]; save()
                except Exception: log(event="prefetch-failed", model=nxt["model_id"])
            break
    maybe_release()
    S["queue"] = [m["model_id"] for m in queue(rows)]; publish.publish(rows, S)


def main():
    log(event="start", pid=os.getpid())
    while True:
        try:
            open(ST + "/heartbeat", "w").write(str(now()))
            step()
        except Exception:
            log(event="error", tb=traceback.format_exc()[-1500:])
        time.sleep(30)


def cli(cmd, argv):
    import argparse
    if cmd == "daemon":
        return main()
    if cmd == "publish":
        from . import publish; publish.publish(model_rows(), S); print("published"); return
    if cmd == "status":
        q = queue(); r = S.get("running")
        print(json.dumps(dict(running=r and dict(model=r["run_key"], bids=r["bids"], started_cr=cr(r["started"]), progress=r.get("progress")),
                              queue_len=len(q), next=[(m["model_id"], m.get("why")) for m in q[:12]], fails=S["fail"], unavailable=S["unavailable"],
                              last=S["history"][-5:]), indent=1)); return
    if cmd == "run":
        ap = argparse.ArgumentParser(prog="affine-bench run")
        ap.add_argument("--model", required=True, help="king:<hash prefix> | registry id (genesis, qwen3-8b, ...)")
        ap.add_argument("--benchmarks", required=True); ap.add_argument("--rollouts", default="", help="e.g. math500=16")
        ap.add_argument("--tag", default="", help="non-empty = side run kept under state/bench-results/tagged/, never published")
        ap.add_argument("--backfill", action="store_true", help="queue after the new kings (e.g. a new benchmark on old models)")
        a = ap.parse_args(argv); rows = model_rows()
        if a.model.startswith("king:"):
            m = next(x for x in rows if x.get("king", "").startswith(a.model[5:]))
        else:
            m = next(x for x in rows if x["model_id"] == a.model)
        m = dict(m, benchmarks=a.benchmarks.split(","), rollouts=a.rollouts, tag=a.tag or None, why="backfill" if a.backfill else "request",
                 backfill=a.backfill)
        p = f"{REQ}/{int(now())}-{run_key(m)}.json"; wj(p, m); print("queued", p); return
    raise SystemExit("unknown command " + cmd)
