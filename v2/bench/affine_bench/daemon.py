"""Host side: the single queue-driven daemon of the per-green benchmark, plus its CLI commands.

Loop (every 30 s): read the applied kings from v2's journal (source.py; no call into the controller), build the
queue, keep exactly one king running on the node, collect finished benchmarks into BENCH_STATE/results/, write
BENCH_SITE/benchmarks.json. Idempotent and resumable: a result file per (king, benchmark, version) is the only "done"
marker, so a king is benchmarked once however often it is listed; a crashed run is resumed in place
(vf-eval --resume keeps finished rollouts); nothing is published per task.

Queue: every applied king that has a missing result, newest first. Nothing else is ever queued.
Informational only: this service reads kings and writes a table. It has no path to the head, the queue or chain weights.

Pod (affine_bench/pod.py): rented on demand when the queue is non-empty and no pod exists, released after IDLE_RELEASE_S with an
empty queue; a pod Lium reclaims is detected (node-gone) and replaced the same way. Nothing on the pod is deleted by
this service: the king's weights go away with the pod (results on the host are never deleted)."""
import fcntl, json, os, shlex, subprocess, time, traceback
from . import config as C, registry as R
from .node import Node
from .fetch import fetch as fetch_king
from . import pod as P, publish, source

ST = C.STATE; RES = ST + "/results"; PRIV = ST + "/private"
for d in (ST, RES, PRIV):
    os.makedirs(d, exist_ok=True)
os.chmod(PRIV, 0o700)
SF = ST + "/daemon.json"
MAX_FAIL = 3; STALL_S = 2400; POLL_S = 30
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
def model_rows():
    """One row per applied king (source.py is the only reader of v2 state). model_id is stable and file-safe."""
    return source.kings()


def result_path(mid, bid): return f"{RES}/{mid}/{bid}.json"


def missing(m, bs):
    out = []
    for b in bs:
        r = rj(result_path(m["model_id"], b["id"]))
        if not r or r.get("bench_version") != R.bench_version(b):
            out.append(b["id"])
    return out


def queue(rows=None):
    rows = model_rows() if rows is None else rows; bs = R.benchmarks(enabled_only=True, runnable_only=True)
    ok = lambda m: m["model_id"] not in S["unavailable"] and S["fail"].get(m["model_id"], 0) < MAX_FAIL and missing(m, bs)
    seen, out = set(), []
    for m in sorted(rows, key=lambda m: -(m.get("crowned_at") or 0)):
        if m["model_id"] not in seen and ok(m):
            seen.add(m["model_id"]); out.append(dict(m, why="new-king"))
    return out


# ------------------------------------------------------------------------------------------------ node ops
def model_dir(m):
    return f"{C.NODE_KINGS}/{m['king']}"


def have_weights(m):
    return "yes" in NODE.out(f"test -f {model_dir(m)}/.verified && echo yes")


def ensure_weights(m):
    if have_weights(m):
        return True
    r = fetch_king(m["king"], NODE); log(event="fetch", model=m["model_id"], status=r.get("status"), out=r.get("out", "")[-160:])
    return have_weights(m)


def run_key(m):
    return m["model_id"]


def launch(m, bids):
    rk = run_key(m)
    st = R.resolve_settings(m["model_id"], m.get("kind"))
    args = ["affine-bench", "node-run", "--model-id", rk, "--model-dir", model_dir(m), "--reasoning-parser", m.get("reasoning_parser", "qwen3"),
            "--benchmarks", ",".join(bids), "--max-model-len", str(st["max_model_len"])]
    ho = st.get("hf_overrides") or ""  # empty while the per-model profile clamps the window to the native context
    if ho: args += ["--hf-overrides", ho]
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
        r.update(model_id=m["model_id"], model_kind=m["kind"], node=dict(pod=C_POD.get("name"), lium_id=C_POD.get("id"), gpu=C_POD.get("gpu")),
                 collected_at=now(), serve_s=None)
        pt = NODE.out(f"cat {C.NODE_ROOT}/runs/{rk}/{bid}/per_task.jsonl 2>/dev/null", timeout=120)
        os.makedirs(f"{PRIV}/{rk}", exist_ok=True)
        with open(f"{PRIV}/{rk}/{bid}.per_task.jsonl", "w") as f: f.write(pt)
        os.chmod(f"{PRIV}/{rk}/{bid}.per_task.jsonl", 0o600)
        os.makedirs(f"{RES}/{m['model_id']}", exist_ok=True); wj(result_path(m["model_id"], bid), r)
        got[bid] = round(r["score"], 4) if r.get("score") is not None else None
    done = rj_node_done(rk)
    S["history"].append(dict(model=rk, started=run["started"], finished=now(), wall_s=now() - run["started"], node_wall_s=(done or {}).get("wall_seconds"),
                             serve_s=(done or {}).get("serve_seconds"), scores=got, gpu=C_POD.get("gpu"), usd_h=C_POD.get("cost"), pod=C_POD.get("huid")))
    S["history"] = S["history"][-400:]
    log(event="collected", model=rk, scores=got)
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
    rows = model_rows(); bs = R.benchmarks(enabled_only=True, runnable_only=True)
    run = S.get("running")
    if not run and not pod_ready(queue(rows)):
        S["queue"] = [m["model_id"] for m in queue(rows)]; publish.publish(rows, S); return
    if run and run.setdefault("pod_id", C_POD.get("id")) != C_POD.get("id"):
        # The run belonged to a pod that has since been replaced; its run dir went with that pod.
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
        elif st.get("alive") != "1":
            k = run["model"]["model_id"]; S["fail"][k] = S["fail"].get(k, 0) + 1
            log(event="run-died", model=run["run_key"], fails=S["fail"][k]); S["running"] = None; save()
        elif now() - run["last_progress"] > STALL_S:
            k = run["model"]["model_id"]; S["fail"][k] = S["fail"].get(k, 0) + 1
            log(event="stalled", model=run["run_key"]); abort(run); S["running"] = None; save()
    if not S.get("running"):
        q = queue(rows)
        for m in q:
            bids = missing(m, bs)
            if not ensure_weights(m):
                S["fail"][m["model_id"]] = S["fail"].get(m["model_id"], 0) + 1; save(); continue
            nxt = next((x for x in q if x["model_id"] != m["model_id"]), None)
            launch(m, bids)
            if nxt:
                try: ensure_weights(nxt); S["next"] = nxt["model_id"]; save()
                except Exception: log(event="prefetch-failed", model=nxt["model_id"])
            break
    maybe_release()
    S["queue"] = [m["model_id"] for m in queue(rows)]; publish.publish(rows, S)


def only_instance():
    """One daemon per state directory: one pod, one runner per king and one publisher. The lock dies with the process."""
    lock = open(ST + "/daemon.lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: raise SystemExit("another affine-bench daemon holds " + lock.name)
    return lock


def main(once=False):
    """daemon: loop for ever. once: return when every waiting king is benchmarked and the pod is released."""
    log(event="start", pid=os.getpid())
    while True:
        try:
            open(ST + "/heartbeat", "w").write(str(now()))
            step()
            if once and not S.get("running") and not S.get("queue") and not C_POD:
                return log(event="drained")
        except Exception:
            log(event="error", tb=traceback.format_exc()[-1500:])
        time.sleep(POLL_S)


def cli(cmd, argv):
    if cmd == "publish":
        publish.publish(model_rows(), S); print("published"); return
    if cmd == "status":
        q = queue(); r = S.get("running")
        print(json.dumps(dict(running=r and dict(model=r["run_key"], bids=r["bids"], started_cr=cr(r["started"]), progress=r.get("progress")),
                              queue_len=len(q), next=[(m["model_id"], m.get("why")) for m in q[:12]], fails=S["fail"], unavailable=S["unavailable"],
                              last=S["history"][-5:]), indent=1)); return
    if cmd in ("daemon", "once"):
        lock = only_instance()  # held until the process exits
        return main(once=cmd == "once")
    raise SystemExit("unknown command " + cmd)
