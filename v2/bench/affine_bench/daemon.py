"""Host side: the service of the per-row benchmark, the worker of one row, and the CLI commands.

The service (one process, `daemon`), every 30 s: read the rows from v2's public files (source.py: the launch model,
every folded king, and the last king of v1; no journal and no call into the controller), build the queue, start a
worker for each row that has work, and write the table: BENCH_SITE/benchmarks.json and, when a row or a result
changed, the same table to the store. Only the service writes the table. Nothing is published per task.

The worker of a row (one process for each row, `row <model_id>`, class Row): rent the row's pod, set it up, download
the row's weights, run the benchmarks, collect them into BENCH_STATE/results/, and give the pod back when the table in
the store has the row's results. Rows run side by side (Jacob, 2026-11-14). A worker waits for its own rent, setup and
download only: no worker holds a lock, a pod or a file of another row. A row that fails keeps no pod.

Done marker: a (row, benchmark, version) is done when the published table has its result, or a result file is on this
host. The service reads the published table before its first step and starts no worker until it has. So a row is
benchmarked once however often it is listed, and a host with an empty disk does not run a king again. A crashed run is
resumed in place (vf-eval --resume keeps finished rollouts).

Queue: every row that has a missing result, newest first; the launch model (genesis) and the v1 king are the oldest
rows. A row whose weight files are not in the store yet waits outside the queue: no fail, no rent, no worker.
At most P.MAX_PODS rows are in work at a time; the next row starts when one of them is done.
Informational only: this service reads kings and writes a table. It has no path to the head, the queue or chain weights.

The first row: the row that ran on the one pod of the single-node service (before 2026-11-14) keeps that pod, its
record (BENCH_STATE/pod.json) and its ssh wrapper (BENCH_NODE_SSH). No new commit is put on that pod, and no run is
ever started or killed on it by this code: the run goes on with the settings of that pod's checkout
(FIRST_ROW_PROFILE), and each of its results says so. When that run is over, dead or without progress and a result is
missing, the row is `held`: what the node has is brought to the host, the pod goes back, and a new run is the
operator's word (`retry`). A result that it makes with more samples a problem than the settings now ask for is brought to
the settings' count by the fixed rule of derive.py.

Pod (affine_bench/pod.py): one for each row. A pod that Lium reclaims is detected (node-gone) and the row rents again.
Nothing on a pod is deleted by this service: the weights go away with the pod (results on the host are never deleted)."""
import fcntl, json, os, shlex, subprocess, sys, time, traceback
from sn120_v2.errors import InfraError
from . import config as C, derive, registry as R
from .node import Node
from .fetch import fetch as fetch_weights
from . import pod as P, publish, source

ST = C.STATE; RES = ST + "/results"; PRIV = ST + "/private"; ROWS = C.ROWS
for d in (ST, RES, PRIV, ROWS):
    os.makedirs(d, exist_ok=True)
os.chmod(PRIV, 0o700)
SF = ST + "/daemon.json"
MAX_FAIL = 3; STALL_S = 2400; POLL_S = 30
WAIT_LATE_S = 3600  # weight files still not in the store this long after the head was applied: the table says `waiting`
BEAT_S = 300  # while a row runs, the table gets the time and the counts of the worker's look at the node this often
CONFIRM_WAIT_S = 600  # a row that failed waits this long for the table in the store before its pod is given back
RELEASE_RETRY_S = 600  # a release that Lium or a guard refused is tried again this often
ALARM_TRIES = 3  # a release that was refused this many times is an alarm in the table
LIUM_CHECK_S = 600  # the service reads Lium's listing this often for the alarms of the table (reads only)
KEEP_TRIES = 3  # the output of a first row's run that is not whole is asked from the node this many times before its pod goes back
POD_CHECK_AFTER_S = 600; POD_CHECK_EVERY_S = 600
FIRST_ROW_PROFILE = "qwen3.5-family"  # the node of the single-node service has a checkout of before `king_profile` changed: its run is with this profile


def now(): return time.time()
def cr(t): return time.strftime("%Y-%m-%d %H:%M CR", time.gmtime(t - 6 * 3600)) if t else None
def rj(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
def wj(p, o, mode=0o644):
    t = p + ".tmp"; json.dump(o, open(t, "w"), indent=1); os.chmod(t, mode); os.replace(t, p)
def log(**k):
    k["at_cr"] = cr(now()); print(P.mask(json.dumps(k)), flush=True)  # the journal gets no address, whatever text an error carries


S = rj(SF, {}) or {}  # the state of the service. `running`, `fail` and `history` in it are of the single-node service
S.setdefault("fail", {}); S.setdefault("unavailable", []); S.setdefault("history", [])
def save(): wj(SF, S)


# ------------------------------------------------------------------------------------------------ rows
def model_rows():
    """The genesis row, one row per folded king, and the v1 king (source.py is the only reader of v2 state). model_id is stable and file-safe."""
    return source.models() + source.extra_rows()


def result_path(mid, bid): return f"{RES}/{mid}/{bid}.json"
def row_state(mid): return rj(f"{ROWS}/{mid}/daemon.json", {}) or {}
def is_first(mid): return os.path.exists(f"{ROWS}/{mid}/first")
def pod_of(mid): return P.Pod(mid, is_first(mid))
def row_ids(): return sorted(d for d in os.listdir(ROWS) if os.path.isdir(f"{ROWS}/{d}"))


def stopped(mid, s=None):
    """Why the service starts no work for a row until the operator says `retry`, or None."""
    s = row_state(mid) if s is None else s
    if s.get("fails", 0) >= MAX_FAIL: return "failed"
    return next((k for k in ("held", "rent_failed", "expired") if s.get(k)), None)


def published():
    """(model, benchmark, version) of every result in the table: the local copy of what the store holds, plus this host's results."""
    return {publish.row_key(r) for r in (rj(publish.OUT) or {}).get("results") or []}


def missing(m, bs, done=None):
    done = published() if done is None else done
    out = []
    for b in bs:
        v = R.bench_version(b); r = rj(result_path(m["model_id"], b["id"]))
        if (m["model_id"], b["id"], v) not in done and not (r and r.get("bench_version") == v):
            out.append(b["id"])
    return out


def candidates(rows):
    """Rows with a missing result that are not stopped, newest first."""
    bs = R.benchmarks(enabled_only=True, runnable_only=True); done = published()
    seen, out = set(), []
    for m in sorted(rows, key=lambda m: -(m.get("crowned_at") or 0)):
        if m["model_id"] not in seen and not stopped(m["model_id"]) and missing(m, bs, done):
            seen.add(m["model_id"]); out.append(dict(m, why="new-" + m["kind"]))
    return out


def waiting(rows, tell=lambda **k: None):
    """Rows that cannot start: model_id -> dict(file=<a weight file that the store does not hold>, late=<bool>).

    Such a row waits. It gets no fail, no worker and no pod. Each step asks the store again, so the row is queued by the
    step that finds every file. `late` says that the head was applied more than WAIT_LATE_S ago: the table then shows the
    row as `waiting`. A row in work is not asked about: its weights are on its node. A store that fails to answer holds
    no king back: the download on the node decides then, and a failure there (a refused link, a wrong sha256) is a failed
    try as before. A row with a source of its own (the v1 king) waits also then: it starts only when the source was read."""
    out = {}
    for m in candidates(rows):
        if row_state(m["model_id"]).get("running") or pod_of(m["model_id"]).load():
            continue
        try:
            name = source.absent(m)
        except InfraError as e:
            tell(event="store-check-failed", model=m["model_id"], error=str(e)[-200:])
            if m.get("source") is None:
                continue
            name = "the source could not be read"
        if name:
            if m.get("source") is not None:  # it does not start by itself: the operator sets the source
                out[m["model_id"]] = dict(file=name, late=True, what="source")
            else:
                out[m["model_id"]] = dict(file=name, late=now() - (m.get("crowned_at") or 0) > WAIT_LATE_S)
    return out


def note_waiting(rows):
    """waiting(), with one log line when a row starts to wait, when it is late, and when its files are there."""
    was = S.get("waiting") or {}; wait = waiting(rows, log)
    for mid, w in wait.items():
        if mid not in was: log(event="waiting", model=mid, file=w["file"])
        if w["late"] and not w.get("what") and not was.get(mid, {}).get("late"): log(event="waiting-late", model=mid, file=w["file"], minutes=WAIT_LATE_S // 60)
    for mid in was:
        if mid not in wait: log(event="waiting-over", model=mid)
    if wait != was:
        S["waiting"] = wait; save()
    return wait


def queue(rows=None, wait=None):
    rows = model_rows() if rows is None else rows; wait = waiting(rows) if wait is None else wait
    return [m for m in candidates(rows) if m["model_id"] not in wait]


# ------------------------------------------------------------------------------------------------ the worker of one row
def node_of(row):
    return Node(row.pod.ssh)


class Row:
    """One row in work: the state of its worker (BENCH_STATE/rows/<model_id>/daemon.json), its pod and its node."""

    def __init__(self, mid):
        self.mid = mid; self.dir = f"{ROWS}/{mid}"; os.makedirs(self.dir, exist_ok=True)
        self.sf = self.dir + "/daemon.json"
        self.S = row_state(mid); self.S.setdefault("fails", 0); self.S.setdefault("history", [])
        self.attach()

    def attach(self):
        self.first = is_first(self.mid); self.pod = P.Pod(self.mid, self.first); self.rec = self.pod.load(); self.node = node_of(self)

    def save(self): wj(self.sf, self.S)
    def log(self, **k): log(row=self.mid, **k)

    def tried(self, what, f, *a):
        """Call f; a fault of the node or of a file is one log line and None. For the steps that must not stop what comes after them."""
        try:
            return f(*a)
        except Exception as e:
            self.log(event="error", during=what, error=type(e).__name__ + ": " + P.mask(str(e))[-200:])

    def tell(self, state):
        """The row's state for the table: renting, setup, running, released, rent-failed, held, release-refused."""
        if (self.S.get("phase") or {}).get("state") != state:
            self.S["phase"] = dict(state=state, since=now()); self.save()

    # ---- node ops
    def model_dir(self, m): return f"{C.NODE_KINGS}/{m['base_hash']}"
    def have_weights(self, m): return "yes" in self.node.out(f"test -f {self.model_dir(m)}/.verified && echo yes")

    def ensure_weights(self, m):
        if self.have_weights(m):
            return True
        r = fetch_weights(m, self.node); self.log(event="fetch", status=r.get("status"), out=r.get("out", "")[-160:])
        return self.have_weights(m)

    def launch(self, m, bids):
        rk = self.mid
        st = R.resolve_settings(m["model_id"], m.get("kind"))
        args = ["affine-bench", "node-run", "--model-id", rk, "--model-dir", self.model_dir(m), "--reasoning-parser", m.get("reasoning_parser", "qwen3"),
                "--benchmarks", ",".join(bids), "--max-model-len", str(st["max_model_len"])]
        ho = st.get("hf_overrides") or ""  # empty while the per-model profile clamps the window to the native context
        if ho: args += ["--hf-overrides", ho]
        # only the runner is backgrounded (all its fds redirected), so the ssh session returns at once
        cmd = (f"cd {C.NODE_ROOT} && . ./env.sh && rm -f runs/{rk}/DONE && mkdir -p runs/{rk} || exit 1; "
               f"nohup setsid {' '.join(shlex.quote(a) for a in args)} >> runs/{rk}/node-run.log 2>&1 < /dev/null & echo $!")
        try:
            out = self.node.run(cmd, timeout=90).stdout.strip()
        except subprocess.TimeoutExpired:
            out = ""  # the runner may still have started; liveness is read from the node's runner lock, not this pid
        self.S["running"] = dict(model=m, run_key=rk, bids=bids, started=now(), pid=out.split()[-1] if out else None, last_progress=now(), progress=0,
                                 pod_id=self.rec.get("id"))
        self.tell("running"); self.save(); self.log(event="launched", bids=bids, pid=self.S["running"]["pid"])

    def node_status(self, rk):
        o = self.node.out(f"cd {C.NODE_ROOT}/runs/{rk} 2>/dev/null && echo done=$(test -f DONE && echo 1); echo alive=$(flock -n {C.NODE_ROOT}/runs/.node-run.lock true 2>/dev/null || echo 1); "
                          f"echo lines=$(cat */vf/*/traces.jsonl 2>/dev/null | wc -l); echo finished=$(ls */summary.json 2>/dev/null | wc -l); echo ok=1", timeout=90)
        d = {}
        for line in o.splitlines():
            if "=" in line: a, b = line.split("=", 1); d[a] = b.strip()
        return d if d.get("ok") == "1" else None  # None = node unreachable or overloaded: decide nothing this round

    def abort(self, run):
        """Exact PIDs only: the node-run process group (setsid) -> its vf-eval children. The model server is left up."""
        pid = run.get("pid")
        if pid and str(pid).isdigit():
            self.tried("abort", self.node.run, f"C=$(pgrep -P {pid}); kill -TERM {pid} $C 2>/dev/null; sleep 15; kill -KILL {pid} $C 2>/dev/null; true", None, 60)
        self.log(event="aborted", pid=pid)

    def collect(self, run, whole=True):
        """Bring the finished benchmarks of a run to the host. `whole`: the runner said that it is done, so a benchmark
        with no result is a failed try. A result is stored only with its per-problem lines."""
        m, rk, rec = run["model"], run["run_key"], self.rec; got = {}; no_value = {}
        for bid in run["bids"]:
            s = self.node.out(f"cat {C.NODE_ROOT}/runs/{rk}/{bid}/summary.json 2>/dev/null", timeout=60)
            if not s.strip():
                continue
            pt = self.node.out(f"cat {C.NODE_ROOT}/runs/{rk}/{bid}/per_task.jsonl 2>/dev/null", timeout=120)
            if not pt.strip():
                self.log(event="collect-incomplete", bench=bid, why="the per-problem lines could not be read"); continue
            try:  # a result file that is cut, or that lacks a field, is not stored: it never stops the other results or the table
                r = json.loads(s)
                r.update(model_id=m["model_id"], model_kind=m["kind"], node=dict(pod=rec.get("name"), lium_id=rec.get("id"), gpu=rec.get("gpu")),
                         collected_at=now(), serve_s=None)
                os.makedirs(f"{PRIV}/{rk}", exist_ok=True)
                with open(f"{PRIV}/{rk}/{bid}.per_task.jsonl", "w") as f: f.write(pt)
                os.chmod(f"{PRIV}/{rk}/{bid}.per_task.jsonl", 0o600)
                b = R.benchmark(bid); want = R.bench_version(b)
                if r.get("bench_version") != want and int(r.get("rollouts") or 0) > int(b["rollouts"]):
                    # a run with more samples a problem than the settings ask for now (the first row): its value at the settings' count
                    wj(f"{PRIV}/{rk}/{bid}.full.json", r, 0o600)
                    r, why = derive.from_full(r, pt, int(b["rollouts"]), want, b.get("problems"))
                    if r is None:
                        no_value[bid] = why; self.log(event="no-value", bench=bid, why=why); continue
                    self.log(event="derived", bench=bid, rule=derive.RULE, score=round(r["score"], 4), score_full=r["score_full"], samples_full=r["samples_full"])
                publish.row_of(r, b)  # raises when the table could not show this result
            except Exception as e:
                self.log(event="collect-incomplete", bench=bid, why=f"the result of the node cannot be read ({type(e).__name__}: {str(e)[:80]})"); continue
            os.makedirs(f"{RES}/{m['model_id']}", exist_ok=True); wj(result_path(m["model_id"], bid), r)
            if r.get("bench_version") != want:  # the node's checkout is of another benchmark version: kept, and not what this row misses
                self.log(event="version-differs", bench=bid, got=r.get("bench_version"), want=want); continue
            got[bid] = round(r["score"], 4) if r.get("score") is not None else None
        done = self.node_done(rk)
        self.S["history"].append(dict(model=rk, started=run["started"], finished=now(), wall_s=now() - run["started"], node_wall_s=(done or {}).get("wall_seconds"),
                                      serve_s=(done or {}).get("serve_seconds"), scores=got, gpu=rec.get("gpu"), usd_h=rec.get("cost"), pod=rec.get("huid")))
        self.S["history"] = self.S["history"][-400:]
        self.log(event="collected", scores=got)
        lost = [b for b in run["bids"] if b not in got and b not in no_value]
        if no_value:  # the run is over and gave no value at the settings' count: a new run needs the operator's word
            self.S["held"] = dict(at=now(), why="; ".join(f"{b}: {w}" for b, w in sorted(no_value.items())))
        elif whole and lost:
            self.S["fails"] += 1; self.log(event="run-incomplete", benchmarks=lost, fails=self.S["fails"])
        return got

    def node_done(self, rk):
        try: return json.loads(self.node.out(f"cat {C.NODE_ROOT}/runs/{rk}/DONE", timeout=30))
        except Exception: return None

    def node_down(self, run):
        """Unreachable node: decide nothing about the run (an overloaded node is not a dead runner), but after 10 min ask Lium
        whether the pod still exists. A reclaimed pod never comes back (2026-11-02: noble-comet-40 was reclaimed at 06:00 CR and
        the old daemon polled it for 3.5 h), so drop it: the run is requeued without a failure and the row rents a pod of its own."""
        d = self.S["node_down"] = self.S.get("node_down") or dict(since=now(), checked=0, state=None)
        if now() - d["since"] < POD_CHECK_AFTER_S or now() - d["checked"] < POD_CHECK_EVERY_S:
            self.log(event="node-unreachable"); self.save(); return
        d["checked"] = now(); st = P.state(self.rec.get("id")); d["state"] = st
        if st == "gone":
            self.log(event="node-gone", pod=self.rec.get("huid"), lium_id=self.rec.get("id"), down_since_cr=cr(d["since"]))
            self.pod.save(dict(gone=now(), last=dict(self.rec))); self.S["running"] = None; self.S.pop("node_down", None)
            self.not_first()
        else:
            self.log(event="node-unreachable", lium=st, down_min=round((now() - d["since"]) / 60))
        self.save()

    def not_first(self):
        """The pod of the single-node service is gone or given back: from now on this is a row as any other."""
        if self.first:
            os.remove(self.dir + "/first")
        self.attach()

    # ---- pod
    def pod_ready(self):
        """True when the row's pod is set up. A rent blocks this worker for some minutes, and this worker only."""
        if self.rec and (self.rec.get("bootstrapped") or self.first):
            return True
        if self.first:  # the record of the single-node service's pod cannot be read: no rent beside that pod, the operator decides
            self.S["held"] = dict(at=now(), why="the record of the pod of the single-node service cannot be read: no rent, and no run")
            self.log(event="first-row-held", why=self.S["held"]["why"]); self.tell("held"); self.save(); return False
        self.tell("renting")
        rec = self.pod.rent(self.S, self.log, self.tell); self.save()
        if rec:
            self.rec = rec
        elif self.S.get("rent_failed"):
            self.tell("rent-failed")
        elif self.S.get("held"):
            self.tell("held")
        else:
            self.rec = self.pod.load()
        return bool(rec)

    def results_left(self):
        """The results of this row on the host that the table in the store does not have yet."""
        there = {publish.row_key(r) for r in source.table_results()}; out = []
        for b in R.benchmarks(enabled_only=True, runnable_only=True):
            r = rj(result_path(self.mid, b["id"]))
            if r and (self.mid, b["id"], r.get("bench_version") or "") not in there:
                out.append(b["id"])
        return out

    def give_back(self, m, bs):
        """Nothing more to run on the row's pod. The pod is given back when the table in the store has each result that
        was collected. A row that failed waits CONFIRM_WAIT_S for that at most. True: the worker has ended its work."""
        S = self.S
        if not self.rec:
            return True
        if now() - S.get("release_tried", 0) < RELEASE_RETRY_S:
            return False
        whole = not missing(m, bs) or bool(S.get("held"))  # held: the run is over, and what it gave is all that this pod gives
        S.setdefault("idle_since", now())
        try:
            left = self.results_left()
        except Exception as e:  # whatever the store or the table gives: the wait below has an end, and the delete is sent
            left = ["?"]; self.log(event="store-check-failed", error=type(e).__name__ + ": " + str(e)[-200:])
        if left and (whole and not S.get("expired") or now() - S["idle_since"] < CONFIRM_WAIT_S):
            if not S.get("release_waits"):
                S["release_waits"] = now(); self.log(event="release-waits", why="the table in the store does not have these results yet", benchmarks=left)
            self.save(); return False
        why = "row done" if whole and not stopped(self.mid, S) else stopped(self.mid, S)
        if self.first and S.get("held") and not self.tried("keep-output", self.keep_output, missing(m, bs)) and S.get("kept", {}).get("tries", 0) < KEEP_TRIES:
            S["release_tried"] = now(); self.save(); return False
        said = []
        if self.pod.release(lambda **k: (said.append(k), self.log(**k)), why=why, row_done=whole and not left):
            self.rec = {}; self.not_first()
            for k in ("idle_since", "release_waits", "release_tried", "release_refused", "node_down"): S.pop(k, None)
            self.tell("released"); self.save(); return True
        # the pod is still the row's, and it costs money: the table says so (state release-refused, and an alarm after ALARM_TRIES)
        r = S.setdefault("release_refused", dict(since=now(), tries=0)); r["tries"] += 1
        r["why"] = str((said[-1].get("why") or said[-1].get("err") or said[-1].get("event")) if said else "")[:200]
        if r["tries"] >= ALARM_TRIES:
            self.log(event="alarm", what="release-refused", tries=r["tries"], since_cr=cr(r["since"]), why=r["why"])
        S["release_tried"] = now(); self.tell("release-refused"); self.save(); return False

    def keep_output(self, bids):
        """Before the pod of the single-node service goes back with a run that is not whole: what the node has of the
        benchmarks without a result (the samples that the run made) comes to a private file on this host. True: it is
        there, or it was asked KEEP_TRIES times (a node that does not answer keeps no pod for ever)."""
        k = self.S.setdefault("kept", dict(tries=0))
        if k.get("done") or k["tries"] >= KEEP_TRIES or not bids:
            return True
        k["tries"] += 1; os.makedirs(f"{PRIV}/{self.mid}", exist_ok=True); path = f"{PRIV}/{self.mid}/node-output.tar.gz"
        dirs = " ".join(shlex.quote(b) for b in bids)
        k["done"] = bool(self.tried("keep-output", self.node.save, f"cd {C.NODE_ROOT}/runs/{self.mid} && D=$(ls -d {dirs} 2>/dev/null); "
                                    f"tar -czf - ${{D:--T /dev/null}}", path, 900))
        self.log(event="output-kept" if k["done"] else "output-not-kept", benchmarks=bids, tries=k["tries"], of=KEEP_TRIES, file=path if k["done"] else None)
        self.save(); return k["done"] or k["tries"] >= KEEP_TRIES

    # ---- one poll
    def step(self, m, bs):
        """One poll of the row. True: the worker has nothing more to do (the row is done, or it waits for the operator)."""
        S = self.S; run = S.get("running")
        if run and run.setdefault("pod_id", self.rec.get("id")) != self.rec.get("id"):
            # The run belonged to a pod that has since been replaced; its run dir went with that pod.
            # Requeue it from scratch on the new pod; this is not counted as a failure of the model.
            self.log(event="pod-changed", old_pod=run["pod_id"], new_pod=self.rec.get("id")); S["running"] = run = None
            S.pop("node_down", None); self.save()
        age = now() - (self.rec.get("rented_at") or now())
        if self.rec and not self.first and not S.get("expired") and age > P.POD_MAX_HOURS * 3600:
            # written first: whatever the node does now (an ssh that hangs, a result file that is cut), the pod goes back
            S["expired"] = dict(at=now(), hours=round(age / 3600, 2)); self.save()
            self.log(event="pod-expired", pod=self.rec.get("huid"), hours=round(age / 3600, 2), cap_hours=P.POD_MAX_HOURS)
            if run:
                self.tried("abort", self.abort, run); self.tried("collect", self.collect, run, False); S["running"] = run = None; self.save()
        if run:
            st = self.node_status(run["run_key"])
            if st is None:
                self.node_down(run); return False
            if S.pop("node_down", None):
                self.log(event="node-back"); self.save()
            lines = int(st.get("lines") or 0)
            if lines > run.get("progress", 0):
                run["progress"] = lines; run["last_progress"] = now(); self.save()
            if now() - run.get("beat", {}).get("at", 0) >= BEAT_S:  # the table says that the worker looks, and what it sees
                run["beat"] = dict(at=now(), lines=lines, grew=run["last_progress"], finished=int(st.get("finished") or 0)); self.save()
            dead, stalled_ = st.get("alive") != "1", now() - run["last_progress"] > STALL_S
            if st.get("done") == "1":
                self.collect(run); S["running"] = None; S.pop("stalled", None); self.save()
            elif self.first and dead:  # what is finished comes to the host, then the row is held
                self.log(event="run-died", first_row=True); self.tried("collect", self.collect, run, False)
                S["running"] = None; S.pop("stalled", None); self.save()
            elif self.first:  # the runner lives: never a kill on that pod, and the pod stays. A stall is an alarm in the table, no more
                if stalled_ and not S.get("stalled"):
                    S["stalled"] = dict(since=run["last_progress"]); self.save()
                    self.log(event="alarm", what="first-row-stalled", since_cr=cr(run["last_progress"]), why=f"no new line for {STALL_S} seconds, and the runner is alive")
                elif not stalled_ and S.pop("stalled", None):
                    self.log(event="first-row-moves-again"); self.save()
            elif dead:
                S["fails"] += 1; self.log(event="run-died", fails=S["fails"]); S["running"] = None; self.save()
            elif stalled_:
                S["fails"] += 1; self.log(event="stalled"); self.abort(run); S["running"] = None; self.save()
        if S.get("running"):
            return False
        bids = missing(m, bs)
        if bids and self.first and not stopped(self.mid, S):  # never a second run on the pod of the single-node service
            S["held"] = dict(at=now(), why="the run on the pod of the single-node service is over and gave no result for " + ", ".join(bids)
                                           + ": no run is started on that pod")
            self.log(event="first-row-held", benchmarks=bids, why=S["held"]["why"]); self.tell("held"); self.save()
        if not bids or stopped(self.mid, S):
            return self.give_back(m, bs)
        if not self.pod_ready():
            return bool(stopped(self.mid, S))
        self.tell("setup")
        try:  # a fetch that passes its hour, or an ssh that breaks, is a failed try as any other: after MAX_FAIL the pod goes back
            ok = self.ensure_weights(m)
        except Exception as e:
            ok = False; self.log(event="fetch", status="raised", error=type(e).__name__)
        if not ok:
            S["fails"] += 1; self.log(event="weights-failed", fails=S["fails"]); self.save(); return False
        self.launch(m, bids); return False


def row_lock(mid):
    """One worker per row. The lock dies with the process. None: a worker of the row is alive."""
    os.makedirs(f"{ROWS}/{mid}", exist_ok=True)
    lock = open(f"{ROWS}/{mid}/worker.lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: lock.close(); return None
    return lock


def row_main(mid):
    """The worker process of one row: poll until the row is done or stopped. A stop of the service ends it; the run on
    the node goes on, and the worker of the next start finds it."""
    lock = row_lock(mid)
    if lock is None:
        raise SystemExit(f"another worker of row {mid} is alive")
    row = Row(mid); bs = R.benchmarks(enabled_only=True, runnable_only=True); row.log(event="row-start", pid=os.getpid())
    while True:
        try:
            m = row.S.get("model") or next((x for x in model_rows() if x["model_id"] == mid), None)  # read once: a store that is down later holds no pod
            if m is None:
                return row.log(event="row-end", why="the row is not listed")
            row.S["model"] = m
            if row.step(m, bs):
                return row.log(event="row-end", state=(row.S.get("phase") or {}).get("state"), stopped=stopped(mid, row.S))
        except Exception:
            row.log(event="error", tb=traceback.format_exc()[-1500:])
            if row.S.get("expired") and row.rec and not row.first:  # a pod past its hours goes back, whatever fails in the step
                row.tried("release", row.pod.release, row.log, "expired")
        time.sleep(POLL_S)


# ------------------------------------------------------------------------------------------------ the service
KIDS = {}  # model_id -> the worker process that this service started
TOLD = set()  # the result files that this process has named as unreadable in the journal


def start_worker(mid):
    k = KIDS.get(mid)
    if k is not None and k.poll() is None:
        return
    lock = row_lock(mid)
    if lock is None:
        return  # a worker of the row lives (started by hand): never two
    lock.close()
    KIDS[mid] = subprocess.Popen([sys.executable, "-m", "affine_bench.cli", "row", mid])
    log(event="row-worker", row=mid, pid=KIDS[mid].pid)


def take_over_single():
    """Once, at the first start of this code: the run and the pod of the single-node service become the first row.
    `running` stays in the service's state as it was, so that the code of before finds its run if it is put back."""
    if S.get("rows_mode"):
        return
    run = S.get("running")
    for mid, n in S["fail"].items():
        if n and not os.path.exists(f"{ROWS}/{mid}/daemon.json"):
            os.makedirs(f"{ROWS}/{mid}", exist_ok=True); wj(f"{ROWS}/{mid}/daemon.json", dict(fails=n, history=[]))
    if run:
        mid = run["model"]["model_id"]; os.makedirs(f"{ROWS}/{mid}", exist_ok=True)
        wj(f"{ROWS}/{mid}/daemon.json", dict(fails=S["fail"].get(mid, 0), history=[], running=run, model=run["model"],
                                             phase=dict(state="running", since=run.get("started")), node_down=S.get("node_down")))
        open(f"{ROWS}/{mid}/first", "w").write("the row that runs on the pod of the single-node service\n")
        log(event="first-row", row=mid, pod_id=run.get("pod_id"))
    S["rows_mode"] = now(); save()


def plan(rows, wait):
    """The rows that have a worker now: each row that has a pod or a run, then the queued rows, newest first, while fewer
    than P.MAX_PODS rows are in work."""
    alive = lambda mid: mid in KIDS and KIDS[mid].poll() is None
    work = [mid for mid in row_ids() if pod_of(mid).load() or row_state(mid).get("running") or alive(mid)]
    for m in queue(rows, wait):
        if m["model_id"] not in work and len(work) < P.MAX_PODS:
            work.append(m["model_id"])
    return work


def view(rows, wait):
    """What the table says of the rows in work: read from the state files of the workers."""
    v = dict(waiting=wait, queue=S.get("queue") or [], runs=[], fail={}, stopped={}, pods={}, history=list(S["history"]), max_pods=P.MAX_PODS,
             profiles={mid: FIRST_ROW_PROFILE for mid in row_ids() if is_first(mid)}, alarms=list(S.get("alarms") or []))
    for mid in row_ids():
        s = row_state(mid); rec = pod_of(mid).load(); ph = s.get("phase") or {}
        if s.get("running"): v["runs"].append(s["running"])
        if is_first(mid): v["node_down"] = s.get("node_down")
        v["fail"][mid] = s.get("fails", 0); v["history"] += s.get("history") or []
        if stopped(mid): v["stopped"][mid] = stopped(mid)
        if rec and (s.get("release_refused") or {}).get("tries", 0) >= ALARM_TRIES:
            v["alarms"].append(dict(what="release-refused", row=mid, since=s["release_refused"]["since"], tries=s["release_refused"]["tries"]))
        if s.get("stalled") and s.get("running") and is_first(mid):
            v["alarms"].append(dict(what="first-row-stalled", row=mid, since=s["stalled"]["since"]))
        if (s.get("rent_failed") or {}).get("no_answer"):
            v["alarms"].append(dict(what="rent-no-answer", row=mid, since=s["rent_failed"]["at"]))
        if rec or ph:
            last = (rj(pod_of(mid).record) or {}).get("last") or {}
            v["pods"][mid] = dict(name=(rec or last).get("huid"), gpu=(rec or last).get("gpu"), usd_h=(rec or last).get("cost"),
                                  state=ph.get("state"), since=ph.get("since"))
    v["history"].sort(key=lambda h: h.get("finished") or 0)
    return v


def lium_check():
    """Each LIUM_CHECK_S, from one listing of Lium (a read): the alarms that a person must see in the table.
    `kept-pod-without-row`: a pod of P.HELD_FOR_ROW is at Lium and its row does not run on it (nobody gives it back).
    `pod-without-row`: a pod with a bench name that no row's record names (it costs money and no row gives it back).
    A listing that fails leaves the alarms as they were. Nothing is deleted here."""
    if now() - S.get("lium_checked", 0) < LIUM_CHECK_S:
        return
    S["lium_checked"] = now(); pods = P.listed()
    if pods is not None:
        mine = {r["id"] for r in (pod_of(mid).load() for mid in row_ids()) if r}
        coming = {P.name_of(mid) for mid in row_ids() if (row_state(mid).get("phase") or {}).get("state") in ("renting", "setup")}
        out = []
        for p in sorted(pods, key=lambda p: str(p.get("name"))):
            pid, name = str(p.get("id") or ""), str(p.get("name") or "")
            row = next((r for x, r in P.HELD_FOR_ROW.items() if pid.startswith(x)), None)
            if row and not (is_first(row) and pid in mine):
                out.append(dict(what="kept-pod-without-row", row=row, pod=name))
            elif not row and name.startswith(P.PREFIX) and pid not in mine and name not in coming:
                out.append(dict(what="pod-without-row", pod=name))
        for a in out: log(event="alarm", **a)
        S["alarms"] = out
    save()


SYNCED = {}  # set once this process has read the published table


def sync_table():
    """Before the first step of a process: start from the published table. A store that cannot be read raises, so no worker is started."""
    if not SYNCED:
        log(event="table-read", published=source.sync_table(publish.OUT)); SYNCED["at"] = now()


def publish_table(rows, wait=None):
    """Rebuild the local table. When a row or a result changed, write it to the store too: the published table is the base
    of that merge, so a published result is never dropped. A failed upload is tried again next step."""
    v = view(rows, S.get("waiting") or {} if wait is None else wait)
    for f, e in publish.unreadable().items():
        if f not in TOLD: TOLD.add(f); log(event="result-unreadable", file=f, error=e)
    publish.publish(rows, v)
    if publish.signature() != S.get("uploaded"):
        if source.sync_table(publish.OUT): publish.publish(rows, v)
        source.upload_table(publish.OUT); S["uploaded"] = publish.signature(); save()
        log(event="table-written", rows={m["model_id"]: m["status"] for m in rj(publish.OUT)["models"]})


def step():
    sync_table()
    take_over_single()
    rows = model_rows(); wait = note_waiting(rows)
    work = plan(rows, wait)
    for mid in work:
        start_worker(mid)
    S["queue"] = [m["model_id"] for m in queue(rows, wait) if m["model_id"] not in work]
    lium_check()
    publish_table(rows, wait)
    return work


def only_instance():
    """One service per state directory: one publisher, and one starter of workers. The lock dies with the process."""
    lock = open(ST + "/daemon.lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: raise SystemExit("another affine-bench daemon holds " + lock.name)
    return lock


def main(once=False):
    """daemon: loop for ever. once: return when every queued row is benchmarked and every pod is given back. A row whose
    weight files are not in the store is not queued: `once` leaves it for a later run."""
    log(event="start", pid=os.getpid())
    while True:
        try:
            open(ST + "/heartbeat", "w").write(str(now()))
            work = step()
            if once and not work and not S.get("queue") and not any(k.poll() is None for k in KIDS.values()):
                return log(event="drained")
        except Exception:
            log(event="error", tb=traceback.format_exc()[-1500:])
        time.sleep(POLL_S)


# ------------------------------------------------------------------------------------------------ commands
def pods_check():
    """Reads only: every bench pod that Lium lists, with its row; a bench pod that no row claims; a row that should have a
    pod and has none. It prints no address and deletes nothing."""
    pods = P.listed()
    if pods is None:
        raise SystemExit("Lium could not be asked: nothing is known")
    recs = {mid: pod_of(mid).load() for mid in row_ids()}
    by_id = {r["id"]: mid for mid, r in recs.items() if r}
    lines = ["%-9s %-42s %-26s %-12s %7s %7s" % ("ID", "NAME", "ROW", "STATE", "AGE_H", "USD_H")]
    ids = set()
    for p in sorted(pods, key=lambda p: str(p.get("name"))):
        name, pid = str(p.get("name") or ""), str(p.get("id") or "")
        if not name.startswith(P.PREFIX) and pid not in by_id:
            continue
        ids.add(pid); mid = by_id.get(pid); rec = recs.get(mid) or {}
        state = (row_state(mid).get("phase") or {}).get("state") or "-" if mid else "-"
        kept = next((r for x, r in P.HELD_FOR_ROW.items() if pid.startswith(x)), None)
        row = mid or "NO ROW CLAIMS THIS POD" + (f" (IT IS KEPT FOR ROW {kept})" if kept else "")
        if mid and name != pod_of(mid).name:
            row += " (THE NAME IS NOT THE ROW'S)"
        age = "%.1f" % ((now() - rec["rented_at"]) / 3600) if rec.get("rented_at") else "?"
        lines.append("%-9s %-42s %-26s %-12s %7s %7s" % (pid[:8], name, row, state, age, p.get("price_per_hour", rec.get("cost"))))
    for mid, rec in recs.items():
        if rec and rec["id"] not in ids:
            lines.append("%-9s %-42s %-26s %-12s" % (rec["id"][:8], rec.get("name"), mid, "NOT LISTED BY LIUM: the row's record names a pod that is not there"))
        ph = (row_state(mid).get("phase") or {}).get("state")
        if not rec and (row_state(mid).get("running") or ph in ("setup", "running")):
            lines.append("%-9s %-42s %-26s %-12s" % ("-", "-", mid, f"THE ROW IS IN STATE {ph} AND HAS NO POD"))
    print("\n".join(lines)); print(f"{len(ids)} bench pods at Lium; ceiling {P.MAX_PODS} rows in work; price cap {P.MAX_PRICE} $/h")


def cli(cmd, argv):
    if cmd == "status":
        rows = model_rows(); wait = waiting(rows); q = queue(rows, wait)
        work = {mid: dict(state=(row_state(mid).get("phase") or {}).get("state"), stopped=stopped(mid), fails=row_state(mid).get("fails", 0),
                          running=bool(row_state(mid).get("running")), pod=(pod_of(mid).load() or {}).get("huid"))
                     | {k: row_state(mid)[k] for k in ("held", "release_refused") if row_state(mid).get(k)} for mid in row_ids()}
        print(json.dumps(dict(rows=work, queue_len=len(q), next=[(m["model_id"], m.get("why")) for m in q[:12]], waiting=wait), indent=1)); return
    if cmd == "offers":  # reads only: what a rent could take now
        offs, seen = P.offers(); kinds = {}
        for o in offs: kinds.setdefault((o["kind"], o["price"]), []).append(o)
        for (kind, price), group in sorted(kinds.items(), key=lambda kv: kv[0][1]):
            print(f"{kind}  {price} $/h  x{len(group)}")
        print(f"{len(offs)} offers fit: a whole free node of `{P.GPU_PREF}`, secure tier, at or below {P.MAX_PRICE} $/h. The cheapest is taken first."); return
    if cmd == "pods":
        return pods_check()
    if cmd == "row":
        return row_main(argv[0])
    if cmd == "retry":  # a row that is stopped (failed, held, rent-failed, expired) is queued again; no stop of the service is needed
        for mid in argv:
            lock = row_lock(mid)
            if lock is None:
                raise SystemExit(f"the worker of row {mid} is alive: stop the service, or wait until the worker has ended")
            s = row_state(mid)
            for k in ("held", "rent_failed", "expired", "rent_tries", "rent_wait_until", "no_offer_at"): s.pop(k, None)
            s["fails"] = 0; wj(f"{ROWS}/{mid}/daemon.json", s); lock.close()
        print(json.dumps(dict(retry=argv))); return
    if cmd not in ("daemon", "once", "publish", "release"):
        raise SystemExit("unknown command " + cmd)
    lock = only_instance()  # held until the process exits: each of these commands does the service's work
    if cmd in ("daemon", "once"):
        return main(once=cmd == "once")
    if cmd == "publish":
        sync_table(); publish_table(model_rows()); print("published"); return
    for mid in argv:  # release ROW...: give the pod of a row back now; the next start runs what the row still misses on a new pod
        row = Row(mid)
        if row.pod.release(log, why="operator"):
            row.S["running"] = None; row.S.pop("node_down", None); row.not_first(); row.tell("released"); row.save()
