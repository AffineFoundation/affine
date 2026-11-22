"""Host side: the service of the per-row benchmark, the worker of one row, and the CLI commands.

The service (one process, `daemon`), every 30 s: read the rows from v2's public files (source.py: the launch model,
every folded king, and the last king of v1; no journal and no call into the controller), build the queue, start a
worker for each row that has work, and write the table: BENCH_SITE/benchmarks.json and, when a row or a result
changed, the same table to the store. Only the service writes the table. Nothing is published per task.

The worker of a row (one process for each row, `row <model_id>`, class Row): rent the row's pod, set it up, download
the row's weights, run the benchmarks, collect them into BENCH_STATE/results/, stream each public rollouts.jsonl.gz to its
store key, and give the pod back when the table in the store has the row's results. Rollout uploads never gate the aggregate table. Rows run side by side (Jacob, 2026-11-14). A worker waits for its own rent, setup and
download only: no worker holds a lock, a pod or a file of another row. A row that fails keeps no pod.

Done marker: a (row, benchmark, version) is done when the published table has its result, or a result file is on this
host. The service reads the published table before its first step and starts no worker until it has. So a row is
benchmarked once however often it is listed, and a host with an empty disk does not run a king again. A crashed run is
resumed in place (vf-eval --resume keeps finished rollouts).

Queue: every row that has a missing result, newest first; the launch model (genesis) and the v1 king are the oldest
rows. A row whose weight files are not in the store yet waits outside the queue: no fail, no rent, no worker.
At most P.MAX_PODS rows are in work at a time; the next row starts when one of them is done.
Informational only: this service reads kings and writes a table. It has no path to the head, the queue or chain weights.

The single-node service (the code of before 2026-11-14) had one pod and one run at a time. This code never writes the
`running` key of that service's record: a run there is a run of that service, and it is cut at each start that finds
one (take_over_single). The record of the run goes to `cut_run`, the run's row is a row as any other and runs again on
a pod of its own. The pod of that service (BENCH_STATE/pod.json) gets no run from this code: the service gives it back
by its exact id (release_single). No row and no pod has a name in this code, and no run of that service is adopted.

A stored result with more samples a problem than the settings ask for now (a run of before the count was cut) is
brought to the settings' count on this host by the fixed rule of derive.py (derive_stored). Its row is not run again
for it. When the rule gives no value, the row is `held` until the operator says `retry`.

Pod (affine_bench/pod.py): one for each row. A pod that Lium reclaims is detected (node-gone) and the row rents again.
Nothing on a pod is deleted by this service: the weights go away with the pod (results on the host are never deleted).
Only rollout files explicitly observed during collection enter the row's retry list; after three failed attempts the
identity is parked. Legacy results are never scanned to discover rollout files."""
import fcntl, glob, gzip, hashlib, json, os, shlex, subprocess, sys, time, traceback
from datetime import datetime, timezone
from sn120_v2 import layout
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
ROLLOUT_TRIES = 3; ROLLOUT_MAX_ATTEMPTS = 3; ROLLOUT_BACKOFF_S = 300; ROLLOUT_MAX_BACKOFF_S = 86400
ROLLOUT_FETCH_WAIT_S = 600
WAIT_LATE_S = 3600  # weight files still not in the store this long after the head was applied: the table says `waiting`
BEAT_S = 300  # while a row runs, the table gets the time and the counts of the worker's look at the node this often
CONFIRM_WAIT_S = 600  # a row that failed waits this long for the table in the store before its pod is given back
RELEASE_RETRY_S = 600  # a release that Lium or a guard refused is tried again this often
ALARM_TRIES = 3  # a release that was refused this many times is an alarm in the table
LIUM_CHECK_S = 600  # the service reads Lium's listing this often for the alarms of the table (reads only)
POD_CHECK_AFTER_S = 600; POD_CHECK_EVERY_S = 600


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
def pod_of(mid): return P.Pod(mid)
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


# ------------------------------------------------------------------------------------------------ public rollouts of one row
def node_file(node, remote, local):
    """Stream one row's file through its ssh wrapper without putting its bytes in memory."""
    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = local + ".part"
    proc = None
    try:
        with open(tmp, "wb") as out:
            proc = subprocess.Popen([node.ssh, f"cat {shlex.quote(remote)}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            while chunk := proc.stdout.read(8 << 20):
                out.write(chunk)
            if proc.wait(timeout=60) != 0:
                raise IOError("node file transfer failed")
        os.replace(tmp, local)
    except BaseException:
        if proc is not None:
            proc.kill(); proc.wait()
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def fetch_rollout(node, rk, bid, local, tell=log):
    """Hash on this row's node, stream to a private cache, then verify those exact bytes."""
    remote = f"{C.NODE_ROOT}/runs/{rk}/{bid}/rollouts.jsonl.gz"
    quoted = shlex.quote(remote)
    answer = node.out(f"sha256sum {quoted} && stat -c %s {quoted}", timeout=120).split()
    if len(answer) < 2 or len(answer[0]) != 64 or any(c not in "0123456789abcdef" for c in answer[0]) or not answer[-1].isdigit():
        raise IOError("rollout sha256/size unavailable")
    sha, size = answer[0], int(answer[-1])
    for attempt in range(ROLLOUT_TRIES):
        try:
            node_file(node, remote, local)
            h = hashlib.sha256()
            with open(local, "rb") as f:
                while chunk := f.read(8 << 20):
                    h.update(chunk)
            if h.hexdigest() != sha or os.path.getsize(local) != size:
                raise IOError("rollout transfer hash mismatch")
            return dict(sha256=sha, bytes=size)
        except Exception as exc:
            tell(event="rollout-fetch-retry", model=rk, benchmark=bid, attempt=attempt + 1, error=type(exc).__name__)
    raise IOError("rollout transfer failed after three tries")


def upload_rollout(mid, bid, path, result, save_state, tell=log):
    """Durable public model/task data only; no pod, host, address, path or credential is added to the metadata."""
    digest = hashlib.sha256()
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("model_id") != mid or row.get("benchmark_id") != bid:
                raise ValueError("rollout model/benchmark identity mismatch")
            rows += 1
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    data, size = digest.hexdigest(), os.path.getsize(path)
    key = layout.public_rollout_bench(mid, bid)
    store = source.store()
    store.put_file(key, path)
    if store.sha256(key) != data:
        raise IOError("stored rollout sha256 mismatch")
    finished = result.get("finished_at")
    meta = dict(schema="sn120-v2-rollouts-bench-1", model_id=mid, benchmark_id=bid, rows=rows, bytes=size, sha256=data,
                bench_version=result.get("bench_version"), finished_at=datetime.fromtimestamp(finished, timezone.utc).isoformat().replace("+00:00", "Z") if finished is not None else None,
                key=key)
    store.put(layout.public_rollout_bench_meta(mid, bid), json.dumps(meta, sort_keys=True, indent=1).encode())
    save_state(data)
    tell(event="rollouts-uploaded", model=mid, benchmark=bid, rows=rows, bytes=size, sha256=data)


class Row:
    """One row in work: the state of its worker (BENCH_STATE/rows/<model_id>/daemon.json), its pod and its node."""

    def __init__(self, mid):
        self.mid = mid; self.dir = f"{ROWS}/{mid}"; os.makedirs(self.dir, exist_ok=True)
        self.sf = self.dir + "/daemon.json"
        self.S = row_state(mid); self.S.setdefault("fails", 0); self.S.setdefault("history", [])
        self.S.setdefault("rollout_uploads", {})
        self.S.setdefault("rollout_pending", {})  # only files this worker saw; never discovered from legacy results
        self.attach()

    def attach(self):
        self.pod = P.Pod(self.mid); self.rec = self.pod.load(); self.node = node_of(self)

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
            identity = f"{m['model_id']}/{bid}"
            if identity not in self.S["rollout_uploads"] and identity not in self.S["rollout_pending"]:
                exists = self.node.out(f"test -f {C.NODE_ROOT}/runs/{rk}/{bid}/rollouts.jsonl.gz && echo yes", timeout=15).strip() == "yes"
                if exists:
                    # Only a file observed by this code becomes pending; legacy results never enter retry discovery.
                    pending = dict(model=m["model_id"], run_key=rk, bid=bid, next_at=0.0, delay=ROLLOUT_BACKOFF_S, attempts=0)
                    self.S["rollout_pending"][identity] = pending
                    self.save()  # the explicit pending marker survives a transfer crash; it never comes from result discovery
                    self.collect_rollout(bid, r, pending)
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

    # ---- public rollouts: a side effect of score collection, never its gate
    def rollout_path(self, pending):
        return f"{PRIV}/{pending['run_key']}/{pending['bid']}.rollouts.jsonl.gz"

    def rollout_pending(self, only_fetch=False):
        """Explicitly observed identities only. Cached artifacts need no node; uncached ones need this row's node."""
        out = []
        for identity, pending in self.S["rollout_pending"].items():
            if pending.get("parked_at") or (only_fetch and os.path.isfile(self.rollout_path(pending))):
                continue
            if not only_fetch and float(pending.get("next_at", 0)) > now():
                continue
            result = rj(result_path(self.mid, pending["bid"]))
            if result:
                out.append((identity, pending, result))
        return out

    def park_rollout(self, identity, pending, why):
        pending["parked_at"] = now(); pending["why"] = why
        self.save()
        self.log(event="rollouts-lost", benchmark=pending["bid"], why=why,
                 fetch_attempts=pending.get("fetch_attempts", 0), upload_attempts=pending.get("upload_attempts", 0))

    def retry_rollouts(self):
        """At most three explicitly pending identities per poll; only uncached files still need this row's node."""
        ready = []
        for identity, pending, result in self.rollout_pending():
            cached = os.path.isfile(self.rollout_path(pending))
            if cached and float(pending.get("next_at", 0)) > now():
                continue
            ready.append((float(pending.get("next_at", 0)) if cached else 0.0, pending["bid"], identity, result))
        for _, bid, identity, result in sorted(ready)[:ROLLOUT_TRIES]:
            self.collect_rollout(bid, result, self.S["rollout_pending"][identity])

    def collect_rollout(self, bid, result, pending, force_fetch=False):
        mid, rk, identity = self.mid, pending["run_key"], f"{self.mid}/{bid}"
        path = self.rollout_path(pending)
        if not os.path.isfile(path) and not self.rec:
            return self.park_rollout(identity, pending, "node-released")
        try:
            known = self.S["rollout_uploads"].get(identity)
            stored = None
            if known:
                try:
                    stored = source.store().sha256(layout.public_rollout_bench(mid, bid))
                except Exception:
                    pass  # a stored object that cannot be checked is re-uploaded, not mistaken for a completed retry
            if known and stored == known:
                self.S["rollout_pending"].pop(identity, None)
                self.save()
                return
            if not os.path.isfile(path):
                if pending.get("fetch_attempts", 0) >= ROLLOUT_MAX_ATTEMPTS:
                    return self.park_rollout(identity, pending, "fetch-failed")
                pending["fetch_attempts"] = int(pending.get("fetch_attempts", pending.get("attempts", 0))) + 1
                try:
                    fetch_rollout(self.node, rk, bid, path, self.log)
                    pending["fetched_at"] = now()
                    pending.pop("fetch_attempts", None)
                except Exception as exc:
                    self.save()
                    self.log(event="rollouts-fetch-deferred", benchmark=bid, attempts=pending["fetch_attempts"], error=type(exc).__name__)
                    return  # give_back bounds this wait; a successful fetch is a prerequisite of release
            if force_fetch:
                self.save()  # the cache is durable; the R2 retry continues after release, not in this fetch-only phase
                return
            for attempt in range(ROLLOUT_TRIES):
                try:
                    def uploaded(data):
                        self.S["rollout_uploads"][identity] = data
                        self.S["rollout_pending"].pop(identity, None)
                        self.save()
                    upload_rollout(mid, bid, path, result, uploaded, self.log)
                    os.unlink(path)  # R2 is durable; this row's private download is disposable.
                    return
                except Exception as exc:
                    self.log(event="rollouts-upload-retry", benchmark=bid, attempt=attempt + 1, error=type(exc).__name__)
            pending["upload_attempts"] = int(pending.get("upload_attempts", pending.get("attempts", 0))) + 1
            pending["attempts"] = pending["upload_attempts"]  # compatibility with the first rollout writer's state
            if pending["upload_attempts"] >= ROLLOUT_MAX_ATTEMPTS:
                return self.park_rollout(identity, pending, "upload-failed")
            pending["next_at"] = now() + float(pending.get("delay", ROLLOUT_BACKOFF_S))
            pending["delay"] = min(ROLLOUT_MAX_BACKOFF_S, float(pending.get("delay", ROLLOUT_BACKOFF_S)) * 2)
            self.save()
            self.log(event="rollouts-upload-deferred", benchmark=bid, attempts=pending["upload_attempts"])
        except Exception as exc:  # malformed data or store state never gates scores or holds a node forever
            pending["upload_attempts"] = int(pending.get("upload_attempts", 0)) + 1
            self.save()
            self.log(event="rollouts-upload-deferred", benchmark=bid, error=type(exc).__name__, detail=P.mask(traceback.format_exc(limit=3)[-600:]))
            if pending["upload_attempts"] >= ROLLOUT_MAX_ATTEMPTS:
                self.park_rollout(identity, pending, "upload-failed")

    def before_release(self):
        """At most three fetch-only attempts per observed identity before losing its node; cached uploads need no node."""
        started = now()
        for _ in range(ROLLOUT_MAX_ATTEMPTS):
            for identity, pending, _ in self.rollout_pending(only_fetch=True):
                if pending.get("fetch_attempts", 0) >= ROLLOUT_MAX_ATTEMPTS:
                    self.park_rollout(identity, pending, "fetch-failed")
            left = [(identity, pending, result) for identity, pending, result in self.rollout_pending(only_fetch=True)
                    if pending.get("fetch_attempts", 0) < ROLLOUT_MAX_ATTEMPTS]
            if not left:
                return
            self.log(event="rollouts-fetch-wait", benchmarks=[pending["bid"] for _, pending, _ in left])
            for identity, pending, result in left:
                if now() - started >= ROLLOUT_FETCH_WAIT_S:
                    self.park_rollout(identity, pending, "fetch-failed")
                    continue
                if not os.path.isfile(self.rollout_path(pending)):
                    self.collect_rollout(pending["bid"], result, pending, True)
            if self.rollout_pending(only_fetch=True):
                time.sleep(POLL_S)
        for identity, pending, _ in self.rollout_pending(only_fetch=True):
            self.park_rollout(identity, pending, "fetch-failed")

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
            self.attach()
        else:
            self.log(event="node-unreachable", lium=st, down_min=round((now() - d["since"]) / 60))
        self.save()

    # ---- pod
    def pod_ready(self):
        """True when the row's pod is set up. A rent blocks this worker for some minutes, and this worker only."""
        if self.rec and self.rec.get("bootstrapped"):
            return True
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
        if self.rollout_pending(only_fetch=True):
            self.tell("fetching-rollouts")
            self.before_release()
            if self.rollout_pending(only_fetch=True):
                self.save(); return False
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
        said = []
        if self.pod.release(lambda **k: (said.append(k), self.log(**k)), why=why):
            self.attach()
            for k in ("idle_since", "release_waits", "release_tried", "release_refused", "node_down"): S.pop(k, None)
            self.tell("released"); self.save(); return True
        # the pod is still the row's, and it costs money: the table says so (state release-refused, and an alarm after ALARM_TRIES)
        r = S.setdefault("release_refused", dict(since=now(), tries=0)); r["tries"] += 1
        r["why"] = str((said[-1].get("why") or said[-1].get("err") or said[-1].get("event")) if said else "")[:200]
        if r["tries"] >= ALARM_TRIES:
            self.log(event="alarm", what="release-refused", tries=r["tries"], since_cr=cr(r["since"]), why=r["why"])
        S["release_tried"] = now(); self.tell("release-refused"); self.save(); return False

    # ---- one poll
    def step(self, m, bs):
        """One poll of the row. True: the worker has nothing more to do (the row is done, or it waits for the operator)."""
        S = self.S; run = S.get("running")
        self.retry_rollouts()
        if run and run.setdefault("pod_id", self.rec.get("id")) != self.rec.get("id"):
            # The run belonged to a pod that has since been replaced; its run dir went with that pod.
            # Requeue it from scratch on the new pod; this is not counted as a failure of the model.
            self.log(event="pod-changed", old_pod=run["pod_id"], new_pod=self.rec.get("id")); S["running"] = run = None
            S.pop("node_down", None); self.save()
        age = now() - (self.rec.get("rented_at") or now())
        if self.rec and not S.get("expired") and age > P.POD_MAX_HOURS * 3600:
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
            elif dead:
                S["fails"] += 1; self.log(event="run-died", fails=S["fails"]); S["running"] = None; self.save()
            elif stalled_:
                S["fails"] += 1; self.log(event="stalled"); self.abort(run); S["running"] = None; self.save()
        if S.get("running"):
            return False
        bids = missing(m, bs)
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
            if row.S.get("expired") and row.rec:  # a pod past its hours goes back, whatever fails in the step
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
    """A run in work of the single-node service is cut, at every start that finds one: this code never writes `running`
    in that service's record, so a run there is of that service (the first start; also a start after a way back to that
    service, which keeps the key `rows_mode` that it does not know). The record goes from `running` to `cut_run` (kept,
    for the operator), and the run's row is a row as any other: it runs again on a pod of its own. No run is adopted.
    The fails of that service stay with their rows."""
    run = S.get("running")
    if not run and S.get("rows_mode"):
        return
    for mid, n in S["fail"].items():
        if n and not os.path.exists(f"{ROWS}/{mid}/daemon.json"):
            os.makedirs(f"{ROWS}/{mid}", exist_ok=True); wj(f"{ROWS}/{mid}/daemon.json", dict(fails=n, history=[]))
    if run:
        S["cut_run"] = dict(run, cut_at=now()); S["running"] = None; S.pop("node_down", None)
        log(event="single-run-cut", row=(run.get("model") or {}).get("model_id"), started_cr=cr(run.get("started")), pod_id=str(run.get("pod_id") or "")[:8])
    S["rows_mode"] = now(); save()


def single_pod():
    """The pod of the single-node service (BENCH_STATE/pod.json)."""
    return P.Pod("", first=True)


def release_single(force=False):
    """The pod of the single-node service is given back when no run is recorded on it: by its exact id, through
    pod.Pod.release (the only delete), tried again each RELEASE_RETRY_S. True: that pod is not rented any more.
    It stays while the per-problem lines of a scored row are not on this host (derive_stored): they may be on that pod
    only. `force` is the operator's word (`release-single`): no wait, and not that guard. The guards of the delete hold.
    Each return that leaves the pod rented, but the wait between two tries, counts a try: the alarm always comes."""
    rec = single_pod().load()
    if not rec:
        return True
    if not force and now() - S.get("single_release_tried", 0) < RELEASE_RETRY_S:
        return False
    said = []; lost = [f for f in S.get("lines_missing") or [] if not os.path.exists(f)]
    rows = [mid for mid in row_ids() if P.name_of(mid) == P.NAME or pod_of(mid).load().get("id") == rec["id"]]
    if S.get("running"):  # take_over_single cuts it at the start of each step: only a command beside a stopped service comes here
        said.append(dict(why="a run of the single-node service is recorded on this pod"))
        log(event="release-refused", **said[-1])
    elif rows:  # BENCH_POD_NAME is the name of a row's pod, or the record names a row's pod: that pod is the row's, with its run
        said.append(dict(why=f"the record of the single-node service names the pod of row {rows[0]}: that pod is the row's"))
        log(event="release-refused", **said[-1])
    elif lost and not force:  # the lines may be on that pod only: it stays until they are on this host, or until the operator's word (`release-single`)
        said.append(dict(why=f"the per-problem lines of a scored row are not on this host and may be on this pod: {', '.join(os.path.relpath(f, PRIV) for f in lost)}"))
        log(event="release-refused", **said[-1])
    elif single_pod().release(lambda **k: (said.append(k), log(**k)), why="no row runs on the pod of the single-node service"):
        S.pop("single_release", None); S.pop("single_release_tried", None); save(); return True
    r = S.setdefault("single_release", dict(since=now(), tries=0)); r["tries"] += 1
    r["why"] = str((said[-1].get("why") or said[-1].get("err") or said[-1].get("event")) if said else "")[:200]
    S["single_release_tried"] = now(); save(); return False


def hold_row(mid, why):
    """A row with no worker is stopped until the operator says `retry`."""
    s = row_state(mid); s.setdefault("fails", 0); s.setdefault("history", [])
    s["held"] = dict(at=now(), why=why)
    os.makedirs(f"{ROWS}/{mid}", exist_ok=True); wj(f"{ROWS}/{mid}/daemon.json", s)


def derive_stored():
    """A result on this host of a run with more samples a problem than the settings ask for now, of a row that is not in
    work: its value at the settings' count, by the fixed rule of derive.py, from the per-problem lines that the host
    has. Nothing is rented for it. The full run's result is kept beside the lines (`<benchmark>.full.json`), and that
    file says that the result was looked at: once only. No value: the row is `held`; it is never run again by itself.
    The same holds for a row whose full run is in the published table and whose result file is not on this host, or
    cannot be read there: no value, and the row is `held` (`no_value` in the state says that it was looked at)."""
    table = [c for c in (rj(publish.OUT) or {}).get("results") or [] if isinstance(c, dict)]; done = published()
    for b in R.benchmarks(enabled_only=True, runnable_only=True):
        bid, want = b["id"], R.bench_version(b)

        def more(r):
            try:
                return isinstance(r, dict) and (r.get("bench_version") or r.get("env_version")) != want and int(r.get("rollouts") or 0) > int(b["rollouts"])
            except (TypeError, ValueError):
                return False
        told = {c.get("model_id") for c in table if c.get("benchmark_id") == bid and more(c)}  # rows whose full run the table has
        for mid in sorted(x for x in {os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{RES}/*/{bid}.json")} | told if x):
            f = result_path(mid, bid); r = rj(f); full = f"{PRIV}/{mid}/{bid}.full.json"; lines = f"{PRIV}/{mid}/{bid}.per_task.jsonl"; key = f"{mid}/{bid}"
            if (mid, bid, want) in done or (isinstance(r, dict) and r.get("bench_version") == want) or not (more(r) or mid in told):
                continue  # the row has its result at the settings' count, or it has no run with more samples
            if os.path.exists(full) or key in (S.get("no_value") or {}) or row_state(mid).get("running") or pod_of(mid).load():
                continue  # looked at before; or a row in work: its worker brings the result to the settings' count (Row.collect)
            new, why = None, "the result file of the full run is not on this host" if r is None else "the result file of the full run on this host cannot be used"
            try:
                if more(r):
                    os.makedirs(f"{PRIV}/{mid}", exist_ok=True); wj(full, r, 0o600)
                    new, why = derive.from_full(r, open(lines).read(), int(b["rollouts"]), want, b.get("problems")) if os.path.exists(lines) else \
                        (None, "the per-problem lines of the full run are not on this host")
                    if new is not None:
                        publish.row_of(new, b)  # raises when the table could not show this result
            except Exception as e:
                new, why = None, f"the result on this host cannot be read ({type(e).__name__}: {str(e)[:80]})"
            if new is None:
                if not os.path.exists(lines):  # release_single keeps the pod of the single-node service while these lines are not here
                    S["lines_missing"] = sorted(set(S.get("lines_missing") or []) | {lines})
                S.setdefault("no_value", {})[key] = now(); save()
                hold_row(mid, f"{bid}: {why}"); log(event="no-value", row=mid, bench=bid, why=why); continue
            wj(f, new)
            log(event="derived", row=mid, bench=bid, rule=derive.RULE, score=round(new["score"], 4), score_full=new["score_full"], samples_full=new["samples_full"])


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
             alarms=list(S.get("alarms") or []))
    for mid in row_ids():
        s = row_state(mid); rec = pod_of(mid).load(); ph = s.get("phase") or {}
        if s.get("running"): v["runs"].append(s["running"])
        v["fail"][mid] = s.get("fails", 0); v["history"] += s.get("history") or []
        if stopped(mid): v["stopped"][mid] = stopped(mid)
        if rec and (s.get("release_refused") or {}).get("tries", 0) >= ALARM_TRIES:
            v["alarms"].append(dict(what="release-refused", row=mid, since=s["release_refused"]["since"], tries=s["release_refused"]["tries"]))
        if (s.get("rent_failed") or {}).get("no_answer"):
            v["alarms"].append(dict(what="rent-no-answer", row=mid, since=s["rent_failed"]["at"]))
        if rec or ph:
            last = (rj(pod_of(mid).record) or {}).get("last") or {}
            v["pods"][mid] = dict(name=(rec or last).get("huid"), gpu=(rec or last).get("gpu"), usd_h=(rec or last).get("cost"),
                                  state=ph.get("state"), since=ph.get("since"))
    if (S.get("single_release") or {}).get("tries", 0) >= ALARM_TRIES and single_pod().load():
        v["alarms"].append(dict(what="single-pod-not-released", pod=P.NAME, since=S["single_release"]["since"], tries=S["single_release"]["tries"]))
    v["history"].sort(key=lambda h: h.get("finished") or 0)
    return v


def lium_check():
    """Each LIUM_CHECK_S, from one listing of Lium (a read): the alarm that a person must see in the table.
    `pod-without-row`: a pod with a bench name that no row's record names (it costs money and no row gives it back).
    The pod of the single-node service is not one of them: the service gives it back, and says so when it cannot
    (`single-pod-not-released`, view()). A listing that fails leaves the alarms as they were. Nothing is deleted here."""
    if now() - S.get("lium_checked", 0) < LIUM_CHECK_S:
        return
    S["lium_checked"] = now(); pods = P.listed()
    if pods is not None:
        mine = {r["id"] for r in [pod_of(mid).load() for mid in row_ids()] + [single_pod().load()] if r}
        coming = {P.name_of(mid) for mid in row_ids() if (row_state(mid).get("phase") or {}).get("state") in ("renting", "setup")}
        out = []
        for p in sorted(pods, key=lambda p: str(p.get("name"))):
            pid, name = str(p.get("id") or ""), str(p.get("name") or "")
            if name.startswith(P.PREFIX) and pid not in mine and name not in coming:
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
    derive_stored()
    release_single()
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
    by_id = {r["id"]: mid for mid, r in recs.items() if r}; single = single_pod().load().get("id")
    lines = ["%-9s %-42s %-26s %-12s %7s %7s" % ("ID", "NAME", "ROW", "STATE", "AGE_H", "USD_H")]
    ids = set()
    for p in sorted(pods, key=lambda p: str(p.get("name"))):
        name, pid = str(p.get("name") or ""), str(p.get("id") or "")
        if not name.startswith(P.PREFIX) and pid not in by_id and pid != single:
            continue
        ids.add(pid); mid = by_id.get(pid); rec = recs.get(mid) or {}
        state = (row_state(mid).get("phase") or {}).get("state") or "-" if mid else "-"
        row = mid or ("THE POD OF THE SINGLE-NODE SERVICE: no row runs on it, the service gives it back" if pid == single else "NO ROW CLAIMS THIS POD")
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
    if cmd not in ("daemon", "once", "publish", "release", "release-single"):
        raise SystemExit("unknown command " + cmd)
    lock = only_instance()  # held until the process exits: each of these commands does the service's work
    if cmd in ("daemon", "once"):
        return main(once=cmd == "once")
    if cmd == "publish":
        sync_table(); publish_table(model_rows()); print("published"); return
    if cmd == "release-single":  # the pod of the single-node service, when no run is recorded on it
        print(json.dumps(dict(single_pod_rented=not release_single(force=True)))); return
    for mid in argv:  # release ROW...: give the pod of a row back now; the next start runs what the row still misses on a new pod
        row = Row(mid)
        if row.pod.release(log, why="operator"):
            row.S["running"] = None; row.S.pop("node_down", None); row.attach(); row.tell("released"); row.save()
