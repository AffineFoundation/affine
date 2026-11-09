#!/usr/bin/env python3
"""PE warm pool (throughput-v4, 2026-11-07 CR, Jacob t1365u "optimize all the pieces").
Keeps up to MAX_EVAL_NODES eval nodes (lanes + warm, dedicated baseline excluded) and, of those, warm_target() idle nodes
pre-booted with the SAME public bundle / live envs (gate-pool-v2 snapshot, sha-pinned) / 64K harness that every lane installs.
As soon as a candidate's training result is pending (state/life/pending/e<N>.json, i.e. TRAINING-RESULT landed), an idle warm
node is assigned to it and the candidate checkpoint is fetched + served (pe_lane.fetch_ckpt / pe_lane.serve, MAX_SEQS 512).
A lane later CLAIMS that node instead of renting cold.

Semantics: unchanged. Warm/prefetch never sees a draw, indices, plan seed or any private round data (only what a lane ships
before it starts the probe: public bundle, public envs, public checkpoint). The lane re-runs boot+push_bundle (idempotent
verification of bundle sha marker, envs sha pin, pool 21578, harness 64K) and verifies serve readiness itself before probing.
A warm node is used for at most ONE probe and is always released by the lane (finally) afterwards; failed prep/prefetch -> release.
State: state/WARM.json (0600) under state/warm.lock. Workers are detached processes (exact PID tracked), survive controller restarts.
"""
import fcntl, json, os, signal, subprocess, sys, time, contextlib
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
import pe_rent

WARM = ST / "WARM.json"
LOCK = ST / "warm.lock"
PENDING = ST / "life" / "pending"
WAIT_WORKER_MAX_S = 2400
UNCLAIMED = ("renting", "booting", "ready", "prefetching", "served")
is_baseline = pe_rent.is_baseline
eval_nodes = pe_rent.eval_nodes


@contextlib.contextmanager
def locked():
    ST.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        W = rj(WARM, {}) or {}
        W.setdefault("pods", {})
        yield W
        wj(WARM, W, 0o600)


def warm_max_total():
    """flag parallel_eval.warm_max_total (t1368u: 2) = cap on ALL warm machines, claimed or idle; default MAX_EVAL_NODES."""
    try:
        n = int(pe_cfg().get("warm_max_total", MAX_EVAL_NODES))
    except Exception:
        n = MAX_EVAL_NODES
    return max(0, min(n, MAX_EVAL_NODES))


def snapshot():
    W = rj(WARM, {}) or {}
    return W.get("pods") or {}


def _alive(pid, needle):
    try:
        os.kill(int(pid), 0)
        return needle in Path("/proc/%d/cmdline" % int(pid)).read_text(errors="ignore")
    except Exception:
        return False


def worker_alive(e):
    w = e.get("worker") or {}
    return bool(w.get("pid")) and _alive(w["pid"], "pe_warm")


def _spawn(kind, name, *args):
    LOGS.mkdir(parents=True, exist_ok=True)
    lg = open(LOGS / ("warm-%s.log" % name), "ab")
    p = subprocess.Popen([HOST_PY, "-u", "-B", str(CODE / "pe_warm.py"), kind, name, *args], stdout=lg, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(CODE))
    return dict(pid=p.pid, kind=kind, at=time.time(), at_cr=cr())


# ------------------------------------------------------------------ candidates
def _job_state(e):
    c = rj(JOBS / ("live-e%d" % e) / "CTL.json")
    return None if c is None else c.get("state")


def waiting_candidates():
    """[(epoch, candidate, files)] pending PE candidates (training result landed) with no decision and no lane holding a pod yet."""
    out = []
    frm = int(pe_cfg().get("from_epoch") or 0)
    for p in sorted(PENDING.glob("e*.json")):
        d = rj(p) or {}
        try:
            e = int(d.get("epoch"))
        except Exception:
            continue
        C = d.get("candidate")
        if not C or e < frm or (EP / str(e) / "DECISION.json").exists():
            continue
        if _job_state(e) not in (None, "new", "backoff"):
            continue
        out.append((e, C, (d.get("checkpoint") or {}).get("files")))
    return out


def _cand_stale(C, e):
    if e is None:
        return False
    if (EP / str(e) / "DECISION.json").exists():
        return True
    return _job_state(e) in ("done", "scored", "decided", "gave-up", "cancelled")


# ------------------------------------------------------------------ controller tick
def tick():
    """Called every controller tick in live mode. Never raises past the controller (caller wraps)."""
    out = dict(actions=[])
    try:
        ps = lium_ps()
    except Exception as ex:
        out["err"] = "lium ps: %s" % str(ex)[:120]
        return out
    live_ids = {p.get("id") for p in ps}
    nodes = len(eval_nodes(ps))
    to_release = []
    with locked() as W:
        P = W["pods"]
        now = time.time()
        for name, e in list(P.items()):
            st = e.get("state"); pod = e.get("pod") or {}; wa = worker_alive(e)
            if st == "claimed":
                lane = (e.get("claimed_by") or {}).get("pid")
                if not _alive(lane, "pe_lane"):
                    if pod.get("id") in live_ids and now - float((e.get("claimed_by") or {}).get("at") or now) > 300 and not wa:
                        to_release.append((name, pod.get("id"), "warm-claimed-lane-gone"))
                    elif pod.get("id") not in live_ids and not wa:
                        P.pop(name); out["actions"].append("drop-claimed:" + name)
                continue
            if st in ("failed", "released"):
                if not wa:
                    if pod.get("id") in live_ids:
                        to_release.append((name, pod.get("id"), "warm-failed"))
                    else:
                        P.pop(name); out["actions"].append("drop:" + name)
                continue
            if wa:
                continue
            if st in ("renting", "booting", "prefetching"):   # worker died without recording an outcome
                e.update(state="failed", why="worker gone in %s" % st); continue
            if pod.get("id") and pod["id"] not in live_ids:
                e.update(state="released", why="pod vanished"); continue
            if now - float(e.get("created") or now) > WARM_MAX_AGE_S:
                to_release.append((name, pod.get("id"), "warm-recycle-age")); continue
            if e.get("cand") and _cand_stale(e["cand"], e.get("epoch")):
                to_release.append((name, pod.get("id"), "warm-cand-resolved")); continue
        # assign idle ready nodes to waiting candidates (oldest epoch first)
        have = {e.get("cand") for e in P.values() if e.get("cand") and e.get("state") in UNCLAIMED + ("claimed",)}
        unassigned = []
        for ep, C, files in waiting_candidates():
            if C in have:
                continue
            free = [n for n, e in P.items() if e.get("state") == "ready" and not e.get("cand") and not worker_alive(e)]
            if free:
                n = sorted(free, key=lambda k: P[k].get("ready_at") or 0)[0]
                P[n].update(state="prefetching", cand=C, epoch=ep, worker=_spawn("prefetch", n, C, str(ep)))
                have.add(C); out["actions"].append("prefetch:%s->e%d" % (n, ep))
                ev(event="warm_prefetch_start", name=n, epoch=ep, candidate=C[:12])
            else:
                unassigned.append(ep)
        if pe_cfg().get("warm_compile", True) is not False:
            for n, e in P.items():   # upgrade ready nodes prepared before compile-warm existed (one per tick)
                if e.get("state") == "ready" and not e.get("cand") and not e.get("warm_phase") and not worker_alive(e):
                    e.update(warm_phase="compile-queued", worker=_spawn("compile", n)); out["actions"].append("compile:" + n); break
        # t1368u (Jacob 2026-11-07 13:20 CR): at most warm_max_total warm machines IN TOTAL (claimed by a lane or idle).
        # Over the cap: release idle, unassigned, worker-less warm nodes by exact id; lanes beyond it rent cold (cap 5 still holds).
        max_total = warm_max_total()
        live_warm = [n for n, e in P.items() if e.get("state") in UNCLAIMED + ("claimed",)]
        excess = len(live_warm) - max_total
        if excess > 0:
            idle = sorted((n for n in live_warm if P[n].get("state") in ("ready", "booting", "renting") and not P[n].get("cand")
                           and not worker_alive(P[n]) and (P[n].get("pod") or {}).get("id")),
                          key=lambda n: P[n].get("created") or 0)
            for n in idle[:excess]:
                to_release.append((n, P[n]["pod"]["id"], "warm-over-total-cap"))
                P[n]["state"] = "released"; excess -= 1
            live_warm = [n for n, e in P.items() if e.get("state") in UNCLAIMED + ("claimed",)]
        free_n = sum(1 for e in P.values() if e.get("state") in ("renting", "booting", "ready") and not e.get("cand"))
        renting = sum(1 for e in P.values() if e.get("state") == "renting")
        want = warm_target() + len(unassigned)
        if free_n < want and nodes + renting < MAX_EVAL_NODES and renting == 0 and len(live_warm) < max_total:
            n = "%swarm-%s" % (PREFIX, time.strftime("%m%d%H%M%S", time.gmtime()))
            P[n] = dict(name=n, state="renting", created=now, created_cr=cr(now), worker=_spawn("prep", n))
            out["actions"].append("prep:" + n); ev(event="warm_prep_start", name=n, nodes=nodes, free=free_n, want=want)
        out.update(nodes=nodes, free=free_n, want=want, warm_total=len(live_warm), warm_max_total=max_total, pods={k: (v.get("state"), (v.get("cand") or "")[:8]) for k, v in P.items()})
    for name, pid, why in to_release:
        ok = pe_rent.release(pid, reason=why) if pid else True
        with locked() as W:
            e = W["pods"].get(name)
            if e is not None:
                if ok:
                    W["pods"].pop(name)
                else:
                    e["release_err"] = cr()
        out["actions"].append("release:%s:%s:%s" % (name, why, ok))
    return out


# ------------------------------------------------------------------ lane side
PREFETCH_WAIT_MAX_S = 600


def acquire(job, beat=None, generic_only=False):
    """Lane: claim the best warm node for job's candidate. -> dict(pod, name, fetched, served, claim_wait_s) or None.
    v5 (ready-only, flag parallel_eval.warm_ready_only, default on): only nodes whose prep is DONE are claimable -- a node still
    renting/booting is never claimed (E862 waited ~11 min on one); the caller rents cold instead. Claimable: served/prefetching THIS
    candidate (a prefetch in flight is bounded by PREFETCH_WAIT_MAX_S), or an idle ready node; a ready node still compile-warming the
    head gets that compile worker stopped by exact PID (compile is best-effort: SIGTERM -> compile-aborted, node stays prepared).
    generic_only: only idle ready nodes with no candidate (2-node shard helper; never steals another candidate's prefetch)."""
    C = job["candidate"]; t0 = time.time()
    ready_only = pe_cfg().get("warm_ready_only", True) is not False or generic_only
    with locked() as W:
        P = W["pods"]
        def rank(e):
            if e.get("state") not in UNCLAIMED:
                return None
            if e.get("cand") == C and not generic_only:
                if ready_only and not e.get("prep_ok"):
                    return None
                return 0
            if e.get("cand"):
                return None
            if e.get("state") == "ready":
                if ready_only and not e.get("prep_ok"):
                    return None
                return 1 if e.get("warm_phase") in ("compiled", "compile-aborted") and not worker_alive(e) else 1.5
            if ready_only:
                return None
            return {"booting": 2, "renting": 3}.get(e.get("state"))
        r = sorted(((rank(e), e.get("created") or 0, n) for n, e in P.items() if rank(e) is not None))
        if not r:
            return None
        name = r[0][2]; e = P[name]
        e["pre_claim_state"] = e.get("state")
        e.update(state="claimed", claimed_by=dict(job=job.get("id"), attempt=job.get("attempt"), pid=os.getpid(), at=time.time(), at_cr=cr(),
                                                  shard=bool(generic_only)))
        stop_compile = None
        if ready_only and e.get("pre_claim_state") == "ready" and worker_alive(e) and e.get("prep_ok"):
            stop_compile = (e.get("worker") or {}).get("pid")
    ev(event="warm_claimed", name=name, job=job.get("id"), attempt=job.get("attempt"), pre=e.get("pre_claim_state"), cand_match=e.get("cand") == C,
       ready_only=ready_only, generic_only=generic_only, stop_compile=stop_compile)
    if stop_compile and _alive(stop_compile, "pe_warm"):
        os.kill(int(stop_compile), signal.SIGTERM)   # exact PID; compile_warm catches it -> compile-aborted, node stays prepared
    max_wait = PREFETCH_WAIT_MAX_S if ready_only else WAIT_WORKER_MAX_S
    while True:
        e = snapshot().get(name) or {}
        if not worker_alive(e):
            break
        if beat:
            beat(status="warm-wait", warm=name)
        if time.time() - t0 > max_wait:
            w = e.get("worker") or {}
            if _alive(w.get("pid"), "pe_warm"):
                os.kill(int(w["pid"]), signal.SIGTERM)   # exact PID; worker failure path releases its pod
            time.sleep(15)
            break
        time.sleep(10)
    e = snapshot().get(name) or {}
    pod = e.get("pod") or {}
    if not e.get("prep_ok") or not pod.get("id"):
        ev(event="warm_claim_unusable", name=name, job=job.get("id"), why=e.get("why"))
        if pod.get("id"):
            pe_rent.release(pod["id"], reason="warm-claim-unusable")
        with locked() as W:
            W["pods"].pop(name, None)
        return None
    if e.get("cand") == C and e.get("prefetch_ok") is False:
        ev(event="warm_claim_prefetch_failed", name=name, job=job.get("id"), why=e.get("why"))
        if pod.get("id"):
            pe_rent.release(pod["id"], reason="warm-prefetch-failed")
        with locked() as W:
            W["pods"].pop(name, None)
        return None
    return dict(pod=pod, name=name, fetched=e.get("fetched") == C, served=e.get("served") == C,
                claim_wait_s=round(time.time() - t0, 1), prep_s=e.get("prep_s"), fetch_s=e.get("fetch_s"), serve_s=e.get("serve_s"))


# ------------------------------------------------------------------ workers
def _set(name, **kw):
    with locked() as W:
        e = W["pods"].setdefault(name, dict(name=name))
        if e.get("state") == "claimed" and "state" in kw:
            kw = dict(kw); kw.pop("state")   # a lane owns it now: record outcome only
        e.update(kw)


def _sigterm(signum, frame):
    raise SystemExit("warm worker got signal %d" % signum)


def work_prep(name):
    import pe_lane
    signal.signal(signal.SIGTERM, _sigterm); signal.signal(signal.SIGHUP, _sigterm)
    t0 = time.time(); d = None; ok = False
    try:
        d = pe_rent.rent(name, ttl=WARM_TTL, max_eval_nodes=MAX_EVAL_NODES)
        if not d:
            _set(name, state="failed", why="rent failed / no supply / cap", prep_ok=False); return
        _set(name, state="booting", pod=d, gpu=d.get("gpu"), price_h=d.get("price_h"))
        if "5090" in str(d.get("gpu") or "").upper():
            raise ValueError("5090 eval nodes are forbidden")
        P = pod_of(d)
        for _ in range(30):
            try:
                P.run("echo ssh-ok", timeout=30); break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("ssh never ready")
        pe_lane.boot(P); pe_lane.push_bundle(P)
        ok = True
        _set(name, state="ready", prep_ok=True, ready_at=time.time(), ready_at_cr=cr(), prep_s=round(time.time() - t0, 1))
        ev(event="warm_ready", name=name, gpu=d.get("gpu"), price_h=d.get("price_h"), prep_s=round(time.time() - t0, 1))
    except pe_rent.CapReached as ex:
        _set(name, state="failed", prep_ok=False, why="cap: %s" % ex)
    except BaseException as ex:
        _set(name, state="failed", prep_ok=False, why=("%s: %s" % (type(ex).__name__, ex))[:300])
        ev(event="warm_prep_failed", name=name, err=str(ex)[:300])
    finally:
        if not ok:
            if d and d.get("id"):
                pe_rent.release(d["id"], reason="lane-failed-warm-prep")
            else:
                pe_rent.release(name, reason="warm-prep-noid")
            _set(name, state="released")
    if ok and pe_cfg().get("warm_compile", True) is not False:
        compile_warm(name, d)


def compile_warm(name, d):
    """Prime this node's per-replica vLLM/inductor/triton compile caches (serve.sh vcache/r<gpu>) by serving the PUBLIC live head
    once with the identical public serve path (pe_lane.serve: MAX_SEQS 512, bf16, 69632, parity-checked). The lane's later
    `podctl serve <candidate>` stops it and starts the candidate; measured 95.8 s warm vs 267 s cold on the baseline pod.
    Best effort: failure / SIGTERM here never releases a prepared node (prep_ok already true); the lane serves cold instead."""
    import pe_lane
    t0 = time.time()
    try:
        H = live_head()
        if not H:
            raise RuntimeError("no live head")
        _set(name, warm_phase="compile", compile_head=H)
        P = pod_of(d)
        pe_lane.fetch_ckpt(P, H)
        si = pe_lane.serve(P, H, max_num_seqs=512)
        _set(name, warm_phase="compiled", compile_s=round(time.time() - t0, 1), head_serve_s=si.get("serve_s"))
        ev(event="warm_compiled", name=name, head=H[:12], seconds=round(time.time() - t0, 1), serve_s=si.get("serve_s"))
    except BaseException as ex:
        _set(name, warm_phase="compile-aborted", compile_err=("%s: %s" % (type(ex).__name__, ex))[:200])
        ev(event="warm_compile_aborted", name=name, err=str(ex)[:200])


def work_prefetch(name, C, epoch):
    import pe_lane
    signal.signal(signal.SIGTERM, _sigterm); signal.signal(signal.SIGHUP, _sigterm)
    e = snapshot().get(name) or {}
    d = e.get("pod") or {}
    ok = False; t0 = time.time()
    try:
        pend = rj(PENDING / ("e%s.json" % epoch)) or {}
        if pend.get("candidate") != C:
            raise RuntimeError("pending e%s candidate mismatch" % epoch)
        files = (pend.get("checkpoint") or {}).get("files")
        P = pod_of(d)
        fi = pe_lane.fetch_ckpt(P, C, files=files)
        _set(name, fetched=C, fetch_s=fi.get("seconds"))
        si = pe_lane.serve(P, C, max_num_seqs=512)
        ok = True
        _set(name, state="served", served=C, serve_s=si.get("serve_s"), prefetch_ok=True, served_at=time.time(), served_at_cr=cr())
        ev(event="warm_prefetched", name=name, epoch=epoch, candidate=C[:12], fetch_s=fi.get("seconds"), serve_s=si.get("serve_s"),
           total_s=round(time.time() - t0, 1))
    except BaseException as ex:
        _set(name, state="failed", prefetch_ok=False, why=("%s: %s" % (type(ex).__name__, ex))[:300])
        ev(event="warm_prefetch_failed", name=name, epoch=epoch, err=str(ex)[:300])
    finally:
        if not ok:
            cur = snapshot().get(name) or {}
            if cur.get("state") != "claimed" and d.get("id"):   # claimed: the lane releases it
                pe_rent.release(d["id"], reason="lane-failed-warm-prefetch")
                _set(name, state="released")


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else "status"
    if k == "prep":
        work_prep(sys.argv[2])
    elif k == "compile":
        signal.signal(signal.SIGTERM, _sigterm); signal.signal(signal.SIGHUP, _sigterm)
        compile_warm(sys.argv[2], (snapshot().get(sys.argv[2]) or {}).get("pod") or {})
    elif k == "prefetch":
        work_prefetch(sys.argv[2], sys.argv[3], sys.argv[4])
    elif k == "status":
        for n, e in snapshot().items():
            print(n, e.get("state"), (e.get("cand") or "")[:12], e.get("gpu"), e.get("price_h"), e.get("prep_s"), e.get("fetch_s"), e.get("serve_s"))
