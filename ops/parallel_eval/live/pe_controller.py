#!/usr/bin/env python3
"""PE controller v4-throughput (2026-11-07: B300-first rentals, warm pool + checkpoint prefetch via pe_warm; v2 2026-11-06): per-candidate eval lanes with watchdog + bounded automatic re-run.
- Each job = immutable spec (JOB.json: candidate, head, round_id, indices, plan_seed, k, max_tokens) bound at DISPATCH time to
  the round (draw + head baseline) that was current for its head; a later head move never changes an in-flight job's draw.
- Each attempt = one lane process (pe_lane.py) on one fresh affine-pe-* pod; the lane always releases its pod (finally + SIGTERM).
- Watchdog (every TICK): lane process dead before done, no heartbeat/progress for too long, or pod gone from `lium ps`
  -> kill lane by exact PID, release pod by exact id, schedule the next attempt (same draw/indices/plan seed) after backoff.
- max_attempts (default 3); exhausted -> job 'gave-up' (live: infra no_decision, turn not served).
- shadow jobs are only created explicitly: `pe_controller.py submit-shadow <epoch>`; live dispatch only when parallel_eval.enabled.
- wiring-v1 (2026-11-07): live mode also runs pe_live.tick every TICK: RESTORE-REQUEST consumer -> TRAINER-AT, pending ->
  live job dispatch, GREEN -> apply records (fresh) / merge-blocked / stale_overflow, consumed apply -> HEAD-CHAIN.
"""
import json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
import pe_rent

CS = ST / "CONTROLLER.json"
LANE = CODE / ("pe_lane_fake.py" if os.environ.get("PE_FAKE_LANE") else "pe_lane.py")
TICK = 20
MAX_ATTEMPTS = 3
BACKOFF_S = 60
STALL = dict(starting=300, renting=1500, booting=1800, fetching=3000, serving=1500, probing=1500)  # "warm-wait" beats every poll -> default 1800  # s without heartbeat/progress
POD_CHECK_S = 60
TERMINAL = ("scored", "decided", "gave-up", "cancelled")


def pid_alive(pid):
    try:
        pid = int(pid)
        os.kill(pid, 0)
        return "pe_lane" in Path("/proc/%d/cmdline" % pid).read_text(errors="ignore")
    except Exception:
        return False


def jdir(jid):
    return JOBS / jid


def load(jid):
    jd = jdir(jid)
    return rj(jd / "JOB.json"), rj(jd / "STATUS.json") or {}, rj(jd / "CTL.json") or dict(attempts=[], state="new")


def save_ctl(jid, c):
    wj(jdir(jid) / "CTL.json", c)


def launch_attempt(jid, c):
    jd = jdir(jid)
    spec = rj(jd / "JOB.json")
    n = len(c["attempts"])
    att = dict(spec, attempt=n, dir=str(jd), run_name="pe-%s-a%d-%s" % (jid, n, spec["candidate"][:12]))
    ap = jd / ("ATTEMPT-%d.json" % n)
    wj(ap, att)
    lg = open(jd / ("lane-a%d.log" % n), "ab")
    p = subprocess.Popen([HOST_PY, "-u", "-B", str(LANE), str(ap)], stdout=lg, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(CODE))
    c["attempts"].append(dict(n=n, pid=p.pid, launched=time.time(), launched_cr=cr()))
    c["state"] = "running"
    save_ctl(jid, c)
    ev(event="attempt_launched", job=jid, attempt=n, pid=p.pid)


def kill_lane(pid):
    """Exact PID only: SIGTERM (lane finally releases its pod), SIGKILL after 90 s."""
    if not pid_alive(pid):
        return
    os.kill(int(pid), signal.SIGTERM)
    for _ in range(45):
        time.sleep(2)
        if not pid_alive(pid):
            return
    os.kill(int(pid), signal.SIGKILL)


def ensure_released(pod):
    if not pod or not pod.get("id"):
        return None
    try:
        ids = {p.get("id") for p in our_pods()}
    except Exception:
        return None
    if pod["id"] in ids:
        return pe_rent.release(pod["id"], reason="controller-cleanup")
    return True


_pod_seen = {}


def watch(jid, spec, st, c):
    """-> None (healthy / waiting) or failure reason string."""
    a = c["attempts"][-1]
    alive = pid_alive(a["pid"])
    s = st.get("status")
    if st.get("attempt") != a["n"]:   # lane hasn't written its first STATUS yet
        return None if (alive or time.time() - a["launched"] < 60) else "lane died before start"
    if s == "done":
        return None
    if not alive:
        return "lane process gone (status=%s err=%s)" % (s, (st.get("error") or "")[:160])
    last = max(float(st.get("beat") or 0), float(st.get("progress_at") or 0)) if s != "probing" else float(st.get("progress_at") or st.get("beat") or 0)
    lim = STALL.get(s, 1800)
    if last and time.time() - last > lim:
        return "stalled in %s for %ds" % (s, int(time.time() - last))
    pod = st.get("pod") or {}
    if pod.get("id") and time.time() - _pod_seen.get(jid, 0) > POD_CHECK_S:
        _pod_seen[jid] = time.time()
        try:
            ids = {p.get("id") for p in our_pods()}
            if pod["id"] not in ids:
                return "pod %s gone from lium ps" % pod.get("huid")
        except Exception:
            pass
    return None


def maybe_rerun(jid, spec, c):
    """wiring-v1: gate-side probe-error guard asked for ONE re-probe of a live job (same spec/draw/seed)."""
    jd = jdir(jid); rr = rj(jd / "RERUN.json")
    if not rr or rr.get("handled"):
        return False
    n = len(c["attempts"])
    for name in ("RESULT.json", "META.json"):
        if (jd / name).exists():
            os.replace(jd / name, jd / ("%s.rerun-a%d" % (name, n)))
    rr["handled"] = time.time(); wj(jd / "RERUN.json", rr)
    spec["max_attempts"] = int(spec.get("max_attempts") or MAX_ATTEMPTS) + 1; wj(jd / "JOB.json", spec)
    c["state"] = "new"; save_ctl(jid, c)
    ev(event="live_rerun", job=jid, why=rr.get("why"))
    return True


def running_lanes():
    n = 0
    for jd in JOBS.glob("*"):
        c = rj(jd / "CTL.json") or {}
        if c.get("state") == "running" and c.get("attempts") and pid_alive(c["attempts"][-1].get("pid")):
            n += 1
    return n


def tick_job(jid):
    spec, st, c = load(jid)
    if spec and spec.get("kind") == "live" and c.get("state") in ("done", "scored") and maybe_rerun(jid, spec, c):
        return
    if not spec or c.get("state") in TERMINAL:
        return
    if c["state"] == "new":
        if running_lanes() >= MAX_EVAL_NODES:
            return   # t1365u: at most five eval nodes at a time
        launch_attempt(jid, c); return
    if c["state"] == "backoff":
        if time.time() >= c.get("retry_at", 0) and running_lanes() < MAX_EVAL_NODES:
            launch_attempt(jid, c)
        return
    if c["state"] == "running":
        if st.get("status") == "done" and (jdir(jid) / "RESULT.json").exists() and st.get("attempt") == c["attempts"][-1]["n"]:
            c["state"] = "done"; c["done_at_cr"] = cr(); save_ctl(jid, c)
            ev(event="job_done", job=jid, attempts=len(c["attempts"]))
            return
        why = watch(jid, spec, st, c)
        if why is None:
            return
        a = c["attempts"][-1]
        kill_lane(a["pid"])
        st2 = rj(jdir(jid) / "STATUS.json") or {}
        rel = ensure_released(st2.get("pod") or st.get("pod"))
        a.update(failed=why, failed_at=time.time(), failed_at_cr=cr(), pod=(st2.get("pod") or st.get("pod") or {}).get("huid"), released=rel)
        ev(event="attempt_failed", job=jid, attempt=a["n"], why=why, released=rel)
        if len(c["attempts"]) >= int(spec.get("max_attempts") or MAX_ATTEMPTS):
            c["state"] = "gave-up"; c["gave_up_cr"] = cr()
            ev(event="job_gave_up", job=jid, attempts=len(c["attempts"]))
        else:
            c["state"] = "backoff"; c["retry_at"] = time.time() + BACKOFF_S * len(c["attempts"])
            ev(event="retry_scheduled", job=jid, next_attempt=len(c["attempts"]), at_cr=cr(c["retry_at"]))
        save_ctl(jid, c)
        return
    if c["state"] == "done":
        score_job(jid, spec, c)


def score_job(jid, spec, c):
    import pe_score
    jd = jdir(jid)
    if spec.get("kind") == "shadow":
        out = pe_score.score_epoch_shadow(spec["epoch"], jd / "RESULT.json", jd / "SCORE.json")
        out["attempts"] = len(c["attempts"])
        wj(jd / "SCORE.json", out)
        c["state"] = "scored"; save_ctl(jid, c)
        ev(event="shadow_scored", job=jid, agree=out["agree"], pe_z=out["pe_z"], live_z=out["live_z"], z_delta=out["z_delta"], attempts=len(c["attempts"]))
        return
    try:
        dec = pe_score.score(spec["epoch"], spec["round_id"], rj(jd / "RESULT.json"))
    except pe_score.Pending as ex:
        if not c.get("pending_logged"):
            c["pending_logged"] = cr(); save_ctl(jid, c); ev(event="score_pending", job=jid, why=str(ex))
        return
    wj(jd / "PE-DECISION.private.json", dec)
    c["state"] = "scored"; save_ctl(jid, c)
    ev(event="pe_scored", job=jid, decision=dec.get("decision"), z=(dec.get("evaluated") or {}).get("z"))


def submit(spec):
    jid = spec["id"]
    jd = jdir(jid)
    if (jd / "JOB.json").exists():
        raise SystemExit("job exists: %s" % jid)
    jd.mkdir(parents=True, mode=0o700)
    spec.setdefault("max_attempts", MAX_ATTEMPTS)
    spec["submitted_cr"] = cr()
    wj(jd / "JOB.json", spec)
    save_ctl(jid, dict(attempts=[], state="new"))
    ev(event="job_submitted", job=jid, kind=spec.get("kind"), epoch=spec.get("epoch"))
    return jid


def submit_live(spec):
    if (jdir(spec["id"]) / "JOB.json").exists():
        return spec["id"]
    return submit(spec)


def shadow_spec(e, tag=""):
    """Exact live replica of epoch e's candidate probe: same round draw (indices), same plan seed cand_seed(round seed, e)."""
    sys.path.insert(0, str(GATE_CODE))
    import baseline_v1 as BV
    d = EP / str(e)
    dec = rj(d / "DECISION.json"); sm = rj(d / "SAMPLE.private.json"); tl = rj(d / "TIMELINE.json") or {}
    r = rj(HELDOUT / sm["round_id"] / "ROUND.private.json")
    assert list(r["indices"]) == list(sm["indices"]), "round draw != epoch sample"
    return dict(id="shadow-e%d%s" % (e, tag), kind="shadow", epoch=e, candidate=dec["candidate"], head=dec["head"],
                round_id=sm["round_id"], indices=list(r["indices"]), plan_seed=BV.cand_seed(r["seed"], e),
                k=int(r["cfg"]["k_cand"]), max_tokens=int(PG_mt(r["cfg"])), env_id=r["cfg"]["probe_env"], files=tl.get("files"),
                live_decision=dec.get("decision"), live_z=(dec.get("evaluated") or {}).get("z"))


def PG_mt(pv):
    sys.path.insert(0, str(GATE_CODE))
    import prior_gate as PG
    return PG._mt(pv)


def active_jobs():
    out = []
    for jd in sorted(JOBS.glob("*")):
        c = rj(jd / "CTL.json")
        if c and (c.get("state") not in TERMINAL or (c.get("state") == "scored" and (jd / "RERUN.json").exists()
                                                      and not (rj(jd / "RERUN.json") or {}).get("handled"))):
            out.append(jd.name)
    return out


def main():
    ST.mkdir(parents=True, exist_ok=True)
    (ST / "CONTROLLER.pid").write_text(str(os.getpid()))
    S = dict(pid=os.getpid(), started_cr=cr(), version="pe-controller-v4-throughput")
    wj(CS, S)
    ev(event="controller_start", pid=os.getpid(), version="v4-throughput", mode=mode(), gpu_order=list(gpu_order()), warm_target=warm_target())
    while True:
        if (ST / "KILL").exists() or (ST / "STOP").exists():
            ev(event="controller_stop"); break
        if mode() == "live":
            try:
                import pe_live
                S["live"] = pe_live.tick(submit_live)
            except Exception as ex:
                ev(event="pe_live_tick_error", err=str(ex)[:300]); traceback.print_exc()
        for jid in active_jobs():
            try:
                tick_job(jid)
            except Exception as ex:
                ev(event="tick_job_error", job=jid, err=str(ex)[:300]); traceback.print_exc()
        if mode() == "live" and not (ST / "WARM-OFF").exists():
            try:   # throughput-v4 warm pool + checkpoint prefetch (state/WARM-OFF disables without restart)
                import pe_warm
                S["warm"] = pe_warm.tick()
            except Exception as ex:
                ev(event="warm_tick_error", err=str(ex)[:300]); traceback.print_exc()
        if mode() == "live":
            try:   # protect every PE candidate's fp32 master store entry from trainer GC until RED or published (pe_fp32pin)
                import pe_fp32pin
                r = pe_fp32pin.tick()
                if r is not None:
                    S["fp32pin"] = dict(pinned=len(r.get("pinned") or {}), missing=len(r.get("missing") or {}), at_cr=cr())
            except Exception as ex:
                ev(event="fp32pin_tick_error", err=str(ex)[:300]); traceback.print_exc()
        S["last_tick_cr"] = cr(); S["active"] = active_jobs(); wj(CS, S)
        time.sleep(TICK)


if __name__ == "__main__":
    if sys.argv[1:2] == ["submit-shadow"]:
        e = int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else ""
        sp = shadow_spec(e, tag)
        if len(sys.argv) > 4:
            sp["gpu"] = sys.argv[4]
        print(submit(sp))
    elif sys.argv[1:2] == ["status"]:
        for jd in sorted(JOBS.glob("*")):
            c = rj(jd / "CTL.json"); s = rj(jd / "STATUS.json") or {}
            if c: print(jd.name, c.get("state"), len(c.get("attempts") or []), s.get("status"), (s.get("progress") or {}))
    else:
        main()
