#!/usr/bin/env python3
"""Release guard: wait for watcher PID to exit (or deadline), salvage result if missing, ALWAYS release pod by exact id.
usage: pe_guard.py <job_dir> <pod_id> <watch_pid> <deadline_unix>"""
import json, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
import pe_rent

def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False

jd, pid_id, wpid, dl = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
ev(event="guard_start", job=jd.name, pod=pid_id, watch=wpid, deadline_cr=cr(dl))
try:
    while alive(wpid) and time.time() < dl:
        time.sleep(20)
    time.sleep(30)  # let watcher finish its own release/score
    job = rj(jd / "JOB.json") or {}
    st = rj(jd / "STATUS.json") or {}
    if not (jd / "RESULT.json").exists() and st.get("pod"):
        try:
            ps = {p.get("id"): p for p in our_pods()}
            if pid_id in ps:
                d = pe_rent._from_ps(ps[pid_id]); P = pod_of(d)
                rn = job.get("run_name")
                raw = P.run("cat %s/runs/%s/result.json 2>/dev/null || echo NORESULT" % (PODD, rn), timeout=600, check=False)
                if raw and not raw.strip().startswith("NORESULT"):
                    wj(jd / "RESULT.json", json.loads(raw), 0o600)
                    ev(event="guard_salvaged", job=jd.name)
                else:
                    ev(event="guard_no_result", job=jd.name, watcher_alive=alive(wpid))
        except Exception as ex:
            ev(event="guard_salvage_err", job=jd.name, err=str(ex)[:300])
finally:
    ok = False
    for a in range(5):
        try:
            if pid_id not in {p.get("id") for p in our_pods()}:
                ok = True; ev(event="guard_pod_gone", pod=pid_id); break
            ok = pe_rent.release(pid_id, reason="guard-%s" % jd.name)
            if ok: break
        except Exception as ex:
            ev(event="guard_release_err", pod=pid_id, err=str(ex)[:200])
        time.sleep(30)
    print("guard done released_or_gone=%s" % ok, flush=True)
