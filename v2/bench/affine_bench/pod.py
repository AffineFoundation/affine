"""Bench pod lifecycle on Lium: rent on demand when a king is queued, release when idle (Jacob pt1079u-era decision,
2026-11-02: bench only new GREEN kings; ~1 king per 1-3 h and ~30 min per king, so an idle $30/h pod costs more than
re-renting). Runs on the host; the Lium key stays in the operator's Lium config and is never printed or logged.

Guard rails (owner's spending rules): one bench pod at most, always named NAME; never above BENCH_MAX_PRICE $/h; at most
MAX_RENTS_PER_DAY rents per 24 h; release touches only the exact Lium id this module rented (recorded in RECORD) and only
if Lium still lists it under NAME. Other pods (probe, trainer, evaluator, perf-eval, Tobias's) are never read or touched
beyond the account-wide `lium ps` listing."""
import json, os, subprocess, time
from . import config as C

LIUM = C.LIUM; MAX_PRICE = C.MAX_PRICE
NAME = os.environ.get("BENCH_POD_NAME", "affine-v2-bench-1")  # not the v1 name: a pod left over from v1 is never ours
GPU_PREF = os.environ.get("BENCH_GPU_PREF", "B300:8 B200:8 H200:8 H100:8")  # rent.sh walks this best-first
NO_OFFER_RETRY_S = float(os.environ.get("BENCH_NO_OFFER_RETRY_S", "300"))
MAX_RENTS_PER_DAY = int(os.environ.get("BENCH_MAX_RENTS_PER_DAY", "8"))
IDLE_RELEASE_S = float(os.environ.get("BENCH_IDLE_RELEASE_S", "900"))
RECORD = C.STATE + "/pod.json"                      # the live pod: the only pod this service ever touches
RENT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "rent.sh")


def _rj(p):
    try: return json.load(open(p))
    except Exception: return None


def load():
    """Current pod as {name, id, huid, gpu, cost} or {} when there is none."""
    r = _rj(RECORD) or {}
    return r if r.get("id") else {}


def save(r):
    t = RECORD + ".tmp"; json.dump(r, open(t, "w"), indent=1); os.replace(t, RECORD)


def lium(*args, timeout=120):
    return subprocess.run([LIUM, *args], capture_output=True, text=True, timeout=timeout)


def listed():
    """Account pods from `lium ps` (None if Lium could not be asked)."""
    try:
        p = lium("ps", "--format", "json", timeout=60)
        d = json.loads(p.stdout)
        return d if isinstance(d, list) else None
    except Exception:
        return None


def state(pod_id):
    """'gone' if Lium says the pod no longer exists, 'listed' if it does, None if unknown."""
    if not pod_id:
        return None
    try:
        p = lium("describe", pod_id, timeout=60)
    except Exception:
        return None
    o = p.stdout + p.stderr
    if "no longer listed" in o or "DELETED" in o:
        return "gone"
    return "listed" if p.returncode == 0 else None


def rent(S, log):
    """Rent + bootstrap one bench pod. Returns the record or None. Blocking (~5-10 min)."""
    S.setdefault("rents", [])
    S["rents"] = [t for t in S["rents"] if time.time() - t < 86400]
    if len(S["rents"]) >= MAX_RENTS_PER_DAY:
        log(event="rent-refused", why=f"{len(S['rents'])} rents in 24 h (cap {MAX_RENTS_PER_DAY})"); return None
    if time.time() - S.get("no_offer_at", 0) < NO_OFFER_RETRY_S:
        return None  # Lium had no eligible node a moment ago: ask again after NO_OFFER_RETRY_S, without logging every step
    pods = listed()
    if pods is None:
        log(event="rent-deferred", why="lium ps failed"); return None
    mine = [p for p in pods if p.get("name") == NAME]
    if mine:  # never a second bench pod: an unknown pod under our name is a human's call
        log(event="rent-refused", why=f"a pod named {NAME} already exists", pods=[p.get("huid") for p in mine]); return None
    # Only a pod that was actually created counts toward MAX_RENTS_PER_DAY (2026-11-03: four "no offer" failures in two
    # minutes filled the 8/24 h cap and blocked the bench for 7 h).
    p = subprocess.run(["bash", RENT], capture_output=True, text=True, timeout=1500,
                       env=dict(os.environ, MAX_PRICE=str(MAX_PRICE), NAME=NAME, LIUM=LIUM, GPU_PREF=GPU_PREF, NODE_SSH=C.NODE_SSH))
    last = (p.stdout.strip().splitlines() or [""])[-1]
    try:
        info = json.loads(last)
    except Exception:
        info = None
    if p.returncode == 3 or (isinstance(info, dict) and info.get("no_offer")):
        log(event="rent-no-offer", max_usd_h=MAX_PRICE, gpu_pref=GPU_PREF, seen=(info or {}).get("seen") if isinstance(info, dict) else None)
        S["no_offer_at"] = time.time(); return None
    if not isinstance(info, dict) or not info.get("id"):
        made = [q for q in (listed() or []) if q.get("name") == NAME]
        if made:  # `lium up` created the pod but the script failed after: it is a real rent; leave it for the next step's guard
            S["rents"].append(time.time())
        log(event="rent-failed", rc=p.returncode, pod_created=bool(made), err=(p.stderr or p.stdout)[-300:])
        return None
    S.pop("no_offer_at", None)
    S["rents"].append(time.time())
    rec = dict(name=info.get("name"), id=info.get("id"), huid=info.get("huid"), gpu=(info.get("config") or "").replace("×", "x"),
               cost=info.get("price_per_hour"), rented_at=time.time(), bootstrapped=False)
    save(rec); log(event="rented", pod=rec["huid"], lium_id=rec["id"], gpu=rec["gpu"], usd_h=rec["cost"])
    if (rec["cost"] or 0) > MAX_PRICE:  # rent.sh filters on price; this is the belt to its braces
        log(event="rent-over-cap", usd_h=rec["cost"]); release(log, why="over price cap"); return None
    b = subprocess.run([C.NODE_SSH, f"mkdir -p {C.NODE_ROOT}/logs && cd {C.NODE_ROOT} && (test -d repo || git clone -q {C.REPO_URL} repo) && git -C repo checkout -q --detach {C.REPO_REF} "
                        f"&& bash {C.NODE_REPO}/bench/deploy/bootstrap.sh > {C.NODE_ROOT}/logs/bootstrap.log 2>&1; tail -1 {C.NODE_ROOT}/logs/bootstrap.log"],
                       capture_output=True, text=True, timeout=2400)
    if "BOOTSTRAP_OK" not in b.stdout:
        log(event="bootstrap-failed", pod=rec["huid"], out=(b.stdout + b.stderr)[-300:]); release(log, why="bootstrap failed"); return None
    rec["bootstrapped"] = True; save(rec); log(event="bootstrapped", pod=rec["huid"], minutes=round((time.time() - rec["rented_at"]) / 60, 1))
    return rec


def release(log, why):
    """Terminate the recorded bench pod (exact Lium id, and only if Lium lists it under NAME)."""
    rec = _rj(RECORD) or {}
    pid = rec.get("id")
    if not pid:
        return True
    pods = listed()
    if pods is None:
        log(event="release-deferred", why="lium ps failed"); return False
    hit = [p for p in pods if p.get("id") == pid]
    if hit and hit[0].get("name") != NAME:
        log(event="release-refused", why=f"pod {pid} is named {hit[0].get('name')!r}, not {NAME!r}"); return False
    if hit:
        r = lium("rm", pid, "--name-only", "-y", timeout=180)
        if r.returncode != 0:
            log(event="release-failed", pod=rec.get("huid"), err=(r.stderr or r.stdout)[-200:]); return False
    save(dict(released=time.time(), last=rec))
    log(event="released", pod=rec.get("huid"), lium_id=pid, why=why,
        hours=round((time.time() - (rec.get("rented_at") or time.time())) / 3600, 2)); return True
