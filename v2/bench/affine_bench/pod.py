"""Bench pods on Lium: one pod for each row, rented when the row has work and given back when the row is done
(Jacob, 2026-11-14: the rows run side by side; before, one pod ran the rows one after the other). Runs on the host;
the Lium key stays in the operator's Lium config and is never printed or logged.

A Pod is the pod of one row: its name on Lium, its record on this host and its ssh script. Each row has a worker
process of its own (daemon.py) that uses the Pod of its row only.

Guard rails (owner's spending rules):
  - one pod at most for a row, always named `affine-v2-bench-<model_id>`. A pod that Lium lists under that exact name
    and that the row has no record of (an answer of Lium was lost) is taken as the row's pod: never a second one.
  - secure tier only, a whole node of a kind in GPU_PREF, never above BENCH_MAX_PRICE $/h: the cheapest such offer.
  - a failed rent is tried RENT_TRIES times with a growing wait; then the row says `rent-failed` and its worker ends.
  - at most MAX_PODS rows have a pod at a time (daemon.py), and at most MAX_RENTS_PER_DAY pods are made in 24 h, over all rows.
  - a pod lives POD_MAX_HOURS at most (daemon.py).
  - Pod.release() is the only place that sends a delete to Lium. It sends it for the exact Lium id in the row's record,
    only while Lium lists that id under the row's exact name, and never for a protected pod (PROTECTED_IDS,
    PROTECTED_NAMES, HELD_FOR_ROW). No delete is ever made from a name, a prefix or a listing.
  - a listing that does not have a pod is no proof that the pod is gone (a listing can be empty for a moment). A record
    is dropped, or says `released`, only after a delete that Lium took, or when Lium says of that exact id that it was
    deleted (state()). Until then the record stays and the row asks again.
  - the row of HELD_FOR_ROW rents nothing while its pod is at Lium or in the record of the single-node service."""
import fcntl, glob, json, os, re, subprocess, time
from . import config as C

LIUM = C.LIUM; MAX_PRICE = C.MAX_PRICE
PREFIX = "affine-v2-bench-"
NAME_MAX = 29  # the longest pod name that is known to work on Lium
NAME = os.environ.get("BENCH_POD_NAME", PREFIX + "1")  # the one pod of the service before 2026-11-14; the row that ran on it keeps it
# Whole 8-GPU nodes that run a row in about the same time. The cheapest offer at or below the cap is taken; at one
# price the earlier kind of this list. (Before 2026-11-14 the list was walked best-first, so a dearer kind could win.)
GPU_PREF = os.environ.get("BENCH_GPU_PREF", "H100:8 H200:8 B200:8 B300:8")
NO_OFFER_RETRY_S = float(os.environ.get("BENCH_NO_OFFER_RETRY_S", "300"))
RENT_TRIES = int(os.environ.get("BENCH_RENT_TRIES", "4")); RENT_WAIT_S = float(os.environ.get("BENCH_RENT_WAIT_S", "120"))
MAX_RENTS_PER_DAY = int(os.environ.get("BENCH_MAX_RENTS_PER_DAY", "16"))
MAX_PODS = int(os.environ.get("BENCH_MAX_PODS", "8"))
POD_MAX_HOURS = float(os.environ.get("BENCH_POD_MAX_HOURS", "6"))
CLAIM_S = 900.0  # an offer that a row has chosen is left to that row this long
RENT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "rent.sh")
# Never deleted by this service, whatever a record says: the prod GPU box and the operator's own pod (Jacob, 2026-11-14).
PROTECTED_IDS = ("6cb913e4", "c44cb565")
PROTECTED_NAMES = ("sn120-v2-prod-dublin", "casper-h100-ask-before-delete")
# A pod that only the worker of one row gives back, and only when that row's results are in the published table.
HELD_FOR_ROW = {"296c5237": "king-10757e879efe5b91"}


def _rj(p):
    try: return json.load(open(p))
    except Exception: return None


def lium(*args, timeout=120):
    return subprocess.run([LIUM, *args], capture_output=True, text=True, timeout=timeout)


def listed():
    """Account pods from `lium ps` (None if Lium could not be asked, or if its answer is not a list of pods)."""
    try:
        p = lium("ps", "--format", "json", timeout=60)
        d = json.loads(p.stdout)
        return d if p.returncode == 0 and isinstance(d, list) and all(isinstance(x, dict) for x in d) else None
    except Exception:
        return None


def mask(text):
    """A text of ssh or of Lium for the journal: no address in it."""
    return re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "ADDRESS", text or "")


GONE_WORDS = ("DELETED", "REMOVED", "TERMINATED")  # the last status of a pod that Lium has deleted, in the line of its last event


def state(pod_id):
    """'gone' only on Lium's own word for this exact id: `lium describe <id>` says that the pod is no longer listed AND
    its events say that it was deleted. 'listed' if Lium describes a pod that is there, None if unknown. A pod that a
    listing lacks for a moment has no such event, so it is never 'gone'."""
    if not pod_id:
        return None
    try:
        p = lium("describe", pod_id, timeout=60)
    except Exception:
        return None
    o = p.stdout + p.stderr
    if "no longer listed" in o:
        return "gone" if any(w in o for w in GONE_WORDS) else None
    return "listed" if p.returncode == 0 else None


def offers():
    """(the offers that a rent may take now, cheapest first; a short text of what Lium lists). An offer is a whole free
    node of a kind in GPU_PREF, on the secure tier, at or below MAX_PRICE. Reads only."""
    kinds = [(t.partition(":")[0], int(t.partition(":")[2] or 8)) for t in GPU_PREF.split()]
    rows = []
    for g in dict(kinds):
        try:
            d = json.loads(lium("ls", "--gpu", g, "--format", "json", timeout=90).stdout)
        except Exception:
            d = []
        rows += [r for r in d if isinstance(r, dict)] if isinstance(d, list) else []
    out = []
    for i, (g, n) in enumerate(kinds):
        for r in rows:
            p = r.get("price_per_hour"); free = r.get("available_gpu_count")
            if (r.get("gpu_type") == g and r.get("gpu_count") == n and (free if free is not None else n) >= n and r.get("tier") == "secure"
                    and p is not None and p <= MAX_PRICE and r.get("huid")):
                out.append(dict(huid=r["huid"], gpus=n, kind=f"{n}x{g}", price=p, order=i))
    out = sorted({o["huid"]: o for o in out}.values(), key=lambda o: (o["price"], o["order"], o["huid"]))
    seen = sorted({"%sx%s %s/%s %s $%s" % (r.get("gpu_count"), r.get("gpu_type"), r.get("available_gpu_count"), r.get("gpu_count"), r.get("tier"),
                                           r.get("price_per_hour")) for r in rows if (r.get("gpu_count") or 0) >= 4})
    return out, seen[:20]


def rents(add=False, cap=None, drop=None):
    """The times at which a pod was made in the last 24 h, over all rows. `add` writes one more and returns its time;
    with `cap` it writes only while fewer than `cap` are there (None: the day is full), in one step under the lock, so
    that rows that rent at the same moment cannot pass the cap together. `drop` takes a time back: no pod was made."""
    os.makedirs(C.ROWS, exist_ok=True)
    with open(C.ROWS + "/.rents.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = C.ROWS + "/rents.json"; ts = [t for t in (_rj(p) or []) if time.time() - t < 86400]
        if not add and drop is None:
            return ts
        if add and cap is not None and len(ts) >= cap:
            return None
        t = time.time()
        while t in ts: t += 1e-6
        ts = ts + [t] if add else [x for x in ts if x != drop]
        json.dump(ts, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
        return t


def protected(pod_id, *names):
    """True for a pod that this service never deletes, by its id or by any name that a record or Lium has for it."""
    return any(str(pod_id or "").startswith(x) for x in PROTECTED_IDS) or any(n in PROTECTED_NAMES for n in names)


def up(offer, name, ssh):
    """Make the pod on one offer (deploy/rent.sh: `lium up` and the ssh script). Returns (the pod as Lium gave it or None, the error text)."""
    try:
        p = subprocess.run(["bash", RENT, "%s %s %s %s" % (offer["huid"], offer["gpus"], offer["kind"], offer["price"])], capture_output=True, text=True,
                           timeout=1500, env=dict(os.environ, MAX_PRICE=str(MAX_PRICE), NAME=name, LIUM=LIUM, NODE_SSH=ssh))
    except subprocess.TimeoutExpired:
        return None, "`lium up` gave no answer in 25 minutes"  # the pod may be there: rent() looks for it under the row's name
    try:
        info = json.loads((p.stdout.strip().splitlines() or [""])[-1])
    except Exception:
        info = None
    return (info, "") if isinstance(info, dict) and info.get("id") else (None, f"rc {p.returncode}: " + mask(p.stderr or p.stdout)[-300:])


def setup(ssh):
    """Put the pinned commit of this repository on a new pod and run its bootstrap. True when the pod is ready."""
    b = subprocess.run([ssh, f"mkdir -p {C.NODE_ROOT}/logs && cd {C.NODE_ROOT} && (test -d repo || git clone -q {C.REPO_URL} repo) && git -C repo checkout -q --detach {C.REPO_REF} "
                        f"&& bash {C.NODE_REPO}/bench/deploy/bootstrap.sh > {C.NODE_ROOT}/logs/bootstrap.log 2>&1; tail -1 {C.NODE_ROOT}/logs/bootstrap.log"],
                       capture_output=True, text=True, timeout=2400)
    return "BOOTSTRAP_OK" in b.stdout, mask(b.stdout + b.stderr)[-300:]


def name_of(row):
    """The pod name of a row: the first NAME_MAX characters of `affine-v2-bench-<model_id>`. For a king that is
    `affine-v2-bench-king-` and the first 8 hex digits of its base hash, the 8 that the table's label shows."""
    return (PREFIX + row)[:NAME_MAX]


def held_pod(row, pods):
    """The id prefix of the pod that is kept for this row (HELD_FOR_ROW) while that pod is at Lium or in the record of
    the single-node service; else None. A listing without the pod does not clear it while that record names the pod."""
    rec = str((_rj(C.STATE + "/pod.json") or {}).get("id") or "")
    return next((x for x, r in HELD_FOR_ROW.items() if r == row and (rec.startswith(x) or any(str(p.get("id") or "").startswith(x) for p in pods))), None)


def claimed(pod_id):
    """The records (of every row, and of the single-node service) that name this Lium id."""
    return [f for f in glob.glob(C.ROWS + "/*/pod.json") + [C.STATE + "/pod.json"] if (_rj(f) or {}).get("id") == pod_id]


class Pod:
    """The pod of one row. `first`: the row runs on the pod that the single-node service rented; that pod keeps its
    name, its record and its ssh script, and no new commit is put on it."""

    def __init__(self, row, first=False):
        self.row, self.first = row, first
        self.dir = f"{C.ROWS}/{row}"
        self.name = NAME if first else name_of(row)
        self.record = C.STATE + "/pod.json" if first else self.dir + "/pod.json"
        self.ssh = C.NODE_SSH if first else self.dir + "/node-ssh"

    def load(self):
        """Current pod as {name, id, huid, gpu, cost, ...} or {} when there is none."""
        r = _rj(self.record) or {}
        return r if r.get("id") else {}

    def save(self, r):
        os.makedirs(os.path.dirname(self.record), exist_ok=True)
        t = self.record + ".tmp"; json.dump(r, open(t, "w"), indent=1); os.replace(t, self.record)

    def choose(self, offs):
        """The cheapest offer that no other row has chosen in the last CLAIM_S seconds; this row's choice is written down.
        The lock is held for these few file reads only: no row waits for the rent of another."""
        os.makedirs(self.dir, exist_ok=True)
        with open(C.ROWS + "/.choose.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            mine = self.dir + "/offer.json"; taken = set()
            for f in glob.glob(C.ROWS + "/*/offer.json"):
                c = _rj(f) or {}
                if f != mine and time.time() - c.get("at", 0) < CLAIM_S:
                    taken.add(c.get("huid"))
            pick = next((o for o in offs if o["huid"] not in taken), None)
            if pick:
                json.dump(dict(huid=pick["huid"], at=time.time()), open(mine, "w"))
            return pick

    def _write_ssh(self, p):
        """The ssh script to a pod that Lium lists, from the listing (the form that deploy/rent.sh writes). False: no address."""
        ip, port = p.get("ip"), (p.get("ports") or {}).get("22")
        if not ip or not port:
            return False
        with open(self.ssh, "w") as f:
            f.write(f"#!/bin/bash\n# bench pod {p.get('name')} ({p.get('huid')}, lium {p.get('id')}, {p.get('config')})\n"
                    f"exec ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60 -o ServerAliveInterval=30 "
                    f"-o UserKnownHostsFile=$HOME/.lium/known_hosts/{p.get('id')} -p {port} root@{ip} \"$@\"\n")
        os.chmod(self.ssh, 0o700)  # it holds the address of the pod
        return True

    def _failed(self, S, log, why, **more):
        """One failed try of a rent: the next try waits twice as long; after RENT_TRIES the row is `rent-failed`."""
        n = S["rent_tries"] = S.get("rent_tries", 0) + 1
        if n >= RENT_TRIES:
            S["rent_failed"] = dict(at=time.time(), why=why[-200:], tries=n)
        else:
            S["rent_wait_until"] = time.time() + RENT_WAIT_S * 2 ** (n - 1)
        log(event="rent-failed", tries=n, of=RENT_TRIES, gives_up=n >= RENT_TRIES, why=why[-300:], **more)

    def rent(self, S, log, tell=lambda state: None):
        """Rent and set up the pod of this row. Returns the record or None. It blocks for some minutes: only this row waits."""
        if self.first or S.get("rent_failed") or time.time() < max(S.get("rent_wait_until", 0), S.get("no_offer_at", 0) + NO_OFFER_RETRY_S):
            return None  # a failed try or "no offer" a moment ago: the wait runs
        pods = listed()
        if pods is None:
            log(event="rent-deferred", why="lium ps failed"); return None
        kept = held_pod(self.row, pods)
        if kept:  # never a rent for the row whose pod is still there: the operator decides
            S["held"] = dict(at=time.time(), why=f"pod {kept} is kept for this row and is still at Lium or in the record of the single-node service: no rent")
            log(event="rent-refused", why=S["held"]["why"]); return None
        mine = [p for p in pods if p.get("name") == self.name]
        rec = self.load()
        if len(mine) > 1:  # never one more, and no guess which of them is the row's
            self._failed(S, log, f"{len(mine)} pods are named {self.name}: the operator decides"); return None
        if mine and rec and rec.get("id") != mine[0].get("id"):
            self._failed(S, log, f"the pod named {self.name} is not the pod of this row's record: the operator decides"); return None
        if mine and not rec and claimed(mine[0].get("id")):  # two rows whose short names are equal: never two rows on one pod
            self._failed(S, log, f"the pod named {self.name} is the pod of another row's record: the operator decides"); return None
        if mine:  # the row's pod is there (a lost answer, or a setup that did not end): take it, never rent a second
            p = mine[0]
            if not rec:
                rec = dict(name=p.get("name"), id=p.get("id"), huid=p.get("huid"), gpu=(p.get("config") or "").replace("×", "x"),
                           cost=p.get("price_per_hour"), rented_at=time.time(), bootstrapped=False, taken_by_name=True)
                self.save(rec); rents(add=True)
                log(event="pod-taken-by-name", pod=rec["huid"], lium_id=rec["id"], gpu=rec["gpu"], usd_h=rec["cost"])
            if rec.get("taken_by_name") and not self._write_ssh(p):
                self.release(log, why="taken by name, and Lium lists no address for it")
                self._failed(S, log, "a pod under the row's name had no address"); return None
        else:
            if rec:  # the record names a pod that this listing does not have under the row's name
                if state(rec.get("id")) != "gone":  # no proof that it is gone: never a second pod beside it
                    self._failed(S, log, "Lium's listing does not have the pod of this row's record, and Lium did not say that it was deleted"); return None
                log(event="pod-record-dropped", lium_id=rec.get("id"), why="Lium says that the pod was deleted"); self.save(dict(gone=time.time(), last=rec))
            # The day's count is taken before the rent, in one step with its check; it is given back when no pod was made
            # (2026-11-03: four "no offer" failures in two minutes filled the cap and blocked the bench for 7 h).
            slot = rents(add=True, cap=MAX_RENTS_PER_DAY)
            if slot is None:
                log(event="rent-refused", why=f"{len(rents())} pods were made in 24 h over all rows (cap {MAX_RENTS_PER_DAY})"); S["no_offer_at"] = time.time(); return None
            offs, seen = offers(); pick = self.choose(offs)
            if not pick:
                rents(drop=slot); log(event="rent-no-offer", max_usd_h=MAX_PRICE, gpu_pref=GPU_PREF, fit=len(offs), seen=seen)
                S["no_offer_at"] = time.time(); return None
            info, err = up(pick, self.name, self.ssh)
            if not info:
                # `lium up` may still have made the pod (a lost answer). The row writes down the one pod that Lium now lists
                # under its exact name, so that the next try sets it up, or the row gives it back by that id.
                after = listed(); made = [q for q in (after or []) if q.get("name") == self.name]
                if len(made) == 1:
                    q = made[0]
                    self.save(dict(name=q.get("name"), id=q.get("id"), huid=q.get("huid"), gpu=(q.get("config") or "").replace("×", "x"),
                                   cost=q.get("price_per_hour"), rented_at=time.time(), bootstrapped=False, taken_by_name=True))
                    log(event="pod-taken-by-name", pod=q.get("huid"), lium_id=q.get("id"), usd_h=q.get("price_per_hour"))
                elif after is not None and not made:
                    rents(drop=slot)  # Lium answered and has no pod of this name: none was made
                self._failed(S, log, err, pod_made=len(made)); return None
            S.pop("no_offer_at", None)
            rec = dict(name=info.get("name"), id=info.get("id"), huid=info.get("huid"), gpu=(info.get("config") or "").replace("×", "x"),
                       cost=info.get("price_per_hour"), rented_at=time.time(), bootstrapped=False)
            self.save(rec); log(event="rented", pod=rec["huid"], lium_id=rec["id"], gpu=rec["gpu"], usd_h=rec["cost"])
        tell("setup")
        if rec["cost"] is None or rec["cost"] > MAX_PRICE:  # offers() filters on the price; this is the belt to its braces
            log(event="rent-over-cap", usd_h=rec["cost"]); self.release(log, why="over price cap")
            self._failed(S, log, "the pod's price is above the cap"); return None
        ok, out = setup(self.ssh)
        if not ok:
            log(event="bootstrap-failed", pod=rec["huid"], out=out); self.release(log, why="bootstrap failed")
            self._failed(S, log, "the setup of the pod failed"); return None
        rec["bootstrapped"] = True; rec.pop("taken_by_name", None); self.save(rec)
        S.pop("rent_tries", None); S.pop("rent_wait_until", None)
        log(event="bootstrapped", pod=rec["huid"], minutes=round((time.time() - rec["rented_at"]) / 60, 1))
        return rec

    def release(self, log, why, row_done=False):
        """Give back the pod of this row's record. The only place that sends a delete to Lium: for that exact Lium id, and
        only while Lium lists the id under the row's exact name. `row_done`: the row's results are in the published table."""
        rec = _rj(self.record) or {}
        pid = rec.get("id")
        if not pid:
            return True
        pods = listed()
        if pods is None:
            log(event="release-deferred", why="lium ps failed"); return False
        hit = [p for p in pods if p.get("id") == pid]
        names = [rec.get("name"), rec.get("huid")] + [x for p in hit for x in (p.get("name"), p.get("huid"))]
        if protected(pid, *names):
            log(event="release-refused", why=f"pod {str(pid)[:8]} is protected: this service never deletes it"); return False
        held = next((row for x, row in HELD_FOR_ROW.items() if str(pid).startswith(x)), None)
        if held and not (row_done and self.row == held):
            log(event="release-refused", why=f"pod {str(pid)[:8]} is kept until row {held} is done"); return False
        if hit and hit[0].get("name") != self.name:
            log(event="release-refused", why=f"pod {str(pid)[:8]} is named {hit[0].get('name')!r}, not {self.name!r}"); return False
        if hit:
            r = lium("rm", pid, "--name-only", "-y", timeout=180)
            if r.returncode != 0:
                log(event="release-failed", pod=rec.get("huid"), err=mask(r.stderr or r.stdout)[-200:]); return False
        elif state(pid) != "gone":  # a listing without the pod is no proof: the record stays, and the release is tried again
            log(event="release-deferred", why=f"Lium's listing does not have pod {str(pid)[:8]}, and Lium did not say that it was deleted"); return False
        self.save(dict(released=time.time(), why=why, last=rec))
        log(event="released", pod=rec.get("huid"), lium_id=pid, why=why,
            hours=round((time.time() - (rec.get("rented_at") or time.time())) / 3600, 2)); return True
