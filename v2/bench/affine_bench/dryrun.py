"""Host: one measured rebuild of a king on a rented pod (`affine-bench rebuild-dry-run`). The operator starts it.

It takes a king that is a checkpoint: each of its base files is in the store. On a pod it makes that king's base
from an OLDER checkpoint and the LoRAs of the steps between, as a row does for a king without its full files (chain.py,
deploy/chain_rebuild.py). The node's script compares the sha256 of each file of the result with the king's head
manifest: the hashes of the king's own published files. It does this once for each device (default: cpu, cuda:0) and
prints the minutes of each part and the dollars. Without `--rent` it reads only: it prints what it would do.

Money: the pod is the pod of one more row, with the name `dry-rebuild` (pod.py, class Pod, not changed). So every
guard of a row holds: the price cap, the kinds, the secure tier, the count of the day, one pod for the name, and the
delete by the exact id only. It rents only while fewer than P.MAX_PODS rows have or wait for a pod, itself counted, and
the service counts its pod as a row in work. One try: a rent or a rebuild that fails is not tried again. The pod goes
back at the end, after a failure, and after the signals INT, TERM and HUP. A pod is at most P.POD_MAX_HOURS old when a
rebuild on it may still start.

A process that is killed runs no code. For that case the row's state says `held` from the start, with the pod's record
in the place of a row's record: when this process is gone, the service starts the worker of `dry-rebuild`, and that
worker gives the pod back (daemon.Row.give_back), as for any row that is held. It runs no benchmark. The row in that
state has no `base_hash`: after a `retry` by mistake the worker can start nothing, fails three times in three polls, and
gives the pod back. A service that does not run gives nothing back: then `affine-bench release dry-rebuild`.

Nothing public: `dry-rebuild` is no listed row, so the table has no row, no result and no pod of it, and this module
writes nothing to the store. The report goes to the output and to a private file on this host. One true money alarm
can name the pod in the table, as for any pod: `release-refused`, `rent-no-answer`, or `pod-without-row` (settle())."""
import argparse, json, os, re, signal, time
from sn120_v2.errors import InfraError, VerifyError
from . import chain, config as C, daemon as D, pod as P, source
from .fetch import rebuild as rebuild_on_node

ROW = "dry-rebuild"  # the row id of the dry run: the pod's name is P.name_of(ROW)
DEVICES = "cpu,cuda:0"
RELEASE_TRIES = 5; RELEASE_WAIT_S = 60.0
QUIET_S = 60.0  # settle(): the rent starts within this time after a look of the service at Lium's list


class Refuse(Exception):
    """The dry run does not start. Nothing was rented."""


def say(**k):
    """One line in the output, as the service writes them. An output that is closed (a cut terminal) stops nothing."""
    try:
        D.log(row=ROW, **k)
    except (OSError, ValueError):
        pass


def pick(head=None, start=None, steps=1):
    """(the row of the king to make, the way to it from an older checkpoint): the newest king that fits, or the one
    that `head` names (a version id or its first digits, or a model id). `start` names the checkpoint to start from;
    else it is the head `steps` steps before the king. Refuse: no king fits, with the reason of the newest one."""
    if not chain.on():
        raise Refuse("BENCH_CHAIN_REF is not a full commit id, or BENCH_REBUILD_DEVICE is not cpu or cuda:N")
    try:
        heads = chain.heads_of(json.loads(source.store().get(chain.HEAD_CHAIN)))
    except KeyError:
        raise Refuse("the chain record is not in the store") from None
    except (VerifyError, ValueError) as e:
        raise Refuse(f"the chain record cannot be used: {e}") from None
    rows = {m["head_version_id"]: m for m in source.models()}; why = None
    for at in range(len(heads) - 1, 0, -1):
        king = heads[at]; m = rows.get(king["version_id"]); short = king["version_id"][:12]
        if head and not (king["version_id"].startswith(head) or (m or {}).get("model_id") == head):
            continue
        try:
            if m is None:
                raise chain.Wait(f"head {short} is no listed row")
            if king["checkpoint"] is not True or source.absent(m):
                raise chain.Wait(f"head {short} is not a checkpoint with each base file in the store: there is nothing to compare the result with")
            if (king["base_hash"], king["base_fp32_master_digest"]) != (m["base_hash"], chain.manifest_of(m).head.base_fp32_master_digest):
                raise VerifyError(f"the chain record and the manifest of head {short} name other files")
            first = at - steps if start is None else next((i for i in range(at) if heads[i]["version_id"].startswith(start)), -1)
            if first < 0:
                raise chain.Wait(f"head {short} has no head {steps} step(s) before it" if start is None else f"no head {start[:12]} is before head {short}")
            return m, chain.way_from(heads, first, at)
        except (chain.Wait, VerifyError) as e:
            why = why or str(e)
    raise Refuse(why or (f"the chain record has no head {head[:12]} after its first head" if head else "the chain record has no step"))


def in_work():
    """The rows that have a pod or will get one: as the service counts its rows in work (daemon.plan), and each queued row."""
    rows = D.model_rows()
    busy = lambda mid: D.pod_of(mid).load() or D.row_state(mid).get("running") or (D.row_state(mid).get("phase") or {}).get("state") in ("renting", "setup")
    return {mid for mid in D.row_ids() if mid != ROW and busy(mid)} | {m["model_id"] for m in D.queue(rows, D.waiting(rows))}


def settle(sleep=time.sleep):
    """Wait until the service has just looked at Lium's list (daemon.lium_check, each D.LIUM_CHECK_S).

    Between Lium's making of the pod and the write of its record, the pod has a bench name and no record names it. A look
    of the service in that time puts the alarm `pod-without-row` into the table until its next look. A rent that starts
    right after a look has D.LIUM_CHECK_S for that. A service that does not run looks at nothing: no wait."""
    try:
        if D.now() - os.path.getmtime(D.ST + "/heartbeat") > 4 * D.POLL_S:
            return
    except OSError:
        return
    end = D.now() + D.LIUM_CHECK_S + QUIET_S
    while D.now() < end and D.now() - ((D.rj(D.SF) or {}).get("lium_checked") or 0) > QUIET_S:
        sleep(5)


def give_back(row):
    """Give the pod of the dry run back: True when the record names no pod any more. RELEASE_TRIES tries; after them the
    record and `held` stay, and the service's worker of this row goes on with the tries (and the table gets its alarm)."""
    for i in range(RELEASE_TRIES):
        try:
            if row.pod.release(say, why="dry run"):
                return True
        except Exception as e:  # a Lium client that hangs or breaks: one failed try
            say(event="release-failed", err=type(e).__name__ + ": " + P.mask(str(e))[:200])
        if i + 1 < RELEASE_TRIES:
            time.sleep(RELEASE_WAIT_S)
    return False


def adopt(row):
    """After a rent that was cut (a signal while `lium up` ran): the one pod that Lium lists under the name of the dry
    run, and that no record names, gets the record of the dry run, in the form that pod.Pod.rent writes. Then the
    give-back has an id. Returns the record, or {}: Lium lists no such pod now. It may still make it: `affine-bench pods`."""
    pods = P.listed(); mine = [q for q in pods or [] if q.get("name") == row.pod.name]
    if len(mine) != 1 or P.claimed(mine[0].get("id")):
        say(event="alarm", what="dry-run-rent-cut", listed=None if pods is None else len(mine),
            do=f"`affine-bench pods` in some minutes. A pod named {row.pod.name} that no row claims is the pod of this dry run: start the dry run again, it takes that pod and gives it back")
        return {}
    q = mine[0]
    row.pod.save(dict(name=q.get("name"), id=q.get("id"), huid=q.get("huid"), gpu=(q.get("config") or "").replace("×", "x"),
                      cost=q.get("price_per_hour"), rented_at=time.time(), bootstrapped=False, taken_by_name=True))
    say(event="pod-taken-by-name", pod=q.get("huid"), lium_id=q.get("id"), usd_h=q.get("price_per_hour"))
    return row.pod.load()


def measure(m, way, device, node, rec):
    """One rebuild on the node for one device, in a new directory. Returns the run for the report. `status` is that of
    the node's script (rebuilt, refused, failed); `cut` when the script gave no answer (a time limit, a broken ssh)."""
    order = chain.pack(m, way, device); run = dict(device=device, limit_min=round(order["limit_s"] / 60, 1))
    if D.now() - rec["rented_at"] + order["limit_s"] > P.POD_MAX_HOURS * 3600:
        return run | dict(status="not-run", same_files=False, why=f"the pod would be older than {P.POD_MAX_HOURS:.0f} hours at the end of this rebuild's time limit")
    t = D.now()
    try:
        node.run(f"rm -rf {C.NODE_ROOT}/dry", timeout=600)  # only one base at a time takes disk, and no rebuild finds the result of another
        r = rebuild_on_node(m, node, order, root=f"{C.NODE_ROOT}/dry/{device.replace(':', '')}")
        out = json.loads(r["line"]) if r["line"].startswith("{") else {}
        run |= dict(status=out.get("status") or r["status"], why=P.mask(str(out.get("why") or r["err"] or ""))[:300], times=out.get("times") or {},
                    steps=out.get("steps"), started_from=out.get("started_from"), fetched_bytes=out.get("fetched_bytes"))
    except Exception as e:  # this device has no result. The script may still run on the node: no other rebuild beside it
        run |= dict(status="cut", why=type(e).__name__ + ": " + P.mask(str(e))[:200], times={})
    run["same_files"] = run["status"] == "rebuilt"  # the script says `rebuilt` only when each sha256 is that of the king's manifest
    return run | dict(minutes=round((D.now() - t) / 60, 1))


def lines(rep):
    """The report as text: the minutes of each part, and the dollars at the pod's price."""
    usd_h = (rep.get("pod") or {}).get("usd_h") or 0
    row = lambda name, minutes, note="": ("  %-38s %6s min %8s $  %s" % (name, "-" if minutes is None else "%.1f" % minutes, "-" if minutes is None else "%.2f" % (minutes * usd_h / 60), note)).rstrip()
    mins = lambda s: None if s is None else s / 60
    out = [f"dry run of a rebuild: {rep['result']}" + (f" ({rep['why']})" if rep.get("why") else ""),
           f"king {rep['king'][:12]} from checkpoint {rep['start'][:12]}, {rep['steps']} step(s), chain code {rep['chain_ref'][:12]}"]
    if rep.get("pod"):
        out.append(f"pod: {rep['pod']['gpu']} at {usd_h} $/h")
        out += [row("rent (Lium makes the pod)", rep["minutes"].get("rent"), "counted as paid time: Lium's own count can start later"), row("setup of the pod", rep["minutes"].get("setup"))]
    for run in rep.get("runs") or []:
        t = run.get("times") or {}; gb = (t.get("master_bytes") or 0) / 1e9
        out.append(f"device {run['device']}: {run['status']}" + (f" ({run['why']})" if run.get("why") else "") + (f", {t['cores']} CPU cores" if t.get("cores") else ""))
        if run["status"] == "not-run":
            continue
        out.append(row("checkout and install", mins((t.get("checkout_s") or 0) + (t.get("install_s") or 0)) if "checkout_s" in t else None))
        out.append(row("download of the master", mins(t.get("master_s")), "%.2f GB, %.0f MB/s" % (gb, gb * 1000 / t["master_s"]) if t.get("master_s") else ""))
        out.append(row("small files and cast to the base", mins(t.get("cast_s"))))
        for i, step in enumerate(t.get("steps") or [], 1):
            out.append(row(f"fold {i} (with hash check 1)", mins(step.get("fold_s")),
                           "LoRA download %.1f min, after the fold %.1f min" % ((step.get("lora_s") or 0) / 60, (step.get("after_s") or 0) / 60)))
        if not t.get("steps"):
            out.append(row("each fold", None, "the lines of the tool did not give it"))
        out.append(row("hash check 2 (the king's manifest)", mins(t.get("check_s"))))
        out.append(row("the whole rebuild", run.get("minutes"), "disk taken: %.1f GB" % (t["disk_used_bytes"] / 1e9) if "disk_used_bytes" in t else ""))
    if rep.get("pod"):
        out.append(row("give the pod back", rep["minutes"].get("give_back"), "GIVEN BACK" if rep.get("released") else "NOT GIVEN BACK: `affine-bench pods`"))
        out.append(row("total, from the rent to the give-back", rep["minutes"].get("total")))
    out.append("hash check 1: the tool makes the sha256 of each folded file while it writes it, and compares it with the chain record. It has no time of its own.")
    out.append("hash check 2: the node's script makes the sha256 of each file of the base again and compares it with the king's head manifest.")
    return out


def run(m, way, devices):
    """Rent, measure, give back. Returns the exit code: 0 when each device gave the king's files and the pod is back;
    1 a rebuild did not give them; 2 no pod was kept for a rebuild; 3 the pod is NOT given back."""
    lock = D.row_lock(ROW)
    if lock is None:
        raise Refuse("another dry run is alive, or the service's worker gives the pod of a dry run back now")
    row = D.Row(ROW); S = row.S
    if row.rec:
        raise Refuse("the record still names the pod of an earlier dry run: `affine-bench pods`. The service gives it back; without the service, `affine-bench release dry-rebuild`")
    if (S.get("rent_failed") or {}).get("no_answer"):  # as for a row: Lium may still make that pod, and a second `lium up` would make a second one
        raise Refuse("`lium up` gave no answer in an earlier dry run: `affine-bench pods`, then `affine-bench retry dry-rebuild`. The next dry run takes a pod that Lium lists under the name")
    for k in ("rent_failed", "rent_tries", "rent_wait_until", "no_offer_at", "expired", "release_refused", "release_tried", "release_waits", "idle_since", "node_down"):
        S.pop(k, None)
    # `held`, and the row that the worker reads: a worker of the service that finds this row gives its pod back and runs nothing
    S.update(fails=0, running=None, phase=dict(state="dry-run", since=D.now()), held=dict(at=D.now(), why="a measured rebuild: no benchmark runs on this pod, and it goes back"),
             model=dict(model_id=ROW, kind="dry-run", dry_run_of=m["head_version_id"]))
    row.save()
    rep = dict(result="failed", king=m["head_version_id"], model_id=m["model_id"], start=way["start"]["version_id"], steps=len(way["steps"]), chain_ref=chain.REF,
               started_at=D.now(), minutes={}, runs=[])
    said = []; tell = lambda **k: (said.append(k), say(**k)); t = {}
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def stop(sig, frame):
        for s in signals: signal.signal(s, signal.SIG_IGN)  # the give-back is not cut by a second signal
        raise SystemExit(f"signal {sig}")

    def work():
        busy = in_work()
        if len(busy) + 1 > P.MAX_PODS:
            rep.update(result="refused", why=f"{len(busy)} rows have or wait for a pod, and at most {P.MAX_PODS} pods are there at a time: no pod for the dry run"); return 2
        settle()
        t["rent"] = D.now(); rec = row.pod.rent(S, tell, lambda state: None); row.save(); row.attach()
        if not rec:
            last = next((k for k in reversed(said) if k.get("why")), said[-1] if said else {})
            rep.update(result="no-pod", why=P.mask(f"{last.get('event')}: {last.get('why') or 'see the lines above'}")[:300]); return 2
        rep["pod"] = dict(gpu=rec.get("gpu"), usd_h=rec.get("cost"), name=rec.get("huid"))
        rep["minutes"].update(rent=round((rec["rented_at"] - t["rent"]) / 60, 1), setup=round((D.now() - rec["rented_at"]) / 60, 1))
        if len(in_work()) + 1 > P.MAX_PODS:  # a row came while Lium made the pod: the rows go first
            rep.update(result="no-place", why=f"more than {P.MAX_PODS} pods with this one: it goes back at once"); return 2
        for device in devices:
            say(event="dry-run-rebuild", device=device)
            cut = any(r["status"] == "cut" for r in rep["runs"])
            rep["runs"].append(dict(device=device, status="not-run", same_files=False, why="the rebuild before it gave no answer and may still run on the node") if cut
                               else measure(m, way, device, row.node, rec))
        good = all(r["status"] == "rebuilt" for r in rep["runs"])
        rep["result"] = "done" if good else "failed"; return 0 if good else 1

    old = {s: signal.signal(s, stop) for s in signals}
    try:
        code = work()
    except BaseException as e:
        rep.update(result="cut", why=type(e).__name__ + ": " + P.mask(str(e))[:200]); raise
    finally:
        for s in signals: signal.signal(s, signal.SIG_IGN)
        at = D.now(); rec = row.pod.load()
        if not rec and "rent" in t and "pod" not in rep and rep["result"] == "cut":
            rec = adopt(row)
        rep["released"] = give_back(row) if rec else True
        if rec:
            rep.setdefault("pod", dict(gpu=rec.get("gpu"), usd_h=rec.get("cost"), name=rec.get("huid")))
            rep["minutes"].update(give_back=round((D.now() - at) / 60, 1), total=round((D.now() - t["rent"]) / 60, 1))
            rep["usd"] = round(rep["minutes"]["total"] * (rec.get("cost") or 0) / 60, 2)
        if rep["released"]:
            S.pop("held", None)  # no pod is left: the worker of the service has nothing to do for this row
        S["phase"] = dict(state="dry-run-ended", since=D.now()); row.save()
        try:
            os.makedirs(f"{D.PRIV}/{ROW}", exist_ok=True)
            D.wj(f"{D.PRIV}/{ROW}/report-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", rep, 0o600)
            print(P.mask(json.dumps(dict(dry_run=rep))), flush=True); print(P.mask("\n".join(lines(rep))), flush=True)
        except Exception as e:  # the report is words: an output that is closed, or a number that is not there, stops nothing here
            say(event="error", during="report", error=type(e).__name__ + ": " + str(e)[:200])
        if not rep["released"]:
            say(event="alarm", what="dry-run-pod-not-given-back", do="`affine-bench pods`. The service tries again each 10 minutes; without the service, `affine-bench release dry-rebuild`")
        for s, h in old.items(): signal.signal(s, h)
    return code if rep["released"] else 3


def main(argv):
    ap = argparse.ArgumentParser(prog="affine-bench rebuild-dry-run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", help="the king to make: a version id or its first digits, or a model id (default: the newest king that fits)")
    ap.add_argument("--from", dest="start", help="the checkpoint to start from: a version id or its first digits (default: --steps before the king)")
    ap.add_argument("--steps", type=int, default=1, help="how many steps before the king the start is (default 1)")
    ap.add_argument("--devices", default=DEVICES, help=f"the devices, one rebuild for each (default {DEVICES})")
    ap.add_argument("--rent", action="store_true", help="rent a pod and measure. Without it: print the plan, rent nothing")
    a = ap.parse_args(argv); devices = [d for d in a.devices.split(",") if d]
    try:
        if not devices or not all(re.fullmatch(r"cpu|cuda:\d+", d) for d in devices) or a.steps < 1:
            raise Refuse("--devices is a list of cpu and cuda:N, and --steps is 1 or more")
        try:
            m, way = pick(a.head, a.start, a.steps)
        except InfraError as e:
            raise Refuse("the store could not be read: " + str(e)[-200:]) from None
        limit_s = P.UP_TIMEOUT_S + 2400 + len(devices) * (chain.BASE_S + chain.STEP_S * len(way["steps"])) + RELEASE_TRIES * RELEASE_WAIT_S
        plan = dict(king=m["head_version_id"], model_id=m["model_id"], start=way["start"]["version_id"], steps=len(way["steps"]), devices=devices, chain_ref=chain.REF,
                    pod_name=P.name_of(ROW), max_usd_h=P.MAX_PRICE, bound_minutes=round(limit_s / 60), bound_usd=round(limit_s / 3600 * P.MAX_PRICE, 2), rent=a.rent)
        print(json.dumps(dict(dry_run_plan=plan)), flush=True)
        if not a.rent:
            print("nothing was rented: this is the plan. `--rent` rents the pod and measures.", flush=True); return 0
        return run(m, way, devices)
    except Refuse as e:
        print(json.dumps(dict(dry_run=dict(result="refused", why=str(e), rented=False))), flush=True); return 2
