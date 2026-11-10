#!/usr/bin/env python3
"""Rent / release affine-pe-* pods. B300-first-then-cheapest among ALLOWED_GPUS.

Allowed: B300, RTX5090, B200, H200, H100. Exactly 8 GPUs. Optional price ceiling
(PIT). Never touches NEVER_NAMES. One lium-up at a time (rent.lock).
"""
import fcntl, json, re, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
import replace_pod as R

PJ = ST / "pods.json"

def load_pods():
    return rj(PJ, {}) or {}

def save_pods(d):
    wj(PJ, d, 0o600)

AVOID = ST / "AVOID-EXECUTORS.json"  # executors whose lane failed recently (slow uplink etc.)
AVOID_TTL_S = 3 * 3600

def avoided():
    now = time.time()
    return {k for k, v in (rj(AVOID, {}) or {}).items() if float(v.get("until", 0)) > now}

def avoid_executor(huid, why):
    if not huid:
        return
    d = rj(AVOID, {}) or {}
    d[huid] = dict(until=time.time() + AVOID_TTL_S, why=str(why)[:200], at_cr=cr(time.time()))
    wj(AVOID, d, 0o600)
    ev(event="executor_avoided", huid=huid, why=str(why)[:200], ttl_s=AVOID_TTL_S)

def pick_node(prefer=None, fallback=True, price_ceiling=None):
    """-> (gpu, row): B300-first-then-cheapest 8-GPU offer under optional ceiling.

    Collect every allowed class (prefer first, then the rest of gpu_order when
    fallback=True). Prefer the cheapest offer of order[0] (default B300) when
    one exists under the ceiling; otherwise the global cheapest under the
    ceiling. Requires exactly 8 GPUs available.
    """
    if prefer and any(g not in ALLOWED_GPUS for g in prefer):
        raise ValueError("PE requires %s; unsupported GPU class" % "/".join(ALLOWED_GPUS))
    base = list(gpu_order())
    order = list(prefer or [])
    if fallback or not order:
        order += [g for g in base if g not in order]
    if not order or any(g not in ALLOWED_GPUS for g in order):
        raise ValueError("PE requires %s; unsupported GPU class" % "/".join(ALLOWED_GPUS))
    cands = []
    bad = avoided()
    need = int(REQUIRED_GPU_COUNT)
    for g in order:
        try:
            rows = lium_ls(g)
        except Exception as ex:
            ev(event="ls_fail", gpu=g, err=str(ex)[:200])
            continue
        for r in rows:
            avail = int(r.get("available_gpu_count") or 0)
            total = int(r.get("gpu_count") or 0)
            # Require a full 8-GPU pack (lium up always uses -c 8).
            if avail < need:
                continue
            if total and total < need:
                continue
            huid = r.get("huid") or ""
            if any(n in huid or n in (r.get("name") or "") for n in NEVER_NAMES):
                continue
            if huid in bad:
                continue
            cands.append((g, r, float(r.get("price_per_hour") or 999)))
    return pick_best_offer(cands, order=order, ceiling=price_ceiling)

class CapReached(RuntimeError):
    """throughput-v4: renting would exceed MAX_EVAL_NODES live eval nodes (lanes + warm, dedicated baseline excluded)."""


BASELINE_IDS = ("dfd5b1ef-ab22-49a0-83b0-fc4c938f6c5d",)
BASELINE_HUIDS = ("brave-matrix-fc",)


def is_baseline(p):
    return "baseline" in (p.get("name") or "") or p.get("id") in BASELINE_IDS or p.get("huid") in BASELINE_HUIDS


def eval_nodes(ps=None):
    """Live affine-pe-* eval nodes (lanes + warm), dedicated baseline excluded."""
    return [p for p in our_pods(ps) if not is_baseline(p)]


def rent(name, ttl="12h", gpu=None, huid=None, max_eval_nodes=None):
    """max_eval_nodes: checked under rent.lock (every PE rental serializes there, lium up completes inside the lock) so
    concurrent lanes/warm workers can never push the live eval-node count past the cap. Raises CapReached."""
    if gpu is not None and gpu not in ALLOWED_GPUS:
        raise ValueError("PE requires %s; unsupported GPU class" % "/".join(ALLOWED_GPUS))
    if huid is not None:
        raise ValueError("Unverified explicit-node override is disabled; use a supported GPU listing")
    assert name.startswith(PREFIX), name
    assert not any(n in name for n in NEVER_NAMES)
    with open(ST / "rent.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if max_eval_nodes is not None:
            n = len(eval_nodes())
            if n >= int(max_eval_nodes):
                ev(event="rent_cap", name=name, nodes=n, cap=int(max_eval_nodes))
                raise CapReached("eval nodes %d >= cap %d" % (n, int(max_eval_nodes)))
        if huid is None:
            g, row = pick_node([gpu] if gpu else None)
            if not row:
                ev(event="rent_no_supply", name=name, gpu=gpu)
                return None
            huid = row["huid"]
            price = row.get("price_per_hour")
            country = row.get("country_code")
            tier = row.get("tier")
            gpu = g
        else:
            price = country = tier = None
        t0 = time.time()
        ev(event="rent_attempt", name=name, huid=huid, gpu=gpu, price_h=price)
        rc, o, e = lium(
            "up", huid, "-c", "8", "--name", name, "--verify-gpus", "--strict-gpus",
            "--no-volume-encryption", "--no-backup", "-y", "--json", "--timeout", "900",
            "--ttl", ttl, timeout=1000,
        )
        if rc != 0:
            err = (e or o)[-400:]
            ev(event="rent_failed", name=name, huid=huid, seconds=round(time.time() - t0, 1), err=err)
            # adopt if "still being processed"
            if "still being processed" in err or "earlier request" in err:
                deadline = time.time() + 360
                while time.time() < deadline:
                    time.sleep(15)
                    for p in our_pods():
                        if p.get("name") == name:
                            d = _from_ps(p, gpu=gpu, price=price, country=country, tier=tier, rented_at=t0)
                            pj = load_pods(); pj[name] = d; save_pods(pj)
                            ev(event="rent_adopted", name=name, id=d["id"], huid=d.get("huid"))
                            return d
            return None
        info = None
        try:
            raw = json.loads(o)
            info = raw.get("pod") if isinstance(raw, dict) and "pod" in raw else raw
        except Exception:
            info = None
        if not info:
            time.sleep(5)
            for p in our_pods():
                if p.get("name") == name:
                    info = p
                    break
        if not info:
            ev(event="rent_parse_fail", name=name, out=o[-300:], err=e[-300:])
            return None
        d = _from_ps(info, gpu=gpu, price=price, country=country, tier=tier, rented_at=t0)
        d["executor"] = huid
        if not d.get("id") or not d.get("ip") or not d.get("port"):
            time.sleep(3)
            for p in our_pods():
                if p.get("name") == name or p.get("id") == d.get("id"):
                    d = _from_ps(p, gpu=gpu, price=price, country=country, tier=tier, rented_at=t0)
                    d["executor"] = huid
                    break
        for _ in range(60):
            try:
                if d.get("ip") and d.get("port") and R.ssh_ok(d["ip"], d["port"], d["id"]):
                    break
            except Exception:
                pass
            time.sleep(5)
        d["ready_at"] = time.time()
        d["rent_to_ready_s"] = round(d["ready_at"] - t0, 1)
        pj = load_pods(); pj[name] = d; save_pods(pj)
        ev(event="rented", name=name, id=d.get("id"), huid=d.get("huid"), gpu=gpu,
           price_h=d.get("price_h"), seconds=round(time.time() - t0, 1), country=d.get("country"),
           rent_to_ready_s=d.get("rent_to_ready_s"))
        return d

def _from_ps(p, gpu=None, price=None, country=None, tier=None, rented_at=None):
    # ssh_command: ssh -p PORT ... root@IP
    ip = p.get("ip") or p.get("ssh_ip")
    port = p.get("ssh_port") or p.get("port")
    sc = p.get("ssh_command") or ""
    if (not ip or not port) and "ssh" in sc:
        m = re.search(r"-p\s+(\d+).*?root@([0-9.]+)", sc)
        if m:
            port, ip = int(m.group(1)), m.group(2)
    return dict(
        id=p.get("id"), huid=p.get("huid"), name=p.get("name"), ip=ip, port=int(port or 0),
        gpu=gpu or p.get("gpu_model") or p.get("gpu_name"),
        price_h=price if price is not None else p.get("price_per_hour"),
        country=country or p.get("country_code"), tier=tier or p.get("tier"),
        rented_at=rented_at or time.time(), rented_at_cr=cr(rented_at or time.time()),
        status=p.get("status"),
    )

def release(name_or_id, reason="done"):
    pj = load_pods()
    d = None
    key = None
    for k, v in list(pj.items()):
        if k == name_or_id or v.get("id") == name_or_id or v.get("name") == name_or_id:
            d, key = v, k
            break
    if not d:
        # try live ps
        for p in our_pods():
            if p.get("name") == name_or_id or p.get("id") == name_or_id:
                d = _from_ps(p)
                key = p.get("name")
                break
    if not d or not d.get("id"):
        ev(event="release_missing", name=name_or_id, reason=reason)
        return False
    assert (d.get("name") or "").startswith(PREFIX) or str(name_or_id).startswith(PREFIX)
    if any(n in (d.get("name") or "") or n in (d.get("huid") or "") for n in NEVER_NAMES):
        ev(event="release_refused_never", name=d.get("name"), reason=reason)
        return False
    if str(reason).startswith("lane-failed"):
        avoid_executor(d.get("executor"), reason)
    rc, o, e = lium("rm", d["id"], "-y", timeout=120)
    ok = rc == 0
    ev(event="released" if ok else "release_failed", name=d.get("name"), id=d["id"], reason=reason, err=(e or o)[-200:] if not ok else None)
    if key and key in pj:
        pj[key]["released"] = ok
        pj[key]["released_at"] = time.time()
        pj[key]["release_reason"] = reason
        save_pods(pj)
    return ok

def rent_next(tag, ttl="12h", gpu=None):
    """tag is short suffix; full name affine-pe-<tag>."""
    name = PREFIX + tag if not tag.startswith(PREFIX) else tag
    return rent(name, ttl=ttl, gpu=gpu)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ls"
    if cmd == "ls":
        for g in gpu_order():
            try:
                rows = lium_ls(g)
            except Exception as ex:
                print(g, "ERR", ex); continue
            avail = [r for r in rows if (r.get("available_gpu_count") or 0) >= 8]
            print(g, "listed", len(rows), "avail8", len(avail))
            for r in sorted(avail, key=lambda x: x.get("price_per_hour") or 999)[:5]:
                print(" ", r.get("huid"), r.get("price_per_hour"), r.get("country_code"), r.get("tier"))
    elif cmd == "ps":
        for p in our_pods():
            print(p.get("name"), p.get("huid"), p.get("status"), p.get("price_per_hour"))
    elif cmd == "rent":
        tag = sys.argv[2]; gpu = sys.argv[3] if len(sys.argv) > 3 else None
        print(json.dumps(rent_next(tag, gpu=gpu), default=str, indent=1))
    elif cmd == "rm":
        print("released" if release(sys.argv[2], "cli") else "FAILED")
