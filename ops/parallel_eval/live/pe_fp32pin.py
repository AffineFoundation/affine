#!/usr/bin/env python3
"""PE fp32-master protection (2026-11-07 CR) + head pins + upcast/chain alerts (fp32-fix v2, 2026-11-08 CR, Jacob pt1391u).

Under PE the GREEN decision lags training by 40-80 min and many epochs, while the trainer GC (host-v834c
mimo_host_trainer_v83._fp32_gc) keeps only the newest FP32_KEEP store entries + entries whose PIN file is < 3 h old.
c59c06ff's committed master (E843, entry 0b1158b4...) was GC'd before fp32_master_publish.py could upload it, and after the
E836 rollback 8fcdb0b5's master entry was already GC'd, so the trainer silently bootstrap-upcast from bf16 (E840, E852).

Pins (same PIN protocol the publisher/pod_upload already use: `touch <store>/<entry>/PIN`, refreshed every tick):
  * every PE candidate output entry from TRAINING-RESULT until RED / no_decision (undecided capped at 12 h);
  * every GREEN entry until the publisher says 'published' (v2: no 48 h cap);
  * v2: the CURRENT head's master entry, always;
  * v2: the PREVIOUS head's master entry until the current head's master is 'published'.
Alerts (v2, alert-only; never changes anything): state/FP32-ALERT.json {ok, alerts[]} + events, for
  * any TRAINING-RESULT with fp32_master.source == 'bootstrap-upcast' (epoch >= ALERT_FROM_EPOCH);
  * any turn whose input_master_digest != the expected master digest of its input head (head -> GREEN epoch output digest,
    or state/FP32-CHAIN-OVERRIDES.json {head_checkpoint: accepted_master_digest} for deliberate ROOT-signed changes);
  * a CURRENT head whose master entry is missing on the pod.
Only touches PIN files on the trainer pod (via the gate's rssh.sh); never deletes anything, never touches the trainer.
"""
import json, re, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *

RSSH = "/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh"
STORE = "/workspace/affine-mimo-trainer-prod/fp32-master"
PUB_STATE = RA / "20261104-fp32-master-publish/state/publisher.json"
OUT = ST / "FP32-PIN.json"
ALERT = ST / "FP32-ALERT.json"
OVERRIDES = ST / "FP32-CHAIN-OVERRIDES.json"
EVERY_S = 600
UNDECIDED_MAX_AGE_S = 12 * 3600
ALERT_FROM_EPOCH = 978          # E840-E862 upcasts are the known, already-reported losses (RA/20261108-no-improvement-audit)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
_last = dict(at=0.0, want=None)


def fp32_of(e):
    tr = rj(LIFE / "epochs" / str(e) / "TRAINING-RESULT.json") or {}
    rep = (tr.get("payload") or {}).get("report") or {}
    fm = (((rep.get("training") or {}).get("optimizer") or {}).get("sgd") or {}).get("fp32_master") or {}
    return rep, fm


def store_entry(e):
    _, fm = fp32_of(e)
    ent = fm.get("output_store_entry")
    return (ent if isinstance(ent, str) and HEX64.match(ent) else None), fm.get("output_master_digest")


def head_epochs():
    """{head checkpoint: epoch whose TRAINING-RESULT produced it} for every applied head (HEAD-CHAIN + genesis)."""
    hc = rj(ST / "HEAD-CHAIN.json", {}) or {}
    out = {}
    for a in hc.get("applied") or []:
        if a.get("new_head") and a.get("epoch"):
            out[a["new_head"]] = int(a["epoch"])
    g = hc.get("genesis")
    if g and g not in out:
        for d in sorted((LIFE / "epochs").glob("*"), key=lambda p: -int(p.name) if p.name.isdigit() else 0):
            if d.name.isdigit() and int(d.name) < int(pe_cfg().get("from_epoch") or 10 ** 9):
                rep, _ = fp32_of(int(d.name))
                if rep.get("output_checkpoint") == g:
                    out[g] = int(d.name); break
    return out


def current_head():
    cur = rj(LIFE / "CURRENT.json", {}) or {}
    return ((cur.get("checkpoint") or {}).get("id"))


def head_pins(pub):
    """CURRENT head always; previous head until CURRENT's master is published."""
    out = {}
    hc = rj(ST / "HEAD-CHAIN.json", {}) or {}
    chain = [hc.get("genesis")] + [a.get("new_head") for a in hc.get("applied") or []]
    he = head_epochs(); cur = current_head()
    if not cur or cur not in he:
        return out, cur
    ent, dig = store_entry(he[cur])
    if ent:
        out[ent] = dict(epoch=he[cur], why="head-current", master_digest=dig)
    cur_pub = (pub.get(str(he[cur])) or {}).get("status") == "published"
    if not cur_pub and cur in chain and chain.index(cur) > 0:
        prev = chain[chain.index(cur) - 1]
        if prev in he:
            pe_, pd = store_entry(he[prev])
            if pe_:
                out.setdefault(pe_, dict(epoch=he[prev], why="head-previous-until-current-published", master_digest=pd))
    return out, cur


def wanted(now=None):
    now = now or time.time()
    frm = int(pe_cfg().get("from_epoch") or 0)
    pub = (rj(PUB_STATE, {}) or {}).get("done") or {}
    out = {}
    for d in (LIFE / "epochs").glob("*"):
        if not d.name.isdigit() or int(d.name) < frm:
            continue
        e = int(d.name); trp = d / "TRAINING-RESULT.json"
        if not trp.exists():
            continue
        age = now - trp.stat().st_mtime
        dec = rj(EP / str(e) / "DECISION.json") or {}
        if not dec.get("decision"):
            if age > UNDECIDED_MAX_AGE_S:
                continue
            why = "undecided"
        elif dec.get("decision") == "green" and not dec.get("no_decision"):
            if (pub.get(str(e)) or {}).get("status") in ("published", "store-entry-gone"):   # gone = known permanent loss
                continue
            why = "green-unpublished"
        else:
            continue
        ent, dig = store_entry(e)
        if ent:
            out[ent] = dict(epoch=e, why=why, master_digest=dig)
    hp, _ = head_pins(pub)
    for k, v in hp.items():
        out[k] = v          # head reasons win (never expire)
    return out


def chain_alerts():
    """Alert-only scan: bootstrap-upcast and master hash-chain breaks for epochs >= ALERT_FROM_EPOCH."""
    he = head_epochs(); ov = rj(OVERRIDES, {}) or {}
    expect = {}
    for h, e in he.items():
        _, dig = store_entry(e)
        if dig:
            expect[h] = dig
    alerts = []; seen = 0
    for d in (LIFE / "epochs").glob("*"):
        if not d.name.isdigit() or int(d.name) < ALERT_FROM_EPOCH or not (d / "TRAINING-RESULT.json").exists():
            continue
        e = int(d.name); rep, fm = fp32_of(e); seen += 1
        if not fm:
            continue
        if fm.get("source") == "bootstrap-upcast":
            alerts.append(dict(kind="bootstrap-upcast", epoch=e, input_checkpoint=(rep.get("checkpoint") or "")[:16]))
        h = rep.get("checkpoint"); want = expect.get(h)
        if e >= int(((ov.get("stage") or {}).get(h) or {}).get("from_epoch", 10**12)):
            want = (ov.get("accept") or {}).get(h, want)
        got = fm.get("input_master_digest")
        if want and got and got != want:
            alerts.append(dict(kind="chain-break", epoch=e, input_checkpoint=(h or "")[:16], expected=want[:16], got=got[:16],
                               source=fm.get("source")))
    return sorted(alerts, key=lambda a: a["epoch"]), seen


def tick(force=False):
    now = time.time(); want = wanted(now)
    if not force and now - _last["at"] < EVERY_S and set(want) == set(_last["want"] or ()):
        return None
    _last.update(at=now, want=set(want))
    res = dict(at_cr=cr(now), pinned={}, missing={})
    if want:
        cmd = "; ".join("if test -f %s/%s/meta.json; then touch %s/%s/PIN && echo OK %s; else echo MISSING %s; fi" % (STORE, x, STORE, x, x, x)
                        for x in sorted(want))
        try:
            p = subprocess.run([RSSH, cmd], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
            lines = p.stdout.split()
        except Exception as ex:
            ev(event="fp32_pin_error", err=scrub(str(ex))[:200]); lines = None
        if lines is not None:
            ok = {lines[i + 1] for i in range(0, len(lines) - 1) if lines[i] == "OK"}
            miss = {lines[i + 1] for i in range(0, len(lines) - 1) if lines[i] == "MISSING"}
            res = dict(at_cr=cr(now), pinned={k: want[k] for k in sorted(ok)}, missing={k: want[k] for k in sorted(miss)})
            wj(OUT, res)
            for k in miss:
                if want[k]["why"] in ("green-unpublished", "head-current", "head-previous-until-current-published"):
                    ev(event="fp32_green_entry_missing", epoch=want[k]["epoch"], entry=k[:12], why=want[k]["why"])
            ev(event="fp32_pin", pinned=len(ok), missing=len(miss), epochs=sorted(v["epoch"] for v in want.values()))
    else:
        wj(OUT, res)
    try:
        alerts, seen = chain_alerts()
        for k, v in (res.get("missing") or {}).items():
            if v.get("why") == "head-current":
                alerts.append(dict(kind="head-master-missing-on-pod", epoch=v["epoch"], entry=k[:16]))
        prev = rj(ALERT, {}) or {}
        new = [a for a in alerts if a not in (prev.get("alerts") or [])]
        wj(ALERT, dict(at_cr=cr(now), ok=not alerts, alerts=alerts, scanned_from_epoch=ALERT_FROM_EPOCH, scanned=seen,
                       note="alert-only (fp32-fix v2): bootstrap-upcast / master chain break / head master missing on pod"))
        for a in new:
            ev(event="fp32_master_alert", **a)
    except Exception as ex:
        ev(event="fp32_alert_error", err=scrub(str(ex))[:200])
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["--dry"]:
        print(json.dumps(dict(want=wanted(), alerts=chain_alerts()), indent=1, default=str))
    else:
        print(json.dumps(tick(force=True), indent=1, default=str))
