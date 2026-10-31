#!/usr/bin/env python3
"""Every head (king) the network has had, from the lifecycle record: head of epoch e = TRAINING-RESULT input_checkpoint. A king's
'crowned' time = completion time (payload.at) of the training that produced it (pre-gate: every trained checkpoint became the next head;
gate era: the GREEN decision time from the gate ledger). Joined with the R2 listing (r2_checkpoints.json) -> recoverable or not.
Writes head_history.json (public facts: ids, epochs, times)."""
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
L = "/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs"
LED = "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/LEDGER.public.jsonl"
r2 = json.load(open(HERE + "/r2_checkpoints.json"))
made = {}; heads = []   # (epoch, input)
for d in sorted(glob.glob(L + "/*"), key=lambda p: int(p.rsplit("/", 1)[1]) if p.rsplit("/", 1)[1].isdigit() else -1):
    e = d.rsplit("/", 1)[1]
    if not e.isdigit(): continue
    try: r = json.load(open(d + "/TRAINING-RESULT.json"))
    except Exception: continue
    p = r.get("payload") or r; rep = p.get("report") or {}; pr = (rep.get("publication_receipt") or {}).get("payload") or {}
    i, o = pr.get("input_checkpoint") or rep.get("checkpoint"), pr.get("output_checkpoint") or rep.get("output_checkpoint")
    if o: made.setdefault(o, dict(epoch=int(e), at=p.get("at") or rep.get("completed_at")))
    if i: heads.append((int(e), i))
green = {}
for l in open(LED):
    try: x = json.loads(l)
    except Exception: continue
    if x.get("decision") == "green" and x.get("applied") == "green":
        green[x["candidate"]] = dict(epoch=x["epoch"], at=(x.get("timestamps") or {}).get("decided_at"), at_cr=x.get("decided_at_cr"))
seen = {}; order = []
for e, ck in heads:
    if ck not in seen:
        seen[ck] = e; order.append(ck)
for ck in green:
    if ck not in seen: order.append(ck); seen[ck] = green[ck]["epoch"] + 1
out = []
for ck in order:
    g = green.get(ck); m = made.get(ck)
    crowned_epoch = g["epoch"] if g else (m["epoch"] if m else None)
    at = g["at"] if g else (m["at"] if m else None)
    out.append(dict(king=ck, first_head_epoch=seen[ck], made_in_epoch=m["epoch"] if m else None, crowned_epoch=crowned_epoch, crowned_at=at,
                    era="gate" if g else ("pre-gate" if m else "initial"), in_r2=ck in r2, r2_model_bytes=(r2.get(ck) or {}).get("model_bytes")))
json.dump(out, open(HERE + "/head_history.json", "w"), indent=1)
import time
cr = lambda t: time.strftime("%Y-%m-%d %H:%M CR", time.gmtime(t - 21600)) if t else "?"
rec = [k for k in out if k["in_r2"]]
print("distinct heads:", len(out), "recoverable from R2:", len(rec), "| gate-era:", sum(k["era"] == "gate" for k in out), "pre-gate:", sum(k["era"] == "pre-gate" for k in out), "initial:", sum(k["era"] == "initial" for k in out))
print("first:", out[0]["king"][:12], out[0]["first_head_epoch"], cr(out[0]["crowned_at"]), "| first recoverable:", rec[0]["king"][:12], rec[0]["first_head_epoch"], cr(rec[0]["crowned_at"]))
print("missing from R2:", [(k["king"][:8], k["first_head_epoch"]) for k in out if not k["in_r2"]][:20])
print("R2 checkpoints that were never a head:", len(set(r2) - {k["king"] for k in out}))
