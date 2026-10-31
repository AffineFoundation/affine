#!/usr/bin/env python3
"""King (head) history from the gate ledger: every applied GREEN = a new king. Prints/writes kings.json (public facts only)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_king import descriptor
G = "/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/LEDGER.public.jsonl"
def kings():
    rows = [json.loads(l) for l in open(G) if l.strip()]
    out = []
    for r in rows:
        if r.get("decision") == "green" and r.get("applied") in (None, "green", True, "promoted", "promote") :
            out.append(dict(king=r["candidate"], epoch=r["epoch"], parent=r["head"], crowned_at=r["timestamps"].get("decided_at"),
                            crowned_at_cr=r.get("decided_at_cr"), z=r.get("z")))
    return out
if __name__ == "__main__":
    ks = kings()
    for k in ks: k["descriptor"] = descriptor(k["king"]) is not None
    json.dump(ks, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kings.json"), "w"), indent=1)
    for k in ks: print(k["epoch"], k["king"][:8], k["parent"][:8], k["crowned_at_cr"], k["descriptor"])
    print("applied values:", sorted({str(json.loads(l).get("applied")) for l in open(G) if '"green"' in l}))
