#!/usr/bin/env python3
"""Final sweep table: corrected-serving rows only, grouped by arm.

Reads results.jsonl, keeps rows with temperature>0 (the corrected serving
config), and prints resolve-rate, Submitted, RepeatedAction per run so the
arm-vs-ctrl comparison is one glance.
"""
import json

LEDGER = "/home/const/subnet120/research/factory/results.jsonl"

rows = []
for line in open(LEDGER):
    r = json.loads(line)
    if r.get("state") != "completed" or not r.get("temperature"):
        continue
    ex = r.get("exits") or {}
    rows.append({
        "arm": r["arm"], "step": r["step"], "n": r["n"],
        "think": bool(r.get("think")),
        "score": r["score"], "resolved": r["n_resolved"],
        "submitted": ex.get("Submitted", 0),
        "repact": ex.get("RepeatedAction", 0),
        "cwe": ex.get("ContextWindowExceededError", 0),
        "job": r["job"],
    })

rows.sort(key=lambda r: (-r["n"], -(r["score"] or 0), r["arm"]))
hdr = f"{'arm':<28} {'step':>5} {'n':>3} {'think':>5} {'score':>6} " \
      f"{'resolved':>8} {'submit':>6} {'repact':>6} {'ctxexc':>6}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['arm']:<28} {r['step']:>5} {r['n']:>3} "
          f"{str(r['think']):>5} {r['score']:>6} {r['resolved']:>8} "
          f"{r['submitted']:>6} {r['repact']:>6} {r['cwe']:>6}")
