import json, sys, collections
SRC = {"terminal_lego": 1, "scaleswe": 2, "swesmith": 1}   # attempts wanted (all king seats)
solved = {s: set() for s in SRC}; att = {s: collections.Counter() for s in SRC}
for l in open("/root/rollouts-data/state.jsonl"):
    try: r = json.loads(l)
    except ValueError: continue
    s = r.get("source")
    if s not in SRC: continue
    pid = r.get("policy_id") or ""
    if r.get("outcome") == "error" and not r.get("n_calls"): continue
    if pid.startswith("king_"): att[s][r["uid"]] += 1
    elif r.get("outcome") == "resolved" and not pid.startswith("backfill_"): solved[s].add(r["uid"])
out = {s: sorted(u for u in solved[s] if att[s][u] < SRC[s]) for s in SRC}
json.dump({"solved": {s: len(solved[s]) for s in SRC}, "owed_uids": out}, sys.stdout)
