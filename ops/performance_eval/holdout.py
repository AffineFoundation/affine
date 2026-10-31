#!/usr/bin/env python3
"""Freeze the SN120 performance holdout (Jacob pt911u 2026-10-31): MATH test (5,000, DigitalLearningGmbH/MATH-lighteval test via
RA/20261031-pool-overlap/math_test.json) minus every problem without a reference \\boxed answer and every problem that matches a pool task:
exact (overlap.py norm) or token-3-gram Jaccard >= 0.9, or Jaccard >= 0.8 with the same normalised answer. Exhaustive candidate search
(no document-frequency cap). Output: holdout.private.json (0600, NEVER publish), holdout.public.json (size + hash + rule only)."""
import json, os, hashlib, collections, sys
OV = os.path.expanduser("~/subnet120-rewrite/state/root-audits/20261031-pool-overlap"); MYDIR = os.path.dirname(os.path.abspath(__file__)); HERE = OV
sys.path.insert(0, OV); os.chdir(OV)
src = open(OV + "/overlap.py").read().split("NEAR = float")[0].replace("os.path.abspath(__file__)", repr(OV + "/overlap.py")); exec(src)   # norm, shingles, jacc, ans_norm, pool, test, P_norm, P_sh, inv
kept, removed = [], collections.Counter(); rm_detail = []
for ti, r in enumerate(test):
    ans = last_boxed_body(r["solution"] or "")
    if not ans: removed["no_boxed_answer"] += 1; continue
    n = norm(r["problem"])
    if n in P_norm: removed["exact"] += 1; rm_detail.append((ti, "exact", 1.0, P_norm[n])); continue
    sh = shingles(r["problem"]); cnt = collections.Counter()
    for g in sh:
        for k in inv.get(g, ()): cnt[k] += 1
    hit = None
    for k, c in cnt.items():
        if c < 0.8 * len(sh): continue          # J >= 0.8 needs |A&B| >= 0.8|A|
        j = jacc(sh, P_sh[k])
        if j >= 0.9 or (j >= 0.8 and ans_norm(pool[k]["answer"]) == ans_norm(ans)):
            if hit is None or j > hit[0]: hit = (j, k)
    if hit:
        removed["near_J>=0.9" if hit[0] >= 0.9 else "near_J>=0.8_same_answer"] += 1; rm_detail.append((ti, "near", round(hit[0], 4), hit[1])); continue
    kept.append(dict(id="math-test-%04d" % ti, test_index=ti, problem=r["problem"], answer=ans, type=r.get("type"), level=r.get("level"),
                     problem_sha256=hashlib.sha256(r["problem"].encode()).hexdigest()))
lines = "\n".join("%s:%s:%s" % (t["id"], t["problem_sha256"], hashlib.sha256(t["answer"].encode()).hexdigest()) for t in kept)
H = hashlib.sha256(lines.encode()).hexdigest()
priv = dict(version="sn120-perf-holdout-v1", n=len(kept), sha256=H, tasks=kept, removed=dict(removed), removed_detail=rm_detail)
p = MYDIR + "/holdout.private.json"
with open(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f: json.dump(priv, f)
pub = dict(version="sn120-perf-holdout-v1", n=len(kept), sha256=H,
           source="MATH test split (5,000 problems; DigitalLearningGmbH/MATH-lighteval test)", removed=dict(removed),
           rule="drop problems without a reference \\boxed answer and any problem matching a pool (MATH train) task: exact after "
                "normalisation, token-3-gram Jaccard >= 0.9, or Jaccard >= 0.8 with the same normalised answer",
           hash_rule="sha256 of newline-joined '<id>:<sha256(problem)>:<sha256(answer)>' over the frozen list in id order; the list itself is private")
json.dump(pub, open(MYDIR + "/holdout.public.json", "w"), indent=1)
print(json.dumps(pub, indent=1))
