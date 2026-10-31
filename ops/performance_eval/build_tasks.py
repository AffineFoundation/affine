#!/usr/bin/env python3
"""Build the pod task file (PRIVATE, 0600): full = the 7,496 affine_math pool tasks in env index order (prompt + answer from the
pinned assets/original-math7496.tasks.json used by the probe's lenient grader; prompt token ids from the probe's own prep cache,
which the evaluator's env reset + chat template produced) + the frozen holdout (holdout.private.json)."""
import json, os, hashlib, glob
HERE = os.path.dirname(os.path.abspath(__file__)); R = "/home/const/subnet120-rewrite"
A = json.load(open(R + "/assets/original-math7496.tasks.json"))
pool = json.load(open(R + "/state/root-audits/20261031-pool-overlap/pool.json"))
PCF = glob.glob(R + "/state/root-audits/20261031-shadow-renorm-step/probe-runtime/prepcache/*.json")[0]
PC = json.load(open(PCF))["rows"]
H = json.load(open(HERE + "/holdout.private.json"))
assert len(A) == 7496 == len(pool) == len(PC), (len(A), len(pool), len(PC))
full = []; mism = 0
for i, t in enumerate(A):
    d = t["data"]
    if d["prompt"] != pool[i]["problem"] or d["answer"] != pool[i]["answer"]: mism += 1
    full.append(dict(set="full", id=str(i), prompt=d["prompt"], answer=d["answer"], ids=PC[str(i)][1]))
hold = [dict(set="holdout", id=t["id"], prompt=t["problem"], answer=t["answer"]) for t in H["tasks"]]
out = dict(version="sn120-perf-tasks-v1", full_n=len(full), holdout_n=len(hold), holdout_sha256=H["sha256"],
           prepcache=os.path.basename(PCF), tasks=full + hold)
p = HERE + "/tasks.private.json"
with open(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f: json.dump(out, f)
print("full", len(full), "holdout", len(hold), "assets-vs-pool.json mismatches", mism, "sha256", hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
