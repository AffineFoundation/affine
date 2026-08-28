#!/usr/bin/env python3
"""Compact turn manifest so the pod does not need the 7 pairs_* files.

For every turn that had at least one non-empty old-teacher reference (same
selection rule gad_gen.load_teacher used), emit turn_id, repo, split. The pod
generates its own Qwen3-32B teacher rollouts; the old GLM refs are only a
turn-selection filter here so the turn pool matches earlier experiments.
"""
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, "/home/const/subnet120/research/scripts")
from disc_text import as_list, normalize  # noqa: E402

DATA = "/home/const/subnet120/research/data/disc_pairs"
OUT = "/home/const/subnet120/research/trackB/turn_meta.jsonl.gz"

seen = {}
for f in sorted(glob.glob(os.path.join(DATA, "pairs_*.jsonl.gz"))):
    with gzip.open(f, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            tid = r["turn_id"]
            if tid in seen:
                continue
            refs = [normalize(x) for x in as_list(r.get("teacher_z"))]
            if not any(refs):
                continue
            seen[tid] = {"turn_id": tid, "repo": (r.get("repo") or "").lower(),
                         "split": r.get("split")}

with gzip.open(OUT, "wt") as fh:
    for tid in sorted(seen):
        fh.write(json.dumps(seen[tid]) + "\n")

n_tr = sum(1 for v in seen.values() if v["split"] == "train")
n_te = sum(1 for v in seen.values() if v["split"] == "test")
print(f"turns={len(seen)} train={n_tr} test={n_te} -> {OUT}")
