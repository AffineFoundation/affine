#!/usr/bin/env python3
"""Diagnose judge_miss: replay recent reign-2 duel pairs against the live
judge (:8000) exactly the way koth_eval.py scores them, and record what the
top-20 next-token candidates actually are when neither A nor B appears.

Read-only with respect to the driver; sends ~100 single-token completion
requests to :8000 (a few percent of one normal eval batch).
"""
import gzip
import json
import math
import os
import sys

import requests

sys.path.insert(0, "/root/work")
os.environ.setdefault("HF_HOME", "/dshare/hf")

from gad_common import both_text  # noqa: E402
from judge_common import fit_ids  # noqa: E402
from trackM_common import render_prefix  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

JUDGE_URL = "http://127.0.0.1:8000"
D_MAX_LEN = 3584
N_PAIRS = 24

# find the currently served discriminator adapter name (d_v2 etc.)
models = [m["id"] for m in requests.get(
    f"{JUDGE_URL}/v1/models", timeout=30).json()["data"]]
d_names = sorted(m for m in models if m.startswith("d_"))
if not d_names:
    sys.exit(f"no d_* adapter on judge server; models={models}")
D_NAME = d_names[-1]
BASE = "Qwen/Qwen3.8-27B"
print(f"judge server models: {models}; probing adapter={D_NAME} vs base={BASE}")

# last N valid reign-2 rows from the archive
rows = []
with open("/dshare/koth/archive.jsonl", "rb") as fh:
    fh.seek(0, 2)
    fh.seek(max(0, fh.tell() - 8_000_000))
    fh.readline()
    for line in fh:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("dver") == "v2" and r.get("valid") and r.get("score") is not None:
            rows.append(r)
rows = rows[-N_PAIRS:]
tids = {r["turn_id"] for r in rows}
print(f"replaying {len(rows)} archived v2 pairs, {len(tids)} distinct turns")

prefixes = {}
with gzip.open("/root/work/turns.jsonl.gz", "rt") as fh:
    for line in fh:
        if len(prefixes) == len(tids):
            break
        try:
            t = json.loads(line)
        except Exception:
            continue
        if t.get("turn_id") in tids:
            prefixes[t["turn_id"]] = t["prefix"]
missing = tids - set(prefixes)
if missing:
    print(f"WARNING: {len(missing)} turn_ids not found in turns.jsonl.gz")

tok = AutoTokenizer.from_pretrained(BASE)


def one(model, ids):
    r = requests.post(f"{JUDGE_URL}/v1/completions", json={
        "model": model, "prompt": ids, "max_tokens": 1,
        "temperature": 0.0, "logprobs": 20}, timeout=300)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    lp = ch.get("logprobs") or {}
    tops = (lp.get("top_logprobs") or [{}])[0] or {}
    gen = (lp.get("tokens") or [""])[0]
    pa = sum(math.exp(v) for t, v in tops.items() if t.strip() == "A")
    pb = sum(math.exp(v) for t, v in tops.items() if t.strip() == "B")
    return pa, pb, gen, tops


summary = {}
miss_examples = []
for model in (D_NAME, BASE):
    calls = misses = 0
    for r in rows:
        if r["turn_id"] not in prefixes:
            continue
        pfx = render_prefix(prefixes[r["turn_id"]], D_MAX_LEN * 6)
        mine = both_text(r["z"], r["y"])
        ref = r["ref"]
        for a, b in ((ref, mine), (mine, ref)):
            ids = fit_ids(tok, pfx, a, b, D_MAX_LEN)
            try:
                pa, pb, gen, tops = one(model, ids)
            except Exception as e:
                print(f"  request error: {e}")
                continue
            calls += 1
            if pa + pb <= 0:
                misses += 1
                if model == D_NAME and len(miss_examples) < 6:
                    top5 = sorted(tops.items(), key=lambda kv: -kv[1])[:5]
                    miss_examples.append(
                        {"turn": r["turn_id"], "gen": gen, "top5": top5})
    summary[model] = (calls, misses)
    print(f"{model}: calls={calls} misses={misses} "
          f"miss_rate={misses / max(calls, 1):.3f}")

print("\nexample miss top-5 tokens (adapter):")
for ex in miss_examples:
    print(f"  turn={ex['turn'][:50]} generated={ex['gen']!r}")
    for t, v in ex["top5"]:
        print(f"      {t!r}: {v:.3f}")
