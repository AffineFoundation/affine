#!/usr/bin/env python3
"""[fix] A1 verification: replay ~50 archived duel pairs against the live
judge (:8000, current d_* adapter) and score each slot order with BOTH
methods:

  OLD: unconstrained call, read top-20 logprobs, miss -> 0.5 fallback
  NEW: two forced-token calls (allowed_token_ids=[A] / [B]), exact raw
       logprob ratio, miss structurally impossible

Reports miss rates, exactly-0.5 rates and old-vs-new score agreement on the
calls where the old method did NOT miss. Read-only wrt the driver; ~300
single-token requests.
"""
import json
import math
import os
import sys

import requests

sys.path.insert(0, "/root/work")
os.environ.setdefault("HF_HOME", "/dshare/hf")

from gad_common import both_text                     # noqa: E402
from judge_common import ab_token_ids, fit_ids       # noqa: E402
from trackM_common import render_prefix              # noqa: E402
from transformers import AutoTokenizer               # noqa: E402

JUDGE_URL = "http://127.0.0.1:8000"
D_MAX_LEN = 3584
N_PAIRS = 50
BASE = "Qwen/Qwen3.8-27B"

models = [m["id"] for m in requests.get(
    f"{JUDGE_URL}/v1/models", timeout=30).json()["data"]]
d_names = sorted(m for m in models if m.startswith("d_"))
if not d_names:
    sys.exit(f"no d_* adapter on judge server; models={models}")
D_NAME = d_names[-1]
print(f"probing adapter={D_NAME} on {JUDGE_URL}")

rows = []
with open("/dshare/koth/archive.jsonl", "rb") as fh:
    fh.seek(0, 2)
    fh.seek(max(0, fh.tell() - 12_000_000))
    fh.readline()
    for line in fh:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("valid") and r.get("score") is not None:
            rows.append(r)
rows = rows[-N_PAIRS:]
tids = {r["turn_id"] for r in rows}
print(f"replaying {len(rows)} archived pairs, {len(tids)} distinct turns")

import gzip  # noqa: E402
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

tok = AutoTokenizer.from_pretrained(BASE)
AB = ab_token_ids(tok)


def old_method(ids):
    r = requests.post(f"{JUDGE_URL}/v1/completions", json={
        "model": D_NAME, "prompt": ids, "max_tokens": 1,
        "temperature": 0.0, "logprobs": 20}, timeout=300)
    r.raise_for_status()
    tops = ((r.json()["choices"][0].get("logprobs") or {})
            .get("top_logprobs") or [{}])[0] or {}
    pa = sum(math.exp(v) for t, v in tops.items() if t.strip() == "A")
    pb = sum(math.exp(v) for t, v in tops.items() if t.strip() == "B")
    if pa + pb <= 0:
        return 0.5, True
    return pa / (pa + pb), False


def new_method(ids):
    lps = []
    for tid in AB:
        r = requests.post(f"{JUDGE_URL}/v1/completions", json={
            "model": D_NAME, "prompt": ids, "max_tokens": 1,
            "temperature": 0.0, "logprobs": 1,
            "allowed_token_ids": [tid]}, timeout=300)
        r.raise_for_status()
        tl = (r.json()["choices"][0].get("logprobs") or {}).get(
            "token_logprobs")
        if not tl or tl[0] is None:
            return None, True
        lps.append(float(tl[0]))
    la, lb = lps
    return 1.0 / (1.0 + math.exp(min(max(lb - la, -60.0), 60.0))), False


calls = 0
old_miss = new_miss = 0
old_half = new_half = 0
diffs = []
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

jobs = []
for r in rows:
    if r["turn_id"] not in prefixes:
        continue
    pfx = render_prefix(prefixes[r["turn_id"]], D_MAX_LEN * 6)
    mine = both_text(r["z"], r["y"])
    for a, b in ((r["ref"], mine), (mine, r["ref"])):
        jobs.append(fit_ids(tok, pfx, a, b, D_MAX_LEN))


def work(ids):
    return old_method(ids), new_method(ids)


with ThreadPoolExecutor(max_workers=12) as ex:
    for (so, mo), (sn, mn) in ex.map(work, jobs):
        calls += 1
        old_miss += int(mo)
        new_miss += int(mn)
        old_half += int(abs(so - 0.5) < 1e-9)
        if sn is not None:
            new_half += int(abs(sn - 0.5) < 1e-9)
        if not mo and sn is not None:
            diffs.append(abs(so - sn))

print(f"calls={calls}")
print(f"OLD  miss={old_miss} miss_rate={old_miss/max(calls,1):.3f} "
      f"exactly_0.5={old_half}")
print(f"NEW  miss={new_miss} miss_rate={new_miss/max(calls,1):.3f} "
      f"exactly_0.5={new_half}")
if diffs:
    print(f"agreement on old-hit calls: n={len(diffs)} "
          f"mean|d|={sum(diffs)/len(diffs):.5f} max|d|={max(diffs):.5f}")
