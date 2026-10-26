#!/usr/bin/env python3
"""Diagnose kept=0 in the contrastive arm.

For a sample of train turns: generate k miner thoughts, then report the two
components of the contrastive score separately:
  raw   = [lp_T(y_true|h,z) - lp_T(y_true|h)] / n
  dpen  = max(0, [lp_T(y_dis|h,z) - lp_T(y_dis|h)] / n)
  S     = raw - dpen
Tells us whether thoughts fail because raw lift is negative (hint never helps)
or because the distractor lift eats it (specificity too strict).
"""
import json
import os
import random
import shutil
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

sys.path.insert(0, "/root/work")
from transformers import AutoTokenizer  # noqa: E402
from gad_common import load_rollout_cache, load_turns, bash_body  # noqa: E402
import reason_family_loop as rfl  # noqa: E402

OUT = "/root/work/diag_contrastive"
os.makedirs(OUT, exist_ok=True)
# reuse cached baselines without racing the main loop's append handle
src = "/dshare/ckpt/contrastive/baselines.jsonl"
if os.path.exists(src):
    shutil.copy(src, os.path.join(OUT, "baselines.jsonl"))

args = SimpleNamespace(score="contrastive", teacher="Qwen/Qwen3.6-35B-A3B",
                       teacher_url="http://127.0.0.1:8010",
                       miner_url="http://127.0.0.1:8021",
                       out=OUT, max_z_tokens=256)

ttok = AutoTokenizer.from_pretrained(args.teacher)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
turns = load_turns("/root/work/turns.jsonl.gz")
cache = load_rollout_cache("/root/work/teacher_train.jsonl")
tids = [t for t in cache if t in turns]
rng = random.Random(7)
sample = rng.sample(tids, 32)
scorer = rfl.Scorer(args, ttok, turns, cache, {})

rows = []


def work(tid):
    ep = rfl.elicit_prompt(args, tok, turns, {}, tid)
    if ep is None or len(ep) > 60000:
        return
    try:
        zs = rfl.gen_thoughts(args, tok, ep, 4, "Qwen/Qwen3-8B")
    except Exception as e:
        print(f"genfail {tid}: {e}", flush=True)
        return
    p, y, key = scorer.targets(tid)
    rolls = cache[tid]
    dis = None
    if len(rolls) > 1 and bash_body(rolls[1]["raw"]) != bash_body(rolls[0]["raw"]):
        dis = rolls[1]["raw"]
    for z in zs:
        z = z.strip()
        if not z:
            continue
        try:
            lp0, n0 = scorer._base(key, p, y)
            pz = scorer.hinted_prompt(tid, z)
            lp1, n1 = rfl.lp_of(args.teacher_url, args.teacher, ttok, pz, y)
            raw = (lp1 - lp0) / max(min(n0, n1), 1)
            dpen = 0.0
            if dis is not None:
                dlp0, dn0 = scorer._base(f"d:{tid}", p, dis)
                dlp1, dn1 = rfl.lp_of(args.teacher_url, args.teacher, ttok, pz, dis)
                dpen = max((dlp1 - dlp0) / max(min(dn0, dn1), 1), 0.0)
            rows.append({"tid": tid, "raw": raw, "dpen": dpen,
                         "S": raw - dpen, "has_dis": dis is not None,
                         "z_toks": len(tok(z)["input_ids"])})
        except Exception as e:
            print(f"scorefail {tid}: {e}", flush=True)


with ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(work, sample))

if not rows:
    print("NO ROWS — everything failed upstream")
    sys.exit(1)

raws = [r["raw"] for r in rows]
dpens = [r["dpen"] for r in rows]
svals = [r["S"] for r in rows]
n = len(rows)
print(f"n_thoughts={n} turns={len(set(r['tid'] for r in rows))} "
      f"with_distractor={sum(r['has_dis'] for r in rows)}")
print(f"raw:  median={statistics.median(raws):.4f} "
      f"frac>0={sum(v > 0 for v in raws)/n:.2f} max={max(raws):.4f}")
print(f"dpen: median={statistics.median(dpens):.4f} "
      f"frac>0={sum(v > 0 for v in dpens)/n:.2f}")
print(f"S:    median={statistics.median(svals):.4f} "
      f"frac>0={sum(v > 0 for v in svals)/n:.2f} max={max(svals):.4f}")
with open(os.path.join(OUT, "rows.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("DIAG_DONE")
