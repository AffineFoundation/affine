#!/usr/bin/env python3
"""[fix] Re-measure Track S judge readouts with CONSTRAINED scoring.

Quantifies how much of the flat-0.5 story was top-20-miss dilution:
  * fresh held-out samples (teacher x2, g_r12, g_r0) -> live/ctrl/tt pairs,
    saved to disk so later D versions can be scored on the SAME pairs;
  * recent d_pending pairs (the drifted rounds) re-scored;
  * every set scored under BOTH D adapters (v9, v12) with BOTH methods
    (OLD unconstrained top-20 w/ 0.5 fallback vs NEW forced-token).

Modes:
  sample:      build + save fresh held pairs, then score
  score-saved: score saved held pairs under --adapters (new method only)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "/root/work")
import requests  # noqa: E402

from gad_common import both_text, load_turns  # noqa: E402
from trackM_common import Judge, now, render_prefix, sample_model  # noqa: E402

W = "/dshare/trackS"


def status(line):
    with open(f"{W}/status.log", "a") as fh:
        fh.write(f"{now()} [fix] {line}\n")
    print(line, flush=True)


class OldJudge(Judge):
    """The pre-fix scorer: unconstrained top-20 readout, 0.5 on miss."""

    def _one(self, ids):
        body = {"model": self.serve, "prompt": ids, "max_tokens": 1,
                "temperature": 0.0, "logprobs": 20}
        r = requests.post(f"{self.url}/v1/completions", json=body,
                          timeout=300)
        r.raise_for_status()
        lp = r.json()["choices"][0].get("logprobs") or {}
        tops = (lp.get("top_logprobs") or [{}])[0] or {}
        pa = pb = 0.0
        for t, v in tops.items():
            s = t.strip()
            if s == "A":
                pa += math.exp(v)
            elif s == "B":
                pb += math.exp(v)
        if pa + pb <= 0:
            self.miss += 1
            return 0.5
        return pa / (pa + pb)


def summarize(scores):
    ok = [s for s in scores if s]
    if not ok:
        return {"n": 0}
    acc = sum(1 for s in ok if s["p_teacher"] > 0.5) / len(ok)
    exact05 = sum(1 for s in ok if abs(s["p_teacher"] - 0.5) < 1e-9)
    return {"acc": round(acc, 4),
            "p": round(sum(s["p_teacher"] for s in ok) / len(ok), 4),
            "exact05": exact05, "n": len(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sample",
                    choices=["sample", "score-saved"])
    ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--sample-url", default="http://127.0.0.1:8000")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8001")
    ap.add_argument("--adapters", default="v9,v12",
                    help="D versions to score under (dirs in d_versions/)")
    ap.add_argument("--g-cur", default="g_r12")
    ap.add_argument("--held-n", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1792)
    ap.add_argument("--d-max-len", type=int, default=3584)
    ap.add_argument("--pairs-file", default=f"{W}/remeasure_held_pairs.json")
    ap.add_argument("--pending-rounds", default="10,11,12,13")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    vers = args.adapters.split(",")

    if args.mode == "sample":
        turns = load_turns("/root/work/turns.jsonl.gz")
        held_tids = [t for t in open("/root/work/held_tids.txt").read().split()
                     if t in turns]
        held_fixed = sorted(held_tids)[: args.held_n]

        t0 = time.time()
        t_ref, tvr, _ = sample_model(args.sample_url, args.base, tok, turns,
                                     held_fixed, 2, args.temp,
                                     args.max_tokens)
        g_cur, gvr, _ = sample_model(args.sample_url, args.g_cur, tok, turns,
                                     held_fixed, 1, args.temp,
                                     args.max_tokens)
        g_zero, zvr, _ = sample_model(args.sample_url, "g_r0", tok, turns,
                                      held_fixed, 1, args.temp,
                                      args.max_tokens)
        sets = {"live": {}, "ctrl": {}, "tt": {}}
        for tid in held_fixed:
            refs = t_ref.get(tid) or []
            if not refs or not refs[0]["valid"]:
                continue
            pfx = render_prefix(turns[tid], args.d_max_len * 6)
            ref = both_text(refs[0]["z"], refs[0]["y"])

            def put(name, c):
                if c and c["valid"]:
                    sets[name][tid] = {"prefix_text": pfx, "ref": ref,
                                       "mine": both_text(c["z"], c["y"]),
                                       "mine_len": len(c["z"])}
            put("live", (g_cur.get(tid) or [None])[0])
            put("ctrl", (g_zero.get(tid) or [None])[0])
            put("tt", refs[1] if len(refs) > 1 else None)
        json.dump(sets, open(args.pairs_file, "w"))
        status(f"CONSTRAINED-SCORING deployed; fresh held samples "
               f"(n live/ctrl/tt = {len(sets['live'])}/{len(sets['ctrl'])}/"
               f"{len(sets['tt'])}, valid t/g/g0 = {tvr:.3f}/{gvr:.3f}/"
               f"{zvr:.3f}, wall={time.time()-t0:.0f}s) saved to "
               f"{os.path.basename(args.pairs_file)}")
    else:
        sets = json.load(open(args.pairs_file))

    # make sure both adapters are registered on the judge server
    for v in vers:
        try:
            requests.post(f"{args.judge_url}/v1/load_lora_adapter",
                          json={"lora_name": f"d_{v}",
                                "lora_path": f"{W}/d_versions/{v}"},
                          timeout=600)
        except Exception:
            pass

    for v in vers:
        for name in ("live", "ctrl", "tt"):
            pairs = list(sets[name].values())
            new = summarize(Judge(args.judge_url, f"d_{v}", tok,
                                  args.d_max_len).score_pairs(pairs))
            if args.mode == "sample":
                oj = OldJudge(args.judge_url, f"d_{v}", tok, args.d_max_len)
                old = summarize(oj.score_pairs(pairs))
                old["miss"] = oj.miss
                status(f"re-measured {v} held_{name}: NEW acc={new.get('acc')} "
                       f"p={new.get('p')} n={new.get('n')} | OLD "
                       f"acc={old.get('acc')} p={old.get('p')} "
                       f"exact0.5={old.get('exact05')} miss={old.get('miss')} "
                       f"of {2*old.get('n', 0)} calls")
            else:
                status(f"re-measured {v} held_{name}: NEW acc={new.get('acc')} "
                       f"p={new.get('p')} n={new.get('n')}")

    # recent-round training pairs (the drifted ones)
    want = {int(x) for x in args.pending_rounds.split(",")}
    rows = [json.loads(x) for x in open(f"{W}/d_pending.jsonl")
            if json.loads(x).get("round") in want]
    if rows:
        pairs = [{"prefix_text": r["prefix_text"], "ref": r["ref"],
                  "mine": r["mine"]} for r in rows]
        for v in vers:
            nj = Judge(args.judge_url, f"d_{v}", tok, args.d_max_len)
            new = summarize(nj.score_pairs(pairs))
            oj = OldJudge(args.judge_url, f"d_{v}", tok, args.d_max_len)
            old = summarize(oj.score_pairs(pairs))
            status(f"re-measured rounds {sorted(want)} G-best pairs under {v}: "
                   f"NEW d_acc={new.get('acc')} p_teacher={new.get('p')} "
                   f"(fool={round(1-new.get('p', 0.5), 4)}) n={new.get('n')} | "
                   f"OLD d_acc={old.get('acc')} p={old.get('p')} "
                   f"exact0.5={old.get('exact05')} miss={oj.miss}")


if __name__ == "__main__":
    main()
