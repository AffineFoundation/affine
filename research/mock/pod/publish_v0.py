#!/usr/bin/env python3
"""Gate a judge adapter on held-out pairs and publish it (VERSION file).

Includes the PERMANENT adapter-effect guard: a mis-keyed LoRA loads into
vLLM without error and applies nothing (shipped once, 2026-08-22, as no-op
judge v0). If the adapter does not measurably change scores vs base on fixed
pairs, publishing is REFUSED.

--src may be an HF/PEFT-format adapter dir; it is remapped to vLLM naming
into d_versions/<ver> (with the HF copy kept under hf/).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remap_lora import remap  # noqa: E402
from trackM_common import Judge, eval_held, judge_effect, now  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-url", default="http://127.0.0.1:8000")
    ap.add_argument("--base-27b", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--work", default="/dshare/koth")
    ap.add_argument("--ver", default="v0")
    ap.add_argument("--src", default="", help="HF-format adapter to remap; "
                    "default: d_versions/<ver> already prepared")
    ap.add_argument("--zero-shot", action="store_true",
                    help="also log the no-adapter baseline")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    d_tok = AutoTokenizer.from_pretrained(args.base_27b)
    judge = Judge(args.judge_url, args.base_27b, d_tok, 3584)

    held = [json.loads(x) for x in open(f"{args.work}/held_pairs.jsonl")]
    guard_pairs = [json.loads(x)
                   for x in open(f"{args.work}/seed_pairs.jsonl")][:8]

    def status(line):
        with open(f"{args.work}/status.log", "a") as fh:
            fh.write(f"{now()} [eval] {line}\n")
        print(line, flush=True)

    adir = f"{args.work}/d_versions/{args.ver}"
    if args.src:
        os.makedirs(adir, exist_ok=True)
        hf_dir = f"{adir}/hf"
        if os.path.abspath(args.src) != os.path.abspath(hf_dir):
            if os.path.isdir(hf_dir):
                shutil.rmtree(hf_dir)
            shutil.copytree(args.src, hf_dir,
                            ignore=shutil.ignore_patterns("hf"))
        hit, tot = remap(hf_dir, adir, prefix="language_model.model.")
        status(f"JUDGE {args.ver}: remapped {hit}/{tot} keys "
               f"(HF -> vLLM language_model.model.*)")

    if args.zero_shot:
        zs = eval_held(judge, held)
        status(f"JUDGE zero-shot (27B base, no adapter): held_acc="
               f"{zs['held_acc']} pos_bias={zs['pos_bias']} "
               f"matched={zs['matched_acc']} n={zs['n']}")

    meta = json.load(open(f"{adir}/train_meta.json"))
    judge.load_adapter(f"d_{args.ver}", adir)

    eff = judge_effect(args.judge_url, args.base_27b, f"d_{args.ver}",
                       d_tok, guard_pairs)
    if eff < 0.01:
        status(f"PUBLISH REFUSED {args.ver}: adapter-effect guard eff="
               f"{eff:.5f} (<0.01) -- adapter is a serving no-op")
        raise SystemExit(2)

    g = eval_held(judge, held)
    status(f"JUDGE {args.ver} gate: steps={meta['steps']} "
           f"train_loss={round(meta['loss'], 4)} effect={eff:.4f} "
           f"held_acc={g['held_acc']} pos_bias={g['pos_bias']} "
           f"matched={g['matched_acc']} n={g['n']} (band 0.6-0.9)")
    if g["held_acc"] > 0.9:
        status(f"JUDGE {args.ver} WARNING held_acc>0.9: miner reward may be "
               f"thin; publishing anyway, watch miner reward distribution")
    if g["held_acc"] < 0.55:
        status(f"JUDGE {args.ver} WARNING held_acc<0.55: barely beats chance; "
               f"publishing anyway (frozen reign will still rank)")
    with open(f"{args.work}/VERSION", "w") as fh:
        fh.write(args.ver + "\n")
    status(f"PUBLISH {args.ver}: adapter={adir} VERSION bumped -- miner may "
           f"start")


if __name__ == "__main__":
    main()
