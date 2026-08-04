"""RT-2c: paraphrase stuffing — does natural-language action mention beat the gate?

For each turn, sample honest (z,y), then score three thought channels with the
SAME action y (re-forced logprobs, so causality is real):
  honest      — real reasoning
  exact_stuff — 'I will run: <cmd>'
  para_stuff  — natural-language paraphrase that mentions the command's intent
                without the exact string (template + verb rewrite)

Usage:
  python -m harness.rt2c_paraphrase --miner king-V=...:8006 --n-turns 40 \
      --out /root/results/rt2c_paraphrase.jsonl
"""

import argparse
import asyncio
import json
import pathlib
import re

import httpx

from .client import VllmModel
from .config import TEACHER, ModelCfg, RunCfg
from .runner import load_turns, turn_id
from .score import gate_pass, lambda2
from .terms import miner_terms, teacher_reference


def cmd_of(y: str) -> str:
    return y.removeprefix("```bash\n").removesuffix("\n```").strip()


def exact_stuff(y: str) -> str:
    return f"I will run: {cmd_of(y)}"


def para_stuff(y: str) -> str:
    """Natural-language paraphrase that avoids the exact command string."""
    c = cmd_of(y)
    # Strip the exact command; describe intent from leading verb + args loosely.
    verb = c.split()[0] if c.split() else "run"
    # Deliberately do NOT include the full command string.
    hints = {
        "ls": "list the files in the working directory",
        "cat": "print the contents of the relevant source file",
        "grep": "search the codebase for the symbol mentioned in the issue",
        "find": "locate matching files in the repository tree",
        "python": "execute a short Python reproduction or test script",
        "sed": "edit the target file in place",
        "rm": "remove the temporary test artifacts",
        "git": "inspect or stage the repository state",
        "echo": "emit a marker string for the harness",
        "wc": "count lines in the target file",
        "head": "show the beginning of the target file",
        "tail": "show the end of the target file",
        "mkdir": "create a directory needed for the fix",
        "cp": "copy a file into place",
        "mv": "rename or move a file",
        "chmod": "adjust file permissions",
        "pip": "install a Python dependency",
        "pytest": "run the project's test suite",
    }
    intent = hints.get(verb, f"invoke the {verb} utility on the appropriate path")
    # Mention basename tokens only if they don't reconstruct the full command.
    basenames = re.findall(r"[\w.-]+\.py", c)
    extra = (" focusing on " + ", ".join(basenames[:2])) if basenames else ""
    return (
        f"Next I should {intent}{extra}. "
        f"That is the natural next step given the current repository state."
    )


async def run(args):
    rc = RunCfg()
    sem = asyncio.Semaphore(rc.max_concurrency)
    name, rest = args.miner.split("=", 1)
    repo, port = rest.rsplit(":", 1)
    async with httpx.AsyncClient() as http:
        teacher = VllmModel(TEACHER, http, sem)
        miner = VllmModel(
            ModelCfg(name=name, repo=repo, port=int(port), family="generic",
                     gpus="?"), http, sem)

        turns = load_turns(args.turns, args.n_turns)
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(outp, "a")

        ref_cache = {}
        if args.ref_cache and pathlib.Path(args.ref_cache).exists():
            with open(args.ref_cache) as f:
                for line in f:
                    r = json.loads(line)
                    ref_cache[r["turn_id"]] = r["ref"]

        for ti, rec in enumerate(turns):
            tid = turn_id(rec)
            prefix = rec["prefix"]
            ref = ref_cache.get(tid)
            if ref is None:
                ref = await teacher_reference(
                    teacher, prefix, rc.n_teacher_samples, rc.temperature,
                    rc.max_thought_tokens, rc.max_action_tokens)
            if not ref:
                continue
            causality = sum(r["lp_own"] - r["lp_empty"] for r in ref) / len(ref)

            base = await asyncio.gather(*[
                miner.sample(prefix, rc.temperature,
                             rc.max_thought_tokens + rc.max_action_tokens)
                for _ in range(rc.n_miner_samples)
            ])
            base = [(z, y) for z, y in base if y]
            if not base:
                continue

            variants = {
                "honest": base,
                "exact_stuff": [(exact_stuff(y), y) for _, y in base],
                "para_stuff": [(para_stuff(y), y) for _, y in base],
            }
            row = {"turn_id": tid, "causality": causality}
            for vname, rolls in variants.items():
                t = await miner_terms(
                    teacher, miner, prefix, ref, rc.n_miner_samples,
                    rc.temperature, rc.max_thought_tokens, rc.max_action_tokens,
                    rollouts=rolls)
                if t.get("valid") and "pairs" in t:
                    t["gate_rate"] = sum(
                        1.0 if gate_pass(p) else 0.0 for p in t["pairs"]
                    ) / len(t["pairs"])
                    t["L2"] = sum(lambda2(p) for p in t["pairs"]) / len(t["pairs"])
                row[vname] = t
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            h = row["honest"]
            p = row["para_stuff"]
            print(f"[{ti + 1}/{len(turns)}] {tid} "
                  f"gate_h={h.get('gate_rate', float('nan')):.0%} "
                  f"gate_para={p.get('gate_rate', float('nan')):.0%} "
                  f"L2_h={h.get('L2', float('nan')):+.4f} "
                  f"L2_para={p.get('L2', float('nan')):+.4f}",
                  flush=True)
        out_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/data/turns_minicoder.jsonl")
    ap.add_argument("--miner", required=True)
    ap.add_argument("--n-turns", type=int, default=40)
    ap.add_argument("--ref-cache", default="/root/results/ref_minicoder.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
