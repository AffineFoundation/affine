#!/usr/bin/env python3
"""Generate affine_scitext variants from open problem sets and verify them.

Seeds (text-only rows, numeric or closed-form answers):
  * Hothan/OlympiadBench  config OE_TO_physics_en_COMP (Apache-2.0)
  * xw27/scibench         default/train (MIT; college textbook problems)
  * TIGER-Lab/TheoremQA   test, rows without a picture (MIT)
HLE is never loaded. For each seed the teacher writes `--variants` new
problems (changed constants, changed asked quantity, same physics) as JSON
{problem, answer, unit, solution_code}; kept iff

  1. `solution_code` (plain Python, math/sympy/numpy) prints a value that
     matches `answer` (numeric 1 % or math-verify) — the answer is checked
     by computation, not by the teacher's word;
  2. the teacher re-solves the variant BLIND from the env prompt in 1-3 of 3
     attempts (0/3 = too hard to reference).

Budget: --budget-usd (plan USD 110 for ~12k variants; default 130).
  python -m affine_scitext_v1.generate --epoch 1 --seeds 6000 --variants 2 --budget-usd 130
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys

from datasets import load_dataset

from affine_gen_v1.store import GenTaskStore, gen_uid
from affine_gen_v1.teacher import BudgetExceeded, Spend, TeacherClient
from affine_gen_v1.verify import answers_equal, boxed_answer, run_python

from affine_scitext_v1.taskset import PACKAGE_DIR, SOURCE, SYSTEM

GEN_SYSTEM = (
    "You write new quantitative science problems by varying a given seed problem: change the numbers, the "
    "asked quantity or the scenario while keeping the underlying physics/chemistry/mathematics. Reply with ONE "
    "JSON object: {\"variants\": [{\"problem\": str, \"answer\": str, \"unit\": str, \"solution_code\": str}]}. "
    "`answer` is the final numeric value (or closed form) the problem asks for, to 3-4 significant figures; "
    "`solution_code` is complete Python (math, sympy, numpy allowed) that computes it from the problem's givens "
    "and prints ONLY the final value. Problems must be self-contained, unambiguous, solvable without images."
)


def seeds(limit: int, rng: random.Random) -> list[dict]:
    out = []
    ob = load_dataset("Hothan/OlympiadBench", "OE_TO_physics_en_COMP", split="train")
    for r in ob:
        fa = r.get("final_answer") or []
        if r.get("question") and fa and not r.get("image_1"):
            out.append({"seed_source": "olympiadbench", "subject": "physics", "problem": r["question"],
                        "answer": str(fa[0]), "unit": r.get("unit") or ""})
    sb = load_dataset("xw27/scibench", split="train")
    for r in sb:
        if r.get("problem_text") and r.get("answer_number"):
            out.append({"seed_source": f"scibench/{r.get('source', '')}", "subject": "chemistry" if "chem" in (r.get("source") or "") else "physics",
                        "problem": r["problem_text"], "answer": str(r["answer_number"]), "unit": r.get("unit") or ""})
    tq = load_dataset("TIGER-Lab/TheoremQA", split="test")
    for r in tq:
        if r.get("Question") and r.get("Answer") and not r.get("Picture") and r.get("Answer_type") in ("float", "integer"):
            out.append({"seed_source": "theoremqa", "subject": "mathematics", "problem": r["Question"],
                        "answer": str(r["Answer"]), "unit": ""})
    rng.shuffle(out)
    return out[:limit]


def parse_variants(text: str) -> list[dict]:
    i, j = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[i:j + 1])
    except Exception:  # noqa: BLE001
        return []
    vs = obj.get("variants") if isinstance(obj, dict) else None
    return [v for v in (vs or []) if isinstance(v, dict) and v.get("problem") and v.get("answer") and v.get("solution_code")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=6000)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--budget-usd", type=float, default=130.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    store = GenTaskStore(SOURCE, PACKAGE_DIR, a.epoch)
    spend = Spend(a.budget_usd)
    teacher = TeacherClient(spend)
    kept, rejects = [], []
    try:
        for s in seeds(a.seeds, rng):
            raw = teacher.complete("generate", GEN_SYSTEM,
                                   f"Seed problem ({s['subject']}):\n{s['problem']}\nSeed answer: {s['answer']} {s['unit']}\n\n"
                                   f"Write {a.variants} variants as the JSON object.", max_tokens=4000, temperature=0.9)
            for v in parse_variants(raw):
                res = run_python(v["solution_code"], timeout=30)
                computed = res.stdout.strip().splitlines()[-1].strip() if res.ok and res.stdout.strip() else None
                if computed is None or not answers_equal(computed, str(v["answer"])):
                    rejects.append({"seed": s["seed_source"], "why": "code_mismatch"}); continue
                prompt = v["problem"].strip() + (f"\n\nGive the answer in {v['unit']}." if v.get("unit") else "")
                passes = 0
                for _ in range(3):
                    reply = teacher.complete("blind_solve", SYSTEM, prompt, max_tokens=6000, temperature=0.8)
                    passes += int(answers_equal(boxed_answer(reply), str(v["answer"])))
                if passes == 0:
                    rejects.append({"seed": s["seed_source"], "why": "teacher_0_of_3"}); continue
                kept.append({"uid": gen_uid("scitext", a.epoch, v["problem"]), "problem": prompt, "answer": str(v["answer"]),
                             "unit": v.get("unit", ""), "subject": s["subject"], "seed_source": s["seed_source"],
                             "teacher_pass": passes})
            print(f"kept {len(kept)} spend USD {spend.usd:.2f}", file=sys.stderr)
    except BudgetExceeded as e:
        print(f"STOP: {e}", file=sys.stderr)
    finally:
        store.write_tasks(kept)
        with gzip.open(store.local_dir / "rejects.jsonl.gz", "wt") as f:
            for r in rejects:
                f.write(json.dumps(r) + "\n")
        spend.write(store.local_dir / "spend.json")
        print(json.dumps({"kept": len(kept), "rejected": len(rejects), "spend": spend.to_dict()}, indent=1))


if __name__ == "__main__":
    main()
