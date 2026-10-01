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

Seeds run in --workers threads (the 17:54 sequential run was over a day for
6,000 seeds). Every kept variant is appended to data/e<epoch>/tasks.jsonl.gz
the moment it passes (a SIGTERM loses nothing); reruns skip uids already in
the file. Budget: --budget-usd (plan USD 110; default 130).
  python -m affine_scitext_v1.generate --epoch 1 --seeds 6000 --variants 2 --workers 8 --budget-usd 130
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
import threading

from datasets import load_dataset

from affine_gen_v1.store import GenTaskStore, gen_uid
from affine_gen_v1.teacher import BudgetExceeded, Spend, TeacherClient
from affine_gen_v1.verify import answers_equal, boxed_answer, run_python

from affine_scitext_v1.taskset import PACKAGE_DIR, SOURCE, SYSTEM

GEN_SYSTEM = (
    "You write new quantitative science problems by varying a given seed problem: change the numbers, the "
    "asked quantity or the scenario while keeping the underlying physics/chemistry/mathematics. Reply with ONE "
    "JSON object inside a ```json fenced block: {\"variants\": [{\"problem\": str, \"answer\": str, \"unit\": str, "
    "\"solution_code\": str}]}. `answer` is the final numeric value (or closed form) the problem asks for, to 3-4 "
    "significant figures; `solution_code` is complete Python (math, sympy, numpy allowed) that computes it from "
    "the problem's givens and prints ONLY the final value. Problems must be self-contained, unambiguous, solvable "
    "without images. Escape backslashes and newlines correctly inside JSON strings."
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
    text = text or ""
    import re
    cands = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S) + [text]
    dec = json.JSONDecoder()
    for c in cands:
        i = c.find("{")
        while i != -1:
            try:
                obj, _ = dec.raw_decode(c[i:])
            except json.JSONDecodeError:
                obj = None
            vs = obj.get("variants") if isinstance(obj, dict) else None
            if isinstance(vs, list):
                return [v for v in vs if isinstance(v, dict) and v.get("problem") and v.get("answer") and v.get("solution_code")]
            i = c.find("{", i + 1)
    return []


class Gen:
    def __init__(self, a, store: GenTaskStore) -> None:
        self.a, self.store = a, store
        self.spend = Spend(a.budget_usd)
        self.teacher = TeacherClient(self.spend)
        self.kept = 0
        self.rejects: dict[str, int] = {}
        self.seen = store.existing_uids()
        self.lock = threading.Lock()
        self.stop = False

    def reject(self, seed: str, why: str, detail: str = "") -> None:
        with self.lock:
            self.rejects[why] = self.rejects.get(why, 0) + 1
        self.store.append_reject({"seed": seed, "why": why, "detail": detail[-800:]})

    def keep(self, rec: dict) -> int:
        with self.lock:
            if rec["uid"] in self.seen:
                return self.kept
            self.seen.add(rec["uid"])
            self.store.append_task(rec)
            self.kept += 1
            self.spend.write(self.store.local_dir / "spend.json")
            return self.kept

    def seed(self, s: dict) -> None:
        if self.stop:
            return
        try:
            raw = self.teacher.complete("generate", GEN_SYSTEM,
                                        f"Seed problem ({s['subject']}):\n{s['problem']}\nSeed answer: {s['answer']} {s['unit']}\n\n"
                                        f"Write {self.a.variants} variants as the JSON object.", max_tokens=6000, temperature=0.9)
            variants = parse_variants(raw)
            if not variants:
                self.reject(s["seed_source"], "bad_json", raw[-500:]); return
            for v in variants:
                res = run_python(v["solution_code"], timeout=30)
                computed = res.stdout.strip().splitlines()[-1].strip() if res.ok and res.stdout.strip() else None
                if computed is None or not answers_equal(computed, str(v["answer"])):
                    self.reject(s["seed_source"], "code_mismatch", f"code={computed!r} answer={v['answer']!r} err={res.stderr[-200:]}"); continue
                prompt = v["problem"].strip() + (f"\n\nGive the answer in {v['unit']}." if v.get("unit") else "")
                passes = 0
                for _ in range(3):
                    reply = self.teacher.complete("blind_solve", SYSTEM, prompt, max_tokens=8000, temperature=0.8)
                    passes += int(answers_equal(boxed_answer(reply), str(v["answer"])))
                if passes == 0:
                    self.reject(s["seed_source"], "teacher_0_of_3"); continue
                n = self.keep({"uid": gen_uid("scitext", self.a.epoch, v["problem"]), "problem": prompt, "answer": str(v["answer"]),
                               "unit": v.get("unit", ""), "subject": s["subject"], "seed_source": s["seed_source"],
                               "teacher_pass": passes})
                print(f"kept {n} ({s['seed_source']}, teacher {passes}/3) spend USD {self.spend.usd:.2f}", file=sys.stderr)
        except BudgetExceeded as e:
            self.stop = True
            print(f"STOP: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - one bad seed must not kill the run
            self.reject(s["seed_source"], f"error_{type(e).__name__}", str(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=6000)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=130.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    store = GenTaskStore(SOURCE, PACKAGE_DIR, a.epoch)
    g = Gen(a, store)
    rows = seeds(a.seeds, rng)
    print(f"{len(rows)} seeds × {a.variants} variants, {a.workers} workers, budget USD {a.budget_usd}, "
          f"{len(g.seen)} already kept", file=sys.stderr)
    try:
        with cf.ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(g.seed, rows))
    finally:
        g.spend.write(store.local_dir / "spend.json")
        print(json.dumps({"seeds": len(rows), "kept_this_run": g.kept, "kept_total": len(g.seen),
                          "rejected": sum(g.rejects.values()), "rejects_by_reason": g.rejects, "spend": g.spend.to_dict()}, indent=1))


if __name__ == "__main__":
    main()
