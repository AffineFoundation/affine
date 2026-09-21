#!/usr/bin/env python3
"""Does the teacher abstain when told it may? The go / no-go for affine_trivia_abstain.

Draws N TriviaQA train rows, asks the model twice per row -- the plain
affine_trivia prompt and the abstention prompt of affine_trivia_abstain --
and reports:

  * plain:   correct / wrong rate (the model's knowledge on these rows);
  * abstain: correct / abstained / wrong rate;
  * the number that decides: of the rows the model got WRONG under the plain
    prompt, how many did it abstain on under the abstention prompt (a
    "rescued hallucination"), and how many correct plain answers turned into
    abstentions (lost knowledge). AA-Omniscience index = +1 correct, 0
    abstain, -1 wrong, reported for both prompts.

Rule of thumb from docs/artificial-analysis-index-path.md §8: worth flipping
the source's share to 1.0 if rescued >= 30 % of the plain-wrong rows and lost
< half of that.

Usage (any box with the ENGY key, or a king box via --base-url/--model):
  python ops/aa/abstain_probe.py --n 100 --model qwen3.8-27b \
      --base-url https://api.engy.ai/v1 --key-env ENGY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rollouts" / "envs"
                       / "affine_trivia_abstain_v1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rollouts" / "envs"
                       / "affine_trivia_v1"))

from affine_trivia_abstain_v1.taskset import (  # noqa: E402
    DATASET_CONFIG, SPLIT, SYSTEM as ABSTAIN_SYSTEM, is_abstention, prediction_of)
from affine_trivia_v1.taskset import SYSTEM as PLAIN_SYSTEM  # noqa: E402
from triviaqa_v1.taskset import DATASET_NAME, DATASET_REVISION  # noqa: E402


async def ask(client: AsyncOpenAI, model: str, system: str, question: str,
              temperature: float, max_tokens: int, sem: asyncio.Semaphore) -> str:
    async with sem:
        r = await client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": question}])
    return r.choices[0].message.content or ""


def grade(reply: str, answers: set[str]) -> str:
    pred = prediction_of(reply)
    if is_abstention(pred):
        return "abstain"
    return "correct" if pred in answers else "wrong"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--key-env", default="ENGY")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT, revision=DATASET_REVISION)
    idx = random.Random(a.seed).sample(range(len(rows)), a.n)
    client = AsyncOpenAI(base_url=a.base_url, api_key=os.environ.get(a.key_env, "x"))
    sem = asyncio.Semaphore(a.concurrency)

    async def one(i: int) -> dict:
        row = rows[i]
        answers = set(row["answer"]["normalized_aliases"])
        plain, abst = await asyncio.gather(
            ask(client, a.model, PLAIN_SYSTEM, row["question"], a.temperature, a.max_tokens, sem),
            ask(client, a.model, ABSTAIN_SYSTEM, row["question"], a.temperature, a.max_tokens, sem))
        return {"question_id": row["question_id"], "question": row["question"],
                "plain": grade(plain, answers), "abstain": grade(abst, answers),
                "plain_reply": plain[-300:], "abstain_reply": abst[-300:]}

    results = await asyncio.gather(*(one(i) for i in idx))
    n = len(results)
    c = lambda cond: sum(1 for r in results if cond(r))   # noqa: E731
    plain_wrong = c(lambda r: r["plain"] == "wrong")
    rescued = c(lambda r: r["plain"] == "wrong" and r["abstain"] == "abstain")
    lost = c(lambda r: r["plain"] == "correct" and r["abstain"] == "abstain")

    def omni(key: str) -> float:
        return (c(lambda r: r[key] == "correct") - c(lambda r: r[key] == "wrong")) / n * 100

    report = {
        "n": n, "model": a.model,
        "plain": {k: c(lambda r, k=k: r["plain"] == k) / n for k in ("correct", "wrong", "abstain")},
        "abstain": {k: c(lambda r, k=k: r["abstain"] == k) / n for k in ("correct", "wrong", "abstain")},
        "plain_wrong": plain_wrong,
        "rescued_hallucinations": rescued,
        "rescued_frac_of_plain_wrong": (rescued / plain_wrong) if plain_wrong else None,
        "lost_correct_to_abstain": lost,
        "omniscience_index_plain": omni("plain"),
        "omniscience_index_abstain": omni("abstain"),
    }
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps({"report": report, "rows": results}, indent=1))
    verdict = ("GO" if plain_wrong and rescued / plain_wrong >= 0.30 and lost < rescued / 2
               else "NO-GO")
    print(f"verdict: {verdict} (rescued {rescued}/{plain_wrong} plain-wrong rows, lost {lost} correct ones)")


if __name__ == "__main__":
    asyncio.run(main())
