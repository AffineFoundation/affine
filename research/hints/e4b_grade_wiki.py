#!/usr/bin/env python
"""Grade the wiki (null-harness) continuations of E4 / E4b offline.

affine-wiki publishes no reward, so a wiki continuation comes back
`unscored`. Same rule as ops/recoverable/grade_wiki.py (PR #13): the final
visible reply is compared with the task's gold answer by normalized
containment and by an LLM judge (the teacher on Engy, T = 0); `solved` =
judge says CORRECT. Rows of other harnesses pass through unchanged.

  ENGY_2=... python e4b_grade_wiki.py --states RUN/e4b/states.jsonl \
      --results RUN/e4b/out/results.jsonl --out RUN/e4b/out/results_graded.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

MODEL = "qwen3.8-27b"
BASE_URL = "https://api.engy.ai/v1"
JUDGE_PROMPT = (
    "You grade trivia answers. Question:\n{question}\n\nGold answer: {gold}\n\n"
    "Candidate reply (the final answer is usually at the end):\n{reply}\n\n"
    "Does the candidate's final answer refer to the same thing as the gold "
    "answer? Minor spelling, extra detail or a different but equivalent name "
    "still count as correct. A reply with no committed answer, or a different "
    "entity, is wrong. Reply with exactly one word: CORRECT or WRONG."
)


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def judge(client: httpx.Client, key: str, question: str, gold: str, reply: str) -> str:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 2048,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                question=question[:4000], gold=gold, reply=(reply or "")[-6000:])}]}
    for attempt in range(4):
        try:
            r = client.post(f"{BASE_URL}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {key}"}, timeout=180)
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            tail = text.upper().split()
            if "CORRECT" in tail[-3:]:
                return "correct"
            if "WRONG" in tail[-3:]:
                return "wrong"
            body["max_tokens"] = 6000
        except (httpx.HTTPError, KeyError, ValueError):
            time.sleep(3 * (attempt + 1))
    return "ungraded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    key = os.environ.get("ENGY_2", "")
    meta = {}
    for line in open(args.states):
        s = json.loads(line)
        meta[s["state_id"]] = s
    rows = [json.loads(l) for l in open(args.results)]
    client = httpx.Client()
    n = 0
    with open(args.out, "w") as out:
        for r in rows:
            s = meta.get(r["state_id"])
            if s and s.get("harness") == "null" and r.get("status") == "ok" and r.get("outcome") == "unscored":
                st = json.loads(Path(s["path"]).read_text()) if Path(s["path"]).exists() else {}
                gold = str(s.get("task_answer") or st.get("task_answer") or "")
                question = str(st.get("task_prompt") or s.get("task_name") or "")
                reply = r.get("last_visible_reply") or ""
                contains = bool(gold) and norm(gold) in norm(reply)
                verdict = judge(client, key, question, gold, reply) if (gold and key) else "ungraded"
                r["wiki_grade"] = {"gold": gold, "contains": contains, "judge": verdict}
                if verdict == "correct":
                    r["outcome"] = "solved"
                elif verdict == "wrong":
                    r["outcome"] = "failed"
                n += 1
            out.write(json.dumps(r) + "\n")
    print(f"graded {n} wiki continuations -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
