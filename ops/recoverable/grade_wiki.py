"""Grade wiki (affine-wiki-v1) continuations offline.

The wiki taskset publishes no reward (no judge endpoint on the datagen
box), so `teacher_solved` for wiki states is decided here: the teacher's
final visible reply is compared with the task's gold answer by

  * normalized containment (gold string inside the reply), and
  * an LLM judge (the teacher model itself on Engy, T=0) asked whether the
    reply's final answer means the same as the gold answer.

`solved` = judge says yes. Both signals are written back into the result
rows (results/<state_id>.json gains `wiki_grade`).

  ENGY_2=... python grade_wiki.py --states states.jsonl --out <out dir>
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
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def judge(client: httpx.Client, key: str, question: str, gold: str, reply: str) -> str:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 2048,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, gold=gold, reply=reply[-6000:])}]}
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
            if not text and body["max_tokens"] < 8192:
                # The model's reasoning ate the whole budget; the visible
                # verdict never came. Once more with room.
                body["max_tokens"] = 8192
                continue
            return "unparsed:" + text[-80:]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            if attempt == 3:
                return f"error:{type(e).__name__}"
            time.sleep(2 * (attempt + 1))
    return "error"


def undecided(grade: dict | None) -> bool:
    """A grade whose judge call produced no verdict (empty / unparsed /
    transport error) -- not a WRONG."""
    j = (grade or {}).get("judge") or ""
    return j.startswith("unparsed") or j.startswith("error")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true", help="re-grade every wiki result")
    ap.add_argument("--redo-undecided", action="store_true",
                    help="re-grade only results whose judge gave no verdict")
    args = ap.parse_args()
    key = os.environ.get("ENGY_2")
    if not key:
        sys.exit("ENGY_2 missing")
    states = {json.loads(l)["state_id"]: json.loads(l)
              for l in open(args.states, encoding="utf-8")}
    client = httpx.Client()
    n = 0
    for p in sorted((args.out / "results").glob("*.json")):
        r = json.loads(p.read_text())
        st = states.get(r["state_id"])
        if not st or st["source"] != "affine_wiki" or r.get("status") != "ok":
            continue
        if r.get("wiki_grade") and not args.force \
                and not (args.redo_undecided and undecided(r["wiki_grade"])):
            continue
        full = json.loads(Path(st["path"]).read_text(encoding="utf-8"))
        gold = str(st.get("task_answer") or full.get("task_answer") or "")
        question = full.get("task_prompt") or ""
        reply = r.get("last_visible_reply") or ""
        finished = r.get("stop_condition") == "agent_completed" and bool(reply.strip())
        grade = {"gold": gold, "finished": finished,
                 "contains": bool(gold) and norm(gold) in norm(reply),
                 "judge": judge(client, key, question, gold, reply) if finished else "no_answer"}
        grade["solved"] = grade["judge"] == "correct"
        r["wiki_grade"] = grade
        p.write_text(json.dumps(r))
        n += 1
        print(r["state_id"], grade["judge"], "contains" if grade["contains"] else "-")
    print(f"graded {n}")


if __name__ == "__main__":
    main()
