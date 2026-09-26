"""IPRM probe — stage 3c (local, optional): fresh KING samples at the primary
(split_states/selected) states, k per state, through the reign-20 king's
vLLM endpoint (hint_design.King). Adds rows origin=king_fresh to
/tmp/iprm/eval_rows.jsonl (so render_eval.py picks them up) and writes
/tmp/iprm/king_samples.jsonl. Cost 0 (our own box).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import hint_design  # noqa: E402
from build_data import action_body_span, fix_fence, parse_action  # noqa: E402

OUT = Path("/tmp/iprm")


async def run(k: int, max_tokens: int, only_primary: bool) -> None:
    states = [s for s in common.read_jsonl(OUT / "eval_states.jsonl") if s["primary"] or not only_primary]
    if not await hint_design.king_reachable():
        print("king endpoint unreachable; skipping")
        return
    king = hint_design.King(concurrency=8)
    rows = common.read_jsonl(OUT / "eval_rows.jsonl")
    have = {(r["state"], r["y"]) for r in rows}
    samples = []

    async def one(s: dict, j: int) -> None:
        try:
            rep = await king.chat(hint_design.KING_MODEL, s["prefix"], temperature=0.8, max_tokens=max_tokens)
        except Exception as e:  # endpoint hiccup
            samples.append({"state": s["state"], "j": j, "error": repr(e)[:200]})
            return
        y = parse_action(fix_fence(rep["content"]), s["kind"]) if rep["think_closed"] else None
        samples.append({"state": s["state"], "j": j, "kind": s["kind"], "think_closed": rep["think_closed"],
                        "finish": rep["finish"], "reasoning_chars": len(rep["reasoning"]),
                        "content": rep["content"][:4000], "y": y})

    await asyncio.gather(*[one(s, j) for s in states for j in range(k)])
    common.write_jsonl(OUT / "king_samples.jsonl", samples)
    n_new = 0
    for smp in samples:
        y = smp.get("y")
        if not y:
            continue
        key = (smp["state"], y)
        if key in have:
            for r in rows:
                if (r["state"], r["y"]) == key:
                    r["origins"].append("king_fresh")
            continue
        have.add(key)
        bs, be = action_body_span(y, smp["kind"])
        rows.append({"state": smp["state"], "kind": smp["kind"], "y": y, "origins": ["king_fresh"], "labels": [],
                     "arms": [], "row_id": hashlib.sha1(f"{smp['state']}|{y}".encode()).hexdigest()[:12],
                     "body_start": bs, "body_end": be, "body": y[bs:be], "n_labels": 0, "p_solved": None})
        n_new += 1
    common.write_jsonl(OUT / "eval_rows.jsonl", rows)
    ok = sum(1 for s in samples if s.get("y"))
    closed = sum(1 for s in samples if s.get("think_closed"))
    print(f"king: {len(samples)} samples at {len(states)} states, think closed {closed}, parsed {ok}, new rows {n_new}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048 + 768)
    ap.add_argument("--all-states", action="store_true")
    a = ap.parse_args()
    asyncio.run(run(a.k, a.max_tokens, not a.all_states))


if __name__ == "__main__":
    main()
