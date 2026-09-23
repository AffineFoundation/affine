"""IPRM probe — parity check (local, ~$0.05 Engy): lpC over the action body of
a handful of rendered rows under the LIVE teacher (Engy vLLM echo) so the
pod's HF-bf16 numbers can be compared. Writes /tmp/iprm/parity_engy.jsonl
{row_id, kind, engy_lpC, n_body}; analyze.py joins it with the pod's lpC.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

OUT = Path("/tmp/iprm")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rows = {r["row_id"]: r for r in common.read_jsonl(OUT / "eval_rows.jsonl")}
    inp = [r for r in common.read_jsonl(OUT / "score_in.jsonl") if r["ctx"] == "own" and len(r["ids"]) < 20000]
    rng = random.Random(a.seed)
    by_kind: dict[str, list[dict]] = {}
    for r in inp:
        by_kind.setdefault(rows[r["row_id"]]["kind"], []).append(r)
    pick: list[dict] = []
    for kind, rs in by_kind.items():
        pick.extend(rng.sample(rs, min(len(rs), max(1, a.n // len(by_kind)))))
    engy = common.Engy(concurrency=4)

    async def one(r: dict) -> dict:
        start = max(min(r["body_tok"]) - 1, 0)
        lps = await engy.echo_ids(r["ids"], start)
        body = [lps[i - start] for i in r["body_tok"] if lps[i - start] is not None]
        return {"row_id": r["row_id"], "kind": rows[r["row_id"]]["kind"], "engy_lpC": sum(body),
                "n_body": len(body), "n_ids": len(r["ids"])}

    res = await asyncio.gather(*(one(r) for r in pick))
    common.write_jsonl(OUT / "parity_engy.jsonl", res)
    for x in res:
        print(json.dumps(x))
    print(f"engy cost ${engy.cost_usd:.4f}  usage {engy.usage}")
    json.dump({"cost_usd": engy.cost_usd, "usage": engy.usage}, open(OUT / "parity_engy_cost.json", "w"))


if __name__ == "__main__":
    asyncio.run(main())
