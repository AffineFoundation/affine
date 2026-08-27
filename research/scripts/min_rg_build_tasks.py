"""Build echo tasks for the min(R, G) grounding test (Phases 2+3).

For each selected duel artifact, emit one task row per turn:
    {challenge_id, turn_id, thoughts: {ref0..ref2, king, chall}}
Ref thoughts (z_C^i) define the per-turn typicality band; king/chall are the
miner thoughts to place against it. Synthetic adversary thoughts (parrot /
boilerplate) are added later, once prefixes are available.

Writes research/data/min_rg_tasks.jsonl and a turn_ids list for prefix
materialization on the eval pod.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVALS_DIR = REPO / "affine" / "state" / "evals"
OUT_TASKS = REPO / "research" / "data" / "min_rg_tasks.jsonl"
OUT_TIDS = REPO / "research" / "data" / "min_rg_turn_ids.json"

# filler crowning / big centered flips / recent defense
DUELS = ["chal-01074", "chal-01093", "chal-01113", "chal-01115"]


def main() -> None:
    tasks = []
    tids: set[str] = set()
    for cid in DUELS:
        d = json.loads(gzip.decompress(
            (EVALS_DIR / f"{cid}.json.gz").read_bytes()))
        refs = d["teacher_refs"]
        kz = {r["turn_id"]: r["pairs"][0]["z_a"]
              for r in d["king_rows"] if r.get("valid") and r.get("pairs")}
        cz = {r["turn_id"]: r["pairs"][0]["z_a"]
              for r in d["challenger_rows"] if r.get("valid") and r.get("pairs")}
        req = d.get("request") or {}
        for tid, rr in refs.items():
            if tid not in kz or tid not in cz:
                continue
            tasks.append({
                "challenge_id": cid,
                "turn_id": tid,
                "king_repo": req.get("king_repo"),
                "challenger_repo": req.get("challenger_repo") or req.get("repo"),
                "thoughts": {
                    **{f"ref{i}": r["z"] for i, r in enumerate(rr)},
                    "king": kz[tid],
                    "chall": cz[tid],
                },
            })
            tids.add(tid)

    OUT_TASKS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TASKS, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    OUT_TIDS.write_text(json.dumps(sorted(tids)))
    print(f"{len(tasks)} tasks over {len(tids)} distinct turns "
          f"from {len(DUELS)} duels")


if __name__ == "__main__":
    main()
