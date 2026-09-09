"""Genesis control for the shadow-teacher experiment.

Scores GENESIS's own frozen rollouts (the king side of early duel artifacts,
where king_repo is the albedo genesis) under the same three shadow teachers.
Genesis optimized against nobody, so:

  genesis holds up under shadows while miners collapse -> collapse = miner
      overfitting to the era teacher (quirk/checkpoint mining confirmed)
  genesis collapses too -> the shadow G band is harsh on everyone
      (apparatus/rendering effect); the overfitting reading weakens

Reuses the engy client + scoring path from shadow_teacher_engy.py.

Usage (from research/):  python scripts/shadow_teacher_genesis_control.py
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import random
import statistics as st
from pathlib import Path

from shadow_teacher_engy import (
    EVALS_DIR, TEACHERS, EngyClient, build_corpus, ensure_chat_template,
    load_engy_key, score_turn,
)

log = logging.getLogger("genesis_control")

GENESIS_MARKER = "king-genesis"
N_TURNS = 30
OUT = Path("results/shadow_genesis_control.jsonl")


def genesis_rollouts() -> dict[str, tuple[str, str]]:
    """turn_id -> (z, y) for genesis, pooled over genesis-king artifacts."""
    out: dict[str, tuple[str, str]] = {}
    for p in sorted(EVALS_DIR.glob("chal-*.json.gz")):
        try:
            d = json.load(gzip.open(p))
        except Exception:
            continue
        if GENESIS_MARKER not in (d.get("request", {}).get("king_repo") or ""):
            continue
        for row in d.get("king_rows") or []:
            pairs = row.get("pairs") or []
            if pairs and row.get("turn_id") and pairs[0].get("z_a"):
                out.setdefault(row["turn_id"],
                               (pairs[0]["z_a"], pairs[0]["y_a"]))
    return out


async def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    client = EngyClient(load_engy_key(), concurrency=12)
    corpus = build_corpus()
    index_by_tid = {r["turn_id"]: r for r in corpus.load_index_rows()}

    rollouts = genesis_rollouts()
    tids = [t for t in sorted(rollouts) if t in index_by_tid]
    rng = random.Random("shadow:genesis-control")
    rng.shuffle(tids)
    tids = tids[:N_TURNS]
    log.info("genesis rollouts: %d total, %d corpus-covered turns selected",
             len(rollouts), len(tids))
    turn_rows = corpus.materialize_turns([index_by_tid[t] for t in tids])
    turns = [(tid, tr["prefix"], *rollouts[tid])
             for tid, tr in zip(tids, turn_rows)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["teacher"] for l in OUT.read_text().splitlines()}

    for model, tok_repo in TEACHERS.items():
        if model in done:
            continue
        tokenizer = ensure_chat_template(tok_repo)
        sem = asyncio.Semaphore(4)
        results, errors = [], []

        async def one(tid, prefix, z, y, model=model, tokenizer=tokenizer,
                      tok_repo=tok_repo, sem=sem, results=results,
                      errors=errors):
            async with sem:
                try:
                    r = await score_turn(client, model, tokenizer, tok_repo,
                                         prefix, z, y, tid)
                    if r is not None:
                        results.append(r)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{tid}: {type(e).__name__}: {e}")

        await asyncio.gather(*[one(*t) for t in turns])
        rec = {
            "repo": "GENESIS-CONTROL",
            "teacher": model,
            "n_turns_scored": len(results),
            "n_errors": len(errors),
            "score": st.mean(r.score for r in results) if results else None,
            "mean_r_leg": (st.mean(r.r_leg for r in results)
                           if results else None),
            "mean_g_leg": (st.mean(r.g_leg for r in results)
                           if results else None),
            "g_bind_frac": (st.mean(1.0 if r.g_leg < r.r_leg else 0.0
                                    for r in results) if results else None),
            "turns": [vars(r) for r in results],
            "errors": errors[:10],
        }
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        log.info("genesis x %s: score=%s (r=%s g=%s) over %d turns, %d err",
                 model, rec["score"], rec["mean_r_leg"], rec["mean_g_leg"],
                 len(results), len(errors))

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
