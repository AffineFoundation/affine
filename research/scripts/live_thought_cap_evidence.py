"""Live-duel evidence for the thought cap (2026-09-09).

Reads stored verdict artifacts (`affine/state/evals/chal-*.json.gz`, min(R,G)
era) and answers three questions about `max_thought_tokens = 1024`:

  1. Where do forfeits land?  Median teacher-reference thought length on the
     turns a side forfeited vs the turns it answered.  If forfeits sit on the
     long-ref turns, the cap is what the miner is hitting.
  2. How much shorter do miners think than the teacher when the teacher
     thinks long?  Ratio of the miner's thought length to the teacher's ref
     thought length, by ref-length bucket.
  3. How many turns does the teacher itself lose?  Turns dropped from the
     slice because fewer than 2 teacher refs parsed within
     max_thought + max_action tokens.

    python research/scripts/live_thought_cap_evidence.py --last 60
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics as st
from collections import defaultdict

BUCKETS = ((0, 300), (300, 800), (800, 1500), (1500, 10**9))


def ref_len(refs: list[dict]) -> float:
    return st.median(len(r.get("z") or "") for r in refs) if refs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="affine/state/evals")
    ap.add_argument("--last", type=int, default=60)
    ap.add_argument("--out", default="research/results/live_thought_cap_evidence")
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.evals}/chal-*.json.gz"))[-args.last:]
    forfeit_ref, valid_ref = [], []
    ratio = defaultdict(list)
    dropped, n_slice, n_verdicts = 0, 0, 0
    forfeits_by_side = defaultdict(int)
    turns_by_side = defaultdict(int)
    forfeit_with_long_ref = 0
    for p in paths:
        v = json.load(gzip.open(p))
        if (v.get("verdict") or {}).get("duel_params", {}).get("score_mode") != "min_rg":
            continue
        n_verdicts += 1
        refs = v.get("teacher_refs") or {}
        turn_ids = v.get("turn_ids") or []
        n_slice += len(turn_ids)
        dropped += sum(1 for t in turn_ids if len(refs.get(t) or []) < 2)
        for side in ("king_rows", "challenger_rows"):
            for row in v.get(side) or []:
                rl = ref_len(refs.get(row["turn_id"]) or [])
                turns_by_side[side] += 1
                if not row.get("valid"):
                    forfeits_by_side[side] += 1
                    forfeit_ref.append(rl)
                    forfeit_with_long_ref += rl >= 1500
                    continue
                valid_ref.append(rl)
                pairs = row.get("pairs") or []
                if pairs and rl > 0:
                    own = len(pairs[0].get("z_a") or "")
                    for lo, hi in BUCKETS:
                        if lo <= rl < hi:
                            ratio[(lo, hi)].append(own / rl)

    lines = [f"live thought-cap evidence — {n_verdicts} min(R,G) verdicts, {n_slice} slice turns",
             "",
             f"1. teacher-ref thought length (chars, median over the turn's refs):",
             f"   forfeited turns  n={len(forfeit_ref):>5}  median {st.median(forfeit_ref):.0f}"
             if forfeit_ref else "   no forfeits",
             f"   answered turns   n={len(valid_ref):>5}  median {st.median(valid_ref):.0f}",
             f"   share of forfeits where the teacher ref itself is >=1500 chars: "
             f"{forfeit_with_long_ref / max(1, len(forfeit_ref)):.0%} "
             f"(base rate among answered turns {sum(1 for x in valid_ref if x >= 1500) / max(1, len(valid_ref)):.0%})",
             "",
             "2. miner thought length / teacher ref thought length, by ref bucket (median ratio):"]
    for (lo, hi), rs in sorted(ratio.items()):
        lines.append(f"   ref {lo:>5}-{hi if hi < 10**9 else 'inf':>5} chars  n={len(rs):>6}  ratio {st.median(rs):.2f}")
    lines += ["",
              f"3. turns dropped because <2 teacher refs parsed inside the cap: {dropped} / {n_slice} = {dropped / max(1, n_slice):.1%}",
              "",
              "forfeit rate per side: " + "  ".join(
                  f"{s.split('_')[0]} {forfeits_by_side[s] / max(1, turns_by_side[s]):.1%}" for s in turns_by_side)]
    text = "\n".join(lines)
    print(text)
    with open(args.out + ".txt", "w") as f:
        f.write(text + "\n")
    with open(args.out + ".json", "w") as f:
        json.dump({"n_verdicts": n_verdicts, "n_slice": n_slice, "dropped": dropped,
                   "forfeit_ref_median": st.median(forfeit_ref) if forfeit_ref else None,
                   "valid_ref_median": st.median(valid_ref) if valid_ref else None,
                   "ratio_by_bucket": {f"{lo}-{hi}": st.median(rs) for (lo, hi), rs in ratio.items()},
                   "forfeit_rate": {s: forfeits_by_side[s] / max(1, turns_by_side[s]) for s in turns_by_side}},
                  f, indent=1)


if __name__ == "__main__":
    main()
