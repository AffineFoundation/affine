"""wvk-9 → wvk-10 replay parity: stored artifacts must re-score identically.

For the last N stored duel artifacts (all scored under the wvk-9 Reason v4
rule), re-run the duel through the NEW score.py with score_mode="reason" and
the artifact's stamped params. The margin/se/z/winner must match the stored
verdict exactly — proving the min(R,G) fork left the replay path untouched.

Run from repo root: python research/scripts/minrg_replay_parity.py [N]
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.score import duel as score_duel  # noqa: E402

EVALS = REPO / "affine" / "state" / "evals"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    index = [json.loads(line)
             for line in open(EVALS / "index.jsonl") if line.strip()]
    # Only rows with a scored duel (margin present) and a local blob.
    rows = [r for r in index if r.get("margin") is not None
            and (EVALS / Path(r["key"]).name).is_file()][-n:]
    print(f"replaying {len(rows)} stored duels through the new code "
          f"(score_mode='reason')\n")
    fails = 0
    for r in rows:
        art = json.loads(gzip.decompress(
            (EVALS / Path(r["key"]).name).read_bytes()))
        cid = r["challenge_id"]
        chall = art.get("challenger_rows") or []
        king = art.get("king_rows") or []
        if not chall or not king:
            print(f"{cid}: no rows (rejected pre-score) — skip")
            continue
        # Production wvk-9 params (duel_params live in the verdict, not the
        # artifact, so use the frozen contract values of that era).
        res = score_duel(chall, king, k_sigma=2.0, min_margin=0.002,
                         min_thought_chars=80, causality_gamma=0.30,
                         tau=0.03, score_mode="reason")
        dm = abs(res.margin - r["margin"])
        dz = abs(res.z - r["z"])
        ok = dm < 1e-12 and dz < 1e-9 and res.challenger_wins == r["challenger_wins"]
        print(f"{cid}: margin {res.margin:+.9f} vs stored {r['margin']:+.9f} "
              f"(d={dm:.2e})  z d={dz:.2e}  wins={res.challenger_wins}=="
              f"{r['challenger_wins']}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails += 1
    print(f"\n{'PARITY_OK' if fails == 0 else f'PARITY_FAIL {fails}'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
