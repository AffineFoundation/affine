"""Extract the filler king's fixed suffix from its crowning duel artifact."""

import gzip
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "affine" / "state" / "evals" / "chal-01093.json.gz"

d = json.loads(gzip.decompress(ART.read_bytes()))
req = d.get("request") or {}
print("king:", req.get("king_repo"))
print("chall:", req.get("challenger_repo") or req.get("repo"))

for side in ("challenger_rows", "king_rows"):
    zs = [r["pairs"][0]["z_a"] for r in d[side]
          if r.get("valid") and r.get("pairs")]
    print(f"\n=== {side}: {len(zs)} thoughts ===")
    tails = Counter(z[-160:] for z in zs if len(z) >= 160)
    for tail, n in tails.most_common(3):
        print(f"--- tail x{n} ---")
        print(repr(tail))
    heads = Counter(z[:120] for z in zs if len(z) >= 120)
    for head, n in heads.most_common(2):
        print(f"--- head x{n} ---")
        print(repr(head))
