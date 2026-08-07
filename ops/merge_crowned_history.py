"""One-shot: collapse verdict+crowned row pairs in history.jsonl into one row.

Old shape: a winning duel wrote a `verdict` row (full payload) then a bare
`crowned` row (no z/margin/gates) — every crowning showed up twice on the
charts, once with values and once as a hole. New validator code writes a
single merged `crowned` row; this script rewrites existing history to match.

Merge rule: for each `crowned` row with a challenge_id that also has a
`verdict` row, fold the verdict row's payload (verdict, accepted, uid,
duration_s, at) into the crowned row and drop the verdict row. Crowns without
a sibling verdict row (seed, revert, retro-crowns) are left alone.

Run:  source .venv/bin/activate && python ops/merge_crowned_history.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

HISTORY = Path(__file__).resolve().parents[1] / "affine" / "state" / "history.jsonl"

MERGE_FIELDS = ("verdict", "accepted", "uid", "duration_s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the merged file (default: dry run)")
    args = ap.parse_args()

    raw = HISTORY.read_text()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]

    verdict_by_cid: dict[str, dict] = {}
    for r in rows:
        if r.get("event") == "verdict" and r.get("challenge_id"):
            verdict_by_cid[r["challenge_id"]] = r

    merged_cids: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if r.get("event") == "crowned":
            v = verdict_by_cid.get(r.get("challenge_id") or "")
            if v is not None:
                for k in MERGE_FIELDS:
                    if k in v and k not in r:
                        r[k] = v[k]
                # Keep the verdict's timestamp: that is when the duel ended.
                r["at"] = v.get("at", r.get("at"))
                merged_cids.add(r["challenge_id"])
                print(f"merge  {r['challenge_id']}  reign #{r.get('reign_number')}  "
                      f"z={((v.get('verdict') or {}).get('z'))}")
        out.append(r)
    out = [r for r in out
           if not (r.get("event") == "verdict"
                   and r.get("challenge_id") in merged_cids)]

    print(f"{len(rows)} rows -> {len(out)} rows "
          f"({len(merged_cids)} crown(s) merged)")
    if not args.apply:
        print("dry run — pass --apply to write")
        return

    # Abort if the validator appended while we worked (tiny race window).
    if HISTORY.read_text() != raw:
        raise SystemExit("history.jsonl changed during merge — rerun")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = HISTORY.with_suffix(f".jsonl.bak-{stamp}-rowmerge")
    shutil.copy2(HISTORY, backup)
    tmp = HISTORY.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in out))
    tmp.replace(HISTORY)
    print(f"wrote {HISTORY} (backup: {backup.name})")


if __name__ == "__main__":
    main()
