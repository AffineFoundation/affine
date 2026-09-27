#!/usr/bin/env python3
"""Build the fold's task-instruction list for SWE-smith: every catalog row
whose `problem_statement` is empty, as the sids the fold sees
(`rollouts.catalog._swesmith_meta`). Output {source: [sid, ...]} for
`[task_instruction].empty_sids` (default
affine/state/task_instruction/empty_problem_statement_sids.json).

2026-09-27 (datagen seat-error audit): py 11,437 / 50,908 (22.5 %), go
6,583 / 8,212 (80.2 %), java 766 / 7,470 (10.3 %), js / ts / rs / cpp 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rollouts"))

from datasets import load_dataset  # noqa: E402
from rollouts.catalog import SWESMITH_LANG_DATASETS, _swesmith_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "affine" / "state" / "task_instruction" / "empty_problem_statement_sids.json"))
    ap.add_argument("--source", default="swesmith")
    a = ap.parse_args()
    sids: list[str] = []
    stats: dict[str, dict] = {}
    for lang_key, dataset, language in SWESMITH_LANG_DATASETS:
        rows = load_dataset(dataset, split="train")
        n = 0
        for row in rows:
            if str(row.get("problem_statement") or "").strip():
                continue
            meta = _swesmith_meta({"instance_id": row.get("instance_id"), "image_name": row.get("image_name") or "x"},
                                  lang_key, language)
            if meta:
                sids.append(meta["sid"])
                n += 1
        stats[lang_key] = {"dataset": dataset, "total": len(rows), "empty": n}
        print(f"{lang_key:5s} {dataset:28s} total {len(rows):6d} empty {n:6d} ({100 * n / max(1, len(rows)):.1f} %)")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"built_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
                               "rule": "problem_statement empty after strip", "stats": stats,
                               "sids": {a.source: sorted(set(sids))}}, indent=0))
    print(f"wrote {len(set(sids))} sids -> {out}")


if __name__ == "__main__":
    main()
