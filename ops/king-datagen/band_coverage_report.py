#!/usr/bin/env python
"""Per-fold king-seat coverage and king-side retire share on the band sources.

Reads the public corpus manifest (`band_filter.per_source`, written by the
fold) and prints one row per replay source (Source.king_attempts >= 2 in
rollouts/sources.toml): tasks the fold keeps in D, tasks the king seat has
tried, coverage, king solve rate, rows retired because the king solved the
task. Meant to be run after every fold and pasted into the datagen worker's
log / requests.md (Jacob 2026-09-23 20:47 "do": report per-fold coverage per
source and the king-side retire share).

    .venv/bin/python ops/king-datagen/band_coverage_report.py [--manifest URL|path] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
MANIFEST = "https://data.affine.io/corpus/manifest.json"


def replay_sources() -> list[str]:
    d = tomllib.load(open(REPO / "rollouts" / "rollouts" / "sources.toml", "rb"))
    return [n for n, s in d["source"].items() if int(s.get("king_attempts", 0)) >= 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.manifest.startswith("http"):
        m = httpx.get(a.manifest, headers={"User-Agent": "affine-band-coverage/0.1"}, timeout=60).json()
    else:
        m = json.loads(Path(a.manifest).read_text())
    per = (m.get("band_filter") or {}).get("per_source") or {}
    rows = []
    for s in replay_sources():
        v = per.get(s) or {}
        pub = v.get("published") or {}
        kept = int(v.get("tasks_kept") or 0)
        seen = int(v.get("tasks_seen") or 0)
        cov = int(v.get("tasks_king_covered") or 0)
        retired = int(pub.get("retired") or 0)
        rows.append({
            "source": s, "epoch": m.get("corpus_epoch"), "tasks_seen": seen, "tasks_kept": kept,
            "tasks_king_covered": cov, "coverage_pct": round(100.0 * cov / seen, 1) if seen else None,
            "king_solve_rate": v.get("king_solve_rate_all"), "teacher_solve_rate": v.get("teacher_solve_rate_all"),
            "rows_kept": int(pub.get("kept") or 0), "rows_retired": retired,
            "rows_retired_king_solved": int(pub.get("retired_king_solved") or 0),
            "retire_share_pct": round(100.0 * retired / (retired + int(pub.get("kept") or 0)), 2) if (retired + int(pub.get("kept") or 0)) else None,
        })
    if a.json:
        print(json.dumps({"epoch": m.get("corpus_epoch"), "published_at": m.get("published_at"), "rows": rows}, indent=1))
        return 0
    print(f"epoch {m.get('corpus_epoch')} published {m.get('published_at')}")
    print(f"{'source':17} {'seen':>6} {'kept':>6} {'king_cov':>8} {'cov%':>6} {'k_solve':>7} {'rows_kept':>9} {'retired':>8} {'k_solved':>8} {'ret%':>6}")
    for r in rows:
        print(f"{r['source']:17} {r['tasks_seen']:6d} {r['tasks_kept']:6d} {r['tasks_king_covered']:8d} "
              f"{(r['coverage_pct'] if r['coverage_pct'] is not None else 0):6.1f} "
              f"{(r['king_solve_rate'] or 0):7.3f} {r['rows_kept']:9d} {r['rows_retired']:8d} "
              f"{r['rows_retired_king_solved']:8d} {(r['retire_share_pct'] or 0):6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
