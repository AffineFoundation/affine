#!/usr/bin/env python
"""Teacher-seat yield baseline for the wvk-25 green watch (datagen side).

Per source, over the last --hours of the datagen pods' state rows for
teacher_* policies: rollouts, graded, solve rate, kept turns per rollout,
cost per rollout, errored share. Captured BEFORE the D2 seat switch
(Qwen3.8-27B on Engy) and re-run after it (GLM-5.3-Flash) so the watch
compares like with like: a source whose solve rate or kept-turn yield drops
> 30 % under GLM is a red flag for the fold worker's band ("teacher_models
both counted") and for the cutover lead.

Reads the state files the pods sync into affine/state/datagen/state/*.jsonl
when present, else the paths given with --state (scp them first). Writes
affine/state/teacher_swap/yield_<label>_<date>.json and prints a table.

    .venv/bin/python ops/v19/teacher_yield_baseline.py --label qwen --hours 48 --state /tmp/d2.jsonl /tmp/d5.jsonl /tmp/d6.jsonl
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "affine" / "state" / "teacher_swap"


def ts(at) -> float | None:
    if isinstance(at, (int, float)):
        return float(at)
    try:
        return calendar.timegm(time.strptime(str(at), "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="qwen | glm | ...")
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--state", nargs="*", default=[], help="state.jsonl files (default: affine/state/datagen/state/*.jsonl)")
    a = ap.parse_args()
    paths = [Path(p) for p in a.state] or sorted((REPO / "affine" / "state" / "datagen" / "state").glob("*.jsonl"))
    cut = time.time() - a.hours * 3600
    agg: dict[str, dict] = defaultdict(lambda: {"rollouts": 0, "graded": 0, "solved": 0, "errored": 0, "kept_turns": 0, "cost_usd": 0.0, "models": defaultdict(int)})
    for p in paths:
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            pid = str(r.get("policy_id") or "")
            if not pid.startswith("teacher_"):
                continue
            t = ts(r.get("at"))
            if t is None or t < cut:
                continue
            s = agg[str(r.get("source"))]
            s["rollouts"] += 1
            oc = r.get("outcome")
            if oc in ("resolved", "unresolved"):
                s["graded"] += 1
            if oc == "resolved":
                s["solved"] += 1
            if oc == "error":
                s["errored"] += 1
            s["kept_turns"] += int(r.get("n_turns") or 0)
            s["cost_usd"] += float(r.get("cost_usd") or 0.0)
            s["models"][str(r.get("provider") or "")] += 1
    rows = {}
    for src, s in sorted(agg.items()):
        n = s["rollouts"] or 1
        rows[src] = {"rollouts": s["rollouts"], "graded": s["graded"],
                     "solve_rate": round(s["solved"] / max(1, s["graded"]), 4),
                     "kept_turns_per_rollout": round(s["kept_turns"] / n, 3),
                     "cost_usd_per_rollout": round(s["cost_usd"] / n, 5),
                     "errored_share": round(s["errored"] / n, 4),
                     "providers": dict(s["models"])}
    out = {"label": a.label, "window_hours": a.hours, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "state_files": [str(p) for p in paths], "sources": rows,
           "totals": {"rollouts": sum(r["rollouts"] for r in rows.values()),
                      "cost_usd": round(sum(agg[s]["cost_usd"] for s in rows), 2)}}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"yield_{a.label}_{time.strftime('%Y-%m-%d', time.gmtime())}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{'source':24} {'rollouts':>8} {'graded':>7} {'solve':>6} {'kept/ro':>8} {'$/ro':>8} {'err%':>5}")
    for src, r in sorted(rows.items(), key=lambda kv: -kv[1]["rollouts"]):
        print(f"{src:24} {r['rollouts']:8d} {r['graded']:7d} {r['solve_rate']:6.3f} {r['kept_turns_per_rollout']:8.2f} {r['cost_usd_per_rollout']:8.4f} {100 * r['errored_share']:5.1f}")
    print(f"total rollouts {out['totals']['rollouts']}, USD {out['totals']['cost_usd']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
