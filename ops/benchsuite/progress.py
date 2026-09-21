#!/usr/bin/env python
"""Live progress of a benchmark-suite run directory: per cell, rollouts done so
far, running mean score, tokens, and whether the cell has finished.

  python progress.py --run-dir ~/benchsuite/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def scan(run_dir: Path) -> list[dict]:
    rows = []
    for traces in sorted(run_dir.glob("*/*/traces.jsonl")):
        d = traces.parent
        n = k = scored = errs = 0
        toks = 0
        with traces.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                n += 1
                t = (e.get("traces") or [{}])[0]
                rw = t.get("rewards") or {}
                if e.get("errors") or t.get("errors"):
                    errs += 1
                if rw:
                    s = sum(float(v["score"]) * float(v.get("weight", 1)) for v in rw.values()
                            if isinstance(v, dict) and v.get("score") is not None)
                    k += s
                    scored += 1
                for c in t.get("calls") or []:
                    toks += int(((c.get("usage") or {}).get("completion_tokens")) or 0)
        rows.append({"model": d.parent.name, "cell": d.name, "done": n, "scored": scored,
                     "mean": round(k / scored, 4) if scored else None, "errored": errs,
                     "tokens_out": toks, "finished": (d / "summary.json").exists()})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    rows = scan(Path(a.run_dir).expanduser())
    fmt = "{:<8} {:<24} {:>6} {:>6} {:>8} {:>5} {:>11} {}"
    print(fmt.format("model", "cell", "done", "scored", "mean", "err", "tokens_out", "state"))
    for r in rows:
        print(fmt.format(r["model"], r["cell"], r["done"], r["scored"],
                         "–" if r["mean"] is None else f"{r['mean']:.3f}", r["errored"],
                         f"{r['tokens_out']:,}", "finished" if r["finished"] else "running"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
