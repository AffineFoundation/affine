#!/usr/bin/env python
"""Scorecard tables from the published benchmark-suite JSONs
(affine/state/benchsuite/*.json, written by publish.py).

  python report.py                      # latest run: env, n, king ± CI, teacher ± CI, delta
  python report.py --run <run_id>       # a specific run
  python report.py --history            # king T=0 score per env across every run (one column per reign)
  python report.py --markdown           # GitHub-flavoured markdown instead of aligned text

Definitions. n = tasks graded. score = mean reward (a solve rate for binary
graders). CI = 95% Wilson score interval for binary graders, normal
approximation otherwise. delta = king − teacher in percentage points. "cap"
= share of king rollouts whose last model call stopped at the token cap.
"t/o/ctx" = rollouts that ran out of the time budget / overflowed the context
(both score 0: the model did not finish the task). Infrastructure errors are
excluded from n and listed as errored.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
STATE_DIR = REPO / SUITE["suite"]["state_dir"]


def load_runs(state_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(state_dir.glob("*.json")):
        try:
            card = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(card, dict) and "rows" in card:
            runs.append(card)
    runs.sort(key=lambda c: c.get("created_at") or "")
    return runs


def pct(x: float | None, d: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{d}f}"


def side(s: dict | None) -> str:
    if not s:
        return "–"
    lo, hi = s["ci95"]
    return f"{pct(s['score'])} [{pct(lo, 0)}–{pct(hi, 0)}]"


def table(rows: list[list[str]], header: list[str], markdown: bool) -> str:
    if markdown:
        out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    return "\n".join([fmt.format(*header), fmt.format(*("-" * w for w in widths))]
                     + [fmt.format(*r) for r in rows])


def scorecard(card: dict, markdown: bool) -> str:
    k = card.get("king") or {}
    w = card.get("where") or {}
    head = [f"run {card['run_id']} — king reign {k.get('reign')} (king-{str(k.get('digest', ''))[:12]}) "
            f"vs teacher {(card.get('teacher') or {}).get('hf_repo')}",
            f"where: {w.get('provider', '?')} {w.get('gpu', '')} · created {card.get('created_at')} · "
            f"pod cost ≈ ${card.get('prime_spent_usd')}"]
    rows = sorted(card["rows"], key=lambda r: ((r.get("group") or ""), r["env"], r["temperature"]))
    body = []
    for r in rows:
        kk, tt = r.get("king"), r.get("teacher")
        d = r.get("delta")
        body.append([
            r["env"], r.get("group") or "", f"{r['temperature']:g}", str(r.get("n") or "–"),
            side(kk), side(tt),
            "–" if d is None else f"{100 * d:+.1f}",
            "–" if not kk else pct(kk.get("finish_length_frac"), 0),
            "–" if not kk else f"{kk.get('n_timeout') or 0}/{kk.get('n_context_overflow') or 0}",
            "–" if not tt else f"{tt.get('n_timeout') or 0}/{tt.get('n_context_overflow') or 0}",
            "–" if not kk else f"{(kk.get('wall_seconds') or 0) / 60:.0f}",
        ])
    header = ["benchmark", "group", "T", "n", "king % [95% CI]", "teacher % [95% CI]",
              "Δ pt", "king cap %", "king t/o/ctx", "teacher t/o/ctx", "min"]
    out = "\n".join(head) + "\n\n" + table(body, header, markdown)
    sk = card.get("skipped") or []
    if sk:
        out += "\n\nnot run: " + "; ".join(f"{s['env']} — {s['why']}" for s in sk)
    return out


def history(runs: list[dict], markdown: bool) -> str:
    envs = sorted({r["env"] for c in runs for r in c["rows"] if r["temperature"] == 0})
    header = ["benchmark"] + [f"reign {(c.get('king') or {}).get('reign')}" for c in runs] + ["teacher (latest)"]
    body = []
    latest = runs[-1]
    for env in envs:
        row = [env]
        for c in runs:
            m = next((r for r in c["rows"] if r["env"] == env and r["temperature"] == 0), None)
            row.append(pct(m["king"]["score"]) if m and m.get("king") else "–")
        m = next((r for r in latest["rows"] if r["env"] == env and r["temperature"] == 0), None)
        row.append(pct(m["teacher"]["score"]) if m and m.get("teacher") else "–")
        body.append(row)
    return table(body, header, markdown)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--run", default="")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    runs = load_runs(Path(a.state_dir))
    if not runs:
        print("no published runs in", a.state_dir)
        return 1
    if a.history:
        print(history(runs, a.markdown))
        return 0
    card = next((c for c in runs if c["run_id"] == a.run), runs[-1]) if a.run else runs[-1]
    print(scorecard(card, a.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
