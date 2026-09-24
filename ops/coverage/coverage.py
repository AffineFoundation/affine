"""Coverage check for the affine.io/#kings tables (operator directive 2026-09-15).

Rule: every row (teacher, genesis, every crowned non-revoked king) must have a
score in every held-out benchmark column and every datagen environment
column, and an environment score must rest on at least MIN_ENV_ROLLOUTS
graded rollouts. This script lists what is missing.

    python coverage.py                 # print the table to stdout
    python coverage.py --json PATH     # also write the machine-readable report
    python coverage.py --markdown PATH # also write a markdown table
    python coverage.py --post          # post the summary to the private Arbos
                                       # channel (DISCORD_BOT_TOKEN_ARBOS_BITTENSOR)

Reads ops/kingboard/state/matrix.json (built by ops/kingboard/build.py). A cell is `missing` when it has
no score, `low` when it is an environment cell with fewer graded rollouts than
MIN_ENV_ROLLOUTS, `running` when a benchmark pass for the row is in flight and
the cell is planned, and `unfillable` when the column cannot be scored at all
(an environment without a grader: the fold and the board both see no grade).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KINGBOARD_STATE = Path(os.environ.get("KINGBOARD_STATE_DIR", HERE.parent / "kingboard" / "state"))
STATE_DIR = Path(os.environ.get("COVERAGE_STATE_DIR", HERE / "state"))
MATRIX_PATH = KINGBOARD_STATE / "matrix.json"
COVERAGE_PATH = STATE_DIR / "coverage.json"
MIN_ENV_ROLLOUTS = 30            # operator: a king env score on < 30 rollouts is an inconsistency
# Operator 2026-09-15 20:27 UTC: rows before Affine-XII (reign 11) are NOT filled.
# Kings with a lower reign stay on the page but are out of the coverage contract.
MIN_KING_REIGN = int(os.environ.get("COVERAGE_MIN_KING_REIGN", "11"))
BACKFILL_ENV_TARGET = 50         # the backfill runs >= 50 rollouts per env per model
DISCORD_CHANNEL = "1510910974498967613"   # private Arbos ops channel (no public posts)
DISCORD_TOKEN_ENV = "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR"


def load_matrix(path: Path = MATRIX_PATH) -> dict:
    return json.loads(path.read_text())


def classify(matrix: dict) -> dict:
    cols = [c for c in matrix["columns"] if c["kind"] in ("bench", "env")]
    rows_out = []
    out_of_scope = []
    totals = {"missing_bench": 0, "missing_env": 0, "low_env": 0, "running_bench": 0,
              "unfillable_env": 0, "cells": 0, "complete": 0}
    for row in matrix["rows"]:
        if row["kind"] == "king" and int(row.get("reign") or 0) < MIN_KING_REIGN:
            out_of_scope.append({"label": row["label"], "reign": row.get("reign"), "digest12": row.get("digest12")})
            continue
        cells = row.get("cells") or {}
        missing_bench, missing_env, low_env, running, unfillable = [], [], [], [], []
        for c in cols:
            v = cells.get(c["key"]) or {}
            totals["cells"] += 1
            if c["kind"] == "env" and c.get("no_grader"):
                unfillable.append(c["env"])
                continue
            if c.get("budget_tag"):
                # `<env>@<tag>` = an older cap / budget kept for continuity; not a coverage target
                totals["cells"] -= 1
                continue
            if v.get("unverified") or v.get("errored_only") or v.get("partial"):
                # `partial` = a job interrupted / still being resumed by its owner: present, not a
                # gap (2026-09-24: 13 partial cells of the watcher's reign-21 pass made autofill
                # rent a second 5-pod fast pass for a card that was already complete)
                # published on purpose without a number (Gaia2 ambiguity: grader result
                # not verified; an env whose rollouts all errored): present, not a gap.
                # Counting these as missing re-queued full rows 51 times on 2026-09-21.
                totals["complete"] += 1
                continue
            if v.get("score") is None:
                if v.get("running"):
                    running.append(c["env"])
                elif c["kind"] == "bench":
                    missing_bench.append(c["env"])
                else:
                    missing_env.append(c["env"])
                continue
            totals["complete"] += 1
            if c["kind"] == "env" and int(v.get("n") or 0) < MIN_ENV_ROLLOUTS:
                low_env.append({"env": c["env"], "n": int(v.get("n") or 0)})
        totals["missing_bench"] += len(missing_bench)
        totals["missing_env"] += len(missing_env)
        totals["low_env"] += len(low_env)
        totals["running_bench"] += len(running)
        totals["unfillable_env"] += len(unfillable)
        rows_out.append({
            "key": row["key"], "kind": row["kind"], "label": row["label"],
            "reign": row.get("reign"), "digest12": row.get("digest12"),
            "model": row.get("model") or row.get("digest12"),
            "missing_bench": missing_bench, "missing_env": missing_env,
            "low_env": low_env, "running_bench": running, "unfillable_env": unfillable,
            "complete": not missing_bench and not missing_env and not low_env and not running,
        })
    n_bench = sum(1 for c in cols if c["kind"] == "bench")
    n_env = sum(1 for c in cols if c["kind"] == "env")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matrix_generated_at": matrix.get("generated_at"),
        "n_rows": len(rows_out), "n_bench": n_bench, "n_env": n_env,
        "min_env_rollouts": MIN_ENV_ROLLOUTS, "min_king_reign": MIN_KING_REIGN,
        "totals": totals, "rows": rows_out, "out_of_scope": out_of_scope,
        "inflight": matrix.get("inflight") or [],
    }


def spend_line() -> str:
    """Backfill spend from the queue (bench pods) and the pod ledger (env
    backfill), when present; one short clause for the nightly post."""
    parts = []
    try:
        q = json.loads((STATE_DIR / "bench_queue.json").read_text())
        pods = sum(float(e.get("pod_usd") or 0) for e in q)
        st = {}
        for e in q:
            st[e.get("status", "?")] = st.get(e.get("status", "?"), 0) + 1
        parts.append(f"bench queue {st} pods ${pods:.0f}")
    except (OSError, ValueError):
        pass
    try:
        led = json.loads((STATE_DIR / "backfill_pods.json").read_text())
        live = [e["pod"] for e in led if not e.get("released_at")]
        parts.append(f"env-backfill pods live {len(live)}")
    except (OSError, ValueError, KeyError):
        pass
    return "; ".join(parts)


def summary_lines(rep: dict) -> list[str]:
    t = rep["totals"]
    lines = [f"kings coverage {rep['generated_at'][:16]}Z — {rep['n_rows']} rows in scope (teacher, genesis, reign ≥ {rep['min_king_reign']}) × "
             f"({rep['n_bench']} benchmarks + {rep['n_env']} envs): {t['complete']}/{t['cells']} cells scored; "
             f"missing {t['missing_bench']} bench + {t['missing_env']} env, {t['low_env']} env cells on "
             f"< {rep['min_env_rollouts']} rollouts, {t['running_bench']} bench cells running, "
             f"{t['unfillable_env']} env cells with no grader" + (f" — {spend_line()}" if spend_line() else "")]
    for r in rep["rows"]:
        if r["complete"]:
            continue
        parts = []
        if r["missing_bench"]:
            parts.append(f"bench −{len(r['missing_bench'])}")
        if r["running_bench"]:
            parts.append(f"bench running {len(r['running_bench'])}")
        if r["missing_env"]:
            parts.append(f"env −{len(r['missing_env'])}")
        if r["low_env"]:
            parts.append(f"env <{rep['min_env_rollouts']}n: " + ", ".join(
                f"{x['env']}({x['n']})" for x in r["low_env"]))
        lines.append(f"• {r['label']} {r.get('digest12') or ''}: " + "; ".join(parts))
    return lines


def markdown(rep: dict) -> str:
    out = [f"# Kings coverage — {rep['generated_at']}", "",
           f"Rows in scope {rep['n_rows']} (teacher, genesis, kings with reign ≥ {rep['min_king_reign']}; "
           f"out of scope: {', '.join(r['label'] for r in rep['out_of_scope']) or 'none'}), "
           f"benchmark columns {rep['n_bench']}, environment columns {rep['n_env']}. "
           f"Scored cells {rep['totals']['complete']} of {rep['totals']['cells']}.", "",
           "| row | model | missing benchmarks | missing envs | env cells on < 30 rollouts | bench running |",
           "|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        out.append("| {label} | `{model}` | {mb} | {me} | {low} | {run} |".format(
            label=r["label"], model=r["model"],
            mb=(f"{len(r['missing_bench'])}: " + ", ".join(r["missing_bench"])) if r["missing_bench"] else "—",
            me=(f"{len(r['missing_env'])}: " + ", ".join(r["missing_env"])) if r["missing_env"] else "—",
            low=", ".join(f"{x['env']} ({x['n']})" for x in r["low_env"]) or "—",
            run=", ".join(r["running_bench"]) or "—"))
    return "\n".join(out) + "\n"


def post_discord(text: str) -> bool:
    token = os.environ.get(DISCORD_TOKEN_ENV)
    if not token:
        print(f"{DISCORD_TOKEN_ENV} unset; not posting", file=sys.stderr)
        return False
    # Discord caps a message at 2,000 characters
    body = json.dumps({"content": text[:1990]}).encode()
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages", data=body,
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json",
                 "User-Agent": "affine-kingboard-coverage/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return 200 <= r.status < 300


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(MATRIX_PATH))
    ap.add_argument("--json", default=str(COVERAGE_PATH))
    ap.add_argument("--markdown", default="")
    ap.add_argument("--post", action="store_true")
    args = ap.parse_args()
    rep = classify(load_matrix(Path(args.matrix)))
    lines = summary_lines(rep)
    print("\n".join(lines))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=1))
    if args.markdown:
        Path(args.markdown).write_text(markdown(rep))
    if args.post:
        ok = post_discord("\n".join(lines))
        print("posted" if ok else "post failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
