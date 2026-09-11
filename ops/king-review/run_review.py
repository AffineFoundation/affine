"""Step 5 -- run the whole review for one king, or watch for crowns.

  python run_review.py --king current                # one pass on the live king
  python run_review.py --king king-0ce59769300c --reign 11
  python run_review.py --watch --poll-seconds 300     # run once per new crown
  python run_review.py --print-pm2                    # pm2 snippet (not deployed)

One pass = select.py -> judge.py -> aggregate.py -> labels_out.py, writing
under `--out-root/reign-<digest12>/`:
  sample.jsonl, sample_cells.json        what was read
  cache/judgments.jsonl, cost.json       every judge response + the ledger
  report.md, report.json                 the per-reign review
  king_pivots/<digest>.jsonl             side-table for the fold (phase 2)
  king_pivots/route_king_pivot.proposed.patch.txt

`--watch` polls the public snapshot (`api/v1/snapshot` -> king.revision)
and runs a pass whenever the digest changes (state in
`<out-root>/watch_state.json`). It never touches the validator, the eval
pod, affine.toml or the fold.

Secrets: the judge key is read from OPENROUTER_API_KEY / OPENROUTER / ENGY_2
or `--key-file`; nothing is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from krlib import DEFAULT_SNAPSHOT_URL, digest12, resolve_current_king

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = Path(os.environ.get("KING_REVIEW_OUT", "/tmp/king-review"))

PM2_SNIPPET = """// pm2 entry for the per-reign king review (NOT deployed yet).
// Add to the validator box's ecosystem file next to affine-king-datagen.
// Needs: the repo venv, OPENROUTER_API_KEY in the env file (never in the
// repo), ~3 GB disk for the trace cache under KING_REVIEW_CACHE.
{
  name: "affine-king-review",
  cwd: "/home/const/subnet120/ops/king-review",
  script: "/home/const/subnet120/.venv/bin/python",
  args: "run_review.py --watch --poll-seconds 300 --max-usd 30 --per-cell 16 --max-total 250",
  interpreter: "none",
  env_file: "/home/const/.affine-validator.env",   // OPENROUTER_API_KEY lives here
  env: {
    KING_REVIEW_CACHE: "/home/const/king-review/traces",
    KING_REVIEW_OUT: "/home/const/king-review",
  },
  autorestart: true,
  max_restarts: 20,
  restart_delay: 60000,
  out_file: "/home/const/.pm2/logs/affine-king-review.log",
  error_file: "/home/const/.pm2/logs/affine-king-review.err",
}
// How to deploy when approved: sync ops/king-review to the box, add the
// block, `pm2 start ecosystem.config.js --only affine-king-review`,
// `pm2 save`. The first tick runs a pass on the current king (the watcher
// state file is empty), later ticks only on a crown. Reports land under
// KING_REVIEW_OUT/reign-<digest12>/; copy report.md + king_pivots/ to the
// Project store / fold input by hand or by a follow-up step.
"""


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def run(step: str, args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / step), *args]
    log("+ " + " ".join(a if "key" not in a.lower() else "<redacted>" for a in cmd[1:]))
    subprocess.run(cmd, check=True, cwd=HERE)


def review_once(*, king: str, reign: str | None, out_root: Path, per_cell: int,
                max_total: int, seed: int, model: str, concurrency: int, max_usd: float,
                key_file: str | None, no_sync: bool) -> Path:
    digest = digest12(king)
    out_dir = out_root / f"reign-{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = out_dir / "sample.jsonl"
    sel = ["--king", digest, "--out", str(sample), "--per-cell", str(per_cell),
           "--max-total", str(max_total), "--seed", str(seed)]
    if no_sync:
        sel.append("--no-sync")
    run("select.py", sel)
    judge = ["--sample", str(sample), "--out-dir", str(out_dir), "--model", model,
             "--concurrency", str(concurrency), "--max-usd", str(max_usd)]
    if key_file:
        judge += ["--key-file", key_file]
    run("judge.py", judge)
    agg = ["--out-dir", str(out_dir)]
    if reign:
        agg += ["--reign", str(reign)]
    run("aggregate.py", agg)
    run("labels_out.py", agg)
    log(f"review done: {out_dir / 'report.md'}")
    return out_dir


def watch(args: argparse.Namespace) -> None:
    state_path = args.out_root / "watch_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    log(f"watch: polling {args.snapshot_url} every {args.poll_seconds}s; "
        f"last reviewed {state.get('digest12')}")
    while True:
        try:
            king = resolve_current_king(args.snapshot_url)
        except Exception as ex:  # network blip: keep watching
            log(f"watch: snapshot unavailable ({type(ex).__name__}); retrying")
            time.sleep(args.poll_seconds)
            continue
        if king["digest12"] != state.get("digest12"):
            log(f"watch: king is {king['digest12']} (reign {king['reign_number']}, "
                f"crowned {king['crowned_at']}); starting review")
            # let the king seat produce rollouts first: a fresh crown has none
            if state.get("digest12") and args.settle_seconds:
                time.sleep(args.settle_seconds)
            try:
                out_dir = review_once(
                    king=king["digest12"], reign=str(king["reign_number"]),
                    out_root=args.out_root, per_cell=args.per_cell,
                    max_total=args.max_total, seed=args.seed, model=args.model,
                    concurrency=args.concurrency, max_usd=args.max_usd,
                    key_file=args.key_file, no_sync=False)
                state = {"digest12": king["digest12"], "reign": king["reign_number"],
                         "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "out_dir": str(out_dir)}
                state_path.write_text(json.dumps(state, indent=1))
            except subprocess.CalledProcessError as ex:
                log(f"watch: review failed ({ex}); will retry next poll")
        time.sleep(args.poll_seconds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--king", default="current", help="digest (12+ hex) or 'current'")
    ap.add_argument("--reign", default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--settle-seconds", type=int, default=6 * 3600,
                    help="watch mode: wait this long after a crown so the king seat has rollouts")
    ap.add_argument("--snapshot-url", default=DEFAULT_SNAPSHOT_URL)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--per-cell", type=int, default=16)
    ap.add_argument("--max-total", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro-0813")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-usd", type=float, default=30.0)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--print-pm2", action="store_true")
    args = ap.parse_args()
    if args.print_pm2:
        print(PM2_SNIPPET)
        return
    if args.watch:
        watch(args)
        return
    reign = args.reign
    king = args.king
    if king == "current":
        cur = resolve_current_king(args.snapshot_url)
        king, reign = cur["digest12"], reign or str(cur["reign_number"])
    review_once(king=king, reign=reign, out_root=args.out_root, per_cell=args.per_cell,
                max_total=args.max_total, seed=args.seed, model=args.model,
                concurrency=args.concurrency, max_usd=args.max_usd,
                key_file=args.key_file, no_sync=args.no_sync)


if __name__ == "__main__":
    main()
