"""Step 5 -- run the whole review for one king, or run it as a service.

  python run_review.py --king current                       # one stratified pass
  python run_review.py --king king-0ce59769300c --reign 11 --all --merge-into-dir ../../affine/state/king_pivots
  python run_review.py --watch --state-json ../../affine/state/state.json --daily-at 14:30
  python run_review.py --print-pm2                          # pm2 block for the box

One pass = select.py -> judge.py -> aggregate.py -> labels_out.py, writing
under `--out-root/reign-<digest12>/`:
  sample.jsonl, sample_cells.json        what was read
  cache/judgments.jsonl, cost.json,      every judge response + the ledger
  ledger.jsonl
  report.md, report.json                 the per-reign review
  king_pivots/<digest>.jsonl             side-table (also merged into
                                         --merge-into-dir for the fold)
  king_pivots/routing_summary.json       admitted / already published / will route

Service mode (`--watch`, the pm2 entry on the validator box):
  * every `--poll-seconds` the king is read from `--state-json` (the
    validator's own state.json; the public api/v1/snapshot when no path is
    given). A NEW king is reviewed as soon as the king seat has produced
    `--min-failed` failed rollouts for it (checked against the trace
    manifest each tick, so no blind settle timer).
  * once a day at `--daily-at` (UTC) the current king is reviewed again in
    `--all` mode: every failed rollout, of which only the not-yet-judged
    ones cost anything (cache), and the side-table is merged into the
    fold's directory. The fold cron is 16:00 UTC; 14:30 leaves the table on
    disk in time.
  State: `<out-root>/watch_state.json`. It never touches the validator, the
  eval pod, affine.toml or the fold.

Secrets: the judge key is read from OPENROUTER_API_KEY / OPENROUTER / ENGY_2
or `--key-file`; nothing is printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from krlib import (DEFAULT_SNAPSHOT_URL, KING_PREFIX, TraceStore, digest12,
                   load_king_pivot_config, resolve_current_king)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_OUT_ROOT = Path(os.environ.get("KING_REVIEW_OUT", "/tmp/king-review"))

PM2_SNIPPET = """// pm2 entry for the per-reign king review on the validator box.
// The wrapper loads ~/.affine-validator.env the way ops/run_validator.sh
// does (OPENROUTER_API_KEY lives there; nothing in the repo), points the
// trace cache + outputs at affine/state/king_review/ and merges the
// side-table into affine/state/king_pivots/<digest>.jsonl, where the fold
// ([king_pivot] in rollouts/rollouts/sources.toml) reads it at 16:00 UTC.
//
//   pm2 start /home/const/subnet120/ops/king-review/run_king_review.sh \\
//       --name affine-king-review --interpreter bash -- watch
//   pm2 save
//
// One-off runs through the same env:
//   ops/king-review/run_king_review.sh once --all --max-usd 90
{
  name: "affine-king-review",
  script: "/home/const/subnet120/ops/king-review/run_king_review.sh",
  args: "watch",
  interpreter: "bash",
  cwd: "/home/const/subnet120/ops/king-review",
  autorestart: true,
  max_restarts: 50,
  restart_delay: 60000,
  out_file: "/home/const/.pm2/logs/affine-king-review.log",
  error_file: "/home/const/.pm2/logs/affine-king-review.err",
}
"""


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def run(step: str, args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / step), *args]
    shown = []
    hide_next = False
    for a in cmd[1:]:
        shown.append("<redacted>" if hide_next else a)
        hide_next = a == "--key-file"
    log("+ " + " ".join(shown))
    subprocess.run(cmd, check=True, cwd=HERE)


def review_once(a: argparse.Namespace, *, king: str, reign: str | None,
                all_failed: bool) -> Path:
    digest = digest12(king)
    out_dir = a.out_root / f"reign-{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = out_dir / "sample.jsonl"
    sel = ["--king", digest, "--out", str(sample), "--per-cell", str(a.per_cell),
           "--max-total", str(a.max_total), "--seed", str(a.seed), "--procs", str(a.procs)]
    if all_failed:
        sel.append("--all")
    if a.include_excluded_sources:
        sel.append("--include-excluded-sources")
    if a.no_sync:
        sel.append("--no-sync")
    run("select.py", sel)
    judge = ["--sample", str(sample), "--out-dir", str(out_dir), "--model", a.model,
             "--concurrency", str(a.concurrency), "--max-usd", str(a.max_usd)]
    if a.key_file:
        judge += ["--key-file", a.key_file]
    run("judge.py", judge)
    agg = ["--out-dir", str(out_dir)]
    if reign:
        agg += ["--reign", str(reign)]
    run("aggregate.py", agg)
    lab = list(agg)
    if a.merge_into_dir:
        lab += ["--merge-into-dir", str(a.merge_into_dir)]
    if a.check_published:
        lab.append("--check-published")
    run("labels_out.py", lab)
    log(f"review done: {out_dir / 'report.md'}")
    return out_dir


def failed_rollouts_for(king12: str, procs: int, log_fn=log) -> int:
    """How many failed rollouts the trace store holds for this king (syncs
    the manifest + new chunks first)."""
    ts = TraceStore()
    ts.sync(log=log_fn)
    rows = ts.index(log=log_fn, procs=procs)
    fold = load_king_pivot_config()
    return sum(1 for r in rows if r["king"] == f"king-{king12}" and r["outcome"] == "failed"
               and r["n_replies"] > 0 and r["policy_id"].startswith(KING_PREFIX)
               and r["source"] not in fold["exclude_sources"])


def parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def daily_due(state: dict, now: dt.datetime, at: tuple[int, int]) -> bool:
    today = now.date().isoformat()
    if state.get("last_daily_date") == today:
        return False
    return (now.hour, now.minute) >= at


def watch(a: argparse.Namespace) -> None:
    state_path = a.out_root / "watch_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    at = parse_hhmm(a.daily_at)
    log(f"watch: king from {a.state_json or a.snapshot_url}, poll {a.poll_seconds}s, "
        f"daily --all run at {a.daily_at} UTC, last reviewed {state.get('digest12')} "
        f"(daily {state.get('last_daily_date')})")

    def save() -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1))

    while True:
        now = dt.datetime.now(dt.timezone.utc)
        try:
            king = resolve_current_king(a.snapshot_url, state_json=a.state_json)
        except Exception as ex:  # unreadable state / network blip: keep watching
            log(f"watch: king unavailable ({type(ex).__name__}: {str(ex)[:120]}); retrying")
            time.sleep(a.poll_seconds)
            continue
        reason = None
        if king["digest12"] != state.get("digest12"):
            try:
                n = failed_rollouts_for(king["digest12"], a.procs, log_fn=lambda m: None)
            except Exception as ex:
                log(f"watch: trace sync failed ({type(ex).__name__}); retrying")
                time.sleep(a.poll_seconds)
                continue
            if n >= a.min_failed:
                reason = f"new king {king['digest12']} (reign {king['reign_number']}, " \
                         f"{n} failed rollouts on record)"
            else:
                log(f"watch: king {king['digest12']} has {n} failed rollouts (< {a.min_failed}); waiting")
        elif daily_due(state, now, at):
            reason = f"daily run {now.date().isoformat()}"
        if reason:
            log(f"watch: starting review -- {reason}")
            try:
                out_dir = review_once(a, king=king["digest12"], reign=str(king["reign_number"]),
                                      all_failed=(not reason.startswith("new king")) or a.all)
                state.update(digest12=king["digest12"], reign=king["reign_number"],
                             reviewed_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             out_dir=str(out_dir), last_reason=reason)
                if reason.startswith("daily"):
                    state["last_daily_date"] = now.date().isoformat()
                save()
            except subprocess.CalledProcessError as ex:
                log(f"watch: review failed ({ex}); will retry next poll")
                if reason.startswith("daily"):
                    # do not hammer the judge all afternoon on a persistent failure
                    state["last_daily_date"] = now.date().isoformat()
                    save()
        time.sleep(a.poll_seconds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--king", default="current", help="digest (12+ hex) or 'current'")
    ap.add_argument("--reign", default=None)
    ap.add_argument("--all", action="store_true", help="every failed rollout, not a sample")
    ap.add_argument("--include-excluded-sources", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=600)
    ap.add_argument("--daily-at", default="14:30", help="UTC HH:MM of the daily --all run")
    ap.add_argument("--min-failed", type=int, default=20,
                    help="watch: review a new king once it has this many failed rollouts")
    ap.add_argument("--state-json", type=Path, default=None,
                    help="validator state.json to read the king from (else the public API)")
    ap.add_argument("--snapshot-url", default=DEFAULT_SNAPSHOT_URL)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--merge-into-dir", type=Path, default=None,
                    help="fold side-table dir, e.g. <repo>/affine/state/king_pivots")
    ap.add_argument("--check-published", action="store_true")
    ap.add_argument("--per-cell", type=int, default=16)
    ap.add_argument("--max-total", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro-0813")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-usd", type=float, default=30.0)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--print-pm2", action="store_true")
    a = ap.parse_args()
    if a.print_pm2:
        print(PM2_SNIPPET)
        return
    if a.watch:
        watch(a)
        return
    reign = a.reign
    king = a.king
    if king == "current":
        cur = resolve_current_king(a.snapshot_url, state_json=a.state_json)
        king, reign = cur["digest12"], reign or str(cur["reign_number"])
    review_once(a, king=king, reign=reign, all_failed=a.all)


if __name__ == "__main__":
    main()
