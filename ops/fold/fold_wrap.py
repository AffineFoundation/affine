#!/usr/bin/env python
"""Wrapper around ops/corpus_build.py (the daily / 6-hourly fold).

Failure this closes (2026-09-17): the 16:00 UTC cron fold died on a KeyError,
published nothing, and nobody knew for a day — pm2 cron jobs exit quietly.

What it does:
  * writes affine/state/fold/last_run.json at START (status running, pid,
    argv, git HEAD, sha256 of rollouts/rollouts/sources.toml, the pm2 cron
    expression of affine-corpus-refresh) and again at EXIT (exit_code,
    duration, traceback tail, attempt count, classification);
  * tees the fold's output to affine/state/fold/runs/<utc>.log (pm2 keeps
    its own copy);
  * on a non-zero exit that looks transient (network / HTTP 5xx / timeout)
    retries ONCE after `retry_wait_s`; any other failure is final;
  * pages the private Discord channel on every final non-zero exit with the
    traceback tail; a fold refused by the lock (another fold running) is
    recorded as `skipped_lock`, no page (the health monitor's epoch-age check
    catches a fold that never happens);
  * appends one row per run to affine/state/fold/history.jsonl.

Usage (pm2 runs ops/fold/run_fold.sh which execs this):
    python ops/fold/fold_wrap.py [-- <corpus_build.py args>]
    python ops/fold/fold_wrap.py --dry-run -- --no-publish --no-announce
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ops" / "health"))
import common  # noqa: E402

FOLD_DIR = REPO / "affine" / "state" / "fold"
LAST_RUN = FOLD_DIR / "last_run.json"
HISTORY = FOLD_DIR / "history.jsonl"
RUNS_DIR = FOLD_DIR / "runs"
SOURCES_TOML = REPO / "rollouts" / "rollouts" / "sources.toml"
CORPUS_BUILD = REPO / "ops" / "corpus_build.py"
PM2_NAME = "affine-corpus-refresh"
SKIP_NEXT = HERE / "SKIP_NEXT"
TAG = "fold-wrap"

TRANSIENT_RE = re.compile(
    r"(ConnectionError|ConnectTimeout|ReadTimeout|Timeout|RemoteDisconnected|"
    r"IncompleteRead|SSLError|ChunkedEncodingError|HTTPSConnectionPool|"
    r"Connection reset by peer|temporarily unavailable|\b50[234]\b|Service Unavailable|"
    r"Max retries exceeded|Name or service not known|EndpointConnectionError)", re.I)
LOCK_RE = re.compile(r"another fold is running", re.I)


def log(msg: str) -> None:
    common.log(TAG, msg)


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       text=True, timeout=20).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def sources_dirty() -> bool | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--",
                            str(SOURCES_TOML.relative_to(REPO))], timeout=20)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    return r.returncode != 0


def cron_expr() -> str | None:
    p = common.pm2_process(common.pm2_jlist(), PM2_NAME)
    return (p or {}).get("cron")


def traceback_tail(text: str, n_lines: int = 40) -> str:
    """The last Python traceback in `text` (or the last n lines)."""
    idx = text.rfind("Traceback (most recent call last)")
    chunk = text[idx:] if idx >= 0 else text
    lines = [l for l in chunk.splitlines() if l.strip()]
    return "\n".join(lines[-n_lines:])


def classify(exit_code: int, tail: str) -> str:
    if exit_code == 0:
        return "ok"
    if LOCK_RE.search(tail):
        return "skipped_lock"
    if TRANSIENT_RE.search(tail):
        return "transient"
    return "error"


def run_once(args: list[str], attempt: int, log_path: Path, dry_run: bool) -> tuple[int, str, float]:
    cmd = [common.python_bin(), str(CORPUS_BUILD), *args]
    log(f"attempt {attempt}: {common.shell_quote(cmd)}")
    started = time.time()
    if dry_run:
        return 0, "", 0.0
    buf: list[str] = []
    with open(log_path, "a") as lf:
        lf.write(f"=== attempt {attempt} {common.now_iso()} {common.shell_quote(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            buf.append(line)
            if len(buf) > 4000:
                del buf[:1000]
        rc = proc.wait()
    return rc, "".join(buf[-400:]), time.time() - started


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="record + page logic without running the fold")
    ap.add_argument("--retry-wait-s", type=int, default=90)
    ap.add_argument("--no-page", action="store_true")
    ap.add_argument("fold_args", nargs="*", help="arguments passed to corpus_build.py (after --)")
    args = ap.parse_args()

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if SKIP_NEXT.exists():
        # `pm2 start` runs the script at once as well as on the cron; the
        # re-point helper sets this flag so that first run is a no-op.
        SKIP_NEXT.unlink()
        log("SKIP_NEXT flag present: skipping this run (pm2 start after re-point); next cron run folds")
        return 0
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = RUNS_DIR / f"{stamp}.log"
    started = time.time()
    rec = {
        "status": "running", "started_at": common.iso(started), "started_ts": started,
        "pid": os.getpid(), "argv": args.fold_args, "git_head": git_head(),
        "sources_toml_sha256": common.sha256_file(SOURCES_TOML),
        "sources_toml_dirty": sources_dirty(),
        "cron": cron_expr(), "pm2_name": PM2_NAME, "log": str(log_path),
        "attempts": 0, "exit_code": None, "classification": None, "traceback_tail": None,
        "ended_at": None, "duration_s": None, "dry_run": args.dry_run,
    }
    common.atomic_write_json(LAST_RUN, rec)

    attempt = 0
    rc, tail, dur = 1, "", 0.0
    while attempt < 2:
        attempt += 1
        rec["attempts"] = attempt
        rc, tail, dur = run_once(args.fold_args, attempt, log_path, args.dry_run)
        cls = classify(rc, tail)
        rec.update({"exit_code": rc, "classification": cls,
                    "traceback_tail": traceback_tail(tail) if rc != 0 else None,
                    "last_attempt_duration_s": round(dur, 1)})
        common.atomic_write_json(LAST_RUN, rec)
        if cls == "transient" and attempt < 2:
            log(f"attempt {attempt} failed transiently (exit {rc}); retry in {args.retry_wait_s}s")
            time.sleep(args.retry_wait_s)
            continue
        break

    ended = time.time()
    rec.update({"status": "done", "ended_at": common.iso(ended),
                "duration_s": round(ended - started, 1)})
    common.atomic_write_json(LAST_RUN, rec)
    with open(HISTORY, "a") as fh:
        fh.write(json.dumps({k: v for k, v in rec.items() if k != "traceback_tail"},
                                   sort_keys=True, default=str) + "\n")

    cls = rec["classification"]
    if cls == "ok":
        log(f"fold ok in {rec['duration_s']}s (attempts {attempt})")
    elif cls == "skipped_lock":
        log("fold skipped: another fold holds ops/corpus_build/fold.lock (no page)")
    else:
        tb = (rec.get("traceback_tail") or "").splitlines()
        short = "\n".join(tb[-8:])
        text = (f"FOLD FAILED exit {rc} ({cls}, {attempt} attempt(s), {rec['duration_s']}s; "
                f"HEAD {rec['git_head']}, cron `{rec['cron']}`) — log {log_path.name}\n"
                f"```\n{short[:1200]}\n```")
        log(text.replace("\n", " | "))
        if not args.no_page:
            common.discord_post(text, prefix="[fold]", dry_run=args.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
