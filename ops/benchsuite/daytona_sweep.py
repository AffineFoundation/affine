#!/usr/bin/env python3
"""Delete orphaned Daytona sandboxes: a sandbox whose Harbor trial already has a result.json (the job
finished or moved on) or whose trial belongs to no job directory on this box. In-flight trials (a trial
dir without result.json under a job whose harbor process is alive) are kept.

  daytona_sweep.py            # sweep everything
  daytona_sweep.py --job DIR  # only sandboxes of trials under this harbor job dir (called by harbor_cell at job end)
  daytona_sweep.py --dry-run
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

RUNS = Path(os.environ.get("BENCH_HOME", str(Path.home() / "benchsuite"))) / "runs"


def api_key() -> str:
    k = os.environ.get("DAYTONA_API_KEY")
    if k:
        return k
    out = subprocess.run(["op", "read", "--no-newline", "op://Arbos/fywmj6vtq5delybw5c7a53l2qa/notesPlain"],
                         capture_output=True, text=True, env={**os.environ, "PATH": os.environ.get("PATH", "") + ":" + str(Path.home() / ".local/bin")}).stdout
    m = re.search(r"dtn_[A-Za-z0-9_-]+", out)
    if not m:
        sys.exit("no DAYTONA_API_KEY")
    return m.group(0)


def sandboxes(key: str) -> list[dict]:
    req = urllib.request.Request("https://app.daytona.io/api/sandbox", headers={"Authorization": "Bearer " + key})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return d if isinstance(d, list) else d.get("items", d)


def delete(key: str, sid: str) -> bool:
    req = urllib.request.Request(f"https://app.daytona.io/api/sandbox/{sid}?force=true", headers={"Authorization": "Bearer " + key}, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=60)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  delete {sid} failed: {e}")
        return False


def trial_dirs() -> dict[str, list[Path]]:
    """trial name -> every trial dir with that name under any harbor job on this box."""
    out: dict[str, list[Path]] = {}
    for d in RUNS.glob("*/*/*/harbor/*/"):
        out.setdefault(d.name, []).append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="", help="only trials under this harbor job dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-age-min", type=int, default=5, help="never touch sandboxes younger than this")
    a = ap.parse_args()
    key = api_key()
    items = sandboxes(key)
    job = Path(a.job).resolve() if a.job else None
    tdirs = trial_dirs()
    now = time.time()
    kept = removed = unknown = 0
    for x in items:
        labels = x.get("labels") or {}
        if not labels.get("harbor.managed"):
            kept += 1; continue
        sid = x.get("id")
        created = x.get("createdAt") or ""
        try:
            age_min = (now - time.mktime(time.strptime(created[:19], "%Y-%m-%dT%H:%M:%S"))) / 60
        except ValueError:
            age_min = 1e9
        if age_min < a.min_age_min:
            kept += 1; continue
        trial = re.sub(r"__env$", "", labels.get("harbor.session_id") or "")
        dirs = tdirs.get(trial, [])
        if job:
            dirs = [d for d in dirs if job in d.parents or d.parent == job]
            if not dirs:
                kept += 1; continue          # not this job's sandbox
        if not dirs:
            reason = "no trial dir on this box"
            unknown += 1
        elif all((d / "result.json").exists() for d in dirs):
            reason = "trial finished (result.json present)"
        else:
            kept += 1; continue              # in flight
        print(f"orphan {sid} {trial} ({reason}, age {age_min:.0f} min)" + (" [dry-run]" if a.dry_run else ""))
        if not a.dry_run and delete(key, sid):
            removed += 1
    print(f"daytona sweep: {len(items)} sandboxes, kept {kept}, removed {removed} (of which no-trial-dir {unknown}){' (dry-run)' if a.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
