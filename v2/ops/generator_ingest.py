#!/usr/bin/env python3
"""Process accepted generator epochs into data-only reserve PRs.

Dry-run only scans local inputs: no clone/network, commit, branch, PR, merge or
state write. Real operation creates a fresh worktree from origin/v2 per epoch
and merges only a PR whose changed files all live below
v2/sn120_v2/envs/reserve/. GH_TOKEN is read only by gh; never printed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

EPOCH = re.compile(r"accepted-([A-Za-z0-9][A-Za-z0-9_.:-]*)\.jsonl\Z")
RESERVE_PREFIX = "v2/sn120_v2/envs/reserve/"


def run(args: list[str], *, cwd: Path | None = None, capture: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=capture, check=True, env=env)


def epochs(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("accepted-*.jsonl")):
        match = EPOCH.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        found[match.group(1)] = path
    return found


def processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def mark_processed(path: Path, epoch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(epoch + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def changed_files(worktree: Path, base: str, runner=run) -> list[str]:
    return runner(["git", "diff", "--name-only", f"origin/{base}...HEAD"], cwd=worktree).stdout.splitlines()


def _merge_sha(number: int, base: str, runner=run) -> str:
    # gh pr merge does not necessarily fetch the new base ref locally. Read the
    # durable server-side PR and only print its confirmed, merged commit SHA.
    runner(["gh", "pr", "merge", str(number), "--merge"], capture=True)
    row = json.loads(runner(["gh", "pr", "view", str(number), "--json", "state,mergeCommit", "--jq", "."], capture=True).stdout)
    if row.get("state") != "MERGED" or not isinstance(row.get("mergeCommit"), dict) or not row["mergeCommit"].get("oid"):
        raise RuntimeError(f"PR {number} did not report a confirmed merge commit")
    return row["mergeCommit"]["oid"]


def process_epoch(epoch: str, source: Path, *, work: Path, remote: str, base: str, prior_dir: Path,
                  merge: bool, dry_run: bool, runner=run) -> dict[str, Any]:
    report: dict[str, Any] = {"epoch": epoch, "input": str(source), "dry_run": dry_run}
    if dry_run:
        report.update({"accepted": str(source), "prior_dir": str(prior_dir),
                       "would_branch": f"generator/{epoch}", "would_merge": merge})
        return report
    if work.exists():
        raise RuntimeError(f"worktree already exists: {work}")
    repository = Path(__file__).resolve().parents[2]
    runner(["git", "worktree", "add", "--detach", str(work), f"origin/{base}"], cwd=repository)
    try:
        env = os.environ.copy()
        env["SN120_RESERVE_DIR"] = str(work / "v2/sn120_v2/envs/reserve")
        env["PYTHONPATH"] = str(work / "v2")
        python = os.environ.get("PYTHON", sys.executable)
        ingest = json.loads(runner([python, "-m", "sn120_v2.envs.reserve", "ingest-generator", str(source),
                                    "--prior-dir", str(prior_dir)], cwd=work / "v2", capture=True, env=env).stdout)
        runner([python, "-m", "sn120_v2.envs.reserve", "verify"], cwd=work / "v2", capture=True, env=env)
        status = [line for line in runner(["git", "status", "--porcelain"], cwd=work).stdout.splitlines() if line.strip()]
        outside = [line[3:] for line in status if not line[3:].startswith(RESERVE_PREFIX)]
        if outside:
            raise RuntimeError(f"refusing non-reserve changes: {outside}")
        if not status:
            report.update({"ingested": 0, "merged": False, "merge_sha": None})
            return report
        branch = f"generator/{epoch}"
        runner(["git", "checkout", "-b", branch], cwd=work)
        runner(["git", "add", "--", RESERVE_PREFIX], cwd=work)
        runner(["git", "commit", "-m", f"generator: accepted tasks for {epoch}"], cwd=work)
        outside = [name for name in changed_files(work, base, runner) if not name.startswith(RESERVE_PREFIX)]
        if outside:
            raise RuntimeError(f"refusing non-reserve commit: {outside}")
        runner(["git", "push", "-u", remote, branch], cwd=work)
        pr_url = runner(["gh", "pr", "create", "--base", base, "--head", branch,
                         "--title", f"generator: accepted tasks for {epoch}",
                         "--body", f"Data-only accepted generator tasks for {epoch}."], capture=True).stdout.strip().splitlines()[-1]
        number = int(pr_url.rstrip("/").rsplit("/", 1)[-1])
        files = runner(["gh", "pr", "view", str(number), "--json", "files", "--jq", ".files[].path"], capture=True).stdout.splitlines()
        if any(not name.startswith(RESERVE_PREFIX) for name in files):
            raise RuntimeError(f"refusing PR {number}: non-reserve files changed")
        result = _merge_sha(number, base, runner) if merge else None
        report.update({"ingested": ingest["ingested"], "merged": merge, "pr": number, "merge_sha": result})
        return report
    finally:
        runner(["git", "worktree", "remove", "--force", str(work)], cwd=repository)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="/home/box/sn120-scratch/generator")
    parser.add_argument("--prior-dir", required=True, help="verified sn120-math-prior-v1 release directory")
    parser.add_argument("--state", default=str(Path.home() / ".sn120-generator-ingest" / "processed.txt"))
    parser.add_argument("--work", default="", help="temporary parent for per-epoch worktrees")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--base", default="v2")
    parser.add_argument("--no-merge", action="store_true", help="open the PR but do not merge it")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    state = Path(args.state)
    done = processed(state)
    pending = [(epoch, path) for epoch, path in epochs(Path(args.directory)).items() if epoch not in done]
    if not pending:
        print(json.dumps({"processed": 0, "pending": 0, "dry_run": args.dry_run}))
        return 0
    outputs = []
    for epoch, path in pending:
        with tempfile.TemporaryDirectory(prefix="sn120-generator-ingest-") as temp:
            work = Path(args.work) / epoch if args.work else Path(temp) / "worktree"
            result = process_epoch(epoch, path, work=work, remote=args.remote, base=args.base, prior_dir=Path(args.prior_dir),
                                   merge=not args.no_merge, dry_run=args.dry_run)
            outputs.append(result)
            if not args.dry_run:
                mark_processed(state, epoch)
    print(json.dumps({"processed": len(outputs), "pending": len(pending), "epochs": outputs, "dry_run": args.dry_run}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
