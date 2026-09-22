"""Stage 4: build every rendered task and check it in Docker.

For each `<out>/e<epoch>/tasks/<task_id>/` (needs a Docker daemon; run on a
datagen pod or a rented builder box, never on the eval pod or the
benchmark-fill pods):

  1. `docker build` the image under the tag task.toml declares  -> else drop `build_failed`
  2. run tests on the UNTOUCHED image      -> reward must be 0  else drop `trivially_solved`
  3. run solution/solve.sh, then the tests -> reward must be 1  else drop `gold_fails`
  4. repeat 3 in a fresh container         -> reward must be 1  else drop `nondeterministic`

Rewards are read from /logs/verifier/reward.txt exactly as the pod's Harbor
verifier reads them. Containers run with `--network none` (task.toml says
allow_internet = false), 1 cpu, 2 GB. Results: `<out>/e<epoch>/validation.jsonl`
(one row per task with `ok` and `reason`) and `validation_summary.json`.
Images of dropped tasks are removed; `--keep-images` keeps everything.

    python validate.py --epoch 63 --out ~/terminal_gen/out --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import out_dir, write_jsonl

log = logging.getLogger("terminal_gen.validate")

BUILD_TIMEOUT = 900
SOLVE_TIMEOUT = 600
TEST_TIMEOUT = 180


def sh(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def image_of(task_dir: Path) -> str:
    return tomllib.loads((task_dir / "task.toml").read_text())["environment"]["docker_image"]


def run_in_container(image: str, task_dir: Path, script: str, timeout: int) -> tuple[int | None, int, str]:
    """(reward or None, exit code, tail of output). `script` runs under bash as root."""
    with tempfile.TemporaryDirectory(prefix="tg-logs-") as logs:
        cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", "1", "--memory", "2g",
               "-v", f"{task_dir / 'tests'}:/tests:ro",
               "-v", f"{task_dir / 'solution'}:/solution:ro",
               "-v", f"{logs}:/logs",
               image, "bash", "-c", script]
        code, tail = sh(cmd, timeout)
        reward_file = Path(logs) / "verifier" / "reward.txt"
        reward = None
        if reward_file.exists():
            try:
                reward = int(float(reward_file.read_text().strip() or "0"))
            except ValueError:
                reward = None
        return reward, code, tail


def validate_task(task_dir: Path, keep_images: bool) -> dict:
    tid = task_dir.name
    t0 = time.time()
    row: dict = {"task_id": tid, "ok": False}
    try:
        image = image_of(task_dir)
    except Exception as exc:  # noqa: BLE001
        row.update(reason="bad_task_toml", detail=str(exc)[:300])
        return row
    row["image"] = image
    code, tail = sh(["docker", "build", "-q", "-t", image, str(task_dir / "environment")], BUILD_TIMEOUT)
    row["build_seconds"] = round(time.time() - t0, 1)
    if code != 0:
        row.update(reason="build_failed", detail=tail[-1200:])
        return row
    try:
        reward, code, tail = run_in_container(image, task_dir, "bash /tests/test.sh", TEST_TIMEOUT)
        row["pristine_reward"] = reward
        if reward is None:
            row.update(reason="verifier_error", detail=tail[-800:])
            return row
        if reward != 0:
            row.update(reason="trivially_solved")
            return row
        solve_and_test = "bash /solution/solve.sh; echo SOLVE_EXIT=$? ; bash /tests/test.sh"
        for attempt in (1, 2):
            reward, code, tail = run_in_container(image, task_dir, solve_and_test, SOLVE_TIMEOUT + TEST_TIMEOUT)
            row[f"solve{attempt}_reward"] = reward
            if "SOLVE_EXIT=0" not in tail:
                row.update(reason="solve_failed" if attempt == 1 else "nondeterministic", detail=tail[-800:])
                return row
            if reward != 1:
                row.update(reason="gold_fails" if attempt == 1 else "nondeterministic", detail=tail[-800:])
                return row
        row["ok"] = True
        row["reason"] = ""
        return row
    finally:
        row["seconds"] = round(time.time() - t0, 1)
        # the package ships task dirs, the pods build images themselves;
        # keeping thousands of validated images filled the builder's overlay
        if not keep_images:
            sh(["docker", "rmi", "-f", image], 120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out", default="~/terminal_gen/out")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated task ids")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if shutil.which("docker") is None:
        raise SystemExit("docker not found; run on a datagen pod or a builder box")
    out = out_dir(args.out, args.epoch)
    dirs = sorted(d for d in (out / "tasks").iterdir() if d.is_dir() and (d / "task.toml").exists())
    if args.only:
        want = set(args.only.split(","))
        dirs = [d for d in dirs if d.name in want]
    log.info("validating %d tasks with %d workers", len(dirs), args.concurrency)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda d: validate_task(d, args.keep_images), dirs))
    for r in rows:
        log.info("%s %s %s", r["task_id"], "OK" if r["ok"] else "DROP", r.get("reason", ""))
    write_jsonl(out / "validation.jsonl", rows)
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["ok"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    n_built = sum(1 for r in rows if r.get("reason") != "build_failed" and "bad_task_toml" != r.get("reason"))
    summary = {"tasks": len(rows), "built": n_built, "ok": sum(r["ok"] for r in rows), "drops": reasons,
               "build_success_rate": round(n_built / len(rows), 3) if rows else None,
               "validated_yield": round(sum(r["ok"] for r in rows) / len(rows), 3) if rows else None}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("validate: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
