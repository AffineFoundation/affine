#!/usr/bin/env python
"""Push finished benchmark-suite cells to Prime Evals after the fact.

The live path (`uv run eval --push`) opens a platform run before the first
rollout and streams episodes into it through the `prime_runs` SDK. It failed
on every run of 2026-09-12/13 because the account had no username; this
replays the stored `traces.jsonl(.gz)` of each cell through the same SDK calls
(`pr.init` → `run.log_episodes` → `run.finish`), one platform run per cell,
named like the live runs (`affine-bench-<run_id>-<model>-<env>-t<T>`), and
writes the resulting URLs into the cell's summary.json and the run manifest
(`prime_evals`), so publish.py carries them into the scorecard.

  <verifiers>/.venv/bin/python push_evals.py --run-dir ~/benchsuite/runs/<run_id> [--models king] [--public]

Needs PRIME_API_KEY. Idempotent: cells whose summary already has `prime_eval_url` are skipped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import prime_runs as pr
from verifiers.v1.episode import Episode

HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[push_evals] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def load_episodes(cell: Path) -> list[Episode]:
    path = cell / "traces.jsonl"
    opener = open
    if not path.exists():
        path = cell / "traces.jsonl.gz"
        opener = gzip.open
    episodes = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                episodes.append(Episode.model_validate_json(line))
    return episodes


def push_cell(cell: Path, run_id: str, public: bool) -> str | None:
    summ_path = cell / "summary.json"
    if not summ_path.exists():
        return None
    summ = json.loads(summ_path.read_text())
    if summ.get("prime_eval_url"):
        log(f"skip {cell.parent.name}/{cell.name}: already pushed -> {summ['prime_eval_url']}")
        return summ["prime_eval_url"]
    if summ.get("reused_from"):
        return None          # a copied teacher baseline: its traces live in the source run
    episodes = load_episodes(cell)
    name = f"affine-bench-{run_id}-{summ['model']}-{summ['env']}-t{summ['temperature']:g}"
    run = pr.init(mode="online", name=name, environments=[summ["taskset"]], model=summ["model"],
                  framework="verifiers",
                  config={"model": summ["model"], "num_examples": len({e.task.key for e in episodes}),
                          "rollouts_per_example": summ.get("rollouts_per_example", 1),
                          "temperature": summ["temperature"], "max_tokens": summ["max_tokens"],
                          "affine_run_id": run_id, "public": public})
    run.log_episodes(episodes)
    try:
        summary = pr.metrics.from_episodes(episodes)
    except Exception as e:  # noqa: BLE001 - close the run even without its headline
        log(f"metrics failed ({e!r}); finishing without a summary")
        summary = None
    run.finish(summary, status=pr.RunStatus.COMPLETED)
    url = run.url
    summ["prime_eval_url"] = url
    summ["prime_eval_id"] = run.id
    summ_path.write_text(json.dumps(summ, indent=1))
    log(f"pushed {cell.parent.name}/{cell.name}: {len(episodes)} episodes -> {url}")
    return url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default="king,teacher")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    run_dir = Path(a.run_dir).expanduser()
    run_id = run_dir.name
    urls = {}
    for model in a.models.split(","):
        for cell in sorted((run_dir / model).glob("*__t*")):
            if not cell.is_dir():
                continue
            try:
                url = push_cell(cell, run_id, a.public)
            except Exception as e:  # noqa: BLE001 - one bad cell must not stop the rest
                log(f"FAILED {model}/{cell.name}: {type(e).__name__}: {e}")
                continue
            if url:
                urls[f"{model}/{cell.name}"] = url
    man_path = run_dir / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        man.setdefault("prime_evals", {}).update(urls)
        man["prime_evals_account"] = "arbos"
        man_path.write_text(json.dumps(man, indent=1))
    log(f"{len(urls)} cells on Prime Evals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
