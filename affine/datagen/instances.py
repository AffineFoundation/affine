"""Instance pool: multi-source loading, benchmark exclusion, batch selection.

The 25-instance advisory bench panel (evalsrv/data/swe_rebench_lite_ids.json)
must stay disjoint from the training corpus, so both the pinned instance ids
AND every instance from the same repos are hard-excluded — across ALL
configured source datasets.

Sources must be swebench-harness-format compatible: instance rows need the
eval fields (patch/test_patch/FAIL_TO_PASS/...) plus a pullable docker image
(docker_image/image_name). Rows are tagged with `_source_dataset` /
`_source_split` (stripped before materialization) so the eval phase can run
per source.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import threading
import time
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

log = logging.getLogger("datagen.instances")

# datagen and evalsrv ship side by side (affine/ in the repo, site-packages
# when installed), so the pinned bench panel resolves relative to this file.
PIN_PATH = (Path(__file__).resolve().parent.parent
            / "evalsrv" / "data" / "swe_rebench_lite_ids.json")


def instance_repo(instance_id: str) -> str:
    """owner__name-123 -> owner/name (swebench instance id convention)."""
    return instance_id.rsplit("-", 1)[0].replace("__", "/", 1)


def pinned_exclusions() -> tuple[set[str], set[str]]:
    pin = json.loads(PIN_PATH.read_text())
    ids = set(pin["instance_ids"])
    return ids, {instance_repo(i) for i in ids}


def load_pool(sources: tuple[tuple[str, str], ...]) -> list[dict]:
    """Load candidates from every source, dropping bench-pinned ids/repos,
    image-less instances, and cross-source duplicates (first source wins)."""
    excl_ids, excl_repos = pinned_exclusions()
    pool: list[dict] = []
    seen: set[str] = set()
    for dataset, split in sources:
        ds = load_dataset(dataset, split=split)
        n_kept = n_no_image = n_excluded = n_dup = 0
        for row in ds:
            if not (row.get("docker_image") or row.get("image_name")):
                n_no_image += 1
                continue
            iid = row["instance_id"]
            if (iid in excl_ids or row.get("repo") in excl_repos
                    or instance_repo(iid) in excl_repos):
                n_excluded += 1
                continue
            if iid in seen:
                n_dup += 1
                continue
            seen.add(iid)
            row["_source_dataset"] = dataset
            row["_source_split"] = split
            pool.append(row)
            n_kept += 1
        log.info("pool: +%d from %s[%s] (%d bench-excluded, %d without "
                 "docker image, %d cross-source dups)", n_kept, dataset,
                 split, n_excluded, n_no_image, n_dup)
    log.info("pool: %d candidates total from %d source(s)",
             len(pool), len(sources))
    return pool


def select_batch(pool: list[dict], done: set[str], n: int,
                 seed: int) -> list[dict]:
    """Next n unprocessed instances in a seed-deterministic shuffle order, so
    restarts resume exactly where the previous run left off."""
    order = list(pool)
    random.Random(seed).shuffle(order)
    return [r for r in order if r["instance_id"] not in done][:n]


def materialize_subset(rows: list[dict], out_dir: Path) -> Path:
    """Write rows as a local HF dataset dir mini-extra can consume
    (--subset <dir> --split test), same shape as swerunner._prepare_subset.
    Internal `_source_*` tags are stripped."""
    clean = [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in rows]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    DatasetDict({"test": Dataset.from_list(clean)}).save_to_disk(str(out_dir))
    return out_dir


class ProcessedState:
    """Append-only JSONL of processed instance ids + outcomes. Restart-safe:
    an instance recorded here is never attempted again. Thread-safe: the
    pipelined loop marks errors from the generation thread and outcomes from
    the eval thread."""

    def __init__(self, path: Path):
        self.path = path
        self.outcomes: dict[str, str] = {}
        self._lock = threading.Lock()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    self.outcomes[rec["instance_id"]] = rec.get("outcome", "")
            log.info("state: %d instances already processed", len(self.outcomes))

    @property
    def done(self) -> set[str]:
        with self._lock:
            return set(self.outcomes)

    def mark(self, instance_id: str, outcome: str, **extra) -> None:
        rec = {"instance_id": instance_id, "outcome": outcome,
               "ts": time.time(), **extra}
        with self._lock:
            self.outcomes[instance_id] = outcome
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
