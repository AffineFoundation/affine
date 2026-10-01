"""Generated-task storage: package data dir first, then data.affine.io."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Iterator

import httpx
from filelock import FileLock

REMOTE_BASE = os.environ.get("AFFINE_GEN_REMOTE", "https://data.affine.io/envs")
CACHE_DIR = Path(os.environ.get("AFFINE_GEN_CACHE", Path.home() / ".cache" / "affine_gen"))
GEN_RE = re.compile(r"\[GEN:[^\]]+\]")


def gen_uid(prefix: str, epoch: int, payload: str) -> str:
    """`<prefix>-<12 hex of payload>[GEN:e<epoch>]` — the fold's decontamination
    rule (`require_gen_marker`) admits a task only with the marker; the
    epoch inside it keeps pools of different generations apart."""
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}[GEN:e{epoch}]"


def has_gen_marker(uid: str) -> bool:
    return bool(GEN_RE.search(uid or ""))


class GenTaskStore:
    """Read/write `tasks.jsonl.gz` for one source and epoch.

    Layout: `<package>/data/e<epoch>/tasks.jsonl.gz` (committed when small)
    or `<REMOTE_BASE>/<source>/e<epoch>/tasks.jsonl.gz` (published from the
    datagen box; cached under ~/.cache/affine_gen). Side files (docqa's
    document bundles) live next to it as `<name>.json.gz`."""

    def __init__(self, source: str, package_dir: Path, epoch: int) -> None:
        self.source, self.package_dir, self.epoch = source, Path(package_dir), int(epoch)

    @property
    def local_dir(self) -> Path:
        return self.package_dir / "data" / f"e{self.epoch}"

    def _remote(self, name: str) -> str:
        return f"{REMOTE_BASE}/{self.source}/e{self.epoch}/{name}"

    def path(self, name: str) -> Path:
        """Local path of a side file, fetching from data.affine.io if the
        package does not carry it."""
        local = self.local_dir / name
        if local.exists():
            return local
        cached = CACHE_DIR / self.source / f"e{self.epoch}" / name
        cached.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cached) + ".lock"):
            if not cached.exists():
                resp = httpx.get(self._remote(name), timeout=600, follow_redirects=True)
                resp.raise_for_status()
                tmp = cached.with_suffix(cached.suffix + ".tmp")
                tmp.write_bytes(resp.content)
                tmp.rename(cached)
        return cached

    def tasks(self) -> Iterator[dict]:
        with gzip.open(self.path("tasks.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def read_json(self, name: str) -> dict:
        with gzip.open(self.path(name), "rt", encoding="utf-8") as f:
            return json.load(f)

    # -- writing (generators run on the datagen box) ---------------------------
    # Durability rule (2026-09-22 17:54, a SIGTERM lost 27 + 29 kept items that
    # were only written at exit): every kept item is APPENDED to the epoch
    # file the moment it passes verification. gzip members concatenate, so
    # `tasks.jsonl.gz` opened in "at" mode stays a valid file that `tasks()`
    # reads whole; the exit path only writes the ledger and a summary.
    _append_lock = threading.Lock()

    def existing_uids(self) -> set[str]:
        path = self.local_dir / "tasks.jsonl.gz"
        if not path.exists():
            return set()
        return {r["uid"] for r in self.tasks() if "uid" in r}

    def append_task(self, rec: dict) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        with self._append_lock, gzip.open(self.local_dir / "tasks.jsonl.gz", "at", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def append_reject(self, rec: dict) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        with self._append_lock, gzip.open(self.local_dir / "rejects.jsonl.gz", "at", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_tasks(self, rows: list[dict]) -> Path:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        out = self.local_dir / "tasks.jsonl.gz"
        with gzip.open(out, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return out

    def write_json(self, name: str, obj: dict) -> Path:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        out = self.local_dir / name
        with gzip.open(out, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        return out
