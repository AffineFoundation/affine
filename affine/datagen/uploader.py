"""Upload accumulated turn shards to a HF dataset repo, sha-pinned.

Follows the spirit of the production corpus manifest (affine.toml [dataset]):
immutable shard files named with their content sha256, plus a manifest.json
listing every shard with its hash. This service only accumulates a candidate
dataset — retargeting the production contract is a separate operator step.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError

log = logging.getLogger("datagen.uploader")

MANIFEST_NAME = "manifest.json"
SCHEMA = "affine-turns-v1"


def shard_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def shard_name(sha: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"turns-{stamp}-{sha[:12]}.jsonl"


class TurnUploader:
    def __init__(self, repo_id: str, private: bool):
        self.repo_id = repo_id
        self.private = private
        self.api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        self._repo_ready = False

    def _ensure_repo(self) -> None:
        if self._repo_ready:
            return
        self.api.create_repo(self.repo_id, repo_type="dataset",
                             private=self.private, exist_ok=True)
        self._repo_ready = True

    def _load_manifest(self) -> dict:
        try:
            path = self.api.hf_hub_download(
                self.repo_id, MANIFEST_NAME, repo_type="dataset")
            return json.loads(Path(path).read_text())
        except EntryNotFoundError:
            return {"schema": SCHEMA, "shards": []}

    def upload_shard(self, shard: Path) -> dict:
        """Upload one shard jsonl + refreshed manifest. Raises on failure so
        the caller keeps the shard queued for retry."""
        self._ensure_repo()
        sha = shard_sha256(shard)
        with open(shard, "rb") as f:
            n_turns = sum(1 for line in f if line.strip())
        key = f"shards/{shard.name}"
        self.api.upload_file(
            path_or_fileobj=str(shard), path_in_repo=key,
            repo_id=self.repo_id, repo_type="dataset",
            commit_message=f"add shard {shard.name} ({n_turns} turns)")

        manifest = self._load_manifest()
        entry = {
            "key": key,
            "sha256": sha,
            "n_turns": n_turns,
            "uploaded_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        }
        manifest["shards"] = [s for s in manifest.get("shards", [])
                              if s.get("key") != key] + [entry]
        manifest["schema"] = SCHEMA
        manifest["updated_at"] = entry["uploaded_at"]
        self.api.upload_file(
            path_or_fileobj=json.dumps(manifest, indent=2,
                                       sort_keys=True).encode(),
            path_in_repo=MANIFEST_NAME,
            repo_id=self.repo_id, repo_type="dataset",
            commit_message=f"manifest: +{shard.name}")
        log.info("uploaded %s (%d turns, sha %s) to %s",
                 key, n_turns, sha[:12], self.repo_id)
        return entry
