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

from huggingface_hub import CommitOperationAdd, HfApi
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

    def upload_shards(self, shards: list[Path]) -> list[dict]:
        """Upload shard jsonls + refreshed manifest in ONE commit.

        The Hub rate-limits commits per account (~128/h). Two commits per
        shard (file, then manifest) across a backlog of outbox shards plus
        the trace mirror burned that budget and stalled staging for hours.
        One commit per flush keeps the manifest consistent with the shards
        and costs the same whether one or a hundred shards are queued.
        Raises on failure so the caller keeps every shard queued for retry.
        """
        if not shards:
            return []
        self._ensure_repo()
        manifest = self._load_manifest()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entries: list[dict] = []
        ops: list[CommitOperationAdd] = []
        for shard in shards:
            with open(shard, "rb") as f:
                n_turns = sum(1 for line in f if line.strip())
            key = f"shards/{shard.name}"
            entries.append({"key": key, "sha256": shard_sha256(shard),
                            "n_turns": n_turns, "uploaded_at": now})
            ops.append(CommitOperationAdd(path_in_repo=key,
                                          path_or_fileobj=str(shard)))
        keys = {e["key"] for e in entries}
        manifest["shards"] = [s for s in manifest.get("shards", [])
                              if s.get("key") not in keys] + entries
        manifest["schema"] = SCHEMA
        manifest["updated_at"] = now
        ops.append(CommitOperationAdd(
            path_in_repo=MANIFEST_NAME,
            path_or_fileobj=json.dumps(manifest, indent=2,
                                       sort_keys=True).encode()))
        total = sum(e["n_turns"] for e in entries)
        self.api.create_commit(
            repo_id=self.repo_id, repo_type="dataset", operations=ops,
            commit_message=f"add {len(entries)} shard(s) ({total} turns)")
        for e in entries:
            log.info("uploaded %s (%d turns, sha %s) to %s",
                     e["key"], e["n_turns"], e["sha256"][:12], self.repo_id)
        return entries

    def upload_shard(self, shard: Path) -> dict:
        """Upload one shard jsonl + refreshed manifest (single commit)."""
        return self.upload_shards([shard])[0]
