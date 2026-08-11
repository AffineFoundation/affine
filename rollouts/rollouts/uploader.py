"""HF uploads: turn shards to the staging dataset, trace chunks to the
rollout mirror.

Turn shards reuse datagen.uploader.TurnUploader unchanged (sha-manifested,
same staging repo the fold consumes). Trace chunks are mirrored file-per-
chunk plus the local store manifest, so the pod disk is never the only copy
of the system of record.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

from datagen.uploader import TurnUploader, shard_sha256

from rollouts.store import TraceStore

__all__ = ["TurnUploader", "TraceMirror", "shard_sha256"]

log = logging.getLogger("rollouts.uploader")


class TraceMirror:
    def __init__(self, repo_id: str, private: bool = True):
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

    def mirror(self, store: TraceStore) -> int:
        """Upload every unmirrored chunk + the refreshed manifest. Returns
        the number of chunks mirrored; raises on failure so the caller
        retries next cycle (chunks stay marked unmirrored)."""
        pending = store.unmirrored_chunks()
        if not pending:
            return 0
        self._ensure_repo()
        done: list[str] = []
        for chunk in pending:
            path = store.root / chunk["key"]
            if not path.exists():
                log.warning("skipping missing chunk %s", chunk["key"])
                continue
            self.api.upload_file(
                path_or_fileobj=str(path), path_in_repo=chunk["key"],
                repo_id=self.repo_id, repo_type="dataset",
                commit_message=(f"add {Path(chunk['key']).name} "
                                f"({chunk['n_rollouts']} rollouts)"))
            done.append(chunk["key"])
        if done:
            store.mark_mirrored(done)
            self.api.upload_file(
                path_or_fileobj=str(store.manifest_path),
                path_in_repo="manifest.json",
                repo_id=self.repo_id, repo_type="dataset",
                commit_message=f"manifest: +{len(done)} chunk(s)")
            log.info("mirrored %d trace chunk(s) to %s", len(done),
                     self.repo_id)
        return len(done)
