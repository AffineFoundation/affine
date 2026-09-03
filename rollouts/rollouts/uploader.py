"""HF cold copy of the trace chunks (secondary; the R2 mirror is canonical).

Kept behind ROLLOUTS_HF_TRACE_MIRROR for 30 days after the trace-first
cutover (2026-09-02) so a bad R2 day never leaves the pod disk as the only
copy; drop the module once nothing reads unconst/affine-rollout-traces.
Turn-shard staging on HF is retired: D is derived from the published traces
by ops/corpus_build.py on the validator box.
"""

from __future__ import annotations

import logging
import os

from huggingface_hub import CommitOperationAdd, HfApi

from rollouts.store import TraceStore

__all__ = ["TraceMirror", "MAX_FILES_PER_COMMIT"]

log = logging.getLogger("rollouts.uploader")

# Hub budgets (observed 2026-09-02): ~128 commits/hour AND 1000 API calls
# per 5-minute window, where LFS verify costs one call per file. One file
# per commit starves the former; a 600-file backlog in one commit starves
# the latter. 64 files per commit fits both with room for the other loops.
MAX_FILES_PER_COMMIT = 64
FIELD = "hf_mirrored_at"


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
        """Upload every chunk not yet on HF plus the refreshed manifest in
        batched commits. Returns the number of chunks mirrored; raises on
        failure so the caller retries next cycle."""
        pending = [c for c in store.unmirrored_chunks(FIELD)
                   if (store.root / c["key"]).exists()]
        if not pending:
            return 0
        self._ensure_repo()
        total = 0
        for i in range(0, len(pending), MAX_FILES_PER_COMMIT):
            batch = pending[i:i + MAX_FILES_PER_COMMIT]
            keys = [c["key"] for c in batch]
            ops = [CommitOperationAdd(path_in_repo=k,
                                      path_or_fileobj=str(store.root / k))
                   for k in keys]
            # Mark first so the manifest we commit already lists these
            # chunks as mirrored; on commit failure, unmark so they retry.
            store.mark_mirrored(keys, field=FIELD)
            ops.append(CommitOperationAdd(
                path_in_repo="manifest.json",
                path_or_fileobj=str(store.manifest_path)))
            try:
                self.api.create_commit(
                    repo_id=self.repo_id, repo_type="dataset", operations=ops,
                    commit_message=f"add {len(keys)} chunk(s)")
            except Exception:
                store.mark_mirrored(keys, mirrored=False, field=FIELD)
                raise
            total += len(keys)
            log.info("HF cold copy: %d trace chunk(s) to %s (%d/%d)", len(keys),
                     self.repo_id, total, len(pending))
        return total
