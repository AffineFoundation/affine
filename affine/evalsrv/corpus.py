"""Corpus sync on the eval machine: manifest + shards from the public bucket.

The corpus D is append-only shards plus an immutable-manifest pointer:

  turns/manifest.json              mutable pointer (current manifest bytes)
  turns/manifests/{sha256}.json    immutable revision the pointer must match
  turns/shards/*.jsonl.gz          immutable shards; sha256 over UNCOMPRESSED

Sync is fail-closed and anonymous (the bucket is public-read; no Hippius
credentials ever reach the pod):

  1. GET the pointer, hash its bytes.
  2. Verify the same bytes exist at turns/manifests/{hash}.json — a rewritten
     pointer that was never published as an immutable revision is rejected.
  3. Download any missing active shards, verify each uncompressed sha256.
  4. Concatenate active shards into turns.jsonl.tmp → atomic rename.

On any failure the previously verified corpus keeps serving and the sync
reports stale. Refresh runs between duels, never mid-duel (server-side gate).

CLI (used by bootstrap.sh, fail-closed):  python -m evalsrv.corpus --sync
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import os
import posixpath
import sys
import time
from pathlib import Path

import httpx

from affine.config import load_config

log = logging.getLogger("evalsrv.corpus")

FETCH_TIMEOUT_S = 300.0


class CorpusVerificationError(Exception):
    """Manifest or shard failed verification; the old corpus stays active."""


class CorpusSync:
    def __init__(self, base_url: str, manifest_key: str, data_dir: Path):
        self.base = base_url.rstrip("/")
        self.manifest_key = manifest_key
        self.data_dir = data_dir
        self.turns_path = data_dir / "turns.jsonl"
        self.shards_dir = data_dir / "shards"
        self.manifest_path = data_dir / "current_manifest.json"
        self.manifest: dict | None = None
        self.manifest_sha256: str = ""
        self.synced_at: float = 0.0
        self.stale: bool = False
        self._load_local()

    # -- public ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.manifest is not None and self.turns_path.exists()

    def info(self) -> dict:
        """Manifest metadata for /health and verdict slice stamping."""
        return {
            "manifest_sha256": self.manifest_sha256,
            "corpus_epoch": (int(self.manifest.get("corpus_epoch", 0))
                             if self.manifest else 0),
            "synced_at": self.synced_at,
            "stale": self.stale,
            "ready": self.ready,
        }

    def refresh(self) -> bool:
        """Sync to the current manifest. Returns readiness (old corpus counts:
        a failed refresh leaves the last verified window active but stale)."""
        try:
            self._refresh()
            self.stale = False
        except (httpx.HTTPError, CorpusVerificationError,
                json.JSONDecodeError, OSError) as e:
            self.stale = True
            log.warning("corpus refresh failed (%s); serving last verified "
                        "corpus (ready=%s)", e, self.ready)
        return self.ready

    # -- internals ---------------------------------------------------------------
    def _load_local(self) -> None:
        """Adopt a previously verified corpus after a restart."""
        if self.manifest_path.exists() and self.turns_path.exists():
            raw = self.manifest_path.read_bytes()
            self.manifest = json.loads(raw)
            self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
            self.synced_at = self.manifest_path.stat().st_mtime
            log.info("adopted local corpus: epoch=%s manifest=%s",
                     self.manifest.get("corpus_epoch"), self.manifest_sha256[:12])

    def _get(self, key: str) -> bytes:
        r = httpx.get(f"{self.base}/{key}", timeout=FETCH_TIMEOUT_S,
                      follow_redirects=True)
        r.raise_for_status()
        return r.content

    def _immutable_manifest_key(self, mhash: str) -> str:
        return posixpath.join(posixpath.dirname(self.manifest_key),
                              "manifests", f"{mhash}.json")

    def _refresh(self) -> None:
        raw = self._get(self.manifest_key)
        mhash = hashlib.sha256(raw).hexdigest()
        if mhash == self.manifest_sha256 and self.turns_path.exists():
            self.synced_at = time.time()
            return

        # Tamper-evidence: the pointer must match its published immutable copy.
        frozen = self._get(self._immutable_manifest_key(mhash))
        if hashlib.sha256(frozen).hexdigest() != mhash:
            raise CorpusVerificationError(
                f"manifest pointer {mhash[:12]} does not match its immutable copy")

        manifest = json.loads(raw)
        active = [s for s in manifest.get("shards", []) if s.get("active")]
        if not active:
            raise CorpusVerificationError("manifest has no active shards")

        self.shards_dir.mkdir(parents=True, exist_ok=True)
        for shard in active:
            self._ensure_shard(shard)

        # Rebuild turns.jsonl in manifest order, atomically.
        tmp = self.turns_path.with_suffix(".jsonl.tmp")
        with open(tmp, "wb") as out:
            for shard in active:
                with gzip.open(self._shard_path(shard["key"]), "rb") as f:
                    while chunk := f.read(1 << 20):
                        out.write(chunk)
        tmp.replace(self.turns_path)

        mtmp = self.manifest_path.with_suffix(".json.tmp")
        mtmp.write_bytes(raw)
        mtmp.replace(self.manifest_path)
        self.manifest = manifest
        self.manifest_sha256 = mhash
        self.synced_at = time.time()
        log.info("corpus synced: epoch=%s manifest=%s shards=%d (%d bytes)",
                 manifest.get("corpus_epoch"), mhash[:12], len(active),
                 self.turns_path.stat().st_size)

    def _shard_path(self, key: str) -> Path:
        return self.shards_dir / posixpath.basename(key)

    def _ensure_shard(self, shard: dict) -> None:
        """Download + verify one shard. A `.sha` sidecar marks local files we
        already verified, so refreshes don't re-hash hundreds of MB."""
        want = str(shard["sha256"]).lower()
        path = self._shard_path(shard["key"])
        sidecar = path.with_suffix(path.suffix + ".sha")
        if path.exists() and sidecar.exists() and sidecar.read_text().strip() == want:
            return
        log.info("fetching shard %s", shard["key"])
        blob = self._get(shard["key"])
        h = hashlib.sha256()
        with gzip.GzipFile(fileobj=io.BytesIO(blob)) as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        got = h.hexdigest()
        if got != want:
            raise CorpusVerificationError(
                f"shard {shard['key']} sha mismatch: {got} != {want}")
        path.write_bytes(blob)
        sidecar.write_text(want)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync the turn corpus (fail-closed)")
    ap.add_argument("--sync", action="store_true", required=True)
    args = ap.parse_args()
    assert args.sync
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    data_dir = Path(os.environ.get("AFFINE_DATA_DIR", "/root/affine_data"))
    sync = CorpusSync(cfg.dataset.corpus_base_url, cfg.dataset.manifest_key,
                      data_dir)
    if not sync.refresh():
        sys.exit("[corpus] FATAL: no verified corpus could be established "
                 "(fail-closed policy)")
    if sync.stale:
        log.warning("refresh failed but a previously verified corpus is present")
    print(f"[corpus] ready: {json.dumps(sync.info())}")


if __name__ == "__main__":
    main()
