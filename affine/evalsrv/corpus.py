"""Corpus sync on the eval machine: manifest + shards from the public bucket.

The corpus D is append-only objects plus an immutable-manifest pointer:

  turns/manifest.json              mutable pointer (current manifest bytes)
  turns/manifests/{sha256}.json    immutable revision the pointer must match
  turns/shards/*.jsonl.gz          schema v1: per-turn JSONL (legacy)
  turns/chunks/*.jsonl.gz          schema v2: trajectory records
  turns/index/turns_*.parquet      schema v2: turn index for sampling

Sync is fail-closed and anonymous (the bucket is public-read; no Hippius
credentials ever reach the pod):

  1. GET the pointer, hash its bytes.
  2. Verify the same bytes exist at turns/manifests/{hash}.json — a rewritten
     pointer that was never published as an immutable revision is rejected.
  3. Download any missing active objects, verify each sha256.
  4. schema v1: concatenate active shards into turns.jsonl
     schema v2: keep parquet index + chunks; duel samples the index.

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
import pyarrow.parquet as pq

from affine.config import load_config
from affine.corpus.materialize import materialize_turn

log = logging.getLogger("evalsrv.corpus")

FETCH_TIMEOUT_S = 300.0


class CorpusVerificationError(Exception):
    """Manifest or shard failed verification; the old corpus stays active."""


def schema_version(manifest: dict | None) -> int:
    if not manifest:
        return 1
    return int(manifest.get("schema_version") or 1)


class CorpusSync:
    def __init__(self, base_url: str, manifest_key: str, data_dir: Path,
                 lazy_chunks: bool = False):
        # lazy_chunks: refresh verifies manifest + index only; chunk objects
        # are fetched (and sha-verified) on first access. Used by the dash
        # dataset browser, where most chunks are never viewed. Eval pods keep
        # the default eager, fail-closed sync.
        self.base = base_url.rstrip("/")
        self.manifest_key = manifest_key
        self.data_dir = data_dir
        self.lazy_chunks = lazy_chunks
        self.turns_path = data_dir / "turns.jsonl"
        self.index_path = data_dir / "turns_index.parquet"
        self.shards_dir = data_dir / "shards"
        self.chunks_dir = data_dir / "chunks"
        self.manifest_path = data_dir / "current_manifest.json"
        self.manifest: dict | None = None
        self.manifest_sha256: str = ""
        self.synced_at: float = 0.0
        self.stale: bool = False
        self._index_rows: list[dict] | None = None
        self._chunk_line_cache: dict[str, list[dict]] = {}
        self._load_local()

    # -- public ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        if self.manifest is None:
            return False
        if schema_version(self.manifest) >= 2:
            return self.index_path.exists()
        return self.turns_path.exists()

    @property
    def schema_version(self) -> int:
        return schema_version(self.manifest)

    def info(self) -> dict:
        """Manifest metadata for /health and verdict slice stamping."""
        return {
            "manifest_sha256": self.manifest_sha256,
            "corpus_epoch": (int(self.manifest.get("corpus_epoch", 0))
                             if self.manifest else 0),
            "schema_version": self.schema_version,
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
                json.JSONDecodeError, OSError, ValueError) as e:
            self.stale = True
            log.warning("corpus refresh failed (%s); serving last verified "
                        "corpus (ready=%s)", e, self.ready)
        return self.ready

    def load_index_rows(self) -> list[dict]:
        """Return Parquet index rows as dicts (schema v2). Cached in memory."""
        if self.schema_version < 2:
            raise RuntimeError("load_index_rows requires schema_version>=2")
        if self._index_rows is not None:
            return self._index_rows
        table = pq.read_table(self.index_path)
        cols = {name: table.column(name).to_pylist()
                for name in table.column_names}
        n = table.num_rows
        rows = [{k: cols[k][i] for k in cols} for i in range(n)]
        self._index_rows = rows
        return rows

    def load_v1_turns(self) -> list[dict]:
        if not self.turns_path.exists():
            raise RuntimeError("v1 turns.jsonl missing")
        with open(self.turns_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def materialize_turns(self, index_rows: list[dict]) -> list[dict]:
        """Expand index rows into scorable turn dicts (prefix + reference)."""
        out: list[dict] = []
        for row in index_rows:
            traj = self._traj_at(row["chunk_key"], int(row["traj_line"]))
            meta = next(
                t for t in traj["turns"]
                if int(t["turn_idx"]) == int(row["turn_idx"]))
            out.append(materialize_turn(traj, meta))
        return out

    # -- internals ---------------------------------------------------------------
    def _load_local(self) -> None:
        """Adopt a previously verified corpus after a restart."""
        if not self.manifest_path.exists():
            return
        raw = self.manifest_path.read_bytes()
        self.manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.synced_at = self.manifest_path.stat().st_mtime
        if schema_version(self.manifest) >= 2:
            if self.index_path.exists():
                log.info("adopted local corpus v2: epoch=%s manifest=%s",
                         self.manifest.get("corpus_epoch"),
                         self.manifest_sha256[:12])
        elif self.turns_path.exists():
            log.info("adopted local corpus v1: epoch=%s manifest=%s",
                     self.manifest.get("corpus_epoch"),
                     self.manifest_sha256[:12])

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
        sv = schema_version(json.loads(raw))
        ready_marker = (self.index_path if sv >= 2 else self.turns_path)
        if mhash == self.manifest_sha256 and ready_marker.exists():
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

        if schema_version(manifest) >= 2:
            self._refresh_v2(raw, mhash, manifest, active)
        else:
            self._refresh_v1(raw, mhash, manifest, active)

    def _refresh_v1(self, raw: bytes, mhash: str, manifest: dict,
                    active: list[dict]) -> None:
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        for shard in active:
            self._ensure_gzip_object(shard, self.shards_dir)

        tmp = self.turns_path.with_suffix(".jsonl.tmp")
        with open(tmp, "wb") as out:
            for shard in active:
                with gzip.open(self.shards_dir / posixpath.basename(shard["key"]),
                               "rb") as f:
                    while chunk := f.read(1 << 20):
                        out.write(chunk)
        tmp.replace(self.turns_path)
        self._commit_manifest(raw, mhash, manifest)
        self._index_rows = None
        self._chunk_line_cache.clear()
        log.info("corpus synced v1: epoch=%s manifest=%s shards=%d (%d bytes)",
                 manifest.get("corpus_epoch"), mhash[:12], len(active),
                 self.turns_path.stat().st_size)

    def _refresh_v2(self, raw: bytes, mhash: str, manifest: dict,
                    active: list[dict]) -> None:
        index = manifest.get("index")
        if not index or not index.get("key") or not index.get("sha256"):
            raise CorpusVerificationError("v2 manifest missing index")
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        if not self.lazy_chunks:
            for shard in active:
                if (shard.get("format") or "") != "traj_v1":
                    # Skip non-chunk actives (should not happen post-migrate).
                    log.warning("skipping non-traj active shard %s",
                                shard["key"])
                    continue
                self._ensure_gzip_object(shard, self.chunks_dir)

        # Index: sha over raw parquet bytes (not gzip).
        want = str(index["sha256"]).lower()
        idx_tmp = self.index_path.with_suffix(".parquet.tmp")
        sidecar = self.index_path.with_suffix(self.index_path.suffix + ".sha")
        if not (self.index_path.exists() and sidecar.exists()
                and sidecar.read_text().strip() == want):
            log.info("fetching index %s", index["key"])
            blob = self._get(index["key"])
            got = hashlib.sha256(blob).hexdigest()
            if got != want:
                raise CorpusVerificationError(
                    f"index sha mismatch: {got} != {want}")
            idx_tmp.write_bytes(blob)
            idx_tmp.replace(self.index_path)
            sidecar.write_text(want)

        if int(index.get("n_turns") or 0) != pq.read_metadata(
                self.index_path).num_rows:
            # Soft check — metadata row count vs claim.
            n = pq.read_table(self.index_path, columns=["turn_id"]).num_rows
            if n != int(index["n_turns"]):
                raise CorpusVerificationError(
                    f"index n_turns claim {index['n_turns']} != {n}")

        self._commit_manifest(raw, mhash, manifest)
        self._index_rows = None
        self._chunk_line_cache.clear()
        log.info("corpus synced v2: epoch=%s manifest=%s chunks=%d index=%s",
                 manifest.get("corpus_epoch"), mhash[:12],
                 sum(1 for s in active if s.get("format") == "traj_v1"),
                 want[:12])

    def _commit_manifest(self, raw: bytes, mhash: str, manifest: dict) -> None:
        mtmp = self.manifest_path.with_suffix(".json.tmp")
        mtmp.write_bytes(raw)
        mtmp.replace(self.manifest_path)
        self.manifest = manifest
        self.manifest_sha256 = mhash
        self.synced_at = time.time()

    def _ensure_gzip_object(self, shard: dict, dest_dir: Path) -> None:
        """Download + verify one gzipped jsonl. Sidecar marks verified sha."""
        want = str(shard["sha256"]).lower()
        path = dest_dir / posixpath.basename(shard["key"])
        sidecar = path.with_suffix(path.suffix + ".sha")
        if path.exists() and sidecar.exists() and sidecar.read_text().strip() == want:
            return
        log.info("fetching %s", shard["key"])
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

    def ensure_chunk(self, chunk_key: str) -> None:
        """Fetch + sha-verify one chunk on demand (lazy_chunks mode)."""
        path = self.chunks_dir / posixpath.basename(chunk_key)
        sidecar = path.with_suffix(path.suffix + ".sha")
        if path.exists() and sidecar.exists():
            return
        shard = next((s for s in (self.manifest or {}).get("shards", [])
                      if s.get("key") == chunk_key), None)
        if shard is None:
            raise CorpusVerificationError(
                f"chunk {chunk_key} not in the active manifest")
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_gzip_object(shard, self.chunks_dir)

    # In lazy mode only a handful of chunks are viewed at a time; cap the
    # parsed-line cache so the dash process does not grow with corpus size.
    _LAZY_CACHE_CHUNKS = 4

    def _traj_at(self, chunk_key: str, traj_line: int) -> dict:
        if chunk_key not in self._chunk_line_cache:
            if self.lazy_chunks:
                self.ensure_chunk(chunk_key)
                while len(self._chunk_line_cache) >= self._LAZY_CACHE_CHUNKS:
                    self._chunk_line_cache.pop(
                        next(iter(self._chunk_line_cache)))
            path = self.chunks_dir / posixpath.basename(chunk_key)
            with gzip.open(path, "rt", encoding="utf-8") as f:
                self._chunk_line_cache[chunk_key] = [
                    json.loads(line) for line in f if line.strip()]
        lines = self._chunk_line_cache[chunk_key]
        if traj_line < 0 or traj_line >= len(lines):
            raise CorpusVerificationError(
                f"traj_line {traj_line} OOB in {chunk_key}")
        return lines[traj_line]


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
