"""Publish a corpus revision (schema_version 3) to the [data_r2] bucket.

Same invariants as scripts/corpus_push.py had on Hippius:
  * objects are immutable — an existing key is reused only if its sha
    matches, never overwritten;
  * the manifest revision is written at corpus/manifests/{sha256}.json
    BEFORE the pointer corpus/manifest.json moves (evalsrv rejects a pointer
    without its immutable copy);
  * removal is logical (active=false in a new revision).

Fail-loud: any inconsistency raises; a partial publish must abort.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import posixpath
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .pack import PackResult

IMMUTABLE = "public, max-age=31536000, immutable"


class PublishError(RuntimeError):
    pass


class CorpusPublisher:
    def __init__(self, *, bucket: str, endpoint: str, access_key_id: str,
                 secret_access_key: str, manifest_key: str = "corpus/manifest.json",
                 key_prefix: str = "", log=print):
        # key_prefix (e.g. "staging/") is applied to every object operation
        # but never to the keys written INTO the manifest, so a reader rooted
        # at <base>/<prefix> sees the same relative layout as production.
        self.bucket = bucket
        self.key_prefix = key_prefix
        self.manifest_key = manifest_key
        self.manifests_prefix = posixpath.join(
            posixpath.dirname(manifest_key), "manifests")
        self.log = log
        self.s3 = boto3.client(
            "s3", endpoint_url=endpoint, region_name="auto",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4", read_timeout=300,
                          retries={"max_attempts": 5, "mode": "standard"}))

    # -- s3 -------------------------------------------------------------------
    def _k(self, key: str) -> str:
        return self.key_prefix + key

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()

    def put(self, key: str, body: bytes, content_type: str,
            cache_control: str = IMMUTABLE) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._k(key), Body=body,
                           ContentType=content_type, CacheControl=cache_control,
                           Metadata={"sha256": hashlib.sha256(body).hexdigest()})
        self.log(f"put {self._k(key)} ({len(body):,} bytes)")

    # -- manifest -------------------------------------------------------------
    def current_manifest(self) -> tuple[dict | None, str | None]:
        """(manifest, sha256 of its bytes) or (None, None)."""
        if not self.exists(self.manifest_key):
            return None, None
        raw = self.get(self.manifest_key)
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    def publish_manifest(self, manifest: dict) -> str:
        raw = json.dumps(manifest, indent=2, sort_keys=True).encode()
        mhash = hashlib.sha256(raw).hexdigest()
        self.put(f"{self.manifests_prefix}/{mhash}.json", raw, "application/json")
        self.put(self.manifest_key, raw, "application/json", "no-cache")
        return mhash

    # -- objects --------------------------------------------------------------
    def upload_gzip_jsonl(self, local, key: str) -> str:
        """Uncompressed local jsonl -> immutable .jsonl.gz; returns the
        uncompressed sha256 (the manifest convention evalsrv verifies)."""
        raw = local.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if self.exists(key):
            got = hashlib.sha256(gzip.decompress(self.get(key))).hexdigest()
            if got != sha:
                raise PublishError(f"{key} exists with a different sha")
            self.log(f"reuse {key} (sha verified)")
            return sha
        self.put(key, gzip.compress(raw, compresslevel=6), "application/gzip")
        return sha

    def upload_bytes(self, local, key: str, content_type: str) -> str:
        raw = local.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if self.exists(key):
            if hashlib.sha256(self.get(key)).hexdigest() != sha:
                raise PublishError(f"{key} exists with a different sha")
            self.log(f"reuse {key} (sha verified)")
            return sha
        self.put(key, raw, content_type)
        return sha

    # -- revision -------------------------------------------------------------
    def publish_revision(self, pack: PackResult, *, epoch: int, view_spec: str,
                         prev_manifest: str | None, prev_shards: list[dict],
                         extra: dict | None = None) -> tuple[dict, str]:
        """Upload the pack's chunks + index, then a schema-3 manifest that
        keeps `prev_shards` (already-published view chunks) active and adds
        the new ones. Returns (manifest, sha256)."""
        shards = list(prev_shards)
        for local, meta in zip(pack.chunk_paths, pack.chunk_meta):
            sha = self.upload_gzip_jsonl(local, meta["key"])
            if sha != meta["sha256"]:
                raise PublishError(f"chunk sha drift for {meta['key']}")
            shards.append(dict(meta))
        index_key = f"views/{view_spec}/index/turns_{epoch:04d}.parquet"
        index_sha = self.upload_bytes(pack.index_path, index_key,
                                      "application/vnd.apache.parquet")
        if index_sha != pack.index_sha256:
            raise PublishError("index sha drift")
        manifest = {
            "corpus_epoch": epoch,
            "schema_version": 3,
            "view_spec": view_spec,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "index": {"key": index_key, "sha256": index_sha,
                      "n_turns": pack.n_turns},
            "shards": shards,
            "prev_manifest": prev_manifest,
            **(extra or {}),
        }
        mhash = self.publish_manifest(manifest)
        self.log(f"manifest {mhash[:16]} epoch={epoch} schema=3 "
                 f"active={sum(1 for s in shards if s.get('active'))} "
                 f"n_turns={pack.n_turns:,}")
        return manifest, mhash
