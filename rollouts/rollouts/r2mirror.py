"""Trace chunks straight to the public corpus bucket (data.affine.io).

The trace store is canonical; D is a view derived from it by the fold on
the validator box. So the pod publishes traces the same way the corpus was
always published — immutable, content-named chunks behind an immutable
manifest, pointer written last:

    traces/chunks/<source>-<utc>-<sha12>.jsonl.gz   envelopes, never rewritten
    traces/manifests/<sha256>.json                  one per publish, immutable
    traces/manifest.json                            pointer = latest manifest

A chunk is marked mirrored only after its put succeeded; the manifest lists
mirrored chunks only, so a reader never sees a key that is not there yet.
A crash between put and manifest leaves the chunk unlisted until the next
pass republishes (idempotent: same content, same key).

Several pods (ROLLOUTS_SHARD=i/N) publish into the same prefix: the
manifest is the union of the pointer's chunks and the local store, so each
pod adds its own chunks and never removes another pod's.
"""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from rollouts.schema import utc_now_iso
from rollouts.store import TraceStore

log = logging.getLogger("rollouts.r2mirror")

TRACE_MANIFEST_SCHEMA = "affine-traces-v1"
IMMUTABLE = "public, max-age=31536000, immutable"
MANIFEST_MERGE_ROUNDS = 3


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class R2TraceMirror:
    def __init__(self, *, bucket: str, endpoint: str, access_key_id: str,
                 secret_access_key: str, prefix: str = "traces/"):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.s3 = boto3.client(
            "s3", endpoint_url=endpoint, region_name="auto",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4",
                          retries={"max_attempts": 5, "mode": "standard"}))

    # -- s3 -------------------------------------------------------------------
    def _exists(self, key: str, sha: str | None = None) -> bool:
        try:
            head = self.s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return sha is None or head.get("Metadata", {}).get("sha256") == sha

    def _put(self, key: str, body: bytes, content_type: str,
             cache_control: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                           ContentType=content_type, CacheControl=cache_control,
                           Metadata={"sha256": _sha(body)})

    def _get_bytes(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    # -- chunks ---------------------------------------------------------------
    def remote_key(self, store_key: str) -> str:
        """`chunks/<name>` in the local store -> `traces/chunks/<name>`."""
        return self.prefix + store_key

    def push_chunks(self, store: TraceStore) -> list[str]:
        """Put every unmirrored chunk present on disk. Marks each one
        mirrored (with the gzip sha256 + size readers verify) as soon as its
        put succeeded, so a failure mid-pass keeps the finished ones."""
        pending = [c for c in store.unmirrored_chunks()
                   if (store.root / c["key"]).exists()]
        done: list[str] = []
        for c in pending:
            body = (store.root / c["key"]).read_bytes()
            gz_sha = _sha(body)
            key = self.remote_key(c["key"])
            if not self._exists(key, gz_sha):
                self._put(key, body, "application/gzip", IMMUTABLE)
            store.mark_mirrored([c["key"]], details={
                c["key"]: {"gz_sha256": gz_sha, "bytes": len(body)}})
            done.append(c["key"])
            log.info("mirrored %s (%d rollouts, %d B)", key,
                     c["n_rollouts"], len(body))
        return done

    # -- manifest -------------------------------------------------------------
    def build_manifest(self, store: TraceStore, current: dict | None,
                       prev_sha: str | None) -> dict:
        """Union of what the pointer already lists and this pod's mirrored
        chunks. Several pods publish into the same prefix (fleet, 2026-09-02);
        each only knows its own store, so a manifest built from the local
        store alone would drop every other pod's chunks. Chunk keys are
        content-named and never rewritten, so a key seen anywhere stays."""
        by_key = {c["key"]: c for c in (current or {}).get("chunks", [])}
        for c in store.chunks():
            if not c.get("mirrored_at") or not c.get("gz_sha256"):
                continue
            by_key[self.remote_key(c["key"])] = {
                "key": self.remote_key(c["key"]),
                "sha256": c["gz_sha256"],
                "bytes": c["bytes"],
                "payload_sha256": c["sha256"],
                "n_rollouts": c["n_rollouts"],
                "sources": c["sources"],
                "created_at": c["created_at"],
            }
        chunks = sorted(by_key.values(), key=lambda c: c["key"])
        return {
            "schema": TRACE_MANIFEST_SCHEMA,
            "prev_manifest": prev_sha,
            "n_chunks": len(chunks),
            "n_rollouts": sum(c["n_rollouts"] for c in chunks),
            "chunks": chunks,
        }

    def publish_manifest(self, store: TraceStore) -> str | None:
        """Write the immutable manifest, then the pointer. Returns the last
        manifest sha written, or None when the pointer already lists every
        chunk this pod mirrored (nothing to publish).

        Two pods can race on the pointer (read-merge-write, last writer
        wins), so after writing we re-read it: if another pod overwrote it
        meanwhile, its copy lacks our chunks and we merge again. Bounded —
        a lost round only delays listing until the next cycle; the chunk
        objects themselves are already in the bucket."""
        pointer_key = self.prefix + "manifest.json"
        published: str | None = None
        for _attempt in range(MANIFEST_MERGE_ROUNDS):
            current_raw = self._get_bytes(pointer_key)
            current = json.loads(current_raw) if current_raw else None
            prev_sha = _sha(current_raw) if current_raw else None
            manifest = self.build_manifest(store, current, prev_sha)
            listed = {c["key"] for c in (current or {}).get("chunks", [])}
            if listed == {c["key"] for c in manifest["chunks"]}:
                return published
            manifest["published_at"] = utc_now_iso()
            # Same convention as the corpus manifest: the revision's name is
            # the sha256 of its exact bytes, so a reader verifies the pointer
            # by hashing what it fetched and finding it under manifests/.
            body = json.dumps(manifest, indent=1, sort_keys=True).encode()
            sha = _sha(body)
            self._put(posixpath.join(self.prefix, "manifests", f"{sha}.json"),
                      body, "application/json", IMMUTABLE)
            self._put(pointer_key, body, "application/json", "no-cache")
            log.info("published trace manifest %s (%d chunks, %d rollouts)",
                     sha[:12], manifest["n_chunks"], manifest["n_rollouts"])
            published = sha
            after = self._get_bytes(pointer_key)
            if after is not None and _sha(after) == sha:
                return published
            log.warning("trace manifest pointer changed under us; re-merging")
        return published

    def mirror(self, store: TraceStore) -> int:
        """One pass: push chunks, then republish the manifest if it changed.
        Raises on failure so the caller retries next cycle."""
        done = self.push_chunks(store)
        self.publish_manifest(store)
        return len(done)
