"""Cloudflare R2 behind the ``ObjectStore`` and ``Presigner`` seams (S3 API).

The rules this file holds:

* ``sha256`` reads the stored bytes back and hashes them. It never trusts an
  ETag, object metadata or a value the uploader sent.
* ``copy`` uses multipart UploadPartCopy above ``part_size``. One CopyObject
  call on a ~4 GB shard timed out in v1.
* No write returns before the store confirms it. Upload bodies are signed.
  ``put`` compares the MD5 the store answers with; ``put_file`` and ``copy``
  compare the stored size with the intended size. A mismatch removes the
  object and raises ``InfraError``.
* A missing key is ``KeyError``. Every other failure is ``InfraError``, so the
  controller records ``no_decision`` and promotes nothing.
* Credentials come from the environment at run time. They are never logged and
  never put in an error message. A presigned URL is a secret: do not log it.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Mapping

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

from .errors import ContractError, InfraError, V2Error
from .params import require_env

MIB = 1024 * 1024
MAX_PRESIGN_S = 7 * 24 * 3600  # the longest life SigV4 allows for a presigned URL
PUT_HEADERS = {"Content-Type": "application/octet-stream"}  # signed: an upload must send exactly these
_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class R2Store:
    """One R2 bucket, optionally under a key prefix (for example an isolated smoke-test namespace)."""

    def __init__(
        self, *, endpoint: str, bucket: str, access_key_id: str, secret_access_key: str,
        prefix: str = "", part_size: int = 64 * MIB, attempts: int = 5,
    ) -> None:
        if part_size < 5 * MIB:
            raise ContractError("part_size must be at least 5 MiB (S3 multipart minimum)")
        self._bucket = bucket
        self._part_size = part_size
        self._prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
        # R2 needs every part but the last to have the same size; a fixed chunk size gives that.
        self._transfer = TransferConfig(multipart_threshold=part_size, multipart_chunksize=part_size, max_concurrency=8)
        self._s3 = boto3.client(
            "s3", endpoint_url=endpoint, region_name="auto",
            aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,
            config=Config(
                signature_version="s3v4",
                # Signed payloads: the store rejects an upload body that changed on the way.
                s3={"addressing_style": "path", "payload_signing_enabled": True},
                retries={"max_attempts": attempts, "mode": "standard"},
                connect_timeout=10,
                read_timeout=120,
                max_pool_connections=16,
                # R2 does not need the optional CRC trailers; integrity is checked here instead.
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
        # Fail at start-up on a wrong bucket or bad credentials. A HEAD on a missing
        # bucket looks the same as a missing key, so this must be known before any call.
        with self._guard("open bucket", bucket):
            self._s3.list_objects_v2(Bucket=bucket, Prefix=self._prefix, MaxKeys=1)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **options: Any) -> "R2Store":
        """Build from ``SN120_R2_*`` variables. A missing one is ``ConfigError``; only the prefix is optional."""
        environ = os.environ if environ is None else environ
        return cls(
            endpoint=require_env("SN120_R2_ENDPOINT", environ),
            bucket=require_env("SN120_R2_BUCKET", environ),
            access_key_id=require_env("SN120_R2_ACCESS_KEY_ID", environ),
            secret_access_key=require_env("SN120_R2_SECRET_ACCESS_KEY", environ),
            prefix=environ.get("SN120_R2_PREFIX", ""),
            **options,
        )

    def _k(self, key: str) -> str:
        if not isinstance(key, str) or not key or key.startswith("/"):
            raise ContractError(f"bad object key {key!r}")
        return self._prefix + key

    @contextlib.contextmanager
    def _guard(self, action: str, key: str) -> Iterator[None]:
        """Map a store failure to ``KeyError`` (missing key) or ``InfraError`` (everything else)."""
        try:
            yield
        except V2Error:
            raise
        except ClientError as exc:
            error = exc.response.get("Error", {})
            if str(error.get("Code")) in _MISSING_CODES:
                raise KeyError(key) from None
            raise InfraError(f"r2 {action} {key!r} failed: {error.get('Code')}") from exc
        except Exception as exc:  # network, timeout, short read, local disk
            raise InfraError(f"r2 {action} {key!r} failed: {type(exc).__name__}") from exc

    def _head(self, key: str) -> dict[str, Any]:
        with self._guard("head", key):
            return self._s3.head_object(Bucket=self._bucket, Key=self._k(key))

    def _stat(self, key: str) -> dict[str, Any]:
        """Like ``_head``, but a missing key gives an empty dict."""
        with contextlib.suppress(KeyError):
            return self._head(key)
        return {}

    def _reject(self, action: str, key: str, why: str) -> None:
        """A write went wrong after the store accepted it: remove the object, then fail."""
        with contextlib.suppress(InfraError):
            self.delete(key)
        raise InfraError(f"r2 {action} {key!r}: {why}")

    def _settle(self, action: str, key: str, want: int) -> None:
        """After a write: the stored size must equal the intended size."""
        got = self._stat(key).get("ContentLength", -1)
        if got != want:
            self._reject(action, key, f"store holds {got} bytes, expected {want}")

    def put(self, key: str, data: bytes) -> None:
        with self._guard("put", key):
            reply = self._s3.put_object(Bucket=self._bucket, Key=self._k(key), Body=data)
        if reply.get("ETag", "").strip('"') != hashlib.md5(data, usedforsecurity=False).hexdigest():
            self._reject("put", key, "the store answered with a different MD5")

    def get(self, key: str) -> bytes:
        with self._guard("get", key):
            return self._s3.get_object(Bucket=self._bucket, Key=self._k(key))["Body"].read()

    def put_file(self, key: str, path: Path) -> None:
        with self._guard("put_file", key):
            size = Path(path).stat().st_size
            self._s3.upload_file(str(path), self._bucket, self._k(key), Config=self._transfer)
        self._settle("put_file", key, size)

    def get_file(self, key: str, path: Path) -> None:
        path = Path(path)
        want = self.size(key)
        with self._guard("get_file", key):
            path.parent.mkdir(parents=True, exist_ok=True)
            if want == 0:  # no ranged GET on an empty object
                path.write_bytes(b"")
                return
            # Writes a temporary file and renames it, so a failed download leaves nothing at ``path``.
            self._s3.download_file(self._bucket, self._k(key), str(path), Config=self._transfer)
            if path.stat().st_size != want:
                path.unlink()
                raise InfraError(f"r2 get_file {key!r}: short download")

    def exists(self, key: str) -> bool:
        return bool(self._stat(key))

    def size(self, key: str) -> int:
        return int(self._head(key)["ContentLength"])

    def sha256(self, key: str) -> str:
        digest, seen = hashlib.sha256(), 0
        with self._guard("sha256", key):
            reply = self._s3.get_object(Bucket=self._bucket, Key=self._k(key))
            for chunk in reply["Body"].iter_chunks(4 * MIB):
                digest.update(chunk)
                seen += len(chunk)
            if seen != int(reply["ContentLength"]):
                raise InfraError(f"r2 sha256 {key!r}: read {seen} of {reply['ContentLength']} bytes")
        return digest.hexdigest()

    def copy(self, src: str, dst: str) -> None:
        before = self._head(src)
        if src == dst:
            return
        size = int(before["ContentLength"])
        source = {"Bucket": self._bucket, "Key": self._k(src)}
        target = {"Bucket": self._bucket, "Key": self._k(dst)}
        with self._guard("copy", src):
            if size <= self._part_size:
                self._s3.copy_object(CopySource=source, **target)
            else:
                self._copy_parts(source, target, size)
        if self._stat(src).get("ETag") != before["ETag"]:  # parts may come from two versions of the source
            self._reject("copy", dst, "the source changed during the copy")
        self._settle("copy", dst, size)

    def _copy_parts(self, source: dict[str, str], target: dict[str, str], size: int) -> None:
        """Server-side copy in ranges. Only plain S3 calls that R2 documents; an error aborts the upload."""
        upload = {**target, "UploadId": self._s3.create_multipart_upload(**target)["UploadId"]}

        def copy_part(number: int) -> dict[str, Any]:
            first = (number - 1) * self._part_size
            span = f"bytes={first}-{min(first + self._part_size, size) - 1}"
            reply = self._s3.upload_part_copy(**upload, PartNumber=number, CopySource=source, CopySourceRange=span)
            return {"PartNumber": number, "ETag": reply["CopyPartResult"]["ETag"]}

        pool = ThreadPoolExecutor(max_workers=8)
        try:
            try:
                parts = list(pool.map(copy_part, range(1, -(-size // self._part_size) + 1)))
            finally:
                pool.shutdown(wait=True, cancel_futures=True)  # no part may still run at the abort
            self._s3.complete_multipart_upload(**upload, MultipartUpload={"Parts": parts})
        except BaseException:
            with contextlib.suppress(Exception):
                self._s3.abort_multipart_upload(**upload)
            raise

    def delete(self, key: str) -> None:
        with self._guard("delete", key):
            self._s3.delete_object(Bucket=self._bucket, Key=self._k(key))

    def list(self, prefix: str) -> list[str]:
        """Sorted keys, matched by path segment as ``ObjectStore`` states: ``l1`` never lists ``l10``."""
        whole = not prefix or prefix.endswith("/")
        keys = []
        with self._guard("list", prefix):
            pages = self._s3.get_paginator("list_objects_v2").paginate(Bucket=self._bucket, Prefix=self._prefix + prefix)
            for page in pages:
                for item in page.get("Contents", []):
                    key = item["Key"][len(self._prefix):]
                    if whole or key == prefix or key.startswith(prefix + "/"):
                        keys.append(key)
        return sorted(keys)

    def presign_get(self, key: str, ttl_s: int) -> str:
        return self._presign("get_object", key, ttl_s, {})

    def presign_put(self, key: str, ttl_s: int) -> tuple[str, dict[str, str]]:
        """A single-request upload link (at most 5 GiB on R2). The uploader must send the returned headers."""
        return self._presign("put_object", key, ttl_s, {"ContentType": PUT_HEADERS["Content-Type"]}), dict(PUT_HEADERS)

    def _presign(self, operation: str, key: str, ttl_s: int, extra: dict[str, str]) -> str:
        if not 0 < int(ttl_s) <= MAX_PRESIGN_S:
            raise ContractError(f"presign ttl must be 1..{MAX_PRESIGN_S} seconds, got {ttl_s}")
        with self._guard("presign", key):
            params = {"Bucket": self._bucket, "Key": self._k(key), **extra}
            return self._s3.generate_presigned_url(operation, Params=params, ExpiresIn=int(ttl_s))
