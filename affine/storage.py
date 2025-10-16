from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import boto3
from botocore.exceptions import ClientError

from .config import settings
from .core import Block
from .validators import load_block


class StorageError(RuntimeError):
    """Raised when storage operations fail."""


@dataclass(frozen=True)
class StorageConfig:
    endpoint: str
    bucket: str
    prefix: str
    access_key: str
    secret_key: str
    region: str
    create: bool
    public_base: str
    cache_dir: Path


def _prefix_path(prefix: str) -> str:
    return prefix.rstrip("/") if prefix else ""


def _key(prefix: str, validator: str, block_index: int, digest: str) -> str:
    stamp = int(time.time() * 1_000_000)
    prefix_norm = _prefix_path(prefix)
    base = f"{validator or 'validator'}-{block_index:08d}-{stamp}-{digest}"
    return f"{prefix_norm}/{base}.json" if prefix_norm else f"{base}.json"


class BucketStorage:
    def __init__(self, config: StorageConfig) -> None:
        if not config.bucket:
            raise StorageError("No bucket configured.")
        self._config = config
        self._session = boto3.session.Session()
        self._client = self._session.client(
            "s3",
            endpoint_url=config.endpoint or None,
            region_name=None if config.region in {"", "auto"} else config.region,
            aws_access_key_id=config.access_key or None,
            aws_secret_access_key=config.secret_key or None,
        )
        self._ensured = False
        self._cache_dir = config.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def bucket(self) -> str:
        return self._config.bucket

    @property
    def prefix(self) -> str:
        return self._config.prefix

    def ensure_bucket(self) -> None:
        if self._ensured:
            return
        try:
            self._client.head_bucket(Bucket=self.bucket)
            self._ensured = True
            return
        except ClientError as exc:
            if not self._config.create:
                raise StorageError(
                    f"Bucket {self.bucket} not accessible: {exc}"
                ) from exc
        try:
            params: Mapping[str, object]
            if self._config.region in {"", "auto"}:
                params = {"Bucket": self.bucket}
            else:
                params = {
                    "Bucket": self.bucket,
                    "CreateBucketConfiguration": {
                        "LocationConstraint": self._config.region
                    },
                }
            self._client.create_bucket(**params)  # type: ignore[arg-type]
            self._ensured = True
        except ClientError as exc:  # pragma: no cover - defensive
            raise StorageError(f"Failed to create bucket {self.bucket}: {exc}") from exc

    def upload_block(self, block: Block, digest: str) -> Mapping[str, str]:
        self.ensure_bucket()
        payload = json.dumps(
            {"hash": digest, "block": block.canonical_dict()}, ensure_ascii=False
        )
        key = _key(
            self.prefix, block.header.validator, block.header.block_index, digest
        )
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload.encode("utf-8"),
            ContentType="application/json",
        )
        cache_path = self._cache_path(block.header.validator, digest)
        cache_path.write_text(payload)
        url = self.public_url(key)
        result = {"key": key, "hash": digest}
        if url:
            result["url"] = url
        return result

    def public_url(self, key: str) -> Optional[str]:
        base = self._config.public_base.strip()
        if not base:
            return None
        return f"{base.rstrip('/')}/{key.lstrip('/')}"

    def list_latest_keys(self, limit: int = 20, suffix: str = ".json") -> List[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        prefix_norm = _prefix_path(self.prefix)
        prefix = f"{prefix_norm}/" if prefix_norm else ""
        entries: List[tuple[str, datetime]] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item.get("Key")
                if not key or (suffix and not key.endswith(suffix)):
                    continue
                last_modified = item.get("LastModified") or datetime.fromtimestamp(0)
                entries.append((key, last_modified))
        entries.sort(key=lambda pair: pair[1], reverse=True)
        return [key for key, _ in entries[:limit]]

    def download_blocks(self, keys: Sequence[str]) -> List[Mapping[str, object]]:
        payloads: List[Mapping[str, object]] = []
        for key in keys:
            obj = self._client.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read().decode("utf-8")
            data = json.loads(body)
            payloads.append(data)
            cache_path = self._cache_path_from_key(key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(body)
        return payloads

    def load_blocks(self, keys: Sequence[str]) -> List[Block]:
        blocks: List[Block] = []
        for data in self.download_blocks(keys):
            block_payload = data.get("block", data)
            if not isinstance(block_payload, Mapping):
                continue
            blocks.append(load_block(block_payload))
        return blocks

    def _cache_path(self, validator: str, digest: str) -> Path:
        safe_validator = validator or "validator"
        return self._cache_dir / f"{safe_validator}-{digest}.json"

    def _cache_path_from_key(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self._cache_dir / safe


def get_storage(
    require_write: bool = True, create: bool | None = None
) -> BucketStorage:
    if not settings.bucket_configured:
        raise StorageError(
            "Storage is not configured; set AFFINE_BUCKET_NAME and related variables."
        )
    if require_write and not settings.bucket_write_enabled:
        raise StorageError(
            "Write access to storage requires AFFINE_BUCKET_ACCESS_KEY and AFFINE_BUCKET_SECRET_KEY."
        )
    config = StorageConfig(
        endpoint=settings.bucket_endpoint,
        bucket=settings.bucket_name,
        prefix=settings.bucket_prefix,
        access_key=settings.bucket_access_key,
        secret_key=settings.bucket_secret_key,
        region=settings.bucket_region,
        create=settings.bucket_create if create is None else create,
        public_base=settings.bucket_public_base,
        cache_dir=settings.cache_path,
    )
    return BucketStorage(config)


__all__ = ["BucketStorage", "StorageConfig", "StorageError", "get_storage"]
