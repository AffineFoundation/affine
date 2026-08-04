"""Hippius S3 client for dashboard state + website publishing.

All writes are best-effort presentation updates: they must never break the
validator loop. On failure we cool down and skip further writes for a while
(Hippius is flaky on long-lived connections), then retry.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import time
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

log = logging.getLogger("affine.hippius")

COOLDOWN_S = 300

# mimetypes has no answer for these; serve as text so code/ renders in-browser.
_TEXT_EXTENSIONS = {".py", ".toml", ".txt", ".md", ".jsonl"}


class Hippius:
    def __init__(self, endpoint: str, bucket: str,
                 access_key: str, secret_key: str):
        self.bucket = bucket
        self.enabled = bool(access_key and secret_key)
        self._retry_after = 0.0
        if self.enabled:
            self.client = boto3.client(
                "s3", endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    connect_timeout=5, read_timeout=15,
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        else:
            self.client = None
            log.warning("Hippius credentials missing; dashboard writes disabled")

    def _available(self) -> bool:
        return self.enabled and time.monotonic() >= self._retry_after

    def _mark_failure(self, key: str, exc: Exception) -> None:
        self._retry_after = time.monotonic() + COOLDOWN_S
        log.warning("Hippius write failed for %s; cooling down %ss: %s",
                    key, COOLDOWN_S, exc)

    def put_bytes(self, key: str, body: bytes, content_type: str,
                  cache_control: str | None = None) -> bool:
        if not self._available():
            return False
        extra = {"CacheControl": cache_control} if cache_control else {}
        try:
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=body,
                ContentType=content_type, **extra)
            return True
        except Exception as exc:
            self._mark_failure(key, exc)
            return False

    def put_json(self, key: str, data) -> bool:
        body = json.dumps(data, default=str).encode()
        return self.put_bytes(key, body, "application/json",
                              cache_control="no-cache")

    def push_directory(self, local_dir: Path, prefix: str = "") -> int:
        """Upload a static site directory (website/). Returns files pushed."""
        pushed = 0
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            key = f"{prefix}{rel}" if prefix else rel
            if path.suffix in _TEXT_EXTENSIONS:
                ctype = "text/plain; charset=utf-8"
            else:
                ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if self.put_bytes(key, path.read_bytes(), ctype,
                              cache_control="max-age=60"):
                pushed += 1
        return pushed
