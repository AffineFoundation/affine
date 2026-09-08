"""Cloudflare R2 glue: S3 clients, the account API, and prefix operations.

Three buckets carry the private-submission flow (affine.toml [submission.r2]):

  private_bucket  miner uploads under models/registrations/{registration_id}/
                  (no public read; miners write with a prefix-scoped
                  temporary credential minted here)
  public_bucket   crowned models, copied to models/sha256/{model_digest}/
                  (public read — the location is published only on a win)
  dash_bucket     dashboard JSON + the sealed mailbox blobs
                  mailbox/v1/{registration_id}/generations/{n:020d}.bin

Credential model (no presigned URLs): per registration the validator creates
a Cloudflare *account API token* whose only permission is "Workers R2 Storage
Bucket Item Write" on the private bucket, then asks R2 for a *temporary
credential* derived from it, restricted to the registration's prefix and a
TTL. Revoking the parent token kills every temporary credential minted from
it at once — that is how `ready` closes the write window.

This module is pure transport: no chain, no policy. The validator's access
controller (affine.registrations) decides *when* to call it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import boto3
import httpx
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

log = logging.getLogger("affine.r2")

CF_API = "https://api.cloudflare.com/client/v4"

# Cloudflare permission-group names (resolved to ids at runtime; the ids are
# stable per account but looked up by name so a copy of this file works on
# any account).
PERM_BUCKET_ITEM_WRITE = "Workers R2 Storage Bucket Item Write"
PERM_BUCKET_ITEM_READ = "Workers R2 Storage Bucket Item Read"

# Temporary-credential permission scopes R2 accepts.
SCOPE_OBJECT_RW = "object-read-write"
SCOPE_OBJECT_RO = "object-read-only"

# R2's hard TTL ceiling for temporary credentials (7 days).
MAX_TEMP_CREDENTIAL_TTL_S = 604_800
PARENT_PROPAGATION_S = 90

# Single CopyObject caps at 5 GB on R2; safetensors shards are routinely
# ~4-5 GB, so promotion copies go multipart above 1 GiB.
_COPY_CONFIG = TransferConfig(multipart_threshold=1 << 30,
                              multipart_chunksize=1 << 30,
                              max_concurrency=8)


def r2_endpoint(account_id: str) -> str:
    return f"https://{account_id}.r2.cloudflarestorage.com"


def s3_client(endpoint: str, access_key_id: str, secret_access_key: str,
              session_token: str | None = None, *,
              connect_timeout: float = 10.0, read_timeout: float = 120.0,
              max_pool_connections: int = 32):
    """boto3 S3 client tuned for R2 (region auto, sigv4, checksums only when
    the operation requires them — R2 rejects the newer default CRC trailers)."""
    return boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token or None,
        region_name="auto",
        config=BotoConfig(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            connect_timeout=connect_timeout, read_timeout=read_timeout,
            retries={"max_attempts": 4, "mode": "standard"},
            max_pool_connections=max_pool_connections,
        ),
    )


@dataclass(frozen=True)
class TempCredential:
    access_key_id: str
    secret_access_key: str
    session_token: str
    expires_at: float  # unix seconds


class CloudflareError(RuntimeError):
    """Cloudflare API returned success=false or a transport error."""


class CloudflareR2Admin:
    """Account-level R2 management through the Cloudflare REST API.

    Needs an account API token with: Workers R2 Storage Write (buckets,
    domains, lifecycle, temporary credentials) + Account API Tokens Write
    (create/revoke the per-registration parent tokens)."""

    def __init__(self, account_id: str, api_token: str,
                 timeout_s: float = 30.0):
        if not account_id or not api_token:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required")
        self.account_id = account_id
        self._headers = {"Authorization": f"Bearer {api_token}",
                         "Content-Type": "application/json"}
        self._timeout = timeout_s
        self._perm_ids: dict[str, str] = {}

    # -- transport -------------------------------------------------------------
    def _call(self, method: str, path: str, body: dict | None = None,
              params: dict | None = None) -> dict:
        url = f"{CF_API}/accounts/{self.account_id}{path}"
        try:
            r = httpx.request(method, url, headers=self._headers,
                              json=body, params=params, timeout=self._timeout)
        except httpx.HTTPError as e:
            raise CloudflareError(f"{method} {path}: {e}") from e
        try:
            data = r.json()
        except ValueError:
            raise CloudflareError(f"{method} {path}: HTTP {r.status_code} "
                                  f"non-JSON: {r.text[:200]}")
        if r.status_code >= 400 or not data.get("success", False):
            raise CloudflareError(f"{method} {path}: HTTP {r.status_code} "
                                  f"{json.dumps(data.get('errors'))[:400]}")
        return data.get("result")

    # -- buckets -----------------------------------------------------------------
    def list_buckets(self) -> list[str]:
        res = self._call("GET", "/r2/buckets")
        return [b["name"] for b in (res or {}).get("buckets", [])]

    def ensure_bucket(self, name: str) -> bool:
        """Create the bucket if missing. Returns True when it was created."""
        if name in self.list_buckets():
            return False
        self._call("POST", "/r2/buckets", {"name": name})
        log.info("created R2 bucket %s", name)
        return True

    def set_managed_public_domain(self, bucket: str, enabled: bool) -> dict:
        """Toggle the r2.dev public URL for a bucket (dev-grade; the custom
        domain below is what production traffic should use)."""
        return self._call("PUT", f"/r2/buckets/{bucket}/domains/managed",
                          {"enabled": bool(enabled)})

    def get_managed_public_domain(self, bucket: str) -> dict:
        return self._call("GET", f"/r2/buckets/{bucket}/domains/managed")

    def list_custom_domains(self, bucket: str) -> list[dict]:
        res = self._call("GET", f"/r2/buckets/{bucket}/domains/custom")
        return list((res or {}).get("domains", []))

    def add_custom_domain(self, bucket: str, domain: str, zone_id: str) -> dict:
        """Attach a Cloudflare-zone hostname to a bucket (public GET through
        the CDN with caching; no r2.dev rate limits)."""
        return self._call("POST", f"/r2/buckets/{bucket}/domains/custom",
                          {"domain": domain, "zoneId": zone_id,
                           "enabled": True, "minTLS": "1.2"})

    def put_lifecycle(self, bucket: str, rules: list[dict]) -> dict:
        return self._call("PUT", f"/r2/buckets/{bucket}/lifecycle",
                          {"rules": rules})

    def put_public_read_cors(self, bucket: str) -> dict:
        """Browser GET/HEAD from any origin (what the Hippius corpus host
        advertised). Write stays S3-signed and is unaffected by CORS."""
        return self._call("PUT", f"/r2/buckets/{bucket}/cors", {"rules": [{
            "id": "public-read",
            "allowed": {"origins": ["*"], "methods": ["GET", "HEAD"],
                        "headers": ["*"]},
            "exposeHeaders": ["ETag", "Content-Length", "Content-Type"],
            "maxAgeSeconds": 86400,
        }]})

    # -- tokens ------------------------------------------------------------------
    def _permission_id(self, name: str) -> str:
        if name not in self._perm_ids:
            groups = self._call("GET", "/tokens/permission_groups") or []
            for g in groups:
                self._perm_ids[g["name"]] = g["id"]
        if name not in self._perm_ids:
            raise CloudflareError(f"permission group {name!r} not found")
        return self._perm_ids[name]

    def _bucket_resource(self, bucket: str, jurisdiction: str = "default") -> str:
        return f"com.cloudflare.edge.r2.bucket.{self.account_id}_{jurisdiction}_{bucket}"

    def list_tokens(self, name_prefix: str = "") -> list[dict]:
        out: list[dict] = []
        page = 1
        while True:
            res = self._call("GET", "/tokens",
                             params={"page": page, "per_page": 50}) or []
            out.extend(t for t in res
                       if not name_prefix or str(t.get("name", "")).startswith(name_prefix))
            if len(res) < 50:
                return out
            page += 1

    def create_bucket_token(self, name: str, bucket: str, *,
                            write: bool, ttl_s: int | None = None) -> tuple[str, str]:
        """Create an account API token scoped to ONE bucket's objects.

        Returns (token_id, token_value). As an S3 credential the pair is
        access_key_id = token_id, secret_access_key = sha256(token_value);
        the same token_id is the `parentAccessKeyId` for temporary creds.
        The value is shown once — persist what you need immediately."""
        perm = PERM_BUCKET_ITEM_WRITE if write else PERM_BUCKET_ITEM_READ
        body: dict = {
            "name": name,
            "policies": [{
                "effect": "allow",
                "resources": {self._bucket_resource(bucket): "*"},
                "permission_groups": [{"id": self._permission_id(perm)}],
            }],
        }
        if ttl_s:
            body["expires_on"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + ttl_s))
        res = self._call("POST", "/tokens", body)
        return str(res["id"]), str(res["value"])

    def delete_token(self, token_id: str) -> bool:
        """Revoke a token. Idempotent: an unknown id is treated as revoked."""
        try:
            self._call("DELETE", f"/tokens/{token_id}")
            return True
        except CloudflareError as e:
            if "HTTP 404" in str(e) or "not found" in str(e).lower():
                return True
            raise

    def mint_temp_credentials(self, parent_token_id: str, bucket: str,
                              prefix: str, ttl_s: int,
                              scope: str = SCOPE_OBJECT_RW) -> TempCredential:
        """Prefix-scoped temporary S3 credential derived from a parent token.
        Dies with the parent (delete_token) or at the TTL, whichever first."""
        if not prefix.endswith("/"):
            raise ValueError("credential prefix must be slash-terminated")
        ttl_s = max(60, min(int(ttl_s), MAX_TEMP_CREDENTIAL_TTL_S))
        body = {
            "bucket": bucket,
            "parentAccessKeyId": parent_token_id,
            "permission": scope,
            "ttlSeconds": ttl_s,
            "prefixes": [prefix],
        }
        # A parent token created moments ago answers 401 until it propagates
        # (~10 s observed); poll for up to PARENT_PROPAGATION_S.
        deadline = time.time() + PARENT_PROPAGATION_S
        while True:
            try:
                res = self._call("POST", "/r2/temp-access-credentials", body)
                break
            except CloudflareError as e:
                if "HTTP 401" not in str(e) or time.time() >= deadline:
                    raise
                time.sleep(3)
        return TempCredential(
            access_key_id=str(res["accessKeyId"]),
            secret_access_key=str(res["secretAccessKey"]),
            session_token=str(res["sessionToken"]),
            expires_at=time.time() + ttl_s)


# -- object helpers (any S3 client) -----------------------------------------------

def list_prefix(s3, bucket: str, prefix: str,
                max_keys: int = 20_000) -> list[dict]:
    """Every object under `prefix`: [{key, size, last_modified, etag}]."""
    out: list[dict] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        res = s3.list_objects_v2(**kw)
        for o in res.get("Contents", []):
            out.append({"key": o["Key"], "size": int(o["Size"]),
                        "last_modified": o.get("LastModified"),
                        "etag": o.get("ETag")})
            if len(out) > max_keys:
                raise ValueError(f"prefix {prefix} lists > {max_keys} objects")
        if not res.get("IsTruncated"):
            return out
        token = res.get("NextContinuationToken")


def get_bytes(s3, bucket: str, key: str, max_bytes: int) -> bytes:
    """Download a small object, refusing anything above `max_bytes` (the
    caller is reading attacker-controlled keys like manifest/config)."""
    head = s3.head_object(Bucket=bucket, Key=key)
    size = int(head.get("ContentLength", 0))
    if size > max_bytes:
        raise ValueError(f"{key} is {size} bytes > {max_bytes} cap")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError(f"{key} exceeded {max_bytes} bytes while reading")
    return body


def object_exists(s3, bucket: str, key: str) -> bool | None:
    """True/False when R2 answered; None when the probe was inconclusive."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        log.warning("head_object %s/%s inconclusive: %s", bucket, key, code)
        return None
    except Exception:
        log.warning("head_object %s/%s failed", bucket, key, exc_info=True)
        return None


def put_json(s3, bucket: str, key: str, data, *,
             cache_control: str = "public, max-age=15, must-revalidate",
             metadata: dict | None = None) -> None:
    body = json.dumps(data, default=str, sort_keys=True).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body,
                  ContentType="application/json; charset=utf-8",
                  CacheControl=cache_control, Metadata=metadata or {})


def delete_prefix(s3, bucket: str, prefix: str) -> int:
    """Delete every object under a prefix (1000 per batch). Returns count."""
    if not prefix or not prefix.endswith("/"):
        raise ValueError("refusing to delete a non-slash-terminated prefix")
    keys = [o["key"] for o in list_prefix(s3, bucket, prefix)]
    for i in range(0, len(keys), 1000):
        chunk = keys[i:i + 1000]
        s3.delete_objects(Bucket=bucket, Delete={
            "Objects": [{"Key": k} for k in chunk], "Quiet": True})
    return len(keys)


def copy_prefix(s3, src_bucket: str, src_prefix: str,
                dst_bucket: str, dst_prefix: str,
                *, expected: dict[str, int] | None = None) -> list[str]:
    """Server-side copy of every object under src_prefix to dst_prefix
    (multipart above 1 GiB). `expected` = {relative_path: size} lets the
    caller pin the exact inventory being promoted; extra or missing objects
    abort before anything is copied. Returns the relative paths copied."""
    objs = list_prefix(s3, src_bucket, src_prefix)
    rel = {o["key"][len(src_prefix):]: o["size"] for o in objs}
    if expected is not None:
        want = dict(expected)
        want.setdefault("manifest.json", rel.get("manifest.json", -1))
        if set(rel) != set(want) or any(rel[p] != s for p, s in want.items()
                                         if s >= 0):
            raise ValueError(
                f"inventory drift under {src_prefix}: have={sorted(rel)[:5]}.. "
                f"want={sorted(want)[:5]}..")
    copied: list[str] = []
    for path in sorted(rel):
        s3.copy({"Bucket": src_bucket, "Key": src_prefix + path},
                dst_bucket, dst_prefix + path, Config=_COPY_CONFIG)
        copied.append(path)
    return copied
