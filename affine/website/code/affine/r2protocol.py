"""The private-submission wire contract: ids, keys, payloads, manifest.

Pure functions only (hashlib/json/base64) so the validator, the eval pod and
the standalone miner CLI (scripts/submit.py, which vendors a copy) all agree
byte-for-byte. Anything that touches the chain, a keypair or a bucket lives
elsewhere (affine.registrations, affine.r2).

Model references
  r2://{bucket}/{prefix}/       an R2 prefix holding a servable checkpoint
                                (+ manifest.json). Used wherever the old
                                `hf_repo` string went (queue, king, verdicts,
                                /duel requests). `revision` for such a ref is
                                the manifest's model_digest (64 hex).

On-chain payloads (timelock commit-reveal, same transport as affine1)
  affine2|activate|{hotkey}|{b64url ed25519 sig}
      sig over ACTIVATE_MESSAGE(netuid, hotkey, registration_id). Proves the
      hotkey is Ed25519 (an sr25519 signature fails Ed25519 verification) so
      the validator can seal the mailbox to it.
  affine2|ready|{registration_id}|{manifest_sha256}
      manifest_sha256 = sha256 of the uploaded manifest.json bytes.

registration_id = sha256("affine-registration-v1\\0" +
                         canonical_json({"hotkey", "netuid"}))
  Deterministic from the hotkey alone: one hotkey = one prefix, forever.
  Re-activation (lost credentials, expired TTL) reuses the id with a new
  credential generation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re

PROTOCOL_VERSION = 1
REVEAL_PREFIX = "affine2"
R2_SCHEME = "r2://"

REGISTRATION_DOMAIN = b"affine-registration-v1\0"
ACTIVATE_DOMAIN = "affine-activate|v1"
ENVELOPE_DOMAIN = b"affine-mailbox-envelope-v1\0"
MANIFEST_DOMAIN = b"affine-manifest-v1\0"

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,50}$")
BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
PRIVATE_PREFIX_RE = re.compile(r"^models/registrations/[0-9a-f]{64}/$")
PUBLIC_PREFIX_RE = re.compile(r"^models/sha256/[0-9a-f]{64}/$")
MAILBOX_KEY_RE = re.compile(r"^mailbox/v1/[0-9a-f]{64}/generations/[0-9]{20}\.bin$")
# Files a checkpoint may ship. Anything else in the prefix fails intake
# (no *.py, no junk that only burns pod disk).
ALLOWED_FILE_RE = re.compile(
    r"^(config\.json|generation_config\.json|tokenizer\.json|tokenizer_config\.json|"
    r"special_tokens_map\.json|vocab\.json|merges\.txt|tokenizer\.model|"
    r"chat_template\.jinja|chat_template\.json|added_tokens\.json|preprocessor_config\.json|"
    r"video_preprocessor_config\.json|model\.safetensors|model\.safetensors\.index\.json|"
    r"model-\d{5}-of-\d{5}\.safetensors|README\.md|LICENSE(\.[a-z]+)?|\.gitattributes)$")

MANIFEST_FIELDS = ("protocol_version", "signature_scheme", "registration_id",
                   "hotkey", "model_name", "files", "model_digest", "signature")
ENVELOPE_FIELDS = (
    "protocol_version", "validator_identity", "netuid", "hotkey",
    "registration_id", "credential_generation", "r2_endpoint",
    "private_model_bucket", "allowed_prefix", "credential_scope",
    "revocation_event", "submission_policy", "access_key_id",
    "secret_access_key", "session_token", "expires_at", "issued_at",
    "signature_scheme", "validator_signature")


# -- encoding ----------------------------------------------------------------------

def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


# -- ids and keys --------------------------------------------------------------------

def registration_id(netuid: int, hotkey: str) -> str:
    body = canonical_json({"hotkey": hotkey, "netuid": int(netuid)})
    return sha256_hex(REGISTRATION_DOMAIN + body)


def private_prefix(reg_id: str) -> str:
    return f"models/registrations/{reg_id}/"


def public_prefix(model_digest: str) -> str:
    return f"models/sha256/{model_digest}/"


def mailbox_key(reg_id: str, generation: int) -> str:
    if generation < 1:
        raise ValueError("credential generation starts at 1")
    return f"mailbox/v1/{reg_id}/generations/{int(generation):020d}.bin"


def mailbox_prefix(reg_id: str) -> str:
    return f"mailbox/v1/{reg_id}/generations/"


def r2_ref(bucket: str, prefix: str) -> str:
    if not BUCKET_RE.match(bucket):
        raise ValueError(f"invalid bucket name {bucket!r}")
    if not prefix.endswith("/"):
        raise ValueError("prefix must be slash-terminated")
    return f"{R2_SCHEME}{bucket}/{prefix}"


def is_r2_ref(repo: str) -> bool:
    return str(repo or "").startswith(R2_SCHEME)


def parse_r2_ref(repo: str) -> tuple[str, str]:
    """`r2://bucket/prefix/` -> (bucket, prefix). Raises ValueError."""
    if not is_r2_ref(repo):
        raise ValueError(f"not an r2 ref: {repo!r}")
    rest = repo[len(R2_SCHEME):]
    bucket, _, prefix = rest.partition("/")
    if not BUCKET_RE.match(bucket):
        raise ValueError(f"invalid bucket in ref {repo!r}")
    if not prefix.endswith("/") or "//" in prefix or ".." in prefix:
        raise ValueError(f"invalid prefix in ref {repo!r}")
    if not (PRIVATE_PREFIX_RE.match(prefix) or PUBLIC_PREFIX_RE.match(prefix)):
        raise ValueError(f"prefix {prefix!r} is not a registration or sha256 prefix")
    return bucket, prefix


def public_model_url(base_url: str, model_digest: str) -> str:
    return f"{base_url.rstrip('/')}/{public_prefix(model_digest)}manifest.json"


# -- on-chain payloads -------------------------------------------------------------

def activate_message(netuid: int, hotkey: str, reg_id: str) -> bytes:
    return f"{ACTIVATE_DOMAIN}|{int(netuid)}|{hotkey}|{reg_id}".encode()


def build_activate_payload(hotkey: str, signature: bytes) -> str:
    if not SS58_RE.match(hotkey):
        raise ValueError(f"invalid hotkey ss58: {hotkey!r}")
    if len(signature) != 64:
        raise ValueError("ed25519 signature must be 64 bytes")
    return f"{REVEAL_PREFIX}|activate|{hotkey}|{b64url(signature)}"


def build_ready_payload(reg_id: str, manifest_sha256: str) -> str:
    if not HEX64_RE.match(reg_id) or not HEX64_RE.match(manifest_sha256):
        raise ValueError("registration_id and manifest_sha256 must be 64 lowercase hex")
    return f"{REVEAL_PREFIX}|ready|{reg_id}|{manifest_sha256}"


def parse_payload(payload: str) -> dict:
    """Parse an affine2 payload → {"kind": "activate"|"ready", ...}.
    Raises ValueError on anything else."""
    parts = (payload or "").strip().split("|")
    if len(parts) != 4 or parts[0] != REVEAL_PREFIX:
        raise ValueError(f"expected {REVEAL_PREFIX}|activate|hotkey|sig or "
                         f"{REVEAL_PREFIX}|ready|registration_id|manifest_sha256")
    kind = parts[1]
    if kind == "activate":
        hotkey, sig_b64 = parts[2].strip(), parts[3].strip()
        if not SS58_RE.match(hotkey):
            raise ValueError(f"invalid hotkey in activate: {hotkey!r}")
        try:
            sig = b64url_decode(sig_b64)
        except Exception as e:
            raise ValueError(f"activate signature is not base64url: {e}") from e
        if len(sig) != 64:
            raise ValueError("activate signature must decode to 64 bytes")
        return {"kind": "activate", "hotkey": hotkey, "signature": sig}
    if kind == "ready":
        reg_id, msha = parts[2].strip().lower(), parts[3].strip().lower()
        if not HEX64_RE.match(reg_id):
            raise ValueError("ready: registration_id must be 64 hex")
        if not HEX64_RE.match(msha):
            raise ValueError("ready: manifest_sha256 must be 64 hex")
        return {"kind": "ready", "registration_id": reg_id,
                "manifest_sha256": msha}
    raise ValueError(f"unknown {REVEAL_PREFIX} kind {kind!r}")


# -- manifest ------------------------------------------------------------------------

def model_digest_from_inventory(files: list[dict]) -> str:
    """sha256 over the sorted (path, size, sha256) inventory — the content
    identity of a checkpoint, independent of where it is stored."""
    rows = sorted((str(f["path"]), int(f["size"]), str(f["sha256"]).lower())
                  for f in files)
    return sha256_hex(MANIFEST_DOMAIN + canonical_json(rows))


def manifest_signing_bytes(manifest: dict) -> bytes:
    body = {k: v for k, v in manifest.items() if k != "signature"}
    return canonical_json(body)


def validate_manifest_shape(manifest: dict, *, max_files: int = 5000) -> None:
    """Structural checks; raises ValueError with a miner-facing reason.
    Does not verify the signature (needs a keypair) or the objects (needs
    the bucket)."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json is not an object")
    extra = set(manifest) - set(MANIFEST_FIELDS)
    missing = set(MANIFEST_FIELDS) - set(manifest)
    if extra or missing:
        raise ValueError(f"manifest fields: missing={sorted(missing)} extra={sorted(extra)}")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"manifest protocol_version {manifest['protocol_version']!r} "
                         f"!= {PROTOCOL_VERSION}")
    if manifest["signature_scheme"] != "ed25519":
        raise ValueError("manifest signature_scheme must be 'ed25519'")
    if not HEX64_RE.match(str(manifest["registration_id"])):
        raise ValueError("manifest registration_id must be 64 hex")
    if not SS58_RE.match(str(manifest["hotkey"])):
        raise ValueError("manifest hotkey is not an ss58 address")
    name = str(manifest["model_name"])
    if not (1 <= len(name) <= 128):
        raise ValueError("manifest model_name must be 1..128 chars")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a non-empty list")
    if len(files) > max_files:
        raise ValueError(f"manifest lists {len(files)} files > {max_files}")
    seen: set[str] = set()
    for f in files:
        if not isinstance(f, dict) or set(f) != {"path", "size", "sha256"}:
            raise ValueError("each file entry needs exactly path, size, sha256")
        path = str(f["path"])
        if path in seen:
            raise ValueError(f"duplicate file path {path!r}")
        seen.add(path)
        if "/" in path or path.startswith(".") and path != ".gitattributes":
            raise ValueError(f"file path {path!r} must be a bare filename")
        if path == "manifest.json":
            raise ValueError("manifest.json must not list itself")
        if not ALLOWED_FILE_RE.match(path):
            raise ValueError(f"file {path!r} is not an allowed checkpoint file")
        if not isinstance(f["size"], int) or f["size"] < 0:
            raise ValueError(f"file {path!r} size must be a non-negative int")
        if not HEX64_RE.match(str(f["sha256"])):
            raise ValueError(f"file {path!r} sha256 must be 64 lowercase hex")
    if not HEX64_RE.match(str(manifest["model_digest"])):
        raise ValueError("manifest model_digest must be 64 hex")
    if manifest["model_digest"] != model_digest_from_inventory(files):
        raise ValueError("manifest model_digest does not match its file inventory")
    if not isinstance(manifest["signature"], str) or not manifest["signature"]:
        raise ValueError("manifest signature missing")


def build_manifest(*, reg_id: str, hotkey: str, model_name: str,
                   files: list[dict]) -> dict:
    """Unsigned manifest (signature = "" placeholder; sign
    manifest_signing_bytes(m) and fill it in)."""
    files = sorted(({"path": str(f["path"]), "size": int(f["size"]),
                     "sha256": str(f["sha256"]).lower()} for f in files),
                   key=lambda f: f["path"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "signature_scheme": "ed25519",
        "registration_id": reg_id,
        "hotkey": hotkey,
        "model_name": model_name,
        "files": files,
        "model_digest": model_digest_from_inventory(files),
        "signature": "",
    }


# -- mailbox envelope ------------------------------------------------------------------

def envelope_signing_bytes(envelope: dict) -> bytes:
    body = {k: v for k, v in envelope.items() if k != "validator_signature"}
    return ENVELOPE_DOMAIN + canonical_json(body)


def validate_envelope_shape(env: dict) -> None:
    if not isinstance(env, dict):
        raise ValueError("envelope is not an object")
    extra = set(env) - set(ENVELOPE_FIELDS)
    missing = set(ENVELOPE_FIELDS) - set(env)
    if extra or missing:
        raise ValueError(f"envelope fields: missing={sorted(missing)} extra={sorted(extra)}")
    if env["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("envelope protocol_version mismatch")
    if not PRIVATE_PREFIX_RE.match(str(env["allowed_prefix"])):
        raise ValueError("envelope allowed_prefix is not a registration prefix")
    if env["credential_scope"] != "object-read-write":
        raise ValueError("envelope credential_scope unexpected")
    if env["signature_scheme"] != "ed25519":
        raise ValueError("envelope signature_scheme unexpected")
