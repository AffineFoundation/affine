"""Access controller for private R2 submissions (affine2 reveals).

One registration per hotkey. Lifecycle, driven by on-chain payloads the
validator sees in RevealedCommitments:

  activate  → verify the Ed25519 signature (proves key type), create a
              parent Cloudflare token on the PRIVATE bucket, mint a
              prefix-scoped temporary credential, seal the signed envelope
              to the hotkey, publish it at the public mailbox key.
  ready     → revoke the parent token (all derived credentials die), delete
              the mailbox blobs, fetch + verify the signed manifest against
              the uploaded objects, run the same hygiene/arch gates the HF
              path runs, and enqueue `r2://private/models/registrations/<id>/`
              with revision = model_digest.
  crown     → copy the prefix to the PUBLIC bucket under
              models/sha256/<model_digest>/ (validator calls `promote`).

Durable state is `state/registrations.json` on the validator (mirrored
best-effort to the private bucket for disaster recovery). No secrets are
persisted: the parent token *value* is used once to mint the credential and
dropped; only its id (needed to revoke) is stored.

Failure policy mirrors the HF path: miner-caused problems (bad signature,
manifest/inventory mismatch, hygiene) burn the hotkey's one slot; our
transport problems (Cloudflare/R2 down) leave the payload unprocessed so the
next scan retries it.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from bittensor.wallet import Keypair

from . import r2, r2protocol as proto
from .config import Config
from .model_store import R2Reader, RepoInfo
from .state import QueueEntry, State, now_iso

log = logging.getLogger("affine.registrations")

ED25519 = 0  # bittensor KeypairType.ED25519
MAX_MANIFEST_BYTES = 2 << 20
STORE_MIRROR_INTERVAL_S = 60.0
STORE_MIRROR_KEY = "state/registrations.json"
STORE_MIRROR_MAX_BYTES = 64 << 20
# Parent tokens outlive the credential TTL by a day so a validator that misses
# `ready` still sees Cloudflare expire the write path on its own.
PARENT_TOKEN_GRACE_S = 86_400
MAX_ACTIVATIONS_PER_SCAN = 4
TOKEN_GC_INTERVAL_S = 1800.0

# Registration states (public on the dashboard).
ACTIVATED = "activated"      # credentials in the mailbox; waiting for upload
QUEUED = "queued"            # manifest verified; on the duel queue
REJECTED = "rejected"        # miner-side failure after ready; slot burned
CROWNED = "crowned"          # promoted to the public bucket


def ed25519_verify(hotkey: str, message: bytes, signature: bytes) -> bool:
    try:
        return bool(Keypair(ss58_address=hotkey, crypto_type=ED25519)
                    .verify(message, signature))
    except Exception:
        return False


def seal_to_hotkey(hotkey: str, plaintext: bytes) -> bytes:
    """NaCl sealed box to the Ed25519 hotkey (bittensor converts the key to
    X25519 internally). Only that hotkey's secret key can open it."""
    return Keypair.encrypt_for(hotkey, plaintext, crypto_type=ED25519)


def signer_from_seed(seed_hex: str) -> Keypair:
    seed = bytes.fromhex(seed_hex.strip().removeprefix("0x"))
    if len(seed) != 32:
        raise ValueError("AFFINE_MAILBOX_SIGNING_SEED must be 32 bytes (64 hex)")
    return Keypair.create_from_seed(seed, crypto_type=ED25519)


class RegistrationStore:
    """registrations.json: {records: {reg_id: rec}, processed: [hotkey:block]}."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, dict] = {}
        self.processed: set[str] = set()
        self._lock = threading.RLock()
        self._dirty = False
        if path.exists():
            d = json.loads(path.read_text())
            self.records = dict(d.get("records") or {})
            self.processed = set(d.get("processed") or [])

    def by_hotkey(self, hotkey: str) -> dict | None:
        for rec in self.records.values():
            if rec.get("hotkey") == hotkey:
                return rec
        return None

    def mark(self, hotkey: str, block: int) -> None:
        with self._lock:
            self.processed.add(f"{hotkey}:{int(block)}")
            self._dirty = True

    def is_processed(self, hotkey: str, block: int) -> bool:
        return f"{hotkey}:{int(block)}" in self.processed

    def upsert(self, rec: dict) -> None:
        with self._lock:
            rec["updated_at"] = now_iso()
            self.records[rec["registration_id"]] = rec
            self._dirty = True

    def flush(self) -> bytes | None:
        """Atomic write; returns the serialized bytes when something changed
        (so the caller can mirror them), else None."""
        with self._lock:
            if not self._dirty:
                return None
            body = json.dumps({"records": self.records,
                               "processed": sorted(self.processed),
                               "flushed_at": now_iso()}, indent=1).encode()
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_bytes(body)
            tmp.replace(self.path)
            self._dirty = False
            return body

    def public_view(self, mailbox_base_url: str,
                    public_models_base_url: str) -> list[dict]:
        """Dashboard rows — never includes credentials (none are stored)."""
        rows = []
        for rec in self.records.values():
            row = {k: rec.get(k) for k in (
                "registration_id", "hotkey", "state", "generation",
                "activate_block", "ready_block", "model_digest", "model_name",
                "challenge_id", "created_at", "updated_at", "detail",
                "credential_expires_at")}
            if rec.get("state") == ACTIVATED and rec.get("mailbox_key"):
                row["mailbox_url"] = f"{mailbox_base_url}/{rec['mailbox_key']}"
            if rec.get("public_ref") and rec.get("model_digest"):
                row["public_url"] = proto.public_model_url(
                    public_models_base_url, rec["model_digest"])
            rows.append(row)
        rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return rows


class AccessController:
    def __init__(self, cfg: Config, state: State,
                 hygiene_check: Callable[[RepoInfo], str | None]):
        self.cfg = cfg
        self.r2cfg = cfg.submission.r2
        self.state = state
        self.hygiene_check = hygiene_check
        sec = cfg.secrets
        self.admin = r2.CloudflareR2Admin(sec.cloudflare_account_id,
                                          sec.cloudflare_api_token)
        self.s3 = r2.s3_client(sec.r2_endpoint, sec.r2_access_key_id,
                               sec.r2_secret_access_key)
        self.signer = signer_from_seed(sec.mailbox_signing_seed)
        self.reader = R2Reader(self.s3)
        store_path = cfg.state_dir / "registrations.json"
        if not store_path.exists():
            self._restore_store_from_mirror(store_path)
        self.store = RegistrationStore(store_path)
        self._last_mirror = 0.0
        self._last_token_gc = 0.0
        log.info("access controller ready: validator identity %s, private=%s "
                 "public=%s dash=%s", self.signer.ss58_address,
                 self.r2cfg.private_bucket, self.r2cfg.public_bucket,
                 self.r2cfg.dash_bucket)

    def _restore_store_from_mirror(self, path: Path) -> None:
        """Fresh validator box: pull the last mirrored registrations.json so
        processed commitments are not replayed and open registrations keep
        their parent-token ids (needed to revoke them on ready)."""
        try:
            body = r2.get_bytes(self.s3, self.r2cfg.private_bucket,
                                STORE_MIRROR_KEY, STORE_MIRROR_MAX_BYTES)
        except Exception as e:
            log.info("no registrations mirror to restore (%s: %s)",
                     type(e).__name__, str(e)[:120])
            return
        json.loads(body)  # refuse to seed the store from a corrupt object
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        log.warning("restored %s from the R2 mirror (%d bytes)", path, len(body))

    @classmethod
    def build_if_configured(cls, cfg: Config, state: State,
                            hygiene_check) -> "AccessController | None":
        r2c = cfg.submission.r2
        if not r2c.configured:
            return None
        sec = cfg.secrets
        missing = [n for n, v in (
            ("CLOUDFLARE_ACCOUNT_ID", sec.cloudflare_account_id),
            ("CLOUDFLARE_API_TOKEN", sec.cloudflare_api_token),
            ("R2_ACCESS_KEY_ID", sec.r2_access_key_id),
            ("R2_SECRET_ACCESS_KEY", sec.r2_secret_access_key),
            ("AFFINE_MAILBOX_SIGNING_SEED", sec.mailbox_signing_seed)) if not v]
        if missing:
            raise SystemExit(f"[submission.r2] enabled but env missing: {missing}")
        return cls(cfg, state, hygiene_check)

    # -- scan entry point ------------------------------------------------------------
    def handle_commitments(self, all_reveals: dict[str, list[tuple[int, str]]]) -> None:
        """Route every unprocessed affine2 payload, oldest first."""
        prefix = self.r2cfg.reveal_prefix + "|"
        items: list[tuple[int, str, str]] = []
        for hotkey, entries in all_reveals.items():
            for block, payload in entries:
                if not str(payload).lstrip().startswith(prefix):
                    continue
                if self.store.is_processed(hotkey, block):
                    continue
                items.append((int(block), hotkey, payload))
        items.sort()
        activations = 0
        for block, hotkey, payload in items:
            if payload.lstrip().startswith(prefix + "activate|"):
                # Each activation is several Cloudflare round trips (token,
                # propagation wait, mint, mailbox put) on the validator's
                # tick thread; bound the work per scan so a go-live rush
                # cannot stall the loop for minutes. The rest stay
                # unprocessed and are picked up next tick, oldest first.
                if activations >= MAX_ACTIVATIONS_PER_SCAN:
                    continue
                activations += 1
            try:
                self._handle_one(hotkey, block, payload)
            except (r2.CloudflareError, OSError) as e:
                # Our transport: leave unprocessed, retry next scan.
                log.warning("affine2 payload from %s@%s deferred (infra): %s",
                            hotkey[:16], block, e)
            except Exception as e:
                if _is_transport_error(e):
                    log.warning("affine2 payload from %s@%s deferred (r2): %s",
                                hotkey[:16], block, e)
                    continue
                log.exception("affine2 payload from %s@%s failed", hotkey[:16], block)
                self.state.record_intake(
                    hotkey=hotkey, block=block, decision="rejected_bad_payload",
                    detail=f"internal error: {type(e).__name__}: {e}")
                self.store.mark(hotkey, block)
        self.flush()
        self.gc_tokens()

    def gc_tokens(self) -> None:
        """Delete affine-reg-* parent tokens that no ACTIVATED registration
        owns (expired, or orphaned by a crash between create and upsert).
        Cloudflare caps account API tokens at 500; expired ones still count."""
        now = time.monotonic()
        if now - self._last_token_gc < TOKEN_GC_INTERVAL_S:
            return
        self._last_token_gc = now
        live = {rec.get("parent_token_id") for rec in self.store.records.values()
                if rec.get("state") == ACTIVATED and rec.get("parent_token_id")}
        try:
            tokens = self.admin.list_tokens("affine-reg-")
        except r2.CloudflareError as e:
            log.warning("token gc: list failed: %s", e)
            return
        removed = 0
        for t in tokens:
            tid = str(t.get("id", ""))
            if tid and tid not in live:
                try:
                    self.admin.delete_token(tid)
                    removed += 1
                except r2.CloudflareError as e:
                    log.warning("token gc: delete %s failed: %s", tid, e)
        if removed:
            log.info("token gc: removed %d orphaned parent token(s), %d live",
                     removed, len(live))

    def _handle_one(self, hotkey: str, block: int, payload: str) -> None:
        try:
            parsed = proto.parse_payload(payload)
        except ValueError as e:
            self.state.record_intake(hotkey=hotkey, block=block,
                                     decision="rejected_bad_payload",
                                     detail=str(e)[:300])
            self.store.mark(hotkey, block)
            return
        if block <= self.cfg.min_submission_block:
            self.state.record_intake(
                hotkey=hotkey, block=block, decision="skipped_min_block",
                detail=(f"reveal block {block} ≤ min_submission_block "
                        f"{self.cfg.min_submission_block}"))
            self.store.mark(hotkey, block)
            return
        if parsed["kind"] == "activate":
            self.on_activate(hotkey, block, parsed["signature"])
        else:
            self.on_ready(hotkey, block, parsed["registration_id"],
                          parsed["manifest_sha256"])

    # -- activate ----------------------------------------------------------------------
    def on_activate(self, hotkey: str, block: int, signature: bytes) -> None:
        reg_id = proto.registration_id(self.cfg.netuid, hotkey)
        king = self.state.king
        if king is not None and hotkey == king.hotkey:
            self._skip(hotkey, block, "skipped_king", "hotkey is the reigning king")
            return
        if hotkey in self.state.seen_hotkeys:
            self._skip(hotkey, block, "skipped_slot_burned",
                       "hotkey already used its one eval slot")
            return
        if not ed25519_verify(hotkey, proto.activate_message(
                self.cfg.netuid, hotkey, reg_id), signature):
            self._skip(hotkey, block, "rejected_not_ed25519",
                       "activate signature does not verify as Ed25519 for this "
                       "hotkey — create an Ed25519 hotkey (submit.py hotkey)")
            return
        rec = self.store.records.get(reg_id)
        if rec is not None and rec.get("state") not in (ACTIVATED,):
            self._skip(hotkey, block, "skipped_slot_burned",
                       f"registration is {rec.get('state')}")
            return
        generation = int(rec.get("generation", 0)) + 1 if rec else 1
        if rec is not None:
            # Re-activation: the old credential dies with its parent token.
            self._revoke(rec, reason="re-activation")
        prefix = proto.private_prefix(reg_id)
        token_name = f"affine-reg-{reg_id[:16]}-g{generation}"
        for stale in self.admin.list_tokens(token_name):
            self.admin.delete_token(str(stale["id"]))
        ttl = int(self.r2cfg.credential_ttl_s)
        token_id, _token_value = self.admin.create_bucket_token(
            token_name, self.r2cfg.private_bucket, write=True,
            ttl_s=ttl + PARENT_TOKEN_GRACE_S)
        try:
            cred = self.admin.mint_temp_credentials(
                token_id, self.r2cfg.private_bucket, prefix, ttl)
        except Exception:
            self.admin.delete_token(token_id)
            raise
        envelope = self._envelope(hotkey, reg_id, generation, prefix, cred)
        blob = seal_to_hotkey(hotkey, proto.canonical_json(envelope))
        key = proto.mailbox_key(reg_id, generation)
        self.s3.put_object(
            Bucket=self.r2cfg.dash_bucket, Key=key, Body=blob,
            ContentType="application/octet-stream",
            CacheControl="public, max-age=15, must-revalidate",
            Metadata={"sha256": proto.sha256_hex(blob),
                      "registration-id": reg_id,
                      "generation": str(generation)})
        now = now_iso()
        self.store.upsert({
            "registration_id": reg_id, "hotkey": hotkey,
            "netuid": self.cfg.netuid, "state": ACTIVATED,
            "generation": generation, "parent_token_id": token_id,
            "activate_block": int(block), "ready_block": None,
            "mailbox_key": key,
            "credential_expires_at": _iso(cred.expires_at),
            "manifest_sha256": None, "model_digest": None, "model_name": None,
            "challenge_id": None, "public_ref": None,
            "created_at": (rec or {}).get("created_at") or now,
            "detail": "credentials sealed to hotkey; waiting for upload + ready",
        })
        self.store.mark(hotkey, block)
        self.state.record_intake(
            hotkey=hotkey, block=block, decision="r2_activated",
            detail=(f"gen {generation} credentials at "
                    f"{self.r2cfg.mailbox_base_url}/{key}"))
        log.info("activated %s (reg %s gen %d) mailbox=%s", hotkey[:16],
                 reg_id[:12], generation, key)

    def _envelope(self, hotkey: str, reg_id: str, generation: int,
                  prefix: str, cred: r2.TempCredential) -> dict:
        env = {
            "protocol_version": proto.PROTOCOL_VERSION,
            "validator_identity": self.signer.ss58_address,
            "netuid": self.cfg.netuid,
            "hotkey": hotkey,
            "registration_id": reg_id,
            "credential_generation": generation,
            "r2_endpoint": self.cfg.secrets.r2_endpoint,
            "private_model_bucket": self.r2cfg.private_bucket,
            "allowed_prefix": prefix,
            "credential_scope": r2.SCOPE_OBJECT_RW,
            "revocation_event": "ready_signal",
            "submission_policy": "one_per_hotkey",
            "access_key_id": cred.access_key_id,
            "secret_access_key": cred.secret_access_key,
            "session_token": cred.session_token,
            "expires_at": _iso(cred.expires_at),
            "issued_at": now_iso(),
            "signature_scheme": "ed25519",
            "validator_signature": "",
        }
        sig = self.signer.sign(proto.envelope_signing_bytes(env))
        env["validator_signature"] = base64.b64encode(sig).decode("ascii")
        proto.validate_envelope_shape(env)
        return env

    def _revoke(self, rec: dict, reason: str) -> None:
        token_id = rec.get("parent_token_id")
        if token_id:
            self.admin.delete_token(str(token_id))
            log.info("revoked parent token for reg %s (%s)",
                     rec["registration_id"][:12], reason)
        rec["parent_token_id"] = None
        deleted = r2.delete_prefix(self.s3, self.r2cfg.dash_bucket,
                                   proto.mailbox_prefix(rec["registration_id"]))
        if deleted:
            log.info("deleted %d mailbox blob(s) for reg %s", deleted,
                     rec["registration_id"][:12])

    # -- ready -------------------------------------------------------------------------
    def on_ready(self, hotkey: str, block: int, reg_id: str,
                 manifest_sha256: str) -> None:
        rec = self.store.records.get(reg_id)
        if rec is None or rec.get("hotkey") != hotkey:
            self._skip(hotkey, block, "rejected_bad_payload",
                       "ready for an unknown registration (activate first, and "
                       "wait for the mailbox before uploading)")
            return
        if rec.get("state") != ACTIVATED:
            self._skip(hotkey, block, "skipped_slot_burned",
                       f"registration is already {rec.get('state')}")
            return
        if int(block) <= int(rec.get("activate_block") or 0):
            self._skip(hotkey, block, "rejected_bad_payload",
                       "ready block precedes activation")
            return
        # 1. Close the write window before reading anything.
        self._revoke(rec, reason="ready")
        rec["ready_block"] = int(block)
        rec["manifest_sha256"] = manifest_sha256
        self.store.upsert(rec)

        # 2. Verify what was uploaded. Miner-side failures burn the slot.
        prefix = proto.private_prefix(reg_id)
        ref_repo = proto.r2_ref(self.r2cfg.private_bucket, prefix)
        try:
            manifest, info = self._verify_upload(rec, prefix, manifest_sha256)
            reason = self.hygiene_check(info)
            if reason:
                raise _MinerFault("repo_hygiene_rejected", reason)
        except _MinerFault as f:
            self._reject(rec, hotkey, block, ref_repo, f.code, f.detail)
            return

        # 3. Enqueue (burns the slot; dedupes on model_digest).
        digest = str(manifest["model_digest"])
        entry = self.state.enqueue(hotkey, ref_repo, digest, int(block),
                                   self.cfg.min_submission_block)
        rec["model_digest"] = digest
        rec["model_name"] = manifest["model_name"]
        if entry is None:
            # State.enqueue recorded why (slot burned / digest seen / king).
            rec["state"] = REJECTED
            rec["detail"] = "not enqueued (see intake decision)"
        else:
            rec["state"] = QUEUED
            rec["challenge_id"] = entry.challenge_id
            rec["detail"] = f"manifest verified; duel queue ← {entry.challenge_id}"
            log.info("enqueued %s for reg %s (%s, %d files, %.1f GB)",
                     entry.challenge_id, reg_id[:12], manifest["model_name"],
                     len(manifest["files"]), info.total_repo_bytes / 1e9)
        self.store.upsert(rec)
        self.store.mark(hotkey, block)

    def _verify_upload(self, rec: dict, prefix: str,
                       manifest_sha256: str) -> tuple[dict, RepoInfo]:
        bucket = self.r2cfg.private_bucket
        key = prefix + "manifest.json"
        exists = r2.object_exists(self.s3, bucket, key)
        if exists is None:
            raise OSError(f"r2 probe of {key} inconclusive")
        if not exists:
            raise _MinerFault("manifest_missing",
                              "ready committed but manifest.json is not in the prefix")
        try:
            raw = r2.get_bytes(self.s3, bucket, key, MAX_MANIFEST_BYTES)
        except ValueError as e:
            raise _MinerFault("manifest_invalid", str(e)) from e
        if proto.sha256_hex(raw) != manifest_sha256:
            raise _MinerFault("manifest_sha_mismatch",
                              "sha256(manifest.json) differs from the ready payload")
        try:
            manifest = json.loads(raw)
            proto.validate_manifest_shape(
                manifest, max_files=self.cfg.submission.max_repo_files)
        except (ValueError, TypeError) as e:
            raise _MinerFault("manifest_invalid", str(e)) from e
        if manifest["registration_id"] != rec["registration_id"]:
            raise _MinerFault("manifest_invalid", "manifest registration_id mismatch")
        if manifest["hotkey"] != rec["hotkey"]:
            raise _MinerFault("manifest_invalid", "manifest hotkey mismatch")
        try:
            sig = base64.b64decode(manifest["signature"], validate=True)
        except Exception as e:
            raise _MinerFault("manifest_invalid", f"signature not base64: {e}") from e
        if len(sig) != 64 or not ed25519_verify(
                rec["hotkey"], proto.manifest_signing_bytes(manifest), sig):
            raise _MinerFault("manifest_bad_signature",
                              "manifest signature does not verify for the hotkey")
        # Inventory: exactly the manifest's files (+ manifest.json), same sizes.
        objs = r2.list_prefix(self.s3, bucket, prefix,
                              max_keys=self.cfg.submission.max_repo_files + 1)
        have = {o["key"][len(prefix):]: o for o in objs}
        want = {f["path"]: f for f in manifest["files"]}
        extra = sorted(set(have) - set(want) - {"manifest.json"})
        missing = sorted(set(want) - set(have))
        if extra or missing:
            raise _MinerFault(
                "inventory_mismatch",
                f"objects under prefix differ from manifest: missing={missing[:5]} "
                f"extra={extra[:5]}")
        bad = [p for p, f in want.items() if have[p]["size"] != f["size"]]
        if bad:
            raise _MinerFault("inventory_mismatch",
                              f"object sizes differ from manifest: {bad[:5]}")
        try:
            info = self.reader.repo_info_from_manifest(
                bucket, prefix, manifest, uploaded_at=have["manifest.json"]["last_modified"])
        except ValueError as e:
            raise _MinerFault("repo_hygiene_rejected", str(e)) from e
        return manifest, info

    def _reject(self, rec: dict, hotkey: str, block: int, repo: str,
                code: str, detail: str) -> None:
        log.info("rejecting reg %s (%s): %s", rec["registration_id"][:12], code, detail)
        self.state.seen_hotkeys.add(hotkey)
        self.state.record_failure_raw(hotkey, repo, rec.get("manifest_sha256") or "",
                                      code, detail)
        self.state.record_intake(hotkey=hotkey, block=block, repo=repo,
                                 revision=rec.get("manifest_sha256") or "",
                                 decision=f"rejected_{code}", detail=detail)
        rec["state"] = REJECTED
        rec["detail"] = f"{code}: {detail}"[:500]
        self.store.upsert(rec)
        self.store.mark(hotkey, block)

    def _skip(self, hotkey: str, block: int, decision: str, detail: str) -> None:
        log.info("affine2 %s from %s@%s: %s", decision, hotkey[:16], block, detail)
        self.state.record_intake(hotkey=hotkey, block=block, decision=decision,
                                 detail=detail)
        self.store.mark(hotkey, block)

    # -- crown -------------------------------------------------------------------------
    def promote(self, entry: QueueEntry) -> str:
        """Copy the private registration prefix to the public bucket. Returns
        the public r2 ref. Raises on any drift or transport failure (the
        caller then crowns the private ref — still servable by the pods)."""
        src_bucket, src_prefix = proto.parse_r2_ref(entry.repo)
        manifest = self.reader.fetch_manifest(src_bucket, src_prefix)
        if manifest["model_digest"] != entry.revision:
            raise ValueError("manifest digest drifted since enqueue")
        dst_prefix = proto.public_prefix(entry.revision)
        expected = {f["path"]: int(f["size"]) for f in manifest["files"]}
        copied = r2.copy_prefix(self.s3, src_bucket, src_prefix,
                                self.r2cfg.public_bucket, dst_prefix,
                                expected=expected)
        public_ref = proto.r2_ref(self.r2cfg.public_bucket, dst_prefix)
        log.info("promoted %s → %s (%d objects)", entry.repo, public_ref, len(copied))
        rec = self.store.by_hotkey(entry.hotkey)
        if rec is not None:
            rec["state"] = CROWNED
            rec["public_ref"] = public_ref
            rec["detail"] = f"crowned by {entry.challenge_id}; public copy live"
            self.store.upsert(rec)
        return public_ref

    # -- persistence -------------------------------------------------------------------
    def flush(self) -> None:
        body = self.store.flush()
        if body is None:
            return
        now = time.monotonic()
        if now - self._last_mirror < STORE_MIRROR_INTERVAL_S:
            return
        self._last_mirror = now
        try:
            self.s3.put_object(Bucket=self.r2cfg.private_bucket,
                               Key=STORE_MIRROR_KEY, Body=body,
                               ContentType="application/json")
        except Exception as e:
            log.warning("registrations mirror to R2 failed: %s", e)

    def dashboard_rows(self) -> list[dict]:
        return self.store.public_view(self.r2cfg.mailbox_base_url,
                                      self.r2cfg.public_models_base_url)


class _MinerFault(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _is_transport_error(e: Exception) -> bool:
    name = type(e).__name__
    mod = type(e).__module__ or ""
    return (mod.startswith("botocore") or mod.startswith("httpx")
            or name in ("EndpointConnectionError", "ConnectTimeoutError",
                        "ReadTimeoutError", "ConnectionClosedError"))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
