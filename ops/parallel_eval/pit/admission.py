"""Ephemeral box admission (DESIGN §12).

Each box generates an ephemeral ed25519 keypair. The host would normally obtain
a short-lived ROOT-signed admission certificate binding that public key to a
box/job. This module:

* generates ephemeral box keys (local, in-memory; never logged)
* builds an admission *request* body (no ROOT signature attached)
* verifies an admission cert when a ROOT signature *blob is supplied by the
  caller as opaque fixture bytes* (tests use a local stand-in key)

This PR never generates, embeds, or requests a real ROOT signature. Production
ROOT signing stays off-box with the operator.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

# Placeholder only — not a real ROOT public key. Tests inject their own.
PLACEHOLDER_ROOT_PUBKEY_HEX = '00' * 32

# Admission cert lifetime defaults (seconds). Short-lived by design.
DEFAULT_TTL_S = 3600
MAX_TTL_S = 4 * 3600


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_nacl():
    try:
        from nacl.signing import SigningKey, VerifyKey
        from nacl.exceptions import BadSignatureError
    except ImportError as exc:  # pragma: no cover - exercised when PyNaCl missing
        raise RuntimeError('PyNaCl (nacl) required for PIT admission cryptography') from exc
    return SigningKey, VerifyKey, BadSignatureError


@dataclass
class EphemeralBoxKey:
    """In-memory ephemeral ed25519 key for one box. Never serialize the seed."""

    public_key_hex: str
    _signing_key: Any  # nacl.signing.SigningKey; private

    def sign(self, message: bytes) -> bytes:
        return bytes(self._signing_key.sign(message).signature)

    def public_bytes(self) -> bytes:
        return bytes(self._signing_key.verify_key)

    def __repr__(self) -> str:
        return 'EphemeralBoxKey(public_key_hex=%r)' % self.public_key_hex

    def __str__(self) -> str:
        return self.__repr__()


def generate_ephemeral_box_key() -> EphemeralBoxKey:
    """Create a fresh box key. Seed material never leaves this process."""
    SigningKey, _, _ = _require_nacl()
    sk = SigningKey.generate()
    return EphemeralBoxKey(public_key_hex=bytes(sk.verify_key).hex(), _signing_key=sk)


@dataclass(frozen=True)
class AdmissionRequest:
    """Unsigned request the host would present to ROOT for a short-lived cert.

    Deliberately contains no signature field and no private material.
    """

    version: str
    box_public_key_hex: str
    item_id: str
    bound_head: str
    bound_master_digest: str
    not_before_unix: float
    not_after_unix: float
    request_id: str

    def body(self) -> dict:
        return {
            'version': self.version,
            'box_public_key_hex': self.box_public_key_hex,
            'item_id': self.item_id,
            'bound_head': self.bound_head,
            'bound_master_digest': self.bound_master_digest,
            'not_before_unix': self.not_before_unix,
            'not_after_unix': self.not_after_unix,
            'request_id': self.request_id,
        }

    def digest(self) -> str:
        return _digest(self.body())


def build_admission_request(
    box_key: EphemeralBoxKey,
    *,
    item_id: str,
    bound_head: str,
    bound_master_digest: str,
    ttl_s: int = DEFAULT_TTL_S,
    now: Optional[float] = None,
    request_id: Optional[str] = None,
) -> AdmissionRequest:
    """Build an admission request. Does **not** obtain or attach a ROOT signature."""
    if ttl_s <= 0 or ttl_s > MAX_TTL_S:
        raise ValueError('ttl_s out of range')
    now = float(now if now is not None else time.time())
    if request_id is None:
        # Random opaque id — not a secret; derived from os.urandom then hex'd.
        request_id = os.urandom(16).hex()
    return AdmissionRequest(
        version='pit-admission-request-v1',
        box_public_key_hex=box_key.public_key_hex,
        item_id=item_id,
        bound_head=bound_head,
        bound_master_digest=bound_master_digest,
        not_before_unix=now,
        not_after_unix=now + float(ttl_s),
        request_id=request_id,
    )


@dataclass(frozen=True)
class AdmissionCert:
    """Verified (or to-verify) short-lived admission certificate.

    ``root_signature_hex`` is opaque fixture/operator material. This module never
    mints a real ROOT signature.
    """

    version: str
    request: Dict[str, Any]
    root_pubkey_hex: str
    root_signature_hex: str
    issued_at_unix: float

    def body_for_verify(self) -> bytes:
        # Signature covers the request body + issued_at + root pubkey id.
        return _canonical({
            'version': self.version,
            'request': self.request,
            'root_pubkey_hex': self.root_pubkey_hex,
            'issued_at_unix': self.issued_at_unix,
        })


def verify_admission_cert(
    cert: AdmissionCert,
    *,
    expected_root_pubkey_hex: str,
    now: Optional[float] = None,
    box_public_key_hex: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify a ROOT-signed admission cert. Fail closed on any mismatch.

    ``expected_root_pubkey_hex`` must be supplied by the operator/test fixture.
    Never defaults to a production ROOT key from this repository.
    """
    _, VerifyKey, BadSignatureError = _require_nacl()
    now = float(now if now is not None else time.time())
    if cert.version != 'pit-admission-cert-v1':
        raise ValueError('unsupported admission cert version')
    if cert.root_pubkey_hex != expected_root_pubkey_hex:
        raise ValueError('root pubkey mismatch')
    if expected_root_pubkey_hex == PLACEHOLDER_ROOT_PUBKEY_HEX:
        raise ValueError('placeholder ROOT pubkey cannot verify real certs')
    req = cert.request
    if not isinstance(req, Mapping):
        raise ValueError('cert request must be a mapping')
    nbf = float(req['not_before_unix'])
    naf = float(req['not_after_unix'])
    if not (nbf <= now <= naf):
        raise ValueError('admission cert expired or not yet valid')
    if box_public_key_hex is not None and req.get('box_public_key_hex') != box_public_key_hex:
        raise ValueError('box public key mismatch')
    try:
        vk = VerifyKey(bytes.fromhex(expected_root_pubkey_hex))
    except Exception as exc:
        raise ValueError('invalid root pubkey hex') from exc
    try:
        sig = bytes.fromhex(cert.root_signature_hex)
    except Exception as exc:
        raise ValueError('invalid root signature hex') from exc
    try:
        vk.verify(cert.body_for_verify(), sig)
    except BadSignatureError as exc:
        raise ValueError('admission cert signature invalid') from exc
    return {
        'ok': True,
        'box_public_key_hex': req['box_public_key_hex'],
        'item_id': req['item_id'],
        'bound_head': req['bound_head'],
        'bound_master_digest': req['bound_master_digest'],
        'not_after_unix': naf,
    }


def fixture_sign_admission_cert(
    request: AdmissionRequest,
    *,
    fixture_root_signing_key: Any,
    issued_at_unix: Optional[float] = None,
) -> AdmissionCert:
    """TEST-ONLY helper: sign an admission cert with a local fixture key.

    Never call this with a real ROOT key. Production ROOT signing is out of band.
    """
    SigningKey, _, _ = _require_nacl()
    if not isinstance(fixture_root_signing_key, SigningKey):
        raise TypeError('fixture_root_signing_key must be nacl.signing.SigningKey')
    issued = float(issued_at_unix if issued_at_unix is not None else time.time())
    root_pubkey_hex = bytes(fixture_root_signing_key.verify_key).hex()
    cert = AdmissionCert(
        version='pit-admission-cert-v1',
        request=request.body(),
        root_pubkey_hex=root_pubkey_hex,
        root_signature_hex='',  # filled below
        issued_at_unix=issued,
    )
    sig = bytes(fixture_root_signing_key.sign(cert.body_for_verify()).signature)
    return AdmissionCert(
        version=cert.version,
        request=cert.request,
        root_pubkey_hex=root_pubkey_hex,
        root_signature_hex=sig.hex(),
        issued_at_unix=issued,
    )


def request_real_root_signature(*_a, **_k):  # noqa: ANN001
    """Explicitly refused. Real ROOT signatures are operator-out-of-band only."""
    raise RuntimeError(
        'refused: this PR never requests or generates a real ROOT admission signature'
    )
