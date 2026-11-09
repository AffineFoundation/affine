"""Ephemeral box signer + ROOT admission (BRIEF §1.5, DESIGN §7).

Real ROOT signatures are produced by the operator ROOT signing path, never by
this module. Code here defines the envelope, verification, and a **test double**
that signs with a local fixture key.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol

ADMISSION_VERSION = 'pit-box-admission-v1'
PLACEHOLDER_ROOT_PUBKEY_HEX = '00' * 32
DEFAULT_TTL_S = 3600
MAX_TTL_S = 4 * 3600


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_nacl():
    from nacl.signing import SigningKey, VerifyKey
    from nacl.exceptions import BadSignatureError
    return SigningKey, VerifyKey, BadSignatureError


@dataclass
class EphemeralBoxKey:
    public_key_hex: str
    _signing_key: Any

    def sign(self, message: bytes) -> bytes:
        return bytes(self._signing_key.sign(message).signature)

    def __repr__(self) -> str:
        return 'EphemeralBoxKey(public_key_hex=%r)' % self.public_key_hex


def generate_ephemeral_box_key() -> EphemeralBoxKey:
    SigningKey, _, _ = _require_nacl()
    sk = SigningKey.generate()
    return EphemeralBoxKey(public_key_hex=bytes(sk.verify_key).hex(), _signing_key=sk)


@dataclass(frozen=True)
class AdmissionEnvelope:
    """Unsigned admission body. ROOT signature is attached separately by ops."""

    version: str
    pubkey: str
    pod_id: str
    epoch: int
    binding_H: str
    master_digest: str
    expires_at: float
    payable: bool = True

    def body(self) -> dict:
        return {
            'version': self.version,
            'pubkey': self.pubkey,
            'pod_id': self.pod_id,
            'epoch': self.epoch,
            'binding_H': self.binding_H,
            'master_digest': self.master_digest,
            'expires_at': self.expires_at,
            'payable': self.payable,
        }

    def digest(self) -> str:
        return _digest(self.body())


def build_admission_envelope(
    box_key: EphemeralBoxKey,
    *,
    pod_id: str,
    epoch: int,
    binding_H: str,
    master_digest: str,
    ttl_s: int = DEFAULT_TTL_S,
    now: Optional[float] = None,
) -> AdmissionEnvelope:
    if ttl_s <= 0 or ttl_s > MAX_TTL_S:
        raise ValueError('ttl_s out of range')
    now = float(now if now is not None else time.time())
    return AdmissionEnvelope(
        version=ADMISSION_VERSION,
        pubkey=box_key.public_key_hex,
        pod_id=pod_id,
        epoch=int(epoch),
        binding_H=binding_H,
        master_digest=master_digest,
        expires_at=now + float(ttl_s),
        payable=True,
    )


@dataclass(frozen=True)
class AdmissionCert:
    version: str
    admission: Dict[str, Any]
    root_pubkey_hex: str
    root_signature_hex: str
    issued_at_unix: float

    def body_for_verify(self) -> bytes:
        return _canonical({
            'version': self.version,
            'admission': self.admission,
            'root_pubkey_hex': self.root_pubkey_hex,
            'issued_at_unix': self.issued_at_unix,
        })


class RootSigner(Protocol):
    """Operator ROOT signing path. Production impl lives off this repo."""

    def public_key_hex(self) -> str: ...

    def sign_admission(self, envelope: AdmissionEnvelope, *, issued_at_unix: float) -> AdmissionCert: ...


class FixtureRootSigner:
    """TEST DOUBLE — local ed25519 key. Never a real ROOT key."""

    def __init__(self, signing_key: Optional[Any] = None):
        SigningKey, _, _ = _require_nacl()
        self._sk = signing_key or SigningKey.generate()
        if not isinstance(self._sk, SigningKey):
            raise TypeError('fixture signing_key must be nacl.signing.SigningKey')

    def public_key_hex(self) -> str:
        return bytes(self._sk.verify_key).hex()

    def sign_admission(self, envelope: AdmissionEnvelope, *, issued_at_unix: float) -> AdmissionCert:
        cert = AdmissionCert(
            version='pit-admission-cert-v1',
            admission=envelope.body(),
            root_pubkey_hex=self.public_key_hex(),
            root_signature_hex='',
            issued_at_unix=float(issued_at_unix),
        )
        sig = bytes(self._sk.sign(cert.body_for_verify()).signature)
        return AdmissionCert(
            version=cert.version,
            admission=cert.admission,
            root_pubkey_hex=cert.root_pubkey_hex,
            root_signature_hex=sig.hex(),
            issued_at_unix=cert.issued_at_unix,
        )


def verify_admission_cert(
    cert: AdmissionCert,
    *,
    expected_root_pubkey_hex: str,
    now: Optional[float] = None,
    pubkey: Optional[str] = None,
    epoch: Optional[int] = None,
    binding_H: Optional[str] = None,
) -> Dict[str, Any]:
    _, VerifyKey, BadSignatureError = _require_nacl()
    now = float(now if now is not None else time.time())
    if cert.version != 'pit-admission-cert-v1':
        raise ValueError('unsupported admission cert version')
    if cert.root_pubkey_hex != expected_root_pubkey_hex:
        raise ValueError('root pubkey mismatch')
    if expected_root_pubkey_hex == PLACEHOLDER_ROOT_PUBKEY_HEX:
        raise ValueError('placeholder ROOT pubkey cannot verify real certs')
    adm = cert.admission
    if not isinstance(adm, Mapping) or adm.get('version') != ADMISSION_VERSION:
        raise ValueError('bad admission body')
    if float(adm['expires_at']) < now:
        raise ValueError('admission cert expired')
    if pubkey is not None and adm.get('pubkey') != pubkey:
        raise ValueError('pubkey mismatch')
    if epoch is not None and int(adm.get('epoch')) != int(epoch):
        raise ValueError('epoch mismatch')
    if binding_H is not None and adm.get('binding_H') != binding_H:
        raise ValueError('binding_H mismatch')
    vk = VerifyKey(bytes.fromhex(expected_root_pubkey_hex))
    try:
        vk.verify(cert.body_for_verify(), bytes.fromhex(cert.root_signature_hex))
    except BadSignatureError as exc:
        raise ValueError('admission cert signature invalid') from exc
    return {'ok': True, 'pubkey': adm['pubkey'], 'epoch': adm['epoch'], 'binding_H': adm['binding_H']}


def sign_training_result(box_key: EphemeralBoxKey, payload: Mapping[str, Any]) -> dict:
    """Box signs TRAINING-RESULT payload with the ephemeral key."""
    body = dict(payload)
    msg = _canonical(body)
    return {
        'payload': body,
        'signer_kind': 'pit-box',
        'pubkey': box_key.public_key_hex,
        'signature_hex': box_key.sign(msg).hex(),
    }


def verify_training_result(
    signed: Mapping[str, Any],
    *,
    cert: AdmissionCert,
    expected_root_pubkey_hex: str,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    _, VerifyKey, BadSignatureError = _require_nacl()
    info = verify_admission_cert(cert, expected_root_pubkey_hex=expected_root_pubkey_hex, now=now)
    if signed.get('signer_kind') != 'pit-box':
        raise ValueError('signer_kind must be pit-box')
    if signed.get('pubkey') != info['pubkey']:
        raise ValueError('training-result pubkey != cert pubkey')
    payload = signed.get('payload')
    if not isinstance(payload, Mapping):
        raise ValueError('missing payload')
    if payload.get('binding_H') != cert.admission.get('binding_H'):
        raise ValueError('payload binding mismatch')
    if int(payload.get('epoch')) != int(cert.admission.get('epoch')):
        raise ValueError('payload epoch mismatch')
    vk = VerifyKey(bytes.fromhex(signed['pubkey']))
    try:
        vk.verify(_canonical(payload), bytes.fromhex(signed['signature_hex']))
    except BadSignatureError as exc:
        raise ValueError('training-result signature invalid') from exc
    return {'ok': True, 'candidate': payload.get('candidate_id')}


def request_real_root_signature(*_a, **_k):  # noqa: ANN001
    raise RuntimeError(
        'refused: real ROOT admission signatures are operator-out-of-band only'
    )


# Back-compat aliases used by earlier PIT draft tests/docs.
build_admission_request = build_admission_envelope
generate_ephemeral_key = generate_ephemeral_box_key
