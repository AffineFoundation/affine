"""Ephemeral box signer + ROOT admission (BRIEF §1.5, DESIGN §7, CUTOVER-PLAN §5.2).

**Live path (owner 2026-11-09): delegated host key only.**
  1. Operator ceremony: ROOT signs a host admission key once (see
     ``ceremony_sign_host_key.py`` / docs/PIT.md). Never embed key material.
  2. Host loads the ROOT-signed host-key cert + host private key (0600) and
     issues short-lived per-box admission certs signed by the host key.
  3. Verifier: box payload → host pubkey (admission) → host-key cert → ROOT.

``FixtureRootSigner`` remains the unit-test double (never a real ROOT).
``PerRoundRootSigner`` is kept as an unused alternative behind the same
``RootSigner`` protocol for experiments; live config refuses it.

Never adds a bootstrap override.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol

ADMISSION_VERSION = 'pit-box-admission-v1'
HOST_KEY_CERT_VERSION = 'pit-host-key-cert-v1'
PLACEHOLDER_ROOT_PUBKEY_HEX = '00' * 32
DEFAULT_TTL_S = 3600
MAX_TTL_S = 4 * 3600
# Live default: delegated host key.
LIVE_ROOT_MODE = 'delegated_host'
ALT_ROOT_MODE = 'per_round'  # unused in live; refused by load_pit_config when mode=live

HOST_KEY_ENV = 'AFFINE_PIT_HOST_KEY_FILE'
HOST_CERT_ENV = 'AFFINE_PIT_HOST_CERT_FILE'
ROOT_PUBKEY_ENV = 'AFFINE_PIT_ROOT_PUBKEY_HEX'


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
    # Delegated path: host signs the admission; ROOT signed the host key once.
    signer_kind: str = 'root'  # 'root' | 'delegated_host'
    host_pubkey_hex: Optional[str] = None
    host_key_cert: Optional[Dict[str, Any]] = None

    def body_for_verify(self) -> bytes:
        body = {
            'version': self.version,
            'admission': self.admission,
            'root_pubkey_hex': self.root_pubkey_hex,
            'issued_at_unix': self.issued_at_unix,
            'signer_kind': self.signer_kind,
        }
        if self.signer_kind == 'delegated_host':
            body['host_pubkey_hex'] = self.host_pubkey_hex
            body['host_key_cert'] = self.host_key_cert
        return _canonical(body)


class RootSigner(Protocol):
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
            signer_kind='root',
        )
        sig = bytes(self._sk.sign(cert.body_for_verify()).signature)
        return AdmissionCert(
            version=cert.version,
            admission=cert.admission,
            root_pubkey_hex=cert.root_pubkey_hex,
            root_signature_hex=sig.hex(),
            issued_at_unix=cert.issued_at_unix,
            signer_kind='root',
        )


# ---------------------------------------------------------------------------
# Host-key ceremony material (ROOT signs once; never embed seeds in repo)
# ---------------------------------------------------------------------------

def host_key_cert_body(
    *,
    host_pubkey_hex: str,
    root_pubkey_hex: str,
    issued_at_unix: float,
    expires_at_unix: float,
    host_id: str = 'affinevalidator',
) -> dict:
    if expires_at_unix <= issued_at_unix:
        raise ValueError('host key cert expires_at must be after issued_at')
    return {
        'version': HOST_KEY_CERT_VERSION,
        'host_id': host_id,
        'host_pubkey_hex': host_pubkey_hex,
        'root_pubkey_hex': root_pubkey_hex,
        'issued_at_unix': float(issued_at_unix),
        'expires_at_unix': float(expires_at_unix),
        'purpose': 'pit-box-admission',
        'bootstrap_override': False,  # never available
    }


def sign_host_key_cert(
    *,
    root_signing_key: Any,
    host_pubkey_hex: str,
    issued_at_unix: float,
    expires_at_unix: float,
    host_id: str = 'affinevalidator',
) -> dict:
    """ROOT-side ceremony helper. Operator runs this with the ROOT key offline.

    Returns a JSON-serializable cert. Never writes the ROOT key. Never enables
    bootstrap override.
    """
    root_pubkey_hex = bytes(root_signing_key.verify_key).hex()
    body = host_key_cert_body(
        host_pubkey_hex=host_pubkey_hex,
        root_pubkey_hex=root_pubkey_hex,
        issued_at_unix=issued_at_unix,
        expires_at_unix=expires_at_unix,
        host_id=host_id,
    )
    sig = bytes(root_signing_key.sign(_canonical(body)).signature)
    return {'cert': body, 'root_signature_hex': sig.hex()}


def verify_host_key_cert(
    cert_doc: Mapping[str, Any],
    *,
    expected_root_pubkey_hex: str,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    _, VerifyKey, BadSignatureError = _require_nacl()
    now = float(now if now is not None else time.time())
    body = cert_doc.get('cert')
    sig = cert_doc.get('root_signature_hex')
    if not isinstance(body, Mapping) or body.get('version') != HOST_KEY_CERT_VERSION:
        raise ValueError('bad host key cert')
    if body.get('bootstrap_override') is not False:
        raise ValueError('host key cert must refuse bootstrap override')
    if body.get('root_pubkey_hex') != expected_root_pubkey_hex:
        raise ValueError('host key cert root pubkey mismatch')
    if expected_root_pubkey_hex == PLACEHOLDER_ROOT_PUBKEY_HEX:
        raise ValueError('placeholder ROOT pubkey cannot verify host key cert')
    if float(body['expires_at_unix']) < now:
        raise ValueError('host key cert expired')
    vk = VerifyKey(bytes.fromhex(expected_root_pubkey_hex))
    try:
        vk.verify(_canonical(body), bytes.fromhex(str(sig)))
    except BadSignatureError as exc:
        raise ValueError('host key cert signature invalid') from exc
    return {'ok': True, 'host_pubkey_hex': body['host_pubkey_hex'], 'expires_at_unix': body['expires_at_unix']}


class DelegatedHostSigner:
    """Live ROOT path: host key (ROOT-signed once) issues short-lived box certs."""

    def __init__(
        self,
        *,
        host_signing_key: Any,
        host_key_cert: Mapping[str, Any],
        root_pubkey_hex: str,
        now_fn=time.time,
    ):
        self._sk = host_signing_key
        self._host_pubkey = bytes(host_signing_key.verify_key).hex()
        self._cert = dict(host_key_cert)
        self._root_pubkey = root_pubkey_hex
        self._now = now_fn
        # Validate cert immediately.
        info = verify_host_key_cert(self._cert, expected_root_pubkey_hex=root_pubkey_hex, now=self._now())
        if info['host_pubkey_hex'] != self._host_pubkey:
            raise ValueError('host key does not match ROOT-signed host cert')

    def public_key_hex(self) -> str:
        # Verifier's trust anchor remains ROOT; expose ROOT pubkey for expected_root checks.
        return self._root_pubkey

    def host_public_key_hex(self) -> str:
        return self._host_pubkey

    def ready(self) -> bool:
        return True

    def sign_admission(self, envelope: AdmissionEnvelope, *, issued_at_unix: float) -> AdmissionCert:
        # Re-check host cert not expired at issue time.
        verify_host_key_cert(self._cert, expected_root_pubkey_hex=self._root_pubkey, now=issued_at_unix)
        cert = AdmissionCert(
            version='pit-admission-cert-v1',
            admission=envelope.body(),
            root_pubkey_hex=self._root_pubkey,
            root_signature_hex='',
            issued_at_unix=float(issued_at_unix),
            signer_kind='delegated_host',
            host_pubkey_hex=self._host_pubkey,
            host_key_cert=dict(self._cert),
        )
        sig = bytes(self._sk.sign(cert.body_for_verify()).signature)
        return AdmissionCert(
            version=cert.version,
            admission=cert.admission,
            root_pubkey_hex=cert.root_pubkey_hex,
            root_signature_hex=sig.hex(),
            issued_at_unix=cert.issued_at_unix,
            signer_kind='delegated_host',
            host_pubkey_hex=cert.host_pubkey_hex,
            host_key_cert=cert.host_key_cert,
        )


def load_delegated_host_signer(
    *,
    host_key_file: Optional[str] = None,
    host_cert_file: Optional[str] = None,
    root_pubkey_hex: Optional[str] = None,
) -> DelegatedHostSigner:
    SigningKey, _, _ = _require_nacl()
    key_path = Path(host_key_file or os.environ.get(HOST_KEY_ENV) or '')
    cert_path = Path(host_cert_file or os.environ.get(HOST_CERT_ENV) or '')
    root_pk = root_pubkey_hex or os.environ.get(ROOT_PUBKEY_ENV) or ''
    if not key_path or not key_path.is_file():
        raise FileNotFoundError('delegated host key file missing (%s)' % HOST_KEY_ENV)
    if not cert_path or not cert_path.is_file():
        raise FileNotFoundError('delegated host cert file missing (%s)' % HOST_CERT_ENV)
    if not root_pk or root_pk == PLACEHOLDER_ROOT_PUBKEY_HEX:
        raise ValueError('AFFINE_PIT_ROOT_PUBKEY_HEX required for delegated host signer')
    mode = key_path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError('host key file must be mode 0600 (or tighter)')
    raw = key_path.read_bytes()
    # Hex text may have a trailing newline; raw 32-byte seeds must not be stripped
    # (a seed byte can legitimately be 0x0a / 0x20).
    text = raw.strip()
    if len(text) == 64 and all(chr(c) in '0123456789abcdefABCDEF' for c in text):
        seed = bytes.fromhex(text.decode('ascii'))
    elif len(raw) == 32:
        seed = raw
    else:
        raise ValueError('host key file must be 32 raw bytes or 64 hex chars')
    sk = SigningKey(seed)
    cert = json.loads(cert_path.read_text())
    return DelegatedHostSigner(host_signing_key=sk, host_key_cert=cert, root_pubkey_hex=root_pk)


class PerRoundRootSigner:
    """Unused live alternative: ROOT signs each admission directly.

    Kept behind the RootSigner protocol for offline experiments. Live config
    refuses root_mode=per_round.
    """

    def __init__(self, signing_key: Any):
        self._sk = signing_key

    def public_key_hex(self) -> str:
        return bytes(self._sk.verify_key).hex()

    def sign_admission(self, envelope: AdmissionEnvelope, *, issued_at_unix: float) -> AdmissionCert:
        cert = AdmissionCert(
            version='pit-admission-cert-v1',
            admission=envelope.body(),
            root_pubkey_hex=self.public_key_hex(),
            root_signature_hex='',
            issued_at_unix=float(issued_at_unix),
            signer_kind='root',
        )
        sig = bytes(self._sk.sign(cert.body_for_verify()).signature)
        return AdmissionCert(
            version=cert.version,
            admission=cert.admission,
            root_pubkey_hex=cert.root_pubkey_hex,
            root_signature_hex=sig.hex(),
            issued_at_unix=cert.issued_at_unix,
            signer_kind='root',
        )


# Back-compat alias: old name pointed at direct ROOT signing.
HostRootSigner = PerRoundRootSigner


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

    if cert.signer_kind == 'delegated_host':
        if not cert.host_key_cert or not cert.host_pubkey_hex:
            raise ValueError('delegated admission missing host key cert')
        verify_host_key_cert(cert.host_key_cert, expected_root_pubkey_hex=expected_root_pubkey_hex, now=now)
        if cert.host_pubkey_hex != (cert.host_key_cert.get('cert') or {}).get('host_pubkey_hex'):
            raise ValueError('delegated admission host pubkey mismatch')
        vk = VerifyKey(bytes.fromhex(cert.host_pubkey_hex))
    elif cert.signer_kind == 'root':
        vk = VerifyKey(bytes.fromhex(expected_root_pubkey_hex))
    else:
        raise ValueError('unsupported signer_kind %r' % cert.signer_kind)
    try:
        vk.verify(cert.body_for_verify(), bytes.fromhex(cert.root_signature_hex))
    except BadSignatureError as exc:
        raise ValueError('admission cert signature invalid') from exc
    return {
        'ok': True,
        'pubkey': adm['pubkey'],
        'epoch': adm['epoch'],
        'binding_H': adm['binding_H'],
        'signer_kind': cert.signer_kind,
    }


def sign_training_result(box_key: EphemeralBoxKey, payload: Mapping[str, Any]) -> dict:
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


def request_real_root_signature(
    envelope: AdmissionEnvelope,
    *,
    signer: Optional[RootSigner] = None,
    issued_at_unix: Optional[float] = None,
) -> AdmissionCert:
    """Issue a short-lived box admission cert via the live delegated host signer."""
    if signer is None:
        signer = load_delegated_host_signer()
    return signer.sign_admission(
        envelope,
        issued_at_unix=float(issued_at_unix if issued_at_unix is not None else time.time()),
    )


build_admission_request = build_admission_envelope
generate_ephemeral_key = generate_ephemeral_box_key
