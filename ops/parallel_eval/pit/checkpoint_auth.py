"""Lifecycle checkpoint authorization for PIT box-trained candidates.

Signed with the delegated host key (or a test signer). Never embeds seeds.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

CHECKPOINT_AUTH_VERSION = 'pit-checkpoint-authorization-v1'


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def build_checkpoint_auth_payload(
    *,
    checkpoint_id: str,
    input_head: str,
    optimizer_step: int,
    files: Optional[Mapping[str, str]] = None,
    publication_receipt_sha256: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    payload = dict(
        version=CHECKPOINT_AUTH_VERSION,
        checkpoint_id=checkpoint_id,
        input_head=input_head,
        input_checkpoint=input_head,
        optimizer_step=int(optimizer_step),
        source='pit',
    )
    if files:
        payload['files'] = dict(files)
    if publication_receipt_sha256:
        payload['publication_receipt_sha256'] = publication_receipt_sha256
    if extra:
        payload.update(dict(extra))
    return payload


def _signing_key(signer: Any) -> Any:
    if hasattr(signer, '_sk'):
        return signer._sk
    if hasattr(signer, 'sign'):
        return signer
    raise TypeError('signer must expose _sk or sign')


def _pubkey_for(signer: Any) -> tuple[str, str]:
    from .signer import DelegatedHostSigner

    if isinstance(signer, DelegatedHostSigner):
        return 'delegated_host', signer.host_public_key_hex()
    if hasattr(signer, 'host_public_key_hex'):
        return 'delegated_host', signer.host_public_key_hex()
    if hasattr(signer, 'public_key_hex'):
        pk = signer.public_key_hex()
        kind = 'delegated_host' if pk != getattr(signer, '_root_pubkey', pk) else 'fixture'
        return kind, pk
    raise TypeError('cannot resolve signer pubkey')


def sign_checkpoint_authorization(
    payload: Mapping[str, Any],
    *,
    signer: Optional[Any] = None,
) -> dict:
    """Return a lifecycle-acceptable signed checkpoint_authorization envelope."""
    if signer is None:
        from .signer import FixtureRootSigner

        signer = FixtureRootSigner()
    body = dict(payload)
    sk = _signing_key(signer)
    signer_kind, pubkey = _pubkey_for(signer)
    msg = _canonical(body)
    signature_hex = bytes(sk.sign(msg).signature).hex()
    return dict(
        payload=body,
        signer_kind=signer_kind,
        pubkey=pubkey,
        signature_hex=signature_hex,
    )


def verify_checkpoint_authorization(
    signed: Mapping[str, Any],
    *,
    expected_input_head: Optional[str] = None,
    expected_checkpoint_id: Optional[str] = None,
) -> dict:
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    if not isinstance(signed, Mapping):
        raise ValueError('signed authorization required')
    payload = signed.get('payload')
    if not isinstance(payload, Mapping):
        raise ValueError('missing payload')
    if payload.get('version') != CHECKPOINT_AUTH_VERSION:
        raise ValueError('unsupported checkpoint authorization version')
    if payload.get('source') != 'pit':
        raise ValueError('source must be pit')
    pubkey = signed.get('pubkey')
    sig = signed.get('signature_hex')
    if not pubkey or not sig:
        raise ValueError('missing pubkey/signature')
    if expected_input_head is not None and payload.get('input_head') != expected_input_head:
        raise ValueError('input_head mismatch')
    if expected_checkpoint_id is not None and payload.get('checkpoint_id') != expected_checkpoint_id:
        raise ValueError('checkpoint_id mismatch')
    vk = VerifyKey(bytes.fromhex(str(pubkey)))
    try:
        vk.verify(_canonical(dict(payload)), bytes.fromhex(str(sig)))
    except BadSignatureError as exc:
        raise ValueError('checkpoint authorization signature invalid') from exc
    return {
        'ok': True,
        'checkpoint_id': payload.get('checkpoint_id'),
        'input_head': payload.get('input_head'),
        'signer_kind': signed.get('signer_kind'),
        'digest': hashlib.sha256(_canonical(dict(payload))).hexdigest(),
    }
