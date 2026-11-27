#!/usr/bin/env python3
"""ROOT admission tests — live path is delegated host key only."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional, Tuple

from nacl.signing import SigningKey

from .signer import (
    PLACEHOLDER_ROOT_PUBKEY_HEX,
    AdmissionCert,
    DelegatedHostSigner,
    FixtureRootSigner,
    PerRoundRootSigner,
    build_admission_envelope,
    generate_ephemeral_box_key,
    load_delegated_host_signer,
    request_real_root_signature,
    sign_host_key_cert,
    sign_training_result,
    verify_admission_cert,
    verify_host_key_cert,
    verify_training_result,
)


H0, M0 = 'a' * 64, 'b' * 64


def _ceremony(
    tmp: Path,
    *,
    ttl_s: float = 86400.0,
    now: Optional[float] = None,
) -> Tuple[SigningKey, SigningKey, dict, Path, Path, float]:
    now_f = float(now if now is not None else time.time())
    root = SigningKey.generate()
    host = SigningKey.generate()
    cert = sign_host_key_cert(
        root_signing_key=root,
        host_pubkey_hex=bytes(host.verify_key).hex(),
        issued_at_unix=now_f,
        expires_at_unix=now_f + ttl_s,
    )
    key_path = tmp / 'host.key'
    cert_path = tmp / 'host.cert.json'
    key_path.write_bytes(bytes(host))
    os.chmod(key_path, 0o600)
    cert_path.write_text(json.dumps(cert))
    return root, host, cert, key_path, cert_path, now_f


class TestSigner(unittest.TestCase):
    def test_ephemeral_repr_hides_seed(self):
        key = generate_ephemeral_box_key()
        text = repr(key)
        self.assertIn(key.public_key_hex, text)
        self.assertNotIn('seed', text.lower())
        self.assertNotIn('private', text.lower())

    def test_envelope_has_no_signature(self):
        key = generate_ephemeral_box_key()
        env = build_admission_envelope(
            key, pod_id='pit-box-1', epoch=10, binding_H=H0, master_digest=M0,
            ttl_s=600, now=1_700_000_000.0,
        )
        body = env.body()
        self.assertNotIn('signature', body)
        self.assertEqual(body['version'], 'pit-box-admission-v1')

    def test_fixture_root_roundtrip(self):
        root = FixtureRootSigner()
        key = generate_ephemeral_box_key()
        now = time.time()
        env = build_admission_envelope(
            key, pod_id='pod', epoch=1, binding_H=H0, master_digest=M0, now=now,
        )
        cert = root.sign_admission(env, issued_at_unix=now)
        out = verify_admission_cert(
            cert, expected_root_pubkey_hex=root.public_key_hex(), now=now + 1,
            pubkey=key.public_key_hex, epoch=1, binding_H=H0,
        )
        self.assertTrue(out['ok'])
        self.assertEqual(out['signer_kind'], 'root')

    def test_delegated_host_live_path_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root, host, hcert, key_path, cert_path, now = _ceremony(tmp)
            signer = DelegatedHostSigner(
                host_signing_key=host,
                host_key_cert=hcert,
                root_pubkey_hex=bytes(root.verify_key).hex(),
                now_fn=lambda: now,
            )
            self.assertTrue(signer.ready())
            self.assertEqual(signer.public_key_hex(), bytes(root.verify_key).hex())
            box = generate_ephemeral_box_key()
            env = build_admission_envelope(
                box, pod_id='pod', epoch=2, binding_H=H0, master_digest=M0, now=now,
            )
            cert = request_real_root_signature(env, signer=signer, issued_at_unix=now)
            self.assertEqual(cert.signer_kind, 'delegated_host')
            out = verify_admission_cert(
                cert, expected_root_pubkey_hex=bytes(root.verify_key).hex(), now=now + 1,
                pubkey=box.public_key_hex, epoch=2, binding_H=H0,
            )
            self.assertTrue(out['ok'])
            self.assertEqual(out['signer_kind'], 'delegated_host')
            loaded = load_delegated_host_signer(
                host_key_file=str(key_path),
                host_cert_file=str(cert_path),
                root_pubkey_hex=bytes(root.verify_key).hex(),
            )
            cert2 = loaded.sign_admission(env, issued_at_unix=now)
            verify_admission_cert(
                cert2, expected_root_pubkey_hex=bytes(root.verify_key).hex(), now=now + 1,
            )

    def test_delegated_host_refuses_world_readable_key(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root, _host, _hcert, key_path, cert_path, _now = _ceremony(tmp)
            os.chmod(key_path, 0o644)
            with self.assertRaises(PermissionError):
                load_delegated_host_signer(
                    host_key_file=str(key_path),
                    host_cert_file=str(cert_path),
                    root_pubkey_hex=bytes(root.verify_key).hex(),
                )

    def test_host_key_cert_refuses_bootstrap_override(self):
        root = SigningKey.generate()
        host = SigningKey.generate()
        now = time.time()
        cert = sign_host_key_cert(
            root_signing_key=root,
            host_pubkey_hex=bytes(host.verify_key).hex(),
            issued_at_unix=now,
            expires_at_unix=now + 1000,
        )
        cert['cert']['bootstrap_override'] = True
        with self.assertRaises(ValueError):
            verify_host_key_cert(cert, expected_root_pubkey_hex=bytes(root.verify_key).hex(), now=now)

    def test_host_key_cert_expired(self):
        with tempfile.TemporaryDirectory() as td:
            root, _host, hcert, _, _, now = _ceremony(Path(td), ttl_s=60.0)
            with self.assertRaises(ValueError):
                verify_host_key_cert(
                    hcert,
                    expected_root_pubkey_hex=bytes(root.verify_key).hex(),
                    now=now + 120,
                )

    def test_per_round_unused_alternative_still_verifies(self):
        """per_round kept as unused alternative; live config refuses it."""
        root = PerRoundRootSigner(SigningKey.generate())
        box = generate_ephemeral_box_key()
        now = time.time()
        env = build_admission_envelope(
            box, pod_id='pod', epoch=3, binding_H=H0, master_digest=M0, now=now,
        )
        cert = root.sign_admission(env, issued_at_unix=now)
        self.assertEqual(cert.signer_kind, 'root')
        verify_admission_cert(
            cert, expected_root_pubkey_hex=root.public_key_hex(), now=now + 1,
        )

    def test_tampered_payload_rejected(self):
        root = FixtureRootSigner()
        key = generate_ephemeral_box_key()
        now = time.time()
        env = build_admission_envelope(
            key, pod_id='pod', epoch=1, binding_H=H0, master_digest=M0, now=now,
        )
        cert = root.sign_admission(env, issued_at_unix=now)
        signed = sign_training_result(key, {
            'epoch': 1, 'binding_H': H0, 'candidate_id': 'c' * 64, 'ok': True,
        })
        signed['payload'] = dict(signed['payload'], candidate_id='d' * 64)
        with self.assertRaises(ValueError):
            verify_training_result(
                signed, cert=cert, expected_root_pubkey_hex=root.public_key_hex(), now=now + 1,
            )

    def test_expired_cert(self):
        root = FixtureRootSigner()
        key = generate_ephemeral_box_key()
        now = 1_700_000_000.0
        env = build_admission_envelope(
            key, pod_id='pod', epoch=1, binding_H=H0, master_digest=M0, ttl_s=60, now=now,
        )
        cert = root.sign_admission(env, issued_at_unix=now)
        with self.assertRaises(ValueError):
            verify_admission_cert(cert, expected_root_pubkey_hex=root.public_key_hex(), now=now + 120)

    def test_placeholder_refused(self):
        with self.assertRaises(ValueError):
            verify_admission_cert(
                AdmissionCert(
                    version='pit-admission-cert-v1',
                    admission={'version': 'pit-box-admission-v1', 'pubkey': 'ab' * 32,
                               'pod_id': 'p', 'epoch': 1, 'binding_H': H0, 'master_digest': M0,
                               'expires_at': time.time() + 100, 'payable': True},
                    root_pubkey_hex=PLACEHOLDER_ROOT_PUBKEY_HEX,
                    root_signature_hex='00' * 64,
                    issued_at_unix=time.time(),
                ),
                expected_root_pubkey_hex=PLACEHOLDER_ROOT_PUBKEY_HEX,
            )

    def test_live_request_without_delegated_files_refused(self):
        key = generate_ephemeral_box_key()
        env = build_admission_envelope(
            key, pod_id='pod', epoch=1, binding_H=H0, master_digest=M0, now=time.time(),
        )
        for var in ('AFFINE_PIT_HOST_KEY_FILE', 'AFFINE_PIT_HOST_CERT_FILE', 'AFFINE_PIT_ROOT_PUBKEY_HEX'):
            os.environ.pop(var, None)
        with self.assertRaises((FileNotFoundError, ValueError)):
            request_real_root_signature(env)


if __name__ == '__main__':
    unittest.main()
