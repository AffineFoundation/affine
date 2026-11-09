#!/usr/bin/env python3
import os
import tempfile
import time
import unittest
from pathlib import Path

from .signer import (
    PLACEHOLDER_ROOT_PUBKEY_HEX, AdmissionCert, FixtureRootSigner, HostRootSigner,
    build_admission_envelope, generate_ephemeral_box_key, request_real_root_signature,
    sign_training_result, verify_admission_cert, verify_training_result,
)


H0, M0 = 'a' * 64, 'b' * 64


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

    def test_host_root_signer_from_identity(self):
        from nacl.signing import SigningKey

        class _Id:
            def __init__(self):
                self.key = SigningKey.generate()
                self.id = bytes(self.key.verify_key).hex()

        root = HostRootSigner(identity=_Id())
        self.assertTrue(root.ready())
        key = generate_ephemeral_box_key()
        now = time.time()
        env = build_admission_envelope(
            key, pod_id='pod', epoch=2, binding_H=H0, master_digest=M0, now=now,
        )
        cert = request_real_root_signature(env, signer=root, issued_at_unix=now)
        out = verify_admission_cert(
            cert, expected_root_pubkey_hex=root.public_key_hex(), now=now + 1,
            pubkey=key.public_key_hex, epoch=2, binding_H=H0,
        )
        self.assertTrue(out['ok'])

    def test_host_root_signer_from_seed_file(self):
        seed = os.urandom(32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'root.seed'
            path.write_bytes(seed)
            os.chmod(path, 0o600)
            root = HostRootSigner(seed_file=str(path))
            self.assertTrue(root.ready())
            key = generate_ephemeral_box_key()
            now = time.time()
            env = build_admission_envelope(
                key, pod_id='pod', epoch=3, binding_H=H0, master_digest=M0, now=now,
            )
            cert = root.sign_admission(env, issued_at_unix=now)
            verify_admission_cert(
                cert, expected_root_pubkey_hex=root.public_key_hex(), now=now + 1,
            )

    def test_host_root_signer_refuses_world_readable_seed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'root.seed'
            path.write_bytes(b'\x00' * 32)
            os.chmod(path, 0o644)
            with self.assertRaises(PermissionError):
                HostRootSigner(seed_file=str(path))

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

    def test_real_root_without_config_refused(self):
        key = generate_ephemeral_box_key()
        env = build_admission_envelope(
            key, pod_id='pod', epoch=1, binding_H=H0, master_digest=M0, now=time.time(),
        )
        old = os.environ.pop('AFFINE_PIT_ROOT_SEED_FILE', None)
        try:
            with self.assertRaises(RuntimeError):
                request_real_root_signature(env)
        finally:
            if old is not None:
                os.environ['AFFINE_PIT_ROOT_SEED_FILE'] = old


if __name__ == '__main__':
    unittest.main()
