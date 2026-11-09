#!/usr/bin/env python3
"""Unit tests: ephemeral admission request + cert verify (fixture ROOT only)."""
import time
import unittest

from .admission import (
    PLACEHOLDER_ROOT_PUBKEY_HEX,
    AdmissionCert,
    build_admission_request,
    fixture_sign_admission_cert,
    generate_ephemeral_box_key,
    request_real_root_signature,
    verify_admission_cert,
)


H0 = 'a' * 64
M0 = 'b' * 64


class TestAdmission(unittest.TestCase):
    def test_ephemeral_key_has_no_seed_in_repr(self):
        key = generate_ephemeral_box_key()
        text = repr(key) + str(key)
        self.assertIn(key.public_key_hex, text)
        self.assertNotIn('seed', text.lower())
        self.assertNotIn('private', text.lower())

    def test_build_request_has_no_signature_field(self):
        key = generate_ephemeral_box_key()
        req = build_admission_request(
            key, item_id='item1', bound_head=H0, bound_master_digest=M0,
            ttl_s=600, now=1_700_000_000.0, request_id='reqfixture01',
        )
        body = req.body()
        self.assertNotIn('signature', body)
        self.assertNotIn('root_signature', body)
        self.assertEqual(body['box_public_key_hex'], key.public_key_hex)
        self.assertEqual(len(req.digest()), 64)

    def test_fixture_cert_verifies_with_fixture_root(self):
        from nacl.signing import SigningKey
        fixture_root = SigningKey.generate()
        key = generate_ephemeral_box_key()
        now = time.time()
        req = build_admission_request(
            key, item_id='item1', bound_head=H0, bound_master_digest=M0,
            ttl_s=600, now=now, request_id='reqfixture02',
        )
        cert = fixture_sign_admission_cert(req, fixture_root_signing_key=fixture_root, issued_at_unix=now)
        out = verify_admission_cert(
            cert,
            expected_root_pubkey_hex=bytes(fixture_root.verify_key).hex(),
            now=now + 1,
            box_public_key_hex=key.public_key_hex,
        )
        self.assertTrue(out['ok'])
        self.assertEqual(out['item_id'], 'item1')

    def test_placeholder_root_cannot_verify(self):
        cert = AdmissionCert(
            version='pit-admission-cert-v1',
            request={
                'version': 'pit-admission-request-v1',
                'box_public_key_hex': 'ab' * 32,
                'item_id': 'x',
                'bound_head': H0,
                'bound_master_digest': M0,
                'not_before_unix': time.time() - 10,
                'not_after_unix': time.time() + 100,
                'request_id': 'r',
            },
            root_pubkey_hex=PLACEHOLDER_ROOT_PUBKEY_HEX,
            root_signature_hex='00' * 64,
            issued_at_unix=time.time(),
        )
        with self.assertRaises(ValueError):
            verify_admission_cert(cert, expected_root_pubkey_hex=PLACEHOLDER_ROOT_PUBKEY_HEX)

    def test_real_root_signature_request_refused(self):
        with self.assertRaises(RuntimeError):
            request_real_root_signature(item_id='nope')

    def test_expired_cert_fails_closed(self):
        from nacl.signing import SigningKey
        fixture_root = SigningKey.generate()
        key = generate_ephemeral_box_key()
        now = 1_700_000_000.0
        req = build_admission_request(
            key, item_id='item1', bound_head=H0, bound_master_digest=M0,
            ttl_s=60, now=now, request_id='reqfixture03',
        )
        cert = fixture_sign_admission_cert(req, fixture_root_signing_key=fixture_root, issued_at_unix=now)
        with self.assertRaises(ValueError):
            verify_admission_cert(
                cert,
                expected_root_pubkey_hex=bytes(fixture_root.verify_key).hex(),
                now=now + 120,
            )


if __name__ == '__main__':
    unittest.main()
