#!/usr/bin/env python3
"""ROOT → host-key ceremony script (never embeds key material)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from nacl.signing import SigningKey

from . import ceremony_sign_host_key as ceremony
from .signer import verify_host_key_cert


class TestCeremony(unittest.TestCase):
    def test_sign_host_key_once(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = SigningKey.generate()
            host = SigningKey.generate()
            seed = tmp / 'root.seed'
            seed.write_bytes(bytes(root))
            os.chmod(seed, 0o600)
            out = tmp / 'host.cert.json'
            rc = ceremony.main([
                '--host-pubkey-hex', bytes(host.verify_key).hex(),
                '--out', str(out),
                '--root-seed-file', str(seed),
                '--ttl-days', '30',
            ])
            self.assertEqual(rc, 0)
            doc = json.loads(out.read_text())
            info = verify_host_key_cert(
                doc, expected_root_pubkey_hex=bytes(root.verify_key).hex(),
            )
            self.assertEqual(info['host_pubkey_hex'], bytes(host.verify_key).hex())
            self.assertFalse(doc['cert']['bootstrap_override'])
            # Seed never written into the cert.
            text = out.read_text()
            self.assertNotIn(bytes(root).hex(), text)

    def test_refuses_world_readable_seed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            root = SigningKey.generate()
            host = SigningKey.generate()
            seed = tmp / 'root.seed'
            seed.write_bytes(bytes(root))
            os.chmod(seed, 0o644)
            with self.assertRaises(SystemExit):
                ceremony.main([
                    '--host-pubkey-hex', bytes(host.verify_key).hex(),
                    '--out', str(tmp / 'c.json'),
                    '--root-seed-file', str(seed),
                ])

    def test_refuses_missing_seed_env(self):
        os.environ.pop('AFFINE_PIT_ROOT_SEED_FILE', None)
        rc = ceremony.main([
            '--host-pubkey-hex', 'ab' * 32,
            '--out', '/tmp/should-not-exist-pit-cert.json',
        ])
        self.assertEqual(rc, 2)


if __name__ == '__main__':
    unittest.main()
