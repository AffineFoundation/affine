#!/usr/bin/env python3
"""Prove B300/B200/H200 non-sharded train_pit path stays on host.checked + admitted files."""
from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAIN_PIT = REPO / 'ops' / 'parallel_eval' / 'live' / 'train_pit.py'
TRAINER_E = REPO / 'miner' / 'validator' / 'trainer-v834e'
# Committed sha256 inventory — no git history required.
ADMITTED_MANIFEST = (
    REPO / 'ops' / 'parallel_eval' / 'pit' / 'host_v834e_admitted.sha256.json'
)


class TestTrainPitNonShardedUnchanged(unittest.TestCase):
    def test_train_pit_calls_host_checked_not_private_json(self):
        src = TRAIN_PIT.read_text()
        self.assertIn('host.checked(HOST / \'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json\')', src)
        self.assertNotIn('host.private_json(', src)
        self.assertNotIn('pub_doc.get(\'engine\')', src)
        self.assertNotIn('trainer-config.public.json', src)
        # Default host selection prefers host-v834e for non-sharded.
        self.assertIn("v834e = W / 'host-v834e'", src)

    def test_admitted_host_files_byte_identical_to_manifest(self):
        """Non-sharded HOST_FILES must match the committed sha256 manifest."""
        man = json.loads(ADMITTED_MANIFEST.read_text())
        expected = man['host_files']
        tree = ast.parse((TRAINER_E / 'mimo_host_trainer_v83.py').read_text())
        host_files = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == 'HOST_FILES':
                        host_files = ast.literal_eval(node.value)
        self.assertIsNotNone(host_files)
        self.assertNotIn('v835_zero3.py', host_files)
        self.assertEqual(set(host_files), set(expected))
        for rel in host_files:
            path = TRAINER_E / rel
            self.assertTrue(path.is_file(), 'missing admitted host file %s' % rel)
            cur = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                cur, expected[rel],
                'host file drifted from admitted manifest: %s' % rel,
            )

    def test_pe_box_defaults_to_signed_release_not_inrepo(self):
        from ops.parallel_eval.live import pe_box as B
        src = Path(B.__file__).read_text()
        self.assertIn("BOX_HOST = TW + '/host-v834e'", src)
        self.assertIn('PIT_USE_INREPO_TRAINER', src)
        # Implicit in-repo preference must not be the default branch.
        self.assertIn('use_inrepo = os.environ.get(\'PIT_USE_INREPO_TRAINER\'', src)


class TestHostV834fSigningArtifact(unittest.TestCase):
    def test_manifest_fail_closed_no_pyc(self):
        root = Path(__file__).resolve().parent / 'host_v834f_signing'
        man = (root / 'SIGNED-MANIFEST.sha256').read_text().splitlines()
        self.assertTrue(man)
        for line in man:
            if not line.strip():
                continue
            digest, _, rel = line.partition('  ')
            self.assertTrue(rel.startswith('./trainer-host-v834f/'), rel)
            self.assertFalse(rel.endswith('.pyc'), rel)
            self.assertNotIn('__pycache__', rel)
            local = root / 'stage' / rel[len('./trainer-host-v834f/'):]
            self.assertTrue(local.is_file(), 'manifest path missing (fail closed): %s' % rel)
            got = hashlib.sha256(local.read_bytes()).hexdigest()
            self.assertEqual(got, digest, rel)
        # Proposal inventories every HOST_FILES entry.
        prop = __import__('json').loads((root / 'PROPOSAL.unsigned.json').read_text())
        for rel in prop['host_files']:
            self.assertTrue((root / 'stage' / rel).is_file(), rel)
        self.assertIn('v835_zero3.py', prop['host_files'])
        self.assertTrue((root / 'TODO-ROOT-SIGN.md').is_file())

    def test_digest_check_fails_closed_when_file_missing(self):
        import importlib.util
        import tempfile
        import shutil
        root = Path(__file__).resolve().parent / 'host_v834f_signing'
        sign_path = root / 'sign_host_v834f.py'
        spec = importlib.util.spec_from_file_location('sign_host_v834f', sign_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            # Point the helper at a stage missing v835_zero3.py.
            shutil.copytree(root / 'stage', tmp_p / 'stage')
            (tmp_p / 'stage' / 'v835_zero3.py').unlink()
            shutil.copy2(root / 'PROPOSAL.unsigned.json', tmp_p / 'PROPOSAL.unsigned.json')
            shutil.copy2(root / 'SIGNED-MANIFEST.sha256', tmp_p / 'SIGNED-MANIFEST.sha256')
            mod.HERE = tmp_p
            mod.STAGE = tmp_p / 'stage'
            with self.assertRaises(SystemExit) as ctx:
                mod.main([])
            msg = str(ctx.exception)
            self.assertIn('missing', msg.lower())
            self.assertIn('v835_zero3.py', msg)


if __name__ == '__main__':
    unittest.main()
