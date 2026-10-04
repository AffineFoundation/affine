import base64
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nacl.signing import SigningKey
from ops.readonly_source_admission import approve, canonical, checkpoint_inventory, digest
import ops.readonly_source_admission as admission


class ReadonlyAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate(); self.authority = self.key.verify_key.encode().hex()
        files = {'config.json': 'a'*64, 'model.safetensors': 'b'*64}
        self.plan = dict(revision='immutable-readonly-source-admission-v1', role='mine',
                         created_at=10, expires_at=110, GPU_jobs=0, chain_transactions=0,
                         live_configuration_writes=0, allows_concurrent_mining=True,
                         helper_sha256=digest(admission.__file__), checkpoint=dict(files=files,
                         id=hashlib.sha256(canonical(files)).hexdigest()),
                         context_indices=[1, 2], native_control_indices=[1])

    def sign(self, p):
        return dict(payload=p, signer=self.authority,
                    signature=base64.b64encode(self.key.sign(canonical(p)).signature).decode())

    def test_explicit_concurrent_reads_need_no_false_idle_claim(self):
        self.assertEqual(approve(self.sign(self.plan), self.authority, 'mine', now=20), self.plan)
        self.assertNotIn('all_scientific_roles_idle', self.plan)

    def test_wrong_role_signer_and_modified_envelope_refused(self):
        with self.assertRaises(ValueError): approve(self.sign(self.plan), self.authority, 'train', now=20)
        with self.assertRaises(ValueError): approve(self.sign(self.plan), SigningKey.generate().verify_key.encode().hex(), 'mine', now=20)
        doc = self.sign(self.plan); doc['payload']['role'] = 'train'
        with self.assertRaises(Exception): approve(doc, self.authority, 'train', now=20)

    def test_gpu_writes_stale_deadlines_and_wrong_helpers_refused(self):
        changes = [{'GPU_jobs': True}, {'GPU_jobs': 1}, {'chain_transactions': 1},
                   {'live_configuration_writes': 1}, {'helper_sha256': '0'*64},
                   {'allows_concurrent_mining': False}, {'expires_at': 20},
                   {'expires_at': 10000}, {'created_at': 21},
                   {'context_indices': [True]}, {'context_indices': [1, 1]},
                   {'native_control_indices': [3]}, {'native_control_indices': []}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                approve(self.sign({**self.plan, **change}), self.authority, 'mine', now=20)

    def test_complete_checkpoint_identity_and_safe_names_required(self):
        for files in ({'config.json': 'a'*64}, {'../config.json': 'a'*64, 'model.safetensors': 'b'*64}):
            p = copy.deepcopy(self.plan); p['checkpoint'] = dict(files=files, id=hashlib.sha256(canonical(files)).hexdigest())
            with self.assertRaises(ValueError): approve(self.sign(p), self.authority, 'mine', now=20)
        p = copy.deepcopy(self.plan); p['checkpoint']['id'] = '0'*64
        with self.assertRaises(ValueError): approve(self.sign(p), self.authority, 'mine', now=20)

    def test_actual_bytes_extra_members_changes_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'checkpoint'; root.mkdir()
            (root/'config.json').write_bytes(b'{}'); (root/'model.safetensors').write_bytes(b'weights')
            files = {p.name: digest(p) for p in root.iterdir()}
            first = checkpoint_inventory(root, files)
            self.assertEqual(first['model.safetensors']['bytes'], 7)
            (root/'extra.json').write_bytes(b'{}')
            with self.assertRaises(ValueError): checkpoint_inventory(root, files)
            (root/'extra.json').unlink(); (root/'model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError): checkpoint_inventory(root, files)
            (root/'model.safetensors').unlink(); (root/'model.safetensors').symlink_to(root/'config.json')
            with self.assertRaises(ValueError): checkpoint_inventory(root, files)

    def test_unverified_source_is_refused_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'; source.mkdir()
            marker = root/'imported'; package = source/'subnet'; package.mkdir()
            (package/'__init__.py').write_text('')
            bootstrap = package/'source_bootstrap.py'
            bootstrap.write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("executed")\n')
            archive = root/'archive.tar.gz'; archive.write_bytes(b'original archive')
            p = copy.deepcopy(self.plan)
            p['source'] = dict(descriptor=dict(sha256=digest(archive)),
                               source_files={'subnet/__init__.py': digest(package/'__init__.py'),
                                             'subnet/source_bootstrap.py': '0'*64})
            with patch('ops.readonly_source_admission.time.time', return_value=20), patch.dict('os.environ', CUDA_VISIBLE_DEVICES=''):
                with self.assertRaisesRegex(ValueError, 'before candidate imports'):
                    admission.run(self.sign(p), self.authority, 'mine', source, root/'checkpoint', archive, root/'receipt.json')
            self.assertFalse(marker.exists())


if __name__ == '__main__': unittest.main()
