import os
import shutil
import tempfile
import unittest
from pathlib import Path
from ops.prepare_native_eog_portable_source import BASELINE,PORTABLE,prepare,transform
import hashlib


class PortableSourceControls(unittest.TestCase):
    def test_exact_published_baseline_reproduces_qualified_source_hashes(self):
        for name in BASELINE:
            result=transform(name,Path(name).read_bytes())
            self.assertEqual(hashlib.sha256(result).hexdigest(),PORTABLE[name])

    def test_unreviewed_input_rejected(self):
        name=next(iter(BASELINE))
        with self.assertRaisesRegex(ValueError,'baseline'):
            transform(name,Path(name).read_bytes()+b'\n')

    def test_protected_tree_rejected(self):
        root=Path.cwd()
        with self.assertRaisesRegex(ValueError,'independent source copy'):
            prepare(root,root)

    def test_hardlinks_cannot_modify_protected_source(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)
            for name in BASELINE:
                destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
                os.link(Path(name),destination)
            with self.assertRaisesRegex(ValueError,'hard-linked'):
                prepare(target,Path.cwd())

    def test_independent_copy_prepared_without_mutating_original(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)
            for name in BASELINE:
                destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(name,destination)
            evidence=prepare(target,Path.cwd())
            self.assertFalse(evidence['services_started'])
            for name in BASELINE:
                self.assertEqual(hashlib.sha256((target/name).read_bytes()).hexdigest(),PORTABLE[name])
                self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),BASELINE[name])


if __name__=='__main__':unittest.main()
