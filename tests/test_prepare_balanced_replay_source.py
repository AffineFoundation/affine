import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ops import prepare_balanced_replay_source as recipe


class SourcePreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'; (self.source/'subnet').mkdir(parents=True)
        self.destination = self.root/'output'

    def test_symlinked_destination_parent_cannot_mutate_source(self):
        alias = self.root/'alias'; alias.symlink_to(self.source,target_is_directory=True)
        with self.assertRaises(ValueError): recipe.prepare(self.source,alias/'new-source')
        self.assertFalse((self.source/'new-source').exists())

    def test_private_source_symlink_rejected_before_copy(self):
        secret = self.root/'private-key'; secret.write_text('private fixture')
        (self.source/'subnet'/'leak.py').symlink_to(secret)
        with self.assertRaises(ValueError): recipe.prepare(self.source,self.destination)
        self.assertFalse(self.destination.exists())

    def test_mutated_base_does_not_create_destination(self):
        for name in ('backend_jobs.py','remote_backend.py','gpu_service.py'):
            (self.source/'subnet'/name).write_text('changed base')
        with self.assertRaisesRegex(ValueError,'base source'): recipe.prepare(self.source,self.destination)
        self.assertFalse(self.destination.exists())

    def test_tampered_patch_inventory_is_rejected(self):
        fake = self.root/'ops'; (fake/'replay_source_patches').mkdir(parents=True)
        (fake/'recipe.py').write_text('placeholder')
        (fake/'replay_source_patches'/'pins.json').write_text('{}')
        with patch.object(recipe,'__file__',str(fake/'recipe.py')):
            with self.assertRaisesRegex(ValueError,'patch inventory'):
                recipe.prepare(self.source,self.destination,Path(recipe.__file__).parent)
        self.assertFalse(self.destination.exists())
