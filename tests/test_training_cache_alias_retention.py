import os
import unittest
from pathlib import Path
from unittest.mock import patch

import test_training_retention as fixtures
from ops.training_retention import hash_file
from ops.training_cache_alias_retention import remove_training_cache_alias


class TrainingAliasRetention(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingRetention(methodName='runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.plan = self.f.plan(kind='checkpoint-export', step=3)
        self.source = Path(self.plan['directory'])
        self.alias = self.f.workspace/'checkpoints'/self.f.checkpoint
        self.alias.mkdir(parents=True)
        for n in self.f.contents:
            os.link(self.source/n, self.alias/n)
        self.f.report['new_checkpoint'].update(path=str(self.source), files={n:v['sha256'] for n,v in self.plan['files'].items()})
        self.f.write_evidence()
        self.plan.update(directory=str(self.alias), source_directory=str(self.source), active_checkpoints=[],
                         **{n+'_sha256':hash_file(p) for n,p in self.f.paths.items()})

    def test_removes_only_alias_and_keeps_original_evidence_and_source(self):
        result = remove_training_cache_alias(self.plan)
        self.assertTrue(result['source_preserved'])
        self.assertEqual(result['physical_bytes_freed'], 0)
        self.assertFalse(self.alias.exists())
        for n, body in self.f.contents.items():
            self.assertEqual((self.source/n).read_bytes(), body)
            self.assertEqual((self.source/n).stat().st_nlink, 1)
        self.assertTrue(all(p.exists() for p in self.f.paths.values()))

    def test_alias_then_archived_export_frees_bytes_and_keeps_evidence(self):
        from ops.training_retention import remove_training_replica
        remove_training_cache_alias(self.plan)
        result = remove_training_replica(dict(self.plan, directory=str(self.source)))
        self.assertEqual(result['bytes'], sum(len(v) for v in self.f.contents.values()))
        self.assertFalse(self.source.exists())
        self.assertFalse(self.alias.exists())
        self.assertTrue(all(p.exists() for p in self.f.paths.values()))

    def test_current_active_unarchived_and_foreign_aliases_are_preserved(self):
        for change in ({'protected_checkpoints':[self.f.checkpoint]}, {'active_checkpoints':[self.f.checkpoint]},
                       {'archive_authenticated':False}, {'directory':str(self.f.workspace/'other')},
                       {'source_directory':str(self.source.with_name('checkpoint-step-1'))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                remove_training_cache_alias(dict(self.plan, **change))
            self.assertTrue(self.alias.exists())

    def test_unknown_third_links_are_preserved(self):
        os.link(self.source/'model.safetensors', self.f.workspace/'third')
        with self.assertRaisesRegex(ValueError, 'two known'):
            remove_training_cache_alias(self.plan)
        self.assertTrue(self.alias.exists())

    def test_open_alias_or_source_and_gpu_use_are_preserved(self):
        for root in (self.source, self.alias):
            with (root/'model.safetensors').open('rb'):
                with self.assertRaisesRegex(ValueError, 'still open'):
                    remove_training_cache_alias(self.plan)
        with patch('ops.training_retention.gpu_processes', return_value=['123']):
            with self.assertRaisesRegex(ValueError, 'idle GPU'):
                remove_training_cache_alias(self.plan)
        self.assertTrue(self.alias.exists())

    def test_corrupt_shared_bytes_are_preserved(self):
        (self.source/'model.safetensors').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'two known'):
            remove_training_cache_alias(self.plan)
        self.assertTrue(self.alias.exists())

    def test_report_substitution_is_refused_even_when_file_hash_is_updated(self):
        self.f.report['new_checkpoint']['path'] = str(self.source.with_name('other'))
        self.f.write_evidence()
        self.plan['report_sha256'] = hash_file(self.f.paths['report'])
        with self.assertRaisesRegex(ValueError, 'original final'):
            remove_training_cache_alias(self.plan)
        self.assertTrue(self.alias.exists())


if __name__ == '__main__':
    unittest.main()
