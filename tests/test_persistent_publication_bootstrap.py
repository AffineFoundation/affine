"""Pinned publication admission must not prevent a fresh trainer startup."""
import importlib
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

from subnet import backend_jobs


class PublicationBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.modules = mock.patch.dict(sys.modules)
        self.modules.start()
        self.finders = list(sys.meta_path)
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        (self.root / 'subnet').mkdir()

    def tearDown(self):
        sys.meta_path[:] = self.finders
        self.modules.stop()
        self.directory.cleanup()

    def test_admission_module_is_reloaded_from_pinned_source(self):
        name = 'subnet.persistent_publication'
        preloaded = types.ModuleType(name)
        preloaded.marker = 'untrusted bootstrap implementation'
        sys.modules[name] = preloaded
        (self.root / 'subnet/persistent_publication.py').write_text(
            "marker = 'fresh pinned implementation'\n")
        backend_jobs.install_source_loader(
            self.root, ('subnet/persistent_publication.py',))
        fresh = importlib.import_module(name)
        self.assertIsNot(fresh, preloaded)
        self.assertEqual(fresh.marker, 'fresh pinned implementation')

    def test_model_runtime_preload_is_still_rejected(self):
        sys.modules['subnet.persistent_publication'] = types.ModuleType(
            'subnet.persistent_publication')
        sys.modules['subnet.gpu_runtime'] = types.ModuleType('subnet.gpu_runtime')
        with self.assertRaisesRegex(ValueError, 'fresh process'):
            backend_jobs.install_source_loader(
                self.root, ('subnet/persistent_publication.py',))

    def test_unspecified_publication_module_is_not_silently_evicted(self):
        preloaded = types.ModuleType('subnet.persistent_publication')
        sys.modules['subnet.persistent_publication'] = preloaded
        backend_jobs.install_source_loader(self.root)
        self.assertIs(sys.modules['subnet.persistent_publication'], preloaded)

    def test_bad_source_bytes_fail_before_bootstrap_eviction(self):
        name = 'subnet.persistent_publication'
        preloaded = types.ModuleType(name)
        sys.modules[name] = preloaded
        job = {'source_files': {'subnet/persistent_publication.py': '0' * 64}}
        with mock.patch.object(backend_jobs, '_validate', return_value=(job, {})):
            with self.assertRaisesRegex(ValueError, 'worker source mismatch'):
                backend_jobs.execute({}, 'unused', self.root)
        self.assertIs(sys.modules[name], preloaded)


if __name__ == '__main__':
    unittest.main()
