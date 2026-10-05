"""Publication bootstrap controls run in fresh processes even in large suites."""
import pathlib
import subprocess
import sys
import unittest

SCRIPT = r'''
import importlib, pathlib, sys, tempfile, types
from unittest import mock
from subnet import backend_jobs as b
mode = sys.argv[1]
name = 'subnet.persistent_publication'
preloaded = types.ModuleType(name)
preloaded.marker = 'untrusted bootstrap implementation'
sys.modules[name] = preloaded
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    (root / 'subnet').mkdir()
    (root / 'subnet/persistent_publication.py').write_text(
        "marker = 'fresh pinned implementation'\n")
    if mode == 'reload':
        b.install_source_loader(root, ('subnet/persistent_publication.py',))
        fresh = importlib.import_module(name)
        assert fresh is not preloaded
        assert fresh.marker == 'fresh pinned implementation'
    elif mode == 'runtime-preload':
        sys.modules['subnet.gpu_runtime'] = types.ModuleType('subnet.gpu_runtime')
        try:
            b.install_source_loader(root, ('subnet/persistent_publication.py',))
        except ValueError as error:
            assert 'fresh process' in str(error)
        else:
            raise AssertionError('runtime preload accepted')
    elif mode == 'undeclared':
        b.install_source_loader(root)
        assert sys.modules[name] is preloaded
    elif mode == 'bad-hash':
        job = {'source_files': {'subnet/persistent_publication.py': '0' * 64}}
        with mock.patch.object(b, '_validate', return_value=(job, {})):
            try:
                b.execute({}, 'unused', root)
            except ValueError as error:
                assert 'worker source mismatch' in str(error)
            else:
                raise AssertionError('bad source hash accepted')
        assert sys.modules[name] is preloaded
    else:
        raise AssertionError('unknown control')
print('passed')
'''

class PublicationBootstrapTests(unittest.TestCase):
    def control(self, mode):
        result = subprocess.run(
            [sys.executable, '-B', '-c', SCRIPT, mode],
            cwd=pathlib.Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('passed', result.stdout)

    def test_admission_module_is_reloaded_from_pinned_source(self):
        self.control('reload')

    def test_model_runtime_preload_is_still_rejected(self):
        self.control('runtime-preload')

    def test_unspecified_publication_module_is_not_silently_evicted(self):
        self.control('undeclared')

    def test_bad_source_bytes_fail_before_bootstrap_eviction(self):
        self.control('bad-hash')

if __name__ == '__main__':
    unittest.main()
