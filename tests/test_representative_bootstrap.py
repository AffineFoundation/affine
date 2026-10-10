"""Representative admission must reload metadata without admitting model preloads."""
import pathlib
import subprocess
import sys
import unittest


SCRIPT = r'''
import importlib, pathlib, sys, tempfile, types
from unittest import mock
from subnet import backend_jobs as b

mode = sys.argv[1]
name = 'subnet.training_task_representatives'
member = 'subnet/training_task_representatives.py'
root = pathlib.Path(b.__file__).resolve().parent.parent
preloaded = importlib.import_module(name)
if mode == 'reload':
    b.install_source_loader(root, (member,))
    fresh = importlib.import_module(name)
    assert fresh is not preloaded
    assert pathlib.Path(fresh.__file__).resolve() == root / member
    assert isinstance(fresh.__loader__, b.importlib.abc.Loader)
elif mode in ('model', 'gpu_runtime'):
    other = 'subnet.' + mode
    runtime = types.ModuleType(other)
    sys.modules[other] = runtime
    try:
        b.install_source_loader(root, (member,))
    except ValueError as error:
        assert 'fresh process' in str(error)
    else:
        raise AssertionError('runtime preload accepted')
    assert sys.modules[other] is runtime
elif mode == 'undeclared':
    b.install_source_loader(root)
    assert sys.modules[name] is preloaded
elif mode == 'bad-hash':
    job = {'source_files': {member: '0' * 64}}
    with tempfile.TemporaryDirectory() as directory:
        with mock.patch.object(b, '_validate', return_value=(job, {})):
            try:
                b.execute({}, 'unused', directory)
            except ValueError as error:
                assert 'worker source mismatch' in str(error)
            else:
                raise AssertionError('bad source hash accepted')
        assert not list(pathlib.Path(directory).iterdir())
    assert sys.modules[name] is preloaded
else:
    raise AssertionError('unknown control')
print('passed')
'''


class RepresentativeBootstrapTests(unittest.TestCase):
    def control(self, mode):
        result = subprocess.run(
            [sys.executable, '-B', '-c', SCRIPT, mode],
            cwd=pathlib.Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('passed', result.stdout)

    def test_actual_metadata_module_reloads_from_pinned_finder(self):
        self.control('reload')

    def test_model_and_gpu_runtime_preloads_still_fail(self):
        for module in ('model', 'gpu_runtime'):
            with self.subTest(module=module):
                self.control(module)

    def test_undeclared_helper_is_not_silently_evicted(self):
        self.control('undeclared')

    def test_source_hash_failure_precedes_metadata_eviction(self):
        self.control('bad-hash')


if __name__ == '__main__':
    unittest.main()
