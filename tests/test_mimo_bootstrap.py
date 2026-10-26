import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mimo_bootstrap', SOURCE / 'ops/bootstrap_mimo_runtime.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        lock = root / 'requirements.txt'
        lock.write_text('\n'.join(n+'=='+versions[0] for n,versions in bootstrap.REQUIRED.items()))
        wheel = root / 'toploc-0.1.6-cp312-cp312-linux_x86_64.whl'
        wheel.write_bytes(b'fixture-wheel')
        self.args = SimpleNamespace(destination=str(root/'new-worker'),requirements=str(lock),
            requirements_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),toploc_wheel=str(wheel),
            toploc_wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
            max_jobs=2,source_root=str(SOURCE),uv='uv',python='3.12',plan=True)

    def test_plan_never_mutates_existing_environment_or_resolves_dependencies(self):
        plan=bootstrap.bootstrap(self.args)
        self.assertFalse(Path(self.args.destination).exists())
        for command in plan['commands']:
            if 'install' in command:self.assertIn('--no-deps',command)
        self.assertIn('torch==2.14.0',plan['commands'][1])
        self.assertNotIn('--index-url',plan['commands'][1])
        Path(self.args.destination).mkdir()
        with self.assertRaisesRegex(ValueError,'must be new'):bootstrap.bootstrap(self.args)

    def test_hash_and_unpinned_dependency_fail_before_creation(self):
        self.args.requirements_sha256='0'*64
        with self.assertRaisesRegex(ValueError,'hash mismatch'):bootstrap.bootstrap(self.args)
        self.assertFalse(Path(self.args.destination).exists())
        for raw in (b'torch>=2.14',b'--extra-index-url https://example.test',b'torch==2.14.0\ntorch==2.14.0'):
            with self.assertRaises(ValueError):bootstrap.resolved_lock(raw)

    def test_explicit_cuda_distribution_preserves_its_exact_metadata_pin(self):
        plan=bootstrap.commands('/tmp/fixture','uv','3.12','fixture.whl',torch_pin='2.14.0+cu130')
        self.assertIn('torch==2.14.0+cu130',plan[1])
        self.assertIn(bootstrap.TORCH_INDEX,plan[1])

    def test_failed_install_records_attempt_without_overwriting_worker(self):
        self.args.plan=False
        calls=[]
        def failure(command,**kwargs):
            calls.append(command)
            self.assertEqual(kwargs['env']['MAX_JOBS'],'2')
            raise RuntimeError('simulated package failure')
        with self.assertRaisesRegex(RuntimeError,'package failure'):
            bootstrap.bootstrap(self.args,runner=failure)
        import json
        report=json.loads((Path(self.args.destination)/'BOOTSTRAP.json').read_text())
        self.assertFalse(report['completed'])
        self.assertFalse(report['qualified'])
        self.assertEqual(len(calls),1)
        with self.assertRaisesRegex(ValueError,'must be new'):
            bootstrap.bootstrap(self.args,runner=failure)


if __name__ == '__main__':unittest.main()
