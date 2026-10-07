"""Mining hydration is not an authorization bypass for other role caches."""
import copy
import hashlib
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from ops import miner_checkpoint_materialization as hydration
from subnet.storage import canonical


class MiningHydrationTests(unittest.TestCase):
    def setUp(self):
        self.source = 'a' * 64
        self.runtime = {'subnet/fixture.py': 'b' * 64}
        self.grant = dict(version=hydration.VERSION, roles=['mine'], source_sha256=self.source,
                          scientific_runtime_files_sha256=self.digest(self.runtime))
        self.manifest = dict(source_bundle=dict(sha256=self.source), checkpoint=dict(id='c' * 64),
                             payable=False, hourly_execution_policy=dict(version='fixture'))
        self.calls = []
        calls = self.calls
        runtime = self.runtime

        class Jobs:
            def __init__(self): self.metadata = dict(source_files=runtime)
            def run(self, label, role, manifest, cache=None, dispatch_only=False, **fields):
                calls.append(dict(role=role, cache=cache, dispatch_only=dispatch_only, fields=fields))
                return dict(dispatch_only=dispatch_only)

        self.module = types.SimpleNamespace(RemoteJobs=Jobs)
        hydration.install(self.module, self.grant, self.source, self.runtime, self.digest)
        self.jobs = Jobs()

    @staticmethod
    def digest(value): return hashlib.sha256(canonical(value)).hexdigest()

    def test_mining_drops_optimistic_cache_and_preserves_attempt_fields(self):
        self.jobs.run('job', 'mine', self.manifest, '/unhydrated', True, seed_start=27, search_budget=1000)
        self.assertIsNone(self.calls[-1]['cache'])
        self.assertTrue(self.calls[-1]['dispatch_only'])
        self.assertEqual(self.calls[-1]['fields'], dict(seed_start=27, search_budget=1000))

    def test_other_roles_keep_explicit_cache_authority(self):
        for role in ('train', 'evaluate', 'verify', 'upload'):
            with self.subTest(role=role):
                self.jobs.run('job', role, self.manifest, '/approved')
                self.assertEqual(self.calls[-1]['cache'], '/approved')

    def test_mining_rejects_wrong_source_and_runtime(self):
        wrong = copy.deepcopy(self.manifest)
        wrong['source_bundle']['sha256'] = 'd' * 64
        with self.assertRaises(ValueError): self.jobs.run('job', 'mine', wrong, '/cache')
        self.jobs.metadata = dict(source_files={})
        with self.assertRaises(ValueError): self.jobs.run('job', 'mine', self.manifest, '/cache')
        self.assertEqual(self.calls, [])

    def test_grant_does_not_allow_non_mining_roles(self):
        grant = dict(self.grant, roles=['mine', 'train'])
        with self.assertRaises(ValueError):
            hydration.install(self.module, grant, self.source, self.runtime, self.digest)

    def test_actual_router_optimistic_hint_does_not_reach_dispatch(self):
        from subnet.role_router import RoutedJobs
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            router = types.SimpleNamespace(
                owners={}, initial_role='mine', roles=dict(mine=self.jobs),
                caches={'mine': {'c' * 64: '/optimistic'}}, cache_lock=threading.RLock(),
                cache_path=root / 'caches.json', owner_path=root / 'owners.json',
                checkpoint_path=lambda role, checkpoint: '/default/' + checkpoint)
            RoutedJobs.run(router, 'job', 'mine', self.manifest, dispatch_only=True, seed_start=27)
            self.assertIsNone(self.calls[-1]['cache'])
            self.assertEqual(self.calls[-1]['fields']['seed_start'], 27)

    def test_unchanged_checkpoint_hydrates_defaults_but_rejects_bad_approved_cache(self):
        from subnet import backend_jobs, model
        content = b'checkpoint fixture'
        expected = hashlib.sha256(content).hexdigest()
        manifest = dict(checkpoint=dict(id='e' * 64, files={'fixture.bin': expected},
                                       read_urls={'fixture.bin': 'https://fixture.invalid'}))
        def get(url, digest, path, limit): path.write_bytes(content)
        def files(root):
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory() as temp, patch.object(backend_jobs, 'get_object', get), patch.object(model, 'model_files', files):
            target = backend_jobs.checkpoint(manifest, Path(temp), cache=None)
            self.assertEqual((target / 'fixture.bin').read_bytes(), content)
            (target / 'fixture.bin').write_bytes(b'bad')
            with self.assertRaises(ValueError): backend_jobs.checkpoint(manifest, Path(temp), cache=str(target))
            backend_jobs.checkpoint(manifest, Path(temp), cache=None)
            self.assertEqual((target / 'fixture.bin').read_bytes(), content)


if __name__ == '__main__': unittest.main()
