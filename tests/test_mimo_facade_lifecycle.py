"""CPU ownership controls; retained closed actors must not retain model weights."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import weakref

from subnet.complete_group_runtime import MimoGroupRuntime
from subnet.mimo_parallel_mining import wrap_runtime
from subnet.miner import Miner
from test_complete_group_mimo import mimo_manifest


class ModelSentinel:
    pass


class BackendSentinel:
    def __init__(self, manifest, spec):
        self.model = ModelSentinel()
        self.tokenizer = object()
        self.eos_token_ids = (248046, 248044)
        self.checkpoint_id = manifest['checkpoint']['id']
        self.spec = spec
        self.trainable = False
        self.close_calls = 0
        self.on_close = lambda: None

    def guard(self):
        assert self.model is not None

    def close(self):
        self.close_calls += 1
        self.model = None
        self.on_close()


@contextmanager
def facade_fixture(*, borrowed=False):
    manifest = mimo_manifest()
    manifest['deadline'] = int(time.time()) + 60
    manifest['model_runtime_revision'] = 'qwen3_5-mimo-v1'
    manifest['backend_profile'] = {'device': 'cuda'}
    definition = manifest['environments'][0]
    spec = SimpleNamespace(**{k: definition['harness'][k] for k in
        ('enable_thinking', 'max_context_tokens', 'temperature', 'top_p', 'top_k')})
    backend = BackendSentinel(manifest, spec)

    def bind(runtime, manifest, miner):
        runtime.sampling_context = {'miner': miner}
        return runtime

    with patch('subnet.mimo_runtime.validate_runtime_spec', return_value=spec), \
         patch('subnet.mimo_runtime.MimoRuntime.load', return_value=backend) as load, \
         patch('subnet.complete_group_runtime.EnvironmentSpec.from_dict',
               side_effect=lambda value: SimpleNamespace(**value)), \
         patch('subnet.forced_sampling.bind_runtime', side_effect=bind):
        facade = MimoGroupRuntime('fixture', manifest, definition['spec'],
            definition['harness'], miner='a' * 64, backend=backend if borrowed else None)
        assert load.call_count == int(not borrowed)
        yield facade, backend, manifest, definition


class FacadeLifecycleTests(unittest.TestCase):
    def test_owned_facade_releases_model_once_and_cannot_create_closed_views(self):
        with facade_fixture() as (facade, backend, _, definition):
            model = weakref.ref(backend.model)
            facade.close()
            facade.close()
            self.assertIsNone(model())
            self.assertIsNone(facade.backend)
            self.assertIsNone(facade.model)
            self.assertEqual(backend.close_calls, 1)
            with self.assertRaisesRegex(ValueError, 'open MiMo task facade'):
                facade.for_environment(definition['spec'], definition['harness'])

    def test_task_view_borrows_owner_backend(self):
        with facade_fixture() as (owner, backend, _, definition):
            model = weakref.ref(backend.model)
            view = owner.for_environment(definition['spec'], definition['harness'])
            view.close()
            view.close()
            self.assertIsNone(view.model)
            self.assertIsNone(view.backend)
            self.assertIs(owner.model, model())
            self.assertEqual(backend.close_calls, 0)
            backend.guard()
            owner.close()
            self.assertIsNone(model())
            self.assertEqual(backend.close_calls, 1)

    def test_miner_close_does_not_close_borrowed_factory_backend(self):
        with facade_fixture(borrowed=True) as (facade, backend, _, definition):
            model = weakref.ref(backend.model)
            view = facade.for_environment(definition['spec'], definition['harness'])
            actor = object.__new__(Miner)
            actor.runtime = facade
            actor.runtimes = {'original': facade, 'view': view}
            actor.close()
            actor.close()
            self.assertIsNone(facade.backend)
            self.assertIsNone(view.backend)
            self.assertIsNotNone(model())
            self.assertEqual(backend.close_calls, 0)
            backend.guard()
            backend.close()  # The external factory remains the sole owner.
            self.assertIsNone(model())

    def test_epoch_turnover_releases_all_models_without_collecting_closed_actors(self):
        retained = []
        model_refs = []
        with tempfile.TemporaryDirectory() as directory:
            for replicas in (1, 4):
                for epoch in range(4):
                    with self.subTest(replicas=replicas, epoch=epoch), \
                         facade_fixture() as (facade, backend, manifest, definition):
                        model = weakref.ref(backend.model)
                        model_refs.append(model)
                        actor = object.__new__(Miner)
                        actor.manifest = manifest
                        actor.checkpoint = 'fixture'
                        actor.identity = SimpleNamespace(id='a' * 64)
                        journal = SimpleNamespace(close=Mock())
                        actor._group_journal = journal
                        pool = SimpleNamespace(_processes={}, shutdown=Mock())
                        with patch('subnet.mimo_parallel_mining.ProcessPoolExecutor', return_value=pool):
                            runtime = wrap_runtime(facade, actor=actor, replicas=replicas,
                                state_root=Path(directory)/f'{replicas}-{epoch}', admit=False)
                        views = [runtime.for_environment(definition['spec'], definition['harness'])
                                 for _ in range(2)]
                        actor.runtime = runtime
                        actor.runtimes = {'owner': runtime, 'one': views[0],
                                          'duplicate': views[0], 'two': views[1]}

                        def released_before_backend_cleanup():
                            self.assertIsNone(model())
                            journal.close.assert_not_called()
                        backend.on_close = released_before_backend_cleanup
                        actor.close()
                        actor.close()
                        self.assertEqual(backend.close_calls, 1)
                        self.assertIsNone(actor.runtime)
                        self.assertEqual(actor.runtimes, {})
                        journal.close.assert_called_once_with()
                        if replicas > 1:
                            pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
                            self.assertIsNone(runtime._replicas.actor)
                            self.assertIsNone(runtime._replicas.base)
                        retained.append((actor, runtime, facade, views, backend))
                        self.assertTrue(all(reference() is None for reference in model_refs))


if __name__ == '__main__':
    unittest.main()
