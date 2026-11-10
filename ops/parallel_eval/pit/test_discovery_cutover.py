"""Cutover discovery binding: PIT heartbeat replaces dead PE controller PID.

Under ``parallel_eval.pit.cutover`` / ``WARM-OFF``, publish_mimo_public_discovery
accepts a fresh affine-pit-host heartbeat when the lifecycle/controller unit is
not running. Other pin/signature checks stay fail-closed. Cutover off keeps the
controller PID requirement.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from ops import publish_mimo_public_discovery as observer
from ops.parallel_eval.pit.run_host import write_host_heartbeat


class _Policy:
    mode = 'public'
    policy_sha256 = 'p' * 64
    scope = {
        'immutable_manifest': {
            'source_bundle': {'sha256': 's' * 64},
        },
    }

    def __init__(self, *_a, **_k):
        pass


class DiscoveryCutoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pe_state = self.root / 'pe-state'
        self.pit = self.pe_state / 'pit'
        self.pit.mkdir(parents=True)
        self.flag = self.root / 'flag.json'
        self.addCleanup(self.tmp.cleanup)

    def _env(self, *, cutover: bool | None = True, warm_off: bool = False):
        if warm_off:
            (self.pe_state / 'WARM-OFF').write_text('{}')
        if cutover is None:
            flag_body = {}
        else:
            flag_body = dict(
                parallel_eval=dict(
                    enabled=True,
                    pit=dict(enabled=True, mode='live', cutover=bool(cutover)),
                ),
            )
        self.flag.write_text(json.dumps(flag_body))
        return {
            'PE_FLAG': str(self.flag),
            'PE_STATE': str(self.pe_state),
            'PIT_STATE': str(self.pit),
        }

    def _write_heartbeat(self, *, at: float | None = None):
        at = time.time() if at is None else at
        return write_host_heartbeat(self.pit, now=at)

    def _selection(self):
        entry = self.root / 'ops' / 'mimo_lifecycle.py'
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text('# entry\n')
        cfg = self.root / 'lifecycle.private.json'
        cfg.write_text('{}')
        cfg.chmod(0o600)
        return dict(
            enabled=True,
            authority='test-authority',
            lifecycle_unit='pe-controller.service',
            lifecycle_entry=dict(path=str(entry), sha256='e' * 64),
            lifecycle_config=dict(path=str(cfg), sha256='c' * 64),
            source_sha256='s' * 64,
        )

    def _run_selector(self, env, *, inspect='MainPID=0\nActiveState=inactive\n'):
        selection = self._selection()
        config = dict(
            enabled=True,
            release_policy={},
            adapters=dict(source_files={'ops/mimo_lifecycle.py': 'x'}),
        )
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(observer, '_pinned_reference', side_effect=lambda ref: Path(ref['path'])):
                with mock.patch.object(observer, 'private_json', return_value=config):
                    with mock.patch.object(observer, 'ReleasePolicy', _Policy):
                        with mock.patch(
                            'subnet.mimo_worker.source_inventory',
                            return_value=config['adapters']['source_files'],
                        ):
                            return observer.selector(
                                selection,
                                proc=self.root / 'proc',
                                inspect_unit=lambda: inspect,
                            )

    def test_cutover_fresh_pit_heartbeat_accepts_without_controller(self):
        env = self._env(cutover=True)
        now = time.time()
        self._write_heartbeat(at=now)
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertTrue(observer.cutover_active())
            self.assertTrue(observer.pit_executor_live(now=now))
        _config, _policy, record = self._run_selector(env)
        self.assertEqual(record.get('executor'), 'pit')
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertTrue(observer.process_live(record, now=now))

    def test_cutover_without_pit_heartbeat_fails_closed(self):
        env = self._env(cutover=None, warm_off=True)
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertTrue(observer.cutover_active())
            self.assertFalse(observer.pit_executor_live())
        with self.assertRaisesRegex(ValueError, 'not running'):
            self._run_selector(env)

    def test_cutover_off_still_requires_controller_pid(self):
        env = self._env(cutover=False)
        now = time.time()
        self._write_heartbeat(at=now)
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertFalse(observer.cutover_active())
            self.assertTrue(observer.pit_executor_live(now=now))
        with self.assertRaisesRegex(ValueError, 'not running'):
            self._run_selector(env)


if __name__ == '__main__':
    unittest.main()
