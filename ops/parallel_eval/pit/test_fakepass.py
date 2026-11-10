#!/usr/bin/env python3
"""Forced-GREEN fakepass harness: isolation, deny-list, simulate, sever, restart."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.pit import fakepass as FP
from ops.parallel_eval.pit.pe_rent import PREFIX_FAKEPASS, is_fakepass_name


C = 'f' * 64


class TestFakepassIsolation(unittest.TestCase):
    def test_refuses_live_root_collision(self):
        with self.assertRaises(FP.FakepassError):
            FP.assert_isolated(
                state_root=Path('/home/const/subnet120/fakepass'),
                live_roots=[Path('/home/const/subnet120')],
            )

    def test_refuses_sandbox_path_under_live(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as td:
            root = Path(td)
            paths = {'root': root, 'pit': root / 'pit'}
            paths['pit'].mkdir()
            with self.assertRaises(FP.FakepassError):
                FP.assert_paths_outside_live(
                    paths,
                    live_roots=[root],  # sandbox path resolves under "live"
                )

    def test_sandbox_under_tmp_and_cleaned(self):
        paths = FP.make_sandbox('testisol')
        root = paths['root']
        try:
            self.assertTrue(str(root).startswith('/tmp') or str(root).startswith(tempfile.gettempdir()))
            self.assertTrue((root / 'FAKEPASS-ISOLATION.json').is_file())
            doc = json.loads((root / 'FAKEPASS-ISOLATION.json').read_text())
            self.assertFalse(doc['opens_live_ledger'])
            self.assertFalse(doc.get('promote_live', True))
        finally:
            FP.cleanup_sandbox(paths)
        self.assertFalse(root.exists())


class TestFakepassDenyList(unittest.TestCase):
    def test_env_denylist_scrubs_live_keys(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as td:
            root = Path(td)
            paths = {
                'root': root,
                'pit': root / 'pit',
                'pe': root / 'pe',
                'life': root / 'life',
                'uploads': root / 'uploads',
                'fp32_pending': root / 'fp32-pending',
                'fp32_public': root / 'fp32-master',
                'r2_mirror': root / 'r2-mirror',
                'bench': root / 'bench',
                'logs': root / 'logs',
            }
            for p in paths.values():
                p.mkdir(parents=True, exist_ok=True)
            with mock.patch.dict(os.environ, {
                'PE_FLAG': '/live/flag.json',
                'PE_HELDOUT': '/live/heldout',
                'PE_SYNC_EXPECTED_HASH': '/live/sync.py',
                'PE_SYNC_EXPECTED_HASH_PYTHON': '/usr/bin/python',
                'PE_SYNC_EXPECTED_HASH_LOG': '/live/sync.log',
                'HG_GATE_STATE': '/live/gate',
            }, clear=False):
                env = FP.env_for_sandbox(paths, 'deny1')
            for key in (
                'PE_FLAG', 'PE_HELDOUT', 'PE_SYNC_EXPECTED_HASH',
                'PE_SYNC_EXPECTED_HASH_PYTHON', 'PE_SYNC_EXPECTED_HASH_LOG',
                'HG_GATE_STATE',
            ):
                self.assertNotIn(key, env)
            self.assertEqual(env['PIT_R2_PENDING_PREFIX'], 'pit-fakepass/deny1')
            self.assertTrue(env['PIT_FAKEPASS'])

    def test_dry_isolation_check_cli(self):
        rc = FP.main(['--dry-isolation-check', '--run-id', 'drycli'])
        self.assertEqual(rc, 0)


class TestFakepassSimulateLocal(unittest.TestCase):
    def test_simulate_local_happy_path_no_tmp_leak(self):
        self.assertTrue(is_fakepass_name(PREFIX_FAKEPASS + '0-a0'))
        # Capture root via a hook before cleanup.
        roots: list[Path] = []
        real_make = FP.make_sandbox

        def tracking_make(run_id, **kwargs):
            paths = real_make(run_id, **kwargs)
            roots.append(paths['root'])
            return paths

        with mock.patch.object(FP, 'make_sandbox', side_effect=tracking_make):
            rc = FP.main([
                '--simulate-local',
                '--run-id', 'simhappy',
                '--hold-vllm-s', '0',
                '--liveness-interval-s', '0.05',
                '--candidate-id', C,
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(roots)
        self.assertFalse(roots[0].exists(), 'sandbox /tmp must be cleaned')

    def test_fake_result_never_promotes(self):
        doc = FP.fake_result_doc(candidate_id=C)
        self.assertTrue(doc['FAKE'])
        self.assertEqual(doc['z'], FP.FAKE_Z)
        stamp = FP.fake_green_fresh_stamp(candidate_id=C)
        self.assertFalse(stamp['promote_live'])
        self.assertTrue(stamp['sandbox_only'])


class TestFakepassSeverAfterGreen(unittest.TestCase):
    def test_sever_after_green_apply_from_upload(self):
        run_id = 'sever1'
        live_roots = FP.live_roots_from_env()
        paths = FP.make_sandbox(run_id, live_roots=live_roots)
        env = FP.env_for_sandbox(paths, run_id)
        run = FP.FakepassRun(
            run_id=run_id,
            mode=FP.FakepassMode.SIMULATE,
            paths=paths,
            env=env,
            live_roots=live_roots,
            hold_vllm_s=0,
            sever_after_green=True,
            candidate_id=C,
            liveness_interval_s=0.05,
        )
        try:
            out = run.run()
            self.assertTrue(out['ok'])
            self.assertTrue(out['sever_after_green'])
            self.assertTrue(out['apply_result']['apply_from_upload'])
            self.assertFalse(out['promote_live'])
            self.assertFalse(out['green_stamp']['promote_live'])
            # Sever model returns rc 255.
            sev = FP.simulate_sever_ssh()
            self.assertEqual(sev['ssh_rc'], 255)
            self.assertTrue(sev['apply_from_upload'])
            # Prove Pod.run is severed after green.
            self.assertIsInstance(run.pod, FP.SeveredPod)
            self.assertTrue(run.pod.severed)
            with self.assertRaises(FP.FakepassError):
                run.pod.run('true', check=False)
        finally:
            if not run.cleaned:
                FP.cleanup_sandbox(paths)
        self.assertFalse(paths['root'].exists())


class TestFakepassRestartHostMidEval(unittest.TestCase):
    def test_restart_host_mid_eval_pid_invariant(self):
        run_id = 'rehost1'
        live_roots = FP.live_roots_from_env()
        paths = FP.make_sandbox(run_id, live_roots=live_roots)
        env = FP.env_for_sandbox(paths, run_id)
        run = FP.FakepassRun(
            run_id=run_id,
            mode=FP.FakepassMode.SIMULATE,
            paths=paths,
            env=env,
            live_roots=live_roots,
            hold_vllm_s=0,
            restart_host_mid_eval=True,
            candidate_id=C,
            liveness_interval_s=0.05,
        )
        try:
            out = run.run()
            self.assertTrue(out['ok'])
            rst = out['restart_result']
            self.assertIsNotNone(rst)
            self.assertTrue(rst['no_free_gpus'])
            self.assertTrue(rst['no_place_cand'])
            self.assertTrue(rst['eval_alive'])
            self.assertIsInstance(rst['eval_pid_unchanged'], int)
            self.assertNotEqual(rst['eval_pid_unchanged'], rst['killed_host_pid'])
            self.assertFalse(run.called_free_gpus)
            self.assertFalse(run.called_place_cand)
            with self.assertRaises(FP.FakepassError):
                FP.simulate_restart_host_mid_eval(eval_pid=5, host_pid=5)
        finally:
            if not run.cleaned:
                FP.cleanup_sandbox(paths)
        self.assertFalse(paths['root'].exists())


class TestFakepassPrefix(unittest.TestCase):
    def test_box_prefix_matches_pe_rent(self):
        self.assertEqual(FP.BOX_NAME_PREFIX, PREFIX_FAKEPASS)
        self.assertTrue(is_fakepass_name(FP.BOX_NAME_PREFIX + 'x'))


if __name__ == '__main__':
    unittest.main()
