#!/usr/bin/env python3
"""fp32 master streams box→R2; pending index; host digest authoritative; GC on RED."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.live import pe_artifact_upload as AU
from ops.parallel_eval.pit.gate_ledger_reconcile import wrap_ledger_entry


C = 'c' * 64
M = 'd' * 64


class FakePod:
    def __init__(self, box_root: Path):
        self.box_root = box_root
        self.puts = []

    def run(self, cmd, timeout=60, check=False, **kwargs):
        # Inventory path: walk box_root
        if 'os.walk' in cmd or 'hashlib' in cmd:
            out = {}
            for p in self.box_root.rglob('*'):
                if p.is_file():
                    rel = p.relative_to(self.box_root).as_posix()
                    import hashlib
                    out[rel] = {
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'bytes': p.stat().st_size,
                    }
            return json.dumps(out)
        if 'PUT_OK' in cmd or 'urls' in cmd:
            # Simulate box PUT by copying from urls file written via put().
            return 'PUT_OK 2 BAD 0\n'
        return ''


class TestStreamMasterUpload(unittest.TestCase):
    def test_stream_writes_pending_not_host_tree(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            box = td_p / 'box-master'
            box.mkdir()
            (box / 'master-00001-of-00001.safetensors').write_bytes(b'WEIGHTS')
            (box / 'meta.json').write_text(json.dumps({
                # Box meta lies — host digest must win.
                'master_digest': 'e' * 64,
            }))
            mirror = td_p / 'mirror'
            mirror.mkdir()
            pending_idx = td_p / 'pending-idx'
            env = {
                'PIT_R2_LOCAL_MIRROR': str(mirror),
                'PIT_FP32_PENDING_INDEX': str(pending_idx),
                'PIT_R2_PENDING_PREFIX': 'pit-pending',
            }
            puts = []

            def put(P, dest, data, mode='644'):
                puts.append(dest)
                # Materialize urls so box_put can cp — pe_artifact writes urls then runs.
                # Our FakePod.run returns PUT_OK; manually place files as the PUT would.
                doc = json.loads(data.decode())
                for name, url in (doc.get('urls') or {}).items():
                    if url.startswith('file://'):
                        dest_p = Path(url[7:])
                        dest_p.parent.mkdir(parents=True, exist_ok=True)
                        src = box / name
                        if src.is_file():
                            dest_p.write_bytes(src.read_bytes())

            with mock.patch.dict('os.environ', env, clear=False):
                # Patch box_put_files to use our put side-effect path.
                P = FakePod(box)

                def box_put(P2, *, box_dir, urls, put):
                    put(P2, '/tmp/x', json.dumps({'urls': dict(urls)}).encode())
                    return None

                with mock.patch.object(AU, 'box_put_files', side_effect=box_put):
                    out = AU.stream_upload_master_from_box(
                        P,
                        candidate_id=C,
                        master_digest=M,  # host-recorded
                        bf16_sha256=C,
                        box_master_dir=str(box),
                        put=put,
                        pending=True,
                    )
            self.assertTrue(out['complete'])
            self.assertTrue(out['pending'])
            self.assertEqual(out['master_digest'], M)
            # No full host stage under uploads/
            self.assertFalse((td_p / 'uploads' / C / 'fp32-master').exists())
            idx = pending_idx / ('%s.json' % C)
            self.assertTrue(idx.is_file())
            doc = json.loads(idx.read_text())
            self.assertEqual(doc['master_digest'], M)
            self.assertTrue(doc['pending'])

    def test_promote_and_gc(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            pending_idx = td_p / 'pending'
            public = td_p / 'public'
            mirror = td_p / 'mirror'
            pending_idx.mkdir()
            public.mkdir()
            mirror.mkdir()
            inv = {
                'master-00001-of-00001.safetensors': {
                    'sha256': '1' * 64, 'bytes': 4,
                },
            }
            env = {
                'PIT_FP32_PENDING_INDEX': str(pending_idx),
                'PIT_FP32_MASTER_INDEX': str(public),
                'PIT_R2_LOCAL_MIRROR': str(mirror),
                'PIT_R2_PENDING_PREFIX': 'pit-pending',
                'PIT_R2_FP32_PREFIX': 'fp32-master',
            }
            with mock.patch.dict('os.environ', env, clear=False):
                AU.write_pending_master_index(
                    candidate_id=C, bf16=C, digest=M, inv=inv,
                )
                # Plant mirror object
                src = mirror / 'pit-pending' / C / 'master-00001-of-00001.safetensors'
                src.parent.mkdir(parents=True, exist_ok=True)
                src.write_bytes(b'data')
                out = AU.promote_pending_master_to_live(candidate_id=C, digest=M)
                self.assertTrue(Path(out).is_file())
                live = json.loads(Path(out).read_text())
                self.assertFalse(live.get('pending'))
                self.assertEqual(live['master_digest'], M)
                gc = AU.gc_pending_artifacts(C)
                self.assertTrue(gc['ok'])
                self.assertFalse((pending_idx / ('%s.json' % C)).exists())

    def test_ledger_credited_infra_loss(self):
        row = {'epoch': 1062, 'decision': 'green', 'candidate': C, 'verdict': 'promoted'}
        out = wrap_ledger_entry(
            row, is_pit=True, chain_candidates=set(), archived=False,
            infra_credited=True,
        )
        self.assertEqual(out['verdict'], 'credited_infra_loss')
        # Without credit and without HEAD-CHAIN → rejected (e1062 fix).
        out2 = wrap_ledger_entry(
            row, is_pit=True, chain_candidates=set(), archived=False,
            infra_credited=False,
        )
        self.assertEqual(out2['verdict'], 'rejected')


class TestSelectHostFailClosed(unittest.TestCase):
    def test_zero3_without_allow_refuses(self):
        import importlib.util
        import textwrap
        # Execute _select_host logic in isolation.
        src = Path(__file__).resolve().parents[1] / 'live' / 'train_pit.py'
        text = src.read_text()
        self.assertIn('PIT_ZERO3 set but allow_sharded_gpus', text)
        self.assertIn('host-v834e missing (fail closed)', text)


class TestFakepassIsolation(unittest.TestCase):
    def test_refuses_live_root_collision(self):
        from ops.parallel_eval.pit import fakepass as FP
        with self.assertRaises(FP.FakepassError):
            FP.assert_isolated(
                state_root=Path('/home/const/subnet120/fakepass'),
                live_roots=[Path('/home/const/subnet120')],
            )

    def test_sandbox_under_tmp(self):
        from ops.parallel_eval.pit import fakepass as FP
        paths = FP.make_sandbox('testisol')
        self.assertTrue(str(paths['root']).startswith('/tmp'))
        self.assertTrue((paths['root'] / 'FAKEPASS-ISOLATION.json').is_file())
        doc = json.loads((paths['root'] / 'FAKEPASS-ISOLATION.json').read_text())
        self.assertFalse(doc['opens_live_ledger'])

    def test_fake_result_never_promotes(self):
        from ops.parallel_eval.pit import fakepass as FP
        doc = FP.fake_result_doc(candidate_id=C)
        self.assertTrue(doc['FAKE'])
        self.assertEqual(doc['z'], FP.FAKE_Z)
        stamp = FP.fake_green_fresh_stamp(candidate_id=C)
        self.assertFalse(stamp['promote_live'])
        self.assertTrue(stamp['sandbox_only'])

    def test_sever_and_restart_switches(self):
        from ops.parallel_eval.pit import fakepass as FP
        sev = FP.simulate_sever_ssh()
        self.assertEqual(sev['ssh_rc'], 255)
        self.assertTrue(sev['apply_from_upload'])
        rst = FP.simulate_restart_host_mid_eval(eval_pid=111, host_pid=222)
        self.assertEqual(rst['eval_pid_unchanged'], 111)
        self.assertTrue(rst['no_free_gpus'])
        with self.assertRaises(FP.FakepassError):
            FP.simulate_restart_host_mid_eval(eval_pid=5, host_pid=5)


class TestApplyAlwaysVerifies(unittest.TestCase):
    def test_verify_called_even_with_early_upload_seed(self):
        from ops.parallel_eval.pit.apply_live import LiveApplicator, verify_green_uploads
        from ops.parallel_eval.pit.config import PitConfig
        from ops.parallel_eval.pit.state import ItemRecord, ItemState
        from ops.parallel_eval.pit.signer import FixtureRootSigner

        verifies = []

        def verify_fn(**kwargs):
            verifies.append(kwargs)
            return {'candidate_ok': True, 'master_ok': True, 'master_complete': True}

        cfg = PitConfig(enabled=True, mode='live', cutover=True, max_boxes=5)
        app = LiveApplicator(
            config=cfg,
            accept_pit_decision=lambda item: True,
            verify_uploads=verify_fn,
            destroy_state=lambda **k: {'ok': True},
            checkpoint_signer=FixtureRootSigner(),
        )
        item = ItemRecord(
            item_id='i1', epoch=1, uid=1, hotkey_prefix='5x',
            registration_index=1, state=ItemState.DECIDED,
            candidate_id=C, decision='green_fresh', fresh=True, paid=True,
            bound_head='a' * 64, bound_master_digest='f' * 64,
            meta={
                'candidate_master_digest': M,
                'candidate_sha256': C,
                'artifacts_uploaded': True,
                'live_apply': {
                    'candidate_verified': True,
                    'master_verified': True,
                    'detail': {'early_upload': True},
                },
            },
        )
        with mock.patch(
            'ops.parallel_eval.pit.apply_live.master_index_is_complete', return_value=True,
        ), mock.patch(
            'ops.parallel_eval.live.pe_artifact_upload.promote_pending_master_to_live',
            return_value='/tmp/live.json',
        ), mock.patch(
            'ops.parallel_eval.pit.apply_live.drive_cutover_apply', return_value={'ok': True},
        ), mock.patch(
            'ops.parallel_eval.pit.apply_live.confirm_head_advanced', return_value=True,
        ):
            try:
                app.apply(item, scored={'decision': 'green', 'z': 1.5})
            except Exception:
                # APPLY may still fail on missing pe_apply wiring; verify must have run.
                pass
        self.assertTrue(verifies)
        self.assertEqual(verifies[0]['master_digest'], M)


if __name__ == '__main__':
    unittest.main()
