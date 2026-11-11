#!/usr/bin/env python3
"""Forced-GREEN fakepass harness: isolation, deny-list, simulate, sever, restart.

REAL-path tests drive the production code (live-source resolution, rent
policy, LiveApplicator → pe_apply/pe_live → lifecycle → record_applied,
hash-sync, promoted-index checks) against sandbox state and moto R2. The
rented-box end-to-end run needs Lium + R2 credentials and an explicit opt-in.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Dict, Iterator
from unittest import mock

try:
    import boto3
    from moto import mock_aws
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore
    mock_aws = None  # type: ignore

from ops.parallel_eval.live import pe_artifact_upload as AU
from ops.parallel_eval.live import pe_box
from ops.parallel_eval.pit import fakepass as FP
from ops.parallel_eval.pit import fakepass_real as R
from ops.parallel_eval.pit.fakepass_sandbox import SANDBOX_REBIND, lium_credential_source
from ops.parallel_eval.pit.pe_rent import PREFIX_FAKEPASS, RentError, is_fakepass_name


C = 'f' * 64
HEAD = 'a' * 64
HEAD_BF16 = 'b' * 64
HEAD_MASTER = 'c' * 64
LIVE_HEAD = '7' * 64
BUCKET = 'affine-fakepass-moto'
_SKIP_NO_MOTO = unittest.skipIf(mock_aws is None, 'moto not installed')


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wj(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True) + '\n')
    return path


def _epoch(life: Path, n: int, *, head: str = HEAD, signed: bool = True) -> Path:
    """One lifecycle epoch. Live finalizes every epoch as no-update right after training."""
    ep = life / 'epochs' / str(n)
    _wj(ep / 'TRAINING-JOB.json', {'payload': {
        'manifest': {'payload': {'checkpoint': {'id': head, 'files': {'model.safetensors': '1' * 64}}}},
        'training': {'config': {'lr': 1e-6}, 'optimizer_step': 40 + n},
    }})
    _wj(ep / 'TRAIN-DISPATCH-FF.json', {
        'signature': 'sig' if signed else '',
        'signer': 'ROOT' if signed else 'SHADOW-UNSIGNED',
        'payload': {'epoch': 'run-e%06d' % n, 'checkpoint': head},
    })
    _wj(ep / 'FINALIZED.json', {'payload': {'input_checkpoint': head, 'result': {'status': 'no-update'}}})
    return ep


def _public_turn(turns: Path, ep: Path, *, decision: str, candidate: str = C) -> Path:
    """The public record of one decided epoch, in the shape k1_transparency.publish_ff_turn writes.

    The batch file is published for a GREEN only; a RED keeps it private.
    """
    n = int(ep.name)
    head = json.loads((ep / 'TRAINING-JOB.json').read_text())['payload']['manifest']['payload']['checkpoint']['id']
    green = decision == 'green'
    return _wj(turns / str(n) / 'turn.json', {
        'schema': 'sn120-k1-transparency-v1',
        'epoch': n,
        'format': 'affine-batch-v2',
        'promoted': green,
        'decision': {'decision': decision, 'head': head, 'candidate': candidate, 'decided_at': 1000.0 + n,
                     'mode': 'live', 'z_min': 1.0, 'z_inclusive': False},
        'batch': ({'published': True, 'url': '/k1/turns/%d/batch.json' % n, 'sha256': '5' * 64, 'bytes': 1}
                  if green else {'published': False, 'note': 'RED: the miner file stays private.'}),
        'dispatch': {
            'envelope_file_sha256': _sha((ep / 'TRAIN-DISPATCH-FF.json').read_bytes()),
            'signer': 'ROOT',
            'header': {'checkpoint': head, 'epoch': 'run-e%06d' % n},
        },
        'candidate': {'id': candidate, 'promoted': green},
    })


def _live_fixture(root: Path) -> Dict[str, str]:
    """Read-only stand-in for the host live tree and the public record.

    Epoch 10 trained from HEAD and went GREEN: its candidate LIVE_HEAD is the
    live head now, and its batch is public. Epochs 11-13 train from LIVE_HEAD:
    11 is unsigned, 12 is decided RED (private batch), 13 is still in flight.
    Epoch 9 is an older GREEN whose input master is not published.
    """
    life = root / 'live-life'
    pe = root / 'live-pe'
    idx = root / 'live-fp32-master'
    turns = root / 'live-web' / 'k1' / 'turns'
    _wj(life / 'CURRENT.json', {
        'epoch_index': 14,
        'checkpoint': {'id': LIVE_HEAD},
        'optimizer_step': 52,
        'checkpoint_authorization': {'input': HEAD, 'id': LIVE_HEAD},
    })
    _public_turn(turns, _epoch(life, 9, head='9' * 64), decision='green', candidate=HEAD)
    _public_turn(turns, _epoch(life, 10), decision='green', candidate=LIVE_HEAD)
    _public_turn(turns, _epoch(life, 11, head=LIVE_HEAD, signed=False), decision='green')
    _public_turn(turns, _epoch(life, 12, head=LIVE_HEAD), decision='red')
    _epoch(life, 13, head=LIVE_HEAD)
    _wj(idx / ('%s.json' % HEAD), {
        'bf16_sha256': HEAD_BF16,
        'master_digest': HEAD_MASTER,
        'files': {
            'model-00001-of-00001.safetensors': {'sha256': '1' * 64, 'bytes': 1, 'r2_key': 'fp32-master/h/s'},
            'meta.json': {'sha256': '2' * 64, 'bytes': 1, 'r2_key': 'fp32-master/h/meta.json'},
        },
        'meta_json': {'r2_key': 'fp32-master/h/meta.json'},
    })
    _wj(pe / 'BUNDLE.json', {'bundle': 'probe'})
    return {'PE_LIFE': str(life), 'PE_STATE': str(pe), 'PIT_FP32_MASTER_INDEX': str(idx),
            'PIT_PUBLIC_TURNS': str(turns)}


def _offline(url: str) -> bytes:
    raise OSError('offline: %s' % url)


@contextlib.contextmanager
def _fresh_live_modules() -> Iterator[None]:
    """Stub the host-only Lium library and restore sys.modules afterwards."""
    rp = types.ModuleType('replace_pod')

    class _Pod:
        pass

    rp.Pod = _Pod
    names = R.LIVE_BOUND_MODULES + ('replace_pod',)
    prev = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    sys.modules['replace_pod'] = rp
    try:
        yield
    finally:
        for n, mod in prev.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


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


class TestRealModeIsReal(unittest.TestCase):
    def test_fakepassrun_refuses_real(self):
        with self.assertRaises(FP.FakepassError):
            FP.FakepassRun(
                run_id='x', mode=FP.FakepassMode.REAL, paths={}, env={}, live_roots=[],
            )

    def test_real_cli_without_creds_refuses_before_sandbox(self):
        with tempfile.TemporaryDirectory() as home:
            env = {'HOME': home, 'PATH': os.environ.get('PATH', '')}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(FP.main(['--real', '--run-id', 'nocreds']), 2)
        self.assertEqual(list(Path(tempfile.gettempdir()).glob('pit-fakepass-nocreds-*')), [])

    def test_real_does_not_use_scripted_box_io(self):
        src = Path(R.__file__).read_text()
        self.assertNotIn('ScriptedBoxIO', src)
        self.assertIn('HostBoxIO().train_v834e', src)
        self.assertIn('pe_box._early_upload_artifacts', src)


class TestRealLiveSources(unittest.TestCase):
    def _refusal(self, env, **kw) -> str:
        with self.assertRaises(FP.FakepassError) as ctx:
            R.resolve_live_sources(env, fetch=_offline, **kw)
        return str(ctx.exception)

    def test_picks_newest_decided_public_epoch_not_the_live_head_epoch(self):
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            src = R.resolve_live_sources(env, fetch=_offline)
            # Epochs 13 (in flight), 12 (RED) and 11 (unsigned) train from the live head: all skipped.
            self.assertEqual(src.epoch, 10)
            self.assertEqual(src.head, HEAD)
            self.assertEqual(src.live_head, LIVE_HEAD)
            self.assertEqual(src.decision['decision'], 'green')
            self.assertEqual(src.decision['candidate'], LIVE_HEAD)
            self.assertEqual(src.master_digest, HEAD_MASTER)
            self.assertEqual(src.head_bf16, HEAD_BF16)
            self.assertTrue(src.master_index_origin.endswith('%s.json' % HEAD))
            turn = Path(env['PIT_PUBLIC_TURNS']) / '10' / 'turn.json'
            self.assertEqual(src.public_turn_origin, str(turn))
            self.assertEqual(src.provenance[str(turn)], _sha(turn.read_bytes()))
            self.assertEqual(len(src.provenance), 5)

    def test_explicit_epoch_must_be_signed_decided_and_public(self):
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            self.assertEqual(R.resolve_live_sources(env, epoch=10, fetch=_offline).epoch, 10)
            self.assertIn('ROOT-signed', self._refusal(env, epoch=11))
            self.assertIn('decided red: the batch is not published', self._refusal(env, epoch=12))
            self.assertIn('no public turn record', self._refusal(env, epoch=13))
            self.assertIn('M(H) unpublished', self._refusal(env, epoch=9))
            self.assertIn('ROOT-signed', self._refusal(env, epoch=77))

    def test_public_record_must_name_the_local_signed_dispatch(self):
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            turn = Path(env['PIT_PUBLIC_TURNS']) / '10' / 'turn.json'
            good = json.loads(turn.read_text())
            for label, edit in (
                ('does not name the local signed dispatch', lambda d: d['dispatch'].update(envelope_file_sha256='0' * 64)),
                ('names epoch 12', lambda d: d.update(epoch=12)),
                ('no final decision', lambda d: d['decision'].update(decision='pending')),
                ('the batch is not published', lambda d: d['batch'].update(published='yes')),
            ):
                doc = json.loads(json.dumps(good))
                edit(doc)
                _wj(turn, doc)
                self.assertIn(label, self._refusal(env, epoch=10))
            turn.write_text('not json')
            self.assertIn('not a JSON object', self._refusal(env, epoch=10))
            # No usable epoch is left: the automatic pick refuses and points at --epoch.
            why = self._refusal(env)
            self.assertIn('no decided public epoch', why)
            self.assertIn('--epoch', why)
            self.assertIn('e9: M(H) unpublished', why)

    def test_public_record_falls_back_to_public_url(self):
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            turns = Path(env.pop('PIT_PUBLIC_TURNS'))
            urls = {R.PUBLIC_TURN_URL % int(p.parent.name): p.read_bytes() for p in turns.glob('*/turn.json')}
            seen = []

            def fetch(url):
                seen.append(url)
                if url not in urls:
                    raise OSError('404')
                return urls[url]

            src = R.resolve_live_sources(env, fetch=fetch)
            self.assertEqual(src.epoch, 10)
            self.assertEqual(src.public_turn_origin, R.PUBLIC_TURN_URL % 10)
            # Unsigned epoch 11 costs no lookup; the M(H) index is local.
            self.assertEqual(seen, [R.PUBLIC_TURN_URL % n for n in (13, 12, 10)])
            self.assertEqual(src.provenance[R.PUBLIC_TURN_URL % 10], _sha(urls[R.PUBLIC_TURN_URL % 10]))

    def test_automatic_pick_bounds_public_lookups(self):
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            life = Path(env['PE_LIFE'])
            env.pop('PIT_PUBLIC_TURNS')
            for n in range(14, 14 + R.PUBLIC_SCAN_LIMIT + 5):
                _epoch(life, n, head=LIVE_HEAD)
            seen = []

            def fetch(url):
                seen.append(url)
                raise OSError('404')

            with self.assertRaises(FP.FakepassError) as ctx:
                R.resolve_live_sources(env, fetch=fetch)
            self.assertEqual(len(seen), R.PUBLIC_SCAN_LIMIT)
            self.assertIn('newest %d signed epochs' % R.PUBLIC_SCAN_LIMIT, str(ctx.exception))

    def test_index_falls_back_to_public_url(self):
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            doc = json.loads((Path(env['PIT_FP32_MASTER_INDEX']) / ('%s.json' % HEAD)).read_text())
            env.pop('PIT_FP32_MASTER_INDEX')
            seen = []

            def fetch(url):
                seen.append(url)
                return json.dumps(doc).encode()

            src = R.resolve_live_sources(env, fetch=fetch)
            self.assertEqual(seen, [R.PUBLIC_MASTER_URL % HEAD])
            self.assertEqual(src.master_index_origin, seen[0])

    def test_missing_bundle_or_index_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            (Path(env['PE_STATE']) / 'BUNDLE.json').unlink()
            with self.assertRaises(FP.FakepassError):
                R.resolve_live_sources(env)
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            env.pop('PIT_FP32_MASTER_INDEX')

            def unpublished(_url):
                raise OSError('404')

            with self.assertRaises(FP.FakepassError):
                R.resolve_live_sources(env, fetch=unpublished)

    def test_stage_copies_inputs_without_touching_live(self):
        with tempfile.TemporaryDirectory() as td:
            env = _live_fixture(Path(td))
            before = {p: p.read_bytes() for p in Path(td).rglob('*') if p.is_file()}
            src = R.resolve_live_sources(env)
            paths = FP.make_sandbox('stage1', live_roots=FP.live_roots_from_env(env))
            try:
                staged = R.stage_live_sources(src, paths)
                cur = json.loads((paths['life'] / 'CURRENT.json').read_text())
                self.assertEqual(cur['checkpoint']['id'], HEAD)
                self.assertEqual(cur['epoch_index'], 10)
                self.assertEqual(cur['optimizer_step'], 50)
                # The live authorization names LIVE_HEAD; it does not follow the run to the earlier HEAD.
                self.assertIsNone(cur['checkpoint_authorization'])
                chain = json.loads((paths['pe'] / 'HEAD-CHAIN.json').read_text())
                self.assertEqual(chain, {'genesis': HEAD, 'applied': []})
                flag = json.loads(staged['flag'].read_text())
                self.assertFalse(flag['parallel_eval']['pit']['allow_sharded_gpus'])
                self.assertTrue((paths['fp32_public'] / ('%s.json' % HEAD)).is_file())
                self.assertTrue(staged['hash_sync'].is_file())
            finally:
                FP.cleanup_sandbox(paths)
            after = {p: p.read_bytes() for p in Path(td).rglob('*') if p.is_file()}
            self.assertEqual(before, after)


class TestRealEnvAndPreflight(unittest.TestCase):
    def _sandbox(self, run_id):
        paths = FP.make_sandbox(run_id)
        self.addCleanup(FP.cleanup_sandbox, paths)
        return paths

    def test_real_env_rebinds_deny_keys_into_sandbox(self):
        paths = self._sandbox('renv1')
        live = {
            'PE_FLAG': '/live/flag.json', 'PE_HELDOUT': '/live/heldout',
            'PE_SYNC_EXPECTED_HASH': '/live/sync.py', 'HG_GATE_STATE': '/live/gate',
            'PIT_SYNC_EXPECTED_HASH': '/live/pit-sync.py',
            'PIT_PRESIGN_PUT_JSON': '/tmp/inject.json',
        }
        with mock.patch.dict(os.environ, live, clear=False):
            env = R.real_env_for_sandbox(FP.env_for_sandbox(paths, 'renv1'), paths)
        root = paths['root']
        for key in SANDBOX_REBIND:
            self.assertTrue(env[key].startswith(str(root)), key)
        self.assertNotIn('PE_HELDOUT', env)
        self.assertNotIn('PIT_SYNC_EXPECTED_HASH', env)
        self.assertNotIn('PIT_R2_LOCAL_MIRROR', env)
        self.assertNotIn('PIT_PRESIGN_PUT_JSON', env)
        self.assertEqual(env['PIT_FAKEPASS_MODE'], 'real')
        self.assertEqual(env['PIT_R2_PENDING_PREFIX'], 'pit-fakepass/renv1')

    def test_assert_env_sandboxed_refuses_live_paths(self):
        paths = self._sandbox('renv2')
        env = R.real_env_for_sandbox(FP.env_for_sandbox(paths, 'renv2'), paths)
        for key, value in (
            ('PE_FLAG', '/home/const/subnet120/flag.json'),
            ('PE_SYNC_EXPECTED_HASH_LOG', '/var/log/sync.log'),
            ('PIT_FP32_MASTER_INDEX', '/srv/website/k1/fp32-master'),
            ('PE_HELDOUT', str(paths['root']) + '/../escape'),
        ):
            with self.assertRaises(FP.FakepassError, msg=key):
                R.assert_env_sandboxed(dict(env, **{key: value}), paths['root'])

    def test_preflight_refuses_injection_and_overlay(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _wj(Path(td) / 'life.json', {'bucket': {}})
            base = {'LIUM_API_KEY': 'x', 'PIT_LIFE_CFG': str(cfg), 'PIT_BUNDLE_JSON': str(cfg)}
            for key in R.INJECTION_ENV + R.OVERLAY_ENV:
                with self.assertRaises(FP.FakepassError, msg=key):
                    R.preflight_real(dict(base, **{key: '1'}))
            with self.assertRaises(FP.FakepassError):
                R.preflight_real({k: v for k, v in base.items() if k != 'LIUM_API_KEY'})
            with self.assertRaises(FP.FakepassError):
                R.preflight_real(dict(base, PIT_BUNDLE_JSON=str(Path(td) / 'missing')))

    def test_preflight_accepts_standard_lium_config_ini(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _wj(Path(td) / 'life.json', {'bucket': {}})
            ini = Path(td) / '.lium' / 'config.ini'
            ini.parent.mkdir()
            ini.write_text('[api]\napi_key = sk_fixture_key\n')
            env = {
                'HOME': td, 'PATH': os.environ.get('PATH', ''),
                'PIT_LIFE_CFG': str(cfg), 'PIT_BUNDLE_JSON': str(cfg),
                'PIT_V834E_RELEASE': str(Path(td) / 'no-release'),
            }
            with mock.patch.dict(os.environ, env, clear=True):
                # Past the credential gate: the next refusal is the trainer host dir.
                with self.assertRaisesRegex(FP.FakepassError, 'trainer-host-v834e dir missing'):
                    R.preflight_real(os.environ)
                ini.write_text('[api]\napi_key =\n')
                with self.assertRaisesRegex(FP.FakepassError, 'Lium credentials missing'):
                    R.preflight_real(os.environ)

    def test_import_live_modules_binds_sandbox_roots(self):
        paths = self._sandbox('renv3')
        with tempfile.TemporaryDirectory() as td:
            src = R.resolve_live_sources(_live_fixture(Path(td)))
            R.stage_live_sources(src, paths)
        env = R.real_env_for_sandbox(FP.env_for_sandbox(paths, 'renv3'), paths)
        with _fresh_live_modules(), R.swapped_environ(env):
            mods = R.import_live_modules()
            bound = R.assert_live_modules_sandboxed(mods, paths['root'])
            self.assertTrue(all(v.startswith(str(paths['root'])) for v in bound.values()), bound)
            with self.assertRaises(FP.FakepassError):
                R.import_live_modules()
        self.assertNotEqual(os.environ.get('PIT_FAKEPASS_MODE'), 'real')


class TestLiumCredentialSources(unittest.TestCase):
    """Nov 10: preflight refused a host that only has the standard ~/.lium/config.ini."""

    KEY = 'sk_fixture_not_a_real_key'

    def _home(self, ini=None) -> str:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        if ini is not None:
            cfg = Path(td.name) / '.lium' / 'config.ini'
            cfg.parent.mkdir()
            cfg.write_text(ini)
        return td.name

    def test_standard_config_ini_api_key(self):
        env = {'HOME': self._home('[api]\napi_key = %s\n' % self.KEY)}
        self.assertEqual(lium_credential_source(env), 'config.ini:[api] api_key')
        self.assertTrue(FP.has_lium_credentials(env))

    def test_env_keys_follow_cli_order(self):
        home = self._home('[api]\napi_key = %s\n' % self.KEY)
        self.assertEqual(
            lium_credential_source({'HOME': home, 'LIUM_API_KEY': 'a', 'LIUM_API_API_KEY': 'b'}),
            'env:LIUM_API_API_KEY',
        )
        self.assertEqual(lium_credential_source({'HOME': home, 'LIUM_API_KEY': 'a'}), 'env:LIUM_API_KEY')

    def test_stored_default_workspace_key(self):
        ini = '[workspaces]\nactive = Team\n\n[workspace.team]\napi_key = %s\n' % self.KEY
        self.assertEqual(
            lium_credential_source({'HOME': self._home(ini)}),
            'config.ini:[workspace.team] api_key',
        )

    def test_requested_workspace_needs_its_own_saved_key(self):
        ini = '[api]\napi_key = %s\n\n[workspace.team]\napi_key = %s\n' % (self.KEY, self.KEY)
        home = self._home(ini)
        self.assertEqual(
            lium_credential_source({'HOME': home, 'LIUM_WORKSPACE': 'Team'}),
            'config.ini:[workspace.team] api_key',
        )
        # The CLI refuses here instead of using another workspace's key.
        self.assertIsNone(lium_credential_source({'HOME': home, 'LIUM_WORKSPACE': 'other', 'LIUM_API_KEY': 'a'}))

    def test_keyless_or_broken_config_is_not_a_credential(self):
        for ini in ('', '[api]\napi_key =\n', '[ssh]\nkey_path = ~/.ssh/id_ed25519\n', 'not an ini file\n'):
            env = {'HOME': self._home(ini)}
            self.assertIsNone(lium_credential_source(env), ini)
            self.assertFalse(FP.has_lium_credentials(env), ini)
        self.assertFalse(FP.has_lium_credentials({'HOME': self._home()}))

    def test_label_never_contains_the_key(self):
        home = self._home('[api]\napi_key = %s\n' % self.KEY)
        for env in ({'HOME': home}, {'HOME': home, 'LIUM_API_KEY': self.KEY}):
            self.assertNotIn(self.KEY, lium_credential_source(env))


class TestRealAdmittedHost(unittest.TestCase):
    def _host(self, td: Path):
        host = td / 'host-v834e'
        files = {'v83_dp.py': b'dp\n', 'patched/mimo_grpo.service.py': b'svc\n', 'math_levels.json': b'{}\n'}
        for rel, data in files.items():
            (host / rel).parent.mkdir(parents=True, exist_ok=True)
            (host / rel).write_bytes(data)
        (host / R.ROOT_ADMISSION).write_text('{}\n')
        manifest = _wj(td / 'admitted.json', {'host_files': {k: _sha(v) for k, v in files.items()}})
        return host, manifest

    def test_byte_identical_host_passes(self):
        with tempfile.TemporaryDirectory() as td:
            host, manifest = self._host(Path(td))
            out = R.check_admitted_trainer_host(host, manifest)
            self.assertFalse(out['overlay'])
            self.assertEqual(out['admitted_files'], 3)

    def test_drift_overlay_or_missing_root_admission_refuses(self):
        for mutate in (
            lambda h: (h / 'v83_dp.py').write_bytes(b'dp-patched\n'),
            lambda h: (h / 'v835_zero3.py').write_text('# overlay\n'),
            lambda h: (h / R.ROOT_ADMISSION).unlink(),
            lambda h: (h / 'patched/mimo_grpo.service.py').unlink(),
        ):
            with tempfile.TemporaryDirectory() as td:
                host, manifest = self._host(Path(td))
                mutate(host)
                with self.assertRaises(FP.FakepassError):
                    R.check_admitted_trainer_host(host, manifest)

    def test_committed_manifest_matches_admitted_inventory(self):
        doc = json.loads(R.ADMITTED_MANIFEST.read_text())
        self.assertIn('v83_dp.py', doc['host_files'])
        self.assertFalse(any('zero3' in k for k in doc['host_files']))


class TestTrainerHostEnvEqualsSignedDefault(unittest.TestCase):
    """Nov 10: PIT_TRAINER_HOST naming the signed default was refused as an overlay."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.root = Path(td.name)
        self.release = self.root / 'release-v834e-signed'
        self.signed = self.release / 'trainer-host-v834e'
        # The real admitted host files; the committed manifest pins their bytes.
        shutil.copytree(R.REPO / 'miner' / 'validator' / 'trainer-v834e', self.signed)
        (self.signed / R.ROOT_ADMISSION).write_text('{}\n')
        cfg = _wj(self.root / 'life.json', {'bucket': {}})
        self.env = {
            'HOME': str(self.root), 'PATH': os.environ.get('PATH', ''), 'LIUM_API_KEY': 'x',
            'PIT_LIFE_CFG': str(cfg), 'PIT_BUNDLE_JSON': str(cfg),
            'PIT_V834E_RELEASE': str(self.release),
        }

    def _preflight(self, **extra):
        with mock.patch.dict(os.environ, dict(self.env, **extra), clear=True):
            return R.preflight_real(os.environ)

    def test_unset_and_equal_ship_the_same_admitted_dir(self):
        base = self._preflight()
        self.assertEqual(base['trainer_host_env'], [])
        self.assertEqual(base['lium_credentials'], 'env:LIUM_API_KEY')
        for key in R.TRAINER_HOST_ENV:
            out = self._preflight(**{key: str(self.signed)})
            self.assertEqual(out['host_dir'], base['host_dir'])
            self.assertEqual(out['admitted_files'], base['admitted_files'])
            self.assertEqual(out['trainer_host_env'], [key])
            self.assertFalse(out['overlay'])

    def test_symlink_and_dotted_spellings_of_the_signed_default_are_equal(self):
        link = self.root / 'link-to-signed'
        link.symlink_to(self.signed)
        for spelling in (link, self.release / '.' / 'trainer-host-v834e', str(self.signed) + '/'):
            self.assertTrue(self._preflight(PIT_TRAINER_HOST=str(spelling))['admitted_files'])

    def test_any_other_dir_still_refuses(self):
        copy = self.root / 'byte-identical-copy'
        shutil.copytree(self.signed, copy)
        for bad in (copy, self.release, self.root / 'missing', '1'):
            with self.assertRaisesRegex(FP.FakepassError, 'overlay', msg=str(bad)):
                self._preflight(PIT_TRAINER_HOST=str(bad))

    def test_equal_path_never_skips_the_admitted_inventory_check(self):
        for mutate, why in (
            (lambda h: (h / 'v83_dp.py').write_bytes(b'patched\n'), 'drifted'),
            (lambda h: (h / 'v835_zero3.py').write_text('# overlay\n'), 'non-admitted overlay'),
            (lambda h: (h / R.ROOT_ADMISSION).unlink(), 'ROOT-ADMITTED'),
        ):
            mutate(self.signed)
            with self.assertRaisesRegex(FP.FakepassError, why):
                self._preflight(PIT_TRAINER_HOST=str(self.signed))
            shutil.rmtree(self.signed)
            shutil.copytree(R.REPO / 'miner' / 'validator' / 'trainer-v834e', self.signed)
            (self.signed / R.ROOT_ADMISSION).write_text('{}\n')

    def test_other_overlay_switches_refuse_next_to_an_equal_trainer_host(self):
        others = [k for k in R.OVERLAY_ENV if k not in R.TRAINER_HOST_ENV]
        self.assertTrue(others)
        for key in others:
            with self.assertRaisesRegex(FP.FakepassError, 'overlay', msg=key):
                self._preflight(PIT_TRAINER_HOST=str(self.signed), **{key: '1'})


class _FakeRent:
    __name__ = 'fake_pe_rent'

    def __init__(self, offers, rented=None):
        self.offers = offers
        self.rented = rented or {}
        self.calls = []

    def pick_node(self, gpus, fallback=True, price_ceiling=None):
        self.calls.append(('pick', tuple(gpus), fallback, price_ceiling))
        g = gpus[0]
        price = self.offers.get(g)
        if price is None or price > price_ceiling:
            return None, None
        return g, {'price_per_hour': price}

    def rent(self, name, ttl, gpu=None):
        self.calls.append(('rent', name, ttl, gpu))
        return self.rented.get(gpu, {'id': 'pod-%s' % gpu, 'name': name, 'gpu': gpu,
                                     'price_h': self.offers[gpu], 'port': 40022, 'host': '1.2.3.4'})

    def release(self, pod, reason=None):
        self.calls.append(('release', pod, reason))


class TestRealRentPolicy(unittest.TestCase):
    NAME = PREFIX_FAKEPASS + 'run1'

    def test_order_is_b300_b200_h200_never_sharded(self):
        self.assertEqual(R.FAKEPASS_GPU_ORDER, ('B300', 'B200', 'H200'))
        with self.assertRaises(FP.FakepassError):
            R.make_fakepass_host_rent(_FakeRent({}), order=('B300', 'H100'))

    def test_falls_through_to_b200_under_ceiling(self):
        fake = _FakeRent({'B300': 80.0, 'B200': 40.0, 'H200': 20.0})
        out = R.make_fakepass_host_rent(fake)(self.NAME, gpu='B300')
        self.assertEqual(out['gpu'], 'B200')
        self.assertTrue(out['fakepass'])
        picks = [c for c in fake.calls if c[0] == 'pick']
        self.assertEqual([c[1] for c in picks], [('B300',), ('B200',)])
        self.assertTrue(all(c[2] is False and c[3] == float(R.PRICE_CEILING_USD_H) for c in picks))
        self.assertEqual([c for c in fake.calls if c[0] == 'rent'], [('rent', self.NAME, R.RENT_TTL, 'B200')])

    def test_overpriced_or_wrong_gpu_rental_is_released(self):
        fake = _FakeRent({'B300': 30.0}, rented={'B300': {
            'id': 'pod-x', 'name': self.NAME, 'gpu': 'H100', 'price_h': 30.0, 'port': 1,
        }})
        with self.assertRaises(RentError):
            R.make_fakepass_host_rent(fake)(self.NAME)
        self.assertIn(('release', 'pod-x', 'fakepass-rent-mismatch'), fake.calls)

    def test_no_offer_and_bad_name_refuse(self):
        with self.assertRaises(RentError):
            R.make_fakepass_host_rent(_FakeRent({'B300': 99.0}))(self.NAME)
        with self.assertRaises(RentError):
            R.make_fakepass_host_rent(_FakeRent({'B300': 1.0}))('affine-pe-live-1')


@contextlib.contextmanager
def _lium_fixture_module(fake: _FakeRent, pods=()) -> Iterator[types.ModuleType]:
    """Importable stand-in for the host Lium rental module (the external boundary)."""
    mod = types.ModuleType('fixture_lium_pe_rent')
    mod.pick_node = fake.pick_node
    mod.rent = fake.rent
    mod.release = fake.release
    mod.our_pods = lambda ps=None: list(pods)
    sys.modules[mod.__name__] = mod
    try:
        yield mod
    finally:
        sys.modules.pop(mod.__name__, None)


class TestRealRentNoteCallback(unittest.TestCase):
    """Nov 10 real-box fakepass: the rent note callback received ``event`` twice."""

    RUN = 'rentnote'

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        live_env = _live_fixture(Path(td.name))
        live_roots = FP.live_roots_from_env(live_env)
        self.paths = FP.make_sandbox(self.RUN, live_roots=live_roots)
        self.addCleanup(FP.cleanup_sandbox, self.paths)
        src = R.resolve_live_sources(live_env)
        staged = R.stage_live_sources(src, self.paths)
        env = R.real_env_for_sandbox(FP.env_for_sandbox(self.paths, self.RUN), self.paths)
        self.run_ = R.RealFakepass(
            run_id=self.RUN, paths=self.paths, env=env, live_roots=live_roots,
            sources=src, staged=staged,
        )

    def _timeline(self):
        return [json.loads(line) for line in self.run_.log_path.read_text().splitlines()]

    def test_rent_through_real_service_writes_one_rent_attempt_row(self):
        fake = _FakeRent({'B300': 30.0})
        with _lium_fixture_module(fake) as mod:
            svc = self.run_.build_rent(mod)
            out = svc.rent(item_id=self.run_.item_id, epoch=self.run_.sources.epoch, attempt=0)
        self.assertEqual(out['gpu'], 'B300')
        self.assertTrue(is_fakepass_name(out['name']))
        # One paid rental: the TypeError signature fallback must not run.
        self.assertEqual(len([c for c in fake.calls if c[0] == 'rent']), 1)
        rows = [r for r in self._timeline() if r['event'] == 'rent_attempt']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['gpu'], 'B300')
        self.assertEqual(rows[0]['offer_price_h'], 30.0)
        self.assertEqual(rows[0]['phase'], self.run_.phase.value)

    def test_note_contract_is_positional_event(self):
        seen = []

        def note(event, **extra):
            seen.append((event, extra))

        R.make_fakepass_host_rent(_FakeRent({'B300': 30.0}), note=note)(PREFIX_FAKEPASS + 'n1')
        self.assertEqual(seen, [('rent_attempt', {'gpu': 'B300', 'offer_price_h': 30.0})])


class _CapturePod:
    def __init__(self, out='OK'):
        self.cmds = []
        self.out = out

    def run(self, cmd, *a, **k):
        self.cmds.append(cmd)
        return self.out


class TestRealBoxCommandTags(unittest.TestCase):
    def test_real_free_gpus_and_place_cand_commands_are_tagged(self):
        pod = _CapturePod()
        with contextlib.suppress(Exception):
            pe_box.free_gpus(pod, also_vllm=True)
        with contextlib.suppress(Exception):
            pe_box.place_cand(pod, candidate_id=C)
        tags = [R.classify_box_command(c) for c in pod.cmds]
        self.assertIn('free_gpus', tags)
        self.assertIn('place_cand', tags)

    def test_hold_and_serve_like_commands_are_untagged(self):
        cmd = R.hold_command(job_dir='/workspace/pst/pit-jobs/x', candidate_id=C, hold_s=10)
        self.assertIsNone(R.classify_box_command(cmd))
        self.assertIsNone(R.classify_box_command('nvidia-smi --query-gpu=memory.used'))

    def test_recording_pod_logs_and_reads_forbidden_rows(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / 'pod.jsonl'
            pod = R.RecordingPod(_CapturePod(), log)
            pod.run('true')
            mark = len(pod.records)
            pod.run("pgrep -f 'train_pit.py|v83_dp.py' || true")
            self.assertEqual(pod.tags_since(mark), ['free_gpus'])
            FP.append_timeline(log, {'event': 'forbidden-call', 'tag': 'place_cand'})
            self.assertEqual(R.read_command_tags(log), ['free_gpus', 'place_cand'])


class TestRealIndexValidation(unittest.TestCase):
    def _pending(self, prefix='pit-fakepass/r1'):
        files = {
            'model-00001-of-00001.safetensors': {'sha256': '1' * 64, 'r2_key': prefix + '/%s/s' % C},
            'meta.json': {'sha256': '2' * 64, 'r2_key': prefix + '/%s/meta.json' % C},
        }
        return {'bf16_sha256': 'e' * 64, 'master_digest': 'd' * 64, 'files': files,
                'meta_json': {'r2_key': files['meta.json']['r2_key']}, 'pending': True}

    def test_pending_index_contract(self):
        R.validate_pending_index(self._pending(), candidate_id=C, run_prefix='pit-fakepass/r1', store_entry='e' * 64)
        for mutate in (
            lambda d: d['files'].pop('meta.json'),
            lambda d: d.update(meta_json={'version': 'v833f-fp32-head-master-v1', 'bf16': C}),
            lambda d: d.update(bf16_sha256=C),
            lambda d: d['files']['model-00001-of-00001.safetensors'].update(r2_key='pit-pending/x/s'),
        ):
            doc = self._pending()
            mutate(doc)
            with self.assertRaises(FP.FakepassError):
                R.validate_pending_index(doc, candidate_id=C, run_prefix='pit-fakepass/r1', store_entry='e' * 64)

    def test_promoted_index_rejects_synthetic_meta_json(self):
        live = 'pit-fakepass/r1/fp32-master/%s' % C
        doc = self._pending()
        doc.update(pending=False)
        for f in doc['files'].values():
            f['r2_key'] = live + '/' + f['r2_key'].rsplit('/', 1)[-1]
        doc['meta_json'] = {'r2_key': doc['files']['meta.json']['r2_key']}
        R.validate_promoted_index(doc, candidate_id=C, live_prefix=live, master_digest='d' * 64)
        for bad in (
            {'version': 'v833f-fp32-head-master-v1', 'bf16': C, 'master_digest': 'd' * 64},
            {'r2_key': doc['meta_json']['r2_key'], 'version': 'x'},
            {'r2_key': live + '/other.json'},
        ):
            with self.assertRaises(FP.FakepassError):
                R.validate_promoted_index(dict(doc, meta_json=bad), candidate_id=C,
                                          live_prefix=live, master_digest='d' * 64)
        with self.assertRaises(FP.FakepassError):
            R.validate_promoted_index(dict(doc, pending=True), candidate_id=C, live_prefix=live, master_digest='d' * 64)

    def test_sweep_refuses_outside_run_prefix(self):
        for bad in ('pit-pending/x', 'pit-fakepass', 'pit-fakepass/', 'fp32-master/a', 'pit-fakepass/../k1'):
            with self.assertRaises(FP.FakepassError, msg=bad):
                R.sweep_r2_prefix(bad, bucket=object())


class TestRealHashSyncScript(unittest.TestCase):
    def _run(self, idx_doc, *, head=C):
        paths = FP.make_sandbox('hsync')
        self.addCleanup(FP.cleanup_sandbox, paths)
        idx = paths['fp32_public']
        if idx_doc is not None:
            _wj(idx / ('%s.json' % head), idx_doc)
        script = paths['hash_sync'] / 'sync_expected_hash.py'
        script.write_bytes(R.HASH_SYNC_SRC.read_bytes())
        env = {'PATH': os.environ.get('PATH', ''), 'PIT_FP32_MASTER_INDEX': str(idx)}
        proc = subprocess.run([sys.executable, str(script), '--head=%s' % head],
                              env=env, capture_output=True, text=True, timeout=60)
        return proc, paths['hash_sync']

    def _doc(self, **kw):
        key = 'pit-fakepass/h/fp32-master/%s/meta.json' % C
        doc = {
            'bf16_sha256': 'e' * 64, 'master_digest': 'd' * 64, 'pending': False,
            'files': {'model-00001-of-00001.safetensors': {'sha256': '1' * 64, 'r2_key': 'k'},
                      'meta.json': {'sha256': '2' * 64, 'r2_key': key}},
            'meta_json': {'r2_key': key},
        }
        doc.update(kw)
        return doc

    def test_writes_expected_hash_for_promoted_index(self):
        proc, out = self._run(self._doc())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual((out / 'fp32-expected-master.hash').read_text().strip(), 'd' * 64)
        self.assertTrue((out / 'LAST-SYNC.json').is_file())

    def test_alerts_on_synthetic_meta_json_or_pending(self):
        for doc in (
            self._doc(meta_json={'version': 'v833f-fp32-head-master-v1', 'bf16': C}),
            self._doc(pending=True),
            None,
        ):
            proc, out = self._run(doc)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertFalse((out / 'fp32-expected-master.hash').exists())
            self.assertTrue((out / 'ALERT.json').is_file())

    def test_refuses_outside_fakepass_sandbox(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / 'sync_expected_hash.py'
            script.write_bytes(R.HASH_SYNC_SRC.read_bytes())
            proc = subprocess.run([sys.executable, str(script), '--head=%s' % C],
                                  env={'PATH': os.environ.get('PATH', '')},
                                  capture_output=True, text=True, timeout=60)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse((Path(td) / 'fp32-expected-master.hash').exists())


@_SKIP_NO_MOTO
class TestRealApplyChainHermetic(unittest.TestCase):
    """REAL LiveApplicator → pe_apply/pe_live → lifecycle → record_applied on sandbox + moto R2."""

    RUN = 'applyhm'

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        root = Path(self.td.name)
        self.live_env = _live_fixture(root)
        creds = root / 'r2.env'
        creds.write_text('R2_ACCESS_KEY_ID=testing\nR2_SECRET_ACCESS_KEY=testing\n')
        self.life_cfg = _wj(root / 'life.json', {'bucket': {
            'bucket': BUCKET, 'endpoint': 'https://s3.amazonaws.com', 'credentials_file': str(creds),
        }})
        self.aws = mock_aws()
        self.aws.start()
        self.addCleanup(self.aws.stop)
        self.s3 = boto3.client('s3', region_name='us-east-1',
                               aws_access_key_id='testing', aws_secret_access_key='testing')
        self.s3.create_bucket(Bucket=BUCKET)
        self.live_roots = FP.live_roots_from_env(self.live_env)
        self.paths = FP.make_sandbox(self.RUN, live_roots=self.live_roots)
        self.addCleanup(lambda: FP.cleanup_sandbox(self.paths) if self.paths['root'].exists() else None)
        self.src = R.resolve_live_sources(self.live_env)
        staged = R.stage_live_sources(self.src, self.paths)
        base = FP.env_for_sandbox(self.paths, self.RUN)
        base.update(PIT_LIFE_CFG=str(self.life_cfg), AFFINE_LIFE_CFG=str(self.life_cfg))
        self.env = R.real_env_for_sandbox(base, self.paths)
        self.run_ = R.RealFakepass(
            run_id=self.RUN, paths=self.paths, env=self.env, live_roots=self.live_roots,
            sources=self.src, staged=staged, sever_after_green=True,
        )

    def _seed_early_upload(self):
        """State _early_upload_artifacts leaves: pending R2 objects, pending index, UPLOAD receipt."""
        shard = b'fp32-shard-bytes'
        meta = json.dumps({'version': 'v833f-fp32-head-master-v1', 'bf16': C,
                           'master_digest': 'd' * 64, 'tensor_file': 'master.safetensors'}).encode()
        bf16 = b'bf16-weights'
        store_entry = _sha(b'store-entry')
        prefix = AU.pending_r2_prefix(C)
        inv = {'model-00001-of-00001.safetensors': shard, 'meta.json': meta}
        for name, data in inv.items():
            self.s3.put_object(Bucket=BUCKET, Key='%s/%s' % (prefix, name), Body=data)
        self.s3.put_object(Bucket=BUCKET, Key='%s/model/model.safetensors' % prefix, Body=bf16)
        idx = AU.write_pending_master_index(
            candidate_id=C, bf16=store_entry, digest='d' * 64,
            inv={n: {'sha256': _sha(d), 'bytes': len(d)} for n, d in inv.items()},
        )
        R.validate_pending_index(json.loads(idx.read_text()), candidate_id=C,
                                 run_prefix=R.run_r2_prefix(self.RUN), store_entry=store_entry)
        files = {'model.safetensors': _sha(bf16)}
        _wj(self.paths['uploads'] / C / 'UPLOAD.json', {
            'id': C, 'sha256': C, 'files': files, 'verified_files': files, 'pending': True,
        })
        _wj(self.paths['uploads'] / C / 'CHECKPOINT.json', {'id': C, 'files': files})
        self.run_.candidate_id = C
        self.run_.store_entry = store_entry
        self.run_.receipt = {
            'candidate_id': C, 'candidate_sha256': C, 'candidate_files': files,
            'candidate_master_digest': 'd' * 64, 'artifacts_uploaded': True,
            'live_apply_seed': {'candidate_verified': True, 'master_verified': True,
                                'master_complete': True, 'pending': True},
        }

    def _keys(self, prefix):
        page = self.s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        return sorted(o['Key'] for o in page.get('Contents') or [])

    def test_green_reaches_sandbox_head_chain_and_hash_sync_with_box_severed(self):
        with _fresh_live_modules(), R.swapped_environ(self.env):
            self.run_.phase_isolate()
            self._seed_early_upload()
            self.run_.pod = FP.SeveredPod(_CapturePod())
            self.run_.pod.arm()
            self.run_.phase_apply_tail()
            out = self.run_.apply_result
            self.assertTrue(out['head_advanced'])
            self.assertEqual(out['new_head'], C)
            self.assertEqual(out['hash_sync'], 'd' * 64)
            self.assertEqual(out['ssh_attempts'], 0)
            chain = json.loads((self.paths['head_chain'] / 'HEAD-CHAIN.json').read_text())
            self.assertEqual(chain['genesis'], HEAD)
            self.assertTrue(any(a.get('new_head') == C for a in chain['applied']))
            promoted = json.loads(Path(out['promoted_index']).read_text())
            live_prefix = AU.live_fp32_r2_prefix(C)
            R.validate_promoted_index(promoted, candidate_id=C, live_prefix=live_prefix, master_digest='d' * 64)
            self.assertIn('%s/meta.json' % live_prefix, self._keys(live_prefix))
            cur = json.loads((self.paths['life'] / 'CURRENT.json').read_text())
            self.assertEqual(cur['checkpoint']['id'], C)
            self.run_.gc_r2()
            self.assertEqual(self._keys(R.run_r2_prefix(self.RUN) + '/'), [])
        live_life = Path(self.live_env['PE_LIFE'])
        self.assertEqual(json.loads((live_life / 'CURRENT.json').read_text())['checkpoint']['id'], LIVE_HEAD)

    def test_missing_pending_meta_json_object_blocks_promote_and_head(self):
        with _fresh_live_modules(), R.swapped_environ(self.env):
            self.run_.phase_isolate()
            self._seed_early_upload()
            self.s3.delete_object(Bucket=BUCKET, Key='%s/meta.json' % AU.pending_r2_prefix(C))
            self.run_.pod = _CapturePod()
            with self.assertRaisesRegex(Exception, 'promote failed'):
                self.run_.phase_apply_tail()
            self.assertFalse((self.paths['fp32_public'] / ('%s.json' % C)).exists())
            chain = json.loads((self.paths['pe'] / 'HEAD-CHAIN.json').read_text())
            self.assertEqual(chain['applied'], [])
            self.assertFalse((self.paths['hash_sync'] / 'fp32-expected-master.hash').exists())


_REAL_E2E = os.environ.get('PIT_FAKEPASS_REAL_E2E') == '1'


@unittest.skipUnless(
    _REAL_E2E and FP.has_lium_credentials() and os.environ.get('PIT_LIFE_CFG'),
    'REAL fakepass rents a paid box: needs Lium + R2 creds and PIT_FAKEPASS_REAL_E2E=1',
)
class TestRealFakepassEndToEnd(unittest.TestCase):
    def test_real_restart_host_mid_eval(self):
        evidence = Path(tempfile.mkdtemp(prefix='fakepass-evidence-'))
        self.addCleanup(lambda: subprocess.run(['rm', '-rf', str(evidence)], check=False))
        proc = subprocess.run(
            [sys.executable, '-m', 'ops.parallel_eval.pit.fakepass', '--real',
             '--run-id', 'e2e%d' % os.getpid(), '--restart-host-mid-eval',
             '--evidence-out', str(evidence)],
            cwd=str(R.REPO), capture_output=True, text=True, timeout=6 * 3600,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-4000:] + proc.stderr[-4000:])
        report = json.loads(proc.stdout)
        self.assertTrue(report['ok'])
        self.assertTrue(report['restart_result']['no_free_gpus'])
        self.assertTrue(report['restart_result']['no_place_cand'])
        self.assertTrue(report['apply_result']['head_advanced'])
        self.assertEqual(report['second_step']['fp32_prepared']['source'], 'disk')


if __name__ == '__main__':
    unittest.main()
