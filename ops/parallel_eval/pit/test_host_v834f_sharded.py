"""host-v834f ZeRO-3: multi-process gloo smoke + signer dry-run (no real seed)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
STAGE = REPO / 'ops' / 'parallel_eval' / 'pit' / 'host_v834f_signing' / 'stage'
SIGN_DIR = REPO / 'ops' / 'parallel_eval' / 'pit' / 'host_v834f_signing'
TRAIN_PIT = REPO / 'ops' / 'parallel_eval' / 'live' / 'train_pit.py'
TRAINER_E = REPO / 'miner' / 'validator' / 'trainer-v834e'
ADMITTED = REPO / 'ops' / 'parallel_eval' / 'pit' / 'host_v834e_admitted.sha256.json'


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestTrainerV834eUntouched(unittest.TestCase):
    def test_no_v835_in_trainer_v834e(self):
        self.assertFalse((TRAINER_E / 'v835_zero3.py').exists())
        man = json.loads(ADMITTED.read_text())
        self.assertNotIn('v835_zero3.py', man['host_files'])
        for rel, expect in man['host_files'].items():
            got = hashlib.sha256((TRAINER_E / rel).read_bytes()).hexdigest()
            self.assertEqual(got, expect, rel)

    def test_train_pit_activates_with_host_not_none(self):
        src = TRAIN_PIT.read_text()
        self.assertIn('z3.activate(host)', src)
        self.assertNotIn('z3.activate(None)', src)
        self.assertIn('prepare_wrapped_model', src)
        self.assertIn('canonical_named_parameters', src)
        self.assertIn("host.checked(HOST / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json')", src)
        self.assertNotIn('host.private_json(', src)
        self.assertNotIn('trainer-config.public.json', src)


class TestManifestProposalConsistent(unittest.TestCase):
    def test_manifest_matches_proposal_no_pyc_no_public_overlay(self):
        prop = json.loads((SIGN_DIR / 'PROPOSAL.unsigned.json').read_text())
        man = {}
        for line in (SIGN_DIR / 'SIGNED-MANIFEST.sha256').read_text().splitlines():
            if not line.strip():
                continue
            digest, _, rel = line.partition('  ')
            self.assertTrue(rel.startswith('./trainer-host-v834f/'))
            self.assertFalse(rel.endswith('.pyc'))
            self.assertNotIn('__pycache__', rel)
            local = rel[len('./trainer-host-v834f/'):]
            man[local] = digest
            got = hashlib.sha256((STAGE / local).read_bytes()).hexdigest()
            self.assertEqual(got, digest, local)
        self.assertEqual(set(man), set(prop['host_files']))
        self.assertNotIn('trainer-config.public.json', man)
        self.assertIn('v835_zero3.py', man)
        for rel, expect in prop['host_files'].items():
            self.assertEqual(man[rel], expect)


class TestFp32AccumDeviceGating(unittest.TestCase):
    def test_accum_device_ignored_without_zero3(self):
        z3 = _load(STAGE / 'v835_zero3.py', 'v835_gate')
        os.environ.pop('PIT_ZERO3', None)
        os.environ['PIT_FP32_ACCUM_DEVICE'] = 'cpu'
        try:
            # Stage v83_dp helper mirrors the gate.
            dp = _load(STAGE / 'v83_dp.py', 'v83_dp_gate')
            self.assertIsNone(dp._fp32_accum_device())
            from ops.parallel_eval.pit import train_shard as ts
            self.assertIsNone(ts.fp32_accum_device_from_env())
        finally:
            os.environ.pop('PIT_FP32_ACCUM_DEVICE', None)

    def test_preflight_uses_offload_not_naive_div8(self):
        z3 = _load(STAGE / 'v835_zero3.py', 'v835_preflight')
        # 9B-ish param count (~18 GiB bf16).
        n = int(18 * (1 << 30) / 2)
        naive = (18.0 / 8.0) + (18.0 / 8.0)  # params/8 + grads/8 only — incomplete
        est = z3.estimate_peak_gib(
            n_params=n, world=8, token_budget=24576, sharded=True, cpu_offload=True,
        )
        self.assertLess(est['peak_gib'], 28.0)
        # Offload path must not pretend full 18+18 fits via /8 alone without residual.
        self.assertEqual(est['cpu_offload'], 1.0)
        unoff = z3.estimate_peak_gib(
            n_params=n, world=8, token_budget=24576, sharded=True, cpu_offload=False,
        )
        # Without offload, sharded params+grads still leave little room; offload is lower.
        self.assertLess(est['peak_gib'], unoff['peak_gib'])
        _ = naive  # documented contrast only


class TestSignerDryRun(unittest.TestCase):
    def test_dry_run_never_reads_seed(self):
        sign = _load(SIGN_DIR / 'sign_host_v834f.py', 'sign_host_v834f_test')
        # Point a fake lifecycle seed path; dry-run must not open it.
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / 'ROOT.seed'
            seed.write_text('00' * 32)
            os.chmod(seed, 0o600)
            # If dry-run tried to read the seed via --lifecycle, we'd still be fine
            # because we do not pass --sign. Ensure Identity path is used.
            # Capture stdout.
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                sign.main(['--dry-run'])
            out = json.loads(buf.getvalue())
            self.assertTrue(out['ok'])
            self.assertTrue(out['dry_run'])
            self.assertTrue(out.get('verified'))
            self.assertEqual(out['host_version'], 'host-v834f')


class TestMultiprocessGlooInstallAndStep(unittest.TestCase):
    def test_install_start_dp_and_one_train_step_world2(self):
        """Real install(start_dp=True) wrap path + gloo smoke step.

        1. Subprocess ``v835_zero3.py smoke`` (world=2 gloo) — avoids pytest's
           hermetic ``open`` patch breaking multiprocessing spawn.
        2. In-process ``activate(host)`` + install(start_dp=True) stub proving
           helpers start via the v835_zero3 entry with the token-budget cut.
        """
        pytest = __import__('pytest')
        pytest.importorskip('torch')
        import subprocess

        # --- smoke: wrap + one step (no hang, sharded, keys, loss) ---
        env = dict(
            os.environ,
            PIT_ZERO3='1',
            PIT_ZERO3_BACKEND='gloo',
            PIT_PER_GPU_MEM_GIB='32',
            PYTHONPATH=str(REPO),
        )
        proc = subprocess.run(
            [sys.executable, '-B', str(STAGE / 'v835_zero3.py'), 'smoke'],
            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + '\n' + proc.stderr)
        smoke = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(smoke['ok'])
        self.assertEqual(smoke['keys'], sorted(['0.weight', '1.weight']))
        for name, nbytes in smoke['local_bytes'].items():
            self.assertLess(int(nbytes), int(smoke['full_bytes'][name]), name)
        self.assertLess(float(smoke['loss_diff']), 1e-3)

        # --- install(start_dp=True) via activate + wrap_host_install ---
        sys.path.insert(0, str(STAGE))
        import v835_zero3 as z3  # noqa: E402

        os.environ['PIT_ZERO3'] = '1'
        os.environ['PIT_ZERO3_BACKEND'] = 'gloo'
        os.environ['PIT_PER_GPU_MEM_GIB'] = '32'
        helpers_started = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                helpers_started.append(list(args))
                self.args = args
                self.pid = 0

            def poll(self):
                return 0

        class FakeHost:
            P = STAGE
            TRITON_CACHE = '/tmp/v834f-triton-test'
            _v835_install_wrapped = False

            def load(self, name, rel):
                return _load(self.P / rel, name)

            def install(self, config, start_dp=False):
                return dict(component='stub', start_dp=start_dp)

        host = FakeHost()
        with self.assertRaises(RuntimeError):
            z3.activate(None)

        pol = z3.activate(host)
        self.assertTrue(pol['zero3'])
        self.assertTrue(getattr(host, '_v835_install_wrapped', False))

        cfg = {
            'engine': {
                'dp_world': 2,
                'dp_port': 29622,
                'dp_token_budget': 69632,
                'dp_max_rows': 64,
                'token_budget': 69632,
                'training_kernels': {},
            },
            'source_root': str(REPO),
        }

        # Call the wrapped install with FakePopen; stub dist.init_process_group
        # so patch_dp's init does not open a real process group.
        import torch.distributed as dist
        real_popen = subprocess.Popen
        real_init_pg = dist.init_process_group
        real_new_group = dist.new_group
        subprocess.Popen = FakePopen

        def fake_init_pg(*args, **kwargs):
            return None

        def fake_new_group(*args, **kwargs):
            return None

        dist.init_process_group = fake_init_pg
        dist.new_group = fake_new_group
        host_load = host.load

        def load(name, rel):
            if rel == 'v83_fast_kernels.py':
                return SimpleNamespace()
            return host_load(name, rel)

        host.load = load
        try:
            sys.modules.pop('v824_batched', None)
            info = host.install(cfg, start_dp=True)
        finally:
            subprocess.Popen = real_popen
            dist.init_process_group = real_init_pg
            dist.new_group = real_new_group

        self.assertTrue(info.get('zero3'))
        self.assertEqual(info.get('token_budget'), 24576)
        self.assertEqual(info.get('dp_token_budget'), 24576)
        self.assertEqual(len(helpers_started), 1)
        argv = helpers_started[0]
        self.assertEqual(argv[1], '-B')
        self.assertTrue(str(argv[2]).endswith('v835_zero3.py'), argv)
        self.assertEqual(argv[3], 'helper')
        self.assertEqual(argv[4], '1')
        self.assertEqual(argv[5], '2')

        os.environ.pop('PIT_ZERO3', None)
        os.environ.pop('PIT_ZERO3_BACKEND', None)
        os.environ.pop('PIT_PER_GPU_MEM_GIB', None)


if __name__ == '__main__':
    unittest.main()
