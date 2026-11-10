"""host-v834f sharded (ZeRO-3) trainer: real torch FSDP in real processes on the CPU (gloo), and the ROOT signer.

What these tests run, and what they cannot run, is stated in host_v834f_signing/harness/fixture.py:
the launcher, v83_dp, v8211_batched and v835_zero3 are the staged host files; the model is a tiny random
production-class model; the source closure is a stand-in; CUDA, NCCL, the 9B model and the fla kernels are not
exercised. train_pit.py itself is only inspected, not run.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from nacl.signing import SigningKey
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from ops.parallel_eval.live import pe_artifact_upload
from ops.parallel_eval.pit import train_shard
from subnet.storage import canonical, sha

REPO = Path(__file__).resolve().parents[3]
PIT = REPO / 'ops' / 'parallel_eval' / 'pit'
SIGNING = PIT / 'host_v834f_signing'
STAGE = SIGNING / 'stage'
HARNESS = SIGNING / 'harness'
TRAIN_PIT = REPO / 'ops' / 'parallel_eval' / 'live' / 'train_pit.py'
TRAINER_E = REPO / 'miner' / 'validator' / 'trainer-v834e'
ADMITTED = PIT / 'host_v834e_admitted.sha256.json'
CHANGED = ('mimo_host_trainer_v83.py', 'v835_zero3.py')
# every key host-v834e writes into the meta.json of an fp32 master entry (_fp32_finish + _fp32_save)
MASTER_META_KEYS = ['bf16_sha256', 'files', 'input_bf16_sha256', 'input_master_digest', 'master_digest', 'optimizer_step',
                    'save_seconds', 'saved_at', 'shard_bytes', 'source', 'tensor_file', 'tensor_sha256', 'version']
REMOVED_SWITCHES = ('PIT_FP32_ACCUM_DEVICE', 'PIT_LOW_MEM_SHARD', 'buffer_device')
# Runs the signer in its own interpreter with an audit hook that records every file the process opens.
AUDITED_SIGNER = r'''
import io, json, sys, types
from contextlib import redirect_stdout
from pathlib import Path
opened = []
sys.addaudithook(lambda event, args: opened.append(str(args[0])) if event == 'open' and args else None)
signer, here, root_id = sys.argv[1:4]
module = types.ModuleType('sign_host_v834f_audited'); module.__file__ = signer
exec(compile(Path(signer).read_bytes(), signer, 'exec'), module.__dict__)
module.HERE, module.STAGE, module.ROOT_ID = Path(here), Path(here) / 'stage', root_id
out = io.StringIO()
with redirect_stdout(out):
    module.main(sys.argv[4:])
print(json.dumps(dict(result=json.loads(out.getvalue()), opened=opened)))
'''


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(path, name):
    """A module from a source file, without writing a bytecode file next to it."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def child_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIT_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.update(extra)
    return env


class LocalBox:
    """The PIT host reaches the training box over ssh. Here the box is this machine: the upload path's own
    commands run in a local shell, on the files the trainer wrote."""

    def run(self, cmd, timeout=60, check=False, **kwargs):
        proc = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, errors='replace', timeout=timeout)
        if check and proc.returncode != 0:
            raise RuntimeError('box command failed: ' + proc.stderr[-400:])
        return proc.stdout


def run_driver(work, mode, *args, env=None, timeout=600):
    started = time.monotonic()
    proc = subprocess.run([sys.executable, '-B', str(HARNESS / 'zero3_e2e.py'), '--work', str(work), '--mode', mode, *args],
                          capture_output=True, text=True, timeout=timeout, env=env or child_env())
    return proc, time.monotonic() - started


class TestAdmittedHostUntouched(unittest.TestCase):
    def test_trainer_v834e_is_the_admitted_inventory_and_nothing_else(self):
        admitted = json.loads(ADMITTED.read_text())['host_files']
        files = sorted(p.relative_to(TRAINER_E).as_posix() for p in TRAINER_E.rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts)
        self.assertEqual(files, sorted(list(admitted) + ['trainer-config.public.json']))
        self.assertNotIn('v835_zero3.py', files)
        for rel, digest in admitted.items():
            self.assertEqual(sha256(TRAINER_E / rel), digest, rel)

    def test_stage_differs_from_admitted_only_in_the_declared_files(self):
        admitted = json.loads(ADMITTED.read_text())['host_files']
        differing = sorted(rel for rel in admitted if sha256(STAGE / rel) != admitted[rel])
        self.assertEqual(differing, ['mimo_host_trainer_v83.py'])
        self.assertTrue((STAGE / 'v835_zero3.py').is_file())
        self.assertEqual(sha256(STAGE / 'v83_dp.py'), admitted['v83_dp.py'])

    def test_staged_launcher_keeps_the_admitted_fp32_master_code(self):
        """Definition by definition, the host-v834f launcher differs from the admitted one only in HOST_FILES and
        install(), plus the new zero3(). So master prepare, load, step, finish, save and GC are host-v834e's."""
        def definitions(path):
            text, found = path.read_text(), {}
            for node in ast.parse(text).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    found[node.name] = ast.get_source_segment(text, node)
                elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    found[node.targets[0].id] = ast.get_source_segment(text, node)
            return found
        admitted = definitions(TRAINER_E / 'mimo_host_trainer_v83.py')
        staged = definitions(STAGE / 'mimo_host_trainer_v83.py')
        self.assertEqual(sorted(set(staged) - set(admitted)), ['zero3'])
        self.assertEqual(sorted(set(admitted) - set(staged)), [])
        self.assertEqual(sorted(name for name in admitted if admitted[name] != staged[name]), ['HOST_FILES', 'install'])
        for name in ('_fp32_prepare', '_fp32_load', '_fp32_finish', '_fp32_save', '_fp32_gc', '_digest_map', 'sgd_step',
                     '_fresh_adam_step', 'FP32_STORE', 'FP32_SHARD_BYTES', 'FP32_VERSION'):
            self.assertEqual(staged[name], admitted[name], name)

    def test_removed_accumulator_switches_are_read_nowhere(self):
        """PIT_FP32_ACCUM_DEVICE / PIT_LOW_MEM_SHARD cannot change behaviour: no trainer file reads them."""
        files = [p for p in STAGE.rglob('*.py')] + [TRAIN_PIT, PIT / 'train_shard.py',
                                                    REPO / 'subnet' / 'mimo_grpo' / 'optimizer.py',
                                                    REPO / 'subnet' / 'mimo_grpo' / 'training.py']
        for path in files:
            tree = ast.parse(path.read_text())
            docstring = ast.get_docstring(tree, clean=False)
            for node in ast.walk(tree):
                names = [node.value] if isinstance(node, ast.Constant) and isinstance(node.value, str) else \
                    [node.id] if isinstance(node, ast.Name) else [node.arg] if isinstance(node, (ast.arg, ast.keyword)) else \
                    [node.attr] if isinstance(node, ast.Attribute) else []
                for name in names:
                    if name is None or name == docstring:
                        continue
                    for needle in REMOVED_SWITCHES:
                        self.assertNotIn(needle, name, '%s in %s' % (needle, path.relative_to(REPO)))

    def test_sharded_code_runs_only_under_pit_zero3(self):
        launcher = (STAGE / 'mimo_host_trainer_v83.py').read_text()
        self.assertIn('sharded = z3.enabled()', launcher)
        self.assertIn('if sharded else dp.make_train_update(batched, fast_kernels)', launcher)
        zero3 = load_source(STAGE / 'v835_zero3.py', 'v835_policy_probe')
        saved = os.environ.pop('PIT_ZERO3', None)
        try:
            self.assertFalse(zero3.enabled())
            os.environ['PIT_FP32_ACCUM_DEVICE'] = 'cpu'
            self.assertFalse(zero3.enabled())
            self.assertFalse(zero3.needs_zero3(180.0))
            self.assertTrue(zero3.needs_zero3(32.0))
        finally:
            os.environ.pop('PIT_FP32_ACCUM_DEVICE', None)
            if saved is not None:
                os.environ['PIT_ZERO3'] = saved


class TestTrainPit(unittest.TestCase):
    def setUp(self):
        self.src = TRAIN_PIT.read_text()

    def test_checked_on_every_path_and_activate_gets_the_launcher(self):
        self.assertEqual(self.src.count("cfg = host.checked(HOST / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json')"), 1)
        self.assertLess(self.src.index('host.checked('), self.src.index('host.zero3()'))
        self.assertLess(self.src.index('z3.activate(host)'), self.src.index('host.install(cfg, start_dp=True)'))
        self.assertNotIn('activate(None)', self.src)
        self.assertNotIn('host.private_json(', self.src)
        self.assertNotIn('trainer-config.public.json', self.src)
        self.assertNotIn('wrap_fsdp', self.src)

    def test_train_call_uses_the_budget_the_launcher_applied(self):
        self.assertIn("token_budget = int(info['dp_token_budget']) if sharded else e['token_budget']", self.src)
        self.assertIn('token_budget=token_budget, max_rows=e[\'max_rows\'])', self.src)
        self.assertNotIn("token_budget=e['token_budget']", self.src)

    def test_host_selection_fails_closed(self):
        """Run train_pit._select_host on a scratch tree: host-v834f only with the promotion gate."""
        func = next(n for n in ast.parse(self.src).body if isinstance(n, ast.FunctionDef) and n.name == '_select_host')
        root = Path(tempfile.mkdtemp(prefix='select-host-'))
        self.addCleanup(shutil.rmtree, root, True)
        scope = dict(os=os, Path=Path, W=root)
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(TRAIN_PIT), 'exec'), scope)
        select = scope['_select_host']
        saved = {k: os.environ.pop(k, None) for k in ('PIT_ZERO3', 'PIT_ALLOW_SHARDED_GPUS', 'PIT_PER_GPU_MEM_GIB', 'CUDA_VISIBLE_MEM_GIB')}
        try:
            with self.assertRaises(SystemExit):
                select()                                        # no host-v834e at all
            (root / 'host-v834e').mkdir()
            self.assertEqual(select().name, 'host-v834e')
            os.environ['PIT_PER_GPU_MEM_GIB'] = '32'
            self.assertEqual(select().name, 'host-v834e')       # low memory alone never selects the sharded host
            os.environ['PIT_ZERO3'] = '1'
            with self.assertRaises(SystemExit):
                select()                                        # forced without the gate
            os.environ['PIT_ALLOW_SHARDED_GPUS'] = '1'
            with self.assertRaises(SystemExit):
                select()                                        # gate open but no signed host-v834f on the box
            (root / 'host-v834f').mkdir()
            self.assertEqual(select().name, 'host-v834f')
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value


class SignerCase(unittest.TestCase):
    """A scratch copy of the signing directory and a throwaway key standing in for ROOT (never a real seed)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='v834f-sign-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.here = self.tmp / 'host_v834f_signing'
        manifest = (SIGNING / 'SIGNED-MANIFEST.sha256').read_text()
        for line in manifest.splitlines():
            rel = line.partition('  ')[2][len('./trainer-host-v834f/'):]
            (self.here / 'stage' / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(STAGE / rel, self.here / 'stage' / rel)
        for name in ('SIGNED-MANIFEST.sha256', 'PROPOSAL.unsigned.json'):
            shutil.copyfile(SIGNING / name, self.here / name)
        self.sign = load_source(SIGNING / 'sign_host_v834f.py', 'sign_host_v834f_under_test')
        self.sign.HERE, self.sign.STAGE = self.here, self.here / 'stage'

    def run_main(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            self.sign.main(argv)
        return json.loads(out.getvalue())

    def refused(self, argv=()):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(list(argv))
        return str(ctx.exception)

    def throwaway_root(self):
        self.seed = hashlib.sha256(b'v834f-test-seed-not-a-real-authority').digest()
        key = SigningKey(self.seed)
        self.root = key.verify_key.encode().hex()
        self.sign.ROOT_ID = self.root
        launcher = self.sign.load_launcher(STAGE / 'mimo_host_trainer_v83.py')
        fixture = load_source(HARNESS / 'fixture.py', 'fixture_for_signer')
        proposal = dict(version=launcher.VERSION, scope=launcher.SCOPE, authority=self.root,
                        source_root='/nonexistent-box/source-final', host_files=json.loads(ADMITTED.read_text())['host_files'],
                        original_methods={site: '0' * 64 for site in launcher.SITES},
                        engine=dict(fixture.ENGINE_BLOCK, dp_world=8, dp_max_rows=256))
        payload = dict(version=launcher.VERSION, configuration_sha256=sha(canonical(proposal)), source_unchanged=True,
                       scope=launcher.SCOPE)
        admission = dict(payload=payload, signer=self.root,
                         signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
        self.base = self.tmp / 'host-v834e.private.json'
        self.base.write_bytes(canonical(dict(proposal, admission=admission)))
        self.seed_file = self.tmp / 'root.seed'
        self.seed_file.write_text(self.seed.hex() + '\n')
        os.chmod(self.seed_file, 0o600)
        self.lifecycle = self.tmp / 'lifecycle.private.json'
        self.lifecycle.write_text(json.dumps(dict(authority_seed_file=str(self.seed_file))))
        self.reviewed = json.loads((self.here / 'PROPOSAL.unsigned.json').read_text())['manifest_sha256']


class TestSigningArtifact(SignerCase):
    def test_stage_manifest_and_proposal_agree(self):
        out = self.run_main([])
        self.assertTrue(out['ok'])
        self.assertFalse(out['signed'])
        self.assertEqual(out['host_file_count'], 16)
        proposal = json.loads((SIGNING / 'PROPOSAL.unsigned.json').read_text())
        manifest = {line.partition('  ')[2][len('./trainer-host-v834f/'):]: line.partition('  ')[0]
                    for line in (SIGNING / 'SIGNED-MANIFEST.sha256').read_text().splitlines()}
        self.assertEqual(manifest, dict(proposal['host_files'], **proposal['public_files']))
        self.assertEqual(sorted(proposal['changed_host_files']), sorted(CHANGED))
        self.assertEqual(proposal['manifest_sha256'], sha256(SIGNING / 'SIGNED-MANIFEST.sha256'))
        self.assertFalse([rel for rel in manifest if rel.endswith('.pyc') or '__pycache__' in rel])

    def test_committed_artifacts_are_what_the_stage_generates(self):
        before = {name: (self.here / name).read_bytes() for name in ('SIGNED-MANIFEST.sha256', 'PROPOSAL.unsigned.json',
                                                                    'stage/trainer-config.public.json')}
        self.run_main(['--write-proposal'])
        for name, raw in before.items():
            self.assertEqual((self.here / name).read_bytes(), raw, name + ' is stale: run sign_host_v834f.py --write-proposal')

    def test_public_copy_restates_no_signed_setting(self):
        public = json.loads((STAGE / 'trainer-config.public.json').read_text())
        self.assertEqual(public['host_version'], 'host-v834f')
        self.assertNotIn('engine', public)
        self.assertNotIn('trainer_config', public)
        zero3 = load_source(STAGE / 'v835_zero3.py', 'v835_public_probe')
        self.assertEqual(public['zero3']['token_budget'], zero3.LOW_VRAM_TOKEN_BUDGET)
        self.assertEqual(public['zero3']['logit_chunk'], zero3.LOGIT_CHUNK)
        self.assertEqual(public['host_files_sha256']['v835_zero3.py'], sha256(STAGE / 'v835_zero3.py'))

    def test_fails_closed_on_a_missing_file(self):
        (self.here / 'stage' / 'v835_zero3.py').unlink()
        message = self.refused()
        self.assertIn('missing', message)
        self.assertIn('v835_zero3.py', message)

    def test_fails_closed_on_a_changed_file(self):
        with open(self.here / 'stage' / 'v83_dp.py', 'ab') as stream:
            stream.write(b'\n# changed\n')
        self.assertIn('digest mismatch v83_dp.py', self.refused())

    def test_fails_closed_on_an_unlisted_file_and_on_bytecode(self):
        extra = self.here / 'stage' / 'extra_module.py'
        extra.write_text('x = 1\n')
        self.assertIn('unlisted file in stage', self.refused())
        extra.unlink()
        cache = self.here / 'stage' / '__pycache__'
        cache.mkdir()
        (cache / 'v835_zero3.cpython-312.pyc').write_bytes(b'\0')
        message = self.refused()
        self.assertIn('unlisted file in stage', message)
        self.assertIn('__pycache__', message)

    def test_fails_closed_when_manifest_and_proposal_differ(self):
        path = self.here / 'SIGNED-MANIFEST.sha256'
        path.write_text(''.join(line + '\n' for line in path.read_text().splitlines() if 'trainer-config.public.json' not in line))
        self.assertIn('manifest and proposal list different files', self.refused())


class TestSigner(SignerCase):
    def setUp(self):
        super().setUp()
        self.throwaway_root()

    def test_dry_run_signs_with_a_throwaway_key_and_never_opens_the_seed(self):
        proc = subprocess.run(
            [sys.executable, '-B', '-c', AUDITED_SIGNER, str(SIGNING / 'sign_host_v834f.py'), str(self.here), self.root,
             '--dry-run', '--base-private', str(self.base), '--lifecycle', str(self.lifecycle)],
            capture_output=True, text=True, timeout=300, env=child_env(), cwd=str(REPO))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        audited = json.loads(proc.stdout.strip().splitlines()[-1])
        out, opened = audited['result'], audited['opened']
        self.assertTrue(out['dry_run'] and out['signed'])
        self.assertNotEqual(out['authority'], self.root)
        self.assertEqual(out['verified'], dict(host_checked=True, cwd_emulated=True, host_files=16))
        self.assertIsNone(out['out'])
        self.assertIn(str(self.base), opened)                       # the hook does record this process's opens
        self.assertNotIn(str(self.seed_file), opened)
        self.assertNotIn(str(self.lifecycle), opened)
        self.assertNotIn(self.seed.hex(), proc.stdout + proc.stderr)

    def test_sign_writes_exactly_the_reviewed_inventory_and_the_host_verifier_accepts_it(self):
        release = self.tmp / 'release'
        out = self.run_main(['--sign', '--base-private', str(self.base), '--lifecycle', str(self.lifecycle),
                             '--source-root', str(REPO), '--manifest-sha256', self.reviewed, '--out', str(release)])
        self.assertTrue(out['signed'] and not out['dry_run'])
        self.assertEqual(out['authority'], self.root)
        self.assertTrue(out['verified']['host_checked'])
        host = release / 'trainer-host-v834f'
        reviewed = {line.partition('  ')[2][len('./trainer-host-v834f/'):]: line.partition('  ')[0]
                    for line in (SIGNING / 'SIGNED-MANIFEST.sha256').read_text().splitlines()}
        written = sorted(p.relative_to(host).as_posix() for p in host.rglob('*') if p.is_file())
        self.assertEqual(written, sorted(list(reviewed) + [self.sign.CONFIG_NAME]))
        for rel, digest in reviewed.items():
            self.assertEqual(sha256(host / rel), digest, rel)
        config = json.loads((host / self.sign.CONFIG_NAME).read_text())
        base = json.loads(self.base.read_text())
        for key in ('engine', 'original_methods', 'source_root', 'authority', 'version', 'scope'):
            self.assertEqual(config[key], base[key], key)
        self.assertEqual(sorted(k for k in config['host_files'] if config['host_files'][k] != base['host_files'].get(k)), sorted(CHANGED))
        everything = [p for p in release.rglob('*') if p.is_file()]
        self.assertTrue(all(p.stat().st_mode & 0o077 == 0 for p in everything))
        self.assertFalse([p for p in everything if p.suffix == '.pyc' or '__pycache__' in p.parts])
        self.assertFalse([p for p in everything if self.seed.hex() in p.read_text(errors='ignore')])
        self.assertNotIn(self.seed.hex(), json.dumps(out))
        listed = {line.partition('  ./')[2] for line in (release / 'SIGNED-MANIFEST.sha256').read_text().splitlines()}
        self.assertEqual(listed, {p.relative_to(release).as_posix() for p in everything} - {'SIGNED-MANIFEST.sha256'})
        self.assertIn('output exists', self.refused(['--sign', '--base-private', str(self.base), '--lifecycle', str(self.lifecycle),
                                                     '--source-root', str(REPO), '--manifest-sha256', self.reviewed, '--out', str(release)]))
        # the host verifier, not only the signer, rejects a release whose host file changed after signing
        target = host / 'v835_zero3.py'
        target.write_bytes(target.read_bytes() + b'\n# changed after signing\n')
        with self.assertRaises(SystemExit) as ctx:
            self.sign.host_checked(host, host / self.sign.CONFIG_NAME, REPO)
        self.assertIn('exact admitted host file v835_zero3.py', str(ctx.exception))

    def test_sign_refuses_an_unreviewed_manifest_a_loose_seed_and_a_checkout_output(self):
        common = ['--sign', '--base-private', str(self.base), '--lifecycle', str(self.lifecycle), '--source-root', str(REPO)]
        self.assertIn('not the reviewed one', self.refused(common + ['--manifest-sha256', '0' * 64, '--out', str(self.tmp / 'r1')]))
        self.assertIn('needs --lifecycle', self.refused(['--sign', '--base-private', str(self.base)]))
        self.assertIn('outside the checkout', self.refused(common + ['--manifest-sha256', self.reviewed,
                                                                    '--out', str(SIGNING / 'release-in-checkout')]))
        os.chmod(self.seed_file, 0o640)
        self.assertEqual(self.refused(common + ['--manifest-sha256', self.reviewed, '--out', str(self.tmp / 'r2')]), 'REFUSED: seed mode')
        os.chmod(self.seed_file, 0o600)
        self.seed_file.write_text('not-hex-' + self.seed.hex())
        message = self.refused(common + ['--manifest-sha256', self.reviewed, '--out', str(self.tmp / 'r3')])
        self.assertEqual(message, 'REFUSED: ROOT seed unreadable (ValueError)')
        self.assertFalse((self.tmp / 'r1').exists() or (self.tmp / 'r2').exists() or (self.tmp / 'r3').exists())

    def test_base_must_be_the_root_admitted_host_v834e(self):
        base = json.loads(self.base.read_text())
        other = dict(base, host_files=dict(base['host_files'], **{'v83_dp.py': '0' * 64}))
        (self.tmp / 'other.json').write_text(json.dumps(other))
        self.assertIn('does not bind the base configuration',
                      self.refused(['--dry-run', '--base-private', str(self.tmp / 'other.json')]))
        self.sign.ROOT_ID = 'ab' * 32                               # a base signed by someone else is not ROOT-admitted
        self.assertIn('not ROOT-signed', self.refused(['--dry-run', '--base-private', str(self.base)]))


class TestMemoryModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zero3 = load_source(STAGE / 'v835_zero3.py', 'v835_memory_probe')

    def test_reference_geometry_is_the_pinned_9b_configuration(self):
        with torch.device('meta'):
            model = Qwen3_5ForConditionalGeneration(Qwen3_5Config())
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('model.language_model.') or name.startswith('lm_head.'))
        self.assertEqual(self.zero3.inventory(model), self.zero3.REFERENCE_9B)

    def test_sharded_estimate_is_not_total_over_world(self):
        inv = self.zero3.REFERENCE_9B
        out = self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=8, inv=inv)
        total = sum(inv[k] for k in ('layer_sum', 'embed', 'head', 'other', 'frozen')) * 2 / 2 ** 30
        self.assertEqual(out['tokens'], 36864)                      # one rollout of the full training context
        self.assertGreater(out['certain_gib'], total)               # every rank loads the whole actor once
        self.assertGreater(out['certain_gib'], 3 * total / 8)       # far above "weights and gradients divided by 8"
        self.assertLess(out['estimate_gib'], out['limit_gib'])
        self.assertEqual(out['limit_gib'], 28.0)
        self.assertFalse(out['coefficients']['measured'])

    def test_layer_inputs_go_to_the_host_when_the_estimate_is_near_the_limit(self):
        inv, estimate = self.zero3.REFERENCE_9B, self.zero3.estimate_peak_gib
        budget = self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=8, inv=inv, tokens=24576)
        self.assertFalse(budget['offload_inputs'])                  # a micro-batch of the cut budget: inputs stay on the GPU
        self.assertEqual(budget['offload_host_gib'], 0.0)
        self.assertLess(budget['estimate_gib'], 0.75 * budget['limit_gib'])
        full = self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=8, inv=inv, tokens=36864)
        self.assertTrue(full['offload_inputs'])                     # one full-context rollout: planned with them on the host
        on_gpu = estimate(inv, tokens=36864, world=8)
        self.assertGreater(on_gpu['estimate_gib'], 0.75 * full['limit_gib'])
        self.assertLess(full['phases']['layer']['estimate_gib'], on_gpu['phases']['layer']['estimate_gib'] - 8.0)
        hidden = 36864 * inv['hidden'] * 2 / 2 ** 30
        self.assertAlmostEqual(full['offload_host_gib'], (inv['layers'] + 2 - 3) * hidden, places=6)
        for tokens in (8192, 24576, 36864):                         # more tokens never lower the estimate of a mode
            for mode in (False, True):
                self.assertLessEqual(estimate(inv, tokens=tokens, world=8, offload_inputs=mode)['estimate_gib'],
                                     estimate(inv, tokens=tokens + 4096, world=8, offload_inputs=mode)['estimate_gib'])

    def test_refuses_what_cannot_fit(self):
        with self.assertRaises(self.zero3.VramPreflightError) as ctx:
            self.zero3.preflight_or_raise(per_gpu_gib=16.0, world=8, inv=self.zero3.REFERENCE_9B)
        self.assertIn('exceeds VRAM', str(ctx.exception))
        with self.assertRaises(self.zero3.VramPreflightError):
            self.zero3.preflight_or_raise(per_gpu_gib=None, world=8, inv=self.zero3.REFERENCE_9B)
        with self.assertRaises(self.zero3.VramPreflightError):
            self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=1, n_params=9.4e9, sharded=False)

    def test_token_budget_cut(self):
        self.assertEqual(self.zero3.token_budget(40960), 24576)
        self.assertEqual(self.zero3.token_budget(16384), 16384)

    def test_host_memory_model_counts_the_gradient_once(self):
        inv = self.zero3.REFERENCE_9B
        ram = self.zero3.host_ram_gib(inv)
        trainable = inv['layer_sum'] + inv['embed'] + inv['head'] + inv['other']
        gib = 2.0 ** 30
        self.assertAlmostEqual(ram['rank0_gib'], ((trainable + inv['frozen']) * 2 + trainable * 4) / gib)   # actor + fp32 master
        self.assertAlmostEqual(ram['shards_gib'], (inv['layer_sum'] + inv['other']) * 4 / gib)
        self.assertAlmostEqual(ram['step_gib'], (trainable * 4 + inv['embed'] * 20) / gib)                  # one fp32 gradient + step temporaries
        self.assertAlmostEqual(ram['total_gib'], ram['rank0_gib'] + ram['shards_gib'] + ram['step_gib'])


class TestWatchdog(unittest.TestCase):
    def test_a_failed_helper_stops_the_trainer_at_once(self):
        code = (
            "import sys, threading, time, types\n"
            "from pathlib import Path\n"
            "m = types.ModuleType('v835_zero3'); m.__file__ = sys.argv[1]\n"
            "exec(compile(Path(sys.argv[1]).read_bytes(), sys.argv[1], 'exec'), m.__dict__)\n"
            "class Helper:\n"
            "    def __init__(self, code): self.code = code\n"
            "    def poll(self): return self.code\n"
            "    def kill(self): print('killed', self.code, flush=True)\n"
            "m.WORLD['helpers'] = (Helper(None), Helper(0), Helper(71))\n"
            "threading.Thread(target=m._watch_helpers, daemon=True).start()\n"
            "time.sleep(60)\n")
        started = time.monotonic()
        proc = subprocess.run([sys.executable, '-B', '-c', code, str(STAGE / 'v835_zero3.py')], capture_output=True, text=True,
                              timeout=120, env=child_env())
        self.assertEqual(proc.returncode, 70, proc.stderr[-800:])
        self.assertLess(time.monotonic() - started, 30)
        event = json.loads(next(line for line in proc.stdout.splitlines() if line.startswith('{')))
        self.assertEqual((event['event'], event['rank'], event['returncode']), ('zero3-world-abort', 3, 71))


class TestShardedNumerics(unittest.TestCase):
    """torch FSDP on the production-class model, two processes, against the v8.3.4e gradient definition."""

    def run_tool(self, dtype):
        return train_shard.run_numerics(world=2, dtype=dtype, seed=3)

    def test_production_precision_bf16_forward_fp32_reduction(self):
        out = self.run_tool('bfloat16')
        self.assertTrue(out['ok'], out)
        self.assertTrue(out['all_sharded'] and out['names_equal'] and out['weights_exact'])
        self.assertEqual(out['native_dtypes'], ['torch.float32'])        # layer shards: fp32, gathered in bf16
        self.assertEqual(out['explicit_dtypes'], ['torch.bfloat16'])     # embedding / lm_head shards stay bf16
        self.assertLessEqual(out['reduction']['rel_diff'], 1e-6)         # a bf16 rounding of the reduction would be ~4e-3
        self.assertEqual([2 * b for b in out['shard_bytes']], [out['full_bytes']] * 2)
        self.assertGreaterEqual(max(out['null_steps']), 1)               # lockstep zero-weight steps did run

    def test_fp32_matches_one_unsharded_process(self):
        out = self.run_tool('float32')
        self.assertTrue(out['ok'], out)
        self.assertTrue(out['losses_equal'])
        self.assertLessEqual(out['unsharded']['rel_diff'], 1e-6)
        self.assertLessEqual(out['reduction']['rel_diff'], 1e-6)

    def test_layer_inputs_behind_the_host_hooks_give_the_same_gradient_bit_for_bit(self):
        """No GPU here, so the host copy is a CPU-to-CPU copy: this checks the autograd plumbing and the values."""
        work = Path(tempfile.mkdtemp(prefix='v834f-inputs-'))
        self.addCleanup(shutil.rmtree, work, True)
        plain = train_shard.run_numerics(world=2, dtype='float32', seed=3, work=work / 'plain')
        hooked = train_shard.run_numerics(world=2, dtype='float32', seed=3, work=work / 'hooked', offload_inputs=True)
        self.assertTrue(plain['ok'] and hooked['ok'], (plain, hooked))
        self.assertEqual(plain['packed_inputs'], [0, 0])
        # every step, real or zero-weight, hands the input of every layer to the hooks, on every rank
        self.assertEqual(hooked['packed_inputs'], [hooked['layers'] * hooked['steps']] * 2)
        self.assertGreaterEqual(hooked['layers'] * hooked['steps'], 8)
        first, second = torch.load(plain['gradient_file']), torch.load(hooked['gradient_file'])
        self.assertEqual(list(first), list(second))
        self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))
        self.assertLessEqual(hooked['unsharded']['rel_diff'], 1e-6)


class TestShardedInstallEndToEnd(unittest.TestCase):
    """The staged launcher's real install(start_dp=True) with one helper rank, one sharded update, against the
    unsharded update of the admitted host-v834e files on the same head, batch and fp32 master."""

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='v834f-e2e-'))
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        for mode, args, env in (('prepare', (), child_env()), ('reference', (), child_env()),
                                ('plain', (), child_env(PIT_PER_GPU_MEM_GIB='180')),
                                ('sharded', ('--port', str(free_port())), child_env(PIT_PER_GPU_MEM_GIB='32')),
                                ('fp32-load', ('--source', 'sharded'), child_env()),
                                ('fp32-load', ('--source', 'reference'), child_env())):
            proc, seconds = run_driver(cls.work, mode, *args, env=env)
            if proc.returncode != 0:
                raise AssertionError('%s failed (rc %s):\n%s\n%s' % (mode, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
            if mode == 'sharded':
                cls.sharded_log, cls.sharded_seconds = proc.stdout, seconds
        load = lambda mode: json.loads((cls.work / mode / 'result.json').read_text())
        cls.ref, cls.plain, cls.out = load('reference'), load('plain'), load('sharded')
        cls.loaded, cls.loaded_ref = load('fp32-load-sharded'), load('fp32-load-reference')

    def master_entry(self, run):
        """(directory, meta.json) of the candidate fp32 master that `run` saved into its own store."""
        result = self.out if run == 'sharded' else self.ref
        directory = self.work / run / 'fp32-master' / result['fp32']['output_store_entry']
        return directory, json.loads((directory / 'meta.json').read_text())

    def test_no_hang_and_helpers_start_through_the_zero3_entry(self):
        self.assertLess(self.sharded_seconds, 300)
        self.assertEqual(self.out['helper_returncodes'], [0])
        (argv,) = self.out['helper_argv']
        self.assertEqual(argv[1], '-B')
        self.assertTrue(argv[2].endswith('/host-v834f/v835_zero3.py'))
        self.assertEqual(argv[3:6], ['helper', '1', '2'])
        self.assertTrue(argv[7].endswith('/host-v834f/v83_dp.py'))
        self.assertTrue(re.search(r'train_pit.py|v83_dp.py', ' '.join(argv)))       # the box clean-up pattern still finds it
        events = [json.loads(line) for line in self.sharded_log.splitlines() if line.startswith('{"component": "trainer-host-v834f"')]
        self.assertEqual([e['event'] for e in events], ['helper-zero3', 'helper-sharded'])
        self.assertEqual(self.out['forward'], 'make_train_update.<locals>.log_probs')

    def test_activate_needs_the_launcher_and_decides_from_the_gpu(self):
        self.assertTrue(self.out['activate_none'].startswith('refused:'))
        self.assertEqual(self.out['policy']['per_gpu_mem_gib'], 32.0)
        self.assertTrue(self.out['policy']['zero3'])
        self.assertEqual(self.out['zero3_env'], '1')
        self.assertEqual(self.out['install'], dict(dp_world=2, zero3=True, dp_token_budget=24576))
        # the same host-v834f files on a GPU that fits the update: the v8.3.4e update, budget untouched
        self.assertFalse(self.plain['policy']['zero3'])
        self.assertIsNone(self.plain['zero3_env'])
        self.assertEqual(self.plain['install'], dict(dp_world=0, zero3=False, dp_token_budget=40960))
        self.assertEqual(self.plain['forward'], 'make_train_update.<locals>.fast_blp')

    def test_live_accumulator_signature_and_no_typeerror(self):
        self.assertEqual(self.out['fp32_accumulator_signature'], '(self, named_parameters)')
        self.assertEqual(self.out['status'], 'updated')
        self.assertEqual(self.out['optimizer_step'], 8)

    def test_token_cut_ran_in_the_plan(self):
        micro = self.out['microbatch']
        self.assertEqual(micro['engine'], 'packed-zero3-v835')
        self.assertEqual(micro['token_budget'], 24576)
        self.assertEqual(self.ref['microbatch']['token_budget'], 40960)
        self.assertEqual(micro['logit_chunk'], 1024)
        # the planner the update calls: two 14,000-token rollouts share a bin under 40,960 and not under 24,576
        self.assertEqual(self.out['plan_probe'], dict(bins=4, largest_bin_tokens=14000))
        self.assertEqual(self.plain['plan_probe'], dict(bins=2, largest_bin_tokens=28000))

    def test_lockstep_and_sharded_parameter_sizes(self):
        micro = self.out['microbatch']
        self.assertEqual(micro['world'], 2)
        self.assertGreaterEqual(micro['steps'], 2)
        self.assertGreaterEqual(micro['count'], micro['steps'] + 1)                 # more micro-batches than steps: both ranks worked
        null = [r['null_steps'] for r in micro['ranks']]
        self.assertEqual(sum(null), micro['steps'] * 2 - micro['count'])
        self.assertGreaterEqual(max(null), 1)                                       # the shorter rank ran a zero-weight step
        head = load_file(str(self.work / 'head' / 'model.safetensors'))
        full = sum(t.numel() * 4 for k, t in head.items() if k.startswith(('model.language_model.', 'lm_head.')))
        self.assertEqual([r['shard_bytes'] for r in micro['ranks']], [full // 2, full // 2])

    def test_loss_and_gradient_norm_match_the_unsharded_update(self):
        self.assertAlmostEqual(self.out['objective_loss'], self.ref['objective_loss'], delta=1e-6 * abs(self.ref['objective_loss']))
        self.assertAlmostEqual(self.out['preclip_gradient_norm'], self.ref['preclip_gradient_norm'],
                               delta=1e-6 * self.ref['preclip_gradient_norm'])
        self.assertEqual(sorted(self.out['losses']), sorted(self.ref['losses']))
        for key, value in self.ref['losses'].items():
            self.assertAlmostEqual(self.out['losses'][key], value, delta=1e-6 * max(1.0, abs(value)))
        self.assertEqual(self.out['trained_rollouts'], self.ref['trained_rollouts'])
        self.assertEqual(self.out['trained_tokens'], self.ref['trained_tokens'])
        self.assertAlmostEqual(self.out['raw_update_norm'], self.ref['raw_update_norm'], delta=1e-6 * self.ref['raw_update_norm'])
        self.assertEqual(self.out['trajectory_loss_records'], self.out['trained_rollouts'])
        self.assertEqual(self.out['train_params']['kl_applied'], 0.0)               # the host wrappers still run

    def test_candidate_checkpoint_has_the_unsharded_keys_dtypes_and_shapes(self):
        self.assertEqual(self.out['checkpoint_files'], self.ref['checkpoint_files'])
        got = load_file(self.out['model_dir'] + '/model.safetensors')
        want = load_file(self.ref['model_dir'] + '/model.safetensors')
        head = load_file(str(self.work / 'head' / 'model.safetensors'))
        self.assertEqual(list(got), list(want))
        self.assertEqual(list(got), list(head))
        self.assertFalse([k for k in got if 'fsdp' in k.lower()])
        for key in want:
            self.assertEqual((got[key].dtype, tuple(got[key].shape)), (want[key].dtype, tuple(want[key].shape)), key)
        moved = max(float((want[k] - head[k]).abs().max()) for k in want)
        apart = max(float((got[k] - want[k]).abs().max()) for k in want)
        self.assertGreater(moved, 1e-7)                                             # the step did change the weights
        self.assertLess(apart, 1e-2 * moved)                                        # and both updates moved them the same way
        frozen = [k for k in want if not k.startswith(('model.language_model.', 'lm_head.'))]
        self.assertTrue(frozen and all(torch.equal(got[k], head[k]) for k in frozen))

    def test_fp32_master_semantics(self):
        fp = self.out['fp32']
        self.assertEqual(fp['source'], 'disk')
        self.assertEqual(fp['input_master_digest'], fp['expected'])
        self.assertEqual(fp['input_master_digest'], self.ref['fp32']['input_master_digest'])
        self.assertEqual(self.out['actor_device'], 'cpu')

    def test_fp32_master_is_saved_once_by_the_trainer_process_where_the_upload_path_looks(self):
        fp = self.out['fp32']
        self.assertIsNone(fp['finish_error'])
        self.assertEqual(fp['store'], str(self.work / 'sharded' / 'fp32-master'))
        # the upload path resolves <store>/<output_store_entry>; host-v834e names the entry after the candidate weights
        self.assertEqual(fp['output_store_entry'], fp['output_bf16_sha256'])
        self.assertEqual(len(fp['output_master_digest']), 64)
        # while the gradients are computed the store holds the head's master only; the candidate's master appears
        # once, after the one optimizer step (the same two states a poller sees with host-v834e)
        self.assertEqual(self.out['store_during_training'], [[fp['input_bf16_sha256']]])
        self.assertEqual(self.ref['store_during_training'], [[self.ref['fp32']['input_bf16_sha256']]])
        self.assertEqual(sorted(self.out['store_after']), sorted({fp['input_bf16_sha256'], fp['output_store_entry']}))
        saved = [json.loads(line) for line in self.sharded_log.splitlines() if '"fp32-master-saved"' in line]
        self.assertEqual([row['entry'] for row in saved], [fp['output_store_entry']])
        self.assertNotIn('present', saved[0])                                       # written by this run, not found there

    def test_fp32_master_has_the_v834e_file_layout_and_keys(self):
        """The entry the sharded run wrote against the entry the unsharded host-v834e run wrote (same head, same batch)."""
        got_dir, got = self.master_entry('sharded')
        want_dir, want = self.master_entry('reference')
        files = sorted(p.name for p in got_dir.iterdir())
        self.assertEqual(files, sorted(p.name for p in want_dir.iterdir()))
        shards = [name for name in files if name != 'meta.json']
        self.assertGreaterEqual(len(shards), 3)                                     # several weight files, as a 36 GB master has
        self.assertEqual(shards, ['fp32-master-%05d-of-%05d.safetensors' % (i + 1, len(shards)) for i in range(len(shards))])
        self.assertEqual(sorted(got), MASTER_META_KEYS)
        self.assertEqual(sorted(want), MASTER_META_KEYS)
        for key in ('version', 'tensor_file', 'shard_bytes', 'optimizer_step', 'source', 'input_bf16_sha256', 'input_master_digest'):
            self.assertEqual(got[key], want[key], key)
        self.assertEqual(got['version'], 'v833f-fp32-head-master-v1')
        self.assertEqual(got['bf16_sha256'], got_dir.name)
        self.assertEqual(got['master_digest'], self.out['fp32']['output_master_digest'])
        self.assertEqual(sorted(got['files']), shards)
        for name in shards:
            self.assertEqual(got['files'][name], dict(sha256=sha256(got_dir / name), bytes=(got_dir / name).stat().st_size))
        head = load_file(str(self.work / 'head' / 'model.safetensors'))
        trainable = [key for key in head if key.startswith(('model.language_model.', 'lm_head.'))]
        seen = []
        for name in shards:
            tensors, reference = load_file(str(got_dir / name)), load_file(str(want_dir / name))
            self.assertEqual(list(tensors), list(reference), name)                  # same keys in the same file, same order
            with safe_open(str(got_dir / name), 'pt') as handle:
                self.assertEqual(handle.metadata(), dict(format='pt', fp32_master=got['version']))
            for key, tensor in tensors.items():
                self.assertEqual(got['tensor_file'][key], name)
                self.assertEqual((tensor.dtype, tuple(tensor.shape)), (torch.float32, tuple(head[key].shape)), key)
                self.assertEqual(hashlib.sha256(tensor.numpy().tobytes()).hexdigest(), got['tensor_sha256'][key], key)
                seen.append(key)
        self.assertEqual(sorted(seen), sorted(trainable))                           # the host-v834e names: no FSDP name anywhere
        self.assertEqual(sorted(got['tensor_file']), sorted(trainable))
        digest = hashlib.sha256(json.dumps(got['tensor_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, got['master_digest'])
        # the two updates agree to rounding, so the two masters do too (they are not required to be the same bytes)
        moved = max(float((load_file(str(want_dir / got['tensor_file'][k]))[k] - head[k]).abs().max()) for k in trainable[:8])
        apart = max(float((load_file(str(got_dir / got['tensor_file'][k]))[k]
                           - load_file(str(want_dir / got['tensor_file'][k]))[k]).abs().max()) for k in trainable)
        self.assertGreater(moved, 1e-7)
        self.assertLess(apart, 1e-2 * moved)

    def test_sharded_save_round_trips_through_the_admitted_v834e_fp32_load(self):
        """A fresh process with the ADMITTED host-v834e files builds the optimizer on the candidate the sharded run
        published and loads the master the sharded run saved, with host-v834e's own _fp32_load."""
        admitted = json.loads(ADMITTED.read_text())['host_files']
        launcher = Path(self.loaded['launcher'])
        self.assertEqual(launcher.parent.name, 'host-v834e')
        self.assertEqual(sha256(launcher), admitted['mimo_host_trainer_v83.py'])    # the admitted bytes ran, not the staged file
        self.assertNotIn('v835_zero3.py', self.loaded['host_files'])
        fp, loaded = self.out['fp32'], self.loaded['loaded']
        self.assertEqual(self.loaded['candidate'], self.out['checkpoint'])
        self.assertEqual(self.loaded['entry'], fp['output_store_entry'])            # host-v834e derives the same entry name
        self.assertEqual(self.loaded['store'], fp['store'])
        self.assertTrue(loaded['accepted'], loaded)
        self.assertEqual(loaded['events'], [])                                      # no rejection event
        self.assertEqual(sorted(loaded['meta']), MASTER_META_KEYS)
        self.assertEqual(loaded['meta']['master_digest'], fp['output_master_digest'])
        self.assertEqual(self.loaded['master_digest_after_load'], fp['output_master_digest'])
        _, meta = self.master_entry('sharded')
        self.assertEqual(self.loaded['master_tensor_sha256'], meta['tensor_sha256'])
        self.assertEqual(self.loaded['master_tensors'], len(meta['tensor_file']))
        self.assertEqual(self.loaded['master_dtypes'], ['torch.float32'])
        self.assertTrue(self.loaded['master_nonzero'])                              # the zeroed masters were filled from the files
        self.assertTrue(self.loaded['master_projects_to_candidate'])
        # the same loader does refuse real files that are wrong
        damaged, wrong = self.loaded['damaged_file'], self.loaded['wrong_weights']
        self.assertFalse(damaged['accepted'])
        self.assertIn('load-rejected:master digest', damaged['error'])
        self.assertFalse(wrong['accepted'])
        self.assertIn('load-rejected:projection', wrong['error'])
        self.assertEqual([row['event'] for row in damaged['events'] + wrong['events']], ['fp32-master-load-rejected'] * 2)
        # and it accepts the entry of the unsharded host-v834e run the same way
        self.assertTrue(self.loaded_ref['loaded']['accepted'])
        self.assertEqual(self.loaded_ref['loaded']['meta']['master_digest'], self.ref['fp32']['output_master_digest'])

    def test_upload_path_inventory_accepts_the_sharded_master_entry(self):
        """ops/parallel_eval/live/pe_artifact_upload.inventory_box_master on the real entry (box = this machine)."""
        directory, meta = self.master_entry('sharded')
        inventory = pe_artifact_upload.inventory_box_master(LocalBox(), str(directory))
        self.assertEqual(sorted(inventory), sorted(list(meta['files']) + ['meta.json']))
        for name, row in meta['files'].items():
            self.assertEqual(inventory[name], row, name)                            # the hashes the trainer recorded
        self.assertEqual(inventory['meta.json']['sha256'], sha256(directory / 'meta.json'))
        with self.assertRaises(pe_artifact_upload.ArtifactUploadError):            # the same check refuses a tree without weights
            pe_artifact_upload.assert_master_tree_complete({'meta.json': inventory['meta.json']})


class TestMasterAtProductionPrecision(unittest.TestCase):
    """bf16 weights and an fp32 master with a residual. The launcher runs above cannot have that: the real runtime
    loads fp32 on a CPU. So this is NOT a sharded launcher run. Two FSDP ranks compute the bf16 gradient
    (zero3_numerics.py); then one process runs the host-v834f launcher's own optimizer step on CPU bf16 weights,
    as the sharded update runs it on rank 0, and the admitted host-v834e _fp32_load reads the saved master."""

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='v834f-bf16-'))
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        proc, _ = run_driver(cls.work, 'prepare')
        if proc.returncode != 0:
            raise AssertionError('prepare failed:\n%s' % proc.stderr[-3000:])
        cls.gradient = train_shard.run_numerics(world=2, dtype='bfloat16', seed=3, work=cls.work)
        proc, _ = run_driver(cls.work, 'bf16-master')
        if proc.returncode != 0:
            raise AssertionError('bf16-master failed (rc %s):\n%s\n%s' % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
        cls.out = json.loads((cls.work / 'bf16-master' / 'result.json').read_text())

    def test_step_uses_the_stored_master_and_saves_the_candidate_master(self):
        out = self.out
        self.assertTrue(self.gradient['ok'], self.gradient)
        self.assertEqual(out['weight_dtypes'], ['torch.bfloat16'])
        self.assertEqual(out['actor_device'], 'cpu')
        self.assertEqual(out['gradient_tensors'], self.gradient['tensors'])
        self.assertGreater(out['head_residual_elements'], 0.9 * out['elements'])    # the head master is not the plain upcast
        self.assertEqual(out['source'], 'disk')
        self.assertEqual(out['input_master_digest'], out['head_master_digest'])
        self.assertIsNone(out['finish_error'])
        self.assertGreater(out['projected_changed_elements'], 0)                    # the step changed some bf16 weights
        self.assertNotEqual(out['output_store_entry'], out['head_entry'])
        self.assertEqual(sorted(out['store_after']), sorted({out['head_entry'], out['output_store_entry']}))

    def test_saved_master_round_trips_through_the_admitted_v834e_fp32_load_with_its_residual(self):
        out = self.out
        admitted = json.loads(ADMITTED.read_text())['host_files']
        self.assertEqual(sha256(out['admitted_launcher']), admitted['mimo_host_trainer_v83.py'])
        self.assertEqual(Path(out['admitted_launcher']).parent, TRAINER_E)
        self.assertEqual(out['entry'], out['output_store_entry'])
        self.assertEqual(out['rejected'], [])
        self.assertEqual(out['meta_keys'], MASTER_META_KEYS)
        self.assertEqual(out['meta_master_digest'], out['output_master_digest'])
        self.assertEqual(out['loaded_master_digest'], out['output_master_digest'])
        self.assertEqual(out['master_dtypes'], ['torch.float32'])
        self.assertTrue(out['projects_to_candidate'])                               # bf16(master) == candidate weights, bit for bit
        self.assertTrue(out['equals_memory_master'])                                # the file holds what the step left in memory
        self.assertGreater(out['residual_elements'], 0.9 * out['elements'])         # and that is more than the bf16 weights hold
        self.assertNotEqual(out['output_master_digest'], out['output_store_entry']) # master digest != weights digest here


class TestShardedFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='v834f-fail-'))
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        for mode in ('prepare', 'reference'):
            proc, _ = run_driver(cls.work, mode)
            if proc.returncode != 0:
                raise AssertionError('%s failed:\n%s' % (mode, proc.stderr[-3000:]))
        cls.ref = json.loads((cls.work / 'reference' / 'result.json').read_text())

    def test_step_boundary_error_stops_all_ranks_together_and_the_world_stays_usable(self):
        proc, _ = run_driver(self.work, 'sharded', '--port', str(free_port()), '--tag', 'guard', '--guard-fail', '3',
                             env=child_env(PIT_ZERO3='1'))
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        out = json.loads((self.work / 'guard' / 'result.json').read_text())
        self.assertIn('failed before the optimizer step', out['first_error'])
        self.assertIn('lease lost', out['first_error'])
        self.assertEqual(out['helper_alive_after_error'], [True])
        self.assertEqual(out['status'], 'updated')                                  # the retry in the same world
        self.assertAlmostEqual(out['preclip_gradient_norm'], self.ref['preclip_gradient_norm'],
                               delta=1e-6 * self.ref['preclip_gradient_norm'])
        self.assertEqual(out['helper_returncodes'], [0])

    def test_a_rank_over_the_limit_moves_its_layer_inputs_to_the_host_and_every_rank_measures_again(self):
        """No GPU here: the harness stands in for rank 0's measurement. The helper rank measures nothing."""
        proc, _ = run_driver(self.work, 'sharded', '--port', str(free_port()), '--tag', 'retry', '--fake-peak', 'first-over',
                             env=child_env(PIT_PER_GPU_MEM_GIB='32'))
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        out = json.loads((self.work / 'retry' / 'result.json').read_text())
        self.assertNotIn('first_error', out)
        first, second = out['microbatch']['ranks']
        self.assertEqual([p['offload_inputs'] for p in first['probes']], [False, True])     # over the limit, then under it
        self.assertEqual([p['reserved_gib'] for p in first['probes']], [30.0, 13.0])
        self.assertTrue(first['offload_inputs'])
        self.assertEqual([p['offload_inputs'] for p in second['probes']], [False, False])   # in step with rank 0, own mode kept
        self.assertFalse(second['offload_inputs'])
        self.assertEqual(out['status'], 'updated')
        self.assertAlmostEqual(out['objective_loss'], self.ref['objective_loss'], delta=1e-6 * abs(self.ref['objective_loss']))
        self.assertAlmostEqual(out['preclip_gradient_norm'], self.ref['preclip_gradient_norm'],
                               delta=1e-6 * self.ref['preclip_gradient_norm'])
        self.assertEqual(out['helper_returncodes'], [0])

    def test_a_probe_that_stays_over_the_limit_is_refused_before_any_gradient(self):
        proc, _ = run_driver(self.work, 'sharded', '--port', str(free_port()), '--tag', 'refuse', '--fake-peak', 'always-over',
                             env=child_env(PIT_PER_GPU_MEM_GIB='32'))
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        out = json.loads((self.work / 'refuse' / 'result.json').read_text())
        self.assertTrue(out['first_error'].startswith('VramPreflightError: sharded update refused before any gradient'))
        self.assertIn('probe peak 30.00 GiB reserved', out['first_error'])
        self.assertIn('exceeds VRAM limit 28.0 GiB', out['first_error'])
        self.assertIn('layer inputs on the host', out['first_error'])
        self.assertEqual(out['helper_alive_after_error'], [True])
        self.assertEqual(out['optimizer_step'], 8)                                   # the refusal took no optimizer step
        self.assertEqual(out['status'], 'updated')                                  # the same world then runs the update
        self.assertAlmostEqual(out['preclip_gradient_norm'], self.ref['preclip_gradient_norm'],
                               delta=1e-6 * self.ref['preclip_gradient_norm'])

    def test_a_dead_helper_ends_the_trainer_instead_of_hanging(self):
        proc, seconds = run_driver(self.work, 'sharded', '--port', str(free_port()), '--tag', 'kill', '--kill-helper',
                                   env=child_env(PIT_ZERO3='1'), timeout=300)
        self.assertNotEqual(proc.returncode, 0)
        self.assertLess(seconds, 120)
        self.assertFalse((self.work / 'kill' / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
