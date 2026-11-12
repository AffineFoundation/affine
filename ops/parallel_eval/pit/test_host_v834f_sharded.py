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
from ops.parallel_eval.pit import compare_train_reports, train_shard
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


class TestMasterOwnership(unittest.TestCase):
    """The fp32 master is not sharded. Ranks 1..W-1 cannot save one: their process has neither a master nor the
    code that writes one. Only the launcher in the trainer process (rank 0) loads, steps and saves it."""

    def test_the_sharded_module_never_touches_the_master_or_the_disk(self):
        source = (STAGE / 'v835_zero3.py').read_text()
        tree = ast.parse(source)
        body = source.replace(ast.get_docstring(tree, clean=False), '')
        self.assertLess(len(body), len(source))
        for needle in ('_fp32', 'masters', 'mimo_host_trainer_v83', 'save_file', 'save_pretrained', 'torch.save',
                       'write_text', 'write_bytes', 'open('):
            self.assertNotIn(needle, body, needle)

    def test_a_helper_rank_loads_the_dp_engine_files_only(self):
        tree = ast.parse((STAGE / 'v835_zero3.py').read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'helper_main')
        loaded = sorted(node.args[1].value for node in ast.walk(helper)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'load'
                        and len(node.args) == 2)
        self.assertEqual(loaded, ['v8211_batched.py', 'v83_dp.py', 'v83_fast_kernels.py', 'v83_kernels_policy.py'])


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
        self.assertEqual(public['zero3']['logit_chunk'], zero3.LOGIT_CHUNK)
        self.assertEqual(public['host_files_sha256']['v835_zero3.py'], sha256(STAGE / 'v835_zero3.py'))
        # the settings of the sharded update are published as the stage holds them: both sets, and which is the default
        self.assertEqual(public['zero3']['profile_env'], 'PIT_ZERO3_PROFILE')
        self.assertEqual(public['zero3']['default_profile'], zero3.DEFAULT_PROFILE)
        self.assertEqual(public['zero3']['profiles'], {name: profile._asdict() for name, profile in zero3.PROFILES.items()})
        self.assertEqual(public['zero3']['token_budget'], zero3.PROFILES[zero3.DEFAULT_PROFILE].token_budget)

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


def stage_with_profile(name):
    """A fresh copy of the stage module in a process environment that names the settings `name` (None = no name).
    The module fixes its settings at the first call of settings(), which runs here."""
    saved = os.environ.pop('PIT_ZERO3_PROFILE', None)
    try:
        if name is not None:
            os.environ['PIT_ZERO3_PROFILE'] = name
        module = load_source(STAGE / 'v835_zero3.py', 'v835_profile_probe')
        module.settings()
        return module
    finally:
        os.environ.pop('PIT_ZERO3_PROFILE', None)
        if saved is not None:
            os.environ['PIT_ZERO3_PROFILE'] = saved


class TestSettings(unittest.TestCase):
    """The values the real 8x RTX 5090 runs set are two named sets. A process runs one of them, whole."""

    U6 = dict(name='rtx5090-u6', token_budget=32768, offload_params=False, pin_inputs=True, layer_reduce='native',
              prefetch=1, gather='data')
    U2 = dict(name='reference-u2', token_budget=24576, offload_params=True, pin_inputs=False, layer_reduce='float32',
              prefetch=0, gather='control')

    def test_the_two_sets_and_the_default(self):
        zero3 = stage_with_profile(None)
        self.assertEqual({name: profile._asdict() for name, profile in zero3.PROFILES.items()},
                         {'rtx5090-u6': self.U6, 'reference-u2': self.U2})
        self.assertEqual(zero3.DEFAULT_PROFILE, 'rtx5090-u6')
        self.assertEqual(zero3.settings()._asdict(), self.U6)                       # no name in the environment: the default
        self.assertEqual(zero3.LOGIT_CHUNK, 1024)                                   # a chunk of 4096 ran out of GPU memory with two layers ahead

    def test_the_environment_names_a_whole_set_and_the_first_call_fixes_it(self):
        zero3 = stage_with_profile('reference-u2')
        self.assertNotIn('PIT_ZERO3_PROFILE', os.environ)                           # the name is gone again
        self.assertEqual(zero3.settings()._asdict(), self.U2)                       # the process keeps the set it began with
        self.assertEqual(zero3.token_budget(40960), 24576)
        self.assertEqual(zero3.token_budget(16384), 16384)                          # never above the signed engine budget
        default = stage_with_profile(None)
        self.assertEqual(default.token_budget(40960), 32768)
        self.assertEqual(default.token_budget(16384), 16384)

    def test_an_unknown_name_is_refused(self):
        for name in ('rtx5090', 'RTX5090-U6', 'reference', 'native'):               # never the nearest or the default set
            with self.assertRaises(ValueError) as ctx:
                stage_with_profile(name)
            self.assertIn('unknown PIT_ZERO3_PROFILE %r' % name, str(ctx.exception))

    def test_no_single_value_can_change_and_what_failed_on_the_box_cannot_be_set(self):
        zero3 = stage_with_profile(None)
        u6 = zero3.PROFILES['rtx5090-u6']
        with self.assertRaises(AttributeError):
            u6.token_budget = 65536
        changed = lambda **change: zero3.Settings(**dict(u6._asdict(), **change))
        self.assertEqual(changed(), u6)
        with self.assertRaises(ValueError):
            changed(prefetch=2)                                                     # two layers ahead: out of GPU memory on the box
        for change in (dict(layer_reduce='float16'), dict(gather='gloo'), dict(token_budget=0), dict(token_budget=32768.0)):
            with self.assertRaises(ValueError):
                changed(**change)
        # the stage reads no other switch for these values
        body = (STAGE / 'v835_zero3.py').read_text()
        self.assertEqual(sorted(set(re.findall(r"'(PIT_[A-Z0-9_]+|CUDA_VISIBLE_MEM_GIB)'", body))),
                         ['CUDA_VISIBLE_MEM_GIB', 'PIT_PER_GPU_MEM_GIB', 'PIT_ZERO3', 'PIT_ZERO3_PROFILE'])
        self.assertNotIn('torch.compile', body)                                     # no compile hook in the sharded update


class TestHostBlocks(unittest.TestCase):
    """The store of the page-locked layer inputs (v835_zero3._HostBlocks). No GPU here: the blocks are ordinary
    host memory, so this checks the block arithmetic, the reuse and the bits. The block size is 64 bytes."""

    def setUp(self):
        self.zero3 = load_source(STAGE / 'v835_zero3.py', 'v835_blocks_probe')
        self.assertEqual(self.zero3.HOST_BLOCK_BYTES, 1 << 24)
        self.zero3.HOST_BLOCK_BYTES = 64
        self.generator = torch.Generator().manual_seed(11)

    def random(self, shape, dtype):
        return torch.randn(shape, generator=self.generator).to(dtype)

    @staticmethod
    def raw(tensor):
        return tensor.contiguous().view(-1).view(torch.uint8)

    def test_a_tensor_comes_back_bit_for_bit_from_whole_and_part_blocks(self):
        pool = self.zero3._HostBlocks(pin=False)
        # 160 bytes: 2 blocks and a half. 256 bytes: 4 whole blocks. 672 bytes: 10 blocks and a half.
        tensors = [self.random((2, 5, 8), torch.bfloat16), self.random((1, 16, 8), torch.bfloat16), self.random((3, 7, 8), torch.float32)]
        packed = [pool.store(tensor) for tensor in tensors]
        self.assertEqual([pack[3] for pack in packed], [0, 3, 7])                   # the first block of each tensor
        self.assertEqual((pool.used, len(pool.blocks)), (18, 18))
        self.assertTrue(all(block.numel() == 64 and block.dtype == torch.uint8 for block in pool.blocks))
        for tensor, pack in reversed(list(zip(tensors, packed))):                   # backward asks in the other order
            back = pool.load(pack)
            self.assertEqual((back.dtype, tuple(back.shape), back.device), (tensor.dtype, tuple(tensor.shape), tensor.device))
            self.assertTrue(torch.equal(self.raw(back), self.raw(tensor)))
            self.assertIsNot(back, tensor)

    def test_a_new_step_fills_the_same_blocks_and_an_input_of_the_step_before_is_refused(self):
        pool = self.zero3._HostBlocks(pin=False)
        old = pool.store(torch.arange(100, dtype=torch.float32))                    # 400 bytes: 7 blocks
        pool.reset()
        value = torch.arange(40, dtype=torch.float32) + 0.5
        new = pool.store(value)                                                     # 160 bytes: 3 of the same 7 blocks
        self.assertEqual((pool.used, len(pool.blocks)), (3, 7))
        self.assertTrue(torch.equal(pool.load(new), value))
        with self.assertRaises(ValueError) as ctx:                                  # its blocks hold other data by now
            pool.load(old)
        self.assertIn('earlier micro-batch', str(ctx.exception))

    def test_the_hooks_put_what_fills_a_block_into_the_store_and_copy_the_rest(self):
        pool = self.zero3._HostBlocks(pin=False)
        hooks = self.zero3._SavedInputsOnHost(pool)
        large, small = self.random((4, 8, 8), torch.float32), self.random((3, 5), torch.float32)    # 1024 bytes, 60 bytes
        in_blocks, own_copy = hooks._pack(large), hooks._pack(small)
        self.assertEqual((len(in_blocks), len(own_copy), pool.used), (5, 2, 16))
        self.assertTrue(torch.equal(hooks._unpack(in_blocks), large) and torch.equal(hooks._unpack(own_copy), small))
        # without a store (the reference settings) every tensor is a copy of its own
        plain = self.zero3._SavedInputsOnHost()
        copied = plain._pack(large)
        self.assertEqual(len(copied), 2)
        self.assertIsNot(copied[1], large)
        self.assertTrue(torch.equal(plain._unpack(copied), large))

    def test_autograd_gets_the_saved_tensor_back_from_the_blocks(self):
        pool = self.zero3._HostBlocks(pin=False)
        x = self.random((4, 8, 8), torch.float32).requires_grad_()
        with self.zero3._SavedInputsOnHost(pool):
            y = (x * x).sum()
        self.assertGreaterEqual(pool.used, 16)                                      # autograd handed x to the store
        y.backward()
        self.assertTrue(torch.equal(x.grad, x.detach() + x.detach()))


class TestMemoryModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zero3 = stage_with_profile(None)

    def test_reference_geometry_is_the_pinned_9b_configuration(self):
        with torch.device('meta'):
            model = Qwen3_5ForConditionalGeneration(Qwen3_5Config())
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('model.language_model.') or name.startswith('lm_head.'))
        self.assertEqual(self.zero3.inventory(model), self.zero3.REFERENCE_9B)

    def test_sharded_estimate_is_not_total_over_world(self):
        inv = self.zero3.REFERENCE_9B
        total = sum(inv[k] for k in ('layer_sum', 'embed', 'head', 'other', 'frozen')) * 2 / 2 ** 30
        for profile in ('rtx5090-u6', 'reference-u2'):
            out = self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=8, inv=inv, profile=profile)
            self.assertEqual(out['profile'], profile)
            self.assertEqual(out['tokens'], 36864)                      # one rollout of the full training context
            self.assertGreater(out['certain_gib'], total)               # every rank loads the whole actor once
            self.assertGreater(out['certain_gib'], 3 * total / 8)       # far above "weights and gradients divided by 8"
            self.assertLess(out['estimate_gib'], out['limit_gib'])
            self.assertEqual(out['limit_gib'], 28.0)
            self.assertFalse(out['coefficients']['measured'])
        # no profile given: the settings of the process (here the default)
        self.assertEqual(self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=8, inv=inv)['profile'], 'rtx5090-u6')
        with self.assertRaises(ValueError):
            self.zero3.estimate_peak_gib(inv, tokens=8192, world=8, profile='rtx5090')

    def test_layer_inputs_go_to_the_host_when_the_estimate_is_near_the_limit(self):
        inv, estimate = self.zero3.REFERENCE_9B, self.zero3.estimate_peak_gib
        budget = self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=8, inv=inv, tokens=24576, profile='reference-u2')
        self.assertFalse(budget['offload_inputs'])                  # a micro-batch of the reference budget: inputs stay on the GPU
        self.assertEqual(budget['offload_host_gib'], 0.0)
        self.assertLess(budget['estimate_gib'], 0.75 * budget['limit_gib'])
        full = self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=8, inv=inv, tokens=36864, profile='reference-u2')
        self.assertTrue(full['offload_inputs'])                     # one full-context rollout: planned with them on the host
        on_gpu = estimate(inv, tokens=36864, world=8, profile='reference-u2')
        self.assertGreater(on_gpu['estimate_gib'], 0.75 * full['limit_gib'])
        self.assertLess(full['phases']['layer']['estimate_gib'], on_gpu['phases']['layer']['estimate_gib'] - 8.0)
        hidden = 36864 * inv['hidden'] * 2 / 2 ** 30
        self.assertAlmostEqual(full['offload_host_gib'], (inv['layers'] + 2 - 3) * hidden, places=6)
        for profile in ('rtx5090-u6', 'reference-u2'):
            for tokens in (8192, 24576, 36864):                     # more tokens never lower the estimate of a mode
                for mode in (False, True):
                    self.assertLessEqual(
                        estimate(inv, tokens=tokens, world=8, offload_inputs=mode, profile=profile)['estimate_gib'],
                        estimate(inv, tokens=tokens + 4096, world=8, offload_inputs=mode, profile=profile)['estimate_gib'])

    def test_the_default_settings_keep_the_layer_shards_on_the_gpu_and_plan_a_full_bin_with_its_inputs_on_the_host(self):
        inv, estimate, gib = self.zero3.REFERENCE_9B, self.zero3.estimate_peak_gib, 2.0 ** 30
        u6 = lambda **kwargs: estimate(inv, world=8, profile='rtx5090-u6', **kwargs)
        u2 = lambda **kwargs: estimate(inv, world=8, profile='reference-u2', **kwargs)
        # resident on every GPU: the fp32 shards of the layers and of the final norm, and their fp32 gradient shards
        on_gpu = 2 * (inv['layer_sum'] + inv['other']) * 4 / gib / 8
        self.assertAlmostEqual(on_gpu, 6.44, places=2)
        for phase in ('head', 'head-reduce', 'embed-reduce'):
            self.assertAlmostEqual(u6(tokens=32768)['phases'][phase]['certain_gib'] - u2(tokens=32768)['phases'][phase]['certain_gib'],
                                   on_gpu, places=2)
        # one layer backward also holds bf16 reduce-scatter inputs in place of fp32 ones: two inputs of half the size
        layer = inv['layer_max'] * 2 / gib
        self.assertAlmostEqual(u6(tokens=32768)['phases']['layer']['certain_gib'] - u2(tokens=32768)['phases']['layer']['certain_gib'],
                               on_gpu - 2 * layer + layer / 8, places=2)
        # a micro-batch of the 32768 budget: over the limit with the layer inputs on the GPU, planned with them on the host
        self.assertGreater(u6(tokens=32768)['estimate_gib'], 28.0)
        plan = self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=8, inv=inv, tokens=32768, profile='rtx5090-u6')
        self.assertTrue(plan['offload_inputs'])
        self.assertLess(plan['estimate_gib'], plan['limit_gib'])
        # on the host an input takes whole blocks of 16 MiB: 32768 tokens fill 16, two tokens more need a 17th
        kept = inv['layers'] + 2 - 3
        self.assertEqual(self.zero3.HOST_BLOCK_BYTES, 1 << 24)
        self.assertAlmostEqual(plan['offload_host_gib'], kept * 16 * 2 ** 24 / gib, places=9)
        self.assertAlmostEqual(u6(tokens=32770, offload_inputs=True)['offload_host_gib'], kept * 17 * 2 ** 24 / gib, places=9)
        self.assertAlmostEqual(u2(tokens=32770, offload_inputs=True)['offload_host_gib'], kept * 32770 * inv['hidden'] * 2 / gib, places=9)

    def test_refuses_what_cannot_fit(self):
        with self.assertRaises(self.zero3.VramPreflightError) as ctx:
            self.zero3.preflight_or_raise(per_gpu_gib=16.0, world=8, inv=self.zero3.REFERENCE_9B)
        self.assertIn('exceeds VRAM', str(ctx.exception))
        with self.assertRaises(self.zero3.VramPreflightError):
            self.zero3.preflight_or_raise(per_gpu_gib=None, world=8, inv=self.zero3.REFERENCE_9B)
        with self.assertRaises(self.zero3.VramPreflightError):
            self.zero3.preflight_or_raise(per_gpu_gib=32.0, world=1, n_params=9.4e9, sharded=False)

    def test_host_memory_model_counts_the_gradient_once(self):
        inv = self.zero3.REFERENCE_9B
        trainable = inv['layer_sum'] + inv['embed'] + inv['head'] + inv['other']
        gib = 2.0 ** 30
        for profile, shards in (('reference-u2', (inv['layer_sum'] + inv['other']) * 4 / gib),      # FSDP keeps the layer shards on the host
                                ('rtx5090-u6', 0.0)):                                                 # they stay on the GPU
            ram = self.zero3.host_ram_gib(inv, profile=profile)
            self.assertAlmostEqual(ram['rank0_gib'], ((trainable + inv['frozen']) * 2 + trainable * 4) / gib)   # actor + fp32 master
            self.assertAlmostEqual(ram['shards_gib'], shards)
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
    """torch FSDP on the production-class model, two processes, against the gradient sum that the settings of the
    run define. With no name the tool runs the default settings (rtx5090-u6), as the trainer does."""

    RUNS = {}

    def run_tool(self, dtype, **kwargs):
        key = (dtype,) + tuple(sorted(kwargs.items()))
        if key not in self.RUNS:
            self.RUNS[key] = train_shard.run_numerics(world=2, dtype=dtype, seed=3, **kwargs)
        return self.RUNS[key]

    def test_reference_settings_bf16_forward_fp32_reduction(self):
        out = self.run_tool('bfloat16', profile='reference-u2')
        self.assertTrue(out['ok'], out)
        self.assertEqual(out['profile']['name'], 'reference-u2')
        self.assertTrue(out['all_sharded'] and out['names_equal'] and out['weights_exact'])
        self.assertEqual(out['native_dtypes'], ['torch.float32'])        # layer shards: fp32, gathered in bf16
        self.assertEqual(out['explicit_dtypes'], ['torch.bfloat16'])     # embedding / lm_head shards stay bf16
        self.assertLessEqual(out['reduction']['rel_diff'], 1e-6)         # a bf16 rounding of the reduction would be ~4e-3
        self.assertEqual(out['reduction'], out['fp32_reduction'])        # the definition of these settings is the v8.3.4e accumulator
        self.assertEqual([2 * b for b in out['shard_bytes']], [out['full_bytes']] * 2)
        self.assertGreaterEqual(max(out['null_steps']), 1)               # lockstep zero-weight steps did run
        # every reduce-scatter has an fp32 input: one in each step for each layer, the final norm, lm_head and the embedding
        self.assertEqual(out['reduce_inputs'], {'torch.float32': (out['layers'] + 3) * out['steps']})
        self.assertEqual(out['gather_groups'], [{'control': out['tensors']}] * 2)       # shards to rank 0 over the control group
        self.assertEqual(out['host_blocks'], [None, None])                              # no store of page-locked blocks

    def test_default_settings_add_the_layer_gradients_in_bf16_and_lm_head_and_the_embedding_in_fp32(self):
        out = self.run_tool('bfloat16')
        self.assertTrue(out['ok'], out)
        self.assertEqual(out['profile']['name'], 'rtx5090-u6')
        self.assertTrue(out['profiles_equal'] and out['reduce_inputs_equal'])           # every rank ran the same settings and collectives
        self.assertTrue(out['all_sharded'] and out['names_equal'] and out['weights_exact'])
        self.assertEqual(out['native_dtypes'], ['torch.float32'])                       # the shards and the gradient shards stay fp32
        self.assertEqual([2 * b for b in out['shard_bytes']], [out['full_bytes']] * 2)
        self.assertGreaterEqual(max(out['null_steps']), 1)
        layers, steps, tensors = out['layers'], out['steps'], out['tensors']
        # reduce-scatter inputs: bf16 for each layer and the final norm, fp32 for lm_head and the embedding
        self.assertEqual(out['reduce_inputs'], {'torch.bfloat16': (layers + 1) * steps, 'torch.float32': 2 * steps})
        # lm_head and the embedding: the fp32 sum over the ranks of each step, bit for bit, which is the v8.3.4e
        # accumulator up to the order of fp32 additions
        self.assertEqual((out['explicit_step_reduction']['tensors'], out['explicit_step_reduction']['exact_tensors']), (2, 2))
        self.assertLessEqual(out['explicit_reduction']['rel_diff'], 1e-6)
        # layers: the two ranks of a step added in bf16 and the steps added in fp32, bit for bit
        self.assertTrue(out['layers_in_bf16'] and out['one_order'])
        self.assertEqual((out['layer_reduction']['tensors'], out['layer_reduction']['exact_tensors']), (tensors - 2, tensors - 2))
        self.assertEqual(out['reduction']['exact_tensors'], tensors)
        # that is not the fp32 sum of the reference settings: it differs by the bf16 rounding of one addition, which
        # is at most 2**-9 of its result (this fixture gave a distance of 1.1e-3)
        far = out['layer_fp32_reduction']
        self.assertLess(far['exact_tensors'], far['tensors'])
        self.assertGreater(far['rel_diff'], 1e-5)
        self.assertLess(far['rel_diff'], 1e-2)
        self.assertGreater(far['cosine'], 0.99999)
        self.assertEqual(out['gather_groups'], [{'data': tensors}] * 2)                 # shards to rank 0 over the data group

    def test_default_settings_gather_the_next_layer_before_the_layer_runs(self):
        """All-gathers a rank has issued in a step at the moment each decoder layer starts its forward."""
        ahead, plain = self.run_tool('bfloat16'), self.run_tool('bfloat16', profile='reference-u2')
        layers = plain['layers']
        self.assertGreaterEqual(layers, 3)
        before = plain['gathers_before_layer']
        self.assertEqual(before, list(range(before[0], before[0] + layers)))            # reference: one more for each layer
        # default: the next layer's all-gather is out already; the last layer has none after it
        self.assertEqual(ahead['gathers_before_layer'], [n + 1 for n in before[:-1]] + [before[-1]])

    def test_more_than_two_ranks_add_bf16_gradients_in_the_order_of_the_backend(self):
        """Then the layer sum has no single value. lm_head and the embedding keep the fp32 check, and the layers
        must stay near the fp32 sum. What NCCL gives on 8 GPUs is not measured here."""
        out = train_shard.run_numerics(world=3, dtype='bfloat16', seed=3)
        self.assertTrue(out['ok'], out)
        self.assertTrue(out['layers_in_bf16'])
        self.assertFalse(out['one_order'])
        self.assertLessEqual(out['explicit_reduction']['rel_diff'], 1e-6)
        self.assertTrue(out['layers_near_fp32'])
        self.assertLess(out['layer_fp32_reduction']['rel_diff'], out['layer_bf16_tol'])
        self.assertGreater(out['layer_fp32_reduction']['cosine'], 0.9999)

    def test_fp32_matches_one_unsharded_process(self):
        for profile in (None, 'reference-u2'):                           # in fp32 both settings add in fp32
            out = self.run_tool('float32', **(dict(profile=profile) if profile else {}))
            self.assertTrue(out['ok'], out)
            self.assertTrue(out['losses_equal'])
            self.assertLessEqual(out['unsharded']['rel_diff'], 1e-6)
            self.assertLessEqual(out['reduction']['rel_diff'], 1e-6)
            self.assertLessEqual(out['fp32_reduction']['rel_diff'], 1e-6)
            self.assertEqual(out['reduce_inputs'], {'torch.float32': (out['layers'] + 3) * out['steps']})
            self.assertFalse(out['layers_in_bf16'])

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
        # a layer input of the tiny model is far below one 16 MiB block: every one was a host copy of its own
        self.assertEqual(hooked['host_blocks'], [0, 0])

    def test_layer_inputs_in_host_blocks_give_the_same_gradient_bit_for_bit(self):
        """The default settings keep a layer input that fills a block in the block store. The block size is lowered
        to 64 bytes so that every input of the tiny model fills several. No GPU here: the blocks are ordinary host
        memory and the copies are CPU-to-CPU, so page-locking and the direct GPU copy are not exercised."""
        work = Path(tempfile.mkdtemp(prefix='v834f-blocks-'))
        self.addCleanup(shutil.rmtree, work, True)
        plain = train_shard.run_numerics(world=2, dtype='float32', seed=3, work=work / 'plain')
        blocks = train_shard.run_numerics(world=2, dtype='float32', seed=3, work=work / 'blocks', offload_inputs=True,
                                          host_block_bytes=64)
        self.assertTrue(plain['ok'] and blocks['ok'], (plain, blocks))
        self.assertTrue(blocks['profile']['pin_inputs'])
        layers, steps = blocks['layers'], blocks['steps']
        self.assertEqual(blocks['packed_inputs'], [layers * steps] * 2)
        for held, used in zip(blocks['host_blocks'], blocks['host_blocks_used']):
            self.assertEqual(len(used), steps)
            self.assertTrue(all(count >= 2 * layers for count in used), used)       # every input filled two blocks or more
            self.assertEqual(held, max(used))                                       # a step begins again at the first block:
            self.assertLess(held, sum(used))                                        # a rank holds the blocks of its largest step
        first, second = torch.load(plain['gradient_file']), torch.load(blocks['gradient_file'])
        self.assertEqual(list(first), list(second))
        self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))
        self.assertLessEqual(blocks['unsharded']['rel_diff'], 1e-6)


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
                                # the same update with the reference settings, named as on the box
                                ('sharded', ('--port', str(free_port()), '--tag', 'sharded-u2'),
                                 child_env(PIT_PER_GPU_MEM_GIB='32', PIT_ZERO3_PROFILE='reference-u2')),
                                ('fp32-load', ('--source', 'sharded'), child_env()),
                                ('fp32-load', ('--source', 'reference'), child_env())):
            proc, seconds = run_driver(cls.work, mode, *args, env=env)
            if proc.returncode != 0:
                raise AssertionError('%s failed (rc %s):\n%s\n%s' % (mode, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
            if mode == 'sharded' and '--tag' not in args:
                cls.sharded_log, cls.sharded_seconds = proc.stdout, seconds
        load = lambda mode: json.loads((cls.work / mode / 'result.json').read_text())
        cls.ref, cls.plain, cls.out, cls.u2 = load('reference'), load('plain'), load('sharded'), load('sharded-u2')
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
        self.assertEqual(events[0]['profile'], 'rtx5090-u6')                        # the helper says which settings it runs
        self.assertEqual(self.out['forward'], 'make_train_update.<locals>.log_probs')

    def test_activate_needs_the_launcher_and_decides_from_the_gpu(self):
        self.assertTrue(self.out['activate_none'].startswith('refused:'))
        self.assertEqual(self.out['policy']['per_gpu_mem_gib'], 32.0)
        self.assertTrue(self.out['policy']['zero3'])
        self.assertEqual(self.out['zero3_env'], '1')
        self.assertEqual(self.out['install'], dict(dp_world=2, zero3=True, dp_token_budget=32768))
        self.assertEqual(self.out['policy']['profile'], TestSettings.U6)            # no name given: the default settings
        self.assertEqual(self.out['policy']['token_budget_cap'], 32768)
        # the same host-v834f files on a GPU that fits the update: the v8.3.4e update, budget untouched, no settings read
        self.assertFalse(self.plain['policy']['zero3'])
        self.assertIsNone(self.plain['zero3_env'])
        self.assertIsNone(self.plain['policy']['profile'])
        self.assertIsNone(self.plain['policy']['token_budget_cap'])
        self.assertEqual(self.plain['install'], dict(dp_world=0, zero3=False, dp_token_budget=40960))
        self.assertEqual(self.plain['forward'], 'make_train_update.<locals>.fast_blp')

    def test_the_reference_settings_run_when_the_environment_names_them(self):
        self.assertEqual(self.u2['policy']['profile'], TestSettings.U2)
        self.assertEqual(self.u2['install'], dict(dp_world=2, zero3=True, dp_token_budget=24576))
        self.assertEqual(self.u2['microbatch']['token_budget'], 24576)
        self.assertEqual(self.u2['microbatch']['profile'], TestSettings.U2)
        self.assertEqual(self.u2['plan_probe'], dict(bins=5, largest_bin_tokens=22000))
        self.assertEqual(self.u2['helper_returncodes'], [0])
        self.assertEqual(self.u2['status'], 'updated')
        # the fixture is fp32, where both settings add every gradient in fp32: the numbers of the unsharded update
        for out in (self.u2, self.out):
            self.assertAlmostEqual(out['objective_loss'], self.ref['objective_loss'], delta=1e-6 * abs(self.ref['objective_loss']))
            self.assertAlmostEqual(out['preclip_gradient_norm'], self.ref['preclip_gradient_norm'],
                                   delta=1e-6 * self.ref['preclip_gradient_norm'])
            self.assertAlmostEqual(out['raw_update_norm'], self.ref['raw_update_norm'], delta=1e-6 * self.ref['raw_update_norm'])

    def test_live_accumulator_signature_and_no_typeerror(self):
        self.assertEqual(self.out['fp32_accumulator_signature'], '(self, named_parameters)')
        self.assertEqual(self.out['status'], 'updated')
        self.assertEqual(self.out['optimizer_step'], 8)

    def test_token_cut_ran_in_the_plan(self):
        micro = self.out['microbatch']
        self.assertEqual(micro['engine'], 'packed-zero3-v835')
        self.assertEqual(micro['token_budget'], 32768)
        self.assertEqual(micro['profile'], TestSettings.U6)                         # the report names the settings that ran
        self.assertEqual(self.ref['microbatch']['token_budget'], 40960)
        self.assertEqual(micro['logit_chunk'], 1024)
        # the planner the update calls, on two rollouts each of 19,000, 15,000 and 11,000 tokens (two to a bin here):
        # the two longest share a bin under 40,960; under 32,768 the largest bin is the two of 15,000
        self.assertEqual(self.out['plan_probe'], dict(bins=3, largest_bin_tokens=30000))
        self.assertEqual(self.plain['plan_probe'], dict(bins=3, largest_bin_tokens=38000))

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


class TestRealBoxFindings(unittest.TestCase):
    """Memory faults the first real 8x RTX 5090 run of the sharded trainer found (2026-11-11). A GPU shows each of
    them as an out-of-memory error. Here one real sharded update runs on the staged files in a one-rank world, and
    rank 0 carries the pass-through instruments of harness/zero3_e2e.py (observe()): they count what is still
    alive at fixed points, and the staged code runs unchanged. The head chunk is 3 tokens so that one micro-batch
    has several head chunks, as a real one has."""

    # autograd nodes of one head chunk and of the split of its outputs: the log_q branch (top-k, log-softmax) and
    # the log_p branch (gather, logsumexp)
    HEAD_NODES = ('TopkBackward0', 'LogSoftmaxBackward0', 'LogsumexpBackward0', 'GatherBackward0', 'SqueezeBackward1',
                  'CatBackward0', 'SplitWithSizesBackward0')

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='v834f-observed-'))
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        for mode, args, env in (('prepare', (), child_env()),
                                ('sharded', ('--world', '1', '--port', str(free_port()), '--observe', '--logit-chunk', '3'),
                                 child_env(PIT_PER_GPU_MEM_GIB='32'))):
            proc, _ = run_driver(cls.work, mode, *args, env=env)
            if proc.returncode != 0:
                raise AssertionError('%s failed (rc %s):\n%s\n%s' % (mode, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
        cls.out = json.loads((cls.work / 'sharded' / 'result.json').read_text())
        cls.seen = cls.out['observed']

    def chunks(self, items):
        return -(-sum(output for _, output in items) // self.seen['logit_chunk'])

    def test_the_observed_run_is_one_real_update(self):
        self.assertEqual(self.out['status'], 'updated')
        self.assertEqual(self.out['microbatch']['engine'], 'packed-zero3-v835')
        self.assertEqual(self.out['microbatch']['logit_chunk'], 3)
        self.assertEqual(self.seen['vocab'], 64)

    def test_no_original_parameter_outlives_the_fully_shard_call_that_replaced_it(self):
        """Finding 1. A reference to an original Parameter keeps its full-size tensor after fully_shard has put this
        rank's shard in its place. On the box that was every decoder layer after its fp32 upcast: 17.1 -> 30.0 GiB."""
        units = self.seen['shard']
        self.assertEqual([unit['unit'] for unit in units],
                         ['FSDPQwen3_5DecoderLayer'] * 4 + ['FSDPEmbedding', 'FSDPLinear', 'FSDPQwen3_5Model'])
        self.assertTrue(all(unit['originals'] >= 1 for unit in units))              # the instrument watched every unit
        self.assertEqual([unit['alive'] for unit in units], [0] * len(units))

    def test_backward_holds_no_recomputed_logits_of_the_head_chunks(self):
        """Finding 2. With KL off the loss differentiates log_q only. A log_p that keeps its graph makes the
        non-reentrant head checkpoint hold the fp32 logits it recomputed, for every chunk of the micro-batch, until
        backward is over: 22.65 GiB on one rank of the box for a bin of 32770 tokens."""
        held = [h['logits_alive'] for h in self.seen['head'] if h['phase'] == 'step']
        real = [n for n in self.seen['nodes'] if n['kind'] == 'step' and n['items']]
        self.assertEqual(len(held), len(real))                                      # one count for each real micro-batch
        self.assertEqual(len(held), self.out['microbatch']['steps'])
        self.assertGreaterEqual(max(self.chunks(n['items']) for n in real), 18)     # with many head chunks in one of them
        self.assertEqual(held, [0] * len(held))

    def test_the_stock_forward_holds_them_and_gives_the_same_bits(self):
        """The control of finding 2, on the same sharded actor and the same micro-batch: the forward of the stock
        v8211_batched file (log_p keeps its graph) against the staged sharded forward, with one fixed loss on log_q.
        The count of the stock run proves that the instrument sees the fault. The values and the gradient shards
        prove that the fix changes no number."""
        control = self.seen['control']
        held = {h['phase']: h['logits_alive'] for h in self.seen['head'] if h['phase'].startswith('control-')}
        self.assertEqual((control['items'], control['head_chunks']), ([[6, 19], [3, 25], [3, 12]], 19))
        self.assertGreaterEqual(held['control-stock'], control['head_chunks'])      # at least the logits of every chunk
        self.assertEqual(held['control-staged'], 0)
        self.assertEqual(control['stock_log_p_attached'], [True] * 3)
        self.assertEqual(control['staged_log_p_attached'], [False] * 3)
        self.assertTrue(control['values_equal'])                                    # log_p, log_q and the support mask
        self.assertEqual((control['gradient_tensors'], control['gradient_nonzero']), (56, 56))
        self.assertTrue(control['gradients_equal'])                                 # every fp32 gradient shard, bit for bit

    def test_the_probe_step_runs_the_backward_of_the_real_loss(self):
        """Finding 3. The probe step is the memory measurement before the real steps (null_step). It also summed
        log_p, so its backward ran the log_p branch and freed what the real step keeps: it read about 4.5 GiB low
        on the box."""
        probes = [n for n in self.seen['nodes'] if n['kind'] == 'probe']
        self.assertEqual(len(probes), 1)
        probe = probes[0]
        real = next(n for n in self.seen['nodes'] if n['kind'] == 'step' and n['items'] == probe['items'])
        chunks = self.chunks(probe['items'])
        self.assertEqual((probe['items'], chunks), ([[3, 25], [3, 29]], 18))
        self.assertEqual((real['nodes']['TopkBackward0'], real['nodes']['LogSoftmaxBackward0']), (chunks, chunks))
        self.assertNotIn('LogsumexpBackward0', real['nodes'])                       # the real loss never differentiates log_p
        self.assertEqual({node: probe['nodes'].get(node, 0) for node in self.HEAD_NODES},
                         {node: real['nodes'].get(node, 0) for node in self.HEAD_NODES})
        first = self.seen['head'][:2]                    # the probe, then the real step on the same micro-batch
        self.assertEqual([h['phase'] for h in first], ['probe', 'step'])
        self.assertEqual(first[0]['logits_alive'], first[1]['logits_alive'])

    def test_the_probe_step_follows_the_loss_and_not_the_forward(self):
        """Finding 3, the rule itself. The probe step runs on the stock forward, whose log_p keeps its graph, next
        to a loss on log_q on that same forward. It must run the same head backward and hold the same logits. A
        probe that sums every output with a graph runs the log_p branch too and holds none of them."""
        nodes = {n['kind']: n['nodes'] for n in self.seen['nodes'] if n['kind'].startswith('control-')}
        held = {h['phase']: h['logits_alive'] for h in self.seen['head'] if h['phase'].startswith('control-')}
        chunks = self.seen['control']['head_chunks']
        self.assertEqual((nodes['control-stock']['TopkBackward0'], nodes['control-stock']['LogSoftmaxBackward0']), (chunks, chunks))
        self.assertNotIn('LogsumexpBackward0', nodes['control-stock'])
        self.assertEqual({node: nodes['control-probe'].get(node, 0) for node in self.HEAD_NODES},
                         {node: nodes['control-stock'].get(node, 0) for node in self.HEAD_NODES})
        self.assertGreaterEqual(held['control-stock'], chunks)
        self.assertEqual(held['control-probe'], held['control-stock'])


class TestGradientParityWithTheUnshardedTrainer(unittest.TestCase):
    """The full 8x RTX 5090 run of 2026-11-11 reported a preclip gradient norm 11% above the B300's on the same
    batch. This is the check of the sharded arithmetic that a CPU can make: the same head, batch and loss through

      reference   the admitted host-v834e files, unsharded: all 9 rollouts in ONE micro-batch
      w1          host-v834f sharded, 1 rank, ONE rollout in each micro-batch (9 micro-batches)
      w2, w3      host-v834f sharded, 2 and 3 ranks, 5 micro-batches spread over the ranks, with zero-weight steps

    One rollout is capped, has a negative advantage and 110 tokens, so the derail mask cuts it to 96 tokens: its
    per-token weight is then above that of every other rollout, as on the real batch. A loss that was normalised
    for each micro-batch or each rank, or a rank that weighted its tokens in another way, would give another
    gradient here. The second pair of runs has the loss knobs of a turn on (token-mean aggregation, group-normalised
    and clipped advantages), which rank 0 sends to the helper ranks with the job.

    Not covered: the CUDA kernels, bf16, and the 9B model. The fixture is fp32 with the stock forward, where a
    rollout gives the same numbers in every packing. See fixture.py."""

    CAPPED = 110
    KNOBS = dict(loss_agg='token_mean', adv_center='group_norm', adv_clip=0.5)
    RUNS = (('reference', 'reference', ()), ('w1', 'sharded', ('--world', '1', '--max-rows', '1')),
            ('w2', 'sharded', ('--world', '2')), ('w3', 'sharded', ('--world', '3')),
            ('reference-knobs', 'reference', ('--turn', json.dumps(dict(knobs=KNOBS)))),
            ('w2-knobs', 'sharded', ('--world', '2', '--turn', json.dumps(dict(knobs=KNOBS)))))
    SAME = ('tokens', 'loss_tokens', 'derail_index', 'support_outside_tokens', 'stop', 'outcome')
    CLOSE = ('loss', 'advantage', 'prompt_weight', 'old_logprob_mean', 'support_logq_mean')

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='v834f-parity-'))
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        proc, _ = run_driver(cls.work, 'prepare')
        if proc.returncode != 0:
            raise AssertionError('prepare failed:\n%s' % proc.stderr[-3000:])
        cls.report, cls.result, cls.gradient = {}, {}, {}
        for tag, mode, args in cls.RUNS:
            args = ('--tag', tag, '--capped-tokens', str(cls.CAPPED), '--capture-gradient') + args
            if mode == 'sharded':
                args += ('--port', str(free_port()))
            proc, _ = run_driver(cls.work, mode, *args, env=child_env(PIT_PER_GPU_MEM_GIB='32') if mode == 'sharded' else child_env())
            if proc.returncode != 0:
                raise AssertionError('%s failed (rc %s):\n%s\n%s' % (tag, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
            cls.report[tag] = json.loads((cls.work / tag / 'report.private.json').read_text())
            cls.result[tag] = json.loads((cls.work / tag / 'result.json').read_text())
            cls.gradient[tag] = torch.load(str(cls.work / tag / 'gradient.pt'))

    PAIRS = (('reference', 'w1'), ('reference', 'w2'), ('reference', 'w3'), ('reference-knobs', 'w2-knobs'))

    def rollouts(self, tag):
        return {(row['task_id'], row['nonce']): row for row in self.report[tag]['trajectories']}

    def update(self, tag):
        """candidate fp32 master minus head fp32 master, as the run left them in its own store."""
        fp = self.result[tag]['fp32']
        store = self.work / tag / 'fp32-master'

        def entry(name):
            tensors = {}
            for path in sorted((store / name).glob('*.safetensors')):
                tensors.update(load_file(str(path)))
            return tensors
        head, candidate = entry(fp['input_bf16_sha256']), entry(fp['output_store_entry'])
        self.assertEqual(list(head), list(candidate))
        return {name: candidate[name].double() - head[name].double() for name in head}

    @staticmethod
    def distance(got, want):
        """(|got - want| / |want|, cosine) over every tensor of `got`; `want` is cut to the rows `got` holds."""
        want = {name: want[name][:got[name].shape[0]].double() for name in got}
        got = {name: tensor.double() for name, tensor in got.items()}
        norm = lambda tensors: float(torch.sqrt(sum(t.pow(2).sum() for t in tensors.values())))
        dot = float(sum((got[name] * want[name]).sum() for name in got))
        return norm({name: got[name] - want[name] for name in got}) / norm(want), dot / (norm(got) * norm(want))

    def test_the_runs_pack_the_same_batch_in_different_ways(self):
        micro = {tag: self.report[tag]['microbatch'] for tag in self.report}
        self.assertEqual((micro['reference']['engine'], micro['reference']['count']), ('padded', 1))
        self.assertEqual(self.report['reference']['trained_rollouts'], 9)
        self.assertEqual((micro['w1']['world'], micro['w1']['count'], micro['w1']['steps']), (1, 9, 9))
        self.assertEqual((micro['w2']['world'], micro['w2']['count'], micro['w2']['steps']), (2, 5, 3))
        self.assertEqual((micro['w3']['world'], micro['w3']['count'], micro['w3']['steps']), (3, 5, 2))
        for tag in ('w2', 'w3'):                                                    # every rank worked; one ran a zero-weight step
            self.assertTrue(all(load > 0 for load in micro[tag]['rank_tokens']), micro[tag])
            self.assertEqual(sum(rank['null_steps'] for rank in micro[tag]['ranks']), 1)
        self.assertEqual(len({self.report[tag]['data_sha256'] for tag in self.report}), 1)

    def test_the_derail_mask_cuts_the_capped_rollout_in_every_run(self):
        for tag, report in self.report.items():
            cut = [row for row in report['trajectories'] if row['derail_index'] is not None]
            self.assertEqual(report['derailed_rollouts'], 1, tag)
            self.assertEqual([(row['stop'], row['tokens'], row['derail_index'], row['loss_tokens']) for row in cut],
                             [('max_tokens', self.CAPPED, 31, 96)], tag)                # window 32, tail 64
            self.assertLess(cut[0]['advantage'], 0.0)

    def test_every_rollout_has_the_numbers_of_the_unsharded_trainer(self):
        for want_tag, got_tag in self.PAIRS:
            want, got = self.rollouts(want_tag), self.rollouts(got_tag)
            self.assertEqual(sorted(got), sorted(want))
            self.assertEqual(len(want), 9)
            for key, row in want.items():
                for field in self.SAME:
                    self.assertEqual(got[key][field], row[field], (got_tag, field))
                for field in self.CLOSE:
                    self.assertAlmostEqual(got[key][field], row[field], delta=1e-6 * max(1.0, abs(row[field])), msg=(got_tag, field))
            for field in ('loss_tokens', 'trained_tokens', 'sampled_tokens', 'trained_rollouts', 'masked_rollouts'):
                self.assertEqual(self.report[got_tag][field], self.report[want_tag][field], (got_tag, field))

    def test_one_rollout_in_each_micro_batch_gives_the_gradient_of_one_micro_batch(self):
        got, want = self.gradient['w1'], self.gradient['reference']
        self.assertEqual(list(got), list(want))                                     # a one-rank shard is the whole tensor
        self.assertEqual([tuple(t.shape) for t in got.values()], [tuple(t.shape) for t in want.values()])
        self.assertEqual(len(want), 56)
        apart, cosine = self.distance(got, want)
        self.assertLess(apart, 1e-6)                                                # fp32 summation order: measured 8e-8
        self.assertGreater(cosine, 1.0 - 1e-12)

    def test_rank_0_holds_its_rows_of_the_sum_over_every_rank(self):
        """In a world of W ranks rank 0 keeps the first 1/W of the rows of every tensor of the reduced gradient.
        The micro-batches of the other ranks are in those rows too: the reduction sums over the ranks first."""
        for want_tag, got_tag, world in (('reference', 'w2', 2), ('reference', 'w3', 3), ('reference-knobs', 'w2-knobs', 2)):
            got, want = self.gradient[got_tag], self.gradient[want_tag]
            self.assertEqual(list(got), list(want))
            for name, shard in got.items():
                self.assertEqual(tuple(shard.shape), (min(-(-want[name].shape[0] // world), want[name].shape[0]),) + tuple(want[name].shape[1:]), name)
            apart, cosine = self.distance(got, want)
            self.assertLess(apart, 1e-6, got_tag)
            self.assertGreater(cosine, 1.0 - 1e-12, got_tag)

    def test_norm_loss_and_update_of_every_packing(self):
        for want_tag, got_tag in self.PAIRS:
            want, got = self.report[want_tag], self.report[got_tag]
            self.assertAlmostEqual(got['preclip_gradient_norm'], want['preclip_gradient_norm'],
                                   delta=1e-6 * want['preclip_gradient_norm'], msg=got_tag)
            self.assertAlmostEqual(got['objective_loss'], want['objective_loss'], delta=1e-6 * abs(want['objective_loss']), msg=got_tag)
            # the step of every rank's rows: fp32 master of the candidate minus fp32 master of the head
            apart, cosine = self.distance(self.update(got_tag), self.update(want_tag))
            self.assertLess(apart, 1e-3, got_tag)                                   # measured 2e-6 .. 3e-5
            self.assertGreater(cosine, 1.0 - 1e-6, got_tag)

    def test_the_loss_knobs_of_the_turn_reach_every_rank(self):
        want, got = self.report['reference-knobs'], self.report['w2-knobs']
        for report in (want, got):
            self.assertEqual(report['train_params']['knobs_applied'], dict(step=False, loss=True))
            self.assertEqual({k: report['train_params']['knobs'][k] for k in self.KNOBS}, self.KNOBS)
            self.assertLessEqual(max(abs(row['advantage']) for row in report['trajectories']), 0.5)
        self.assertEqual(self.report['w2']['train_params']['knobs_applied'], dict(step=False, loss=False))
        self.assertGreater(max(abs(row['advantage']) for row in self.report['w2']['trajectories']), 0.5)
        # the knobs change the gradient by more than half of its size; the two knob runs agree as closely as the others
        apart, _ = self.distance(self.gradient['reference-knobs'], self.gradient['reference'])
        self.assertGreater(apart, 0.5)
        apart, _ = self.distance(self.gradient['w2-knobs'], self.gradient['w2'])
        self.assertGreater(apart, 0.5)


class TestCompareTrainReports(unittest.TestCase):
    """compare_train_reports on real update reports of the fixture batch: the unsharded update, the sharded update
    of the same job, and the unsharded update of a job whose derail mask is off (threshold -9.0 against -3.0)."""

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='v834f-compare-'))
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        common = ('--capped-tokens', '110', '--capture-gradient')
        for tag, mode, args, env in (
                (None, 'prepare', (), child_env()),
                ('derail', 'reference', common, child_env()),
                ('no-derail', 'reference', common + ('--derail-threshold', '-9.0'), child_env()),
                ('sharded', 'sharded', common + ('--world', '2', '--port', str(free_port())), child_env(PIT_PER_GPU_MEM_GIB='32'))):
            proc, _ = run_driver(cls.work, mode, *(args + (('--tag', tag) if tag else ())), env=env)
            if proc.returncode != 0:
                raise AssertionError('%s failed (rc %s):\n%s\n%s' % (tag or mode, proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
        cls.path = {tag: cls.work / tag / 'report.private.json' for tag in ('derail', 'no-derail', 'sharded')}
        cls.report = {tag: json.loads(path.read_text()) for tag, path in cls.path.items()}

    def test_two_packings_of_one_job_show_no_difference(self):
        out = compare_train_reports.compare(self.report['derail'], self.report['sharded'])
        self.assertEqual(out['inputs_different'], {})
        self.assertEqual([m['engine'] for m in out['microbatch']], ['padded', 'packed-zero3-v835'])
        self.assertEqual((out['rollouts']['shared'], out['rollouts']['only_a'], out['rollouts']['only_b']), (9, 0, 0))
        self.assertEqual((out['rollouts']['different'], out['rollouts']['different_mask']), (0, 0))
        self.assertEqual(out['rollouts']['same_kept_tokens'], 9)
        self.assertLess(out['rollouts']['old_logprob_mean_abs_difference']['max'], 1e-6)
        self.assertEqual(out['top'], [])
        self.assertAlmostEqual(out['totals']['preclip_gradient_norm']['ratio'], 1.0, delta=1e-6)
        self.assertEqual(out['weight_model']['different_mask_share'], [0, 0])
        self.assertAlmostEqual(out['weight_model']['modelled_norm_ratio'], 1.0, places=12)
        self.assertAlmostEqual(out['weight_model']['modelled_cosine'], 1.0, places=12)

    def test_a_different_derail_cut_is_found_and_weighed(self):
        out = compare_train_reports.compare(self.report['no-derail'], self.report['sharded'])
        self.assertEqual(out['inputs_different'], {'config.settings.derail_threshold': [-9.0, -3.0]})
        self.assertEqual(out['totals']['derailed_rollouts'], dict(a=0, b=1, ratio=None))
        self.assertEqual((out['totals']['loss_tokens']['a'], out['totals']['loss_tokens']['b']), (277, 263))
        rollouts = out['rollouts']
        self.assertEqual((rollouts['different'], rollouts['different_mask'], rollouts['same_kept_tokens']), (1, 1, 8))
        self.assertEqual(rollouts['different_by_field'], dict(derail_index=1, loss_tokens=1, support_outside_tokens=1,
                                                              advantage=0, stop=0, tokens=0))
        self.assertLess(rollouts['old_logprob_mean_abs_difference']['max'], 1e-6)    # the other 8 rollouts: the same forward
        (row,) = out['top']
        self.assertEqual((row['derail_index'], row['loss_tokens'], row['stop']), ([None, 31], [110, 96], ['max_tokens'] * 2))
        self.assertLess(row['pg_tokens'][1], row['pg_tokens'][0])
        self.assertGreater(row['weight_share'][1], row['weight_share'][0])          # fewer tokens: a larger share of the gradient
        self.assertEqual(out['weight_model']['different_mask_share'], row['weight_share'])
        # the model against the two real gradients (both unsharded and whole): same batch, with and without the cut
        with_cut = torch.load(str(self.work / 'derail' / 'gradient.pt'))
        without = torch.load(str(self.work / 'no-derail' / 'gradient.pt'))
        norm = lambda g: float(torch.sqrt(sum(t.double().pow(2).sum() for t in g.values())))
        cosine = float(sum((with_cut[n].double() * without[n].double()).sum() for n in with_cut)) / (norm(with_cut) * norm(without))
        self.assertLess(cosine, 0.999)                                              # one rollout of nine moved the gradient
        self.assertAlmostEqual(out['weight_model']['modelled_cosine'], cosine, delta=2e-3)
        self.assertAlmostEqual(out['weight_model']['measured_norm_ratio'], norm(with_cut) / norm(without), delta=1e-6)

    def test_command_line(self):
        def run(*argv):
            stream = io.StringIO()
            with redirect_stdout(stream):
                self.assertEqual(compare_train_reports.main(list(argv)), 0)
            return stream.getvalue()
        text = run(str(self.path['no-derail']), str(self.path['sharded']))
        self.assertIn('DIFFERENT config.settings.derail_threshold: -9.0 | -3.0', text)
        self.assertIn('with another mask: 1 (derail_index 1, loss_tokens 1, support_outside_tokens 1)', text)
        self.assertIn('derail_index None | 31, loss tokens 110 | 96', text)
        same = run(str(self.path['derail']), str(self.path['sharded']))
        self.assertIn('the same batch, checkpoint, fp32 master, settings and turn parameters', same)
        self.assertIn('with another mask: 0', same)
        self.assertEqual(json.loads(run(str(self.path['derail']), str(self.path['sharded']), '--json'))['rollouts']['different'], 0)

    def test_refuses_a_report_without_an_update(self):
        with self.assertRaises(ValueError):
            compare_train_reports.compare(dict(status='no-update'), self.report['sharded'])


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

    def test_a_helper_with_other_settings_refuses_the_job_and_the_trainer_stops(self):
        """Ranks with different settings would issue different collectives. Rank 0 sends the name of its settings
        with the job; the helper compares it with its own before it shards anything."""
        proc, seconds = run_driver(self.work, 'sharded', '--port', str(free_port()), '--tag', 'mixed',
                                   '--helper-profile', 'reference-u2', env=child_env(PIT_ZERO3='1'), timeout=300)
        self.assertNotEqual(proc.returncode, 0)
        self.assertLess(seconds, 120)
        self.assertFalse((self.work / 'mixed' / 'result.json').exists())            # no update came out of the mixed world
        events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{"component": "trainer-host-v834f"')]
        started = next(e for e in events if e['event'] == 'helper-zero3')
        self.assertEqual(started['profile'], 'reference-u2')
        failed = next(e for e in events if e['event'] == 'zero3-rank-failed')
        self.assertEqual((failed['rank'], failed['where']), (1, 'sharding'))
        self.assertIn("rank 0 runs the settings 'rtx5090-u6', this rank 'reference-u2'", failed['error'])
        self.assertFalse([e for e in events if e['event'] == 'helper-sharded'])     # the helper built no sharded actor

    def test_an_unknown_settings_name_stops_the_trainer_before_any_helper_starts(self):
        proc, _ = run_driver(self.work, 'sharded', '--port', str(free_port()), '--tag', 'unknown',
                             env=child_env(PIT_ZERO3='1', PIT_ZERO3_PROFILE='rtx5090'))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unknown PIT_ZERO3_PROFILE 'rtx5090'", proc.stderr)
        self.assertFalse((self.work / 'unknown' / 'result.json').exists())
        self.assertNotIn('helper-zero3', proc.stdout)


if __name__ == '__main__':
    unittest.main()
