"""CPU unit tests for trainer-v834e fail-closed fp32 master loading, ROOT bootstrap override, and durable PIN GC."""
from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from nacl.signing import SigningKey
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / 'miner' / 'validator' / 'trainer-v834e' / 'mimo_host_trainer_v83.py'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _install_signed_stub():
    """Use the same envelope shape as subnet.backend_jobs.signed (ROOT VerifyKey over canonical payload)."""
    from nacl.signing import VerifyKey
    bj = types.ModuleType('subnet.backend_jobs')

    def signed(value, authority):
        if value.get('signer') != authority:
            raise ValueError('job authority')
        VerifyKey(bytes.fromhex(authority)).verify(
            _canonical(value['payload']), base64.b64decode(value['signature'], validate=True))
        return value['payload']

    bj.signed = signed
    sub = sys.modules.get('subnet') or types.ModuleType('subnet')
    sub.backend_jobs = bj
    sys.modules['subnet'] = sub
    sys.modules['subnet.backend_jobs'] = bj
    return bj


def _load_host():
    _install_signed_stub()
    name = 'trainer_v834e_fp32_host'
    spec = importlib.util.spec_from_file_location(name, HOST)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sign(sk, payload):
    sig = sk.sign(_canonical(payload)).signature
    return dict(signer=sk.verify_key.encode().hex(),
                signature=base64.b64encode(sig).decode(),
                payload=payload)


class Opt:
    def __init__(self, params):
        self.parameters = tuple(params)
        self.masters = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in self.parameters}
        self.global_step = 0
        self.failed = False

    def zero_grad(self):
        pass


def _params(seed=1):
    g = torch.Generator().manual_seed(seed)
    return [
        ('a.weight', torch.randn(8, 4, generator=g).bfloat16()),
        ('b.bias', torch.randn(4, generator=g).bfloat16()),
    ]


class TrainerV834eFp32MasterTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_host()
        self.tmp = Path(tempfile.mkdtemp(prefix='v834e-fp32-'))
        self.store = self.tmp / 'fp32-master'
        self.store.mkdir()
        self.mod.FP32_STORE = str(self.store)
        self.mod.FP32_SHARD_BYTES = 1 << 20
        self.mod.FP32_SAVE_DEFER_MAX = 0
        self.mod.FP32_KEEP = 2
        self.mod.FP32_BOOTSTRAP_OVERRIDE = None
        self.mod.FP32_EXPECTED_MASTER = None
        self.mod.FP32_AUTHORITY = None
        self.mod._FP32_HEADS = {'current_input': None, 'previous_admitted': None}
        self.events = []
        self.emit = self.events.append
        self.sk = SigningKey.generate()
        self.authority = self.sk.verify_key.encode().hex()
        self.mod.FP32_AUTHORITY = self.authority

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_master(self, opt, entry=None, *, corrupt_digest=False, mutate_tensor=None):
        bf, _ = self.mod._digest_map(list(opt.parameters))
        entry = entry or bf
        dig, per = self.mod._digest_map([(n, opt.masters[n]) for n, _ in opt.parameters])
        if corrupt_digest:
            dig = '0' * 64
        d = self.store / entry
        d.mkdir(parents=True, exist_ok=True)
        fn = 'fp32-master-00001-of-00001.safetensors'
        tensors = {n: opt.masters[n].detach().clone() for n, _ in opt.parameters}
        if mutate_tensor:
            tensors[mutate_tensor] = tensors[mutate_tensor] + 1.0
        save_file(tensors, str(d / fn), metadata=dict(format='pt', fp32_master=self.mod.FP32_VERSION))
        meta = dict(version=self.mod.FP32_VERSION, bf16_sha256=entry, master_digest=dig, tensor_sha256=per,
                    files={fn: dict(sha256='x', bytes=1)}, tensor_file={n: fn for n, _ in opt.parameters})
        (d / 'meta.json').write_text(json.dumps(meta, sort_keys=True, indent=1))
        return entry, dig, bf

    def test_master_present_loads_from_disk(self):
        opt = Opt(_params(1))
        # Give masters a distinct residual vs plain bf16 upcast.
        with torch.no_grad():
            for n, p in opt.parameters:
                opt.masters[n].copy_(p.float() + 1e-4)
                # Keep projection exact: live bf16 must equal bf16_rne(master).
                p.copy_(opt.masters[n].bfloat16())
        want = {n: opt.masters[n].detach().clone() for n, _ in opt.parameters}
        entry, dig, bf = self._write_master(opt)
        # Scramble in-memory masters so prepare must load from disk.
        with torch.no_grad():
            for n in opt.masters:
                opt.masters[n].zero_()
        opt._v833f_bf16 = None
        info = self.mod._fp32_prepare(opt, None, self.emit)
        self.assertEqual(info['source'], 'disk')
        self.assertEqual(info['fp32_master_source'], 'disk')
        self.assertEqual(info['input_master_digest'], dig)
        self.assertEqual(info['fp32_master_hash_in'], dig)
        self.assertEqual(info['input_bf16_sha256'], bf)
        for n, t in want.items():
            self.assertTrue(torch.equal(opt.masters[n].detach(), t), n)

    def test_master_missing_fails_closed(self):
        opt = Opt(_params(2))
        opt._v833f_bf16 = None
        with self.assertRaises(self.mod.Fp32MasterError) as ctx:
            self.mod._fp32_prepare(opt, None, self.emit)
        self.assertEqual(ctx.exception.reason, 'master-missing')
        self.assertTrue(any(e.get('event') == 'fp32-master-missing' for e in self.events))
        self.assertFalse(any((e.get('source') or '').startswith('bootstrap-upcast') for e in self.events))

    def test_corrupt_master_fails_closed_with_reason(self):
        opt = Opt(_params(3))
        with torch.no_grad():
            for n, p in opt.parameters:
                opt.masters[n].copy_(p.float() + 2e-4)
                p.copy_(opt.masters[n].bfloat16())
        entry, dig, bf = self._write_master(opt, corrupt_digest=True)
        with torch.no_grad():
            for n in opt.masters:
                opt.masters[n].zero_()
        opt._v833f_bf16 = None
        with self.assertRaises(self.mod.Fp32MasterError) as ctx:
            self.mod._fp32_prepare(opt, None, self.emit)
        self.assertIn('load-rejected', ctx.exception.reason)
        self.assertTrue(any(e.get('event') == 'fp32-master-load-rejected' for e in self.events))
        # Must not fall through to silent bootstrap-upcast (no override accepted).
        self.assertFalse(any(e.get('event') == 'fp32-master-bootstrap-upcast' for e in self.events))
        self.assertNotEqual(getattr(opt, '_v833f_bf16', None), bf)

    def test_mismatched_expected_master_fails_closed(self):
        opt = Opt(_params(4))
        with torch.no_grad():
            for n, p in opt.parameters:
                opt.masters[n].copy_(p.float() + 3e-4)
                p.copy_(opt.masters[n].bfloat16())
        entry, dig, bf = self._write_master(opt)
        with torch.no_grad():
            for n in opt.masters:
                opt.masters[n].zero_()
        opt._v833f_bf16 = None
        self.mod.FP32_EXPECTED_MASTER = 'ab' * 32
        with self.assertRaises(self.mod.Fp32MasterError) as ctx:
            self.mod._fp32_prepare(opt, None, self.emit)
        self.assertEqual(ctx.exception.reason, 'expected-master-mismatch')

    def test_root_override_valid_upcasts(self):
        opt = Opt(_params(5))
        bf, _ = self.mod._digest_map(list(opt.parameters))
        payload = dict(version=self.mod.FP32_BOOTSTRAP_KIND, head_bf16_sha256=bf)
        path = self.tmp / 'override.json'
        path.write_text(json.dumps(_sign(self.sk, payload)))
        self.mod.FP32_BOOTSTRAP_OVERRIDE = str(path)
        opt._v833f_bf16 = None
        with torch.no_grad():
            for n in opt.masters:
                opt.masters[n].fill_(9.0)
        info = self.mod._fp32_prepare(opt, None, self.emit)
        self.assertEqual(info['source'], 'bootstrap-upcast-root-override')
        self.assertEqual(info['fp32_master_source'], 'bootstrap-upcast-root-override')
        for n, p in opt.parameters:
            self.assertTrue(torch.equal(opt.masters[n].detach(), p.detach().float()))
        self.assertTrue(any(e.get('event') == 'fp32-master-bootstrap-override-accepted' for e in self.events))

    def test_override_wrong_head_fails_closed(self):
        opt = Opt(_params(6))
        bf, _ = self.mod._digest_map(list(opt.parameters))
        payload = dict(version=self.mod.FP32_BOOTSTRAP_KIND, head_bf16_sha256='ff' * 32)
        path = self.tmp / 'override-bad-head.json'
        path.write_text(json.dumps(_sign(self.sk, payload)))
        self.mod.FP32_BOOTSTRAP_OVERRIDE = str(path)
        opt._v833f_bf16 = None
        with self.assertRaises(self.mod.Fp32MasterError) as ctx:
            self.mod._fp32_prepare(opt, None, self.emit)
        self.assertEqual(ctx.exception.reason, 'bootstrap-override-head-mismatch')
        self.assertNotEqual(bf, 'ff' * 32)

    def test_override_bad_signature_fails_closed(self):
        opt = Opt(_params(7))
        bf, _ = self.mod._digest_map(list(opt.parameters))
        other = SigningKey.generate()
        payload = dict(version=self.mod.FP32_BOOTSTRAP_KIND, head_bf16_sha256=bf)
        path = self.tmp / 'override-bad-sig.json'
        path.write_text(json.dumps(_sign(other, payload)))
        self.mod.FP32_BOOTSTRAP_OVERRIDE = str(path)
        opt._v833f_bf16 = None
        with self.assertRaises(self.mod.Fp32MasterError) as ctx:
            self.mod._fp32_prepare(opt, None, self.emit)
        self.assertEqual(ctx.exception.reason, 'bootstrap-override-signature')

    def test_gc_never_deletes_pinned_current_or_previous_head(self):
        # Create five complete store entries; pin one; mark current + previous heads.
        import os, time
        names = []
        for i in range(5):
            name = ('%02d' % i) + ('a' * 62)
            d = self.store / name
            d.mkdir()
            (d / 'meta.json').write_text('{}')
            names.append(name)
        pinned, current, previous = names[0], names[1], names[2]
        (self.store / pinned / 'PIN').write_text('operator')
        # Age the PIN far beyond the old 3h rule — must still survive.
        now = time.time()
        os.utime(self.store / pinned / 'PIN', (now - 100000, now - 100000))
        # Stagger directory mtimes AFTER creating PIN so pin write does not make it "newest".
        for i, name in enumerate(names):
            os.utime(self.store / name, (now - 100 + i, now - 100 + i))
        self.mod._FP32_HEADS = {'current_input': current, 'previous_admitted': previous}
        self.mod._fp32_gc((), self.emit)
        alive = {p.name for p in self.store.iterdir() if p.is_dir() and not p.name.startswith('.')}
        # newest 2 by mtime = names[3], names[4]; plus pin/current/previous
        for must in (pinned, current, previous, names[3], names[4]):
            self.assertIn(must, alive, 'GC deleted protected entry ' + must)
        victim = 'zz' + ('b' * 62)
        vd = self.store / victim
        vd.mkdir()
        (vd / 'meta.json').write_text('{}')
        os.utime(vd, (now - 1000, now - 1000))
        self.mod._fp32_gc((), self.emit)
        self.assertFalse((self.store / victim).exists(), 'unprotected old entry must be GC deleted')
        self.assertTrue((self.store / pinned).exists())
        self.assertTrue((self.store / current).exists())
        self.assertTrue((self.store / previous).exists())

    def test_memory_hit_emits_report_fields(self):
        opt = Opt(_params(8))
        with torch.no_grad():
            for n, p in opt.parameters:
                opt.masters[n].copy_(p.float() + 5e-5)
                p.copy_(opt.masters[n].bfloat16())
        bf, _ = self.mod._digest_map(list(opt.parameters))
        dig, _ = self.mod._digest_map([(n, opt.masters[n]) for n, _ in opt.parameters])
        opt._v833f_bf16 = bf
        opt._v833g_dig = dig
        info = self.mod._fp32_prepare(opt, None, self.emit)
        self.assertEqual(info['source'], 'memory')
        self.assertEqual(info['fp32_master_source'], 'memory')
        self.assertEqual(info['fp32_master_hash_in'], dig)
        self.assertIsNone(info['fp32_master_hash_out'])


if __name__ == '__main__':
    unittest.main()
