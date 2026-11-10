#!/usr/bin/env python3
"""A PIT-published fp32 master must be loadable by the admitted trainer.

End to end, with real files and the trainer's own code (no loader mocks):

1. the admitted trainer module (``miner/validator/trainer-v834e``) writes a
   candidate master store entry with its real ``_fp32_finish`` / ``_fp32_save``;
2. PIT publishes it exactly as ``LiveApplicator._apply_fresh`` does
   (``pe_box.upload_master`` with ``bf16_sha256=<checkpoint id>``);
3. the next epoch's box fetches it with the real ``fetch_master_on_box`` /
   ``gate_fetch.py`` using the index entry ``train_v834e`` would read;
4. the trainer's real ``_fp32_prepare`` / ``_fp32_load`` load it and verify
   digest + bf16 projection under ``AFFINE_FP32_EXPECTED_MASTER_HASH``.

Only transport is local: the "box" is a temp dir, ``scp`` is a PATH shim and R2
is ``PIT_R2_LOCAL_MIRROR`` (file:// presigns).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from ops.parallel_eval.live import pe_box

REPO = Path(__file__).resolve().parents[3]
TRAINER_SRC = REPO / 'miner' / 'validator' / 'trainer-v834e' / 'mimo_host_trainer_v83.py'
GATE_FETCH_SRC = REPO / 'ops' / 'parallel_eval' / 'live' / 'gate_fetch.py'

SCP_SHIM = textwrap.dedent('''\
    #!/usr/bin/env python3
    """Local stand-in for scp: root@host:/path is a path on this machine."""
    import os, shutil, sys
    args, av, i = [], sys.argv[1:], 0
    while i < len(av):
        a = av[i]
        if a in ('-o', '-P', '-i', '-F', '-l'):
            i += 2
            continue
        if a.startswith('-'):
            i += 1
            continue
        args.append(a)
        i += 1
    src, dst = (a.split(':', 1)[1] if a.startswith('root@') else a for a in args)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    shutil.copyfile(src, dst)
''')


def load_trainer():
    """Load the admitted host trainer the way ``train_pit.py`` does."""
    spec = importlib.util.spec_from_file_location('pit_test_mimo_host_trainer_v83', TRAINER_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LocalBox:
    """Pod stand-in: box commands run in a local shell (paths are real temp paths)."""

    ip = '127.0.0.1'
    port = 22

    def __init__(self, box_id: str):
        self.id = box_id

    def run(self, cmd, inp=None, timeout=600, check=True, retries=0):
        if isinstance(inp, str):
            inp = inp.encode()
        p = subprocess.run(['bash', '-c', cmd], input=inp, capture_output=True, timeout=timeout)
        if check and p.returncode != 0:
            raise RuntimeError('box rc=%d: %s' % (p.returncode, p.stderr.decode('utf-8', 'replace')[-400:]))
        return p.stdout.decode('utf-8', 'replace')


class Optimizer:
    """The attributes the trainer's fp32 helpers use on CPUAdamW."""

    def __init__(self, parameters, masters, global_step):
        self.parameters = list(parameters)
        self.masters = dict(masters)
        self.global_step = int(global_step)
        self._v833f_lock = threading.Lock()


def candidate_weights():
    gen = torch.Generator().manual_seed(1066)
    shapes = {
        'model.language_model.layers.0.weight': (24, 16),
        'model.language_model.layers.1.weight': (16, 16),
        'lm_head.weight': (32, 16),
    }
    masters = {n: torch.randn(s, generator=gen, dtype=torch.float32) for n, s in shapes.items()}
    bf16 = {n: m.to(torch.bfloat16) for n, m in masters.items()}
    return masters, bf16


class TestPitMasterLoadsInTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        scp = self.bin / 'scp'
        scp.write_text(SCP_SHIM)
        scp.chmod(scp.stat().st_mode | stat.S_IXUSR)
        self.mirror = self.root / 'r2'
        self.index = self.root / 'k1-fp32-master'
        self.presign = self.root / 'presign-keys.json'
        for p in (self.mirror, self.index):
            p.mkdir()
        self.env = {
            'HOME': str(self.root),
            'PATH': '%s:%s' % (self.bin, os.environ.get('PATH', '')),
            'PE_STATE': str(self.root / 'pe'),
            'PIT_UPLOAD_ROOT': str(self.root / 'uploads'),
            'PIT_R2_LOCAL_MIRROR': str(self.mirror),
            'PIT_FP32_MASTER_INDEX': str(self.index),
            'PIT_PRESIGN_KEYS_JSON': str(self.presign),
        }
        self.host = load_trainer()
        self.events = []
        self.masters, self.bf16 = candidate_weights()
        # C is the checkpoint (descriptor) id; it is NOT the trainer's store entry.
        self.candidate = hashlib.sha256(b'pit-candidate-descriptor').hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def emit(self, row):
        self.events.append(row)

    def train_box_writes_master(self, store: Path):
        """Candidate master store entry, written by the trainer's own save path."""
        opt = Optimizer(
            [(n, self.bf16[n]) for n in self.bf16], {n: m.clone() for n, m in self.masters.items()}, 41,
        )
        info = dict(source='disk', input_bf16_sha256='1' * 64, input_master_digest='2' * 64)
        with mock.patch.object(self.host, 'FP32_STORE', str(store)), \
                mock.patch.object(self.host, 'FP32_SHARD_BYTES', 2048):
            info = self.host._fp32_finish(opt, info, self.emit)
            self.host._PUBLISHED.set()
            with opt._v833f_lock:
                pass
        entry = store / info['output_store_entry']
        self.assertTrue((entry / 'meta.json').is_file(), self.events)
        self.assertGreater(len(list(entry.glob('*.safetensors'))), 1, 'want several shards')
        return info

    def publish_like_apply_fresh(self, info, box_master_dir: Path):
        """The exact call ``LiveApplicator._apply_fresh`` makes on GREEN."""
        return pe_box.upload_master(
            candidate_id=self.candidate,
            path=str(self.root / 'host-report' / 'fp32-master'),
            master_digest=info['output_master_digest'],
            bf16_sha256=self.candidate,
            pod=LocalBox('train-box'),
            box_master_dir=str(box_master_dir),
            candidate_on_box=True,
            defer_candidate_upload=True,
            store_entry='',
        )

    def write_presign_map(self, published):
        self.presign.write_text(json.dumps({
            meta['r2_key']: 'file://%s' % (self.mirror / meta['r2_key'])
            for meta in published['files'].values()
        }))

    def next_box_fetches_master(self, store: Path, digest: str):
        """What ``train_v834e`` does for the next item bound to head C."""
        pst = self.root / 'next-box' / 'pst'
        pst.mkdir(parents=True)
        shutil.copyfile(GATE_FETCH_SRC, pst / 'gate_fetch.py')
        published = pe_box._published_master(self.candidate)
        self.write_presign_map(published)
        with mock.patch.object(pe_box, 'FP32', str(store)), mock.patch.object(pe_box, 'PST', str(pst)):
            out = pe_box.fetch_master_on_box(
                LocalBox('next-box'), head=self.candidate, expected_master=digest,
                head_bf16=str(published['bf16_sha256']), published=published,
            )
        return published, out

    def test_published_master_trains_the_next_epoch(self):
        train_store = self.root / 'train-box' / 'fp32-master'
        next_store = self.root / 'next-box' / 'fp32-master'
        with mock.patch.dict(os.environ, self.env):
            info = self.train_box_writes_master(train_store)
            entry, digest = info['output_store_entry'], info['output_master_digest']
            self.assertNotEqual(entry, self.candidate)
            pub = self.publish_like_apply_fresh(info, train_store / entry)
            self.assertTrue(pub['complete'])

            index = json.loads((self.index / ('%s.json' % self.candidate)).read_text())
            self.assertEqual(index['master_digest'], digest)
            # The index names the trainer's store entry, never the checkpoint id.
            self.assertEqual(index['bf16_sha256'], entry)
            # meta_json is a pointer to the real uploaded meta.json, like the live HEAD index.
            self.assertEqual(index['meta_json'], {'r2_key': index['files']['meta.json']['r2_key']})
            uploaded_meta = self.mirror / index['meta_json']['r2_key']
            self.assertEqual(
                hashlib.sha256(uploaded_meta.read_bytes()).hexdigest(),
                index['files']['meta.json']['sha256'],
            )
            self.assertEqual(
                uploaded_meta.read_bytes(), (train_store / entry / 'meta.json').read_bytes(),
            )

            published, fetched = self.next_box_fetches_master(next_store, digest)
            self.assertEqual(fetched['entry'], entry)
            landed = next_store / entry
            self.assertEqual(
                (landed / 'meta.json').read_bytes(), (train_store / entry / 'meta.json').read_bytes(),
                'the trainer meta.json must survive the fetch byte for byte',
            )

            # Next epoch: actor = bf16 head C, masters not yet loaded.
            opt = Optimizer(
                [(n, self.bf16[n].clone()) for n in self.bf16],
                {n: torch.zeros_like(m) for n, m in self.masters.items()}, 42,
            )
            with mock.patch.object(self.host, 'FP32_STORE', str(next_store)), \
                    mock.patch.dict(os.environ, {'AFFINE_FP32_EXPECTED_MASTER_HASH': digest}):
                prepared = self.host._fp32_prepare(opt, None, self.emit)
                loaded = self.host._fp32_load(
                    Optimizer(
                        [(n, self.bf16[n].clone()) for n in self.bf16],
                        {n: torch.zeros_like(m) for n, m in self.masters.items()}, 42,
                    ),
                    None, entry, self.emit,
                )
        self.assertEqual(prepared['source'], 'disk')
        self.assertEqual(prepared['input_store_entry'], entry)
        self.assertEqual(prepared['input_master_digest'], digest)
        self.assertEqual(prepared['expected_master_digest'], digest)
        self.assertEqual(loaded['master_digest'], digest)
        self.assertEqual(set(loaded['tensor_file']), set(self.masters))
        for name, want in self.masters.items():
            self.assertTrue(torch.equal(opt.masters[name], want), name)
        rejected = [e for e in self.events if 'rejected' in str(e.get('event')) or 'missing' in str(e.get('event'))]
        self.assertEqual(rejected, [])

    def test_trainer_rejects_a_synthetic_meta_json(self):
        """The 5d8f3032 index shape is exactly what the trainer cannot load."""
        store = self.root / 'box' / 'fp32-master'
        with mock.patch.dict(os.environ, self.env):
            info = self.train_box_writes_master(store)
            entry = info['output_store_entry']
            (store / entry / 'meta.json').write_text(json.dumps({
                'version': 'pit-fp32-master-v1', 'bf16_sha256': entry,
                'master_digest': info['output_master_digest'], 'source': 'pit', 'shard_count': 2,
            }))
            opt = Optimizer(
                [(n, self.bf16[n].clone()) for n in self.bf16],
                {n: torch.zeros_like(m) for n, m in self.masters.items()}, 42,
            )
            with mock.patch.object(self.host, 'FP32_STORE', str(store)):
                with self.assertRaises(self.host.Fp32MasterError):
                    self.host._fp32_prepare(opt, None, self.emit)

    def test_fetch_never_overwrites_the_verified_meta_json(self):
        """An inline index body must not replace the sha-verified trainer meta.json."""
        train_store = self.root / 'train-box' / 'fp32-master'
        next_store = self.root / 'next-box' / 'fp32-master'
        with mock.patch.dict(os.environ, self.env):
            info = self.train_box_writes_master(train_store)
            entry, digest = info['output_store_entry'], info['output_master_digest']
            self.publish_like_apply_fresh(info, train_store / entry)
            idx_path = self.index / ('%s.json' % self.candidate)
            doc = json.loads(idx_path.read_text())
            doc['meta_json'] = {
                'version': 'pit-fp32-master-v1', 'bf16_sha256': entry, 'master_digest': digest,
            }
            idx_path.write_text(json.dumps(doc))
            self.next_box_fetches_master(next_store, digest)
        landed = json.loads((next_store / entry / 'meta.json').read_text())
        self.assertEqual(landed['version'], self.host.FP32_VERSION)
        self.assertEqual(set(landed['tensor_file']), set(self.masters))

    def test_publish_refuses_a_master_the_trainer_could_not_load(self):
        store = self.root / 'train-box' / 'fp32-master'
        with mock.patch.dict(os.environ, self.env):
            info = self.train_box_writes_master(store)
            entry = store / info['output_store_entry']
            meta = json.loads((entry / 'meta.json').read_text())

            (entry / 'meta.json').unlink()
            with self.assertRaises(pe_box.BoxIOError):
                self.publish_like_apply_fresh(info, entry)
            self.assertFalse((self.index / ('%s.json' % self.candidate)).exists())

            broken = dict(meta)
            del broken['tensor_file']
            (entry / 'meta.json').write_text(json.dumps(broken))
            with self.assertRaises(pe_box.BoxIOError):
                self.publish_like_apply_fresh(info, entry)

            # A store entry whose digest is not the one the host recorded at train.done.
            (entry / 'meta.json').write_text(json.dumps(meta, sort_keys=True, indent=1))
            with self.assertRaises(pe_box.BoxIOError):
                self.publish_like_apply_fresh(dict(info, output_master_digest='e' * 64), entry)
        self.assertFalse((self.index / ('%s.json' % self.candidate)).exists())


if __name__ == '__main__':
    unittest.main()
