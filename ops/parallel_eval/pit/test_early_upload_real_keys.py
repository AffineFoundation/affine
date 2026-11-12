#!/usr/bin/env python3
"""Early upload must produce records the real APPLY / next-box path can consume.

* fp32 index ``bf16_sha256`` is the trainer store entry (== meta.json
  bf16_sha256), never the checkpoint id — else ``fetch_master_on_box`` lands
  the master under the wrong FP32/<entry> and rejects meta.
* bf16 upload: the object that the host checks is the exact pending key that
  the box sent to (the real box tool against a local S3 server).
* ``pe_verify`` accepts the private pending index as the shard-complete
  record before APPLY promotes it.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.live import pe_artifact_upload as AU
from ops.parallel_eval.live import pe_box
from ops.parallel_eval.live import pe_verify
from ops.parallel_eval.pit.r2_fixture import R2Server, ShellBox


C = 'c' * 64
M = 'd' * 64
ENTRY = 'e' * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _PutOkPod:
    id = 'pod-early'
    ip = '127.0.0.1'
    port = 22

    def __init__(self):
        self.cmds = []

    def run(self, cmd, inp=None, timeout=60, check=True, **_kw):
        self.cmds.append(cmd)
        return 'PUT_OK 1 BAD 0\n'


class TestMasterIndexUsesStoreEntry(unittest.TestCase):
    def test_upload_master_from_box_names_index_by_store_entry(self):
        seen = {}

        def _stream(P, **kw):
            seen.update(kw)
            return {'ok': True, 'complete': True, 'shard_count': 1}

        with mock.patch.object(AU, 'stream_upload_master_from_box', side_effect=_stream):
            pe_box.upload_master_from_box(
                _PutOkPod(), candidate_id=C, master_digest=M, bf16_sha256=C,
                box_master_dir='/box/fp32/' + ENTRY, store_entry=ENTRY,
            )
        self.assertEqual(seen['bf16_sha256'], ENTRY)

    def test_non_hex_store_entry_keeps_explicit_bf16(self):
        seen = {}

        def _stream(P, **kw):
            seen.update(kw)
            return {'ok': True, 'complete': True, 'shard_count': 1}

        with mock.patch.object(AU, 'stream_upload_master_from_box', side_effect=_stream):
            pe_box.upload_master_from_box(
                _PutOkPod(), candidate_id=C, master_digest=M, bf16_sha256=C,
                box_master_dir='/box/fp32/x', store_entry=M + '-g1',
            )
        self.assertEqual(seen['bf16_sha256'], C)

    def test_early_upload_passes_report_store_entry(self):
        seen = {}

        def _stream(P, **kw):
            seen.update(kw)
            return {'ok': True, 'complete': True, 'shard_count': 1, 'index': '/x'}

        receipt = {
            'candidate_id': C,
            'candidate_files': {'model.safetensors': 'a' * 64},
            'candidate_master_digest': M,
            'box_master_dir': '/box/fp32/' + ENTRY,
            'box_candidate_dir': '/box/ckpt/' + C,
            'report_doc': {'optimizer': {'sgd': {'fp32_master': {
                'output_store_entry': ENTRY, 'output_bf16_sha256': ENTRY,
            }}}},
        }
        sent = {'master': {'meta.json': {'sha256': '1' * 64}}, 'candidate': {'model.safetensors': {'sha256': 'a' * 64}}}
        env = {k: v for k, v in os.environ.items()
               if k not in ('PIT_R2_LOCAL_MIRROR', 'PIT_PRESIGN_PUT_JSON', 'PIT_PRESIGN_MASTER_PUT_JSON')}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(AU, 'upload_trees_from_box', return_value=sent) as transfer, \
                mock.patch.object(AU, 'stream_upload_master_from_box', side_effect=_stream), \
                mock.patch.object(pe_box, 'upload_candidate_from_box', return_value={'ok': True}) as candidate:
            out = pe_box._early_upload_artifacts(_PutOkPod(), receipt)
        self.assertTrue(out['artifacts_uploaded'])
        self.assertEqual(seen['bf16_sha256'], ENTRY)
        self.assertEqual(seen['master_digest'], M)
        # One transfer carries both trees; each record step gets its tree of that transfer.
        trees = transfer.call_args.kwargs['trees']
        self.assertEqual(transfer.call_count, 1)
        self.assertEqual((trees['master']['dir'], trees['master']['names']), ('/box/fp32/' + ENTRY, None))
        self.assertEqual((trees['candidate']['dir'], trees['candidate']['names']), ('/box/ckpt/' + C, ['model.safetensors']))
        self.assertEqual(trees['candidate']['prefix'], trees['master']['prefix'] + '/model')
        self.assertEqual(seen['uploaded'], sent['master'])
        self.assertEqual(candidate.call_args.kwargs['uploaded'], sent['candidate'])


class TestPendingBf16VerifyKey(unittest.TestCase):
    """The real upload of a bf16 file: real box tool in a shell, real host code, a local S3 server as R2."""

    PENDING_KEY = 'pit-fakepass/run1/%s/model/model.safetensors' % C

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.r2 = R2Server(root).start()
        self.addCleanup(self.r2.stop)
        self.uploads = root / 'uploads'
        self.box_dir = root / 'box' / 'ckpt' / C
        self.box_dir.mkdir(parents=True)
        self.blob = b'BF16-SHARD'
        (self.box_dir / 'model.safetensors').write_bytes(self.blob)
        patch = mock.patch.dict(os.environ, {
            'PIT_LIFE_CFG': str(self.r2.life_cfg),
            'PIT_R2_PENDING_PREFIX': 'pit-fakepass/run1',
            'PIT_R2_CKPT_PREFIX': 'mimo/affine-mimo9b-grpo-20261013/checkpoints',
            'PIT_UPLOAD_ROOT': str(self.uploads),
            'PIT_R2_BOX_TMP': str(root / 'box-tmp'),
        })
        patch.start()
        self.addCleanup(patch.stop)
        for k in ('PIT_R2_LOCAL_MIRROR', 'PIT_PRESIGN_PUT_JSON', 'PIT_PRESIGN_MASTER_PUT_JSON'):
            os.environ.pop(k, None)

    def test_host_checks_the_pending_key_that_the_box_sent_to(self):
        files = {'model.safetensors': _sha(self.blob)}
        out = pe_box.upload_candidate_from_box(
            ShellBox(), candidate_id=C, files=files,
            box_candidate_dir=str(self.box_dir), pending=True,
        )
        self.assertTrue(out['ok'])
        # The one object is at the pending key; the live checkpoint prefix has nothing.
        self.assertEqual(self.r2.keys(), [self.PENDING_KEY])
        self.assertEqual(self.r2.body(self.PENDING_KEY), self.blob)
        receipt = json.loads((self.uploads / C / 'UPLOAD.json').read_text())
        self.assertEqual(receipt['verified_files'], files)
        self.assertEqual(receipt['objects']['model.safetensors']['r2_key'], self.PENDING_KEY)
        self.assertEqual(receipt['objects']['model.safetensors']['etag'], self.r2.etag(self.PENDING_KEY))
        # The host read the size and the ETag of that key, and it downloaded nothing.
        self.assertEqual(set(self.r2.calls), {'HeadObject'})

    def test_verify_fails_closed_on_wrong_bytes(self):
        files = {'model.safetensors': 'a' * 64}
        with self.assertRaises(pe_box.BoxIOError):
            pe_box.upload_candidate_from_box(
                ShellBox(), candidate_id=C, files=files,
                box_candidate_dir=str(self.box_dir), pending=True,
            )
        self.assertFalse((self.uploads / C).exists())


class TestVerifyAcceptsPendingIndex(unittest.TestCase):
    def _env(self, root: Path) -> dict:
        return {
            'PE_STATE': str(root / 'pe'),
            'PIT_UPLOAD_ROOT': str(root / 'uploads'),
            'PIT_FP32_MASTER_INDEX': str(root / 'public'),
            'PIT_FP32_PENDING_INDEX': str(root / 'pending'),
        }

    def _receipts(self, root: Path) -> tuple[Path, Path]:
        cand = root / 'uploads' / C
        cand.mkdir(parents=True)
        (cand / 'CHECKPOINT.json').write_text(json.dumps({'id': C, 'files': {'m': '1' * 64}}))
        master = root / 'host-out' / 'fp32-master-digest'
        master.mkdir(parents=True)
        (master / 'master_digest.txt').write_text(M + '\n')
        return cand, master

    def test_pending_index_counts_as_complete_before_promote(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'public').mkdir()
            cand, master = self._receipts(root)
            with mock.patch.dict(os.environ, self._env(root), clear=False):
                AU.write_pending_master_index(
                    candidate_id=C, bf16=ENTRY, digest=M,
                    inv={
                        'master-00001-of-00001.safetensors': {'sha256': '2' * 64, 'bytes': 10},
                        'meta.json': {'sha256': '3' * 64, 'bytes': 5},
                    },
                )
                out = pe_verify.verify_uploads(
                    candidate_id=C, candidate_sha256=C, master_digest=M,
                    candidate_path=str(cand), master_path=str(master),
                )
        self.assertTrue(out['candidate_ok'])
        self.assertTrue(out['master_ok'])
        self.assertTrue(out['master_complete'])
        self.assertTrue(out['detail']['master_index_pending'])

    def test_no_index_anywhere_stays_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cand, master = self._receipts(root)
            with mock.patch.dict(os.environ, self._env(root), clear=False):
                out = pe_verify.verify_uploads(
                    candidate_id=C, candidate_sha256=C, master_digest=M,
                    candidate_path=str(cand), master_path=str(master),
                )
        self.assertFalse(out['master_complete'])

    def test_pending_digest_mismatch_fails_master(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cand, master = self._receipts(root)
            with mock.patch.dict(os.environ, self._env(root), clear=False):
                AU.write_pending_master_index(
                    candidate_id=C, bf16=ENTRY, digest='9' * 64,
                    inv={
                        'master-00001-of-00001.safetensors': {'sha256': '2' * 64, 'bytes': 10},
                        'meta.json': {'sha256': '3' * 64, 'bytes': 5},
                    },
                )
                out = pe_verify.verify_uploads(
                    candidate_id=C, candidate_sha256=C, master_digest=M,
                    candidate_path=str(cand), master_path=str(master),
                )
        self.assertFalse(out['master_ok'])
        self.assertFalse(out['master_complete'])


if __name__ == '__main__':
    unittest.main()
