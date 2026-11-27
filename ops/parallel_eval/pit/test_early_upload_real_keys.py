#!/usr/bin/env python3
"""Early upload must produce records the real APPLY / next-box path can consume.

* fp32 index ``bf16_sha256`` is the trainer store entry (== meta.json
  bf16_sha256), never the checkpoint id — else ``fetch_master_on_box`` lands
  the master under the wrong FP32/<entry> and rejects meta.
* bf16 upload verify GETs the exact pending key the box PUT.
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

import boto3

try:
    from moto import mock_aws
except ImportError:  # pragma: no cover
    mock_aws = None  # type: ignore

from ops.parallel_eval.live import pe_artifact_upload as AU
from ops.parallel_eval.live import pe_box
from ops.parallel_eval.live import pe_verify


C = 'c' * 64
M = 'd' * 64
ENTRY = 'e' * 64
BUCKET = 'affine-pit-moto-early'


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
        with mock.patch.object(AU, 'stream_upload_master_from_box', side_effect=_stream), \
                mock.patch.object(pe_box, 'upload_candidate_from_box', return_value={'ok': True}):
            out = pe_box._early_upload_artifacts(_PutOkPod(), receipt)
        self.assertTrue(out['artifacts_uploaded'])
        self.assertEqual(seen['bf16_sha256'], ENTRY)
        self.assertEqual(seen['master_digest'], M)


@unittest.skipIf(mock_aws is None, 'moto not installed')
class TestPendingBf16VerifyKey(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        creds = root / 'r2.env'
        creds.write_text('R2_ACCESS_KEY_ID=testing\nR2_SECRET_ACCESS_KEY=testing\n')
        life_cfg = root / 'life.json'
        life_cfg.write_text(json.dumps({'bucket': {
            'bucket': BUCKET, 'endpoint': 'https://s3.amazonaws.com',
            'credentials_file': str(creds),
        }}))
        self.uploads = root / 'uploads'
        self.env = {
            'PIT_LIFE_CFG': str(life_cfg),
            'PIT_R2_PENDING_PREFIX': 'pit-fakepass/run1',
            'PIT_R2_CKPT_PREFIX': 'mimo/affine-mimo9b-grpo-20261013/checkpoints',
            'PIT_UPLOAD_ROOT': str(self.uploads),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _client(self):
        client = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id='testing', aws_secret_access_key='testing',
        )
        client.create_bucket(Bucket=BUCKET)
        return client

    def test_verify_reads_pending_key_box_put(self):
        blob = b'BF16-SHARD'
        files = {'model.safetensors': _sha(blob)}
        with mock_aws():
            client = self._client()
            client.put_object(
                Bucket=BUCKET, Key='pit-fakepass/run1/%s/model/model.safetensors' % C, Body=blob,
            )
            with mock.patch.dict(os.environ, self.env, clear=False):
                for k in ('PIT_R2_LOCAL_MIRROR', 'PIT_PRESIGN_PUT_JSON', 'PIT_PRESIGN_PAYLOAD_JSON'):
                    os.environ.pop(k, None)
                with mock.patch.object(pe_box, 'put'):
                    out = pe_box.upload_candidate_from_box(
                        _PutOkPod(), candidate_id=C, files=files,
                        box_candidate_dir='/box/ckpt/' + C, pending=True,
                    )
        self.assertTrue(out['ok'])
        receipt = json.loads((self.uploads / C / 'UPLOAD.json').read_text())
        self.assertEqual(receipt['verified_files'], files)

    def test_verify_fails_closed_on_wrong_bytes(self):
        files = {'model.safetensors': 'a' * 64}
        with mock_aws():
            client = self._client()
            client.put_object(
                Bucket=BUCKET, Key='pit-fakepass/run1/%s/model/model.safetensors' % C, Body=b'other',
            )
            with mock.patch.dict(os.environ, self.env, clear=False):
                os.environ.pop('PIT_R2_LOCAL_MIRROR', None)
                with mock.patch.object(pe_box, 'put'):
                    with self.assertRaises(pe_box.BoxIOError):
                        pe_box.upload_candidate_from_box(
                            _PutOkPod(), candidate_id=C, files=files,
                            box_candidate_dir='/box/ckpt/' + C, pending=True,
                        )


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
