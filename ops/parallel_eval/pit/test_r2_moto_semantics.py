#!/usr/bin/env python3
"""Fake-S3 (moto) models real R2 key semantics for pending promote / GC / bf16.

Local-mirror tests alone miss the live failure mode: rewriting the public index
to ``fp32-master/<cid>/`` without copying objects stalls the next box download.
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


C = 'c' * 64
M = 'd' * 64
BUCKET = 'affine-pit-moto-test'

_SKIP_NO_MOTO = unittest.skipIf(mock_aws is None, 'moto not installed')


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@_SKIP_NO_MOTO
class TestMotoR2KeySemantics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.creds = self.root / 'r2.env'
        self.creds.write_text(
            'R2_ACCESS_KEY_ID=testing\n'
            'R2_SECRET_ACCESS_KEY=testing\n'
        )
        self.life_cfg = self.root / 'life.json'
        self.life_cfg.write_text(json.dumps({
            'bucket': {
                'bucket': BUCKET,
                'endpoint': 'https://s3.amazonaws.com',
                'credentials_file': str(self.creds),
            }
        }))
        self.pending_idx = self.root / 'fp32-pending'
        self.public_idx = self.root / 'fp32-master'
        self.pending_idx.mkdir()
        self.public_idx.mkdir()
        self.env = {
            'PIT_LIFE_CFG': str(self.life_cfg),
            'AFFINE_LIFE_CFG': str(self.life_cfg),
            'PIT_FP32_PENDING_INDEX': str(self.pending_idx),
            'PIT_FP32_MASTER_INDEX': str(self.public_idx),
            'PIT_R2_PENDING_PREFIX': 'pit-pending',
            'PIT_R2_FP32_PREFIX': 'fp32-master',
            # Force real Bucket path — no local mirror.
            'PIT_R2_LOCAL_MIRROR': '',
        }
        # Clear any inherited mirror.
        os.environ.pop('PIT_R2_LOCAL_MIRROR', None)

    def tearDown(self):
        self.tmp.cleanup()

    def _start_bucket(self):
        client = boto3.client(
            's3',
            region_name='us-east-1',
            aws_access_key_id='testing',
            aws_secret_access_key='testing',
        )
        client.create_bucket(Bucket=BUCKET)
        return client

    def test_promote_copies_objects_before_public_index(self):
        shard = b'WEIGHTS-PENDING'
        sha = _sha(shard)
        meta_blob = json.dumps({
            'version': 'v833f-fp32-head-master-v1',
            'bf16_sha256': C,
            'master_digest': M,
            'tensor_file': {'layer.weight': 'master.safetensors'},
        }).encode()
        meta_sha = _sha(meta_blob)
        src_key = 'pit-pending/%s/master.safetensors' % C
        meta_key = 'pit-pending/%s/meta.json' % C
        dst_key = 'fp32-master/%s/master.safetensors' % C
        dst_meta = 'fp32-master/%s/meta.json' % C
        with mock_aws():
            client = self._start_bucket()
            client.put_object(Bucket=BUCKET, Key=src_key, Body=shard)
            client.put_object(Bucket=BUCKET, Key=meta_key, Body=meta_blob)
            inv = {
                'master.safetensors': {
                    'sha256': sha, 'bytes': len(shard), 'r2_key': src_key,
                },
                'meta.json': {
                    'sha256': meta_sha, 'bytes': len(meta_blob), 'r2_key': meta_key,
                },
            }
            with mock.patch.dict(os.environ, self.env, clear=False):
                os.environ.pop('PIT_R2_LOCAL_MIRROR', None)
                AU.write_pending_master_index(
                    candidate_id=C, bf16=C, digest=M, inv=inv,
                )
                # Public index must not exist yet; live key must be absent.
                self.assertFalse((self.public_idx / ('%s.json' % C)).exists())
                with self.assertRaises(client.exceptions.NoSuchKey):
                    client.get_object(Bucket=BUCKET, Key=dst_key)
                out = AU.promote_pending_master_to_live(candidate_id=C, digest=M)
            self.assertTrue(Path(out).is_file())
            live = json.loads(Path(out).read_text())
            self.assertFalse(live.get('pending'))
            self.assertEqual(live['files']['master.safetensors']['r2_key'], dst_key)
            # Objects exist at the live key the next box will read.
            body = client.get_object(Bucket=BUCKET, Key=dst_key)['Body'].read()
            self.assertEqual(body, shard)
            self.assertEqual(_sha(body), sha)
            meta_live = client.get_object(Bucket=BUCKET, Key=dst_meta)['Body'].read()
            self.assertEqual(json.loads(meta_live)['tensor_file']['layer.weight'],
                             'master.safetensors')
            self.assertEqual(set(live['meta_json'].keys()), {'r2_key'})
            self.assertEqual(live['meta_json']['r2_key'], dst_meta)
            self.assertNotIn('version', live['meta_json'])

    def test_promote_refuses_when_pending_object_missing(self):
        with mock_aws():
            self._start_bucket()
            inv = {
                'master.safetensors': {
                    'sha256': '1' * 64, 'bytes': 1,
                    'r2_key': 'pit-pending/%s/master.safetensors' % C,
                },
                'meta.json': {
                    'sha256': '2' * 64, 'bytes': 2,
                    'r2_key': 'pit-pending/%s/meta.json' % C,
                },
            }
            with mock.patch.dict(os.environ, self.env, clear=False):
                os.environ.pop('PIT_R2_LOCAL_MIRROR', None)
                AU.write_pending_master_index(
                    candidate_id=C, bf16=C, digest=M, inv=inv,
                )
                with self.assertRaises(AU.ArtifactUploadError):
                    AU.promote_pending_master_to_live(candidate_id=C, digest=M)
                self.assertFalse((self.public_idx / ('%s.json' % C)).exists())

    def test_gc_deletes_r2_objects_not_just_index(self):
        blob = b'PENDING-BLOB'
        keys = [
            'pit-pending/%s/master.safetensors' % C,
            'pit-pending/%s/model/model.safetensors' % C,
            'pit-pending/%s/meta.json' % C,
        ]
        with mock_aws():
            client = self._start_bucket()
            for k in keys:
                client.put_object(Bucket=BUCKET, Key=k, Body=blob)
            inv = {
                'master.safetensors': {
                    'sha256': _sha(blob), 'bytes': len(blob), 'r2_key': keys[0],
                },
                'meta.json': {
                    'sha256': _sha(blob), 'bytes': len(blob), 'r2_key': keys[2],
                },
            }
            with mock.patch.dict(os.environ, self.env, clear=False):
                os.environ.pop('PIT_R2_LOCAL_MIRROR', None)
                AU.write_pending_master_index(
                    candidate_id=C, bf16=C, digest=M, inv=inv,
                )
                out = AU.gc_pending_artifacts(C)
            self.assertTrue(out['ok'])
            self.assertFalse((self.pending_idx / ('%s.json' % C)).exists())
            listed = client.list_objects_v2(
                Bucket=BUCKET, Prefix='pit-pending/%s/' % C,
            )
            self.assertEqual(listed.get('KeyCount') or 0, 0)
            self.assertFalse(listed.get('Contents'))

    def test_pending_bf16_presign_prefix_not_live_ckpt(self):
        files = {'model.safetensors': 'a' * 64}
        with mock_aws():
            self._start_bucket()
            with mock.patch.dict(os.environ, {
                **{k: v for k, v in self.env.items() if k != 'PIT_R2_LOCAL_MIRROR'},
                'PIT_R2_CKPT_PREFIX': 'mimo/affine-mimo9b-grpo-20261013/checkpoints',
            }, clear=False):
                os.environ.pop('PIT_R2_LOCAL_MIRROR', None)
                os.environ.pop('PIT_PRESIGN_PUT_JSON', None)
                urls = pe_box.presign_put_urls(C, files, pending=True)
                self.assertEqual(set(urls), set(files))
                for name, url in urls.items():
                    self.assertIn('pit-pending/%s/model/%s' % (C, name), url)
                    self.assertNotIn(
                        'mimo/affine-mimo9b-grpo-20261013/checkpoints/%s' % C, url,
                    )
                live_urls = pe_box.presign_put_urls(C, files, pending=False)
                for name, url in live_urls.items():
                    self.assertIn(
                        'mimo/affine-mimo9b-grpo-20261013/checkpoints/%s/model/%s'
                        % (C, name),
                        url,
                    )


if __name__ == '__main__':
    unittest.main()
