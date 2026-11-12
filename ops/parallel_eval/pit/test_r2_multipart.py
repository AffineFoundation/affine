"""R2 transfers of the PIT artifacts: the parallel upload from the box, the check by size and ETag, and the
multipart copy at APPLY.

Nothing of the code under test is replaced. The host code is the real pe_box / pe_artifact_upload /
pe_r2_multipart, the box tool is the real pe_r2_box_upload.py run in a shell, and R2 is a real S3 server on
this machine (see r2_fixture). Parts are 5 MiB here (the R2 minimum) and 256 MiB on a box.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from botocore.exceptions import ReadTimeoutError

from ops.parallel_eval.live import pe_artifact_upload as AU
from ops.parallel_eval.live import pe_box
from ops.parallel_eval.live import pe_r2_box_upload as BOX
from ops.parallel_eval.live import pe_r2_multipart as R2MP
from ops.parallel_eval.live import pe_verify
from ops.parallel_eval.pit.r2_fixture import BUCKET, MIB, R2Server, ShellBox, sha256

C = 'c' * 64            # candidate id
M = 'd' * 64            # fp32 master digest that the host recorded
ENTRY = 'e' * 64        # name of the fp32 store entry on the box
PART = 5 * MIB
PENDING = 'pit-shadow/pending/%s' % C
LIVE = 'fp32-master/%s' % C


class R2Case(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.r2 = R2Server(self.root).start()
        self.addCleanup(self.r2.stop)
        self.box = ShellBox()
        self.master_dir = self.root / 'box' / 'fp32' / ENTRY
        self.cand_dir = self.root / 'box' / 'ckpt' / C
        self.master = {
            'master-00001-of-00002.safetensors': os.urandom(2 * PART + 2 * MIB + 123),     # 3 parts
            'master-00002-of-00002.safetensors': os.urandom(PART + 2 * MIB),               # 2 parts
            'meta.json': json.dumps({'version': 'v833f-fp32-head-master-v1', 'master_digest': M,
                                     'tensor_file': {'w': 'master-00001-of-00002.safetensors'}}).encode(),
        }
        self.cand = {'model.safetensors': os.urandom(2 * PART + MIB), 'config.json': b'{"hidden": 8}'}
        for folder, files in ((self.master_dir, self.master), (self.cand_dir, self.cand)):
            folder.mkdir(parents=True)
            for name, data in files.items():
                (folder / name).write_bytes(data)
        self.files = {name: sha256(data) for name, data in self.cand.items()}
        self.pending_index = self.root / 'pending-index'
        self.public_index = self.root / 'public-index'
        env = {
            'PIT_LIFE_CFG': str(self.r2.life_cfg), 'PIT_R2_PENDING_PREFIX': 'pit-shadow/pending',
            'PIT_R2_FP32_PREFIX': 'fp32-master', 'PIT_R2_CKPT_PREFIX': 'checkpoints',
            'PIT_UPLOAD_ROOT': str(self.root / 'uploads'), 'PIT_FP32_PENDING_INDEX': str(self.pending_index),
            'PIT_FP32_MASTER_INDEX': str(self.public_index), 'PIT_R2_PART_BYTES': str(PART),
            'PIT_R2_BOX_TMP': str(self.root / 'box-tmp'),
        }
        patch = mock.patch.dict(os.environ, env)
        patch.start()
        self.addCleanup(patch.stop)
        for name in ('PIT_R2_LOCAL_MIRROR', 'PIT_PRESIGN_PUT_JSON', 'PIT_PRESIGN_MASTER_PUT_JSON', 'PIT_R2_WORKERS'):
            os.environ.pop(name, None)

    def receipt(self, **over):
        doc = dict(candidate_id=C, candidate_files=dict(self.files), candidate_master_digest=M,
                   box_master_dir=str(self.master_dir), box_candidate_dir=str(self.cand_dir),
                   report_doc={'optimizer': {'sgd': {'fp32_master': {'output_store_entry': ENTRY}}}})
        doc.update(over)
        return doc

    def early_upload(self):
        return pe_box._early_upload_artifacts(self.box, self.receipt())

    def index(self, folder=None):
        return json.loads(((folder or self.pending_index) / ('%s.json' % C)).read_text())

    def count(self, name):
        return self.r2.calls.count(name)


class TestParallelUpload(R2Case):
    def test_one_box_job_sends_both_trees_and_the_host_downloads_nothing(self):
        out = self.early_upload()

        # R2 holds the bytes of the box, under the private pending prefix only.
        for name, data in self.master.items():
            self.assertEqual(self.r2.body('%s/%s' % (PENDING, name)), data)
        for name, data in self.cand.items():
            self.assertEqual(self.r2.body('%s/model/%s' % (PENDING, name)), data)
        self.assertEqual(len(self.r2.keys()), 5)
        self.assertEqual(self.r2.keys(PENDING + '/'), self.r2.keys())

        # One upload command carried the fp32 master and the bf16 candidate.
        uploads = [cmd for cmd in self.box.commands if ' upload ' in cmd]
        self.assertEqual(len(uploads), 1)
        # The host sent no artifact byte and read none back: it signed, joined and read sizes and ETags.
        self.assertEqual(set(self.r2.calls), {'CreateMultipartUpload', 'CompleteMultipartUpload', 'HeadObject'})
        self.assertEqual(self.count('CreateMultipartUpload'), 3)
        self.assertEqual(self.r2.open_uploads(), [])
        # The job file had the signed URLs; the box tool removed it.
        self.assertEqual(sorted(p.name for p in (self.root / 'box-tmp').rglob('*') if p.is_file()),
                         ['pe_r2_box_upload.py', 'stat.json'])

        # The pending index: sha256 of the box file, and the part layout and ETag that R2 has.
        files = self.index()['files']
        self.assertEqual(sorted(files), sorted(self.master))
        for name, data in self.master.items():
            entry = files[name]
            parts = [data[i:i + PART] for i in range(0, len(data), PART)]
            self.assertEqual((entry['sha256'], entry['bytes'], entry['r2_key']),
                             (sha256(data), len(data), '%s/%s' % (PENDING, name)))
            self.assertEqual((entry['multipart'], entry['parts'], entry['part_bytes']), (len(data) > PART, len(parts), PART))
            self.assertEqual(entry['etag'], self.r2.etag(entry['r2_key']))
        self.assertEqual(files['master-00001-of-00002.safetensors']['etag'][-2:], '-3')
        self.assertEqual(self.index()['meta_json'], {'r2_key': PENDING + '/meta.json'})
        self.assertEqual((self.index()['master_digest'], self.index()['bf16_sha256'], self.index()['pending']), (M, ENTRY, True))
        self.assertFalse(self.public_index.exists())

        # The bf16 receipt, and what the train stage hands on.
        upload = json.loads((self.root / 'uploads' / C / 'UPLOAD.json').read_text())
        self.assertEqual(upload['verified_files'], self.files)
        self.assertEqual((upload['transport'], upload['pending']), ('presigned-multipart', True))
        self.assertEqual(upload['objects']['model.safetensors']['r2_key'], PENDING + '/model/model.safetensors')
        self.assertEqual({o['verify'] for o in upload['objects'].values()}, {'etag'})
        self.assertTrue(out['artifacts_uploaded'] and out['artifacts_pending'])
        self.assertEqual(out['early_upload']['master_publish']['transport'], 'box-r2-multipart')
        self.assertEqual((out['early_upload']['master_publish']['downloads'], out['early_upload']['candidate_upload']['downloads']), (0, 0))

        # The APPLY-time verify of the host reads these records and R2 is not needed for it.
        self.r2.calls.clear()
        verified = pe_verify.verify_uploads(candidate_id=C, candidate_sha256=C, master_digest=M,
                                            candidate_path=str(self.root / 'uploads' / C))
        self.assertTrue(verified['candidate_ok'] and verified['master_complete'], verified)
        self.assertEqual(self.r2.calls, [])

    def test_the_etag_of_the_box_tool_is_the_etag_of_the_store(self):
        self.early_upload()
        for name, data in self.master.items():
            parts = [data[i:i + PART] for i in range(0, len(data), PART)]
            hashed = BOX.hash_file(str(self.master_dir / name), BOX.part_ranges(len(data), PART))
            self.assertEqual(hashed['sha256'], sha256(data))
            self.assertEqual(BOX.object_etag(hashed['parts'], len(parts) > 1), self.r2.etag('%s/%s' % (PENDING, name)))

    def test_a_bf16_file_that_is_not_the_file_of_the_descriptor_is_refused(self):
        wrong = dict(self.files, **{'config.json': sha256(b'another config')})
        with self.assertRaises(pe_box.BoxIOError) as caught:
            pe_box.upload_candidate_from_box(self.box, candidate_id=C, files=wrong, box_candidate_dir=str(self.cand_dir))
        self.assertIn('R2 sha mismatch for config.json', str(caught.exception))
        self.assertFalse((self.root / 'uploads' / C).exists())

    def test_a_missing_bf16_file_is_refused_before_any_upload(self):
        missing = dict(self.files, **{'tokenizer.json': sha256(b'x')})
        with self.assertRaises(pe_box.BoxIOError) as caught:
            pe_box.upload_candidate_from_box(self.box, candidate_id=C, files=missing, box_candidate_dir=str(self.cand_dir))
        self.assertIn('files missing on the box', str(caught.exception))
        self.assertEqual((self.r2.keys(), self.r2.open_uploads()), ([], []))

    def test_other_bytes_in_r2_are_found_by_the_etag_and_refused(self):
        # After the box has sent everything, the object of meta.json gets other bytes of the same size.
        key = PENDING + '/meta.json'
        other = bytes(len(self.master['meta.json']))
        self.box.after[' upload '] = lambda: self.r2.client.put_object(Bucket=BUCKET, Key=key, Body=other)
        with self.assertRaises(AU.ArtifactUploadError) as caught:
            AU.stream_upload_master_from_box(self.box, candidate_id=C, master_digest=M, bf16_sha256=ENTRY,
                                             box_master_dir=str(self.master_dir), put=pe_box.put)
        self.assertIn('sha256 in R2', str(caught.exception))
        # Only the object with the unexpected ETag was read back, and no index was written.
        self.assertEqual(self.count('GetObject'), 1)
        self.assertFalse(self.pending_index.exists())

    def test_a_file_that_changes_on_the_box_fails_closed_and_leaves_no_upload(self):
        shard = self.master_dir / 'master-00002-of-00002.safetensors'
        self.box.after[' stat '] = lambda: shard.write_bytes(shard.read_bytes() + b'more')
        with self.assertRaises(AU.ArtifactUploadError) as caught:
            AU.stream_upload_master_from_box(self.box, candidate_id=C, master_digest=M, bf16_sha256=ENTRY,
                                             box_master_dir=str(self.master_dir), put=pe_box.put)
        self.assertIn('the box upload failed', str(caught.exception))
        self.assertIn('bytes any more', str(caught.exception))
        self.assertEqual(self.r2.open_uploads(), [])
        self.assertFalse(self.pending_index.exists())

    def test_the_early_upload_fails_closed_after_its_tries(self):
        (self.master_dir / 'meta.json').unlink()
        with mock.patch.object(pe_box.time, 'sleep'), self.assertRaises(pe_box.EarlyUploadError) as caught:
            self.early_upload()
        self.assertIn('missing meta.json', str(caught.exception))
        self.assertEqual(len([cmd for cmd in self.box.commands if ' upload ' in cmd]), AU.EARLY_UPLOAD_RETRIES)
        self.assertFalse(self.pending_index.exists())
        self.assertFalse((self.root / 'uploads').exists())

    def test_a_second_upload_of_the_same_files_sends_nothing(self):
        trees = {'master': dict(dir=str(self.master_dir), prefix=PENDING, names=None)}
        first = AU.upload_trees_from_box(self.box, candidate_id=C, put=pe_box.put, trees=trees)['master']
        self.assertEqual({entry['reused'] for entry in first.values()}, {False})
        stamps = {key: self.r2.client.head_object(Bucket=BUCKET, Key=key)['LastModified'] for key in self.r2.keys()}
        again = AU.upload_trees_from_box(self.box, candidate_id=C, put=pe_box.put, trees=trees)['master']
        self.assertEqual({entry['reused'] for entry in again.values()}, {True})
        self.assertEqual({name: entry['etag'] for name, entry in again.items()},
                         {name: entry['etag'] for name, entry in first.items()})
        self.assertEqual({entry['verify'] for entry in again.values()}, {'etag'})
        self.assertEqual(stamps, {key: self.r2.client.head_object(Bucket=BUCKET, Key=key)['LastModified'] for key in stamps})
        self.assertEqual(self.r2.open_uploads(), [])

    def test_a_manifest_that_does_not_fit_the_plan_is_refused(self):
        name = 'master-00002-of-00002.safetensors'
        path = self.master_dir / name
        bucket = AU.open_life_bucket()
        plan = {'f': dict(key='%s/%s' % (PENDING, name), path=str(path), bytes=path.stat().st_size)}

        def manifest():
            job, record = R2MP.plan_upload(bucket, plan)
            return record, BOX.upload(job)

        cases = {
            'the bytes sent are not the bytes hashed': lambda doc: doc['files']['f']['sent'].reverse(),
            'the box hashed 1 parts': lambda doc: doc['files']['f']['hashed'].pop(),
            'has no sha256': lambda doc: doc['files']['f'].update(sha256='zz'),
            'the plan has %d' % path.stat().st_size: lambda doc: doc['files']['f'].update(bytes=5),
            'the box upload failed': lambda doc: doc.update(ok=False, errors=['f: disk error']),
            'no upload manifest': lambda doc: doc.update(version='other'),
            'has no file f': lambda doc: doc['files'].clear(),
        }
        for message, damage in cases.items():
            record, doc = manifest()
            damage(doc)
            with self.assertRaises(R2MP.R2TransferError) as caught:
                R2MP.finish_upload(bucket, record, doc)
            self.assertIn(message, str(caught.exception))
            self.assertEqual((self.r2.keys(), self.r2.open_uploads()), ([], []), message)
        record, doc = manifest()
        done = R2MP.finish_upload(bucket, record, doc)['f']
        self.assertEqual((done['sha256'], done['verify'], done['parts']), (sha256(path.read_bytes()), 'etag', 2))

    def test_a_part_that_the_store_refuses_is_named_and_fails_the_upload(self):
        name = 'master-00001-of-00002.safetensors'
        path = self.master_dir / name
        bucket = AU.open_life_bucket()
        job, record = R2MP.plan_upload(bucket, {'f': dict(key='%s/%s' % (PENDING, name), path=str(path),
                                                         bytes=path.stat().st_size)})
        # The store drops the upload before the box sends its parts: it answers every part with an error status.
        self.r2.client.abort_multipart_upload(Bucket=BUCKET, Key=record['f']['key'], UploadId=record['f']['upload_id'])
        doc = BOX.upload(dict(job, attempts=1))
        self.assertFalse(doc['ok'])
        self.assertRegex(doc['errors'][0], r'HTTP [45]\d\d')
        self.assertIn('of f failed 1 times', doc['errors'][0])
        with self.assertRaises(R2MP.R2TransferError) as caught:
            R2MP.finish_upload(bucket, record, doc)
        self.assertIn('the box upload failed', str(caught.exception))
        self.assertEqual((self.r2.keys(), self.r2.open_uploads()), ([], []))


class TestParallelCopy(R2Case):
    def promote(self):
        return Path(AU.promote_pending_master_to_live(candidate_id=C, digest=M))

    def test_promote_copies_in_parts_inside_r2_and_checks_the_copy_by_etag(self):
        self.early_upload()
        self.r2.calls.clear()
        public = self.promote()

        for name, data in self.master.items():
            self.assertEqual(self.r2.body('%s/%s' % (LIVE, name)), data)
            self.assertEqual(self.r2.etag('%s/%s' % (LIVE, name)), self.r2.etag('%s/%s' % (PENDING, name)))
        # 3 + 2 part copies for the two shards, one plain copy for the small meta.json, and no download.
        self.assertEqual((self.count('UploadPartCopy'), self.count('CopyObject')), (5, 1))
        self.assertEqual(set(self.r2.calls), {'HeadObject', 'CreateMultipartUpload', 'UploadPartCopy',
                                              'CompleteMultipartUpload', 'CopyObject'})
        self.assertEqual(self.r2.open_uploads(), [])

        live = json.loads(public.read_text())
        self.assertEqual(public, self.public_index / ('%s.json' % C))
        self.assertEqual({name: entry['r2_key'] for name, entry in live['files'].items()},
                         {name: '%s/%s' % (LIVE, name) for name in self.master})
        self.assertEqual({name: entry['sha256'] for name, entry in live['files'].items()},
                         {name: sha256(data) for name, data in self.master.items()})
        self.assertEqual(live['meta_json'], {'r2_key': LIVE + '/meta.json'})
        self.assertEqual((live['master_digest'], live['bf16_sha256'], live['pending']), (M, ENTRY, False))

    def test_promote_refuses_a_pending_object_that_changed_after_its_upload(self):
        self.early_upload()
        name = 'master-00002-of-00002.safetensors'
        self.r2.client.put_object(Bucket=BUCKET, Key='%s/%s' % (PENDING, name), Body=os.urandom(len(self.master[name])))
        with self.assertRaises(AU.ArtifactUploadError) as caught:
            self.promote()
        self.assertIn('changed after its upload', str(caught.exception))
        self.assertFalse(self.public_index.exists())
        self.assertEqual(self.r2.open_uploads(), [])
        self.assertNotIn('%s/%s' % (LIVE, name), self.r2.keys())

    def test_promote_refuses_another_master_digest_and_a_missing_index(self):
        with self.assertRaises(AU.ArtifactUploadError) as caught:
            self.promote()
        self.assertIn('pending master index missing', str(caught.exception))
        self.early_upload()
        with self.assertRaises(AU.ArtifactUploadError) as caught:
            AU.promote_pending_master_to_live(candidate_id=C, digest='9' * 64)
        self.assertIn('pending master digest mismatch', str(caught.exception))
        self.assertEqual(self.r2.keys(LIVE + '/'), [])
        self.assertFalse(self.public_index.exists())

    def test_an_index_of_the_serial_transport_is_checked_by_a_stream_hash(self):
        # An index with sha256 only (no ETag record): the host cannot tie an ETag to the sha256, so it hashes.
        for name, data in self.master.items():
            self.r2.client.put_object(Bucket=BUCKET, Key='%s/%s' % (PENDING, name), Body=data)
        inventory = {name: dict(sha256=sha256(data), bytes=len(data)) for name, data in self.master.items()}
        AU.write_pending_master_index(candidate_id=C, bf16=ENTRY, digest=M, inv=inventory)
        self.promote()
        self.assertEqual(self.count('GetObject'), 3)
        self.assertEqual(self.count('UploadPartCopy'), 5)
        for name, data in self.master.items():
            self.assertEqual(self.r2.body('%s/%s' % (LIVE, name)), data)

        # The same index with a sha256 that is not the sha256 of the object: refused, no public index.
        (self.public_index / ('%s.json' % C)).unlink()
        inventory['meta.json']['sha256'] = sha256(b'other')
        AU.write_pending_master_index(candidate_id=C, bf16=ENTRY, digest=M, inv=inventory)
        with self.assertRaises(AU.ArtifactUploadError) as caught:
            self.promote()
        self.assertIn('promote verify mismatch meta.json', str(caught.exception))
        self.assertFalse((self.public_index / ('%s.json' % C)).exists())

    def test_a_part_copy_runs_again_after_a_timeout_and_gives_up_after_its_tries(self):
        self.early_upload()
        name = 'master-00001-of-00002.safetensors'
        entry = self.index()['files'][name]
        bucket = AU.open_life_bucket()
        client = R2MP.transfer_client(bucket)
        copy_part, state = client.upload_part_copy, dict(fail=2, calls=0)

        def flaky(**kwargs):
            state['calls'] += 1
            if state['fail']:
                state['fail'] -= 1
                raise ReadTimeoutError(endpoint_url=self.r2.endpoint)
            return copy_part(**kwargs)

        client.upload_part_copy = flaky
        with mock.patch.object(R2MP, 'BACKOFF_S', 0.0):
            copied = R2MP.server_copy(bucket, entry['r2_key'], '%s/%s' % (LIVE, name), size=entry['bytes'],
                                      etag=entry['etag'], part=entry['part_bytes'], client=client)
            self.assertEqual((copied['parts'], copied['etag'], state['calls']), (3, entry['etag'], 5))
            self.assertEqual(self.r2.body('%s/%s' % (LIVE, name)), self.master[name])

            state.update(fail=10 ** 6, calls=0)
            with self.assertRaises(R2MP.R2TransferError) as caught:
                R2MP.server_copy(bucket, entry['r2_key'], LIVE + '/again', client=client, part=PART)
        self.assertIn('copy of part', str(caught.exception))
        self.assertEqual(state['calls'], 3 * R2MP.ATTEMPTS)
        self.assertEqual(self.r2.open_uploads(), [])
        self.assertNotIn(LIVE + '/again', self.r2.keys())

    def test_a_4_gb_shard_is_16_part_copies_with_a_long_read_timeout(self):
        self.assertEqual(len(BOX.part_ranges(4 * 1024 * MIB, R2MP.PART_BYTES)), 16)
        self.assertEqual(BOX.part_ranges(4 * 1024 * MIB + 1, R2MP.PART_BYTES)[-1], (4 * 1024 * MIB, 1))
        os.environ.pop('PIT_R2_PART_BYTES')
        self.assertEqual((R2MP.part_bytes(), R2MP.workers()), (256 * MIB, 16))
        config = R2MP.transfer_client(AU.open_life_bucket()).meta.config
        self.assertEqual((config.read_timeout, config.connect_timeout), (900, 30))
        self.assertEqual(config.retries['total_max_attempts'], R2MP.ATTEMPTS)
        self.assertGreaterEqual(config.max_pool_connections, 16)
        with mock.patch.dict(os.environ, {'PIT_R2_PART_BYTES': str(MIB)}), self.assertRaises(R2MP.R2TransferError):
            R2MP.part_bytes()                               # R2 refuses parts below 5 MiB

    def test_gc_removes_the_pending_objects_and_every_unfinished_upload(self):
        self.early_upload()
        self.r2.client.create_multipart_upload(Bucket=BUCKET, Key=PENDING + '/master-00009.safetensors')
        self.r2.client.create_multipart_upload(Bucket=BUCKET, Key='pit-shadow/pending/%s/other' % ('f' * 64))
        out = AU.gc_pending_artifacts(C)
        self.assertTrue(out['ok'])
        self.assertIn('unfinished-uploads:1', out['removed'])
        self.assertEqual(self.r2.keys(), [])
        self.assertEqual(self.r2.open_uploads(), ['pit-shadow/pending/%s/other' % ('f' * 64)])     # another candidate
        self.assertFalse((self.pending_index / ('%s.json' % C)).exists())


if __name__ == '__main__':
    unittest.main()
