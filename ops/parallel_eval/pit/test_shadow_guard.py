"""Shadow mode can never write live state or a live R2 prefix.

Real writes are tried against a real directory tree and a real S3 server (moto). The guard under test is the
real one. After each set of refused writes the live tree and the live objects are compared with their start.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import boto3

from ops.parallel_eval.pit import run_host
from ops.parallel_eval.pit import shadow_guard as SG
from ops.parallel_eval.pit.durable import DurableStore, pit_state_root
from ops.parallel_eval.pit.r2_fixture import BUCKET, R2Server

LIVE_FILES = (
    'life/CURRENT.json', 'state/HEAD-CHAIN.json', 'state/ledger.jsonl', 'state/weights.json', 'web/mining.json',
    'web/llms.txt', 'web/index.html', 'gate/epochs/e7/DECISION.json',
)
LIVE_KEYS = ('pit-pending/e7/model.safetensors', 'fp32-master/aa/model.safetensors', 'ckpts/aa/model.safetensors',
             'checkpoints/head.json', 'pit-shadow-not/x', 'pit-shadowx')


def digest(root: Path) -> str:
    out = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        out.update(('%s|%s|%o\n' % (path.relative_to(root), path.is_symlink(), path.lstat().st_mode)).encode())
        if path.is_file():
            out.update(path.read_bytes())
    return out.hexdigest()


class GuardCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(os.path.realpath(tmp.name))
        self.live, self.shadow = self.root / 'live', self.root / 'shadow'
        for rel in LIVE_FILES:
            (self.live / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.live / rel).write_text('live %s\n' % rel)
        self.shadow.mkdir()
        self.addCleanup(SG.leave)
        # No scratch directory: the test tree is in the temp directory of the system.
        self.guard = SG.ShadowGuard(state_root=self.shadow, scratch=()).activate()


class TestFiles(GuardCase):
    def test_every_write_primitive_is_refused_on_each_live_file(self):
        before = digest(self.live)
        inside = self.shadow / 'own.json'
        inside.write_text('{}')
        for rel in LIVE_FILES:
            path = self.live / rel
            writes = {
                'open w': lambda: open(path, 'w'),
                'open a': lambda: open(path, 'a'),
                'open r+': lambda: open(path, 'r+'),
                'os.open': lambda: os.open(path, os.O_WRONLY | os.O_CREAT),
                'write_text': lambda: path.write_text('shadow'),
                'write_bytes': lambda: path.write_bytes(b'shadow'),
                'new file': lambda: (path.parent / 'new.json').write_text('shadow'),
                'remove': lambda: os.remove(path),
                'unlink': lambda: path.unlink(),
                'rename away': lambda: os.rename(path, self.shadow / 'taken'),
                'replace onto': lambda: os.replace(inside, path),
                'mkdir': lambda: os.mkdir(path.parent / 'newdir'),
                'makedirs': lambda: os.makedirs(path.parent / 'a' / 'b'),
                'rmdir': lambda: os.rmdir(path.parent),
                'chmod': lambda: os.chmod(path, 0o777),
                'utime': lambda: os.utime(path, (1, 1)),
                'truncate': lambda: os.truncate(path, 0),
                'symlink': lambda: os.symlink(inside, path.parent / 'link'),
                'hard link': lambda: os.link(inside, path.parent / 'hard'),
                'copyfile': lambda: shutil.copyfile(inside, path),
                'move': lambda: shutil.move(str(inside), str(path)),
                'rmtree': lambda: shutil.rmtree(path.parent),
            }
            for name, write in writes.items():
                with self.assertRaises(SG.ShadowWriteError, msg='%s %s' % (name, rel)):
                    write()
        SG.leave()
        self.assertEqual(digest(self.live), before)
        self.assertTrue(inside.is_file())
        self.assertEqual(len(self.guard.denied), len(LIVE_FILES) * 22)

    def test_a_symlink_in_the_shadow_root_does_not_open_live(self):
        before = digest(self.live)
        os.symlink(self.live / 'state', self.shadow / 'state')              # a link in the shadow root is permitted
        os.symlink(self.live / 'web' / 'mining.json', self.shadow / 'mining.json')
        for write in (
            lambda: (self.shadow / 'state' / 'weights.json').write_text('shadow'),
            lambda: (self.shadow / 'state' / 'new.json').write_text('shadow'),
            lambda: (self.shadow / 'mining.json').write_text('shadow'),
            lambda: os.remove(self.shadow / 'state' / 'ledger.jsonl'),
            lambda: (self.shadow / '..' / 'live' / 'web' / 'llms.txt').write_text('shadow'),
        ):
            with self.assertRaises(SG.ShadowWriteError):
                write()
        os.remove(self.shadow / 'mining.json')                             # the link itself is in the shadow root
        SG.leave()
        self.assertEqual(digest(self.live), before)

    def test_shadow_reads_live_and_writes_its_own_root(self):
        self.assertEqual((self.live / 'life' / 'CURRENT.json').read_text(), 'live life/CURRENT.json\n')
        work = self.shadow / 'pit' / 'items'
        work.mkdir(parents=True)
        (work / 'a.json').write_text('{}')
        os.replace(work / 'a.json', work / 'b.json')
        with open(work / 'log.jsonl', 'a') as handle:
            handle.write('{}\n')
        shutil.rmtree(work)
        self.assertEqual(self.guard.denied, [])

    def test_the_guard_is_off_after_leave(self):
        SG.leave()
        (self.live / 'state' / 'weights.json').write_text('a live process writes this')
        self.assertIsNone(SG.active())


class TestClaim(GuardCase):
    def test_a_shadow_claim_is_not_put_in_a_live_directory(self):
        before = digest(self.live)
        env = {SG.STATE_ENV: str(self.shadow), 'PE_STATE': str(self.live / 'state'), 'PE_LIFE_STATE': str(self.live / 'life')}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(pit_state_root(), self.shadow / 'pit')
            store = DurableStore(root=pit_state_root())
            path = store.write_claim(7, candidate='c' * 64, head='a' * 64, item_id='i' * 32)
        self.assertTrue(path.is_file() and str(path).startswith(str(self.shadow)))
        SG.leave()
        self.assertEqual(digest(self.live), before)
        self.assertEqual(self.guard.denied, [])


class TestEnterShadow(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(SG.leave)
        self.root = Path(os.path.realpath(tmp.name))
        self.live = self.root / 'live'
        (self.live / 'state').mkdir(parents=True)

    def test_shadow_does_not_start_without_its_own_state_root(self):
        live_env = {'PE_STATE': str(self.live / 'state'), 'LIFE_ROOT': str(self.live / 'life')}
        bad = {
            'not set': {},
            'relative': {SG.STATE_ENV: 'shadow'},
            'the root directory': {SG.STATE_ENV: '/'},
            'a live root': {SG.STATE_ENV: str(self.live / 'state')},
            'inside a live root': {SG.STATE_ENV: str(self.live / 'state' / 'shadow')},
            'above a live root': {SG.STATE_ENV: str(self.live)},
        }
        for name, env in bad.items():
            with self.assertRaises(SG.ShadowConfigError, msg=name):
                SG.enter_shadow(dict(live_env, **env))
            self.assertIsNone(SG.active(), name)

    def test_enter_shadow_binds_the_state_root_and_the_r2_prefixes(self):
        env = {SG.STATE_ENV: str(self.root / 'shadow'), 'PE_STATE': str(self.live / 'state'), 'PIT_STATE': str(self.live / 'state' / 'pit'),
               'PIT_R2_PENDING_PREFIX': 'pit-pending'}
        guard = SG.enter_shadow(env, scratch=())
        self.assertIs(SG.active(), guard)
        self.assertEqual(env['PIT_STATE'], str(self.root / 'shadow' / 'pit'))
        self.assertEqual(env['PE_STATE'], str(self.live / 'state'))          # a live input: read only
        for name in ('PIT_R2_PENDING_PREFIX', 'PIT_R2_FP32_PREFIX', 'PIT_R2_CKPT_PREFIX'):
            self.assertTrue(env[name].startswith(SG.R2_PREFIX), name)
        with self.assertRaises(SG.ShadowWriteError):
            (self.live / 'state' / 'pit-claim.json').write_text('{}')
        (self.root / 'shadow' / 'ok.json').write_text('{}')

    def test_the_output_prefix_of_the_controller_decides_the_r2_write_prefixes(self):
        run = 'pit-shadow/batchctl/run1/'
        names = ('PIT_R2_OUTPUT_PREFIX', 'PIT_R2_PENDING_PREFIX', 'PIT_R2_FP32_PREFIX', 'PIT_R2_CKPT_PREFIX')
        cases = {
            'the argument': ({}, {'r2_prefix': run}, run),
            'the argument with no "/" at its end': ({}, {'r2_prefix': run.rstrip('/')}, run),
            'the variable': ({'PIT_R2_OUTPUT_PREFIX': run}, {}, run),
            'the argument wins over the variable': ({'PIT_R2_OUTPUT_PREFIX': 'pit-shadow/other/'}, {'r2_prefix': run}, run),
            'no output prefix': ({}, {}, SG.R2_PREFIX),
        }
        for name, (given, kwargs, expected) in cases.items():
            # The live prefixes of the environment never stay.
            env = dict(given, **{SG.STATE_ENV: str(self.root / 'shadow'), 'PIT_R2_FP32_PREFIX': 'fp32-master',
                                 'PIT_R2_PENDING_PREFIX': 'pit-pending'})
            SG.enter_shadow(env, scratch=(), **kwargs)
            self.assertEqual([env[key] for key in names], [
                expected, expected + 'pending', expected + 'fp32-master', expected + 'ckpts'], name)
            SG.leave()

    def test_an_output_prefix_outside_the_allowlist_stops_the_start(self):
        bad = ('fp32-master/', 'pit-pending/run1/', 'pit-shadowx/run1/', 'pit/batchctl/run1/', 'pit-shadow/../fp32-master/',
               '/pit-shadow/run1/', 'pit-shadow//run1/', 'pit-shadow/run 1/', 'pit-shadow/.hidden/', '/', '', 7)
        for prefix in bad:
            ways = [{'r2_prefix': prefix}]
            if isinstance(prefix, str) and prefix:
                ways.append({'PIT_R2_OUTPUT_PREFIX': prefix})          # the same prefix, as the variable
            for way in ways:
                start = {SG.STATE_ENV: str(self.root / 'shadow'), 'PIT_R2_PENDING_PREFIX': 'pit-pending'}
                start.update({key: value for key, value in way.items() if key != 'r2_prefix'})
                env = dict(start)
                with self.assertRaises(SG.ShadowConfigError, msg=repr(way)):
                    SG.enter_shadow(env, scratch=(), r2_prefix=way.get('r2_prefix'))
                self.assertIsNone(SG.active(), way)
                self.assertEqual(env, start, way)                       # a refused start changes nothing

    def test_enter_shadow_puts_the_host_directories_of_the_box_pipeline_in_the_state_root(self):
        shadow = self.root / 'shadow'
        env = {SG.STATE_ENV: str(shadow), 'PE_STATE': str(self.live / 'state'),
               'PIT_UPLOAD_ROOT': str(self.live / 'state' / 'pit' / 'uploads'),
               'PIT_FP32_PENDING_INDEX': str(self.live / 'state' / 'pit' / 'fp32-pending')}
        SG.enter_shadow(env, scratch=())
        self.assertEqual((env['PIT_UPLOAD_ROOT'], env['PIT_FP32_PENDING_INDEX']),
                         (str(shadow / 'pit' / 'uploads'), str(shadow / 'pit' / 'fp32-pending')))
        SG.leave()
        # Not set: the default would be in the live PE state. A directory in the shadow state root stays.
        env = {SG.STATE_ENV: str(shadow), 'PE_STATE': str(self.live / 'state'),
               'PIT_FP32_PENDING_INDEX': str(shadow / 'own-index')}
        SG.enter_shadow(env, scratch=())
        self.assertEqual((env['PIT_UPLOAD_ROOT'], env['PIT_FP32_PENDING_INDEX']),
                         (str(shadow / 'pit' / 'uploads'), str(shadow / 'own-index')))

    def run_host(self, mode, prefix):
        """The host runner with ``--r2-output-prefix``. -> (exit code, error, environment after, files made)."""
        flag = self.root / 'flag.json'
        flag.write_text(json.dumps({'parallel_eval': {
            'enabled': True, 'pit': {'enabled': True, 'mode': mode, 'cutover': mode == 'live'}}}))
        env = {SG.STATE_ENV: str(self.root / 'shadow'), 'PE_STATE': str(self.live / 'state'),
               'PIT_R2_PENDING_PREFIX': 'pit-pending'}
        said = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stderr(said):
            code = run_host.main(['--flag', str(flag), '--once', '--dry-run', '--r2-output-prefix', prefix])
            after = dict(os.environ)
        made = sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*'))
        return code, json.loads(said.getvalue())['error'], after == env, made

    def test_the_host_runner_refuses_an_output_prefix_that_it_can_not_use(self):
        nothing = ['flag.json', 'live', 'live/state']
        # The live lane has fixed prefixes: the argument is for shadow only.
        self.assertEqual(self.run_host('live', 'pit-shadow/batchctl/run1/'),
                         (2, '--r2-output-prefix is shadow-only', True, nothing))
        code, error, same_env, made = self.run_host('shadow', 'fp32-master/')
        self.assertEqual((code, same_env, made), (2, True, nothing))
        self.assertIn('fp32-master/ is not for shadow', error)
        self.assertIsNone(SG.active())


class TestR2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.r2 = R2Server(Path(cls.tmp.name)).start()

    @classmethod
    def tearDownClass(cls):
        cls.r2.stop()
        cls.tmp.cleanup()

    def setUp(self):
        self.r2.reset()
        self.s3 = boto3.client('s3', endpoint_url=self.r2.endpoint, region_name='us-east-1',
                               aws_access_key_id='testing', aws_secret_access_key='testing')
        self.s3.create_bucket(Bucket=BUCKET)
        for key in LIVE_KEYS:
            self.s3.put_object(Bucket=BUCKET, Key=key, Body=b'live ' + key.encode())
        self.addCleanup(SG.leave)
        self.guard = SG.ShadowGuard(state_root=Path(self.tmp.name) / 'shadow', scratch=()).activate()

    def objects(self):
        SG.leave()
        rows = self.s3.list_objects_v2(Bucket=BUCKET).get('Contents', [])
        return {row['Key']: self.s3.get_object(Bucket=BUCKET, Key=row['Key'])['Body'].read() for row in rows}

    def test_every_write_operation_is_refused_on_a_live_prefix(self):
        own = SG.R2_PREFIX + 'pending/e7/model.safetensors'
        self.s3.put_object(Bucket=BUCKET, Key=own, Body=b'shadow')
        for key in LIVE_KEYS + ('pit-shadow/../fp32-master/x',):
            writes = {
                'put_object': lambda: self.s3.put_object(Bucket=BUCKET, Key=key, Body=b'shadow'),
                'delete_object': lambda: self.s3.delete_object(Bucket=BUCKET, Key=key),
                'delete_objects': lambda: self.s3.delete_objects(
                    Bucket=BUCKET, Delete={'Objects': [{'Key': own}, {'Key': key}]}),
                'copy_object': lambda: self.s3.copy_object(Bucket=BUCKET, Key=key, CopySource={'Bucket': BUCKET, 'Key': own}),
                'create_multipart_upload': lambda: self.s3.create_multipart_upload(Bucket=BUCKET, Key=key),
                'upload_part': lambda: self.s3.upload_part(Bucket=BUCKET, Key=key, UploadId='u', PartNumber=1, Body=b'x'),
                'upload_part_copy': lambda: self.s3.upload_part_copy(
                    Bucket=BUCKET, Key=key, UploadId='u', PartNumber=1, CopySource={'Bucket': BUCKET, 'Key': own}),
                'complete_multipart_upload': lambda: self.s3.complete_multipart_upload(
                    Bucket=BUCKET, Key=key, UploadId='u', MultipartUpload={'Parts': []}),
                'upload_fileobj': lambda: self.s3.upload_file(__file__, BUCKET, key),
                'presigned put_object': lambda: self.s3.generate_presigned_url('put_object', Params={'Bucket': BUCKET, 'Key': key}),
                'presigned upload_part': lambda: self.s3.generate_presigned_url(
                    'upload_part', Params={'Bucket': BUCKET, 'Key': key, 'UploadId': 'u', 'PartNumber': 1}),
                'presigned delete_object': lambda: self.s3.generate_presigned_url('delete_object', Params={'Bucket': BUCKET, 'Key': key}),
            }
            for name, write in writes.items():
                with self.assertRaises(SG.ShadowWriteError, msg='%s %s' % (name, key)):
                    write()
        for name, write in {
            'delete_bucket': lambda: self.s3.delete_bucket(Bucket=BUCKET),
            'create_bucket': lambda: self.s3.create_bucket(Bucket='second'),
            'put_bucket_policy': lambda: self.s3.put_bucket_policy(Bucket=BUCKET, Policy='{}'),
        }.items():
            with self.assertRaises(SG.ShadowWriteError, msg=name):
                write()
        expected = {key: b'live ' + key.encode() for key in LIVE_KEYS}
        expected[own] = b'shadow'
        self.assertEqual(self.objects(), expected)

    def test_shadow_reads_live_keys_and_writes_its_own_prefix(self):
        key = LIVE_KEYS[1]
        self.assertEqual(self.s3.get_object(Bucket=BUCKET, Key=key)['Body'].read(), b'live ' + key.encode())
        self.s3.head_object(Bucket=BUCKET, Key=key)
        self.assertIn('fp32-master', self.s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET, 'Key': key}))
        own = SG.R2_PREFIX + 'fp32-master/aa/model.safetensors'
        # A copy reads the live object and writes the shadow key.
        self.s3.copy_object(Bucket=BUCKET, Key=own, CopySource={'Bucket': BUCKET, 'Key': key})
        upload = self.s3.create_multipart_upload(Bucket=BUCKET, Key=own + '.mp')['UploadId']
        self.assertIn('pit-shadow', self.s3.generate_presigned_url(
            'upload_part', Params={'Bucket': BUCKET, 'Key': own + '.mp', 'UploadId': upload, 'PartNumber': 1}))
        self.s3.abort_multipart_upload(Bucket=BUCKET, Key=own + '.mp', UploadId=upload)
        self.s3.delete_objects(Bucket=BUCKET, Delete={'Objects': [{'Key': own}]})
        self.assertEqual(self.guard.denied, [])
        self.assertEqual(sorted(self.objects()), sorted(LIVE_KEYS))


if __name__ == '__main__':
    unittest.main()
