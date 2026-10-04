import base64
import hashlib
import io
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from botocore.exceptions import ClientError
from nacl.signing import SigningKey
from nacl.exceptions import BadSignatureError
from subnet.storage import canonical
from ops.retain_completed_training import sha
from ops.retain_checkpoint_caches import archived_files, protections, run_cycle, scopes_for


class Archive:
    def __init__(self, key, contents):
        self.name = 'test'; self.client = self; self.key = key; self.objects = {}
        self.files = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
        self.checkpoint = hashlib.sha256(canonical(self.files)).hexdigest()
        self.authority = key.verify_key.encode().hex()
        prefix = 'public/checkpoints/' + self.checkpoint + '/'
        self.objects.update({prefix + name: value for name, value in contents.items()})
        self.descriptor = prefix + 'authorities/' + self.authority + '/checkpoint.json'
        payload = {'id': self.checkpoint, 'files': self.files}
        self.objects[self.descriptor] = canonical(self.sign(payload))
        self.read = []

    def sign(self, payload):
        return {'payload': payload, 'signer': self.authority,
                'signature': base64.b64encode(self.key.sign(canonical(payload)).signature).decode()}

    def get(self, key):
        if key not in self.objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
        return self.objects[key]

    def head_object(self, Bucket, Key):
        return {'ContentLength': len(self.objects[Key])}

    def get_object(self, Bucket, Key):
        self.read.append(Key)
        return {'ContentLength': len(self.objects[Key]), 'Body': io.BytesIO(self.objects[Key]), 'ETag': 'test'}


class CacheSupervisorControls(unittest.TestCase):
    def fixture(self, root):
        key = SigningKey.generate(); archive = Archive(key, {'config.json': b'{}', 'model.safetensors': b'obsolete-model'})
        state = root / 'state'; (state / 'roles').mkdir(parents=True)
        (state / 'controller.json').write_bytes(canonical({'checkpoint': {'id': 'a' * 64}, 'active': None}))
        cache = root / 'checkpoints' / archive.checkpoint; cache.mkdir(parents=True)
        for name in archive.files:
            (cache / name).write_bytes(archive.objects['public/checkpoints/' + archive.checkpoint + '/' + name])
        endpoint = {'workspace': str(root / 'workspace'), 'python': sys.executable, 'host': 'isolated.invalid',
                    'port': 22, 'known_hosts': str(root / 'known_hosts')}
        config = root / 'config.json'; config.write_bytes(canonical({'state': str(state), 'bucket': {},
            'remote': {'roles': {'train': endpoint, 'evaluate': endpoint, 'mine': endpoint, 'verify': []}}})); config.chmod(0o600)
        ticks = Path('/proc', str(os.getpid()), 'stat').read_text().rsplit(')', 1)[1].split()[19]
        record = root / 'process.json'; record.write_bytes(canonical({'child_pid': os.getpid(), 'child_ticks': ticks,
                                                                    'config_sha256': sha(config)}))
        scopes = root / 'scopes.json'; scopes.write_bytes(canonical({'schema': 1, 'config_sha256': sha(config),
            'roles': {'train': {'endpoint_sha256': hashlib.sha256(canonical(endpoint)).hexdigest(),
                                'roots': [str(cache.parent)]}}})); scopes.chmod(0o600)
        return archive, cache, config, record, scopes, state

    def test_scopes_refuse_changed_config_endpoint_or_arbitrary_directories(self):
        with tempfile.TemporaryDirectory(prefix='affine-cache-tests-') as directory:
            root = Path(directory); archive, cache, config, record, scopes, state = self.fixture(root)
            c = json.loads(config.read_text()); original = json.loads(scopes.read_text())
            self.assertIn('train', scopes_for(scopes, config, c))
            for change in ({'config_sha256': 'b' * 64}, {'roles': {'train': dict(original['roles']['train'], roots=['/root'])}},
                           {'roles': {'train': dict(original['roles']['train'], endpoint_sha256='b' * 64)}}):
                scopes.write_bytes(canonical(dict(original, **change)))
                with self.assertRaises(ValueError): scopes_for(scopes, config, c)
            scopes.write_bytes(canonical(original)); scopes.chmod(0o644)
            with self.assertRaisesRegex(ValueError, 'private'): scopes_for(scopes, config, c)

    def test_active_manifest_and_incomplete_dispatched_job_protect_old_weights(self):
        with tempfile.TemporaryDirectory(prefix='affine-cache-tests-') as directory:
            root = Path(directory); archive, cache, config, record, scopes, state = self.fixture(root)
            manifest = {'epoch': 'original', 'checkpoint': {'id': archive.checkpoint}}
            (state / 'controller.json').write_bytes(canonical({'checkpoint': {'id': 'a' * 64},
                'active': {'epoch': 'original', 'next_checkpoint': {'id': 'b' * 64}}}))
            (state / 'original-first-signed-manifest.json').write_bytes(canonical(archive.sign(manifest)))
            job = {'role': 'evaluate', 'job_id': 'original-job', 'manifest': archive.sign({'epoch': 'older', 'checkpoint': {'id': 'c' * 64}})}
            roles = state / 'roles'
            (roles / 'older-before.json').write_bytes(canonical({'role': job['role'], 'job_id': job['job_id'], 'job_sha256': hashlib.sha256(canonical(job)).hexdigest()}))
            (roles / 'original-job-job.json').write_bytes(canonical(archive.sign(job)))
            (roles / 'original-job-failure.json').write_bytes(canonical({'reason': 'observation expired'}))
            with sqlite3.connect(roles / 'verifier-queue.sqlite3') as db:
                db.execute('create table jobs(envelope text,status text)')
                db.execute('insert into jobs values (?,?)', (json.dumps(archive.sign({'manifest': archive.sign({'checkpoint': {'id': 'd' * 64}})})), 'leased'))
            _, principal, active = protections(config, record, archive.authority)
            self.assertEqual(principal, {'a' * 64, 'b' * 64})
            self.assertEqual(active, {archive.checkpoint, 'c' * 64, 'd' * 64})
            (state / 'original-first-signed-manifest.json').write_bytes(canonical(archive.sign(dict(manifest, epoch='other'))))
            with self.assertRaisesRegex(ValueError, 'active manifest'): protections(config, record, archive.authority)

    def test_archive_reads_every_byte_and_refuses_corruption_or_wrong_signature(self):
        archive = Archive(SigningKey.generate(), {'config.json': b'{}', 'model.safetensors': b'weights'})
        rows = archived_files(archive, archive.checkpoint, archive.authority)
        self.assertEqual(sum(row['archive_read_bytes'] for row in rows.values()), 9)
        self.assertEqual(len(archive.read), 2)
        key = 'public/checkpoints/' + archive.checkpoint + '/model.safetensors'
        archive.objects[key] = b'corrupt'
        with self.assertRaisesRegex(ValueError, 'bytes changed'): archived_files(archive, archive.checkpoint, archive.authority)
        descriptor = json.loads(archive.objects[archive.descriptor]); descriptor['payload']['files']['model.safetensors'] = '0' * 64
        archive.objects[archive.descriptor] = canonical(descriptor)
        with self.assertRaises(BadSignatureError): archived_files(archive, archive.checkpoint, archive.authority)
        self.assertIsNone(archived_files(archive, 'b' * 64, archive.authority))

    def test_actual_isolated_retirement_preserves_current_cache_and_r2_bytes(self):
        with tempfile.TemporaryDirectory(prefix='affine-cache-tests-') as directory:
            root = Path(directory); archive, cache, config, record, scopes, state = self.fixture(root)
            current = cache.parent / ('a' * 64); current.mkdir(); (current / 'weights').write_bytes(b'current')
            commands = root / 'bin'; commands.mkdir(); gpu = commands / 'nvidia-smi'
            gpu.write_text('#!/bin/sh\nexit 0\n'); gpu.chmod(0o700)
            actual_run = subprocess.run; original_objects = dict(archive.objects)
            def transport(argv, **kwargs):
                if argv[0] == 'ssh':
                    command = shlex.split(argv[-1])
                    if 'result=m.remove_checkpoint_replica' in command[-1]:
                        # The isolated child owns this entire fixture. This host
                        # is unprivileged, so scan its actual /proc FD/maps rather
                        # than pretending it can read other users' processes.
                        command[-1] = command[-1].replace('before=os.statvfs(root)',
                            "m.processes=lambda:[Path('/proc',str(os.getpid()))]\nbefore=os.statvfs(root)")
                    return actual_run(command, **dict(kwargs, env=dict(os.environ, PATH=str(commands) + ':' + os.environ['PATH'])))
                if argv[0] == 'scp':
                    shutil.copyfile(argv[-2], argv[-1].split(':', 1)[1])
                    return subprocess.CompletedProcess(argv, 0, '', '')
                return actual_run(argv, **kwargs)
            with patch('ops.retain_checkpoint_caches.Bucket', return_value=archive), patch('ops.retain_checkpoint_caches.subprocess.run', side_effect=transport):
                result = run_cycle(config, scopes, archive.authority, root / 'out', record)
            self.assertEqual(result['removed_replicas'], 1); self.assertEqual(result['removed_bytes'], 16)
            self.assertFalse(cache.exists()); self.assertEqual((current / 'weights').read_bytes(), b'current')
            self.assertEqual(archive.objects, original_objects)
            self.assertTrue(list((root / 'out').glob('*/actual-retirement.private.json')))

    def test_reference_arriving_during_archive_readback_prevents_retirement(self):
        with tempfile.TemporaryDirectory(prefix='affine-cache-tests-') as directory:
            root = Path(directory); archive, cache, config, record, scopes, state = self.fixture(root)
            c = json.loads(config.read_text())
            probe = subprocess.CompletedProcess([], 0, json.dumps({'free_bytes': 100,
                'gpu_busy': False, 'candidates': [{'checkpoint': archive.checkpoint, 'directory': str(cache)}]}), '')
            with patch('ops.retain_checkpoint_caches.Bucket', return_value=archive), \
                    patch('ops.retain_checkpoint_caches.subprocess.run', return_value=probe) as transport, \
                    patch('ops.retain_checkpoint_caches.protections', side_effect=[
                        (c, {'a' * 64}, set()), (c, {'a' * 64}, {archive.checkpoint})]):
                with self.assertRaisesRegex(ValueError, 'protection changed'):
                    run_cycle(config, scopes, archive.authority, root / 'out', record)
            self.assertEqual(transport.call_count, 1)
            self.assertTrue(cache.exists()); self.assertEqual(len(archive.read), 2)
            self.assertTrue(list((root / 'out').glob('*/fresh-complete-public-readbacks.private.json')))

    def test_busy_gpu_preserves_caches_without_archive_or_helper_operation(self):
        with tempfile.TemporaryDirectory(prefix='affine-cache-tests-') as directory:
            root = Path(directory); archive, cache, config, record, scopes, state = self.fixture(root)
            probe = subprocess.CompletedProcess([], 0, json.dumps({'free_bytes': 100,
                'gpu_busy': True, 'candidates': [{'checkpoint': archive.checkpoint, 'directory': str(cache)}]}), '')
            with patch('ops.retain_checkpoint_caches.Bucket') as bucket, \
                    patch('ops.retain_checkpoint_caches.subprocess.run', return_value=probe) as transport:
                result = run_cycle(config, scopes, archive.authority, root / 'out', record)
            self.assertEqual(result['removed_replicas'], 0); self.assertEqual(result['busy_roles'], 1)
            self.assertTrue(cache.exists()); bucket.assert_not_called(); self.assertEqual(transport.call_count, 1)


if __name__ == '__main__': unittest.main()
