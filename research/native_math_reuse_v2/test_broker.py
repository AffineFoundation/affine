"""CPU-only equivalence/adversarial controls; ephemeral keys never authorize GPUs."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nacl.signing import SigningKey
from . import engine
from . import broker
from .broker import BrokerClient, NativeInfrastructureError, decode_result
from subnet.native_math_grader import ASSET, isolated_argv, runtime_lock


REPO = Path(__file__).resolve().parents[2]
PREPARED = Path('/home/const/.cache/uv/environments-v2/verifiers-cp3.12.3-242b229c745212ce')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed(payload, key):
    return dict(payload=payload, signer=key.verify_key.encode().hex(),
        signature=base64.b64encode(key.sign(engine.canonical(payload)).signature).decode())


class ScalarControls(unittest.TestCase):
    def test_only_original_exact_binary_scalars_are_rewards(self):
        for value in ('NaN', 'inf', '0.5', '1', 'true', '', '1.0\n0.0', '0.0 extra'):
            with self.subTest(value=value), self.assertRaises(NativeInfrastructureError):
                decode_result(dict(returncode=0, stdout=value, stderr=''))
        for value in ('0.0', '1.0'):
            self.assertEqual(decode_result(dict(returncode=0, stdout=value, stderr='')), value)

    def test_unparseable_reference_only_skips_preflight(self):
        row = dict(returncode=75, stdout='', stderr=json.dumps(dict(reason='grader_reference_unparseable')))
        self.assertEqual(decode_result(row, reference_preflight=True), 'reference_unparseable')
        with self.assertRaises(NativeInfrastructureError):
            decode_result(row)

    def test_timeout_crash_bad_json_never_becomes_negative(self):
        for code, stderr in ((75, '{"reason":"grader_timeout"}'), (-9, ''), (75, 'bad'), (75, 'null')):
            for preflight in (False, True):
                with self.subTest(code=code, preflight=preflight), self.assertRaises(NativeInfrastructureError):
                    decode_result(dict(returncode=code, stdout='0.0', stderr=stderr), reference_preflight=preflight)

    def test_fresh_child_limits_are_not_the_wall_timeout(self):
        def grade():
            print(json.dumps(dict(cpu=resource.getrlimit(resource.RLIMIT_CPU),
                memory=resource.getrlimit(resource.RLIMIT_AS), pid=os.getpid())))
        a, pid_a, _ = engine._run_child(grade, dict(reply='x'), '1', 15, 768*1024**2)
        b, pid_b, _ = engine._run_child(grade, dict(reply='x'), '1', 15, 768*1024**2)
        self.assertNotEqual(pid_a, pid_b)
        self.assertEqual(json.loads(a['stdout'])['cpu'], [8, 9])
        self.assertEqual(json.loads(b['stdout'])['memory'], [768*1024**2]*2)
        for pid in (pid_a, pid_b):
            with self.assertRaises(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)

    def test_bounded_child_timeout_is_reaped(self):
        def grade():
            time.sleep(2)
        result, pid, _ = engine._run_child(grade, dict(reply='x'), '1', .05, 768*1024**2)
        self.assertEqual(result['returncode'], 75)
        self.assertIn('timeout', result['stderr'])
        with self.assertRaises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)

    def test_private_limit_ceiling_and_independent_cpu_gate(self):
        self.assertEqual(engine.MAX_REQUESTS, 6912)
        self.assertEqual(engine.MAX_TASKS, 768)
        self.assertEqual(engine.MAX_LIFETIME_SECONDS, 7*3600)
        with self.assertRaises(engine.GraderUnavailable):
            engine._run_child(lambda: print('0.0'), dict(reply='x'), '1', 15, 768*1024**2, (15, 16))

    def test_undrained_large_pipe_has_bounded_write_and_restores_flags(self):
        for module in (engine, broker):
            readfd, writefd = os.pipe()
            started = time.monotonic()
            try:
                with self.subTest(module=module.__name__), self.assertRaises(RuntimeError):
                    module._bounded_write(writefd, b'x'*(1024*1024), .05)
                self.assertLess(time.monotonic()-started, 1)
                self.assertTrue(os.get_blocking(writefd))
            finally:
                os.close(readfd)
                os.close(writefd)


class PreparedNativeControls(unittest.TestCase):
    def setUp(self):
        if not (PREPARED/'bin/python').exists():
            self.skipTest('approved CPU native runtime is unavailable')
        self.tmp = tempfile.TemporaryDirectory(prefix='private-native-reuse-v2-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        files = ['subnet/native_math_grader.py',
            'subnet/vendor/legacy/rollouts/envs/affine_math_v1/affine_math_v1/verify.py',
            'research/native_math_reuse_v2/engine.py',
            'research/native_math_reuse_v2/broker.py']
        for relative in files:
            target = self.source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO/relative, target)
        # Test-only venv facade: unchanged prepared grader files plus host PyNaCl.
        # No dependency installation or mutation of either underlying runtime.
        self.runtime = self.root/'runtime'
        (self.runtime/'bin').mkdir(parents=True)
        (self.runtime/'bin/python').symlink_to((PREPARED/'bin/python').resolve())
        site = self.runtime/'lib/python3.12/site-packages'
        site.mkdir(parents=True)
        for item in (PREPARED/'lib/python3.12/site-packages').iterdir():
            (site/item.name).symlink_to(item)
        for module in ('nacl', 'cffi', 'pycparser', '_cffi_backend'):
            spec = importlib.util.find_spec(module)
            if spec is None or spec.origin is None:
                self.skipTest('CPU test signature dependencies unavailable')
            origin = Path(spec.origin)
            item = origin.parent if spec.submodule_search_locations else origin
            if not (site/item.name).exists():
                (site/item.name).symlink_to(item)
        self.snapshot = self.root/'tasks.json'
        self.rows = [dict(data=dict(answer='1')), dict(data=dict(answer=r'\frac{1}{2}'))]
        self.snapshot.write_bytes(engine.canonical(self.rows))
        now = time.time()
        native = self.source/files[1]
        self.key = SigningKey.generate()
        self.payload = dict(version=engine.VERSION, execute_allowed=True, job_id='cpu-controls',
            created_at=now-1, expires_at=now+90, source_root=str(self.source),
            source_files={name:digest(self.source/name) for name in engine._source_members(self.source)},
            asset_files={str(self.snapshot):digest(self.snapshot)}, grader_path=str(native),
            grader_sha256=digest(native), interpreter=str(PREPARED/'bin/python'),
            broker_interpreter=str(self.runtime/'bin/python'),
            broker_interpreter_sha256=digest((self.runtime/'bin/python').resolve()),
            environment_sha256='a'*64, snapshot_path=str(self.snapshot), snapshot_sha256=digest(self.snapshot),
            tasks={str(i):dict(task_hash=str(i+1)*64,
                row_sha256=hashlib.sha256(engine.canonical(row)).hexdigest()) for i,row in enumerate(self.rows)},
            max_requests=20, outer_timeout_seconds=15, memory_bytes=768*1024**2,
            cpu_soft_seconds=8, cpu_hard_seconds=9,
            native_runtime_binding=hashlib.sha256(engine.canonical(runtime_lock())).hexdigest())
        # Client executes the byte-identical pinned copy, not the working checkout.
        spec = importlib.util.spec_from_file_location('private_broker_fixture', self.source/files[3])
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def client(self, payload=None):
        payload = payload or self.payload
        client = self.module.BrokerClient(signed(payload, self.key),
            authority=self.key.verify_key.encode().hex(), job_id=payload['job_id'],
            environment_sha256=payload['environment_sha256'])
        self.addCleanup(client.close)
        return client

    def original(self, gold, reply):
        argv = isolated_argv(self.runtime/'bin/python', self.payload['grader_path'],
            ['--json-arguments', json.dumps([gold, reply])])
        row = subprocess.run(['/usr/bin/prlimit', '--cpu=8:9', '--as=805306368:805306368', '--core=0:0', '--', *argv],
            capture_output=True, text=True, timeout=15)
        return dict(returncode=row.returncode, stdout=row.stdout, stderr=row.stderr)

    def test_real_native_equivalence_and_fresh_children(self):
        client = self.client()
        self.assertEqual(client.preflight(index=0, task_hash='1'*64, answer='1'), '1.0')
        children = []
        cases = [(0,'1',r'\boxed{1}'), (0,'1',r'\boxed{2}'),
            (0,'1',r'\boxed{2} then \boxed{1}'), (1,r'\frac{1}{2}',r'\boxed{0.5}')]
        for index, gold, reply in cases:
            actual = client.grade(index=index, task_hash=str(index+1)*64, reply=reply)
            self.assertEqual(actual, decode_result(self.original(gold, reply)))
            children.append(client.last_result['fresh_child_pid'])
        self.assertEqual(len(set(children)), len(children))
        parent_pid = client.observations[0]['native_parent_pid']
        client.close()
        self.assertIsNotNone(client.process.poll())
        self.assertFalse(Path('/proc', str(parent_pid)).exists())

    def test_task_identity_and_reference_cannot_be_substituted(self):
        for mode in ('hash', 'reference'):
            client = self.client()
            with self.subTest(mode=mode), self.assertRaises(self.module.NativeInfrastructureError):
                if mode == 'hash':
                    client.grade(index=0, task_hash='2'*64, reply=r'\boxed{1}')
                else:
                    client.preflight(index=0, task_hash='1'*64, answer='2')
            self.assertTrue(client.closed)

    def test_source_mutation_after_success_invalidates_next_grade(self):
        client = self.client()
        self.assertEqual(client.grade(index=0, task_hash='1'*64, reply=r'\boxed{1}'), '1.0')
        path = self.source/'subnet/native_math_grader.py'
        path.write_text(path.read_text()+'\n# changed test fixture\n')
        with self.assertRaises(self.module.NativeInfrastructureError):
            client.grade(index=0, task_hash='1'*64, reply=r'\boxed{2}')
        self.assertTrue(client.closed)

    def test_native_timeout_aborts_no_negative(self):
        client = self.client()
        with self.assertRaises(self.module.NativeInfrastructureError):
            client.grade(index=0, task_hash='1'*64, reply=r'\boxed{2^{2^{1000000}}}')
        self.assertEqual(client.last_result['returncode'], 75)
        self.assertTrue(client.closed)

    def test_scope_request_exhaustion_no_silent_restart(self):
        payload = dict(self.payload, max_requests=1)
        client = self.client(payload)
        self.assertEqual(client.grade(index=0, task_hash='1'*64, reply=r'\boxed{1}'), '1.0')
        with self.assertRaises(self.module.NativeInfrastructureError):
            client.grade(index=0, task_hash='1'*64, reply=r'\boxed{2}')
        self.assertEqual(client.sequence, 2)

    def test_gpu_owner_can_call_without_forking_its_cuda_state(self):
        # The caller may already hold CUDA. Only separately exec'd owners fork.
        import sys
        with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))}):
            with patch('os.fork', side_effect=AssertionError('caller must not fork')):
                client = self.client()
                self.assertEqual(client.grade(index=0, task_hash='1'*64, reply=r'\boxed{1}'), '1.0')
                client.close()

    def test_tampered_signature_rejected_before_process_start(self):
        scope = signed(self.payload, self.key)
        scope['payload']['max_requests'] += 1
        with patch.object(self.module.subprocess, 'Popen') as spawn:
            with self.assertRaises(self.module.NativeInfrastructureError):
                self.module.BrokerClient(scope, authority=self.key.verify_key.encode().hex(),
                    job_id=self.payload['job_id'], environment_sha256=self.payload['environment_sha256'])
            spawn.assert_not_called()

    def test_original_native_interpreter_has_no_signature_dependency(self):
        command = isolated_argv(self.payload['interpreter'], self.payload['grader_path'], ['--runtime-check'])
        ready = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(ready.returncode, 0, ready.stderr)
        self.assertNotEqual(self.payload['interpreter'], self.payload['broker_interpreter'])
        client = self.client()
        self.assertEqual(client.preflight(index=1, task_hash='2'*64), '1.0')


if __name__ == '__main__':
    unittest.main()
