"""Exec a CPU-only owner for the isolated native grader, including after CUDA init.

The caller never forks. The broker never imports torch. Its native parent forks
only fresh disposable grading children, with the original CPU/memory/wall limits.
This research-only service does not sign scopes or renew an expired capability.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

MAX_SCOPE_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESULT_BYTES = 16 * 1024


class NativeInfrastructureError(RuntimeError):
    """Never a negative task outcome; abort this original experiment."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _bounded_write(fd, data, timeout):
    deadline = time.monotonic() + max(0., timeout)
    was_blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    offset = 0
    try:
        while offset < len(data):
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise NativeInfrastructureError('native broker write deadline')
            try:
                wrote = os.write(fd, memoryview(data)[offset:offset+16384])
            except BlockingIOError:
                continue
            if wrote <= 0:
                raise NativeInfrastructureError('native broker write channel closed')
            offset += wrote
    except OSError as exc:
        raise NativeInfrastructureError('native broker write channel failed') from exc
    finally:
        os.set_blocking(fd, was_blocking)


def authenticate(scope, authority, job_id, environment_sha256):
    from nacl.signing import VerifyKey
    try:
        if set(scope) != {'payload', 'signer', 'signature'} or scope['signer'] != authority:
            raise ValueError('signed capability schema/authority')
        VerifyKey(bytes.fromhex(authority)).verify(canonical(scope['payload']),
            base64.b64decode(scope['signature'], validate=True))
        p = scope['payload']
        if (p['job_id'] != job_id or p['environment_sha256'] != environment_sha256
                or p['version'] != 'private-job-scoped-native-math-grader-v2'):
            raise ValueError('private capability version/job/environment')
        return p
    except Exception as exc:
        raise NativeInfrastructureError('native broker capability authentication') from exc


def decode_result(result, *, reference_preflight=False):
    """Keep exact original worker scalar and reference-unparseable semantics."""
    if not isinstance(result, dict) or type(result.get('returncode')) is not int:
        raise NativeInfrastructureError('native result schema')
    if not isinstance(result.get('stdout'), str) or not isinstance(result.get('stderr'), str):
        raise NativeInfrastructureError('native output schema')
    if result['returncode'] != 0:
        try:
            detail = json.loads(result['stderr'].strip())
        except (ValueError, TypeError):
            detail = {}
        if (reference_preflight and isinstance(detail, dict)
                and detail.get('reason') == 'grader_reference_unparseable'):
            return 'reference_unparseable'
        raise NativeInfrastructureError('native infrastructure error; never negative reward')
    scalar = result['stdout'].strip()
    if scalar not in ('0.0', '1.0'):
        raise NativeInfrastructureError('native binary scalar schema')
    return scalar


class BrokerClient:
    """One bounded signed CPU session; no automatic retries or grading cache."""
    def __init__(self, scope, *, authority, job_id, environment_sha256):
        self.process = None
        self.buffer = b''
        self.sequence = 0
        self.closed = False
        self.last_result = None
        self.observations = []
        scope = json.loads(canonical(scope))
        p = authenticate(scope, authority, job_id, environment_sha256)
        self.payload = p
        broker_python = Path(p['broker_interpreter'])
        if (not broker_python.is_absolute() or broker_python.parent.name != 'bin'
                or hashlib.sha256(broker_python.resolve().read_bytes()).hexdigest() != p['broker_interpreter_sha256']):
            raise NativeInfrastructureError('prepared broker interpreter identity')
        # Source and broker bytes are also checked by the authenticated CPU owner.
        script = Path(__file__).resolve()
        from subnet.native_math_grader import isolated_argv
        command = isolated_argv(broker_python, script, ['--cpu-broker'])
        command.insert(1, '-B')
        started = time.monotonic()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, close_fds=True,
                start_new_session=True, env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent',
                    'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            self._send(dict(scope=scope, authority=authority, job_id=job_id,
                environment_sha256=environment_sha256, parent_pid=os.getpid()), MAX_SCOPE_BYTES)
            ready = self._receive(45)
            if (ready.get('status') != 'ready' or ready.get('job_id') != job_id
                    or ready.get('broker_pid') != self.process.pid):
                raise NativeInfrastructureError('CPU broker initialization')
            self.observations.append(dict(stage='native_broker_ready',
                seconds=time.monotonic()-started, broker_pid=self.process.pid,
                native_parent_pid=ready['native_parent_pid']))
        except BaseException:
            self.close()
            raise

    def _send(self, value, cap=MAX_REQUEST_BYTES):
        data = canonical(value) + b'\n'
        if len(data) > cap:
            raise NativeInfrastructureError('native broker request byte limit')
        try:
            _bounded_write(self.process.stdin.fileno(), data,
                min(15., self.payload['expires_at']-time.time()))
        except (OSError, ValueError) as exc:
            raise NativeInfrastructureError('native broker channel closed') from exc

    def _receive(self, timeout):
        deadline = time.monotonic() + max(0., timeout)
        fd = self.process.stdout.fileno()
        while b'\n' not in self.buffer:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise NativeInfrastructureError('native broker response timeout')
            data = os.read(fd, MAX_RESULT_BYTES+1)
            if not data:
                raise NativeInfrastructureError('native broker exited')
            self.buffer += data
            if len(self.buffer) > MAX_RESULT_BYTES:
                raise NativeInfrastructureError('native broker response byte limit')
        row, self.buffer = self.buffer.split(b'\n', 1)
        try:
            result = json.loads(row)
        except (ValueError, UnicodeDecodeError) as exc:
            raise NativeInfrastructureError('native broker invalid JSON') from exc
        if not isinstance(result, dict) or result.get('status') == 'error':
            raise NativeInfrastructureError('native broker failed')
        return result

    def _request(self, mode, index, task_hash, reply):
        started = time.monotonic()
        try:
            if self.closed or not time.time() < self.payload['expires_at']:
                raise NativeInfrastructureError('native broker closed/expired')
            self.sequence += 1
            self._send(dict(mode=mode, sequence=self.sequence, index=index,
                task_hash=task_hash, reply=reply))
            result = self._receive(min(20., self.payload['expires_at']-time.time()+1))
            if (result.get('status') != 'graded' or result.get('sequence') != self.sequence
                    or result.get('job_id') != self.payload['job_id']):
                raise NativeInfrastructureError('native broker response identity')
            self.last_result = result['native']
            observation = dict(stage='native_grade', mode=mode, index=index,
                task_hash=task_hash, seconds=time.monotonic()-started,
                child_seconds=result['native']['elapsed_seconds'],
                fresh_child_pid=result['native']['fresh_child_pid'],
                returncode=result['native']['returncode'])
            self.observations.append(observation)
            return decode_result(result['native'], reference_preflight=(mode == 'preflight'))
        except BaseException:
            self.close()
            raise

    def preflight(self, *, index, task_hash, answer=None):
        # The CPU owner constructs the reference from the authenticated snapshot.
        # Optional caller answer is checked there, never used as a new reference.
        return self._request('preflight', index, task_hash, answer)

    def grade(self, *, index, task_hash, reply):
        return self._request('grade', index, task_hash, reply)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=3)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=3)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _serve():
    import runpy
    import ctypes
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules):
        raise NativeInfrastructureError('CPU broker isolation')
    # This file and sibling engine are pinned by the executable ROOT scope.
    engine = runpy.run_path(str(Path(__file__).with_name('engine.py')))
    read = engine['_bounded_line']
    first, pending = read(0, MAX_SCOPE_BYTES, time.monotonic()+30)
    init = json.loads(first)
    if set(init) != {'scope', 'authority', 'job_id', 'environment_sha256', 'parent_pid'}:
        raise NativeInfrastructureError('CPU broker bootstrap schema')
    p = authenticate(init['scope'], init['authority'], init['job_id'], init['environment_sha256'])
    if (Path(sys.executable).resolve() != Path(p['broker_interpreter']).resolve()
            or hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest() != p['broker_interpreter_sha256']):
        raise NativeInfrastructureError('running broker interpreter identity')
    if type(init['parent_pid']) is not int:
        raise NativeInfrastructureError('CPU broker parent identity')
    def terminate(*_):
        raise SystemExit(75)
    signal.signal(signal.SIGTERM, terminate)
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != init['parent_pid']:
        raise NativeInfrastructureError('CPU broker parent changed')
    sys.path.insert(0, p['source_root'])
    session = None
    try:
        session = engine['JobScopedGrader'](init['scope'], authority=init['authority'],
            job_id=init['job_id'], environment_sha256=init['environment_sha256'],
            before_model_construction=True)
        own = str(Path(__file__).resolve())
        if session._identities.get(own) is None:
            raise NativeInfrastructureError('broker source is not in authenticated scope')
        print(json.dumps(dict(status='ready', job_id=p['job_id'], broker_pid=os.getpid(),
            native_parent_pid=session._process.pid)), flush=True)
        count = 0
        while True:
            row, pending = read(0, MAX_REQUEST_BYTES,
                time.monotonic()+max(0., p['expires_at']-time.time()), pending)
            if row is None:
                break
            request = json.loads(row)
            count += 1
            if (set(request) != {'mode', 'sequence', 'index', 'task_hash', 'reply'}
                    or type(request['sequence']) is not int or request['sequence'] != count
                    or count > p['max_requests'] or request['mode'] not in ('preflight', 'grade')):
                raise NativeInfrastructureError('CPU broker request sequence/mode')
            reply = request['reply']
            if request['mode'] == 'preflight':
                index = request['index']
                if type(index) is not int or str(index) not in p['tasks']:
                    raise NativeInfrastructureError('CPU broker reference task')
                answer = session._tasks[index]['data']['answer']
                if reply is not None and reply != answer:
                    raise NativeInfrastructureError('CPU broker reference mismatch')
                reply = '\\boxed{' + answer + '}'
            result = session.grade(index=request['index'], task_hash=request['task_hash'], reply=reply)
            print(json.dumps(dict(status='graded', job_id=p['job_id'], sequence=count,
                native=result)), flush=True)
    finally:
        if session is not None:
            session.close()


if __name__ == '__main__':
    if sys.argv[1:] != ['--cpu-broker']:
        raise SystemExit('CPU broker only')
    try:
        _serve()
    except Exception as exc:
        print(json.dumps(dict(status='error', kind=type(exc).__name__)), flush=True)
        raise SystemExit(75)
