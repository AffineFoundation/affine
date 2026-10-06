"""Default-off, job-scoped CPU native grader preparation prototype.

No backend enables this module. A signed capability and explicit pre-model
construction call are required. The original grader bytes remain unchanged.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import select
import signal
import stat
import subprocess
import sys
import time

VERSION = 'job-scoped-isolated-fork-native-math-grader-v1'
MAX_SCOPE_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESULT_BYTES = 16 * 1024
MAX_REQUESTS = 128
MAX_LIFETIME_SECONDS = 3600
MAX_TIMEOUT_SECONDS = 300
MAX_MEMORY_BYTES = 1024 * 1024 * 1024
SPLIT = 'if len(sys.argv) == 3 and sys.argv[1] == "--json-arguments":'


class GraderUnavailable(RuntimeError):
    """Infrastructure failure, never a negative native outcome."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metadata(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode, s.st_nlink)


def _identity(path):
    try:
        s = os.lstat(path)
    except OSError as e:
        raise GraderUnavailable('pinned file unavailable') from e
    if not stat.S_ISREG(s.st_mode):
        raise GraderUnavailable('nonregular pinned file')
    return _metadata(s)


class _DirectoryGuard:
    """Cache inventory bookkeeping, never file identities or grading outcomes."""
    def __init__(self, roots, *, excluded=(), suffixes=None):
        self.excluded = set(excluded) | {'__pycache__', '.git'}
        self.suffixes = suffixes
        self.directories = {}
        for root in roots:
            for path, dirs, _ in os.walk(root, followlinks=False):
                dirs[:] = [d for d in dirs if d not in self.excluded]
                self.directories[path] = (_metadata(os.lstat(path)), self._entries(path))

    def _entries(self, path):
        result = set()
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name in self.excluded or entry.name.endswith('.pyc'):
                    continue
                directory = entry.is_dir(follow_symlinks=False)
                link = entry.is_symlink()
                if self.suffixes is None or directory or link or ('.' + entry.name.rsplit('.', 1)[-1]) in self.suffixes:
                    result.add((entry.name, 'directory' if directory else 'link' if link else 'file'))
        return result

    def valid(self):
        for path, (baseline, entries) in self.directories.items():
            try:
                current = _metadata(os.lstat(path))
                # Directory timestamps can share a filesystem clock tick.
                # Check names/types EVERY time, including same-tick additions.
                if (current[0:2] != baseline[0:2] or current[5] != baseline[5]
                        or self._entries(path) != entries):
                    return False
            except OSError:
                return False
        return True


class _PinnedIdentities(dict):
    pass


def _source_members(root):
    return {str(p.relative_to(root)) for p in root.rglob('*')
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts
            and p.suffix != '.pyc' and p.name != '.git'}


def _validate_payload(p, *, now=None):
    now = time.time() if now is None else now
    required = {'version', 'execute_allowed', 'job_id', 'created_at', 'expires_at',
                'source_root', 'source_files', 'asset_files', 'grader_path', 'grader_sha256',
                'interpreter', 'environment_sha256', 'snapshot_path', 'snapshot_sha256',
                'tasks', 'max_requests', 'outer_timeout_seconds', 'memory_bytes',
                'native_runtime_binding'}
    if set(p) != required or p['version'] != VERSION or p['execute_allowed'] is not True:
        raise GraderUnavailable('explicit job-scoped policy missing')
    if not isinstance(p['job_id'], str) or not p['job_id'] or len(p['job_id']) > 256:
        raise GraderUnavailable('job identity')
    if (type(p['created_at']) not in (int, float) or type(p['expires_at']) not in (int, float)
            or not p['created_at'] <= now < p['expires_at']
            or p['expires_at'] - p['created_at'] > MAX_LIFETIME_SECONDS):
        raise GraderUnavailable('job lifetime')
    if type(p['max_requests']) is not int or not 1 <= p['max_requests'] <= MAX_REQUESTS:
        raise GraderUnavailable('request count bound')
    if (type(p['outer_timeout_seconds']) not in (int, float)
            or not 0 < p['outer_timeout_seconds'] <= MAX_TIMEOUT_SECONDS):
        raise GraderUnavailable('original outer timeout bound')
    if type(p['memory_bytes']) is not int or not 256 * 1024**2 <= p['memory_bytes'] <= MAX_MEMORY_BYTES:
        raise GraderUnavailable('memory bound')
    for name in ('environment_sha256', 'grader_sha256', 'snapshot_sha256'):
        if not isinstance(p[name], str) or len(p[name]) != 64:
            raise GraderUnavailable('source/asset digest')
        try:
            bytes.fromhex(p[name])
        except ValueError as e:
            raise GraderUnavailable('source/asset digest') from e
    if not isinstance(p['tasks'], dict) or not 1 <= len(p['tasks']) <= MAX_REQUESTS:
        raise GraderUnavailable('task binding')
    for index, task in p['tasks'].items():
        if (not isinstance(index, str) or not index.isdigit()
                or set(task) != {'task_hash', 'row_sha256'}
                or any(not isinstance(v, str) or len(v) != 64 for v in task.values())):
            raise GraderUnavailable('task context binding')
    root = Path(p['source_root'])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise GraderUnavailable('owned source root')
    if not isinstance(p['source_files'], dict) or not p['source_files']:
        raise GraderUnavailable('complete source pins')
    if _source_members(root) != set(p['source_files']):
        raise GraderUnavailable('full source inventory mismatch')
    pins = {}
    for name, sha in p['source_files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise GraderUnavailable('source member traversal')
        pins[str(root / rel)] = sha
    if not isinstance(p['asset_files'], dict):
        raise GraderUnavailable('asset inventory')
    for name, sha in p['asset_files'].items():
        if not Path(name).is_absolute():
            raise GraderUnavailable('absolute asset pin')
        pins[name] = sha
    for name, sha in ((p['grader_path'], p['grader_sha256']), (p['snapshot_path'], p['snapshot_sha256'])):
        if not Path(name).is_absolute() or pins.get(name) != sha:
            raise GraderUnavailable('grader/snapshot not covered by authenticated inventory')
    original = Path(__file__).parent / 'vendor/legacy/rollouts/envs/affine_math_v1/affine_math_v1/verify.py'
    if p['grader_sha256'] != _digest(original):
        raise GraderUnavailable('byte-identical original native grader required')
    own = str(Path(__file__).resolve())
    if pins.get(own) != _digest(own):
        raise GraderUnavailable('CPU implementation pin')
    identities = _PinnedIdentities()
    identities.tree = _DirectoryGuard([str(root)])
    for name, sha in pins.items():
        before = _identity(name)
        if not isinstance(sha, str) or _digest(name) != sha or _identity(name) != before:
            raise GraderUnavailable('source/asset digest mismatch')
        identities[name] = before
    rows = json.loads(Path(p['snapshot_path']).read_bytes())
    for index, task in p['tasks'].items():
        i = int(index)
        if (not isinstance(rows, list) or i >= len(rows)
                or hashlib.sha256(canonical(rows[i])).hexdigest() != task['row_sha256']
                or not isinstance(rows[i].get('data', {}).get('answer'), str)):
            raise GraderUnavailable('pinned reference/task mismatch')
    return identities, rows


def _check_unchanged(p, identities):
    if not time.time() < p['expires_at']:
        raise GraderUnavailable('expired job')
    if not identities.tree.valid():
        raise GraderUnavailable('source inventory changed')
    if any(_identity(name) != value for name, value in identities.items()):
        raise GraderUnavailable('pinned source/asset changed')


def prepare_job_scoped_grader(scope=None, *, authority=None, job_id=None,
                              before_model_construction=False, environment_sha256=None):
    """Absent policy preserves the existing native path without spawning anything."""
    if scope is None:
        return None
    return JobScopedGrader(scope, authority=authority, job_id=job_id,
                           before_model_construction=before_model_construction,
                           environment_sha256=environment_sha256)


class JobScopedGrader:
    def __init__(self, scope, *, authority, job_id, before_model_construction=False, environment_sha256=None):
        self._process = None
        self._buffer = b''
        self._sequence = 0
        self._closed = False
        if before_model_construction is not True:
            raise GraderUnavailable('must prepare before model construction')
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            raise GraderUnavailable('CUDA already initialized')
        if not isinstance(scope, dict) or set(scope) != {'payload', 'signer', 'signature'}:
            raise GraderUnavailable('signed operator capability required')
        from nacl.signing import VerifyKey
        # Snapshot mutable caller objects before authenticating the payload.
        try:
            wire = canonical(scope)
            if len(wire) > MAX_SCOPE_BYTES:
                raise GraderUnavailable('scope byte bound')
            scope = json.loads(wire)
        except (TypeError, ValueError) as e:
            raise GraderUnavailable('scope encoding') from e
        try:
            if scope['signer'] != authority:
                raise ValueError('signer')
            VerifyKey(bytes.fromhex(authority)).verify(canonical(scope['payload']),
                                                       base64.b64decode(scope['signature'], validate=True))
        except Exception as e:
            raise GraderUnavailable('operator capability signature') from e
        p = scope['payload']
        if p.get('job_id') != job_id:
            raise GraderUnavailable('cross-job capability')
        if p.get('environment_sha256') != environment_sha256:
            raise GraderUnavailable('cross-environment capability')
        identities, rows = _validate_payload(p)
        self.payload = p
        self._identities = identities
        self._tasks = rows
        py = Path(p['interpreter'])
        if not py.is_absolute() or py.parent.name != 'bin' or not py.is_file():
            raise GraderUnavailable('approved prepared interpreter')
        from .native_math_grader import isolated_argv
        env = {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent', 'CUDA_VISIBLE_DEVICES': '',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        try:
            self._process = subprocess.Popen(isolated_argv(py, Path(__file__).resolve(), ['--cpu-parent']),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=env, start_new_session=True, close_fds=True)
            self._send({'payload': p}, MAX_SCOPE_BYTES)
            ready = self._receive(min(30., p['expires_at'] - time.time()))
            if (ready.get('status') != 'ready' or ready.get('job_id') != job_id
                    or ready.get('native_runtime_binding') != p['native_runtime_binding']
                    or ready.get('isolated') != 1 or ready.get('no_site') != 1):
                raise GraderUnavailable('native parent authentication')
        except BaseException:
            self.close()
            raise

    def _send(self, value, cap=MAX_REQUEST_BYTES):
        data = canonical(value) + b'\n'
        if len(data) > cap:
            raise GraderUnavailable('request byte bound')
        try:
            self._process.stdin.write(data)
            self._process.stdin.flush()
        except OSError as e:
            raise GraderUnavailable('native parent channel closed') from e

    def _receive(self, timeout):
        deadline = time.monotonic() + max(0., timeout)
        fd = self._process.stdout.fileno()
        while b'\n' not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise GraderUnavailable('bounded native parent timeout')
            data = os.read(fd, MAX_RESULT_BYTES + 1)
            if not data:
                raise GraderUnavailable('native parent exited')
            self._buffer += data
            if len(self._buffer) > MAX_RESULT_BYTES:
                raise GraderUnavailable('native result byte bound')
        data, self._buffer = self._buffer.split(b'\n', 1)
        try:
            result = json.loads(data)
        except (ValueError, UnicodeDecodeError) as e:
            raise GraderUnavailable('native result malformed') from e
        if not isinstance(result, dict):
            raise GraderUnavailable('native result schema')
        return result

    def grade(self, *, index, task_hash, reply):
        try:
            if self._closed or self._sequence >= self.payload['max_requests']:
                raise GraderUnavailable('closed/exhausted job')
            _check_unchanged(self.payload, self._identities)
            if (type(index) is not int or str(index) not in self.payload['tasks']
                    or task_hash != self.payload['tasks'][str(index)]['task_hash']
                    or not isinstance(reply, str)):
                raise GraderUnavailable('task/prediction binding')
            self._sequence += 1
            self._send(dict(job_id=self.payload['job_id'], sequence=self._sequence,
                            index=index, task_hash=task_hash, reply=reply))
            timeout = min(self.payload['outer_timeout_seconds'] + 3,
                          self.payload['expires_at'] - time.time() + 1)
            result = self._receive(timeout)
            if (set(result) != {'job_id', 'sequence', 'returncode', 'stdout', 'stderr', 'elapsed_seconds', 'fresh_child_pid'}
                    or result['job_id'] != self.payload['job_id'] or result['sequence'] != self._sequence
                    or type(result['returncode']) is not int
                    or not isinstance(result['stdout'], str) or not isinstance(result['stderr'], str)
                    or type(result['fresh_child_pid']) is not int or result['fresh_child_pid'] <= 0
                    or type(result['elapsed_seconds']) not in (int, float)
                    or not math.isfinite(result['elapsed_seconds']) or result['elapsed_seconds'] < 0):
                raise GraderUnavailable('native result binding/schema')
            if result['returncode'] == 0:
                try:
                    score = float(result['stdout'].strip())
                except ValueError as e:
                    raise GraderUnavailable('native scalar result') from e
                if score not in (0., 1.) or not math.isfinite(score):
                    raise GraderUnavailable('native scalar result')
                result['score'] = score
            else:
                result['score'] = None
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        self._closed = True
        if self._process is not None:
            if self._process.poll() is None:
                # This group contains the dedicated CPU parent and its current child only.
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._process.wait(timeout=5)
            for stream in (self._process.stdin, self._process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _runtime_file_guard(namespace):
    stdlib = Path(namespace['sysconfig'].get_path('stdlib')).resolve()
    package_roots = namespace['package_roots']
    tree = _DirectoryGuard([str(stdlib)], excluded=('site-packages', 'dist-packages'), suffixes=('.py', '.so'))
    packages = _DirectoryGuard([str(root) for root in package_roots], suffixes=('.py', '.so'))
    paths = [p for p in stdlib.rglob('*') if p.is_file() and p.suffix in ('.py', '.so')
             and not any(v in p.parts for v in ('site-packages', 'dist-packages', '__pycache__'))]
    for root in package_roots:
        paths.extend(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py', '.so')
                     and '__pycache__' not in p.parts)
    paths.append(Path(sys.executable).resolve())
    baseline = {}
    for path in paths:
        metadata = _metadata(os.lstat(path))
        target = str(path.resolve()) if stat.S_ISLNK(metadata[5]) else str(path)
        baseline[str(path)] = (metadata, target, _identity(target))
    def unchanged():
        if not tree.valid() or not packages.valid():
            return False
        for name, (metadata, target, target_identity) in baseline.items():
            try:
                current = _metadata(os.lstat(name))
            except OSError:
                return False
            if current != metadata or (target != name and _identity(target) != target_identity):
                return False
        return True
    return unchanged


def _run_child(grade, request, gold, timeout, memory_bytes):
    began = time.monotonic()
    readfd, writefd = os.pipe()
    if len(list(Path('/proc/self/task').iterdir())) != 1 or 'torch' in sys.modules:
        os.close(readfd); os.close(writefd)
        raise GraderUnavailable('CPU parent must be single-threaded and CUDA-free')
    # Parent-death signaling prevents a native child surviving an unexpected
    # CPU-parent crash. This is Linux-only like the existing /proc guard.
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    parent_pid = os.getpid()
    pid = os.fork()
    if pid == 0:
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
            os._exit(75)
        os.close(readfd)
        for fd in (0, 1, 2):
            if fd != writefd:
                os.close(fd)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        cpu = max(1, math.ceil(timeout))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        out, err = io.StringIO(), io.StringIO()
        code = 0
        try:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(out), redirect_stderr(err):
                sys.argv = [sys.argv[0], '--json-arguments', json.dumps([gold, request['reply']])]
                grade()
        except SystemExit as e:
            code = e.code or 0
        except BaseException as e:
            code = 75
            err.write(json.dumps({'status': 'indeterminate', 'reason': type(e).__name__}))
        value = dict(returncode=code, stdout=out.getvalue(), stderr=err.getvalue())
        data = canonical(value)
        if len(data) > MAX_RESULT_BYTES // 2:
            data = canonical(dict(returncode=75, stdout='', stderr='native output exceeded bound'))
        os.write(writefd, data)
        os.close(writefd)
        os._exit(0)
    os.close(writefd)
    try:
        deadline = began + timeout
        data = b''
        status = None
        # Drain while the child runs so a bounded response cannot fill the pipe.
        while True:
            if select.select([readfd], [], [], max(0., min(.01, deadline - time.monotonic())))[0]:
                chunk = os.read(readfd, MAX_RESULT_BYTES + 1)
                data += chunk
                if len(data) > MAX_RESULT_BYTES:
                    raise GraderUnavailable('child output bound')
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                while True:
                    chunk = os.read(readfd, MAX_RESULT_BYTES + 1)
                    if not chunk:
                        break
                    data += chunk
                if len(data) > MAX_RESULT_BYTES or status != 0:
                    return dict(returncode=75, stdout='', stderr='native child exited abnormally'), pid, time.monotonic() - began
                try:
                    value = json.loads(data)
                except ValueError:
                    value = dict(returncode=75, stdout='', stderr='native child result malformed')
                return value, pid, time.monotonic() - began
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return dict(returncode=75, stdout='', stderr='native outer timeout'), pid, time.monotonic() - began
    except BaseException:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except ProcessLookupError:
            pass
        raise
    finally:
        os.close(readfd)


def _bounded_line(fd, cap, deadline, pending=b''):
    """Bound partial frames and idle lifetime without a blocking readline."""
    while b'\n' not in pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise GraderUnavailable('CPU parent frame/lifetime deadline')
        chunk = os.read(fd, cap + 1)
        if not chunk:
            if pending:
                raise GraderUnavailable('truncated CPU frame')
            return None, b''
        pending += chunk
        if len(pending) > cap:
            raise GraderUnavailable('CPU frame bound')
    line, pending = pending.split(b'\n', 1)
    return line, pending


def _parent():
    # This is a fresh isolated interpreter, never the GPU worker process.
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment
            and sys.flags.no_user_site and os.environ.get('CUDA_VISIBLE_DEVICES') == ''):
        raise GraderUnavailable('CPU import isolation')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    first, pending = _bounded_line(0, MAX_SCOPE_BYTES, time.monotonic() + 30)
    if first is None:
        raise GraderUnavailable('scope missing')
    p = json.loads(first)['payload']
    identities, rows = _validate_payload(p)
    source = Path(p['grader_path']).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != p['grader_sha256']:
        raise GraderUnavailable('native grader changed during loading')
    if source.count(SPLIT) != 1:
        raise GraderUnavailable('original native grader shape')
    prefix, tail = source.split(SPLIT, 1)
    body = SPLIT + tail
    ns = {'__name__': '__main__', '__file__': p['grader_path']}
    import importlib.metadata
    import sysconfig
    package_roots = [Path(importlib.metadata.distribution(name).locate_file(package)).resolve()
                     for name, package in (('math-verify', 'math_verify'),
                         ('latex2sympy2-extended', 'latex2sympy2_extended'),
                         ('sympy', 'sympy'), ('antlr4-python3-runtime', 'antlr4'), ('mpmath', 'mpmath'))]
    guard = _runtime_file_guard({'sysconfig': sysconfig, 'package_roots': package_roots})
    # Original prefix includes full interpreter/dependency authentication, pinned
    # imports and original parse/verify implementations, before untrusted text.
    exec(compile(prefix, p['grader_path'], 'exec'), ns)
    observed_binding = hashlib.sha256(canonical(ns['RUNTIME_LOCK'])).hexdigest()
    if p['native_runtime_binding'] != observed_binding:
        raise GraderUnavailable('native runtime binding')
    exec(compile('def original_grade():\n' + ''.join('    ' + line + '\n' for line in body.splitlines()),
                 p['grader_path'], 'exec'), ns)
    if not guard():
        raise GraderUnavailable('dependency closure changed during preparation')
    _check_unchanged(p, identities)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    print(json.dumps(dict(status='ready', job_id=p['job_id'], native_runtime_binding=observed_binding,
                          isolated=sys.flags.isolated, no_site=sys.flags.no_site)), flush=True)
    sequence = 0
    while True:
        data, pending = _bounded_line(0, MAX_REQUEST_BYTES,
                                      time.monotonic() + max(0., p['expires_at'] - time.time()), pending)
        if data is None:
            break
        request = json.loads(data)
        _check_unchanged(p, identities)
        if not guard():
            raise GraderUnavailable('approved dependency closure changed')
        sequence += 1
        if (set(request) != {'job_id', 'sequence', 'index', 'task_hash', 'reply'}
                or request['job_id'] != p['job_id'] or request['sequence'] != sequence
                or sequence > p['max_requests'] or type(request['index']) is not int
                or str(request['index']) not in p['tasks']
                or request['task_hash'] != p['tasks'][str(request['index'])]['task_hash']
                or not isinstance(request['reply'], str)):
            raise GraderUnavailable('request job/task binding')
        timeout = min(p['outer_timeout_seconds'], p['expires_at'] - time.time())
        result, pid, elapsed = _run_child(ns['original_grade'], request,
                                          rows[request['index']]['data']['answer'], timeout, p['memory_bytes'])
        # Changed assets/dependencies during grading invalidate that result too.
        _check_unchanged(p, identities)
        if not guard():
            raise GraderUnavailable('approved dependency closure changed during grading')
        result.update(job_id=p['job_id'], sequence=sequence, elapsed_seconds=elapsed, fresh_child_pid=pid)
        if len(canonical(result)) > MAX_RESULT_BYTES - 1:
            raise GraderUnavailable('result bound')
        print(json.dumps(result), flush=True)
        if sequence == p['max_requests']:
            break


if __name__ == '__main__':
    if sys.argv[1:] != ['--cpu-parent']:
        raise SystemExit('CPU parent only')
    _parent()
