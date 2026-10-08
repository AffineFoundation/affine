"""Operator-owned, single-checkpoint verifier process; signed science stays pinned.

No training or miner jobs, no concurrent requests, no automatic replay on EOF.
Per-job source, signature, capability, sampling and outcome checks stay in execute.
"""
import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


def fingerprint(root, files):
    root = Path(root)
    if root.is_symlink() or root.absolute() != root.resolve():
        raise ValueError('resident checkpoint path changed')
    if {p.name for p in root.iterdir()} != set(files):
        raise ValueError('resident checkpoint inventory changed')
    values = []
    for name in sorted(files):
        if Path(name).name != name:
            raise ValueError('checkpoint member path')
        s = (root / name).lstat()
        if not stat.S_ISREG(s.st_mode):
            raise ValueError('resident checkpoint member type changed')
        values.append((name, s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
    return tuple(values)


def binding(job, source):
    if job.get('role') != 'verify':
        raise ValueError('resident backend is verifier-only')
    m = job['manifest']['payload']
    return json.dumps(dict(operator_version='resident-verifier-checkpoint-v1',
        source_archive=m['source_bundle']['sha256'],source=str(Path(source).resolve()), files=job['source_files'],
        versions=job['runtime_versions'], checkpoint=m['checkpoint']['id'],
        checkpoint_files=m['checkpoint']['files'], revision=m['model_runtime_revision'],
        backend_profile=m.get('backend_profile')), sort_keys=True, separators=(',', ':'))


class ResidentRuntime:
    """One immutable GPU model; each job gets fresh environment/binding state."""
    def __init__(self, backend, source, key):
        self.backend, self.source, self.key = backend, Path(source), key
        self.path = self.files = self.snapshot = self.runtime = None
        self.original_checkpoint = backend.checkpoint
        self.original_loader = backend.install_source_loader
        self.loader_installed = False
        self.checkpoint_authentications = 0
        self.model_loads = 0

    def install_loader(self, root, additional_files=()):
        if not self.loader_installed:
            self.original_loader(root, additional_files)
            self.loader_installed = True
            return
        # execute has already authenticated all source bytes for this request.
        # Reuse imports only from this exact immutable science tree, never a
        # different historical source or a caller-supplied runtime factory.
        for name, module in tuple(sys.modules.items()):
            if not name.startswith('subnet.') or not getattr(module, '__file__', None):
                continue
            location = Path(module.__file__)
            expected = self.source.joinpath(*name.split('.')).with_suffix('.py')
            if location.resolve() != expected.resolve():
                raise ValueError('resident imported source origin changed')

    def checkpoint(self, manifest, workspace, cache=None):
        cp = manifest['checkpoint']
        if self.path is None:
            self.path = self.original_checkpoint(manifest, workspace, cache)
            self.files = dict(cp['files'])
            self.snapshot = fingerprint(self.path, self.files)
            self.checkpoint_authentications += 1
        elif cp['files'] != self.files or fingerprint(self.path, self.files) != self.snapshot:
            raise ValueError('resident authenticated checkpoint changed')
        return self.path

    def factory(self, checkpoint, files, environment, harness):
        if fingerprint(checkpoint, files) != self.snapshot:
            raise ValueError('resident checkpoint changed before model access')
        if self.runtime is None:
            from subnet.gpu_runtime import GPURuntime
            revision = json.loads(self.key)['revision']
            self.runtime = GPURuntime(checkpoint, files, environment, harness,
                                      runtime_revision=revision)
            self.model_loads += 1
            if fingerprint(checkpoint, files) != self.snapshot:
                raise ValueError('checkpoint changed during construction')
        return self.runtime.for_environment(environment, harness)

    def execute(self, envelope, authority, workspace, cache=None):
        job = self.backend.signed(envelope, authority)
        if binding(job, self.source) != self.key:
            raise ValueError('resident execution binding changed')
        self.backend.checkpoint = self.checkpoint
        self.backend.install_source_loader = self.install_loader
        report=self.backend.execute(envelope, authority, workspace, cache,
                                    runtime_factory=self.factory)
        # Local operator telemetry, separate from the unchanged scientific
        # report. Two successful jobs can independently establish load-once.
        witness=Path(workspace)/'jobs'/job['job_id']/'resident-lifecycle.json'
        if witness.parent.is_dir():
            with witness.open('x')as stream:
                json.dump(dict(version='resident-verifier-lifecycle-v1',pid=os.getpid(),
                    job_id=job['job_id'],binding_sha256=hashlib.sha256(self.key.encode()).hexdigest(),
                    checkpoint_authentications=self.checkpoint_authentications,
                    model_loads=self.model_loads,model_object_id=id(getattr(self.runtime,'model',self.runtime))),stream,sort_keys=True)
            os.chmod(witness,0o600)
        return report


class ResidentClient:
    def __init__(self, helper=None, idle_seconds=900):
        if type(idle_seconds) is not int or not 1 <= idle_seconds <= 3600:
            raise ValueError('bounded resident idle lifetime')
        self.helper = Path(helper or __file__).resolve()
        self.idle_seconds = idle_seconds
        self.process = None
        self.key = None
        self.leases = ExitStack()
        self.fds = {}
        self.last_used = time.monotonic()
        self.error_log = None

    def prepare(self, job, source):
        key = binding(job, source)
        if self.key != key:
            self.close()
            self.key = key

    @contextmanager
    def lease_checkpoint(self, lifecycle, checkpoint):
        k = (str(lifecycle.root), checkpoint)
        if k not in self.fds:
            self.fds[k] = self.leases.enter_context(lifecycle.lease_checkpoint(checkpoint))
        yield self.fds[k]

    def idle(self):
        if time.monotonic() - self.last_used >= self.idle_seconds:
            self.close()

    def close(self):
        if self.process is not None:
            # Only between requests. EOF naturally retires the same child.
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise RuntimeError('resident backend has not retired; refuse second owner')
            self.process.stdout.close()
            self.process = None
        if self.error_log is not None:
            self.error_log.close()
            self.error_log = None
        self.leases.close()
        self.leases = ExitStack()
        self.fds = {}
        self.key = None

    def run(self, command, *, stdout, stderr, env, pass_fds, cwd):
        # Preserve original command selectors; all source/capacity validation
        # still happens in the child. Never retry an uncertain original job.
        args = list(command)
        if '-m' in args:
            i = args.index('-m')
            if args[i+1] != 'subnet.backend_jobs':
                raise ValueError('resident original backend selector')
            jobpath = args[i+2]
        else:
            jobpath = args[3] if args[1] == '-B' else None
        if not jobpath:
            raise ValueError('resident job selector')
        def value(flag):
            return args[args.index(flag)+1] if flag in args else None
        authority = value('--authority')
        request = dict(job=jobpath, authority=authority, workspace=value('--workspace'),
                       cache=value('--checkpoint-cache'), log=stdout.name,
                       capacity_policy=value('--capacity-policy'))
        if self.process is None:
            self.error_log = open(str(stdout.name)+'.resident-stderr', 'xb')
            os.chmod(self.error_log.name, 0o600)
            self.process = subprocess.Popen([args[0], '-I', '-B', str(self.helper),
                '--serve', '--source', str(cwd), '--key', self.key],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_log,
                env=env, pass_fds=tuple(self.fds.values()), cwd=cwd, text=True)
        self.process.stdin.write(json.dumps(request, separators=(',', ':'))+'\n')
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError('resident original attempt lost; do not replay')
        result = json.loads(response)
        if set(result) != {'returncode'} or type(result['returncode']) is not int:
            raise ValueError('resident original terminal protocol')
        self.last_used = time.monotonic()
        if result['returncode']:
            self.close()
        return subprocess.CompletedProcess(command, result['returncode'])


def serve(source, key):
    source = Path(source).resolve(strict=True)
    sys.path.insert(0, str(source))
    from subnet import backend_jobs
    resident = ResidentRuntime(backend_jobs, source, key)
    for line in sys.stdin:
        request = json.loads(line)
        rc = 1
        with open(request['log'], 'a') as log, redirect_stdout(log), redirect_stderr(log):
            try:
                envelope = backend_jobs.load_job_envelope(request['job'], request['authority'])
                if request['capacity_policy']:
                    # Isolated admission imports no GPU modules and preserves
                    # exact job/checkpoint byte budgets before model use.
                    import importlib.util
                    p = Path(__file__).with_name('capacity_bounded_verifier_backend.py')
                    spec = importlib.util.spec_from_file_location('resident_capacity', p)
                    cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
                    policy = json.loads(Path(request['capacity_policy']).read_bytes())
                    job = backend_jobs.signed(envelope, request['authority'])
                    helper = Path(__file__).with_name('verifier_capacity_admission.py')
                    admission = cap.isolated_transport_admission(job, policy,
                        request['authority'], source, helper,validate_runtime=True)
                    original_get = cap.bind_transport(backend_jobs, job,
                        request['authority'], request['workspace'], policy, admission)
                else:
                    original_get = None
                try:
                    resident.execute(envelope, request['authority'], request['workspace'], request['cache'])
                finally:
                    if original_get is not None:
                        backend_jobs.get_object = original_get
                rc = 0
            except Exception:
                import traceback
                traceback.print_exc()
        sys.stdout.write(json.dumps(dict(returncode=rc))+'\n');sys.stdout.flush()
        if rc:
            return
    resident.runtime = None
    gc.collect()


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--serve', action='store_true')
    p.add_argument('--source', required=True);p.add_argument('--key', required=True)
    a = p.parse_args();serve(a.source, a.key)
