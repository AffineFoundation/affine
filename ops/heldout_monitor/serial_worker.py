"""Single-flight private evaluation supervisor and acknowledged cache retirement."""
import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from nacl.signing import VerifyKey


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def signed(envelope, authority):
    if set(envelope) != {'payload', 'signer', 'signature'} or envelope['signer'] != authority:
        raise ValueError('exact monitor authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),
                                             base64.b64decode(envelope['signature'], validate=True))
    return envelope['payload']


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    tmp.chmod(0o600); tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def safe_root(root):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or root.is_symlink():
        raise ValueError('canonical monitor root')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


@contextlib.contextmanager
def lease(root):
    root = safe_root(root)
    fd = os.open(root / 'serial.lease', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode): raise ValueError('regular serial lease')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally: os.close(fd)


def assignment(path, authority, root):
    envelope = json.loads(Path(path).read_bytes()); p = signed(envelope, authority)
    if p['version'] != 'serial-heldout128-assignment-v1' or p['root'] != str(root):
        raise ValueError('exact private monitor assignment')
    job_id = p['job_id']
    if Path(job_id).name != job_id or not job_id.startswith('step-'):
        raise ValueError('bounded job name')
    directory = root / 'jobs' / job_id
    if Path(path) != directory / 'assignment.json': raise ValueError('canonical assignment path')
    if p['worker_sha256'] != file_hash(__file__): raise ValueError('pinned supervisor')
    for name, expected in p['program_files'].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError('pinned scientific and hydration programs')
    evaluation = signed(json.loads((directory / 'plan.json').read_bytes()), authority)
    hydration = signed(json.loads((directory / 'read-plan.json').read_bytes()), authority)
    cp = evaluation['checkpoint']['id']
    if (evaluation['directory'] != str(directory) or evaluation['checkpoint']['path'] != str(root.parent / 'checkpoints' / cp)
            or p['checkpoint_path'] != evaluation['checkpoint']['path']
            or hydration['destination'] != evaluation['checkpoint']['path'] or hydration['checkpoint'] != cp
            or cp != p['checkpoint'] or evaluation['program_sha256'] != p['program_files']['evaluate.py']):
        raise ValueError('exact model, private directory and scientific program binding')
    return p, directory


def observe(root, directory):
    result = {'job_id': directory.name, 'at': time.time(), 'lease_busy': False}
    try:
        with lease(root): pass
    except BlockingIOError: result['lease_busy'] = True
    status = directory / 'status.json'
    if status.exists(): result.update(json.loads(status.read_bytes()))
    else: result['phase'] = 'not-dispatched'
    result['result_present'] = (directory / 'output/result.json').is_file()
    if not result['lease_busy'] and (directory / 'dispatch.json').exists():
        if result['result_present'] and result['phase'] != 'complete':
            # Presence only routes to the coordinator's complete 128-row/hash
            # validator. It does not itself label the scientific result valid.
            result['phase'] = 'complete-recovered'
        elif result['phase'] not in ('complete', 'failed'):
            result['phase'] = 'abandoned'
    return result


def dispatch(root, path, authority, python):
    p, directory = assignment(path, authority, root)
    with lease(root) as fd:
        marker = directory / 'dispatch.json'
        if marker.exists():
            return {'reused_original_dispatch': True, 'job_id': p['job_id']}
        # Intent is durable before launch. An ambiguous launch is never repeated.
        save(marker, dict(job_id=p['job_id'], assignment_sha256=file_hash(path), at=time.time()))
        save(root / 'active.json', dict(job_id=p['job_id'], checkpoint=p['checkpoint']))
        args = [python, '-I', '-B', str(Path(__file__).resolve()), 'execute', '--root', str(root),
                '--assignment', str(path), '--authority', authority, '--python', python, '--lease-fd', str(fd)]
        with (directory / 'supervisor.log').open('ab') as log:
            child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, pass_fds=(fd,))
        save(directory / 'launch.json', dict(pid=child.pid, at=time.time(), assignment_sha256=file_hash(path)))
        return {'launched': True, 'pid': child.pid, 'job_id': p['job_id']}


def execute(root, path, authority, python, fd):
    p, directory = assignment(path, authority, root)
    # This descriptor is inherited from dispatch, then by every child. Losing
    # the SSH client or supervisor cannot allow a second GPU process.
    if (os.fstat(fd).st_ino, os.fstat(fd).st_dev) != ((root / 'serial.lease').stat().st_ino, (root / 'serial.lease').stat().st_dev):
        raise ValueError('inherited single-flight lease')
    phase = 'hydrating'
    try:
        save(directory / 'status.json', dict(phase=phase, at=time.time(), supervisor_pid=os.getpid()))
        hydrate = [python, '-I', '-B', str(directory / 'hydrate.py'), '--plan', str(directory / 'read-plan.json'),
                   '--plan-sha256', file_hash(directory / 'read-plan.json'), '--operator', authority,
                   '--role', 'heldout128-evaluate', '--retained-UUID', p['retained_UUID'],
                   '--destination', p['checkpoint_path'], '--output', str(directory / 'hydration-receipt.json')]
        if not (directory / 'hydration-receipt.json').exists():
            with (directory / 'hydration.log').open('ab') as log:
                subprocess.run(hydrate, stdout=log, stderr=log, check=True, pass_fds=(fd,), timeout=3550)
        phase = 'evaluating'
        save(directory / 'status.json', dict(phase=phase, at=time.time(), supervisor_pid=os.getpid()))
        with (directory / 'worker.log').open('ab') as log:
            subprocess.run([python, '-I', '-B', str(directory / 'evaluate.py'), '--plan', str(directory / 'plan.json')],
                           stdout=log, stderr=log, check=True, pass_fds=(fd,), timeout=7200)
        if not (directory / 'output/result.json').is_file(): raise ValueError('complete native evaluation result')
        save(directory / 'status.json', dict(phase='complete', at=time.time(), supervisor_pid=os.getpid()))
    except Exception as error:
        save(directory / 'status.json', dict(phase='failed', failed_phase=phase, at=time.time(),
                                            error_type=type(error).__name__, supervisor_pid=os.getpid()))
        raise
    finally: os.close(fd)


def snapshot(path):
    s = Path(path).lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid != os.geteuid():
        raise ValueError('owned single-link regular cache file')
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_uid', 'st_nlink')}


def unused_model(path):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal(): continue
        try:
            if str(path) + '/' in (proc / 'maps').read_text(errors='replace'):
                raise ValueError('model is still memory mapped')
            for fd in (proc / 'fd').iterdir():
                try: target = os.readlink(fd)
                except FileNotFoundError: continue
                if target.startswith(str(path) + '/'):
                    raise ValueError('model file is still open')
        except (FileNotFoundError, ProcessLookupError): continue


def retire(root, grant_path, authority):
    grant = signed(json.loads(Path(grant_path).read_bytes()), authority)
    if grant['version'] != 'archived-heldout128-model-retirement-v1' or grant['root'] != str(root):
        raise ValueError('scoped model retirement')
    plan = signed(grant['evaluation_plan'], authority)
    archive = signed(grant['archive'], authority)
    cp = plan['checkpoint']['id']; path = Path(plan['checkpoint']['path'])
    if (archive['full_readback_verified'] is not True or archive['checkpoint'] != cp or archive['task_count'] != 128
            or archive['plan_sha256'] != digest(canonical(grant['evaluation_plan']))
            or path != root.parent / 'checkpoints' / cp or path.resolve() != path
            or grant['checkpoint'] != cp):
        raise ValueError('durably archived exact completed model; base is protected')
    directory = Path(plan['directory'])
    if directory.resolve() != directory or not directory.is_relative_to(root.parent):
        raise ValueError('owned evaluation lease directory')
    with lease(root):
        lock = os.open(directory / 'evaluation.lease', os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active_path = root / 'active.json'
            if active_path.exists():
                active = json.loads(active_path.read_bytes())
                if active['checkpoint'] == cp and active['job_id'] != directory.name:
                    raise ValueError('another current job owns this checkpoint')
            record = root / 'retirements' / (cp + '.json')
            binding = digest(canonical(grant))
            if record.exists():
                state = json.loads(record.read_bytes())
                if state['grant_sha256'] != binding: raise ValueError('same original retirement authorization')
                if state['phase'] == 'complete': return state
            else:
                if not path.is_dir() or set(x.name for x in path.iterdir()) != set(plan['checkpoint']['files']):
                    raise ValueError('exact owned checkpoint inventory')
                stats = {}
                for name, expected in plan['checkpoint']['files'].items():
                    before = snapshot(path / name)
                    if file_hash(path / name) != expected or snapshot(path / name) != before:
                        raise ValueError('unchanged complete checkpoint hash before retirement')
                    stats[name] = before
                state = dict(phase='prepared', grant_sha256=binding, checkpoint=cp, files=stats,
                             bytes=sum(x['st_size'] for x in stats.values()), at=time.time())
                save(record, state)
            if path.exists():
                if path.is_symlink() or any(x.name not in state['files'] for x in path.iterdir()):
                    raise ValueError('unexpected retirement member')
                unused_model(path)
                for name, expected in state['files'].items():
                    member = path / name
                    if member.exists() and snapshot(member) != expected:
                        raise ValueError('retirement member changed after full hash')
                for name in state['files']:
                    member = path / name
                    if member.exists(): member.unlink()
                path.rmdir()
            state.update(phase='complete', completed_at=time.time(), durable_raw_evidence_retained=True)
            save(record, state)
            return state
        finally: os.close(lock)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('dispatch', 'execute', 'observe', 'retire', 'retire-failed'))
    parser.add_argument('--root', required=True); parser.add_argument('--assignment')
    parser.add_argument('--authority', required=True); parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--lease-fd', type=int); parser.add_argument('--grant')
    args = parser.parse_args(); root = safe_root(args.root)
    if args.action == 'dispatch': result = dispatch(root, Path(args.assignment), args.authority, args.python)
    elif args.action == 'execute': execute(root, Path(args.assignment), args.authority, args.python, args.lease_fd); return
    elif args.action == 'retire': result = retire(root, Path(args.grant), args.authority)
    elif args.action == 'retire-failed':
        import importlib.util
        grant = signed(json.loads(Path(args.grant).read_bytes()), args.authority)
        helper = root / 'failed_cache_retirement.py'
        if file_hash(helper) != grant['cleanup_helper_sha256']: raise ValueError('pinned failed-cache cleanup')
        sys.modules.setdefault('serial_worker', sys.modules[__name__])
        spec = importlib.util.spec_from_file_location('failed_cache_retirement', helper)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        result = module.retire_failed(root, Path(args.grant), args.authority)
    else:
        _, directory = assignment(Path(args.assignment), args.authority, root)
        result = observe(root, directory)
    print(json.dumps(result))


if __name__ == '__main__': main()
