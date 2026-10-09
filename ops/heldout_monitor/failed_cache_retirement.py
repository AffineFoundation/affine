"""Retire exact failed private-evaluation download bytes after durable evidence."""
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import time

from serial_worker import canonical, digest, file_hash, lease, save, signed, snapshot, unused_model


@contextmanager
def file_lease(path, *, optional=False, create=False):
    path = Path(path)
    if optional and not path.exists():
        yield
        return
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('regular owned lifecycle lease')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def retire_failed(root, grant_path, authority):
    root = Path(root)
    grant = signed(json.loads(Path(grant_path).read_bytes()), authority)
    if grant['version'] != 'failed-heldout128-cache-retirement-v1' or grant['root'] != str(root):
        raise ValueError('exact failed cache retirement scope')
    assignment = signed(grant['assignment'], authority)
    read = signed(grant['read_plan'], authority)
    archive = signed(grant['failure_archive'], authority)
    job, cp = assignment['job_id'], assignment['checkpoint']
    plan_hash = digest(canonical(grant['read_plan']))
    if (assignment['version'] != 'serial-heldout128-assignment-v1' or assignment['root'] != str(root)
            or Path(job).name != job or not job.startswith('step-') or not re.fullmatch('[0-9a-f]{64}', cp)
            or assignment['program_files']['read-plan.json'] != plan_hash
            or read['kind'] != 'immutable-checkpoint-read-hydration-v1' or read['checkpoint'] != cp
            or read['role'] != 'heldout128-evaluate' or read['retained_UUID'] != assignment['retained_UUID']):
        raise ValueError('original assignment and read plan binding')
    if (archive['version'] != 'serial-heldout128-failed-attempt-v1'
            or archive['job_id'] != job or archive['checkpoint'] != cp
            or archive['assignment_sha256'] != digest(canonical(grant['assignment']))
            or archive['read_plan_sha256'] != plan_hash
            or archive['phase'] not in ('failed', 'abandoned')
            or archive['full_readback_verified'] is not True or archive['scientific_success'] is not False
            or not isinstance(archive['archive'], dict) or not archive['archive']):
        raise ValueError('durably archived exact failed attempt')
    if type(grant['remove_complete_model']) is not bool:
        raise ValueError('explicit completed-model retirement choice')
    destination = root.parent / 'checkpoints' / cp
    if (root.resolve() != root or read['destination'] != str(destination)
            or assignment['checkpoint_path'] != str(destination) or destination.resolve() != destination):
        raise ValueError('canonical owned model path; base protected')
    descriptor = signed(read['checkpoint_descriptor'], authority)
    files, objects = descriptor['files'], read['objects']
    if (descriptor['id'] != cp or digest(canonical(files)) != cp or set(objects) != set(files)
            or digest(canonical(read['checkpoint_descriptor'])) != read['checkpoint_descriptor_sha256']):
        raise ValueError('exact original checkpoint inventory')
    for name, expected in files.items():
        if (not re.fullmatch('[A-Za-z0-9_.-]+', name) or name.startswith('.')
                or not re.fullmatch('[0-9a-f]{64}', expected)
                or objects[name]['sha256'] != expected or type(objects[name]['bytes']) is not int
                or not 0 < objects[name]['bytes'] <= 20 * 1024**3):
            raise ValueError('bounded exact original object inventory')
    if not 1 <= len(files) <= 32 or sum(x['bytes'] for x in objects.values()) > 64 * 1024**3:
        raise ValueError('bounded complete inventory')
    directory = root / 'jobs' / job
    if directory.resolve() != directory:
        raise ValueError('canonical original job directory')
    stage = destination.parent / ('.' + cp + '.hydrate-' + plan_hash[:16])
    binding = dict(plan_sha256=plan_hash, checkpoint=cp, role=read['role'],
                   retained_UUID=read['retained_UUID'], descriptor_sha256=read['checkpoint_descriptor_sha256'])
    record = root / 'failed-retirements' / (job + '.json')
    grant_hash = digest(canonical(grant))
    with ExitStack() as stack:
        stack.enter_context(lease(root))
        stack.enter_context(file_lease(directory / 'evaluation.lease', optional=True))
        stack.enter_context(file_lease(destination.parent / ('.' + cp + '.hydrate-lock'), create=True))
        active_path = root / 'active.json'
        if active_path.exists():
            active = json.loads(active_path.read_bytes())
            if active['checkpoint'] == cp and active['job_id'] != job:
                raise ValueError('another current job owns checkpoint')
        if stage.exists():
            if stage.is_symlink() or stage.resolve() != stage or not stage.is_dir():
                raise ValueError('canonical original staging path')
            marker = stage / 'binding.json'
            snapshot(marker)
            if json.loads(marker.read_bytes()) != binding:
                raise ValueError('exact original staging binding')
        if record.exists():
            state = json.loads(record.read_bytes())
            if state['grant_sha256'] != grant_hash:
                raise ValueError('original unchanged retirement authorization')
            if state['phase'] == 'complete':
                return state
        else:
            targets = {}
            work = stage / 'objects'
            if work.exists():
                if not work.is_dir() or work.is_symlink():
                    raise ValueError('regular original objects directory')
                for member in work.iterdir():
                    partial = member.name.endswith('.partial')
                    name = member.name[:-8] if partial else member.name
                    if name not in files:
                        raise ValueError('unexpected staged member')
                    before = snapshot(member)
                    if before['st_size'] > objects[name]['bytes']:
                        raise ValueError('staged size exceeds bound')
                    checksum = file_hash(member)
                    if not partial and (before['st_size'] != objects[name]['bytes'] or checksum != files[name]):
                        raise ValueError('completed staged member full hash')
                    if snapshot(member) != before:
                        raise ValueError('unchanged staged file')
                    targets[str(member)] = dict(stat=before, sha256=checksum)
            if grant['remove_complete_model'] and destination.exists():
                if not destination.is_dir() or destination.is_symlink() or set(x.name for x in destination.iterdir()) != set(files):
                    raise ValueError('exact full model inventory')
                for name, expected in files.items():
                    member = destination / name; before = snapshot(member)
                    if before['st_size'] != objects[name]['bytes'] or file_hash(member) != expected or snapshot(member) != before:
                        raise ValueError('unchanged complete model full hash')
                    targets[str(member)] = dict(stat=before, sha256=expected)
            state = dict(phase='prepared', grant_sha256=grant_hash, checkpoint=cp, job_id=job,
                         files=targets, bytes=sum(x['stat']['st_size'] for x in targets.values()), at=time.time())
            save(record, state)
        roots = [stage / 'objects'] + ([destination] if grant['remove_complete_model'] else [])
        for parent in roots:
            if parent.exists():
                if parent.is_symlink() or not parent.is_dir() or any(str(x) not in state['files'] for x in parent.iterdir()):
                    raise ValueError('unexpected member during retirement recovery')
                unused_model(parent)
        for filename, expected in state['files'].items():
            path = Path(filename)
            if path.exists() and snapshot(path) != expected['stat']:
                raise ValueError('retirement member changed since authenticated snapshot')
        for filename in state['files']:
            path = Path(filename)
            if path.exists():
                path.unlink()
        for parent in roots:
            if parent.exists():
                parent.rmdir()
        state.update(phase='complete', completed_at=time.time(), original_attempt_evidence_retained=True)
        save(record, state)
        return state
