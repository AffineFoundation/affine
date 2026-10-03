"""Retain archived evidence while removing completed verifier download replicas.

Planning requires original operator/worker signatures. Applying requires a fresh
full-body archive check supplied by the operator, plus exact local report/file
checks. This never deletes model caches, reports, jobs or submission archives.
"""
import hashlib
import json
import re
import stat
from pathlib import Path


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def process_directories():return Path('/proc').iterdir()


def completed_replicas(row, authority, workspaces, sources, *, now):
    from subnet.distributed_roles import authenticate, Coordinator
    if row['status']!='complete' or row['role']!='verify':raise ValueError('completed verifier job required')
    worker=row['worker']
    if worker not in workspaces:raise ValueError('approved worker required')
    job=authenticate(json.loads(row['envelope']),authority)
    manifest=authenticate(job['manifest'],authority)
    request=authenticate(json.loads(row['report_request']),worker)
    report=json.loads(row['report'])
    jobid=job['job_id']
    if (not re.fullmatch('[A-Za-z0-9_-]+',jobid) or row['id']!=jobid or
            job['role']!='verify' or manifest.get('payable') is not False or
            row['digest']!=digest(job) or row['report_digest']!=digest(report) or
            request.get('action')!='report' or request.get('job_id')!=jobid or
            request.get('token')!=row['token'] or request.get('report')!=report):
        raise ValueError('original completed job/report binding')
    source=manifest['source_bundle']['sha256']
    if source not in sources or not job['source_files'] or any(sources[source].get(name)!=sha for name,sha in job['source_files'].items()):
        raise ValueError('approved original source required')
    checker=Coordinator.__new__(Coordinator);checker.authority=authority;checker.clock=lambda:now
    checker.validate_report(report,job,manifest,row['digest'])
    workspace=Path(workspaces[worker]);attempt=row['attempt']
    if not workspace.is_absolute() or type(attempt) is not int or not 1<=attempt<=10:
        raise ValueError('worker workspace/attempt binding')
    backend=workspace/'backend' if attempt==1 else workspace/jobid/('attempt-'+str(attempt))/'backend'
    root=backend/'jobs'/jobid;result=[]
    for i,submission in enumerate(job['submissions']):
        matching=[r for r in manifest['audit_frozen_receipts'].values() if r['sha256']==submission['sha256']]
        if not matching:raise ValueError('original frozen receipt required')
        receipt=matching[0];key=receipt['frozen_key']
        if not re.fullmatch('public/'+re.escape(manifest['epoch'])+'/submissions/[0-9a-f]{64}\\.zip',key):
            raise ValueError('canonical immutable submission archive required')
        if type(receipt['size']) is not int or not 0<receipt['size']<=2_000_000_000:
            raise ValueError('original submission size required')
        result.append(dict(job_id=jobid,worker=worker,attempt=attempt,workspace=str(workspace),
            path=str(root/('submission-'+str(i)+'.zip')),report_path=str(root/'report.json'),
            report_sha256=row['report_digest'],archive_key=key,sha256=submission['sha256'],size=receipt['size']))
    return result


def remove_verified_replica(plan):
    """Delete one exact duplicate only after independent archive verification."""
    if plan.get('archive_verified') is not True:raise ValueError('fresh full archive readback required')
    path=Path(plan['path']);workspace=Path(plan['workspace']);report=Path(plan['report_path'])
    jobid=plan['job_id'];attempt=plan['attempt']
    if not re.fullmatch('[A-Za-z0-9_-]+',jobid) or type(attempt) is not int or not 1<=attempt<=10:
        raise ValueError('original job/attempt required')
    backend=workspace/'backend' if attempt==1 else workspace/jobid/('attempt-'+str(attempt))/'backend'
    expected=backend/'jobs'/jobid
    if (not workspace.is_absolute() or workspace.resolve()!=workspace or
            path.resolve()!=path or report.resolve()!=report or
            path.parent!=expected or report.parent!=expected or
            re.fullmatch('submission-[0-9]+\\.zip',path.name) is None):
        raise ValueError('exact local download replica path required')
    local_report=json.loads(report.read_text())
    if digest(local_report)!=plan['report_sha256'] or local_report.get('job_id')!=plan['job_id']:
        raise ValueError('actual original local report required')
    if not path.exists():return dict(path=str(path),removed=False,already_absent=True,bytes=0)
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size!=plan['size']:
        raise ValueError('actual local replica size/type changed')
    hashed=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):hashed.update(block)
    if hashed.hexdigest()!=plan['sha256']:raise ValueError('actual local replica hash changed')
    # A completed cache must not still be held by any scientific process.
    for process in process_directories():
        if not process.name.isdecimal():continue
        try:descriptors=list((process/'fd').iterdir())
        except FileNotFoundError:continue
        for fd in descriptors:
            try:target=fd.readlink()
            except FileNotFoundError:continue
            if target==path:raise ValueError('download replica still open by a process')
    after=path.lstat()
    if (after.st_ino,after.st_size,after.st_mtime_ns)!=(before.st_ino,before.st_size,before.st_mtime_ns):
        raise ValueError('local replica changed during verification')
    path.unlink()
    return dict(path=str(path),removed=True,already_absent=False,bytes=before.st_size,
        sha256=plan['sha256'],archive_key=plan['archive_key'],report_preserved=report.exists())
