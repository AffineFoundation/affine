"""Retire archived replicas belonging to an actually completed training job.

The operator independently authenticates and reads back the archive before
issuing a plan. Reports, job requests, runner records and current checkpoints
are retained. A report appearing before child.wait() completes is insufficient.
"""
import base64
import hashlib
import json
import re
import secrets
import stat
import subprocess
from pathlib import Path


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def hash_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def processes():return Path('/proc').iterdir()


def gpu_processes():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()


def completed_job(plan):
    from nacl.signing import VerifyKey
    workspace=Path(plan['workspace']);jobid=plan['job_id'];authority=plan['authority']
    if (not workspace.is_absolute() or workspace.resolve()!=workspace or
            not re.fullmatch('[A-Za-z0-9_-]+',jobid) or not re.fullmatch('[0-9a-f]{64}',authority)):
        raise ValueError('canonical original training workspace/job/authority')
    paths={'job':workspace/(jobid+'.json'),
           'report':workspace/'jobs'/jobid/'report.json',
           'terminal':workspace/'runner-status'/(jobid+'.json')}
    values={}
    for name,path in paths.items():
        if path.resolve()!=path or not path.is_file() or path.is_symlink() or hash_file(path)!=plan[name+'_sha256']:
            raise ValueError('original training evidence bytes changed')
        values[name]=json.loads(path.read_text())
    envelope=values['job']
    if envelope['signer']!=authority:raise ValueError('original training signer')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True))
    job=envelope['payload'];report=values['report'];terminal=values['terminal']
    if (job.get('job_id')!=jobid or job.get('role')!='train' or
            report.get('job_id')!=jobid or report.get('role')!='train' or report.get('success') is not True or
            report.get('operator')!=authority or report.get('job_sha256')!=digest(job) or
            terminal.get('job_id')!=jobid or terminal.get('phase')!='complete' or terminal.get('exit_code')!=0):
        raise ValueError('actual completed original training job required')
    manifest=job['manifest']
    if manifest['signer']!=authority:raise ValueError('original manifest signer')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(manifest['payload']),base64.b64decode(manifest['signature'],validate=True))
    if report.get('epoch')!=manifest['payload']['epoch'] or report.get('checkpoint')!=manifest['payload']['checkpoint']['id']:
        raise ValueError('training report checkpoint/epoch binding')
    for name in ('runner_pid','child_pid'):
        pid=terminal.get(name);ticks=terminal.get(name+'_ticks')
        if type(pid) is not int or not isinstance(ticks,str):raise ValueError('actual runner/child wait identity')
        path=Path('/proc',str(pid),'stat')
        try:fields=path.read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:continue
        if fields[19]==ticks and fields[0]!='Z':raise ValueError('original training process still alive')
    return workspace,job,report


def unreferenced(paths):
    targets=set(paths)
    if gpu_processes():raise ValueError('training retention requires idle GPU')
    for process in processes():
        if not process.name.isdecimal():continue
        try:descriptors=list((process/'fd').iterdir())
        except FileNotFoundError:continue
        for fd in descriptors:
            try:target=fd.readlink()
            except FileNotFoundError:continue
            if target in targets:raise ValueError('training replica still open')
        try:maps=(process/'maps').read_text()
        except FileNotFoundError:continue
        for line in maps.splitlines():
            row=line.split(None,5)
            if len(row)==6 and Path(row[5].removesuffix(' (deleted)')) in targets:
                raise ValueError('training replica still memory mapped')


def remove_training_replica(plan):
    workspace,job,report=completed_job(plan)
    if plan.get('archive_verified') is not True or plan.get('archive_authenticated') is not True:
        raise ValueError('authenticated full archive readback required')
    root=workspace/'jobs'/job['job_id'];kind=plan['kind'];target=Path(plan['directory'])
    if target.resolve()!=target or target.is_symlink():raise ValueError('canonical replica path')
    files=plan['files']
    if kind=='submission':
        number=plan['submission_index']
        if type(number) is not int or not 0<=number<len(job['submissions']):raise ValueError('original submission position')
        name='submission-'+str(number)+'.zip'
        if target!=root or set(files)!={name} or files[name]['sha256']!=job['submissions'][number]['sha256']:
            raise ValueError('original training download binding')
    elif kind=='checkpoint-export':
        step=plan['step'];checkpoint=plan['checkpoint'];protected=plan['protected_checkpoints']
        covered=job.get('training_policy')=='bf16-full-adamw-covered-fixed-reference-v3'
        expected_name=('checkpoint-covered-final' if covered else 'checkpoint-step-'+str(step))
        if covered and (step!=job['steps'] or
                report.get('training',{}).get('training_policy')!=job['training_policy'] or
                report.get('new_checkpoint',{}).get('path')!=str(target)):
            raise ValueError('original covered final checkpoint binding')
        if (type(step) is not int or not 1<=step<=job['steps'] or
                target!=root/expected_name or
                not re.fullmatch('[0-9a-f]{64}',checkpoint) or
                not isinstance(protected,list) or not protected or
                any(not re.fullmatch('[0-9a-f]{64}',c) for c in protected) or checkpoint in protected or
                digest({n:v['sha256'] for n,v in files.items()})!=checkpoint or
                'config.json' not in files or not any(n.endswith('.safetensors') for n in files)):
            raise ValueError('archived unprotected original checkpoint export')
        if target.exists() and {p.name for p in target.iterdir()}!=set(files):raise ValueError('exact export membership')
        if step==job['steps'] and report.get('new_checkpoint',{}).get('id')!=checkpoint:
            raise ValueError('original final checkpoint report binding')
    else:raise ValueError('scoped training replica kind')
    if not isinstance(files,dict) or not 1<=len(files)<=32:raise ValueError('bounded exact object inventory')
    before={}
    for name,expected in files.items():
        if (not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*',name) or
                not re.fullmatch('[0-9a-f]{64}',expected.get('sha256','')) or
                type(expected.get('size')) is not int or not 0<expected['size']<=
                (32 if kind=='checkpoint-export' and name.endswith('.safetensors') else 5)*1024**3):
            raise ValueError('bounded archived object metadata')
        path=target/name
        if not path.exists():
            if kind=='submission':return dict(removed=False,already_absent=True,bytes=0)
            if not target.exists():return dict(removed=False,already_absent=True,bytes=0)
            raise ValueError('partial export preserved')
        original=path.lstat()
        if not stat.S_ISREG(original.st_mode) or original.st_nlink!=1 or original.st_size!=expected['size'] or hash_file(path)!=expected['sha256']:
            raise ValueError('exact regular local replica bytes required')
        before[path]=original
    unreferenced(before)
    completed_job(plan)
    for path,previous in before.items():
        current=path.lstat()
        if (current.st_ino,current.st_size,current.st_mtime_ns,current.st_nlink)!=(previous.st_ino,previous.st_size,previous.st_mtime_ns,1):
            raise ValueError('training replica changed during verification')
    if kind=='checkpoint-export':
        if {p.name for p in target.iterdir()}!=set(files):raise ValueError('export membership changed')
        retired=target.with_name(target.name+'.retired-'+secrets.token_hex(8))
        target.rename(retired)
        for name in files:(retired/name).unlink()
        retired.rmdir()
    else:
        for path in before:path.unlink()
    return dict(removed=True,already_absent=False,bytes=sum(s.st_size for s in before.values()),
                original_jobs_and_reports_preserved=True,kind=kind)
