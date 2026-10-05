"""Retire exactly two known aliases of an obsolete archived checkpoint.

Not a duplicate-current policy: principal/pending/live-request/mapped identities
always refuse. Operator verifies per-reference retirement and complete canonical
R2 readback before delegation. Unknown links, bytes or process refs refuse.
"""
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from ops.verifier_redundant_cache_lifecycle import assert_unreferenced


def _record(path,value):
    parent=Path(path).parent
    if parent.resolve()!=parent.absolute()or parent.stat().st_mode&0o077:raise ValueError('private journal parent')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w')as stream:
        os.fchmod(stream.fileno(),0o600);json.dump(value,stream);stream.flush();os.fsync(stream.fileno())
    _sync(parent)


def _sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def retire_obsolete_alias_cluster(plan,*,apply=False):
    cp=plan['checkpoint'];files=plan['files'];paths=[Path(v)for v in plan['directories']]
    if (len(paths)!=2 or paths[0]==paths[1]or not re.fullmatch('[0-9a-f]{64}',cp)
            or not isinstance(files,dict)or not 1<=len(files)<=32
            or 'config.json'not in files or not any(n.endswith('.safetensors')for n in files)
            or any(not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*',n)or type(v.get('size'))is not int or v['size']<=0
                or not re.fullmatch('[0-9a-f]{64}',v.get('sha256',''))for n,v in files.items())):
        raise ValueError('bounded exact model cluster required')
    fmap={n:r['sha256']for n,r in files.items()}
    if hashlib.sha256(json.dumps(fmap,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=cp:
        raise ValueError('canonical checkpoint filemap required')
    if (plan.get('archive_verified')is not True or plan.get('descriptor_authenticated')is not True
            or plan.get('reference_retirements_verified')is not True):raise ValueError('authenticated archive/reference policy required')
    for field,maximum,minimum in [('protected_checkpoints',32,1),('active_checkpoints',512,0),('worker_mapped_checkpoints',64,0)]:
        values=plan[field]
        if not isinstance(values,list)or not minimum<=len(values)<=maximum or any(not isinstance(v,str)or not re.fullmatch('[0-9a-f]{64}',v)for v in values):
            raise ValueError('bounded explicit identity protections required')
    if cp in set(plan['protected_checkpoints'])|set(plan['active_checkpoints'])|set(plan['worker_mapped_checkpoints']):
        raise ValueError('current pending or referenced identity protected')
    for p in paths:
        if not p.is_absolute()or p.resolve()!=p or not stat.S_ISDIR(p.lstat().st_mode)or p.name!=cp or p.parent.name not in('checkpoint','checkpoints'):
            raise ValueError('ordinary canonical explicit alias directories required')
        if {f.name for f in p.iterdir()}!=set(files):raise ValueError('exact cluster membership required')
    def idle():
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('idle GPU required')
        assert_unreferenced(paths)
    idle();snapshots={};allocated=0
    for n,v in files.items():
        left=(paths[0]/n).lstat();right=(paths[1]/n).lstat()
        if (not stat.S_ISREG(left.st_mode)or not stat.S_ISREG(right.st_mode)
                or left.st_nlink!=2 or right.st_nlink!=2 or left.st_size!=v['size']
                or(left.st_dev,left.st_ino)!=(right.st_dev,right.st_ino)):
            raise ValueError('exact known two-link cluster required')
        h=hashlib.sha256()
        with (paths[0]/n).open('rb')as stream:
            for b in iter(lambda:stream.read(1048576),b''):h.update(b)
        if h.hexdigest()!=v['sha256']:raise ValueError('cluster byte hash changed')
        snapshots[n]=(left.st_dev,left.st_ino,left.st_size,left.st_nlink,left.st_mtime_ns,left.st_ctime_ns)
        allocated+=left.st_blocks*512
    idle()
    for p in paths:
        if {f.name for f in p.iterdir()}!=set(files):raise ValueError('cluster membership changed')
        for n,before in snapshots.items():
            s=(p/n).lstat()
            if(s.st_dev,s.st_ino,s.st_size,s.st_nlink,s.st_mtime_ns,s.st_ctime_ns)!=before:raise ValueError('cluster inode changed')
    fs=os.statvfs(paths[0]);before=fs.f_bavail*fs.f_frsize
    result=dict(checkpoint=cp,review_only=not apply,estimated_reclaim_bytes=allocated,free_before=before,aliases=2,
                job_logs_and_r2_preserved=True,principal_and_active_preserved=True)
    if not apply:return result
    operation=Path(plan['operation_directory'])
    if operation.parent.resolve()!=operation.parent.absolute()or operation.parent.stat().st_mode&0o077:raise ValueError('private canonical operation parent')
    if operation.exists():raise ValueError('prior operation exists; no automatic repeats')
    operation.mkdir(mode=0o700);_sync(operation.parent)
    _record(operation/'operation-start.private.json',dict(result,at=time.time(),plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
    retired=[p.with_name(cp+'.obsolete-cluster-'+operation.name+'-'+str(i))for i,p in enumerate(paths)]
    try:
        for p,r in zip(paths,retired):
            if r.exists():raise ValueError('retired cluster path already exists')
            p.rename(r);_sync(p.parent)
        for r in retired:
            for n in files:(r/n).unlink()
            r.rmdir();_sync(r.parent)
        fs=os.statvfs(paths[0].parent);result.update(completed=True,removed_aliases=2,completed_at=time.time(),free_after=fs.f_bavail*fs.f_frsize)
        _record(operation/'operation-completed.private.json',result)
    except BaseException as error:
        _record(operation/'operation-uncertain.private.json',dict(error_type=type(error).__name__,at=time.time(),retired_paths=[str(p)for p in retired],automatic_repeat_forbidden=True))
        raise
    return result
