"""Opt-in root-local physical-node scheduling. Expiry never proves process death."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

VERSION='shared-reader-evaluator-reservation-v1'
class SharedNodeBusy(RuntimeError):pass

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class SharedNodeReservation:
    def __init__(self,config,*,provider_uuid):
        if set(config)!={'version','directory','provider_UUID'} or config['version']!=VERSION:
            raise ValueError('shared node reservation configuration')
        if str(uuid.UUID(config['provider_UUID']))!=provider_uuid:
            raise ValueError('shared reservation physical provider binding')
        self.provider=provider_uuid;self.root=Path(config['directory'])
        if not self.root.is_absolute():raise ValueError('absolute root-local reservation directory')
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.is_symlink() or self.root.stat().st_uid!=os.getuid() or self.root.stat().st_mode&0o077:
            raise ValueError('private operator reservation directory required')
        self.path=self.root/(provider_uuid+'.json');self.lock=self.root/(provider_uuid+'.lock')
    @contextlib.contextmanager
    def transaction(self):
        fd=os.open(self.lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'r+')as f:
            if os.fstat(f.fileno()).st_uid!=os.getuid()or os.fstat(f.fileno()).st_mode&0o077:raise ValueError('private reservation lock')
            fcntl.flock(f,fcntl.LOCK_EX)
            if self.path.is_symlink():raise ValueError('symlink reservation journal')
            if self.path.exists():
                if self.path.is_symlink()or self.path.stat().st_uid!=os.getuid()or self.path.stat().st_mode&0o077:raise ValueError('private reservation journal')
                v=json.loads(self.path.read_bytes())
                if v.get('version')!=VERSION or v.get('provider_UUID')!=self.provider:raise ValueError('reservation journal binding')
            else:v=dict(version=VERSION,provider_UUID=self.provider,active=None,pending_readers=[],history=[])
            yield v
            tmp=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
            fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb')as out:
                out.write(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode());out.flush();os.fsync(out.fileno())
            os.replace(tmp,self.path)
            d=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(d)
            finally:os.close(d)
    def claim(self,role,request):
        if role not in ('reader','evaluate')or not re.fullmatch('[0-9a-f]{64}',request):raise ValueError('reservation owner identity')
        # Commit priority intent even when a current evaluator cannot be preempted.
        if role=='reader':
            with self.transaction()as v:
                for old in v['history']:
                    if old['request']==request and old['role']==role:return dict(old)
                if request not in v['pending_readers']:v['pending_readers'].append(request)
        with self.transaction()as v:
            a=v['active']
            if not a:
                for old in v['history']:
                    if old['request']==request and old['role']==role:return dict(old)
            if a:
                if (a['role'],a['request'])!=(role,request):raise SharedNodeBusy('physical node retained by original job')
                return dict(a)
            if v['pending_readers'] and (role!='reader'or request!=v['pending_readers'][0]):
                raise SharedNodeBusy('durable trainer-state reader has priority')
            v['active']=dict(role=role,request=request,claimed_at=time.time(),binding=None,handle=None)
            return dict(v['active'])
    def bind(self,request,binding,handle=None):
        if not re.fullmatch('[0-9a-f]{64}',binding):raise ValueError('original signed request SHA')
        if handle is not None:
            if set(handle)!={'pid','ticks'}or type(handle['pid'])is not int or handle['pid']<=0 or not str(handle['ticks']).isdigit():raise ValueError('original child PID/ticks')
        with self.transaction()as v:
            a=v['active']
            if not a or a['request']!=request:
                old=next((h for h in v['history']if h['request']==request),None)
                if old and old['binding']==binding and (handle is None or old['handle']==handle):return
                raise SharedNodeBusy('original reservation absent')
            if a['binding']not in (None,binding)or (handle is not None and a['handle']not in (None,handle)):raise ValueError('original reservation binding changed')
            a['binding']=binding
            if handle is not None:a['handle']=handle
    def terminal(self,request,binding,handle,evidence):
        # Absence, TTL expiry, PID reuse, and failed SSH observations never release.
        if (evidence.get('actual_child_wait_completed')is not True or type(evidence.get('exit_code'))is not int
                or type(evidence.get('finished_at'))not in (int,float) or not math.isfinite(evidence['finished_at'])):
            raise ValueError('actual original wait terminal required')
        with self.transaction()as v:
            a=v['active']
            old=next((h for h in v['history']if h['request']==request),None)
            if old and old['binding']==binding and old['handle']==handle and old['terminal']==evidence:return
            if not a or a['request']!=request or a['binding']!=binding or a['handle']!=handle:
                raise ValueError('terminal original request/PID/ticks changed')
            if evidence['finished_at']<a['claimed_at']:raise ValueError('terminal predates claim')
            v['history'].append(dict(a,terminal=dict(evidence),released_at=time.time()))
            v['active']=None
            if request in v['pending_readers']:v['pending_readers'].remove(request)
    @contextlib.contextmanager
    def executing(self,request):
        if not re.fullmatch('[0-9a-f]{64}',request):raise ValueError('dispatch reservation identity')
        fd=os.open(self.root/(self.provider+'-'+request+'.dispatch.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'r+')as f:
            if os.fstat(f.fileno()).st_uid!=os.getuid()or os.fstat(f.fileno()).st_mode&0o077:raise ValueError('private dispatch lock')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SharedNodeBusy('same original dispatch already observed by another process')
            yield

    def blocked(self):
        with self.transaction()as v:return bool(v['active']or v['pending_readers'])

def reserved_evaluation_run(lease,jobs,label,manifest,cache,fields):
    request=identity(dict(label=label,manifest=manifest,fields=fields))
    with lease.executing(request):
        return _reserved_evaluation_run(lease,jobs,label,manifest,cache,fields)

def _reserved_evaluation_run(lease,jobs,label,manifest,cache,fields):
    """Same RemoteJobs request on recovery; only its durable wait marker releases."""
    import shlex
    request=identity(dict(label=label,manifest=manifest,fields=fields))
    lease.claim('evaluate',request)
    try:return jobs.run(label,'evaluate',manifest,cache,**fields)
    finally:
        priorfile=jobs.state/(label+'.json')
        if priorfile.exists():
            from .backend_jobs import signed,canonical
            prior=json.loads(priorfile.read_bytes());envelope=json.loads((jobs.state/(prior['job_id']+'-job.json')).read_bytes())
            job=signed(envelope,jobs.controller.authority.id)
            if (hashlib.sha256(canonical(job)).hexdigest()!=prior['job_sha256']or job['role']!='evaluate'
                    or signed(job['manifest'],jobs.controller.authority.id)!=manifest
                    or job.get('heldout')!=fields.get('heldout')):raise ValueError('shared evaluator original job binding')
            binding=identity(envelope);lease.bind(request,binding)
            # probe() short-circuits on report existence: inspect the actual
            # runner journal instead, which is written only after child.wait().
            code='import json;from pathlib import Path;p=Path('+repr(jobs.workspace+'/runner-status/'+prior['job_id']+'.json')+');print(p.read_text() if p.is_file() else "{}")'
            try:status=json.loads(jobs.command(shlex.quote(jobs.python)+' -I -B -c '+shlex.quote(code)))
            except Exception:status={}
            if status.get('job_id')!=prior['job_id']:return_marker=None
            else:return_marker=status
            if return_marker and 'child_pid'in status and 'child_pid_ticks'in status:
                handle=dict(pid=status['child_pid'],ticks=status['child_pid_ticks']);lease.bind(request,binding,handle)
                if status.get('phase')in ('complete','failed') and 'exit_code'in status and 'finished_at'in status:
                    lease.terminal(request,binding,handle,dict(actual_child_wait_completed=True,exit_code=status['exit_code'],finished_at=status['finished_at']))
