"""Narrow-key verifier worker; signed job capabilities perform all R2 reads."""
import argparse
from contextlib import ExitStack
import base64
import json
import hashlib
import logging
import math
import re
import stat
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
import requests
from nacl.signing import SigningKey
from nacl.exceptions import BadSignatureError
from .distributed_roles import authenticate, digest
from .storage import canonical
from .cache_lifecycle import CacheLifecycle, snapshot


class ExpiredCompletedLease(ValueError):
    """The original backend has exited, but its lease no longer permits ACK."""


def checkpoint_cache_candidate(path,files):
    """Select a local candidate without reading weights; this is NOT admission.

    backend_jobs.checkpoint still reads and hashes every approved member and
    verifies the full allowlist before GPURuntime can open the model. A corrupt
    candidate therefore fails the job rather than gaining trusted-cache status.
    """
    if (not isinstance(files,dict) or not files or any(not isinstance(name,str) or
            Path(name).name!=name or name in ('.','..') or not isinstance(expected,str) or
            re.fullmatch('[0-9a-f]{64}',expected) is None for name,expected in files.items())):
        raise ValueError('approved checkpoint file inventory')
    root=Path(path)
    try:mode=root.lstat().st_mode
    except FileNotFoundError:return False
    if not stat.S_ISDIR(mode) or root.absolute()!=root.resolve():return False
    observed={entry.name:entry for entry in root.iterdir()}
    return set(observed)==set(files) and all(
        stat.S_ISREG(entry.lstat().st_mode) for entry in observed.values())


def complete_checkpoint_cache(path,files):
    """CPU-only exact inventory/readback; absence/mismatch is not an admission.

    Storage errors propagate rather than becoming a cache miss. No model import,
    deletion, or repair happens here; signed backend hydration remains authoritative.
    """
    if (not isinstance(files,dict) or not files or any(not isinstance(name,str) or
            Path(name).name!=name or name in ('.','..') or not isinstance(expected,str) or
            re.fullmatch('[0-9a-f]{64}',expected) is None for name,expected in files.items())):
        raise ValueError('approved checkpoint file inventory')
    root=Path(path)
    try:mode=root.lstat().st_mode
    except FileNotFoundError:return False
    if not stat.S_ISDIR(mode) or root.absolute()!=root.resolve():return False
    observed={entry.name:entry for entry in root.iterdir()}
    if set(observed)!=set(files):return False
    for name,expected in files.items():
        entry=observed[name]
        if not stat.S_ISREG(entry.lstat().st_mode):return False
        digest_value=hashlib.sha256()
        with entry.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest_value.update(block)
        if digest_value.hexdigest()!=expected:return False
    return True


class Worker:
    def __init__(self, url, seed, authority, workspace, python=sys.executable, checkpoint_caches=None, backend_source=None, source_registry=None, checkpoint_retention=None):
        self.url=url.rstrip('/'); self.key=SigningKey(seed); self.identity=self.key.verify_key.encode().hex()
        self.authority=authority; self.workspace=Path(workspace); self.python=python
        self.checkpoint_retention=checkpoint_retention
        if checkpoint_retention is not None:
            if (set(checkpoint_retention)!={'ttl_seconds','disk_floor_bytes'} or
                isinstance(checkpoint_retention['ttl_seconds'],bool) or
                not isinstance(checkpoint_retention['ttl_seconds'],(int,float)) or
                not 0<checkpoint_retention['ttl_seconds']<=86400 or
                isinstance(checkpoint_retention['disk_floor_bytes'],bool) or
                not isinstance(checkpoint_retention['disk_floor_bytes'],int) or
                checkpoint_retention['disk_floor_bytes']<0):raise ValueError('bounded owned checkpoint retention')
        self.checkpoint_caches=dict(checkpoint_caches or {})
        self.backend_source=backend_source
        self.source_registry=dict(source_registry or {})
        self._source_inventory_checked=set()
        self._expired_input_scan_offset=0
        self.workspace.mkdir(parents=True,exist_ok=True); self.workspace.chmod(0o700)
        if not self.url.startswith(('https://','http://127.0.0.1:','http://localhost:')):
            raise ValueError('coordinator requires TLS or local SSH forwarding')

    def request(self, action, **fields):
        payload=dict(action=action,at=time.time(),nonce=secrets.token_hex(16),**fields)
        envelope=dict(payload=payload,signer=self.identity,signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())
        response=requests.post(self.url+'/request',json=envelope,timeout=30,allow_redirects=False)
        if response.status_code!=200: raise ValueError('coordinator request rejected')
        return authenticate(response.json(),self.authority)

    def source_for_job(self,job):
        """ROOT registry routes pinned archives; manifest never supplies a path.

        Verify each configured runtime inventory once per worker start. Backend
        retains its existing signed per-job source checks. No 2053-file scan or
        checkpoint rehash is added to each job.
        """
        if not self.source_registry:
            if self.backend_source:raise ValueError('operator backend requires explicit pinned source registry')
            return None
        bundle=job['manifest']['payload'].get('source_bundle',{}).get('sha256')
        entry=self.source_registry.get(bundle)
        if entry is None:raise ValueError('job source archive not in ROOT registry')
        code=Path(entry['path'])
        if not code.is_absolute() or code!=code.resolve() or not code.is_dir():raise ValueError('ROOT source path')
        files=entry['source_files']
        if not files or files!=job.get('source_files'):raise ValueError('job runtime source inventory differs from ROOT registry')
        if bundle not in self._source_inventory_checked:
            for name,expected in files.items():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('source member path')
                path=code/relative
                if path!=path.resolve() or not path.is_file():raise ValueError('source member symlink or missing')
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
                if h.hexdigest()!=expected:raise ValueError('staged ROOT source changed')
            self._source_inventory_checked.add(bundle)
        return str(code)

    def retire_expired_inputs(self, attempt):
        """Dispose verified input copies after an original backend lost its lease.

        Local terminal diagnostics do not grant credit or classify fraud. Only
        authenticated job capabilities and preexisting hydration receipts name
        disposable files; inherited checkpoint locks guard any live reuse.
        """
        attempt=Path(attempt).absolute()
        if attempt!=attempt.resolve() or attempt.parent.parent!=self.workspace.absolute():
            raise ValueError('exact owned attempt path')
        number=attempt.name.removeprefix('attempt-')
        if not number.isdecimal() or int(number)<1:raise ValueError('exact attempt number')
        evidence=attempt/'expired-completed-lease.json';jobpath=attempt/'job.json'
        for path in (evidence,jobpath):
            if snapshot(path)['mode']&0o077:raise ValueError('private original diagnostics required')
        diagnostic=json.loads(evidence.read_bytes())
        try:job=authenticate(json.loads(jobpath.read_bytes()),self.authority)
        except BadSignatureError as error:raise ValueError('historical original job authentication')from error
        if not isinstance(diagnostic,dict):raise ValueError('typed terminal diagnostics')
        if (job.get('role')!='verify' or job['job_id']!=attempt.parent.name or
            diagnostic.get('job_id')!=job['job_id'] or diagnostic.get('job_sha256')!=digest(job) or
            type(diagnostic.get('attempt'))is not int or diagnostic['attempt']!=int(number) or
            diagnostic.get('backend_terminal')is not True or type(diagnostic.get('backend_exit'))is not int or
            diagnostic.get('report_acknowledged')is not False or
            diagnostic.get('stage')not in ('backend_completed','report_acknowledgment')):
            raise ValueError('original completed lease diagnostic binding')
        for name in ('lease_until','observed_at'):
            value=diagnostic.get(name)
            if type(value)not in (int,float) or not math.isfinite(value):raise ValueError('terminal observation time')
        if not diagnostic['lease_until']<=diagnostic['observed_at']<=time.time():raise ValueError('actual elapsed original lease')
        runspace=self.workspace/'backend' if int(number)==1 else attempt/'backend'
        if not runspace.is_dir() or runspace!=runspace.resolve():raise ValueError('original owned backend workspace')
        lifecycle=CacheLifecycle(runspace);receipt=lifecycle.meta/('download-'+job['job_id']+'.json')
        if not receipt.exists():return []
        if snapshot(receipt)['mode']&0o077:raise ValueError('private hydration ownership receipt')
        catalog=json.loads(receipt.read_bytes());eligible=[]
        if not isinstance(catalog,dict):raise ValueError('typed hydration catalog')
        for i,obj in enumerate(job.get('submissions',[])):
            relative='jobs/'+job['job_id']+'/submission-'+str(i)+'.zip'
            record=catalog.get(relative)
            if record is not None:
                if not isinstance(record,dict):raise ValueError('typed hydration ownership receipt')
                if record.get('sha256')==obj['sha256']:eligible.append(relative)
        checkpoint=job['manifest']['payload']['checkpoint']['id']
        with ExitStack()as locks:
            locks.enter_context(CacheLifecycle(self.workspace/'backend').lease_checkpoint(checkpoint,blocking=False))
            if runspace!=self.workspace/'backend':locks.enter_context(lifecycle.lease_checkpoint(checkpoint,blocking=False))
            return lifecycle.retire_downloads(job['job_id'],only=eligible)

    def sweep_expired_inputs(self):
        """Bounded idle recovery of terminal inputs; no adoption or proof rehash."""
        candidates=[]
        for evidence in self.workspace.glob('*/attempt-*/expired-completed-lease.json'):
            attempt=evidence.parent
            number=attempt.name.removeprefix('attempt-')
            if not number.isdecimal():continue
            runspace=self.workspace/'backend' if int(number)==1 else attempt/'backend'
            if not (runspace/'.cache-lifecycle'/('download-'+attempt.parent.name+'.json')).exists():continue
            candidates.append(attempt)
        if not candidates:return
        candidates.sort();start=self._expired_input_scan_offset%len(candidates)
        for index in range(min(4,len(candidates))):
            attempt=candidates[(start+index)%len(candidates)]
            try:self.retire_expired_inputs(attempt)
            except (OSError,ValueError):logging.warning('terminal disposable input retirement deferred; retaining diagnostics')
        self._expired_input_scan_offset=start+min(4,len(candidates))

    def sweep_owned_checkpoints(self,lifecycle):
        if self.checkpoint_retention is None:return lifecycle.evict_checkpoints(keep=0)
        return lifecycle.retain_acknowledged_checkpoint(**self.checkpoint_retention)

    def once(self):
        claim=self.request('claim',role='verify')['claim']
        if claim is None:
            # No new work is required to retire already verified owned models.
            # Lease/inode guards retain every active or changed cache; external
            # mapped caches and report/log evidence are outside this catalog.
            try:self.sweep_owned_checkpoints(CacheLifecycle(self.workspace/'backend'))
            except (OSError,ValueError):logging.warning('idle owned checkpoint disposal deferred; retaining evidence')
            try:self.sweep_expired_inputs()
            except (OSError,ValueError):logging.warning('terminal input inventory unavailable; retaining diagnostics')
            return False
        job=authenticate(claim['job'],self.authority)
        if digest(job)!=claim['job_sha256'] or job['role']!='verify': raise ValueError('claim job binding')
        # Retry workspace differs by lease attempt; the immutable signed job ID
        # remains unchanged. No previous attempt directory is overwritten.
        attempt=self.workspace/job['job_id']/('attempt-'+str(claim['attempt']))
        attempt.mkdir(parents=True,exist_ok=False); attempt.chmod(0o700)
        jobpath=attempt/'job.json'; jobpath.write_bytes(canonical(claim['job'])); jobpath.chmod(0o600)
        stopped=threading.Event(); lost=threading.Event()
        def renew():
            while not stopped.wait(max(1,min(30,(claim['lease_until']-time.time())/3))):
                try:
                    result=self.request('renew',job_id=job['job_id'],token=claim['token'])
                    claim['lease_until']=result['lease_until']
                except Exception:
                    # A temporary transport fault must not publish an unchecked
                    # replacement. Retry until the known lease expires.
                    if time.time()>=claim['lease_until']: lost.set(); return
        thread=threading.Thread(target=renew,daemon=True); thread.start()
        cache_leases=ExitStack()
        lifecycle=None;runspace=None;acked_owned_checkpoint=None;expired_attempt=None
        try:
            backend_source=self.source_for_job(job)
            environment=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
            runspace=self.workspace/'backend' if claim['attempt']==1 else attempt/'backend'
            cache=self.workspace/'backend'/'checkpoints'/job['manifest']['payload']['checkpoint']['id']
            command=[self.python,'-B','-m','subnet.backend_jobs',str(jobpath),
                '--authority',self.authority,'--workspace',str(runspace)]
            approved=job['manifest']['payload']['checkpoint']
            lifecycle=CacheLifecycle(runspace)
            shared_lifecycle=CacheLifecycle(self.workspace/'backend')
            shared_lifecycle.evict_checkpoints(exclude=[approved['id']],keep=0)
            lifecycle.evict_checkpoints(exclude=[approved['id']],keep=0)
            lease_fds=[cache_leases.enter_context(shared_lifecycle.lease_checkpoint(approved['id']))]
            if runspace!=self.workspace/'backend':
                lease_fds.append(cache_leases.enter_context(lifecycle.lease_checkpoint(approved['id'])))
            environment['AFFINE_CACHE_LIFECYCLE_ROOT']=str(runspace)
            selected_cache=None
            approved_cache=self.checkpoint_caches.get(approved['id'])
            if approved_cache and checkpoint_cache_candidate(approved_cache,approved['files']):
                selected_cache=Path(approved_cache)
                command+=['--checkpoint-cache',str(approved_cache)]
            elif claim['attempt']>1 and checkpoint_cache_candidate(cache,approved['files']):
                selected_cache=cache
                command+=['--checkpoint-cache',str(cache)]
            with (attempt/'worker.log').open('xb') as output:
                (attempt/'worker.log').chmod(0o600)
                result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=environment,pass_fds=tuple(lease_fds),cwd=backend_source)
            def expired_completed_lease(stage):
                nonlocal expired_attempt
                evidence=attempt/'expired-completed-lease.json'
                with evidence.open('xb') as stream:
                    stream.write(canonical(dict(job_id=job['job_id'],
                        job_sha256=claim['job_sha256'],attempt=claim['attempt'],
                        stage=stage,backend_terminal=True,backend_exit=result.returncode,
                        lease_until=claim['lease_until'],observed_at=time.time(),
                        report_acknowledged=False)))
                evidence.chmod(0o600)
                expired_attempt=attempt
                return ExpiredCompletedLease('completed backend lost its lease; retained diagnostic only')
            if lost.is_set(): raise expired_completed_lease('backend_completed')
            if result.returncode:
                self.request('fail',job_id=job['job_id'],token=claim['token'])
                lifecycle.retire_downloads(job['job_id']); return True
            report=json.loads((runspace/'jobs'/job['job_id']/'report.json').read_text())
            # Persist the exact report before transport; every retry uses the
            # same report bytes and fresh authenticated request nonce.
            pending=attempt/'pending-report.json';pending.write_bytes(canonical(report));pending.chmod(0o600)
            while True:
                try:
                    self.request('report',job_id=job['job_id'],token=claim['token'],report=report);break
                except (requests.RequestException,ValueError) as error:
                    if lost.is_set() or time.time()>=claim['lease_until']:
                        raise expired_completed_lease('report_acknowledgment') from error
                    time.sleep(2)
            # Coordinator ACK releases disposable inputs, never reports/logs.
            # An original, source-pinned backend also supports this operator
            # wrapper: successful report ACK follows its existing digest checks.
            for i,obj in enumerate(job.get('submissions',[])):
                path=runspace/'jobs'/job['job_id']/('submission-'+str(i)+'.zip')
                if path.exists():lifecycle.record_download(path,obj['sha256'])
            lifecycle.retire_downloads(job['job_id'])
            try:
                if selected_cache is None:
                    if lifecycle.record_checkpoint(approved['id'],approved['files']):acked_owned_checkpoint=lifecycle
                elif selected_cache==cache:
                    if shared_lifecycle.record_checkpoint(approved['id'],approved['files']):acked_owned_checkpoint=shared_lifecycle
                # External mapped caches were verified at their own path; never
                # label a different same-ID owned copy as that verified input.
            except ValueError:logging.warning('cache changed after verified job; retaining checkpoint')
            return True
        finally:
            cache_leases.close()
            if expired_attempt is not None:
                try:self.retire_expired_inputs(expired_attempt)
                except (OSError,ValueError):logging.warning('terminal disposable input retirement deferred; retaining diagnostics')
            # Only the successfully ACKed, backend-verified owned checkpoint
            # follows the configured bounded operator retention policy. Release
            # this job's inherited flock first;
            # another live lease or changed inode makes eviction refuse safely.
            # External mapped caches and all reports/logs remain untouched.
            if acked_owned_checkpoint is not None:
                try:
                    if self.checkpoint_retention is not None and acked_owned_checkpoint.root==(self.workspace/'backend').absolute():
                        self.sweep_owned_checkpoints(acked_owned_checkpoint)
                    else:acked_owned_checkpoint.evict_checkpoints(only=[approved['id']],keep=0)
                except (OSError,ValueError):logging.warning('ACKed owned checkpoint disposal deferred; retaining evidence')
            # Retry workspaces are disposable model inputs, not independent
            # long-lived caches. Their report/job diagnostics remain in place.
            if lifecycle is not None and runspace!=self.workspace/'backend':
                lifecycle.evict_checkpoints(keep=0)
            stopped.set(); thread.join(timeout=31)


def serve(worker, pause=time.sleep):
    """Retry transport outages while retaining original jobs and leases.

    A completed backend whose lease expired retains diagnostics and returns to
    polling. This never resubmits its report or restarts that physical attempt.
    Authority, integrity and unclassified failures still stop the worker.
    """
    while True:
        try:
            worked=worker.once()
        except requests.RequestException as error:
            logging.warning('verifier transport unavailable (%s); retrying in 5 seconds',type(error).__name__)
            pause(5)
            continue
        except ExpiredCompletedLease:
            logging.warning('completed verifier attempt lost its lease; retaining diagnostics and polling again')
            pause(5)
            continue
        if not worked:pause(5)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--coordinator',required=True)
    parser.add_argument('--seed-file',required=True); parser.add_argument('--authority',required=True)
    parser.add_argument('--source-registry');parser.add_argument('--backend-source');parser.add_argument('--workspace',required=True); parser.add_argument('--once',action='store_true')
    parser.add_argument('--owned-checkpoint-ttl-seconds',type=int)
    parser.add_argument('--owned-checkpoint-disk-floor-bytes',type=int)
    parser.add_argument('--checkpoint-cache',action='append',default=[],metavar='CHECKPOINT_ID=LOCAL_PATH')
    args=parser.parse_args(); path=Path(args.seed_file)
    if path.stat().st_mode & 0o077: raise ValueError('worker key must be private')
    if (args.owned_checkpoint_ttl_seconds is None)!=(args.owned_checkpoint_disk_floor_bytes is None):raise ValueError('both owned checkpoint retention limits required')
    retention=None if args.owned_checkpoint_ttl_seconds is None else dict(ttl_seconds=args.owned_checkpoint_ttl_seconds,disk_floor_bytes=args.owned_checkpoint_disk_floor_bytes)
    caches={}
    for value in args.checkpoint_cache:
        identifier,local=value.split('=',1)
        if len(bytes.fromhex(identifier))!=32 or not Path(local).is_absolute():raise ValueError('exact checkpoint cache mapping')
        if identifier in caches:raise ValueError('duplicate checkpoint cache mapping')
        caches[identifier]=local
    registry=json.loads(Path(args.source_registry).read_text()) if args.source_registry else {}
    worker=Worker(args.coordinator,bytes.fromhex(path.read_text().strip()),args.authority,args.workspace,checkpoint_caches=caches,backend_source=args.backend_source,source_registry=registry,checkpoint_retention=retention)
    if args.once:
        worker.once()
        return
    serve(worker)

if __name__=='__main__': main()
