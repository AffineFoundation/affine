"""Narrow-key verifier worker; signed job capabilities perform all R2 reads."""
import argparse
from contextlib import ExitStack
import base64
import json
import hashlib
import logging
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
from .distributed_roles import authenticate, digest
from .storage import canonical
from .cache_lifecycle import CacheLifecycle


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
    def __init__(self, url, seed, authority, workspace, python=sys.executable, checkpoint_caches=None, backend_source=None):
        self.url=url.rstrip('/'); self.key=SigningKey(seed); self.identity=self.key.verify_key.encode().hex()
        self.authority=authority; self.workspace=Path(workspace); self.python=python
        self.checkpoint_caches=dict(checkpoint_caches or {})
        self.backend_source=backend_source
        self.workspace.mkdir(parents=True,exist_ok=True); self.workspace.chmod(0o700)
        if not self.url.startswith(('https://','http://127.0.0.1:','http://localhost:')):
            raise ValueError('coordinator requires TLS or local SSH forwarding')

    def request(self, action, **fields):
        payload=dict(action=action,at=time.time(),nonce=secrets.token_hex(16),**fields)
        envelope=dict(payload=payload,signer=self.identity,signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())
        response=requests.post(self.url+'/request',json=envelope,timeout=30,allow_redirects=False)
        if response.status_code!=200: raise ValueError('coordinator request rejected')
        return authenticate(response.json(),self.authority)

    def once(self):
        claim=self.request('claim',role='verify')['claim']
        if claim is None: return False
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
        try:
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
            approved_cache=self.checkpoint_caches.get(approved['id'])
            if approved_cache and checkpoint_cache_candidate(approved_cache,approved['files']):
                command+=['--checkpoint-cache',str(approved_cache)]
            elif claim['attempt']>1 and checkpoint_cache_candidate(cache,approved['files']):
                command+=['--checkpoint-cache',str(cache)]
            with (attempt/'worker.log').open('xb') as output:
                (attempt/'worker.log').chmod(0o600)
                result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=environment,pass_fds=tuple(lease_fds),cwd=self.backend_source)
            if lost.is_set(): raise ValueError('lease expired during execution; retained diagnostic only')
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
                except (requests.RequestException,ValueError):
                    if lost.is_set() or time.time()>=claim['lease_until']: raise
                    time.sleep(2)
            # Coordinator ACK releases disposable inputs, never reports/logs.
            # An original, source-pinned backend also supports this operator
            # wrapper: successful report ACK follows its existing digest checks.
            for i,obj in enumerate(job.get('submissions',[])):
                path=runspace/'jobs'/job['job_id']/('submission-'+str(i)+'.zip')
                if path.exists():lifecycle.record_download(path,obj['sha256'])
            lifecycle.retire_downloads(job['job_id'])
            try:lifecycle.record_checkpoint(approved['id'],approved['files'])
            except ValueError:logging.warning('cache changed after verified job; retaining checkpoint')
            return True
        finally:
            cache_leases.close()
            stopped.set(); thread.join(timeout=31)


def serve(worker, pause=time.sleep):
    """Retry transport outages while retaining original jobs and leases.

    Authority, integrity and unclassified failures still stop the worker.
    """
    while True:
        try:
            worked=worker.once()
        except requests.RequestException as error:
            logging.warning('verifier transport unavailable (%s); retrying in 5 seconds',type(error).__name__)
            pause(5)
            continue
        if not worked:pause(5)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--coordinator',required=True)
    parser.add_argument('--seed-file',required=True); parser.add_argument('--authority',required=True)
    parser.add_argument('--backend-source');parser.add_argument('--workspace',required=True); parser.add_argument('--once',action='store_true')
    parser.add_argument('--checkpoint-cache',action='append',default=[],metavar='CHECKPOINT_ID=LOCAL_PATH')
    args=parser.parse_args(); path=Path(args.seed_file)
    if path.stat().st_mode & 0o077: raise ValueError('worker key must be private')
    caches={}
    for value in args.checkpoint_cache:
        identifier,local=value.split('=',1)
        if len(bytes.fromhex(identifier))!=32 or not Path(local).is_absolute():raise ValueError('exact checkpoint cache mapping')
        if identifier in caches:raise ValueError('duplicate checkpoint cache mapping')
        caches[identifier]=local
    worker=Worker(args.coordinator,bytes.fromhex(path.read_text().strip()),args.authority,args.workspace,checkpoint_caches=caches,backend_source=args.backend_source)
    if args.once:
        worker.once()
        return
    serve(worker)

if __name__=='__main__': main()
