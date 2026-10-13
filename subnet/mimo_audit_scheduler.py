"""MiMo audit adapter on the existing durable role queue, with no learner wait.

The host provides authenticated release and worker source admission. Random draws
and original jobs are durable before dispatch; a restart or lost enqueue ACK
does not redraw. This service has no chain or weight-setting implementation.
"""
import fcntl
import json
import os
from pathlib import Path
import secrets
import time

from .backend_jobs import signed
from .storage import canonical
from .continuous_audit_service import atomic
from .continuous_audit_policy import digest, random_selection
from .complete_group_capture import _immutable
from .mimo_audit_bridge import authenticate_population, build_audit_job, need


class AuditScheduler:
    def __init__(self, directory, bucket, queue, identity, *, manifest_guard, metadata,
                 max_inflight=8, budget_per_tick=8, job_seconds=900, max_execution_retries=3):
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        need(self.root.absolute()==self.root.resolve() and self.root.stat().st_uid==os.getuid() and
             not self.root.stat().st_mode&0o077,'ordinary owner-private audit service directory')
        for v in (max_inflight,budget_per_tick):need(type(v)is int and 1<=v<=128,'bounded continuous audit parallelism')
        need(type(job_seconds)is int and 60<=job_seconds<=86400 and type(max_execution_retries)is int and
             0<=max_execution_retries<=8,'bounded audit lifetime and infrastructure retries')
        self.bucket,self.queue,self.identity=bucket,queue,identity
        need(queue.authority==identity.id and callable(manifest_guard) and callable(metadata),
             'same queue authority and authenticated release callbacks')
        self.guard,self.metadata=manifest_guard,metadata
        self.maximum,self.budget,self.seconds,self.retries=max_inflight,budget_per_tick,job_seconds,max_execution_retries

    def _lock(self):
        fd=os.open(self.root/'audit.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(fd);raise
        return fd

    def _load(self):
        path=self.root/'state.json'
        if not path.exists():return dict(version='mimo-audit-scheduler-v1',populations={},draws={})
        need(not path.is_symlink() and path.stat().st_size<=256*1024**2,'bounded audit state')
        value=json.loads(path.read_bytes())
        need(type(value)is dict and set(value)=={'version','populations','draws'} and
             value['version']=='mimo-audit-scheduler-v1','explicit original audit state')
        return value

    def _save(self,state):atomic(self.root/'state.json',state)

    def admit(self,document):
        fd=self._lock()
        try:
            p=authenticate_population(document,self.identity.id,manifest_guard=self.guard)
            m=signed(p['manifest_document'],self.identity.id);state=self._load();checksum=digest(document)
            prior=state['populations'].get(m['epoch'])
            need(prior is None or prior==checksum,'immutable audit population epoch')
            path=self.root/(checksum+'.population.json')
            if path.exists():need(path.read_bytes()==canonical(document),'original local population bytes')
            else:atomic(path,document)
            _immutable(self.bucket,f'private/{m["epoch"]}/mimo/audit-population.json',canonical(document))
            state['populations'][m['epoch']]=checksum;self._save(state)
            return checksum
        finally:os.close(fd)

    def populations(self):
        state=self._load();result=[]
        for epoch,checksum in sorted(state['populations'].items()):
            path=self.root/(checksum+'.population.json')
            need(not path.is_symlink() and path.stat().st_size<=256*1024**2,'original audit population file')
            doc=json.loads(path.read_bytes());need(digest(doc)==checksum,'immutable population hash')
            p=authenticate_population(doc,self.identity.id,manifest_guard=self.guard)
            need(p['manifest_document']['payload']['epoch']==epoch,'population epoch directory binding')
            result.append(doc)
        return result

    def _signed(self,payload):
        from .mimo_release_policy import sign
        return sign(self.identity,payload)

    def _queue_status(self,identifier):
        # A crash can happen after our durable job write but before enqueue.
        # The public status() method raises on that legitimate missing row;
        # distinguish it explicitly without swallowing validation/DB errors.
        with self.queue.transaction() as db:
            row=db.execute('SELECT status FROM jobs WHERE id=?',(identifier,)).fetchone()
            return None if row is None else dict(row)

    def tick(self,now=None):
        now=time.time() if now is None else now;fd=self._lock()
        try:
            state=self._load();populations={};records=[]
            for doc in self.populations():
                p=signed(doc,self.identity.id);m=signed(p['manifest_document'],self.identity.id)
                populations[m['epoch']]=p
                if p['committed_at']<=now:records.extend(p['records'])
            inflight=0
            for draw in state['draws'].values():
                if draw.get('job_id'):
                    q=self._queue_status(draw['job_id'])
                    if q and q['status'] in ('queued','leased') and draw['expires_at']>now:inflight+=1
            capacity=min(self.budget,max(0,self.maximum-inflight))
            if not capacity:return dict(enqueued=0,selected=0,backpressure=True)
            pending=[]
            for key,draw in state['draws'].items():
                if not draw.get('job_id'):pending.append((key,draw));continue
                q=self._queue_status(draw['job_id'])
                if q is None and draw['expires_at']>now:pending.append((key,draw));continue
                if q is not None and q['status']=='complete':continue
                if ((q is not None and q['status'] in ('expired','failed')) or draw['expires_at']<=now) and draw.get('generation',0)<self.retries:
                    # Renew only failed execution/expired capabilities. The
                    # selected original contribution remains exactly the same.
                    draw.setdefault('prior_jobs',[]).append(draw['job_id'])
                    draw['generation']=draw.get('generation',0)+1
                    draw.pop('job_id');draw.pop('expires_at');pending.append((key,draw))
            seed=secrets.token_hex(32)
            chosen=random_selection(records,seed,max(0,capacity-len(pending)),state['draws'])
            for row in chosen:
                key=digest(row);draw=dict(row=row,seed=seed,selected_at=now,generation=0)
                state['draws'][key]=draw;pending.append((key,draw))
            self._save(state)  # must precede URL minting, signing and enqueue
            count=0
            for key,draw in pending[:capacity]:
                row=draw['row'];p=populations[row['epoch']];m=signed(p['manifest_document'],self.identity.id)
                self.guard(m);jobid='mimo-audit-'+key[:40]+'-'+str(draw.get('generation',0))
                path=self.root/(jobid+'.job.json')
                if path.exists():
                    envelope=json.loads(path.read_bytes());job=signed(envelope,self.identity.id)
                    need(job['job_id']==jobid and job['manifest']==p['manifest_document'], 'original selected audit job')
                else:
                    metadata=self.metadata(m)
                    need(type(metadata)is dict and set(metadata)<= {'source_files','runtime_versions','checkpoint_read_urls'}
                         and {'source_files','runtime_versions'}<=set(metadata), 'exact admitted audit metadata')
                    job=build_audit_job(p,[row],authority=self.identity.id,manifest_guard=self.guard,
                        job_id=jobid,created_at=now,expires_at=now+self.seconds,
                        read_url=lambda key:self.bucket.presign(key,'get_object',self.seconds),**metadata)
                    envelope=self._signed(job);atomic(path,envelope)
                draw['job_id']=jobid;draw['expires_at']=job['expires_at'];self._save(state)
                _immutable(self.bucket,f'private/{m["epoch"]}/mimo/audit-jobs/{jobid}.json',canonical(envelope))
                self.queue.enqueue(envelope);count+=1
            return dict(enqueued=count,selected=len(chosen),backpressure=False)
        finally:os.close(fd)

    def completed_queue_rows(self):
        """One existing-queue snapshot, retaining original signed report requests."""
        state=self._load();ids=[]
        for draw in state['draws'].values():
            ids.extend(draw.get('prior_jobs',[]))
            if draw.get('job_id'):ids.append(draw['job_id'])
        result=[]
        with self.queue.transaction() as db:
            for jobid in ids:
                row=db.execute('SELECT * FROM jobs WHERE id=? AND status=?',(jobid,'complete')).fetchone()
                if row is not None:result.append(dict(row))
        return result
