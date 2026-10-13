"""Concrete R2, fixed-block registration and existing SQLite queue adapters.

Instantiation does not publish anything. This module has no weight-writing
method. The caller must supply an explicitly enabled, qualified release and an
isolated state/queue configuration before running Lifecycle.tick.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .backend_jobs import signed
from .storage import canonical
from .mimo_lifecycle import Services
from .mimo_release_policy import descriptor, digest, need, sign


class R2Services:
    def __init__(self, bucket, identity, policy, queue, chain, *, discovery_key,
                 initial_checkpoint_keys, source_bundle_key, source_files, runtime_versions,
                 trainer_worker, publisher_id, trainer_role_config_file,
                 trainer_config, job_seconds=1800, readback_state):
        from .mimo_grpo.publication import _namespace
        self.bucket,self.identity,self.policy,self.queue,self.chain=bucket,identity,policy,queue,chain
        self.discovery_key=_namespace(discovery_key)
        self.source_bundle_key=_namespace(source_bundle_key)
        self._source_checked=None
        # Public discovery is a signed manifest only; private diagnostics cannot
        # accidentally overwrite a shared public pointer through this adapter.
        need(policy.mode != 'private' or discovery_key.startswith(policy.namespace()+'/'),
             'private discovery must remain in its private run namespace')
        self.initial_keys=dict(initial_checkpoint_keys)
        need(set(self.initial_keys)==set(policy.scope['initial_checkpoint']['files']),
             'complete original checkpoint object mapping')
        for key in self.initial_keys.values(): _namespace(key)
        need(type(job_seconds)is int and 60 <= job_seconds <= 86400, 'bounded trainer job lifetime')
        self.source_files,self.runtime_versions=source_files,runtime_versions
        self.trainer_worker,self.publisher_id=trainer_worker,publisher_id
        self.role_config,self.trainer_config=trainer_role_config_file,trainer_config
        self.job_seconds=job_seconds
        self.readback_state=Path(readback_state).absolute()
        need(self.readback_state.resolve()==self.readback_state,'ordinary model readback state')
        self.readback_state.mkdir(parents=True,exist_ok=True,mode=0o700)

    def services(self):
        return Services(self.registrations,self.committed_miners,self.discovery,
            self.checkpoint_urls,self.make_training_job,self.enqueue_training,
            self.training_result,self.readback_publication,self.source_bundle)

    def source_bundle(self,source):
        expected=self.policy.scope['immutable_manifest']['source_bundle']
        need(source==expected,'exact qualified source archive descriptor')
        need(type(source.get('size'))is int and 0 < source['size'] <=32*1024**2,
             'bounded qualified source archive')
        if self._source_checked != source:
            from .complete_group_capture import _get
            raw,_=_get(self.bucket,self.source_bundle_key,source['size'])
            need(len(raw)==source['size'] and hashlib.sha256(raw).hexdigest()==source['sha256'],
                 'source archive full readback hash')
            self._source_checked=dict(source)
        return dict(source,url=self.bucket.presign(self.source_bundle_key,'get_object',86400))

    def registrations(self):
        """Existing authenticated read-only activation path, one immutable block."""
        rows=self.chain.registrations()
        blocks={r['snapshot_block']for r in rows.values()}
        need(len(blocks)<=1,'one fixed-block activated registration snapshot')
        # The empty case never opens an epoch; this observation only supports status.
        block=next(iter(blocks))if blocks else int(self.chain.chain.block)
        return dict(snapshot_block=block,registrations={hotkey:{k:r[k]for k in
            ('uid','public_key','snapshot_block')}for hotkey,r in rows.items()})

    def committed_miners(self,manifest):
        self.policy.validate_manifest(manifest)
        return self.bucket.complete_commitment_listing(manifest['epoch'],manifest['capabilities'],
            cutoff=time.time()+30)

    def discovery(self,opening,previous_sha256):
        from botocore.exceptions import ClientError
        from .mimo_grpo.publication import _missing
        m=signed(opening,self.identity.id);self.policy.validate_manifest(m)
        expected=digest(opening); conditions={}
        try:
            response=self.bucket.client.get_object(Bucket=self.bucket.name,Key=self.discovery_key)
        except ClientError as error:
            if not _missing(error):raise
            need(previous_sha256 is None,'expected prior discovery is missing')
            conditions['IfNoneMatch']='*'
        else:
            body=response['Body']
            try:
                need(0 < response['ContentLength'] <= 32*1024**2,'bounded current discovery record')
                existing=body.read(response['ContentLength']+1)
                need(len(existing)==response['ContentLength'],'complete current discovery read')
            finally:body.close()
            original=signed(json.loads(existing),self.identity.id)
            if original.get('manifest_sha256')==expected and original.get('epoch')==m['epoch']:
                return dict(sha256=expected,discovery_sha256=hashlib.sha256(existing).hexdigest(),key=self.discovery_key)
            need(hashlib.sha256(existing).hexdigest()==previous_sha256,'discovery changed from the expected predecessor')
            need(type(response.get('ETag'))is str and response['ETag'],'discovery conditional ETag required')
            conditions['IfMatch']=response['ETag']
        # No unsafe fallback if conditional PutObject is unsupported.
        manifest_key=self.policy.namespace()+'/epochs/'+m['epoch']+'/mimo/opening.json'
        from .complete_group_capture import _get
        original_opening,_=_get(self.bucket,manifest_key,len(canonical(opening)))
        need(original_opening==canonical(opening),'discovery points to exact immutable signed opening')
        published=sign(self.identity,dict(epoch=m['epoch'],run_id=m['run_id'],transport_policy='direct-r2-v1',
            manifest_url=self.bucket.presign(manifest_key,'get_object',86400),manifest_sha256=expected,
            source_bundle=m['source_bundle'],checkpoint=m['checkpoint']['id'],
            current_url=self.bucket.presign(self.discovery_key,'get_object',86400),
            current_url_expires_at=time.time()+86400))
        raw=canonical(published)
        self.bucket.client.put_object(Bucket=self.bucket.name,Key=self.discovery_key,
            Body=raw,ContentType='application/json',**conditions)
        observed,_=_get(self.bucket,self.discovery_key,len(raw))
        need(observed==raw,'public discovery full readback differs')
        return dict(sha256=expected,discovery_sha256=digest(published),key=self.discovery_key)

    def checkpoint_urls(self,checkpoint):
        cp=descriptor(checkpoint)
        if cp==self.policy.scope['initial_checkpoint']:keys=self.initial_keys
        else:keys={name:self.policy.namespace()+'/checkpoints/'+cp['id']+'/model/'+name for name in cp['files']}
        return {name:self.bucket.presign(key,'get_object',86400)for name,key in keys.items()}

    def make_training_job(self,opening,dispatch,optimizer_step,namespace):
        from .mimo_worker import VERSION as WORKER
        m=signed(opening,self.identity.id);self.policy.validate_manifest(m)
        d=signed(dispatch,self.identity.id)
        need(namespace==self.policy.namespace() and d['manifest_sha256']==digest(m),'original scoped training dispatch')
        key=namespace+'/epochs/'+m['epoch']+'/mimo/dispatch.json'
        documents={}
        for row in d['selected']:
            doc=row['document']; response=self.bucket.client.head_object(Bucket=self.bucket.name,Key=doc['key'])
            need(type(response['ContentLength'])is int and 0 < response['ContentLength'] <=128*1024**2,
                 'bounded actual captured training document')
            documents[doc['sha256']]=dict(url=self.bucket.presign(doc['key'],'get_object',86400),size=response['ContentLength'])
        need(len(documents)==len(d['selected']),'one original compact document per training group')
        now=time.time()
        value=dict(schema=1,worker_policy=WORKER,job_id='mimo-train-'+digest(dispatch)[:48],role='train',
            created_at=now,expires_at=now+self.job_seconds,manifest=opening,
            source_files=self.source_files,runtime_versions=self.runtime_versions,trainer_worker=self.trainer_worker,
            checkpoint_read_urls=self.checkpoint_urls(m['checkpoint']),training=dict(run_id=m['run_id'],
                optimizer_step=optimizer_step,config=self.trainer_config,
                dispatch=dict(url=self.bucket.presign(key,'get_object',86400),sha256=digest(dispatch),size=len(canonical(dispatch))),
                documents=documents,publication_namespace=namespace,publisher_id=self.publisher_id,
                role_config_file=self.role_config))
        return sign(self.identity,value)

    def enqueue_training(self,envelope):
        from .mimo_lifecycle import TrainingRecoveryRequired
        job=signed(envelope,self.identity.id)
        with self.queue.transaction()as db:
            row=db.execute('SELECT status,expires,digest FROM jobs WHERE id=?',(job['job_id'],)).fetchone()
        if row is not None:
            need(row['digest']==digest(job),'original immutable trainer queue job')
            if row['status']=='complete':return job['job_id']
            if row['status']in('failed','expired')or row['expires']<=time.time():
                raise TrainingRecoveryRequired(job['job_id'],'expired'if row['expires']<=time.time()else row['status'])
        return self.queue.enqueue(envelope)

    def training_result(self,job_id):
        # status() deliberately doesn't expose the signature; read its original
        # authenticated request from the same coordinator-owned SQLite database.
        with self.queue.transaction() as db:
            row=db.execute('SELECT status,expires,report_request FROM jobs WHERE id=?',(job_id,)).fetchone()
        need(row is not None,'original trainer job is absent from queue')
        if row['status']!='complete'and(row['status']in('failed','expired')or row['expires']<=time.time()):
            from .mimo_lifecycle import TrainingRecoveryRequired
            raise TrainingRecoveryRequired(job_id,'expired'if row['expires']<=time.time()else row['status'])
        if row['status']!='complete':return None
        need(row['report_request']is not None,'original worker signature is absent')
        return json.loads(row['report_request'])

    def readback_publication(self,published):
        cp=descriptor(published['descriptor']); need(cp['id']==published['output_checkpoint'],'publication output binding')
        objects=published['objects'];need(set(objects)==set(cp['files']),'complete model readback inventory')
        receipt=digest(published); path=self.readback_state/(receipt+'.json')
        if path.exists():
            need(not path.is_symlink() and json.loads(path.read_bytes())==dict(receipt_sha256=receipt,checkpoint=cp),
                 'same completed immutable model readback')
            return cp
        # Bounded streaming, no checkpoint download/cache on the validator host.
        for name,checksum in cp['files'].items():
            expected=objects[name]
            need(expected['key']==self.policy.namespace()+'/checkpoints/'+cp['id']+'/model/'+name
                 and expected['sha256']==checksum and type(expected['size'])is int
                 and 0 < expected['size'] <=20*1024**3,'scoped bounded published model object')
            response=self.bucket.client.get_object(Bucket=self.bucket.name,Key=expected['key']);body=response['Body']
            try:
                need(response['ContentLength']==expected['size'],'published model byte count')
                checksum_state=hashlib.sha256();count=0
                while True:
                    chunk=body.read(4*1024**2)
                    if not chunk:break
                    count+=len(chunk);need(count<=expected['size'],'published model exceeds declared size')
                    checksum_state.update(chunk)
                need(count==expected['size'] and checksum_state.hexdigest()==checksum,'full model readback hash')
            finally:body.close()
        from .controller import save_manifest
        save_manifest(path,dict(receipt_sha256=receipt,checkpoint=cp))
        return cp
