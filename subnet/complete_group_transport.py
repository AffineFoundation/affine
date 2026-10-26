"""Real versioned R2 complete-group adapter; no legacy pair-document fallback.

Only authenticated manifest-selected code instantiates this adapter. Credentials
stay in the operator bucket client; miners receive only encrypted, scoped PUT
capabilities. This candidate is exercised with mock HTTP/bucket clients only.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time
from urllib.parse import urlparse,unquote,parse_qs
import requests
from .storage import canonical,encrypt
from .batch_contract import REQUIRED_CLIENT_FEATURES,CompleteGroupContract
from .complete_group_codec import limits,encode,decode,loads,need
from .complete_group_admission import (TRANSPORT,context,scope,sha,digest,local_task,
    structural_batch,document,signed_commitment,validate_commitment)
from .complete_group_state import _write

PREPARED='complete-group-prepared-state-v1'
JOURNAL='complete-group-upload-journal-v1'


def paths(manifest,miner,slot=None):
    epoch=manifest['epoch']
    need(type(epoch) is str and epoch and all(c.isalnum() or c in '-_.' for c in epoch)
         and '..' not in epoch,'safe canonical epoch object namespace')
    if slot is None:return f'private/{epoch}/commitments/{miner}.json'
    return (f'private/{epoch}/staging/{miner}/{slot}.zip',f'private/{epoch}/training/{miner}/{slot}.json')


def capability(bucket,manifest,miner,*,registered_miners):
    need(miner in registered_miners,'registered epoch miner')
    contract,_=context(manifest,miner);remaining=int(manifest['deadline']-time.time())
    need(remaining>0,'epoch already closed')
    cap=dict(transport=TRANSPORT,scope_sha256=scope(manifest,miner),deadline=manifest['deadline'],
        headers={'Content-Type':'application/octet-stream'},
        put_url=bucket.presign(paths(manifest,miner),'put_object',remaining),
        upload_headers={'Content-Type':'application/octet-stream','If-None-Match':'*'},
        batch_put_urls=[],training_put_urls=[],batch_get_urls=[],training_get_urls=[])
    for slot in range(contract.max_batches):
        proof,training=paths(manifest,miner,slot)
        for kind,key in (('batch',proof),('training',training)):
            cap[kind+'_put_urls'].append(bucket.client.generate_presigned_url('put_object',
                Params=dict(Bucket=bucket.name,Key=key,ContentType='application/octet-stream',IfNoneMatch='*'),ExpiresIn=remaining))
            cap[kind+'_get_urls'].append(bucket.presign(key,'get_object',remaining))
    validate_capability(cap,manifest,miner)
    return encrypt(miner,cap)


def validate_capability(cap,manifest,miner):
    from .client import direct_r2_url
    contract,_=context(manifest,miner)
    need(type(cap) is dict and set(cap)=={'transport','scope_sha256','deadline','headers','upload_headers','put_url','batch_put_urls','training_put_urls','batch_get_urls','training_get_urls'}
         and cap['transport']==TRANSPORT and cap['scope_sha256']==scope(manifest,miner)
         and type(cap['deadline']) is int and cap['deadline']==manifest['deadline']
         and cap['headers']=={'Content-Type':'application/octet-stream'}
         and cap['upload_headers']=={'Content-Type':'application/octet-stream','If-None-Match':'*'},'exact epoch-scoped group capability')
    for name in ('batch_put_urls','training_put_urls','batch_get_urls','training_get_urls'):
        need(type(cap[name]) is list and len(cap[name])==contract.max_batches,'exact signed capability slot count')
    urls=[(cap['put_url'],paths(manifest,miner),'commit')]
    for slot in range(contract.max_batches):
        proof,training=paths(manifest,miner,slot)
        for kind,key in (('batch',proof),('training',training)):
            urls += [(cap[kind+'_put_urls'][slot],key,'put'),(cap[kind+'_get_urls'][slot],key,'get')]
    endpoint=None;seen=set()
    for url,key,action in urls:
        direct_r2_url(url);parsed=urlparse(url);decoded=unquote(parsed.path)
        suffix='/'+key
        need(decoded.endswith(suffix) and not parsed.fragment,'capability object path binding')
        prefix=(parsed.netloc,decoded[:-len(suffix)])
        need(prefix[1] and (endpoint is None or prefix==endpoint),'one scoped R2 account/bucket')
        if action=='put':
            signed=parse_qs(parsed.query).get('X-Amz-SignedHeaders',[''])[0].split(';')
            need('if-none-match' in signed and 'content-type' in signed,'conditional immutable PUT headers must be signature-bound')
        endpoint=prefix;need((action,decoded) not in seen,'distinct upload capability objects');seen.add((action,decoded))
    return cap


@contextmanager
def _lock(path):
    descriptor=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:os.close(descriptor)


class CompleteGroupTransport:
    features=REQUIRED_CLIENT_FEATURES

    def __init__(self,*,http=None):
        self.http=http if http is not None else requests
        self.path=None;self.previous=[];self.memory_journal={}

    def validate(self,manifest,contract):
        need(isinstance(contract,CompleteGroupContract) and manifest.get('submission_transport_policy')==TRANSPORT,
             'explicit new complete-group transport required')
        need(manifest.get('transport_policy')=='direct-r2-v1','complete groups use scoped R2 capabilities')
        need(type(manifest.get('deadline')) is int and type(manifest.get('start')) is int
             and manifest['start']<manifest['deadline'],'signed epoch upload window')
        need('artifact_policy' not in manifest,'new group transport cannot retain legacy artifact policy')
        limits(manifest)

    def validate_capability(self,cap,manifest,miner):
        return validate_capability(cap,manifest,miner)

    def check_runtime(self,runtime,manifest):
        from .complete_group_sampling import validate_geometry
        validate_geometry(runtime, runtime.sampling_context)

    def _check(self,batch,arrays,manifest,miner):
        policy=manifest['group_transport_policy']
        from .complete_group_distribution import eos_ids
        stops = eos_ids(policy)
        task=local_task(batch,manifest,eos_token_id=stops[0],vocab_size=policy['vocab_size'],
                        eos_token_ids=stops if 'eos_token_ids' in policy else ())
        return structural_batch(batch,manifest,miner,task,arrays)

    def _all(self,prepared,manifest,miner):
        bound=limits(manifest)
        need(type(prepared) is list and len(prepared)<=manifest['max_batches'],'cumulative signed group cap')
        total=0;seen=set();rows=[]
        for batch,data in prepared:
            decoded=decode(data,bound);actual,arrays=decoded[0]
            need(actual==batch,'prepared metadata differs from exact original bytes')
            result=self._check(batch,arrays,manifest,miner)
            need(result['group_id'] not in seen,'one canonical task in cumulative state');seen.add(result['group_id'])
            total+=len(data);need(total<=bound.cumulative_bytes,'cumulative prepared byte bound')
            rows.append(dict(sha256=sha(data),size=len(data),batch_sha256=result['batch_sha256'],group_id=result['group_id']))
        return rows

    def restore(self,manifest,miner,state_path):
        self.path=Path(state_path) if state_path is not None else None
        self.binding=dict(manifest_sha256=digest(manifest),scope_sha256=scope(manifest,miner),miner=miner)
        self.previous=[]
        if self.path is None or not self.path.exists():return []
        need(not self.path.is_symlink() and self.path.stat().st_size<=limits(manifest).commitment_bytes,'private prepared group index')
        value=loads(self.path.read_bytes(),limits(manifest).commitment_bytes)
        need(type(value) is dict and set(value)=={'version','binding','groups'} and value['version']==PREPARED
             and value['binding']==self.binding and type(value['groups']) is list,'prepared complete-group state binding')
        bound=limits(manifest);need(len(value['groups'])<=manifest['max_batches'],'prepared signed cap')
        need(all(type(x) is dict and set(x)=={'sha256','size','batch_sha256','group_id'}
                 and type(x['size']) is int and 0<x['size']<=bound.compressed_bytes for x in value['groups'])
             and sum(x['size'] for x in value['groups'])<=bound.cumulative_bytes,'prepared aggregate bytes before decode')
        directory=Path(str(self.path)+'.complete-groups');need(not directory.is_symlink(),'private prepared directory')
        packed=[]
        for row in value['groups']:
            from .complete_group_admission import is_digest
            need(all(is_digest(row[k]) for k in ('sha256','batch_sha256','group_id')),'prepared digest fields')
            p=directory/(row['sha256']+'.zip');need(not p.is_symlink() and p.stat().st_size==row['size'],'prepared object path/size')
            data=p.read_bytes();need(sha(data)==row['sha256'],'prepared full artifact SHA256')
            batch,_=decode(data,bound)[0];packed.append((batch,data))
        actual=self._all(packed,manifest,miner);need(actual==value['groups'],'prepared original metadata hashes')
        self.previous=actual
        return packed

    def prepare(self,batch,arrays,manifest):
        self._check(batch,arrays,manifest,batch['miner'])
        from .batches import compression_for_manifest
        return encode([(batch,arrays)],limits(manifest),level=compression_for_manifest(manifest))

    def persist(self,manifest,miner,state_path,prepared):
        rows=self._all(prepared,manifest,miner)
        need(len(rows)>=len(self.previous) and rows[:len(self.previous)]==self.previous,'prepared slots are append-only')
        need((Path(state_path) if state_path is not None else None)==self.path,'bound prepared state path')
        if self.path is not None:
            directory=Path(str(self.path)+'.complete-groups');need(not directory.is_symlink(),'private prepared directory')
            directory.mkdir(parents=True,exist_ok=True,mode=0o700);directory.chmod(0o700)
            for row,(_,data) in zip(rows,prepared):
                p=directory/(row['sha256']+'.zip');need(not p.is_symlink(),'private prepared artifact path')
                if p.exists():need(p.read_bytes()==data,'immutable complete-group artifact')
                else:_write(p,data)
            _write(self.path,canonical(dict(version=PREPARED,binding=self.binding,groups=rows)))
        self.previous=rows

    def _remaining(self,manifest):
        from .miner import EpochClosed
        remaining=manifest['deadline']-time.time()
        if remaining<=0:raise EpochClosed('signed complete-group upload deadline')
        return min(120,remaining)

    def _confirm_existing(self,url,body,manifest):
        # Only an uncertain immutable PUT needs a full GET. Content-Length is a
        # bound, not authenticity; exact bytes/digest must match before an ack.
        response=self.http.get(url,stream=True,timeout=self._remaining(manifest),allow_redirects=False)
        try:
            need(response.status_code==200,'immutable retry GET not acknowledged')
            need(response.headers.get('Content-Length')==str(len(body)),'immutable retry object size differs')
            import hashlib
            actual=hashlib.sha256();size=0
            for chunk in response.iter_content(chunk_size=1024*1024):
                size+=len(chunk);need(size<=len(body),'immutable retry body exceeds original bound');actual.update(chunk)
            need(size==len(body) and actual.hexdigest()==sha(body),'immutable retry original bytes differ')
        finally:response.close()

    def _put(self,url,body,manifest,*,read_url=None):
        headers={'Content-Type':'application/octet-stream'}
        if read_url is not None:headers['If-None-Match']='*'
        response=self.http.put(url,data=body,headers=headers,
                               timeout=self._remaining(manifest),allow_redirects=False)
        if response.status_code==412 and read_url is not None:
            self._confirm_existing(read_url,body,manifest)
            return
        need(response.status_code in (200,201,204),'group PUT not acknowledged; redirects forbidden')
        response.raise_for_status()

    def upload(self,identity,manifest,capability,prepared):
        validate_capability(capability,manifest,identity.id)
        rows=self._all(prepared,manifest,identity.id)
        need(rows==self.previous,'upload only original persisted prepared groups')
        commitment=signed_commitment(identity,manifest,prepared)
        if self.path is None:
            return self._upload(identity,manifest,capability,prepared,commitment,None)
        with _lock(Path(str(self.path)+'.upload.lock')):
            return self._upload(identity,manifest,capability,prepared,commitment,Path(str(self.path)+'.uploads.json'))

    def _upload(self,identity,manifest,capability,prepared,commitment,journal_path):
        if journal_path is not None and journal_path.exists():
            need(not journal_path.is_symlink() and journal_path.stat().st_size<=limits(manifest).commitment_bytes,'bounded private upload journal')
            journal=loads(journal_path.read_bytes(),limits(manifest).commitment_bytes)
            need(type(journal) is dict and set(journal)=={'version','binding','slots'} and journal['version']==JOURNAL
                 and journal['binding']==self.binding and type(journal['slots']) is dict,'upload journal binding')
        else:journal=dict(version=JOURNAL,binding=self.binding,slots=dict(self.memory_journal))
        allowed={kind+str(i) for i in range(len(prepared)) for kind in ('artifact-','document-')}
        need(set(journal['slots'])<=allowed,'upload journal references unknown slot')
        for slot,(batch,data) in enumerate(prepared):
            for kind,body,url,read_url in [('artifact-',data,capability['batch_put_urls'][slot],capability['batch_get_urls'][slot]),
                ('document-',document(batch,manifest,identity.id,slot),capability['training_put_urls'][slot],capability['training_get_urls'][slot])]:
                key=kind+str(slot);fingerprint=sha(body);old=journal['slots'].get(key)
                if old is not None:
                    need(old==fingerprint,'acknowledged group bytes cannot change');continue
                self._put(url,body,manifest,read_url=read_url);journal['slots'][key]=fingerprint
                if journal_path is not None:_write(journal_path,canonical(journal))
                self.memory_journal=dict(journal['slots'])
        self._put(capability['put_url'],commitment,manifest)
        return 200


def open_epoch(gateway,manifest,registered_miners):
    """Explicit new-version Gateway entry point, no legacy format reinterpretation."""
    need(gateway.direct_r2 is True,'complete-group R2 gateway required')
    epoch=manifest['epoch'];registered=set(registered_miners)
    with gateway.lock:
        need(epoch not in gateway.epochs,'epoch already exists')
        caps={miner:capability(gateway.bucket,manifest,miner,registered_miners=registered) for miner in sorted(registered)}
        gateway.epochs[epoch]=dict(closed=False,miners=registered,uploads={},transport='direct-r2-v1',
            start=manifest['start'],deadline=manifest['deadline'],max_batches=manifest['max_batches'],
            commitment_binding=dict(version=TRANSPORT,scope='per-miner-complete-group-v1'))
        gateway.persist()
    return caps
