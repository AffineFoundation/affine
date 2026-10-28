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
# v8.2.3: per-identity upload URLs leave the signed manifest. The manifest keeps
# one small sealed pointer per identity (constant size, independent of
# max_batches) that binds a presigned GET for the identity's sealed grant object
# and its exact SHA-256/size. The grant is the unchanged v1 capability.
GRANT_POLICY='r2-upload-grant-v1'
GRANT_FIELDS=frozenset({'transport','scope_sha256','deadline','grant_url','grant_sha256','grant_bytes'})
MAX_GRANT_BYTES=8*1024**2


def paths(manifest,miner,slot=None):
    epoch=manifest['epoch']
    need(type(epoch) is str and epoch and all(c.isalnum() or c in '-_.' for c in epoch)
         and '..' not in epoch,'safe canonical epoch object namespace')
    if slot is None:return f'private/{epoch}/commitments/{miner}.json'
    return (f'private/{epoch}/staging/{miner}/{slot}.zip',f'private/{epoch}/training/{miner}/{slot}.json')


def delivery(manifest):
    value=manifest.get('capability_delivery')
    need(value is None or value=={'version':GRANT_POLICY},'known signed capability delivery policy')
    return None if value is None else GRANT_POLICY


def grant_key(manifest,miner,grant_sha256):
    epoch=paths(manifest,miner).split('/')[1]
    need(type(grant_sha256) is str and len(grant_sha256)==64 and all(c in '0123456789abcdef' for c in grant_sha256),'grant digest')
    need(type(miner) is str and len(miner)==64 and all(c in '0123456789abcdef' for c in miner),'grant identity')
    return f'private/{epoch}/grants/{miner}/{grant_sha256}.json'


def _put_immutable(bucket,key,data):
    from botocore.exceptions import ClientError
    try:
        bucket.client.put_object(Bucket=bucket.name,Key=key,Body=data,ContentType='application/octet-stream',IfNoneMatch='*')
    except ClientError as error:
        if error.response.get('ResponseMetadata',{}).get('HTTPStatusCode')!=412:raise
        # Content-addressed key: an existing object must already be these exact bytes.
        body=bucket.client.get_object(Bucket=bucket.name,Key=key)['Body'].read(len(data)+1)
        need(body==data,'immutable grant collision')


_GRANT_CACHE={}
MINT_WORKERS=32
MINT_PARALLEL_MIN=64


def _pointer(bucket,manifest,miner,sealed):
    blob=sealed.encode()
    need(len(blob)<=MAX_GRANT_BYTES,'bounded sealed upload grant')
    grant_sha256=sha(blob);key=grant_key(manifest,miner,grant_sha256)
    _put_immutable(bucket,key,blob)
    remaining=int(manifest['deadline']-time.time());need(remaining>0,'epoch already closed')
    pointer=dict(transport=GRANT_POLICY,scope_sha256=scope(manifest,miner),deadline=manifest['deadline'],
        grant_url=bucket.presign(key,'get_object',remaining),grant_sha256=grant_sha256,grant_bytes=len(blob))
    validate_pointer(pointer,manifest,miner)
    return encrypt(miner,pointer)


def _mint_parallel(bucket,manifest,miners,workers):
    """Sealed v1 grants for many identities: presign+seal in isolated subprocesses
    (CPU bound, no fork of the multithreaded host), immutable PUTs in threads.
    Credentials travel only over the child's stdin pipe."""
    import json,subprocess,sys
    from concurrent.futures import ThreadPoolExecutor
    root=str(Path(__file__).resolve().parents[1]);miners=sorted(miners)
    chunks=[miners[i::workers] for i in range(workers) if miners[i::workers]]
    request=lambda chunk:json.dumps(dict(options=bucket._client_options,bucket=bucket.name,manifest=manifest,miners=chunk,
        registered=miners)).encode()
    code='import sys;sys.path.insert(0,sys.argv[1]);from subnet.complete_group_transport import _mint_worker;_mint_worker()'
    procs=[subprocess.Popen([sys.executable,'-B','-c',code,root],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,cwd=root) for _ in chunks]
    sealed={}
    try:
        def run(pair):
            proc,chunk=pair;out,err=proc.communicate(request(chunk),timeout=600)
            need(proc.returncode==0,'grant mint worker failed: '+err.decode(errors='replace')[-300:])
            value=json.loads(out);need(sorted(value)==sorted(chunk),'grant mint worker identity set');return value
        with ThreadPoolExecutor(len(procs)) as pool:
            for value in pool.map(run,zip(procs,chunks)):sealed.update(value)
    finally:
        for proc in procs:
            if proc.poll() is None:proc.kill()
    with ThreadPoolExecutor(32) as pool:
        pointers=dict(zip(miners,pool.map(lambda m:_pointer(bucket,manifest,m,sealed[m]),miners)))
    return pointers


def _mint_worker():
    import json,sys,boto3
    value=json.loads(sys.stdin.buffer.read())
    class _Bucket:
        def __init__(self):
            self.name=value['bucket'];self.client=boto3.client('s3',**value['options'])
        def presign(self,key,operation='get_object',expires=604800):
            need(1<=expires<=604800,'R2 URL expiration');params=dict(Bucket=self.name,Key=key)
            if operation=='put_object':params['ContentType']='application/octet-stream'
            return self.client.generate_presigned_url(operation,Params=params,ExpiresIn=expires)
    b=_Bucket();registered=set(value['registered'])
    out={m:_capability(b,value['manifest'],m,registered_miners=registered) for m in value['miners']}
    sys.stdout.write(json.dumps(out));sys.stdout.flush()


def capability(bucket,manifest,miner,*,registered_miners):
    if delivery(manifest) is None:
        return _capability(bucket,manifest,miner,registered_miners=registered_miners)
    need(miner in registered_miners,'registered epoch miner')
    if (len(registered_miners)>=MINT_PARALLEL_MIN and MINT_WORKERS>1
            and isinstance(getattr(bucket,'_client_options',None),dict)):
        key=(manifest['epoch'],manifest['deadline'],frozenset(registered_miners))
        hit=_GRANT_CACHE.get(key)
        if hit is None or not (manifest is hit['manifest'] or digest(manifest)==hit['digest']):
            _GRANT_CACHE.clear()  # one opening at a time; never reuse another epoch's pointers
            hit=dict(manifest=manifest,digest=digest(manifest),
                     pointers=_mint_parallel(bucket,manifest,registered_miners,MINT_WORKERS))
            _GRANT_CACHE[key]=hit
        return hit['pointers'][miner]
    return _pointer(bucket,manifest,miner,_capability(bucket,manifest,miner,registered_miners=registered_miners))


def validate_pointer(pointer,manifest,miner):
    from .client import direct_r2_url
    need(delivery(manifest)==GRANT_POLICY,'grant pointer requires signed grant delivery')
    need(type(pointer) is dict and set(pointer)==GRANT_FIELDS and pointer['transport']==GRANT_POLICY
         and pointer['scope_sha256']==scope(manifest,miner) and type(pointer['deadline']) is int
         and pointer['deadline']==manifest['deadline'] and type(pointer['grant_bytes']) is int
         and 0<pointer['grant_bytes']<=MAX_GRANT_BYTES,'exact epoch-scoped grant pointer')
    key=grant_key(manifest,miner,pointer['grant_sha256'])
    url=direct_r2_url(pointer['grant_url']);parsed=urlparse(url)
    need(unquote(parsed.path).endswith('/'+key) and not parsed.fragment,'grant object path binding')
    return pointer


def resolve_capability(identity,manifest,*,http=None,attempts=4,sleep=time.sleep):
    """Miner side: the decrypted v1 capability for this identity and epoch.

    Inline (legacy) delivery decrypts the manifest entry. Grant delivery
    decrypts the signed pointer, fetches exactly grant_bytes, checks the
    pointer-bound SHA-256, then decrypts and validates the v1 capability."""
    sealed=manifest['capabilities'][identity.id]
    if delivery(manifest) is None:return identity.decrypt(sealed)
    pointer=validate_pointer(identity.decrypt(sealed),manifest,identity.id)
    http=http if http is not None else requests;error=None
    for attempt in range(attempts):
        try:
            response=http.get(pointer['grant_url'],timeout=60,stream=True)
            response.raise_for_status()
            blob=b''
            for chunk in response.iter_content(65536):
                blob+=chunk;need(len(blob)<=pointer['grant_bytes'],'grant larger than signed size')
            need(len(blob)==pointer['grant_bytes'] and sha(blob)==pointer['grant_sha256'],'grant bytes differ from signed digest')
            break
        except ValueError:raise
        except Exception as exc:  # transient transport errors only
            error=exc;sleep(min(8,2**attempt))
    else:raise RuntimeError('upload grant unavailable: '+type(error).__name__)
    cap=identity.decrypt(blob.decode())
    validate_capability(cap,manifest,identity.id)
    return cap


def _capability(bucket,manifest,miner,*,registered_miners):
    need(miner in registered_miners,'registered epoch miner')
    contract,_=context(manifest,miner);remaining=int(manifest['deadline']-time.time())
    need(remaining>0,'epoch already closed')
    cap=dict(transport=TRANSPORT,scope_sha256=scope(manifest,miner),deadline=manifest['deadline'],
        headers={'Content-Type':'application/octet-stream'},
        put_url=bucket.presign(paths(manifest,miner),'put_object',remaining),
        upload_headers={'Content-Type':'application/octet-stream','If-None-Match':'*'},
        batch_put_urls=[],training_put_urls=[],batch_get_urls=[],training_get_urls=[])
    from .batch_lifetime import upload_slots
    for slot in range(upload_slots(manifest)):
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
    from .batch_lifetime import upload_slots
    slots=upload_slots(manifest)
    need(type(cap) is dict and set(cap)=={'transport','scope_sha256','deadline','headers','upload_headers','put_url','batch_put_urls','training_put_urls','batch_get_urls','training_get_urls'}
         and cap['transport']==TRANSPORT and cap['scope_sha256']==scope(manifest,miner)
         and type(cap['deadline']) is int and cap['deadline']==manifest['deadline']
         and cap['headers']=={'Content-Type':'application/octet-stream'}
         and cap['upload_headers']=={'Content-Type':'application/octet-stream','If-None-Match':'*'},'exact epoch-scoped group capability')
    for name in ('batch_put_urls','training_put_urls','batch_get_urls','training_get_urls'):
        need(type(cap[name]) is list and len(cap[name])==slots,'exact signed capability slot count')
    urls=[(cap['put_url'],paths(manifest,miner),'commit')]
    for slot in range(slots):
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
        from .batch_lifetime import policy as lifetime_policy
        if lifetime_policy(manifest) is None:
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
        from .batch_lifetime import policy as lifetime_policy
        # Replace-until-seal signs the commitment after write-once slots are assigned.
        commitment=None if lifetime_policy(manifest) is not None else signed_commitment(identity,manifest,prepared)
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
        if commitment is None:
            return self._replaceable(identity,manifest,capability,prepared,journal,journal_path)
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

    def _replaceable(self,identity,manifest,capability,prepared,journal,journal_path):
        """v8.2.3 replace-until-seal: groups live in write-once slots; the commitment names the set.

        A group already uploaded this epoch keeps its slot. A new group takes the
        lowest never-used slot. The replaceable commitment is written last, so a
        crash leaves the previous commitment in force."""
        from .batch_lifetime import upload_slots
        total=upload_slots(manifest)
        allowed={kind+str(i) for i in range(total) for kind in ('artifact-','document-')}
        need(set(journal['slots'])<=allowed,'upload journal references unknown slot')
        def save():
            if journal_path is not None:_write(journal_path,canonical(journal))
            self.memory_journal=dict(journal['slots'])
        def used():
            return {int(k.split('-',1)[1]) for k in journal['slots']}
        by_artifact={v:int(k.split('-',1)[1]) for k,v in journal['slots'].items() if k.startswith('artifact-') and v!='occupied'}
        placed=[]
        for batch,data in prepared:
            slot=by_artifact.get(sha(data))
            while True:
                if slot is None:
                    free=[i for i in range(total) if i not in used() and i not in {s for s,_ in placed}]
                    need(bool(free),'all write-once upload slots for this epoch are used; the current commitment stays in force')
                    slot=free[0]
                try:
                    for kind,body,url,read_url in [('artifact-',data,capability['batch_put_urls'][slot],capability['batch_get_urls'][slot]),
                        ('document-',document(batch,manifest,identity.id,slot),capability['training_put_urls'][slot],capability['training_get_urls'][slot])]:
                        key=kind+str(slot);fingerprint=sha(body);old=journal['slots'].get(key)
                        if old is not None:
                            need(old==fingerprint,'acknowledged group bytes cannot change');continue
                        self._put(url,body,manifest,read_url=read_url);journal['slots'][key]=fingerprint;save()
                    break
                except ValueError as error:
                    if 'immutable retry' not in str(error) and 'acknowledged group bytes' not in str(error):raise
                    # Slot holds other bytes (e.g. an unjournaled earlier upload): burn it.
                    for kind in ('artifact-','document-'):journal['slots'].setdefault(kind+str(slot),'occupied')
                    save();slot=None
            placed.append((slot,(batch,data)))
        placed.sort(key=lambda x:x[0])
        commitment=signed_commitment(identity,manifest,[p for _,p in placed],[s for s,_ in placed])
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
