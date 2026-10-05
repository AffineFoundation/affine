"""Small signed commitments, immutable snapshots and honest selected-only reports."""
import base64,hashlib,json,re,time
from nacl.signing import VerifyKey
VERSION='small-commitment-pairs-v1'
VERSION2='small-commitment-pairs-v2'
VERSIONS=(VERSION,VERSION2)
MAX_BYTES=65536
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
sha=lambda b:hashlib.sha256(b).hexdigest()
def need(v,msg):
 if not v:raise ValueError(msg)
def is_digest(v):return isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)is not None

def validate(data,epoch,miner,maximum=256):
 need(type(data)is bytes and 0<len(data)<=MAX_BYTES,'bounded commitment')
 envelope=json.loads(data);need(type(envelope)is dict and set(envelope)=={'payload','signer','signature'}and envelope['signer']==miner,'miner signature binding')
 need(data==canonical(envelope),'canonical commitment envelope encoding')
 p=envelope['payload'];VerifyKey(bytes.fromhex(miner)).verify(canonical(p),base64.b64decode(envelope['signature'],validate=True))
 need(type(p)is dict and set(p)=={'version','epoch','miner','checkpoint','source','batches'}and p['version']in VERSIONS and p['epoch']==epoch and p['miner']==miner,'commitment scope')
 need(is_digest(p['checkpoint'])and is_digest(p['source']),'model/source digest')
 need(type(p['batches'])is list and len(p['batches'])<=maximum,'commitment cap');seen=set();payloads=set()
 for i,b in enumerate(p['batches']):
  fields={'slot','env_id','index','batch_sha256','sha256','size'}
  if p['version']==VERSION2:fields|={'training_sha256','training_size'}
  need(type(b)is dict and set(b)==fields and type(b['slot'])is int and b['slot']==i,'ordered batch slots')
  if p['version']==VERSION2:need(is_digest(b['training_sha256'])and type(b['training_size'])is int and 0<b['training_size']<=2_000_000,'bounded token artifact')
  need(b['sha256']not in payloads,'distinct payload slots');payloads.add(b['sha256'])
  need(is_digest(b['sha256'])and is_digest(b['batch_sha256'])and type(b['size'])is int and 0<b['size']<=2_000_000_000,'artifact size/hash')
  need(type(b['env_id'])is str and 0<len(b['env_id'])<=100 and type(b['index'])is int and 0<=b['index']<2**31 and (b['env_id'],b['index'])not in seen,'unique declared index');seen.add((b['env_id'],b['index']))
 return envelope

def make(identity,manifest,packed):
 entries=[]
 for i,(batch,data)in enumerate(packed):entries.append(dict(slot=i,env_id=batch['env_id'],index=batch['index'],batch_sha256=sha(canonical(batch)),sha256=sha(data),size=len(data)))
 version=manifest.get('submission_transport_policy',VERSION);need(version in VERSIONS,'explicit transport version')
 if version==VERSION2:
  from .training_documents import document
  for entry,(batch,_)in zip(entries,packed):
   body=document(batch,manifest,identity.id,entry['slot']);entry.update(training_sha256=sha(body),training_size=len(body))
 p=dict(version=version,epoch=manifest['epoch'],miner=identity.id,checkpoint=manifest['checkpoint']['id'],source=manifest['source_bundle']['sha256'],batches=entries)
 return dict(payload=p,signer=identity.id,signature=base64.b64encode(identity.key.sign(canonical(p)).signature).decode())

def unchecked(manifest,receipt):
 from .forced_sampling import assurance
 from .auditing import assurance as audit_assurance
 return dict(epoch=manifest['epoch'],submission_sha256=receipt['sha256'],commitment_status='not_selected',policy=manifest['audit_policy'],audit_seed=manifest['audit_seed'],sampling_assurance=assurance(manifest),selected_batches=[],assurance=audit_assurance(len(receipt['artifacts']),0),outcomes=[dict(batch=i,env_id=b['env_id'],index=b['index'],valid=None,fully_audited=False,failure_kind='not_selected')for i,b in enumerate(receipt['artifacts'])],accepted=[],training_eligibility='fully-audited-only')

def combine(manifest,receipt,remote):
 """Merge original worker audits without inventing inference or queue evidence."""
 report=unchecked(manifest,receipt);report['commitment_status']='audited_subset';report['artifact_audits']=[];report['selected_batches']=[]
 for audit in remote['audits']:
  matches=[b for b in receipt['artifacts']if b['sha256']==audit['submission_sha256']];need(len(matches)==1,'original artifact receipt')
  b=matches[0];i=b['slot'];need(i not in report['selected_batches'],'unique artifact audit');report['selected_batches'].append(i)
  need(len(audit['accepted'])<=1,'one complete pair per artifact')
  if audit['accepted']:need(sha(canonical(audit['accepted'][0]))==b['batch_sha256'],'actual accepted batch commitment')
  if audit.get('outcomes'):
   need(len(audit['outcomes'])==1,'one batch outcome')
   need(audit['outcomes'][0].get('valid')is None or type(audit['outcomes'][0].get('valid'))is bool,'strict outcome boolean')
   report['outcomes'][i]=dict(audit['outcomes'][0],batch=i)
  report['accepted'].extend(audit['accepted']);report['artifact_audits'].append(dict(audit,remote_job_id=remote['job_id'],backend_profile=remote['backend_profile'],execution_resources_enforced=remote['execution_resources_enforced']))
 report['remote_job_id']=remote['job_id'];report['backend_profile']=remote['backend_profile'];report['execution_resources_enforced']=remote['execution_resources_enforced']
 from .auditing import assurance
 checked=sum(o.get('fully_audited')is True and (o.get('valid')is True or o.get('failure_kind')=='confirmed_invalid')for o in report['outcomes'])
 report['assurance']=assurance(len(receipt['artifacts']),checked)
 return report

def validate_unchecked(manifest,receipt,report):
 need(manifest.get('submission_transport_policy')in VERSIONS,'explicit commitment epoch')
 need(report==unchecked(manifest,receipt),'exact no-credit unaudited report')
 return True

def _read_small_commitment(bucket,key,client=None):
 """Worker reads only bounded commitment bytes, never state or heavy artifacts."""
 response=(bucket.client if client is None else client).get_object(Bucket=bucket.name,Key=key);body=response['Body']
 try:data=body.read(MAX_BYTES+1)
 finally:body.close()
 return dict(data=data,ETag=response['ETag'],LastModified=response['LastModified'])

def _bounded_small_reads(gateway,epoch,miners,cutoff):
 """At most four in-flight GETs; journals remain serial and deterministic."""
 from concurrent.futures import ThreadPoolExecutor
 from collections import deque
 remaining=iter(miners);inflight=deque()
 client=gateway.bucket.commitment_read_client()if cutoff is not None and hasattr(gateway.bucket,'commitment_read_client')else None
 try:
  with ThreadPoolExecutor(max_workers=4)as pool:
   def fill():
    while len(inflight)<4:
     if cutoff is not None and time.time()>=cutoff:return
     try:miner=next(remaining)
     except StopIteration:return
     key='private/'+epoch+'/commitments/'+miner+'.json'
     inflight.append((miner,pool.submit(_read_small_commitment,gateway.bucket,key,client)))
   fill()
   while inflight:
    item=inflight.popleft()
    yield item
    fill()
 finally:
  if client is not None:client.close()

class FreezeMetadataIncomplete(RuntimeError):
 """No finalized receipt set can be inferred from partial infrastructure reads."""

def freeze(gateway,epoch):
 """Capture ALL tiny commitments first; fair <=4 copies, serial durable journals."""
 import secrets
 from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
 from .storage import SubmissionPolicyError
 from botocore.exceptions import ClientError
 state=gateway.epochs[epoch];state['closed']=True;gateway.persist()
 if 'frozen_receipts'in state:
  gateway.bucket.json('public/'+epoch+'/receipts.json',state['frozen_receipts']);return state['frozen_receipts']
 snapshots=state.setdefault('commitment_snapshots',{});pending=state.setdefault('commitment_pending',{});rejections=state.setdefault('rejections',{});failures=[]
 cutoff=state['commitment_binding'].get('freeze_until')
 def expired():return cutoff is not None and time.time()>=cutoff
 def missing(exc):return isinstance(exc,ClientError)and str(exc.response.get('Error',{}).get('Code'))in('NoSuchKey','404','NotFound')
 if not state.get('commitment_capture_complete'):
  allminers=sorted(state['miners']);discovery=state.get('commitment_discovery')
  if discovery is None:
   try:
    if expired():raise TimeoutError('tiny commitment discovery deadline')
    listed=gateway.bucket.complete_commitment_listing(epoch,allminers,cutoff)if hasattr(gateway.bucket,'complete_commitment_listing')else allminers
    need(type(listed)is list and len(listed)==len(set(listed))and set(listed)<=set(allminers),'exact activated commitment discovery')
    discovery=dict(miners=listed,at=time.time(),complete=True,method='complete-prefix-list-advisory'if hasattr(gateway.bucket,'complete_commitment_listing')else'bounded-GET-all')
    state['commitment_discovery']=discovery;gateway.persist()
   except Exception as exc:
    state['commitment_metadata_incomplete']=dict(reason='discovery_infrastructure_incomplete',at=time.time(),error_type=type(exc).__name__);gateway.persist()
    raise FreezeMetadataIncomplete('complete tiny commitment discovery unavailable')from exc
  remaining=[m for m in discovery['miners']if m not in pending and m not in snapshots and m not in rejections]
  reads=_bounded_small_reads(gateway,epoch,remaining,cutoff);metadata_failures=[]
  try:
   for miner,future in reads:
    try:
     r=future.result();data=r['data']
     if not state['start']<=r['LastModified'].timestamp()<state['deadline']:raise SubmissionPolicyError('commitment time')
     try:
      env=validate(data,epoch,miner,state['max_batches'])
      need(env['payload']['version']==state['commitment_binding'].get('version',VERSION),'explicit epoch transport')
      need(env['payload']['checkpoint']==state['commitment_binding']['checkpoint']and env['payload']['source']==state['commitment_binding']['source'],'committed source/model')
     except Exception as exc:raise SubmissionPolicyError('malformed commitment')from exc
     digest=sha(data);root='public/'+epoch+'/submissions/'+miner+'/'+digest
     pending[miner]=dict(key='private/'+epoch+'/commitments/'+miner+'.json',etag=r['ETag'],document=env,sha256=digest,size=len(data),received_at=r['LastModified'].timestamp(),root=root,artifacts=[],artifact_plans={},commitment_copied=False);gateway.persist()
    except SubmissionPolicyError as exc:rejections[miner]=str(exc);gateway.persist()
    except Exception as exc:
     if missing(exc):rejections[miner]='missing completed commitment/artifact';gateway.persist()
     else:metadata_failures.append(exc)
  finally:reads.close()
  unresolved=[m for m in remaining if m not in pending and m not in snapshots and m not in rejections]
  if metadata_failures or unresolved:
   state['commitment_metadata_incomplete']=dict(reason='tiny_GET_infrastructure_incomplete',at=time.time(),unresolved=unresolved,error_types=[type(x).__name__ for x in metadata_failures]);gateway.persist()
   raise FreezeMetadataIncomplete('complete tiny commitment admission unavailable')from(metadata_failures[0]if metadata_failures else None)
  state['commitment_capture_complete']=True;state['commitment_capture_completed_at']=time.time();state.pop('commitment_metadata_incomplete',None);gateway.persist()
 if state['commitment_binding'].get('version')==VERSION2:
  from .training_documents import capture
  capture(gateway,epoch)
  from .training_documents import freeze_receipts
  return freeze_receipts(gateway,epoch)
 if state['commitment_binding'].get('proof_copy_policy') is not None:
  from .selected_proof_copy import freeze_metadata
  return freeze_metadata(gateway,epoch)
 # The unpredictable order is persisted once AFTER every tiny document was
 # authenticated. Recovery never draws again or rereads a mutable commitment.
 if 'commitment_copy_order'not in state:
  seed=secrets.token_hex(32);state['commitment_copy_order_seed']=seed
  state['commitment_copy_order']=sorted(pending,key=lambda m:sha(bytes.fromhex(seed)+bytes.fromhex(m)));gateway.persist()
 def plan(miner):
  progress=pending[miner]
  for b in progress['document']['payload']['batches']:
   if str(b['slot'])in progress['artifact_plans']:continue
   if expired():raise TimeoutError('heavy copy planning cutoff')
   staging='private/'+epoch+'/staging/'+miner+'/'+str(b['slot'])+'.zip'
   meta=gateway.bucket.client.head_object(Bucket=gateway.bucket.name,Key=staging)
   if b['size']>state.get('upload_limit',100_000_000)or meta['ContentLength']!=b['size']or not state['start']<=meta['LastModified'].timestamp()<state['deadline']:raise SubmissionPolicyError('artifact size/time')
   progress['artifact_plans'][str(b['slot'])]=dict(etag=meta['ETag'],received_at=meta['LastModified'].timestamp());gateway.persist()
 def copy_one(miner):
  progress=pending[miner];root=progress['root'];copied=progress['commitment_copied'];artifacts=[]
  try:
   if not copied:
    gateway.bucket.copy(progress['key'],root+'/commitment.json',expected_etag=progress['etag']);copied=True
   for b in progress['document']['payload']['batches']:
    slot=b['slot']
    if any(x['slot']==slot for x in progress['artifacts']):continue
    if expired():raise TimeoutError('heavy copy launch cutoff')
    staging='private/'+epoch+'/staging/'+miner+'/'+str(slot)+'.zip';frozen=root+'/'+str(slot)+'.zip';ap=progress['artifact_plans'][str(slot)]
    gateway.bucket.copy(staging,frozen,expected_etag=ap['etag'])
    artifacts.append(dict(b,key=staging,frozen_key=frozen,etag=ap['etag'],received_at=ap['received_at'],read_url=gateway.bucket.presign(frozen)))
   return miner,copied,artifacts,None
  except Exception as exc:return miner,copied,artifacts,exc
 def finish(result):
  miner,copied,artifacts,exc=result;progress=pending[miner];progress['commitment_copied']=copied;progress['artifacts'].extend(artifacts);gateway.persist()
  if exc is not None:
   if isinstance(exc,SubmissionPolicyError)or missing(exc):rejections[miner]=str(exc);gateway.persist()
   else:failures.append(exc)
   return
  snapshots[miner]=dict(sha256=progress['sha256'],commitment_document=progress['document'],commitment_key=progress['root']+'/commitment.json',artifacts=progress['artifacts'],size=progress['size'],received_at=progress['received_at'],hash_assurance='declared-payload-hashes-until-selected-verifier');gateway.persist()
 candidates=iter(m for m in state['commitment_copy_order']if m not in snapshots and m not in rejections)
 with ThreadPoolExecutor(max_workers=4)as pool:
  active={};exhausted=False
  while active or not exhausted:
   while len(active)<4 and not exhausted and not expired():
    try:miner=next(candidates)
    except StopIteration:exhausted=True;break
    try:plan(miner)
    except Exception as exc:
     if isinstance(exc,SubmissionPolicyError)or missing(exc):rejections[miner]=str(exc);gateway.persist()
     else:failures.append(exc)
     continue
    if expired():break
    active[pool.submit(copy_one,miner)]=miner
   if not active:break
   done,_=wait(active,return_when=FIRST_COMPLETED)
   for future in done:finish(future.result());active.pop(future)
 if failures and not expired():raise failures[0]
 for miner in pending:
  if miner not in snapshots and miner not in rejections:state.setdefault('commitment_deferred',{})[miner]=dict(reason='freeze_infrastructure_budget_deferred',closed_at=time.time())
 if state['commitment_binding'].get('version')==VERSION2:
  from .training_documents import attach
  attach(state,snapshots)
 state['frozen_receipts']=dict(snapshots);gateway.persist();gateway.bucket.json('public/'+epoch+'/receipts.json',state['frozen_receipts']);return state['frozen_receipts']


def deferred(manifest,receipt,reason,at):
 from .hourly_policy import cutoff
 need(reason in ('budget_deferred','infrastructure_deferred'),'typed audit deferral')
 need(cutoff(manifest,'audit')is not None,'signed deferred audit policy')
 need(type(at)in(int,float)and at>=manifest['deadline'],'deferral time')
 if reason=='budget_deferred':need(at>=cutoff(manifest,'audit'),'actual audit cutoff elapsed')
 report=unchecked(manifest,receipt);report['commitment_status']=reason;report['deferred_at']=at
 for outcome in report['outcomes']:outcome['failure_kind']=reason
 return report

def validate_deferred(manifest,receipt,report):
 need(report==deferred(manifest,receipt,report.get('commitment_status'),report.get('deferred_at')),'exact no-credit deferred report')
 return True


def pair_artifact(batch,arrays,manifest):
 """Stable lossless ZIP; absent policy preserves exact historical default bytes."""
 from .batches import pack,compression_for_manifest
 from .artifact_budget import for_manifest
 return pack([(batch,arrays)],budget=for_manifest(manifest),stable=True,compression_level=compression_for_manifest(manifest))


class UploadJournal:
 """Append-only acknowledged slot hashes, durable across owned/client restart."""
 def __init__(self,manifest,path=None):
  from pathlib import Path
  self.path=Path(path)if path else None;self.binding=dict(epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],source=manifest['source_bundle']['sha256']);self.slots={}
  if self.path and self.path.exists():
   value=json.loads(self.path.read_text());need(value.get('binding')==self.binding and type(value.get('slots'))is dict,'local upload journal binding');self.slots=value['slots']
 def known(self,slot,data):
  old=self.slots.get(str(slot))
  if old is None:return False
  need(old==sha(data),'acknowledged pair slots are append-only')
  return True
 def acknowledge(self,slot,data):
  self.slots[str(slot)]=sha(data)
  if self.path:
   self.path.parent.mkdir(parents=True,exist_ok=True);tmp=self.path.with_suffix('.tmp');tmp.write_bytes(canonical(dict(binding=self.binding,slots=self.slots)));tmp.chmod(0o600);tmp.replace(self.path)

def check_prepared_cumulative(packed,manifest,maximum):
    # Sum independent archive sizes/raw framing conservatively: duplicated
    # manifests count toward the SAME historical cumulative caps.
    import io,zipfile,zlib
    from .artifact_budget import for_manifest
    from .batches import UploadBudgetExceeded
    budget=for_manifest(manifest);raw=0;compressed=0;array_raw=0;array_compressed=0;framing=22;records=[]
    if len(packed)>maximum:raise ValueError('owned commitment slot cap')
    for slot,(batch,artifact) in enumerate(packed):
        compressed+=len(artifact)
        with zipfile.ZipFile(io.BytesIO(artifact))as archive:
            raw+=sum(e.file_size for e in archive.infolist())
            if len(archive.infolist())>4096 or archive.getinfo('manifest.json').file_size>2_000_000:raise ValueError('prepared archive metadata budget')
            if raw>budget['raw_bytes']or compressed>budget['compressed_bytes']:raise UploadBudgetExceeded('prepared pairs exceed cumulative artifact budget')
            record=json.loads(archive.read('manifest.json'))
            if len(record)!=1 or record[0]['batch']!=batch:raise ValueError('prepared batch metadata')
            row=record[0];refs=[]
            for turns in row['arrays']:
                renamed=[]
                for name in turns:
                    info=archive.getinfo(name);target=str(slot)+name[name.index('-'):]
                    array_raw+=info.file_size;array_compressed+=info.compress_size
                    framing+=76+2*len(target.encode());renamed.append(target)
                refs.append(renamed)
            records.append(dict(batch=batch,arrays=refs))
    # Array DEFLATE bytes do not depend on member names. Account for exact
    # hypothetical cumulative ZIP framing and combined canonical manifest,
    # including two-digit slot prefixes, without decoding model arrays.
    from .batches import compression_for_manifest
    manifest_bytes=canonical(records);codec=zlib.compressobj(level=compression_for_manifest(manifest),wbits=-15)
    manifest_compressed=codec.compress(manifest_bytes)+codec.flush()
    cumulative_raw=array_raw+len(manifest_bytes)
    cumulative_compressed=array_compressed+framing+76+2*len('manifest.json')+len(manifest_compressed)
    raw=max(raw,cumulative_raw);compressed=max(compressed,cumulative_compressed)
    if raw>budget['raw_bytes']or compressed>budget['compressed_bytes']:raise UploadBudgetExceeded('prepared pairs exceed cumulative artifact budget')

PREPARED_STATE='prepared-miner-pairs-v1'

def write_prepared_state(path,manifest,packed):
 """Persist immutable actual pair bytes, then atomically replace private index."""
 import os
 from pathlib import Path
 check_prepared_cumulative(packed,manifest,manifest['max_batches'])
 path=Path(path);directory=path.with_name(path.name+'.pairs')
 need(not path.is_symlink()and not directory.is_symlink(),'private prepared state path')
 directory.mkdir(parents=True,exist_ok=True,mode=0o700);directory.chmod(0o700)
 rows=[]
 for batch,data in packed:
  digest=sha(data);target=directory/(digest+'.zip');need(not target.is_symlink(),'private prepared artifact path')
  if target.exists():need(target.read_bytes()==data,'immutable prepared local artifact')
  else:
   temporary=directory/(digest+'.tmp-'+str(os.getpid()));need(not temporary.is_symlink(),'private prepared temporary path')
   with temporary.open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
   temporary.chmod(0o600);temporary.replace(target)
  rows.append(dict(sha256=digest,size=len(data),batch_sha256=sha(canonical(batch))))
 value=dict(version=PREPARED_STATE,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],source=manifest['source_bundle']['sha256'],sampling_contract_sha256=sha(canonical(manifest.get('sampling_contract'))),pairs=rows)
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()));need(not temporary.is_symlink(),'private prepared index temporary')
 with temporary.open('xb')as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
 temporary.chmod(0o600);temporary.replace(path)

def read_prepared_state(path,manifest):
 """Resume exact original bytes after full hash/admission; no recompression."""
 import io,zipfile
 from pathlib import Path
 from .batches import unpack
 from .artifact_budget import for_manifest
 path=Path(path);need(not path.is_symlink()and path.stat().st_size<=65536,'private prepared state path');value=json.loads(path.read_bytes())
 need(type(value)is dict and set(value)=={'version','epoch','checkpoint','source','sampling_contract_sha256','pairs'}and value['version']==PREPARED_STATE,'prepared state version')
 need(value['epoch']==manifest['epoch']and value['checkpoint']==manifest['checkpoint']['id']and value['source']==manifest['source_bundle']['sha256']and value['sampling_contract_sha256']==sha(canonical(manifest.get('sampling_contract'))),'stale local prepared miner state')
 need(type(value['pairs'])is list and len(value['pairs'])<=manifest['max_batches'],'prepared state slot cap')
 directory=path.with_name(path.name+'.pairs');need(not directory.is_symlink(),'private prepared artifact directory');packed=[]
 need(all(type(r)is dict and type(r.get('size'))is int and r['size']>0 for r in value['pairs'])and sum(r['size']for r in value['pairs'])<=for_manifest(manifest)['compressed_bytes'],'prepared aggregate compressed cap')
 for row in value['pairs']:
  need(type(row)is dict and set(row)=={'sha256','size','batch_sha256'}and is_digest(row['sha256'])and is_digest(row['batch_sha256'])and type(row['size'])is int and 0<row['size']<=for_manifest(manifest)['compressed_bytes'],'prepared state integrity fields')
  target=directory/(row['sha256']+'.zip');need(not target.is_symlink()and target.stat().st_size==row['size'],'prepared local artifact size/path')
  data=target.read_bytes();need(sha(data)==row['sha256'],'prepared local artifact full SHA256')
  with zipfile.ZipFile(io.BytesIO(data))as archive:
   need(archive.getinfo('manifest.json').file_size<=2_000_000,'prepared archive metadata budget');preview=json.loads(archive.read('manifest.json'))
  need(type(preview)is list and len(preview)==1,'prepared pair single batch');batch=preview[0]['batch']
  check_prepared_cumulative(packed+[(batch,data)],manifest,manifest['max_batches'])
  records=unpack(data,budget=for_manifest(manifest));need(len(records)==1,'prepared pair single batch')
  batch,arrays=records[0];need(sha(canonical(batch))==row['batch_sha256']and batch['epoch']==manifest['epoch']and batch['checkpoint']==manifest['checkpoint']['id'],'prepared local batch binding')
  packed.append((batch,data));del arrays,records
 check_prepared_cumulative(packed,manifest,manifest['max_batches'])
 return packed
