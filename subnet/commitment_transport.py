"""Small signed commitments, immutable snapshots and honest selected-only reports."""
import base64,hashlib,json,re,time
from nacl.signing import VerifyKey
VERSION='small-commitment-pairs-v1'
MAX_BYTES=65536
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
sha=lambda b:hashlib.sha256(b).hexdigest()
def need(v,msg):
 if not v:raise ValueError(msg)
def is_digest(v):return isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)is not None

def validate(data,epoch,miner,maximum=256):
 need(type(data)is bytes and 0<len(data)<=MAX_BYTES,'bounded commitment')
 envelope=json.loads(data);need(type(envelope)is dict and set(envelope)=={'payload','signer','signature'}and envelope['signer']==miner,'miner signature binding')
 p=envelope['payload'];VerifyKey(bytes.fromhex(miner)).verify(canonical(p),base64.b64decode(envelope['signature'],validate=True))
 need(type(p)is dict and set(p)=={'version','epoch','miner','checkpoint','source','batches'}and p['version']==VERSION and p['epoch']==epoch and p['miner']==miner,'commitment scope')
 need(is_digest(p['checkpoint'])and is_digest(p['source']),'model/source digest')
 need(type(p['batches'])is list and len(p['batches'])<=maximum,'commitment cap');seen=set();payloads=set()
 for i,b in enumerate(p['batches']):
  need(type(b)is dict and set(b)=={'slot','env_id','index','batch_sha256','sha256','size'}and type(b['slot'])is int and b['slot']==i,'ordered batch slots')
  need(b['sha256']not in payloads,'distinct payload slots');payloads.add(b['sha256'])
  need(is_digest(b['sha256'])and is_digest(b['batch_sha256'])and type(b['size'])is int and 0<b['size']<=2_000_000_000,'artifact size/hash')
  need(type(b['env_id'])is str and 0<len(b['env_id'])<=100 and type(b['index'])is int and 0<=b['index']<2**31 and (b['env_id'],b['index'])not in seen,'unique declared index');seen.add((b['env_id'],b['index']))
 return envelope

def make(identity,manifest,packed):
 entries=[]
 for i,(batch,data)in enumerate(packed):entries.append(dict(slot=i,env_id=batch['env_id'],index=batch['index'],batch_sha256=sha(canonical(batch)),sha256=sha(data),size=len(data)))
 p=dict(version=VERSION,epoch=manifest['epoch'],miner=identity.id,checkpoint=manifest['checkpoint']['id'],source=manifest['source_bundle']['sha256'],batches=entries)
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
 need(manifest.get('submission_transport_policy')==VERSION,'explicit commitment epoch')
 need(report==unchecked(manifest,receipt),'exact no-credit unaudited report')
 return True

def freeze(gateway,epoch):
 """Persist each successful small receipt; infrastructure failures retry safely."""
 from .storage import SubmissionPolicyError
 from botocore.exceptions import ClientError
 state=gateway.epochs[epoch];state['closed']=True;gateway.persist()
 if 'frozen_receipts'in state:return state['frozen_receipts']
 snapshots=state.setdefault('commitment_snapshots',{});pending=state.setdefault('commitment_pending',{});rejections=state.setdefault('rejections',{});failures=[]
 for miner in sorted(state['miners']):
  if miner in snapshots or miner in rejections:continue
  cutoff=state['commitment_binding'].get('freeze_until')
  if cutoff is not None and time.time()>=cutoff:
   state.setdefault('commitment_deferred',{})[miner]=dict(reason='freeze_infrastructure_budget_deferred',closed_at=time.time());gateway.persist();continue
  try:
   if miner not in pending:
    key='private/'+epoch+'/commitments/'+miner+'.json';r=gateway.bucket.client.get_object(Bucket=gateway.bucket.name,Key=key)
    body=r['Body']
    try:data=body.read(MAX_BYTES+1)
    finally:body.close()
    if not state['start']<=r['LastModified'].timestamp()<state['deadline']:raise SubmissionPolicyError('commitment time')
    try:
     env=validate(data,epoch,miner,state['max_batches'])
     need(env['payload']['checkpoint']==state['commitment_binding']['checkpoint']and env['payload']['source']==state['commitment_binding']['source'],'committed source/model')
    except Exception as exc:raise SubmissionPolicyError('malformed commitment')from exc
    digest=sha(data);root='public/'+epoch+'/submissions/'+miner+'/'+digest
    # Persist exact first observed signed commitment BEFORE conditional copying.
    pending[miner]=dict(key=key,etag=r['ETag'],document=env,sha256=digest,size=len(data),received_at=r['LastModified'].timestamp(),root=root,artifacts=[],artifact_plans={},commitment_copied=False);gateway.persist()
   progress=pending[miner];root=progress['root'];env=progress['document']
   if not progress['commitment_copied']:
    gateway.bucket.copy(progress['key'],root+'/commitment.json',expected_etag=progress['etag']);progress['commitment_copied']=True;gateway.persist()
   for b in env['payload']['batches']:
    slot=b['slot']
    if any(x['slot']==slot for x in progress['artifacts']):continue
    staging='private/'+epoch+'/staging/'+miner+'/'+str(slot)+'.zip';plan=progress['artifact_plans'].get(str(slot))
    if plan is None:
     meta=gateway.bucket.client.head_object(Bucket=gateway.bucket.name,Key=staging)
     if b['size']>state.get('upload_limit',100_000_000)or meta['ContentLength']!=b['size']or not state['start']<=meta['LastModified'].timestamp()<state['deadline']:raise SubmissionPolicyError('artifact size/time')
     plan=dict(etag=meta['ETag'],received_at=meta['LastModified'].timestamp());progress['artifact_plans'][str(slot)]=plan;gateway.persist()
    frozen=root+'/'+str(slot)+'.zip';gateway.bucket.copy(staging,frozen,expected_etag=plan['etag'])
    progress['artifacts'].append(dict(b,key=staging,frozen_key=frozen,etag=plan['etag'],received_at=plan['received_at'],read_url=gateway.bucket.presign(frozen)));gateway.persist()
   snapshots[miner]=dict(sha256=progress['sha256'],commitment_document=env,commitment_key=root+'/commitment.json',artifacts=progress['artifacts'],size=progress['size'],received_at=progress['received_at'],hash_assurance='declared-payload-hashes-until-selected-verifier');gateway.persist()
  except SubmissionPolicyError as exc:rejections[miner]=str(exc);gateway.persist()
  except ClientError as exc:
   if str(exc.response.get('Error',{}).get('Code'))in('NoSuchKey','404','NotFound'):rejections[miner]='missing completed commitment/artifact';gateway.persist()
   else:failures.append(exc)
  except Exception as exc:failures.append(exc)
 cutoff=state['commitment_binding'].get('freeze_until')
 if failures and (cutoff is None or time.time()<cutoff):raise failures[0]
 if failures:
  state.setdefault('commitment_deferred',{}).update({m:dict(reason='freeze_infrastructure_budget_deferred',closed_at=time.time())for m in pending if m not in snapshots and m not in rejections});gateway.persist()
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
 """Stable ZIP framing only; original model arrays and batch bytes unchanged."""
 import io,zipfile
 from .batches import pack
 from .artifact_budget import for_manifest
 original=pack([(batch,arrays)],budget=for_manifest(manifest));out=io.BytesIO()
 with zipfile.ZipFile(io.BytesIO(original))as source,zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED)as target:
  for old in source.infolist():
   info=zipfile.ZipInfo(old.filename,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o600<<16
   target.writestr(info,source.read(old.filename))
 data=out.getvalue()
 if len(data)>for_manifest(manifest)['compressed_bytes']:raise ValueError('stable pair compressed cap')
 return data

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
