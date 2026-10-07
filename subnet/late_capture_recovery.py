"""ROOT-authorized CPU capture windows; never extend miner upload eligibility."""
import base64,math,time
from nacl.signing import VerifyKey
from .commitment_transport import canonical,sha,need
VERSION='same-original-late-capture-recovery-v1'
AUTHORITY='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
FIELD='capture_recovery_authorizations'

def validate(document,state,epoch,*,authority=None):
 authority=AUTHORITY if authority is None else authority
 need(type(document)is dict and set(document)=={'payload','signature','signer'}and document['signer']==authority,'ROOT capture recovery authority')
 signature=base64.b64decode(document['signature'],validate=True)
 need(len(signature)==64,'capture recovery signature')
 VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),signature)
 p=document['payload']
 keys={'version','epoch','start','deadline','original_freeze_until','source','checkpoint','binding_sha256','miners_sha256','first_signed_manifest_sha256','operational_start','operational_until','reason'}
 need(type(p)is dict and set(p)==keys and p['version']==VERSION and p['reason']=='original-controller-infrastructure-recovery','exact typed capture recovery')
 b=state['commitment_binding']
 need(p['epoch']==epoch and p['start']==state['start']and p['deadline']==state['deadline']and p['original_freeze_until']==b['freeze_until']and p['source']==b['source']and p['checkpoint']==b['checkpoint']and p['binding_sha256']==sha(canonical(b))and p['miners_sha256']==sha(canonical(sorted(state['miners']))),'same original capture scope')
 need(type(p['first_signed_manifest_sha256'])is str and len(p['first_signed_manifest_sha256'])==64 and all(c in '0123456789abcdef'for c in p['first_signed_manifest_sha256']),'original signed manifest digest')
 for key in ('operational_start','operational_until'):
  need(type(p[key])in(int,float)and math.isfinite(p[key]),'finite operational recovery window')
 need(b['freeze_until']<=p['operational_start']and 0<p['operational_until']-p['operational_start']<=900 and p['operational_start']-state['deadline']<=86400,'bounded late capture operational window')
 return p

def windows(state,epoch):
 docs=state.get(FIELD,[])
 need(type(docs)is list and len(docs)<=16,'bounded signed recovery history')
 return [validate(doc,state,epoch)for doc in docs]

def cutoff(state,epoch,at=None):
 original=state['commitment_binding'].get('freeze_until')
 if not state.get(FIELD):return original
 at=time.time()if at is None else at
 candidates=[p['operational_until']for p in windows(state,epoch)if p['operational_start']<=at<p['operational_until']]
 return max([original]+candidates)

def valid_capture_time(state,epoch,at):
 if type(at)not in(int,float)or not math.isfinite(at):return False
 original=state['commitment_binding']['freeze_until']
 if state['start']<=at<original:return True
 return any(p['operational_start']<=at<p['operational_until']for p in windows(state,epoch))

def attach(gateway,epoch,document,first_signed_manifest,*,at=None):
 """Append immutable signed operational evidence; upload/binding state unchanged."""
 state=gateway.epochs[epoch];p=validate(document,state,epoch)
 need(sha(canonical(first_signed_manifest))==p['first_signed_manifest_sha256'],'exact original first signed manifest')
 need(type(first_signed_manifest)is dict and first_signed_manifest.get('signer')==AUTHORITY,'original manifest signer')
 VerifyKey(bytes.fromhex(AUTHORITY)).verify(canonical(first_signed_manifest['payload']),base64.b64decode(first_signed_manifest['signature'],validate=True))
 m=first_signed_manifest['payload']
 need(m['epoch']==epoch and m['start']==state['start']and m['deadline']==state['deadline']and m['source_bundle']['sha256']==p['source']and m['checkpoint']['id']==p['checkpoint'],'original signed upload/computation context')
 at=time.time()if at is None else at
 if document in state.get(FIELD,[]):
  windows(state,epoch);return p
 need(p['operational_start']<=at<p['operational_until'],'active authorized recovery admission')
 docs=state.setdefault(FIELD,[])
 if document not in docs:
  need(len(docs)<16,'bounded recovery attempts');docs.append(document);gateway.persist()
 return p
