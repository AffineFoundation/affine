"""Default-off prospective task cooling. No signing, network or model calls.

The admitted operator supplies the finalized-evidence reader. This module never
promotes unsigned trainer reports into authenticated report-content evidence.
"""
import copy,hashlib,json,os,re,stat,tempfile,time
from pathlib import Path

VERSION='prospective-broad-finalized-task-cooling-v1'
DOMAIN='affine-finalized-task-cooling-v1'
SNAPSHOT='immutable-task-cooling-opening-v1'
MAX_SNAPSHOT_BYTES=4*1024**2
C=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
S=lambda v:hashlib.sha256(C(v)).hexdigest()
def need(v,message):
 if not v:raise ValueError(message)
def indices(v):
 need(isinstance(v,list)and 0<len(v)<=10000 and all(type(i)is int and i>=0 for i in v)and len(v)==len(set(v)),'unique integer task indices');return v

def policy(value):
 fields={'version','enabled','first_round','history_epochs','cooldown_count','minimum_allowed','env_id','epoch_prefix','approved_indices_sha256','calibration_anchor_indices','tie_break_domain'}
 need(type(value)is dict and set(value)==fields and value['version']==VERSION,'exact task cooling policy')
 need(type(value['enabled'])is bool and type(value['first_round'])is int and value['first_round']>=0,'explicit prospective state')
 need(value['history_epochs']==12 and type(value['history_epochs'])is int,'fixed twelve finalized epochs')
 need(type(value['cooldown_count'])is int and value['cooldown_count']in(0,256,512),'bounded prospective cooling width')
 need(type(value['minimum_allowed'])is int and 4096<=value['minimum_allowed']<=10000,'large minimum task pool')
 need(value['env_id']=='affine_math'and isinstance(value['epoch_prefix'],str)and bool(re.fullmatch('[A-Za-z0-9_-]{1,90}',value['epoch_prefix'])),'specific mining environment and stream')
 need(bool(re.fullmatch('[0-9a-f]{64}',value['approved_indices_sha256'])),'approved pool digest')
 need(len(indices(value['calibration_anchor_indices']))==2 and value['tie_break_domain']==DOMAIN,'fixed original calibration anchors and tie break')
 return copy.deepcopy(value)

def ref(v):
 need(type(v)is dict and set(v)=={'path','sha256'}and isinstance(v['path'],str)and Path(v['path']).is_absolute()and bool(re.fullmatch('[0-9a-f]{64}',v['sha256'])),'local evidence reference')
 return copy.deepcopy(v)

def normalized_evidence(value,p,round_number,allowed):
 need(type(value)is dict and set(value)=={'version','records','assurance'}and value['version']=='finalized-trained-task-exposure-input-v1','exact finalized exposure input')
 need(value['assurance']=='signed-closure-and-job-linked-unsigned-trainer-report','explicit report authentication limitation')
 rows=value['records'];need(isinstance(rows,list)and len(rows)==p['history_epochs'],'complete twelve finalized epochs')
 out=[];seen_epochs=set();last=-1
 for row in rows:
  fields={'round','epoch','input_checkpoint','output_checkpoint','updates','provenance'}
  need(type(row)is dict and set(row)==fields,'exact closed training exposure row')
  n=row['round'];need(type(n)is int and last<n<round_number,'strict prior finalized round order')
  need(isinstance(row['epoch'],str)and row['epoch'].startswith(p['epoch_prefix']+'-')and row['epoch'].endswith('-'+str(n))and row['epoch']not in seen_epochs,'specific finalized epoch identity')
  for name in ('input_checkpoint','output_checkpoint'):need(bool(re.fullmatch('[0-9a-f]{64}',row[name])),'closed checkpoint identity')
  pv=row['provenance'];need(type(pv)is dict and set(pv)=={'closure','job','report'},'closure/job/report provenance')
  pv={k:ref(v)for k,v in pv.items()}
  updates=row['updates'];need(isinstance(updates,list)and 1<=len(updates)<=64,'actual completed update exposures')
  steps=set();items=[]
  for u in updates:
   need(type(u)is dict and set(u)=={'global_optimizer_step','indices'},'per-update task exposure')
   st=u['global_optimizer_step'];need(type(st)is int and st>0 and st not in steps,'distinct update identity')
   ids=indices(u['indices']);need(set(ids)<=set(allowed),'trained task in original approved pool')
   items.append(dict(global_optimizer_step=st,indices=list(ids)));steps.add(st)
  out.append(dict(row,updates=items,provenance=pv));last=n;seen_epochs.add(row['epoch'])
 return dict(version=value['version'],records=out,assurance=value['assurance'])

def select(approved,p,round_number,epoch,evidence):
 """One count per trained task per completed update, never submitted volume."""
 p=policy(p);approved=indices(approved)
 need(S(approved)==p['approved_indices_sha256']and len(approved)>=p['minimum_allowed'],'unchanged approved broad pool')
 need(approved[:2]==p['calibration_anchor_indices'],'unchanged leading calibration anchors')
 need(type(round_number)is int and round_number>=p['first_round'],'future task cooling round')
 need(isinstance(epoch,str)and epoch.startswith(p['epoch_prefix']+'-')and epoch.endswith('-'+str(round_number)),'opening epoch identity')
 normalized=normalized_evidence(evidence,p,round_number,approved)
 counts={i:0 for i in approved}
 for row in normalized['records']:
  for update in row['updates']:
   for i in update['indices']:counts[i]+=1
 anchors=set(p['calibration_anchor_indices'])
 rank=sorted((i for i in approved if counts[i]>0 and i not in anchors),key=lambda i:(-counts[i],S([p['tie_break_domain'],epoch,round_number,i])))
 count=min(p['cooldown_count'],len(rank),len(approved)-p['minimum_allowed'])
 cooled=rank[:count];exclude=set(cooled);kept=[i for i in approved if i not in exclude]
 need(kept[:2]==approved[:2]and len(kept)>=p['minimum_allowed'],'calibration and broad pool preserved')
 return dict(allowed_indices=kept,cooled_indices=cooled,cooled_exposures=[dict(index=i,updates=counts[i])for i in cooled],evidence=normalized,evidence_sha256=S(normalized),outcome='cooled'if cooled else'no_positive_exposure_to_cool',fallback_reason=None)

def no_cooling(approved,reason):
 return dict(allowed_indices=list(approved),cooled_indices=[],cooled_exposures=[],evidence=None,evidence_sha256=None,outcome='full_original_pool',fallback_reason=reason)

def environment_id(row):
 need(type(row)is dict,'environment row')
 if 'env_id'in row:
  name=row['env_id'];need(isinstance(name,str)and name,'environment identity')
  if 'spec'in row:need(type(row['spec'])is dict and row['spec'].get('id')==name,'same environment identity')
  return name
 need(type(row.get('spec'))is dict and isinstance(row['spec'].get('id'),str),'environment spec identity')
 return row['spec']['id']

def project(original,allowed,p,project_harness):
 result=copy.deepcopy(original);rows=result.get('environments');need(isinstance(rows,list),'opening environments')
 found=[r for r in rows if environment_id(r)==p['env_id']];need(len(found)==1,'one target environment')
 row=found[0];before=row['indices'];need(S(before)==p['approved_indices_sha256']and before[:2]==p['calibration_anchor_indices'],'original full opening pool')
 row['indices']=list(allowed);row['harness']=project_harness(row['harness'],allowed,before)
 # The controller derives the legacy top-level indices during its original open.
 return result

def contract_digest(original):
 # The admitted parent renews this GET capability on every assembly. Bind the
 # immutable archive identity while allowing its expiring access URL to change.
 value=copy.deepcopy(original);bundle=value.get('source_bundle')
 need(type(bundle)is dict and bool(re.fullmatch('[0-9a-f]{64}',bundle.get('sha256',''))),'original source archive identity')
 if 'url'in bundle:
  need(isinstance(bundle['url'],str)and bundle['url'],'original source capability')
  bundle.pop('url')
 return S(value)

def atomic_once(path,value):
 """Publish an immutable local selection before the caller can sign a manifest."""
 path=Path(path);body=C(value);need(len(body)<=MAX_SNAPSHOT_BYTES,'bounded opening snapshot')
 fd,name=tempfile.mkstemp(prefix='.cooling-',dir=path.parent)
 try:
  with os.fdopen(fd,'wb')as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
  os.chmod(name,0o400)
  try:os.link(name,path,follow_symlinks=False)
  except FileExistsError:return False
  fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return True
 finally:os.unlink(name)

def read_snapshot(path,p,round_number,epoch,original,verify_snapshot):
 st=path.lstat();need(stat.S_ISREG(st.st_mode)and not path.is_symlink()and st.st_uid==os.getuid()and st.st_size<=MAX_SNAPSHOT_BYTES,'owned bounded snapshot')
 raw=path.read_bytes();envelope=json.loads(raw);need(C(envelope)==raw,'canonical signed snapshot')
 need(callable(verify_snapshot),'admitted snapshot signature verifier')
 v=verify_snapshot(envelope);need(type(v)is dict,'authenticated snapshot payload')
 need(set(v)=={'version','epoch','round','policy','policy_sha256','original_contract_sha256','selection','created_at'}and v['version']==SNAPSHOT,'exact original opening snapshot')
 need(v['epoch']==epoch and v['round']==round_number and v['policy']==p and v['policy_sha256']==S(p)and v['original_contract_sha256']==contract_digest(original),'immutable original opening bindings')
 selection=v['selection'];need(type(selection)is dict and set(selection)=={'allowed_indices','cooled_indices','cooled_exposures','evidence','evidence_sha256','outcome','fallback_reason'},'original selection shape')
 approved=next(r['indices']for r in original['environments']if environment_id(r)==p['env_id'])
 if selection['evidence']is None:need(selection==no_cooling(approved,selection['fallback_reason'])and isinstance(selection['fallback_reason'],str),'explicit original full-pool fallback')
 else:need(selection==select(approved,p,round_number,epoch,selection['evidence']),'reproduce original frozen exposure selection')
 return v

def install(service,value,evidence_provider,*,project_harness,sign_snapshot=None,verify_snapshot=None):
 """Wrap only future contract assembly; original run/config/state are untouched.

The source-pinned caller owns authenticating the exposure reader and policy.
The reader must return the twelve latest finalized epochs at this opening. Its
exceptions or malformed evidence cause an explicit immutable full-pool fallback.
"""
 p=policy(value);original_contract=service.contract
 if p['enabled']:need(callable(sign_snapshot)and callable(verify_snapshot),'enabled cooling requires authenticated snapshot hooks')
 def contract(config,round_number):
  original=original_contract(config,round_number)
  if not p['enabled']or round_number<p['first_round']:return original
  state=Path(config['state']);need(state.resolve()==state and not state.is_symlink(),'canonical controller state')
  status=json.loads((state/'controller.json').read_bytes());active=status.get('active')or{}
  need(status.get('round')==round_number and active.get('phase')=='opening','exact current prospective opening')
  epoch=active.get('epoch');need(isinstance(epoch,str)and bool(re.fullmatch('[A-Za-z0-9_-]{1,180}',epoch)),'safe active epoch identity')
  row=next(r for r in original['environments']if environment_id(r)==p['env_id']);approved=indices(row['indices'])
  need(S(approved)==p['approved_indices_sha256']and approved[:2]==p['calibration_anchor_indices']and len(approved)>=p['minimum_allowed'],'approved pool and calibration anchors')
  directory=state/'task-cooling-v1';directory.mkdir(mode=0o700,exist_ok=True)
  need(directory.resolve()==directory and not directory.is_symlink()and directory.stat().st_uid==os.getuid(),'owned snapshot directory')
  path=directory/(epoch+'.json');issued=(state/(epoch+'-manifest.json')).exists()or(state/(epoch+'-first-signed-manifest.json')).exists()
  if not path.exists():
   need(not issued,'never reconstruct an issued epoch without its original snapshot')
   if p['cooldown_count']==0:selection=no_cooling(approved,'configured_zero_cooldown')
   else:
    try:selection=select(approved,p,round_number,epoch,evidence_provider(config,round_number,epoch))
    except Exception as error:
     code=getattr(error,'reason_code',None)
     reason=code if isinstance(code,str)and re.fullmatch('[a-z0-9_]{1,96}',code)else type(error).__name__
     selection=no_cooling(approved,'finalized_evidence_unavailable_or_invalid:'+reason)
   value=dict(version=SNAPSHOT,epoch=epoch,round=round_number,policy=p,policy_sha256=S(p),original_contract_sha256=contract_digest(original),selection=selection,created_at=time.time())
   envelope=sign_snapshot(value);need(verify_snapshot(envelope)==value,'exact authority-signed original snapshot')
   atomic_once(path,envelope)
  frozen=read_snapshot(path,p,round_number,epoch,original,verify_snapshot)
  return project(original,frozen['selection']['allowed_indices'],p,project_harness)
 service.contract=contract
 return original_contract
