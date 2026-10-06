"""Prospective local capture WAL; never authenticates numerical computation."""
import os,stat,json,threading,fcntl
from pathlib import Path
from .commitment_transport import canonical,sha,need
VERSION='fsynced-per-epoch-capture-v1'
MAX_RECORD=80000
MAX_FILE=64_000_000
class JournalDurabilityError(RuntimeError):pass

def sync_directory(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)

class CaptureJournal:
 def __init__(self,gateway,epoch):
  need(gateway.state_path is not None,'durable capture requires state path')
  root=Path(gateway.state_path).parent
  need(root.resolve()==root and stat.S_ISDIR(root.lstat().st_mode)and root.lstat().st_uid==os.getuid(),'ordinary capture state directory')
  self.root=root/'capture-journals'
  self.root.mkdir(mode=0o700,exist_ok=True)
  need(self.root.resolve()==self.root and stat.S_ISDIR(self.root.lstat().st_mode)and self.root.lstat().st_uid==os.getuid()and self.root.lstat().st_mode&0o077==0,'ordinary capture journal directory')
  sync_directory(root)
  self.state=gateway.epochs[epoch];self.epoch=epoch;self.lock=threading.Lock();self.intents={};self.commits={}
  self.context=sha(canonical(self.context_value()))
  name=sha((self.phase+':'+epoch).encode());self.path=self.root/(name+'.jsonl')
  self.guard=os.open(self.root/(name+'.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
  self.fd=None
  try:
   guard_stat=os.fstat(self.guard);need(stat.S_ISREG(guard_stat.st_mode)and guard_stat.st_uid==os.getuid()and guard_stat.st_nlink==1 and guard_stat.st_mode&0o077==0,'ordinary private capture lock')
   fcntl.flock(self.guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
   self.fd=os.open(self.path,os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
   st=os.fstat(self.fd);need(stat.S_ISREG(st.st_mode)and st.st_uid==os.getuid()and st.st_nlink==1 and st.st_mode&0o077==0 and st.st_size<=MAX_FILE,'private bounded journal file')
   raw=os.pread(self.fd,st.st_size,0);self.previous='0'*64;self.sequence=0;self.failed=False
   # A trailing incomplete append was never a durable acknowledged record.
   # Its previous durable intent remains and is reconciled against R2 below.
   complete=raw.rfind(b'\n')+1
   need(not raw or complete>0,'unbound incomplete journal header')
   for line in raw[:complete].splitlines():
    need(len(line)<=MAX_RECORD,'bounded journal record');entry=json.loads(line)
    need(canonical(entry)==line and set(entry)=={'payload','sha256'}and sha(canonical(entry['payload']))==entry['sha256'],'journal canonical hash')
    p=entry['payload'];need(set(p)=={'version','context','sequence','previous','kind','miner','receipt'}and p['version']==VERSION and p['context']==self.context and type(p['sequence'])is int and p['sequence']==self.sequence and p['previous']==self.previous,'journal chain and exact capture context')
    self._accept(p);self.sequence+=1;self.previous=entry['sha256']
   if complete!=len(raw):os.ftruncate(self.fd,complete);os.fsync(self.fd)
   if not self.sequence:self._append('header',None,None)
   sync_directory(self.root)
  except BaseException:
   self.close();raise

 phase='tokens'
 def context_value(self):
  return dict(version=VERSION,phase=self.phase,epoch=self.epoch,start=self.state['start'],deadline=self.state['deadline'],binding=self.state['commitment_binding'],parents={k:v['sha256']for k,v in self.state['commitment_pending'].items()})

 def _key(self,miner,receipt):
  need(miner in self.state['commitment_pending'],'journal original miner')
  need(type(receipt)is dict and set(receipt)=={'slot','sha256','size','frozen_key','captured_at','assurance'},'journal receipt schema')
  need(type(receipt['slot'])is int,'journal slot type')
  parent=self.state['commitment_pending'][miner]
  b=next((v for v in parent['document']['payload']['batches']if v['slot']==receipt['slot']),None)
  need(b is not None and receipt['sha256']==b['training_sha256']and type(receipt['size'])is int and receipt['size']==b['training_size']and receipt['frozen_key']==parent['root']+'/training/'+str(receipt['slot'])+'.json'and receipt['assurance']=='unaudited','journal declared bytes and immutable scope')
  at=receipt['captured_at'];need(type(at)in(int,float)and self.state['start']<=at<self.state['commitment_binding']['freeze_until'],'journal original capture time')
  return miner,str(receipt['slot'])

 def _accept(self,p):
  if p['kind']=='header':need(self.sequence==0 and p['miner']is None and p['receipt']is None,'first journal header');return
  need(self.sequence>0 and p['kind']in('intent','commit'),'journal operation')
  key=self._key(p['miner'],p['receipt'])
  if p['kind']=='intent':
   need(key not in self.commits,'no replacement of durable capture');self.intents[key]=p['receipt']
  else:
   need(self.intents.get(key)==p['receipt'],'commit exact prior intent')
   need(key not in self.commits or self.commits[key]==p['receipt'],'conflicting capture commit')
   self.commits[key]=p['receipt']

 def _append(self,kind,miner,receipt):
  if self.failed:raise JournalDurabilityError('capture journal requires recovery after uncertain append')
  try:self._write_record(kind,miner,receipt)
  except BaseException as exc:
   self.failed=True;raise JournalDurabilityError('capture journal append not acknowledged')from exc

 def _write_record(self,kind,miner,receipt):
  p=dict(version=VERSION,context=self.context,sequence=self.sequence,previous=self.previous,kind=kind,miner=miner,receipt=receipt)
  self._accept(p);digest=sha(canonical(p));data=canonical(dict(payload=p,sha256=digest))+b'\n'
  need(len(data)<=MAX_RECORD and os.fstat(self.fd).st_size+len(data)<=MAX_FILE,'bounded capture journal append')
  view=memoryview(data)
  while view:
   n=os.write(self.fd,view);need(n>0,'journal write progress');view=view[n:]
  os.fsync(self.fd);self.sequence+=1;self.previous=digest

 def intent(self,miner,receipt):
  with self.lock:self._append('intent',miner,receipt)

 def commit(self,miner,receipt):
  with self.lock:self._append('commit',miner,receipt)

 def replay(self):
  snapshots=self.state.setdefault('training_document_snapshots',{})
  for (miner,slot),receipt in self.commits.items():
   old=snapshots.setdefault(miner,{}).get(slot);need(old is None or old==receipt,'global checkpoint conflicts with capture WAL');snapshots[miner][slot]=receipt
   self.state.get('training_document_deferred',{}).get(miner,{}).pop(slot,None)

 def unresolved(self):return [(miner,r)for (miner,slot),r in self.intents.items()if (miner,slot)not in self.commits]

 def checkpoint(self,gateway):checkpoint_state(gateway)

 def close(self):
  if self.fd is not None:os.close(self.fd);self.fd=None
  if getattr(self,'guard',None)is not None:os.close(self.guard);self.guard=None

class CommitmentJournal(CaptureJournal):
 """Authenticated tiny commitments before token-stage parent inventory exists."""
 phase='commitments'
 def context_value(self):
  discovery=self.state['commitment_discovery']
  need(discovery['complete']is True and type(discovery['miners'])is list and len(discovery['miners'])==len(set(discovery['miners']))and set(discovery['miners'])<=set(self.state['miners']),'complete exact commitment discovery')
  return dict(version=VERSION,phase=self.phase,epoch=self.epoch,start=self.state['start'],deadline=self.state['deadline'],binding=self.state['commitment_binding'],max_batches=self.state['max_batches'],miners=sorted(self.state['miners']),discovery_sha256=sha(canonical(discovery)))

 def _accept(self,p):
  if p['kind']=='header':need(self.sequence==0 and p['miner']is None and p['receipt']is None,'first commitment journal header');return
  need(self.sequence>0 and p['kind']in('commitment','rejection'),'commitment journal operation')
  miner=p['miner'];value=p['receipt'];need(miner in self.state['commitment_discovery']['miners'],'original discovered commitment miner')
  if p['kind']=='rejection':
   need(type(value)is str and value in('commitment time','malformed commitment','missing completed commitment/artifact'),'original structural rejection')
  else:
   from .commitment_transport import validate
   need(type(value)is dict and set(value)=={'key','etag','document','sha256','size','received_at','root','artifacts','artifact_plans','commitment_copied'},'exact authenticated commitment record')
   data=canonical(value['document']);env=validate(data,self.epoch,miner,self.state['max_batches']);binding=self.state['commitment_binding']
   need(env['payload']['version']==binding.get('version','small-commitment-v1')and env['payload']['checkpoint']==binding['checkpoint']and env['payload']['source']==binding['source'],'commitment replay signed transport/source/checkpoint')
   need(value['sha256']==sha(data)and type(value['size'])is int and value['size']==len(data)and value['key']=='private/'+self.epoch+'/commitments/'+miner+'.json'and value['root']=='public/'+self.epoch+'/submissions/'+miner+'/'+value['sha256'],'commitment original bytes and paths')
   need(type(value['received_at'])in(int,float)and self.state['start']<=value['received_at']<self.state['deadline']and type(value['etag'])is str and 0<len(value['etag'])<=1024,'commitment original R2 metadata')
   need(value['artifacts']==[]and value['artifact_plans']=={}and value['commitment_copied']is False,'initial commitment state')
  need(miner not in self.commits or self.commits[miner]==(p['kind'],value),'conflicting original commitment decision')
  self.commits[miner]=(p['kind'],value)

 def accepted(self,miner,value):
  with self.lock:self._append('commitment',miner,value)

 def rejected(self,miner,reason):
  with self.lock:self._append('rejection',miner,reason)

 def replay(self):
  pending=self.state['commitment_pending'];rejections=self.state['rejections']
  for miner,(kind,value)in self.commits.items():
   if kind=='rejection':
    need(miner not in pending and (miner not in rejections or rejections[miner]==value),'conflicting persisted commitment rejection');rejections[miner]=value
   else:
    need(miner not in rejections,'conflicting persisted accepted commitment')
    if miner in pending:
     for key in('key','etag','document','sha256','size','received_at','root'):need(pending[miner][key]==value[key],'persisted commitment immutable conflict')
    else:pending[miner]=value


def checkpoint_state(gateway):
 gateway.persist()
 fd=os.open(gateway.state_path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  st=os.fstat(fd);need(stat.S_ISREG(st.st_mode)and st.st_uid==os.getuid()and st.st_nlink==1,'ordinary owned gateway checkpoint');os.fsync(fd)
 finally:os.close(fd)
 sync_directory(Path(gateway.state_path).parent)
