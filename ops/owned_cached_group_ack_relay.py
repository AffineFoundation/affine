"""Passive ROOT CPU relay: full physical-file observation -> R2 readback -> ACK.

Never starts GPU jobs, never renews an original expiry, never disposes a model.
Bucket credentials/authority seed stay on this CPU host. Default-off CLI.
"""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
from subnet.backend_jobs import canonical,signed
from ops.owned_cached_group_operator import GroupACKPublisher,private_json,validate_terminal
READ_BODY='''import pathlib,hashlib,os,stat,subprocess
root=pathlib.Path(PLAN['workspace']);code=pathlib.Path(PLAN['source_path']);assert hashlib.sha256(pathlib.Path('/etc/machine-id').read_bytes()).hexdigest()==PLAN['machine_id_sha256'];assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()==PLAN['gpu_uuid']
assert {str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(code/'subnet').glob('*.py')}==PLAN['source_files']
def read(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_uid==os.geteuid()and s.st_nlink==1 and not s.st_mode&0o077 and s.st_size<=8*1024**2;raw=p.read_bytes();after=p.lstat();assert all(getattr(after,k)==getattr(s,k)for k in('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns'));return json.loads(raw)
jid=PLAN['job_id'];job=root/(jid+'.json');marker=root/'runner-status'/(jid+'.json');report=root/'jobs'/jid/'report.json'
if not all(p.exists()for p in(job,marker,report)):print('null')
else:
 envelope=read(job);assert hashlib.sha256(json.dumps(envelope,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==PLAN['envelope_sha256'];status=read(marker);live=False
 for field in('runner_pid','child_pid'):
  pid=status.get(field);ticks=status.get(field+'_ticks')
  assert type(pid)is int and pid>0 and str(ticks).isdigit()
  p=pathlib.Path('/proc')/str(pid)/'stat'
  if p.exists():
   parts=p.read_text().rsplit(')',1)[1].split();live=live or(parts[0]!='Z'and parts[19]==str(ticks))
 print(json.dumps({'original_job':envelope,'terminal':status,'report':read(report),'physical_original_absent':not live,'machine_id_sha256':PLAN['machine_id_sha256'],'gpu_uuid':PLAN['gpu_uuid']}))
'''

def emit_read(plan):
 code='import json\nPLAN=json.loads('+repr(json.dumps(plan,separators=(',',':')))+')\n'+READ_BODY;compile(code,'exact-group-physical-read','exec');return code

class QualifiedGroupObserver:
 def __init__(self,scope,envelopes,authority):
  self.scope=scope;self.authority=authority;self.originals={signed(v,authority)['job_id']:v for v in envelopes};self.endpoint=scope['endpoint']
  if self.endpoint['workspace']!=scope['workspace']or self.endpoint['code']!=scope['source_path']:raise ValueError('exact ROOT physical group route')
  if not str(self.endpoint['known_hosts']).startswith('/')or not str(self.endpoint['python']).startswith('/'):raise ValueError('strict qualified SSH/runtime paths')
 def command(self,code):
  e=self.endpoint;args=['ssh','-p',str(e['port']),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+e['known_hosts'],'-o','ConnectTimeout=10',e.get('user','root')+'@'+e['host'],shlex.quote(e['python'])+' -I -B -']
  r=subprocess.run(args,input=code,text=True,capture_output=True,timeout=30)
  if r.returncode:raise RuntimeError('private original group SSH observation failed; no dispatch')
  return json.loads(r.stdout)
 def read(self,jid):
  envelope=self.originals[jid];plan={k:self.scope[k]for k in('workspace','source_path','source_files','machine_id_sha256','gpu_uuid')};plan.update(job_id=jid,envelope_sha256=hashlib.sha256(canonical(envelope)).hexdigest());v=self.command(emit_read(plan))
  if v is None:return None
  if v['original_job']!=envelope or v['machine_id_sha256']!=self.scope['machine_id_sha256']or v['gpu_uuid']!=self.scope['gpu_uuid']:raise ValueError('physical original source/identity binding')
  return v
 def install_ack(self,jid,envelope):
  payload=signed(envelope,self.authority)
  if payload['original_job']!=self.originals[jid]or payload['workspace']!=self.scope['workspace']:raise ValueError('exact original remote ACK route')
  raw=canonical(envelope);plan=dict(root=self.scope['workspace'],job_id=jid,raw=raw.decode(),sha256=hashlib.sha256(raw).hexdigest())
  code='import json\nPLAN=json.loads('+repr(json.dumps(plan))+')\n'+'''import pathlib,hashlib,os,stat
root=pathlib.Path(PLAN['root']);directory=root/'durable-evaluation-acks';assert not directory.is_symlink();directory.mkdir(mode=0o700,exist_ok=True);p=directory/(PLAN['job_id']+'.json');raw=PLAN['raw'].encode();assert hashlib.sha256(raw).hexdigest()==PLAN['sha256']
if p.exists():
 s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_uid==os.geteuid()and s.st_nlink==1 and not s.st_mode&0o077;assert p.read_bytes()==raw
else:
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'original_ack_installed':True,'ack_sha256':PLAN['sha256']}))
''';compile(code,'exact-original-ACK-install','exec');return self.command(code)

def relay_step(publisher,observer):
 if observer.scope!=publisher.scope:raise ValueError('publisher and physical observer scope changed')
 if not hasattr(observer,'installed'):observer.installed=set()
 for jid,(envelope,job,_)in publisher.originals.items():
  if jid in observer.installed:continue
  v=observer.read(jid)
  if v is None:continue
  # A worker publishes its report before remote_runner atomically finalizes
  # the marker. Report availability is not terminal/physical completion.
  status=v['terminal']
  if status.get('phase')in('pending','running'):
   if status.get('job_id')!=jid or status.get('exit_code')is not None or status.get('finished_at')is not None:
    raise ValueError('nonterminal original marker binding')
   return dict(status='observing-original',job_id=jid,original_phase=status['phase'],GPU_dispatch=False)
  validate_terminal(status,job,publisher.clock())
  if v['terminal']['exit_code']!=0:return dict(status='original-infrastructure-failure',job_id=jid,model_reward=None)
  if v['physical_original_absent']is not True:return dict(status='observing-original',job_id=jid)
  ack=publisher.publish(jid,v['report'],v['terminal'],physical_original_absent=True);observer.install_ack(jid,ack);observer.installed.add(jid)
 return dict(status='all-four-durable-ACKs-installed'if len(observer.installed)==4 else'waiting-originals',durable_ACK_count=len(observer.installed),GPU_dispatch=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--authority',required=True);p.add_argument('--original-jobs',nargs=4,required=True);p.add_argument('--bucket-config');p.add_argument('--authority-seed');p.add_argument('--execute-relay',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
 if not a.execute_relay:print(json.dumps({'dispatch_allowed':False,'ROOT_signing_allowed':False,'reason':'default-off relay'}));return
 if not a.bucket_config or not a.authority_seed:raise ValueError('explicit CPU-only ROOT publisher configuration')
 scope_envelope=private_json(a.scope);scope=signed(scope_envelope,a.authority);jobs=[private_json(p)for p in a.original_jobs]
 from subnet.storage import Bucket,Identity
 seed=Path(a.authority_seed);s=seed.stat()
 if s.st_mode&0o077 or s.st_uid!=os.geteuid()or s.st_nlink!=1 or seed.is_symlink():raise ValueError('private local ROOT seed selector')
 identity=Identity(bytes.fromhex(seed.read_text().strip()))
 if identity.id!=a.authority:raise ValueError('exact approved local ROOT authority')
 config=private_json(a.bucket_config);config=config.get('payload',config);bucket=Bucket(config['bucket'])
 def sign(payload):
  import base64
  return dict(payload=payload,signer=identity.id,signature=base64.b64encode(identity.key.sign(canonical(payload)).signature).decode())
 publisher=GroupACKPublisher(scope_envelope,a.authority,jobs,bucket,sign);observer=QualifiedGroupObserver(scope,jobs,a.authority)
 while True:
  result=relay_step(publisher,observer);print(json.dumps(result,sort_keys=True),flush=True)
  if a.once or result['status']in('all-four-durable-ACKs-installed','original-infrastructure-failure'):return
  if time.time()>=scope['expires_at']:raise ValueError('scope expired; preserve originals for explicit reconciliation')
  time.sleep(2)
if __name__=='__main__':main()
