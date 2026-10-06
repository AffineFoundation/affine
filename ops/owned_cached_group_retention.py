"""Separate default-off operator retention; frozen per-job leases remain intact."""
import fcntl,hashlib,json,os,time
from pathlib import Path
from subnet.backend_jobs import canonical,signed,file_map
from ops.owned_cached_larger_cohort import SOURCE,POLICY
VERSION='root-owned-cached128-group-retention-v1'
CACHE_SHA='86bd543f85d39e0b8e42fbed10ff2c814aaedd55fce0bb2427107bf332227dfa'
def frozen_cache():
 """Read and execute the byte-identical 4db implementation as a CPU dependency.

 Never resolve subnet.cache_lifecycle from the operator's current checkout.
 The frozen scientific runtime map remains unchanged.
 """
 import types
 path=Path(__file__).with_name('owned_cached_group_frozen_cache.py')
 if path.is_symlink()or not path.is_file():raise ValueError('frozen owned-cache dependency path')
 raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=CACHE_SHA:raise ValueError('owned-cache dependency changed before import')
 module=types.ModuleType('owned_cached_group_frozen_cache');module.__file__=str(path)
 exec(compile(raw,str(path),'exec'),module.__dict__);return module

def validate_scope(envelope,authority,workspace,source_files,*,now=None):
 scope=signed(envelope,authority);now=time.time()if now is None else now;root=Path(workspace)
 if scope.get('version')!=VERSION or scope.get('execute_allowed')is not True or not scope['created_at']<=now<scope['expires_at']or scope['expires_at']-scope['created_at']>7200:raise ValueError('fresh explicit ROOT group execution scope')
 if root!=root.absolute()or root.resolve()!=root or str(root)!=scope['workspace']:raise ValueError('fresh owned canonical namespace')
 if scope['source_sha256']!=SOURCE or scope['source_files']!=source_files or len(source_files)!=177 or scope['experiment_id']!='owned-cached-native-heldout128-cap1024-v1':raise ValueError('frozen source and distinct128 experiment')
 groups=scope['groups'];indices=[i for g in groups for i in g['indices']]
 if len(groups)!=4 or sorted(g['group']for g in groups)!=list(range(4))or len(set(indices))!=128 or any(len(g['indices'])!=32 or g['seeds']!=[20261002+i*1000 for i in g['indices']]for g in groups):raise ValueError('four disjoint fixed32 groups')
 jobs=scope['original_jobs']
 if len(jobs)!=4 or len(set(jobs))!=4 or set(v['group']for v in jobs.values())!=set(range(4))or any(len(v['job_sha256'])!=64 for v in jobs.values()):raise ValueError('exact four authentic original job bindings')
 if source_files.get('subnet/cache_lifecycle.py')!=CACHE_SHA:raise ValueError('exact frozen owned-cache implementation')
 if file_map(scope['checkpoint']['files'])!=scope['checkpoint']['id']:raise ValueError('exact checkpoint bytes map')
 return scope
class GroupRetention:
 """A namespace retention lock, deliberately distinct from checkpoint EX locks."""
 def __init__(self,envelope,authority,workspace,source_files,*,now=None):
  self.scope=validate_scope(envelope,authority,workspace,source_files,now=now);self.root=Path(workspace);self.fd=None
 def __enter__(self):
  meta=self.root/'.owned-cached-group-retention';meta.mkdir(mode=0o700,exist_ok=True)
  if meta.is_symlink():raise ValueError('group retention metadata symlink')
  self.fd=os.open(meta/'exclusive.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
  try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BaseException:os.close(self.fd);self.fd=None;raise
  return self
 def __exit__(self,*args):
  if self.fd is not None:os.close(self.fd);self.fd=None

def retire_group(envelope,acks,authority,workspace,source_files,*,now=None,cache_factory=None,live=None):
 """Remove only the group's owned model after every original has a genuine ACK.

 The caller first full-reads each ACK from R2. Authentication and hash checks
 below bind exact reports/jobs; an unsigned flag cannot authorize disposal.
 """
 scope=validate_scope(envelope,authority,workspace,source_files,now=now);root=Path(workspace);cp=scope['checkpoint']
 CacheLifecycle=frozen_cache().CacheLifecycle
 from subnet.evaluator_cache_lifecycle import live_original
 live=live_original if live is None else live;cache_factory=CacheLifecycle if cache_factory is None else cache_factory
 if len(acks)!=4:raise ValueError('all four original durable ACKs required')
 seen=set();failed=[]
 for ack_envelope in acks:
  ack=signed(ack_envelope,authority);job=signed(ack['original_job'],authority);manifest=signed(job['manifest'],authority);jid=job['job_id'];binding=scope['original_jobs'].get(jid);report=ack['original_report']
  if binding is None or jid in seen or job['source_files']!=source_files or job['role']!='evaluate' or job.get('owned_evaluation_policy')!=POLICY or manifest['checkpoint']!=cp or manifest['source_bundle']['sha256']!=SOURCE or job.get('checkpoint_cache')is not None:raise ValueError('exact owned original source/parent/policy; no external cache')
  if (ack.get('version')!='owned-cached-evaluation-durable-ack-v1'or ack.get('durable_report_full_readback')is not True or ack['workspace']!=str(root)or ack['checkpoint']!=cp or ack['job_sha256']!=hashlib.sha256(canonical(job)).hexdigest()or ack['job_sha256']!=binding['job_sha256']or ack['report_sha256']!=hashlib.sha256(canonical(report)).hexdigest()or report['job_id']!=jid or report['job_sha256']!=ack['job_sha256']or report['checkpoint']!=cp['id']):raise ValueError('genuine exact complete report durability ACK')
  group=next(g for g in scope['groups']if g['group']==binding['group']);suite=job['heldout'][0]
  if len(job['heldout'])!=1 or suite['indices']!=group['indices']or suite['seeds']!=group['seeds']or suite['harness']!=group['harness']:raise ValueError('signed group scientific profile')
  status=json.loads((root/'runner-status'/(jid+'.json')).read_bytes())
  if status.get('job_id')!=jid or status.get('phase')not in ('complete','failed')or type(status.get('exit_code'))is not int or (status['phase']=='complete')!=(status['exit_code']==0):raise ValueError('actual original terminal required')
  if live(status):return dict(status='deferred',reason='original-still-live',removed=[])
  if status['exit_code']!=0 or report.get('success')is not True:failed.append(jid)
  seen.add(jid)
 if seen!=set(scope['original_jobs']):raise ValueError('all four original identities')
 if any(live(json.loads(p.read_bytes()))for p in(root/'runner-status').glob('*.json')):return dict(status='deferred',reason='owned-namespace-live-process',removed=[])
 cache=cache_factory(root);directory=root/'checkpoints'/cp['id']
 if not directory.exists():return dict(status='complete',removed=[],already_absent=True,failed_originals=failed,model_reward=None if failed else 'aggregate-separately')
 receipt=cache._receipt(cp['id']);v=json.loads(receipt.read_bytes())
 if v.get('path',str(Path('checkpoints')/cp['id']))!=str(Path('checkpoints')/cp['id'])or v.get('files')!=cp['files']or set(v.get('members',{}))!=set(cp['files']):raise ValueError('exact original owned hydration inventory')
 removed=cache.evict_checkpoints(keep=0,only=[cp['id']])
 return dict(status='complete'if removed else 'deferred',removed=removed,reason=None if removed else 'lease-or-inode-guard',failed_originals=failed,model_reward=None if failed else 'aggregate-separately')

def capacity_admission(lease,free_bytes):
 """Account only the exact already-owned model within an active group lease."""
 if not isinstance(lease,GroupRetention)or lease.fd is None:raise ValueError('active exclusive namespace retention lease')
 scope=lease.scope;cold=scope['minimum_free_cold_bytes'];margin=scope['minimum_free_warm_margin_bytes']
 if type(free_bytes)is not int or type(cold)is not int or type(margin)is not int or cold<20000000000 or margin<5000000000 or cold<margin:raise ValueError('explicit cold and warm capacity margins')
 frozen=frozen_cache();CacheLifecycle=frozen.CacheLifecycle;snapshot=frozen.snapshot
 cache=CacheLifecycle(lease.root);cp=scope['checkpoint'];receipt=cache._receipt(cp['id']);owned_bytes=0
 if receipt.is_file():
  value=json.loads(receipt.read_bytes());directory=cache._path(Path('checkpoints')/cp['id'])
  if value.get('path',str(Path('checkpoints')/cp['id']))!=str(Path('checkpoints')/cp['id'])or value['files']!=cp['files']or set(value['members'])!=set(cp['files'])or set(p.name for p in directory.iterdir())!=set(cp['files']):raise ValueError('exact owned model capacity receipt')
  for name,member in value['members'].items():
   if member['sha256']!=cp['files'][name]or snapshot(directory/name)!=member['stat']:raise ValueError('owned model capacity inode/ownership changed')
   owned_bytes+=(directory/name).stat().st_size
 required=max(cold-owned_bytes,margin)
 return dict(admitted=free_bytes>=required,required_free_bytes=required,existing_owned_model_bytes=owned_bytes,model_download_reused=owned_bytes>0,source_unchanged=SOURCE)
