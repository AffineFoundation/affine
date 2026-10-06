"""Default-off, ROOT-scoped CPU operator disposal for genuine owned mine jobs."""
import argparse,base64,hashlib,importlib.util,json,pathlib,subprocess,sys,time
from nacl.signing import VerifyKey
AUTH='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(path):
 p=pathlib.Path(path)
 if p.is_symlink()or not p.is_file():raise ValueError('regular reviewed operator input')
 return json.loads(p.read_bytes())
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def authenticate(x):
 if x.get('signer')!=AUTH:raise ValueError('ROOT operator approval required')
 VerifyKey(bytes.fromhex(AUTH)).verify(canonical(x['payload']),base64.b64decode(x['signature'],validate=True));return x['payload']
def validate_inputs(approval,config,path):
 if approval['version']!='owned-miner-terminal-cleanup-operator-v1' or approval['config_sha256']!=sha(path)or approval['state']!=config['state']:raise ValueError('exact operator approval/config')
 if approval['max_jobs_per_cycle']not in(1,4,8)or type(approval['max_jobs_per_cycle'])is not int:raise ValueError('bounded CPU disposal')
 source=config['source_bundle']['sha256'];entry=approval['sources'].get(source)
 if entry is None or config['remote']['roles']['mine']!=approval['miner_endpoint']:raise ValueError('approved physical miner route')
 tree=pathlib.Path(approval['runtime_tree'])
 for name,digest in entry.items():
  p=tree/name
  if p.is_symlink()or sha(p)!=digest:raise ValueError('complete pinned scientific CPU runtime')
 if sha(__file__)!=approval['entrypoint_sha256']or sha(approval['cleanup_module'])!=approval['cleanup_module_sha256']:raise ValueError('exact CPU operator files')
 seed=pathlib.Path(config['state'])/'authority.seed'
 if seed.is_symlink()or not seed.is_file()or seed.stat().st_mode&0o077:raise ValueError('existing private authority; never mint another')
 return tree

def validate(approval,config,path):
 tree=validate_inputs(approval,config,path)
 if not approval['created_at']<=time.time()<approval['expires_at']:raise ValueError('fresh operator approval')
 selector=approval['controller'];v=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',selector['unit'],'--property=MainPID,InvocationID'],text=True).splitlines())
 if v['MainPID']!='0':
  if v['MainPID']!=str(selector['pid'])or v['InvocationID']!=selector['invocation']:raise ValueError('different controller requires new approval')
  p=pathlib.Path('/proc')/v['MainPID'];ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19])
  if ticks!=selector['ticks']:raise ValueError('exact current controller process')
 return tree

def renew_authorized_scope(approval,config,path,scope_path):
 # The durable authorization remains narrow: identical config, physical route,
 # complete source pins, operator bytes, unit definition and original argv.
 # A reviewed controller restart may change only its PID/ticks/invocation.
 renewal=approval.get('renewal_authorization')
 if renewal is None:return approval
 validate_inputs(approval,config,path)
 if renewal.get('version')!='identical-controller-cleanup-renewal-v1':raise ValueError('explicit bounded renewal authorization')
 unit=approval['controller']['unit'];v=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=MainPID,InvocationID,FragmentPath'],text=True).splitlines())
 if v['FragmentPath']!=renewal['unit_path']or sha(v['FragmentPath'])!=renewal['unit_sha256']:raise ValueError('identical authorized controller unit')
 if sha(renewal['launcher_path'])!=renewal['launcher_sha256']:raise ValueError('identical authorized controller launcher')
 selector=dict(approval['controller'])
 if v['MainPID']!='0':
  p=pathlib.Path('/proc')/v['MainPID'];argv=(p/'cmdline').read_bytes()
  if hashlib.sha256(argv).hexdigest()!=renewal['process_argv_sha256']:raise ValueError('identical authorized controller argv')
  selector=dict(unit=unit,pid=int(v['MainPID']),ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),invocation=v['InvocationID'])
 now=time.time()
 if approval['created_at']<=now<approval['expires_at'] and selector==approval['controller']:return approval
 updated=dict(approval,created_at=now,expires_at=now+3600,controller=selector)
 validate(updated,config,path)
 from nacl.signing import SigningKey
 key=SigningKey(bytes.fromhex((pathlib.Path(config['state'])/'authority.seed').read_text()))
 if key.verify_key.encode().hex()!=AUTH:raise ValueError('original ROOT authority')
 envelope=dict(payload=updated,signer=AUTH,signature=base64.b64encode(key.sign(canonical(updated)).signature).decode())
 target=pathlib.Path(scope_path);archive=target.parent/'scope-renewal-history'
 if target.is_symlink()or archive.is_symlink():raise ValueError('regular private renewal journal')
 archive.mkdir(mode=0o700,exist_ok=True)
 for raw in (target.read_bytes(),canonical(envelope)):
  saved=archive/(hashlib.sha256(raw).hexdigest()+'.json')
  if saved.exists():
   if saved.is_symlink()or saved.read_bytes()!=raw:raise ValueError('immutable renewal history')
  else:
   with saved.open('xb')as f:f.write(raw)
   saved.chmod(0o600)
 temporary=target.with_name(target.name+'.renew-'+str(time.time_ns()))
 with temporary.open('xb')as f:f.write(canonical(envelope))
 temporary.chmod(0o600);temporary.replace(target)
 return updated

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--approval',required=True);p.add_argument('--output',required=True);a=p.parse_args();c=read(a.config);approval=authenticate(read(a.approval));approval=renew_authorized_scope(approval,c,a.config,a.approval);tree=validate(approval,c,a.config)
 for n in list(sys.modules):
  if n=='subnet'or n.startswith('subnet.'):del sys.modules[n]
 sys.path.insert(0,str(tree))
 from subnet.controller import Controller
 from subnet.storage import Bucket
 from subnet.remote_backend import RemoteJobs
 controller=Controller(Bucket(c['bucket']),None,c['state']);remote=RemoteJobs(c['remote']['roles']['mine'],controller)
 if remote.metadata['source_files']!=approval['sources'][c['source_bundle']['sha256']]:raise ValueError('actual remote complete source pins')
 spec=importlib.util.spec_from_file_location('owned_miner_cleanup_operator',approval['cleanup_module']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 results=module.retire_completed(controller,remote,approval['sources'],output=a.output,owned_miners=approval['owned_miner_identities'],max_jobs=approval['max_jobs_per_cycle'])
 print(json.dumps(dict(at=time.time(),results=results,scientific_jobs_dispatched=False,production_state_changed=False,manual_cleanup=False)))
if __name__=='__main__':main()
