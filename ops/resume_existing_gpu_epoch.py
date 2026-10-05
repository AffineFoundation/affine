"""Guarded observation of ONE already dispatched epoch; never opens a new epoch."""
import argparse,base64,hashlib,json,os,pathlib,re,sys,time
from nacl.signing import VerifyKey

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def load(p):
 p=pathlib.Path(p)
 if not p.is_file()or p.is_symlink()or p.stat().st_size>32*1024*1024:raise ValueError('bounded existing regular file')
 return json.loads(p.read_text())
def signed(v,authority):
 if set(v)!={'payload','signature','signer'}or v['signer']!=authority:raise ValueError('authority signer')
 VerifyKey(bytes.fromhex(authority)).verify(canonical(v['payload']),base64.b64decode(v['signature'],validate=True));return v['payload']
def guard(config_path,expected_hash,epoch,authority,source,working_directory,*,now=None):
 cp=pathlib.Path(config_path);cwd=pathlib.Path(working_directory)
 if cp.is_symlink()or not cp.is_file()or hashlib.sha256(cp.read_bytes()).hexdigest()!=expected_hash:raise ValueError('config drift')
 config=load(cp)
 if config['source_bundle']['sha256']!=source:raise ValueError('config source')
 state=pathlib.Path(config['state']);status=load(state/'controller.json');active=status.get('active')
 if active is None:return None
 if not isinstance(active,dict)or active.get('epoch')!=epoch or active.get('phase')not in ('train','after')or status.get('initial_published')is not True:raise ValueError('different active epoch/phase')
 if not re.fullmatch('[a-zA-Z0-9_-]+',epoch):raise ValueError('epoch path')
 record=load(state/'roles'/(epoch+'-train.json'));jid=record.get('job_id')
 if not isinstance(jid,str)or not re.fullmatch('[a-zA-Z0-9_-]+',jid):raise ValueError('original job path')
 job=signed(load(state/'roles'/(jid+'-job.json')),authority);manifest=signed(job['manifest'],authority)
 if job.get('role')!='train'or record.get('role')!='train'or job.get('job_id')!=jid or record.get('epoch')!=epoch or record.get('job_sha256')!=digest(job)or manifest.get('epoch')!=epoch or record.get('manifest_sha256')!=digest(manifest):raise ValueError('original signed job binding')
 if manifest['source_bundle']['sha256']!=source or record.get('checkpoint')!=manifest['checkpoint']['id']or status['checkpoint']['id']!=manifest['checkpoint']['id']:raise ValueError('original source/checkpoint')
 if record.get('source_files')!=job.get('source_files')or not job.get('source_files'):raise ValueError('original source inventory')
 for name,sha in job['source_files'].items():
  rel=pathlib.PurePosixPath(name)
  if rel.is_absolute()or '..'in rel.parts:raise ValueError('source path')
  f=cwd/name
  if not f.is_file()or f.is_symlink()or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:raise ValueError('sealed working source drift')
 # No request is created here. RemoteJobs' existing signed train record remains
 # authoritative and its original process/report recovery handles are reused.
 return dict(epoch=epoch,phase=active['phase'],job_id=jid,job_sha256=digest(job),config_path=str(cp.resolve()),working_directory=str(cwd.resolve()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--epoch',required=True);p.add_argument('--authority',required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--working-directory',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 result=guard(a.config,a.config_sha256,a.epoch,a.authority,a.source_sha256,a.working_directory)
 if result is None:return 0
 if pathlib.Path.cwd().resolve()!=pathlib.Path(a.working_directory).resolve():raise ValueError('unit working directory changed')
 if a.check_only:print(json.dumps(result,sort_keys=True));return 0
 os.execv(sys.executable,[sys.executable,'-B','-m','subnet.gpu_service','--config',result['config_path'],'--once'])
if __name__=='__main__':raise SystemExit(main())
