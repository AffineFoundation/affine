"""Hydrate an operator-pinned checkpoint through scoped read-only capabilities.

No bucket credentials, model loading, GPU work, optimizer, or publication.
The signed plan binds the full checkpoint inventory and this exact helper.
Existing caches and partial downloads are retained on any failed check.
"""
import sys
import argparse,base64,fcntl,hashlib,json,os,re,stat,time
from pathlib import Path
from urllib.parse import parse_qs,unquote,urlparse
import requests
from nacl.signing import VerifyKey
MAX_FILE=20*1024**3;MAX_TOTAL=64*1024**3

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
 return h.hexdigest()
def authenticate(envelope,authority):
 if envelope.get('signer')!=authority or re.fullmatch('[0-9a-f]{64}',authority or '') is None:raise ValueError('approved signer')
 VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True));return envelope['payload']
def read_url(url,cp,name):
 p=urlparse(url);q=parse_qs(p.query)
 if p.scheme!='https' or not (p.hostname or '').endswith('.r2.cloudflarestorage.com') or p.username or p.password or p.fragment or p.port not in (None,443) or q.get('X-Amz-Algorithm')!=['AWS4-HMAC-SHA256'] or len(q.get('X-Amz-Signature',[]))!=1 or not re.fullmatch('[0-9a-f]{64}',q['X-Amz-Signature'][0]):raise ValueError('scoped direct signed R2 read')
 path=unquote(p.path)
 if '..' in path.split('/') or not path.endswith('/public/checkpoints/'+cp+'/'+name):raise ValueError('read URL exact checkpoint/file')
 return url

def validate(plan,authority,role,uuid,now=None):
 job=authenticate(plan,authority);now=time.time() if now is None else now
 if job.get('kind')!='immutable-checkpoint-read-hydration-v1' or job.get('role')!=role or job.get('retained_UUID')!=uuid or job.get('helper_sha256')!=sha(__file__):raise ValueError('approved scoped helper/actor')
 for k in ['GPU_runs','optimizer_runs','chain_transactions','publication_writes']:
  if type(job.get(k)) is not int or job[k]!=0:raise ValueError('CPU-only hydration')
 created,expires=job.get('created_at'),job.get('expires_at')
 if type(created) not in (int,float) or type(expires) not in (int,float) or not created<=now<expires or not 0<expires-created<=3600:raise ValueError('original read plan deadline')
 descriptor=authenticate(job['checkpoint_descriptor'],job['checkpoint_authority']);cp=descriptor.get('id');files=descriptor.get('files')
 if set(descriptor)!=set(['id','files']) or cp!=job.get('checkpoint') or re.fullmatch('[0-9a-f]{64}',cp or '') is None or not isinstance(files,dict) or not 1<=len(files)<=32 or hashlib.sha256(canonical(files)).hexdigest()!=cp:raise ValueError('normal signed complete learned descriptor')
 if job.get('checkpoint_descriptor_sha256')!=hashlib.sha256(canonical(job['checkpoint_descriptor'])).hexdigest():raise ValueError('descriptor envelope exact binding')
 if set(job['objects'])!=set(files) or set(job['read_urls'])!=set(files):raise ValueError('complete object/read inventory')
 total=0
 for name,digest in files.items():
  if not re.fullmatch('[A-Za-z0-9_.-]+',name) or name.startswith('.') or re.fullmatch('[0-9a-f]{64}',digest or '') is None:raise ValueError('approved file name/hash')
  meta=job['objects'][name]
  if set(meta)!=set(['sha256','bytes']) or meta['sha256']!=digest or type(meta['bytes']) is not int or not 0<meta['bytes']<=MAX_FILE:raise ValueError('actual root size/hash receipts')
  total+=meta['bytes'];read_url(job['read_urls'][name],cp,name)
 if total>MAX_TOTAL or 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):raise ValueError('bounded complete model')
 destination=job.get('destination')
 if not isinstance(destination,str) or not Path(destination).is_absolute() or '..' in Path(destination).parts:raise ValueError('signed exact role-local destination')
 if job.get('allows_concurrent_scientific_reads') is not True:raise ValueError('explicit isolated concurrent-read scope')
 return job,descriptor

def complete(path,files,objects):
 path=Path(path)
 if not path.is_absolute() or not path.is_dir() or path.is_symlink() or path.resolve()!=path or any(p.is_symlink() or not p.is_file() for p in path.iterdir()) or set(p.name for p in path.iterdir())!=set(files):return False
 for name,digest in files.items():
  if (path/name).stat().st_size!=objects[name]['bytes'] or sha(path/name)!=digest:return False
 return True

def save(path,body):
 temp=path.with_suffix('.tmp');temp.write_bytes(canonical(body));temp.chmod(0o600);temp.replace(path)

def download(session,url,partial,expected_bytes,expected_sha,deadline,clock=time.time):
 partial=Path(partial)
 if partial.is_symlink() or (partial.exists() and not partial.is_file()):raise ValueError('regular preserved partial')
 offset=partial.stat().st_size if partial.exists() else 0
 if offset>expected_bytes:raise ValueError('partial exceeds approved object')
 if offset==expected_bytes:
  if sha(partial)!=expected_sha:raise ValueError('preserve mismatching complete partial')
  return {'resumed_bytes':offset,'downloaded_bytes':0}
 if clock()>=deadline:raise ValueError('read plan expired; preserve partial')
 headers={'Accept-Encoding':'identity'}
 if offset:headers['Range']='bytes='+str(offset)+'-'
 with session.get(url,headers=headers,stream=True,timeout=(30,180),allow_redirects=False) as response:
  expected_status=206 if offset else 200
  if response.status_code!=expected_status or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('exact GET/range status/encoding')
  if offset and response.headers.get('Content-Range')!='bytes '+str(offset)+'-'+str(expected_bytes-1)+'/'+str(expected_bytes):raise ValueError('exact resumed object range')
  length=response.headers.get('Content-Length')
  if length is not None and (not length.isdecimal() or int(length)!=expected_bytes-offset):raise ValueError('exact remaining length')
  # Append only to retained same-object partial; never truncate/overwrite any existing bytes.
  with partial.open('ab' if offset else 'xb') as stream:
   partial.chmod(0o600);written=offset
   for part in response.iter_content(1024**2):
    if clock()>=deadline:raise ValueError('read plan expired during transfer')
    if not part:continue
    if written+len(part)>expected_bytes:raise ValueError('object size bound')
    stream.write(part);written+=len(part)
   stream.flush();os.fsync(stream.fileno())
 if partial.stat().st_size!=expected_bytes or sha(partial)!=expected_sha:raise ValueError('complete downloaded size/SHA mismatch; preserve')
 return {'resumed_bytes':offset,'downloaded_bytes':expected_bytes-offset}

def acquire_lock(parent,checkpoint):
 parent=Path(parent);parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 path=parent/('.'+checkpoint+'.hydrate-lock')
 fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 lock=os.fdopen(fd,'a+b')
 if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):lock.close();raise ValueError('regular local hydration lock')
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();raise ValueError('same checkpoint hydration already active') from None
 return lock
def main():
 os.umask(0o077)
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--operator',required=True);p.add_argument('--role',required=True);p.add_argument('--retained-UUID',required=True);p.add_argument('--destination',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 if sha(a.plan)!=a.plan_sha256:raise ValueError('exact root signed read plan file')
 envelope=json.loads(Path(a.plan).read_text());job,descriptor=validate(envelope,a.operator,a.role,a.retained_UUID);dest=Path(a.destination);out=Path(a.output);files=descriptor['files'];objects=job['objects']
 if str(dest)!=job['destination'] or not dest.is_absolute() or dest.is_symlink() or any(p.is_symlink() for p in dest.parents) or out.exists():raise ValueError('fresh private receipt/nonsymlink paths')
 lock=acquire_lock(dest.parent,descriptor['id']) # Kernel releases on actual process exit; no stale lock deletion.
 reused=False;receipts={}
 if dest.exists():
  if not complete(dest,files,objects):raise ValueError('existing incomplete/wrong cache preserved; use a new destination')
  reused=True
 else:
  stage=dest.parent/('.'+descriptor['id']+'.hydrate-'+a.plan_sha256[:16]);marker=stage/'binding.json';body={'plan_sha256':a.plan_sha256,'checkpoint':descriptor['id'],'role':a.role,'retained_UUID':a.retained_UUID,'descriptor_sha256':job['checkpoint_descriptor_sha256']}
  if stage.exists():
   if stage.is_symlink() or not marker.is_file() or json.loads(marker.read_text())!=body:raise ValueError('preserved staging identity collision')
  else:stage.mkdir(mode=0o700,parents=True);save(marker,body)
  work=stage/'objects';work.mkdir(mode=0o700,exist_ok=True)
  if work.is_symlink() or any(p.is_symlink() or not p.is_file() or p.name not in set(files)|{n+'.partial' for n in files} for p in work.iterdir()):raise ValueError('regular exact scoped staging')
  remaining=sum(max(0,meta['bytes']-(work/(name+'.partial')).stat().st_size if (work/(name+'.partial')).exists() else meta['bytes']) for name,meta in objects.items() if not (work/name).exists())
  capacity=os.statvfs(stage);free=capacity.f_bavail*capacity.f_frsize
  if free<remaining+2*1024**3:raise ValueError('actual remaining bytes+reserve capacity')
  with requests.Session() as session:
   for name,digest in files.items():
    final=work/name;partial=work/(name+'.partial')
    if final.exists():
     if final.is_symlink() or final.stat().st_size!=objects[name]['bytes'] or sha(final)!=digest:raise ValueError('preserve invalid prior staged member')
     receipts[name]={'already_staged':True};continue
    try:receipt=download(session,job['read_urls'][name],partial,objects[name]['bytes'],digest,job['expires_at'])
    except Exception:
     save(stage/'last-failure.json',{'file':name,'error_type':sys.exc_info()[0].__name__,'partial_bytes':partial.stat().st_size if partial.exists() else 0,'observed_at':time.time()});raise ValueError('scoped read failed; original partial and typed diagnostic preserved') from None
    partial.rename(final);receipts[name]=receipt;save(stage/'progress.json',{'objects':receipts,'plan_sha256':a.plan_sha256,'observed_at':time.time()})
  if not complete(work,files,objects):raise ValueError('full staged checkpoint readback')
  work.rename(dest) # Fresh destination only; original cache never modified.
 if not complete(dest,files,objects):raise ValueError('actual final full readback')
 if 'torch' in sys.modules:raise ValueError('pure hydration unexpectedly imported Torch')
 out.parent.mkdir(mode=0o700,parents=True,exist_ok=True);save(out,{'CPU_only':True,'torch_imported':'torch' in sys.modules,'model_loaded':False,'GPU_runs':0,'chain_transactions':False,'publication_writes':0,'checkpoint':descriptor['id'],'objects':objects,'actual_files':{n:sha(dest/n) for n in files},'actual_local_destination':str(dest),'reused_existing_cache_only_after_full_hashes':reused,'read_plan_sha256':a.plan_sha256,'descriptor_sha256':job['checkpoint_descriptor_sha256'],'role':a.role,'retained_UUID':a.retained_UUID,'progress':receipts,'completed_at':time.time()})
 print(json.dumps({'CPU_only':True,'checkpoint':descriptor['id'],'files':len(files),'total_bytes':sum(m['bytes'] for m in objects.values()),'receipt_sha256':sha(out)}))
if __name__=='__main__':main()
