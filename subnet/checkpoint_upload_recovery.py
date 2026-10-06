"""Explicit ROOT authorization for a distinct transport-only publication attempt.

Original failed requests remain failed. This selector cannot retry training or
commit model/optimizer authority. Normal full readback still precedes commitment.
"""
import hashlib,json,math,re,time
from pathlib import Path
from .storage import canonical
from .backend_jobs import signed
VERSION='terminal-checkpoint-upload-recovery-v1'
def sha(value):return hashlib.sha256(canonical(value)).hexdigest()
def computation(manifest):
 import copy
 value=copy.deepcopy(manifest);value['checkpoint'].pop('read_urls',None)
 return value

def select_label(controller,manifest,remote_path,label):
 paths=getattr(controller,'checkpoint_upload_recovery_files',{})
 path=paths.get(manifest['epoch'])
 if path is None:return label
 envelope=json.loads(Path(path).read_bytes());p=signed(envelope,controller.authority.id)
 keys={'version','epoch','checkpoint','source_sha256','original_label','replacement_label','original_job_id','original_job_sha256','original_failure_sha256','remote_path','created_at','expires_at','computation_sha256','transport_path','transport_operator_sha256'}
 if set(p)!=keys or p['version']!=VERSION:raise ValueError('exact upload recovery declaration')
 if any(type(p[k])not in(int,float)or not math.isfinite(p[k])for k in ('created_at','expires_at'))or not 0<p['expires_at']-p['created_at']<=86400:raise ValueError('bounded upload recovery authorization')
 if p['epoch']!=manifest['epoch']or p['checkpoint']!=manifest['checkpoint']['id']or p['source_sha256']!=manifest['source_bundle']['sha256']or p['original_label']!=label or p['remote_path']!=remote_path:raise ValueError('exact upload recovery scope')
 replacement=p['replacement_label']
 if type(replacement)is not str or re.fullmatch('[A-Za-z0-9_-]{1,180}',replacement)is None or replacement==label:raise ValueError('distinct upload recovery label')
 roles=controller.state/'roles';record=json.loads((roles/(label+'.json')).read_bytes());jid=p['original_job_id']
 if record['job_id']!=jid or record['job_sha256']!=p['original_job_sha256']:raise ValueError('original upload reservation')
 original=signed(json.loads((roles/(jid+'-job.json')).read_bytes()),controller.authority.id)
 old=signed(original['manifest'],controller.authority.id)
 if original['role']!='upload'or original['job_id']!=jid or sha(original)!=p['original_job_sha256']or old['checkpoint']['files']!=manifest['checkpoint']['files']or old['checkpoint']['id']!=p['checkpoint']or old['source_bundle']['sha256']!=p['source_sha256']:raise ValueError('original failed upload request')
 if computation(old)!=computation(manifest)or sha(computation(old))!=p['computation_sha256']:raise ValueError('original upload computation context changed')
 if type(p['transport_path'])is not str or not Path(p['transport_path']).is_absolute()or type(p['transport_operator_sha256'])is not str or re.fullmatch('[0-9a-f]{64}',p['transport_operator_sha256'])is None:raise ValueError('pinned isolated CPU upload operator')
 failure=json.loads((roles/(jid+'-failure.json')).read_bytes())
 if sha(failure)!=p['original_failure_sha256']or failure.get('job_id')!=jid or failure.get('phase')!='failed'or type(failure.get('exit_code'))is not int or failure['exit_code']==0:raise ValueError('witnessed original upload failure')
 if not original['created_at']<=failure['started_at']<=failure['finished_at']<=p['created_at']:raise ValueError('original failure before new authorization')
 if (roles/(jid+'-report.json')).exists():raise ValueError('failed upload recovery forbidden after report')
 existing=roles/(replacement+'.json')
 if not existing.exists():
  if not p['created_at']<=time.time()<p['expires_at']:raise ValueError('upload recovery dispatch outside authorization')
 else:
  r=json.loads(existing.read_bytes());j=signed(json.loads((roles/(r['job_id']+'-job.json')).read_bytes()),controller.authority.id);m=signed(j['manifest'],controller.authority.id)
  if j['role']!='upload'or j['job_id']==jid or sha(j)!=r['job_sha256']or not p['created_at']<=j['created_at']<p['expires_at']or m['checkpoint']['id']!=p['checkpoint']or m['checkpoint']['files']!=old['checkpoint']['files']or m['source_bundle']['sha256']!=p['source_sha256']or computation(m)!=computation(old):raise ValueError('original authorized recovery attempt')
 return replacement

# Credentialless CPU transport adapter. The immutable scientific runner/backend
# execute unchanged; only their exact authorized upload PUT calls are wrapped.
TRANSPORT_PROGRAM = r'''import base64,fcntl,hashlib,json,os,pathlib,re,runpy,stat,subprocess,sys,time
from nacl.signing import VerifyKey
SOURCE=__SOURCE__
DECLARATION=__DECLARATION__
canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
args=sys.argv[1:];backend=bool(args and args[0]=='--backend')
if backend:args=args[1:]
elif args[:3]==['-B','-m','subnet.remote_runner']:args=args[3:]
else:raise ValueError('upload-only CPU adapter invocation')
authority=args[args.index('--authority')+1];jobpath=pathlib.Path(args[0]);cache=pathlib.Path(args[args.index('--checkpoint-cache')+1])
def signed(d):
 if d['signer']!=authority:raise ValueError('ROOT upload authority')
 VerifyKey(bytes.fromhex(authority)).verify(canon(d['payload']),base64.b64decode(d['signature'],validate=True));return d['payload']
p=signed(json.loads(pathlib.Path(DECLARATION).read_bytes()));job=signed(json.loads(jobpath.read_bytes()));m=signed(job['manifest']);deadline=min(p['expires_at'],job['expires_at'])
if p['version']!='terminal-checkpoint-upload-recovery-v1'or job['role']!='upload'or not re.fullmatch(re.escape(p['replacement_label'])+'-[0-9a-f]{8}',job['job_id']):raise ValueError('distinct upload-only job')
comp=json.loads(canon(m));comp['checkpoint'].pop('read_urls',None)
if hashlib.sha256(canon(comp)).hexdigest()!=p['computation_sha256']or m['source_bundle']['sha256']!=p['source_sha256']or m['checkpoint']['id']!=p['checkpoint']or str(cache)!=p['remote_path']:raise ValueError('exact recovery computation/path')
if pathlib.Path(__file__).absolute()!=pathlib.Path(p['transport_path'])or hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()!=p['transport_operator_sha256']:raise ValueError('CPU transport operator drift')
if not p['created_at']<=job['created_at']<p['expires_at']or time.time()>=deadline:raise ValueError('new original upload lifetime')
for n,h in job['source_files'].items():
 f=pathlib.Path(SOURCE)/n
 if f.is_symlink()or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise ValueError('unchanged sealed scientific modules')
if set(job['put_urls'])!=set(m['checkpoint']['files']):raise ValueError('exact model destination inventory')
sys.dont_write_bytecode=True;sys.path.insert(0,SOURCE)
if not backend:
 original=subprocess.Popen
 def spawn(command,*a,**kw):
  if command[:4]!=[sys.executable,'-B','-m','subnet.backend_jobs']:raise ValueError('only original sealed upload backend child')
  return original([sys.executable,'-I','-B',__file__,'--backend',*command[4:]],*a,**kw)
 subprocess.Popen=spawn
 sys.argv=['subnet.remote_runner',*args];runpy.run_module('subnet.remote_runner',run_name='__main__')
else:
 import requests,threading
 traffic_lock=threading.Lock()
 original_put=requests.put;original_get=requests.get
 rows=[];started=time.monotonic();total=0
 def stamp(v):return(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns,v.st_mode,v.st_uid,v.st_nlink)
 def record(row):
  row=dict(row,at=time.time());rows.append(row)
  log=pathlib.Path(__file__).parent/(job['job_id']+'-transport.jsonl');fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600)
  st=os.fstat(fd)
  if not stat.S_ISREG(st.st_mode)or st.st_uid!=os.getuid()or st.st_nlink!=1 or st.st_mode&0o077:os.close(fd);raise ValueError('owned transport journal')
  try:
   data=canon(row)+b'\n';written=os.write(fd,data)
   if written!=len(data):raise OSError('short journal write')
   os.fsync(fd)
  finally:os.close(fd)
 def account(amount):
  global total
  with traffic_lock:
   total+=amount
   if total>128*1024**3:raise ValueError('bounded all-attempt transport bytes')
 def fresh():
  if time.time()>=deadline:raise TimeoutError('authorized upload transport expired')
 def retryable(error):
  # Certificate verification failures never become transient transport claims.
  if isinstance(error,requests.exceptions.SSLError):return 'EOF'in str(error)and 'CERTIFICATE_VERIFY_FAILED'not in str(error)
  return isinstance(error,(requests.exceptions.Timeout,requests.exceptions.ConnectionError))
 def put(url,**kw):
  nonlocal_dummy=None
  global total
  matches=[n for n,u in job['put_urls'].items()if u==url]
  if len(matches)!=1:raise ValueError('unapproved model PUT capability')
  name=matches[0];body=kw['data'];fd=body.fileno();before=os.fstat(fd);path=cache/name
  if not stat.S_ISREG(before.st_mode)or before.st_nlink!=1 or before.st_uid!=os.getuid()or path.is_symlink()or stamp(before)!=stamp(path.stat()):raise ValueError('owned immutable upload FD')
  body.seek(0);h=hashlib.file_digest(body,'sha256').hexdigest();body.seek(0)
  if h!=m['checkpoint']['files'][name]or stamp(os.fstat(fd))!=stamp(before):raise ValueError('full model SHA before transport')
  from urllib.parse import urlsplit,urlunsplit
  u=urlsplit(url);get_url=urlunsplit((u.scheme,u.netloc,u.path,'',''))
  # PUT URLs cannot authorize GET. ROOT provides separate short GET capabilities
  # in a signed adjacent transport capability envelope.
  caps=signed(json.loads(pathlib.Path(DECLARATION+'.GET.json').read_bytes()))
  if caps['declaration_sha256']!=hashlib.sha256(canon(p)).hexdigest()or caps['checkpoint']!=p['checkpoint']or set(caps['read_urls'])!=set(job['put_urls']):raise ValueError('exact separate model read capabilities')
  missing=False
  for attempt in range(4):
   fresh();t=time.monotonic();size=0;digest=hashlib.sha256()
   try:
    with original_get(caps['read_urls'][name],stream=True,timeout=(10,min(180,max(1,deadline-time.time()))),allow_redirects=False,headers={'Accept-Encoding':'identity'})as response:
     status=response.status_code
     if status==404:
      record(dict(member=name,phase='full-GET-existing',bytes=0,seconds=time.monotonic()-t,attempt=attempt+1,HTTP_status=404));missing=True;break
     if status in(408,429,500,502,503,504):raise requests.exceptions.ConnectionError('retryable read status')
     if status!=200 or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('model full-GET permanent status')
     for chunk in response.iter_content(1024*1024):
      fresh();size+=len(chunk);account(len(chunk))
      if size>before.st_size:raise ValueError('bounded model full-GET bytes')
      digest.update(chunk)
    if size!=before.st_size or digest.hexdigest()!=h:raise ValueError('immutable R2 model collision')
    if stamp(os.fstat(fd))!=stamp(before)or stamp(path.stat())!=stamp(before):raise ValueError('model mutation during GET')
    record(dict(member=name,phase='full-GET-existing',bytes=size,seconds=time.monotonic()-t,attempt=attempt+1,verified=True,skipped_PUT=True));r=requests.Response();r.status_code=204;return r
   except Exception as error:
    record(dict(member=name,phase='full-GET-existing',bytes=size,seconds=time.monotonic()-t,attempt=attempt+1,error_type=type(error).__name__))
    if not retryable(error)or attempt==3:raise RuntimeError('model read transport refused: '+type(error).__name__)from None
    time.sleep(min(2**attempt,max(0,deadline-time.time())))
  if not missing:raise ValueError('model existence not determined')
  for attempt in range(4):
   fresh()
   if stamp(os.fstat(fd))!=stamp(before)or stamp(path.stat())!=stamp(before):raise ValueError('model changed before rewind')
   body.seek(0);t=time.monotonic();status=None
   try:
    response=original_put(url,**dict(kw,timeout=(10,min(600,max(1,deadline-time.time()))),allow_redirects=False))
    status=response.status_code
    if status not in(200,201,204):
     if status in(408,429,500,502,503,504):raise requests.exceptions.ConnectionError('retryable PUT status')
     raise ValueError('permanent model PUT status')
    if stamp(os.fstat(fd))!=stamp(before)or stamp(path.stat())!=stamp(before):raise ValueError('model changed during PUT')
    account(body.tell())
    record(dict(member=name,phase='PUT',bytes=before.st_size,seconds=time.monotonic()-t,attempt=attempt+1,HTTP_status=status,completed=True));return response
   except Exception as error:
    account(body.tell())
    record(dict(member=name,phase='PUT',body_bytes_consumed=body.tell(),seconds=time.monotonic()-t,attempt=attempt+1,HTTP_status=status,error_type=type(error).__name__))
    if stamp(os.fstat(fd))!=stamp(before)or stamp(path.stat())!=stamp(before):raise ValueError('model changed on failed PUT')from None
    if not retryable(error)or attempt==3:raise RuntimeError('model PUT refused: '+type(error).__name__+' status '+str(status))from None
    time.sleep(min(2**attempt,max(0,deadline-time.time())))
 requests.put=put
 sys.argv=['subnet.backend_jobs',*args];runpy.run_module('subnet.backend_jobs',run_name='__main__')
'''

def transport_program(python,source,declaration):
    return '#!'+python+'\n'+TRANSPORT_PROGRAM.replace('__SOURCE__',repr(source)).replace('__DECLARATION__',repr(declaration))

def run_recovery_upload(controller,label,manifest,remote_path,put_urls):
    """Same immutable science/backend; explicit CPU transport adapter only."""
    path=getattr(controller,'checkpoint_upload_recovery_files',{}).get(manifest['epoch'])
    if path is None:return None
    p=signed(json.loads(Path(path).read_bytes()),controller.authority.id)
    from .remote_backend import save
    import os,shlex,types
    owner=controller.jobs.owners.get(remote_path,controller.jobs.initial_role)
    remote=controller.jobs.roles[owner]
    remote_declaration=str(Path(p['transport_path']).parent/'declaration.ROOT-SIGNED.json')
    program=transport_program(remote.python,remote.code,remote_declaration)
    if hashlib.sha256(program.encode()).hexdigest()!=p['transport_operator_sha256']:raise ValueError('approved CPU transport adapter bytes')
    local=Path(path).parent/'CPU-upload-transport.py'
    if local.exists()and local.read_text()!=program:raise ValueError('transport program immutable')
    if not local.exists():local.write_text(program);local.chmod(0o700)
    remote.command('mkdir -p '+shlex.quote(str(Path(p['transport_path']).parent)))
    remote.copy_to(local,p['transport_path']);remote.copy_to(path,remote_declaration)
    capfile=Path(path).with_name(Path(path).name+'.GET.json')
    caps=dict(declaration_sha256=sha(p),checkpoint=p['checkpoint'],read_urls={n:controller.bucket.presign('public/checkpoints/'+p['checkpoint']+'/'+n,expires=3600)for n in manifest['checkpoint']['files']})
    save(capfile,controller.signed(caps));remote.copy_to(capfile,remote_declaration+'.GET.json')
    remote.command('chmod 700 '+shlex.quote(p['transport_path']))
    original=remote.launch_runner
    def launch(identifier,remotejob,cache=None):
        import subprocess
        from .remote_backend import RemoteObservationTimeout,RemoteJobTerminalError
        arguments=[remote.python,'-I','-B',p['transport_path'],'-B','-m','subnet.remote_runner',remotejob,'--authority',controller.authority.id,'--workspace',remote.workspace]
        if cache:arguments+=['--checkpoint-cache',cache]
        code=("import json,os,subprocess,time;from pathlib import Path;"
          "root=Path("+repr(remote.workspace)+");marker=root/"+repr(identifier+'-dispatch-attempt.json')+";"
          "fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);"
          "os.write(fd,json.dumps({'job_id':"+repr(identifier)+",'attempted_at':time.time()}).encode());os.fsync(fd);os.close(fd);"
          "log=open(root/"+repr(identifier+'-runner.log')+",'ab');"
          "child=subprocess.Popen("+repr(arguments)+",cwd="+repr(remote.code)+",stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'));"
          "print(json.dumps({'pid':child.pid,'ticks':Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]}))")
        try:return remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(code),timeout=60)
        except subprocess.TimeoutExpired:
            status=remote.remote_status(identifier,timeout=30)
            if status['phase']=='failed':raise RemoteJobTerminalError('original CPU upload failed')
            if status['phase']not in('running','complete'):raise RemoteObservationTimeout(identifier,'CPU upload launch')from None
    remote.launch_runner=launch
    try:return remote.run(label,'upload',manifest,remote_path,put_urls=put_urls)
    finally:remote.launch_runner=original
