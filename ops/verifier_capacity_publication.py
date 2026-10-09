"""Default-off durable delivery of genuine full-readback model size metadata."""
import base64,fcntl,hashlib,json,logging,os,re,secrets,sqlite3,stat,subprocess,tempfile,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from subnet.distributed_roles import authenticate
from subnet.storage import canonical
from subnet.training_receipts import sha
VERSION='verifier-capacity-publication-v1'
INVENTORY_VERSION='authenticated-model-byte-inventory-v1'

def publish_exclusive(path,value):
    """Same audited exclusive complete-file publication; no history rewrite."""
    temporary=path.with_name('.'+path.name+'.writing-'+secrets.token_hex(16))
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb')as stream:
            stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path,follow_symlinks=False)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:temporary.unlink(missing_ok=True)

def checked_policy(envelope,authority):
    v=authenticate(envelope,authority)
    if set(v)!={'version','outbox','replicas'}or v['version']!=VERSION or not isinstance(v['replicas'],dict)or set(v['replicas']) not in ({'1','2','3','4','5','6','8'}, {'1','2','3','4','5','6','8','9'}, {'1','2','4','5','6','8','9'}):raise ValueError('exact admitted verifier replica size publication policy')
    if not Path(v['outbox']).is_absolute():raise ValueError('owned metadata outbox')
    return v

def enqueue(controller,policy,checkpoint,staged):
    if policy is None:return None
    v=checked_policy(policy,controller.authority.id)
    if staged.get('checkpoint')!=checkpoint['id']or staged.get('operator_independent_hashes')is not True or set(staged.get('objects',{}))!=set(checkpoint['files']):raise ValueError('genuine exact checkpoint readback inventory')
    if not re.fullmatch('[0-9a-f]{64}',checkpoint['id']):raise ValueError('content-addressed model id')
    rows={}
    for name,expected in checkpoint['files'].items():
        row=staged['objects'][name]
        if Path(name).name!=name or name in ('.','..')or row.get('sha256')!=expected or type(row.get('bytes'))is not int or not 0<row['bytes']<=20_000_000_000:raise ValueError('full actual published model bytes')
        rows[name]=dict(sha256=expected,size=row['bytes'])
    descriptor=dict(id=checkpoint['id'],files=checkpoint['files'])
    grant=controller.signed(dict(version=INVENTORY_VERSION,checkpoint_id=checkpoint['id'],descriptor_sha256=sha(descriptor),files=rows))
    root=Path(v['outbox']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if root.resolve()!=root or root.is_symlink()or root.stat().st_uid!=os.getuid():raise ValueError('owned outbox')
    path=root/(checkpoint['id']+'.json')
    if path.exists():
        if path.is_symlink()or json.loads(path.read_bytes())!=grant:raise ValueError('immutable published size inventory')
    else:publish_exclusive(path,grant)
    return grant

def flush(controller,policy,*,replicate,preferred_checkpoint_ids=()):
    if policy is None:return []
    if type(preferred_checkpoint_ids)not in(tuple,list)or any(type(cp)is not str or not re.fullmatch('[0-9a-f]{64}',cp)for cp in preferred_checkpoint_ids)or len(set(preferred_checkpoint_ids))!=len(preferred_checkpoint_ids):raise ValueError('exact preferred checkpoint identities')
    v=checked_policy(policy,controller.authority.id);root=Path(v['outbox']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if root.resolve()!=root or root.is_symlink()or root.stat().st_uid!=os.getuid():raise ValueError('owned outbox')
    fd=os.open(root/'lease',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or st.st_uid!=os.getuid():raise ValueError('owned observer lease')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return []
        results=[];pending=[]
        paths=sorted(root.glob('*.json'));cursor=root/'.observer-cursor'
        if cursor.is_symlink():raise ValueError('owned observer cursor')
        offset=json.loads(cursor.read_bytes())['next']if cursor.exists()else 0
        if type(offset)is not int or offset<0:raise ValueError('bounded observer cursor')
        def finish():
            next_offset=(offset+1)%len(paths)if paths else 0
            handle,name=tempfile.mkstemp(dir=root,prefix='cursor-')
            try:
                with os.fdopen(handle,'wb')as stream:stream.write(canonical(dict(next=next_offset)));stream.flush();os.fsync(stream.fileno())
                os.replace(name,cursor);handle=os.open(root,os.O_RDONLY|os.O_DIRECTORY);os.fsync(handle);os.close(handle)
            finally:
                if os.path.exists(name):os.unlink(name)
            return results
        offset=offset%len(paths)if paths else 0
        priority={cp:i for i,cp in enumerate(preferred_checkpoint_ids)}
        ordered=sorted(paths[offset:]+paths[:offset],key=lambda path:priority.get(path.stem,len(priority)))
        for path in ordered:
            if path.is_symlink():raise ValueError('owned inventory grant')
            grant=json.loads(path.read_bytes());row=authenticate(grant,controller.authority.id)
            if row.get('version')!=INVENTORY_VERSION or row.get('checkpoint_id')!=path.stem:raise ValueError('exact outbox grant')
            receipts=root/path.stem;receipts.mkdir(mode=0o700,exist_ok=True)
            if receipts.is_symlink():raise ValueError('owned replica receipts')
            for replica,config in v['replicas'].items():
                destination=receipts/(replica+'.json')
                expected=dict(version='owned-capacity-inventory-install-v1',checkpoint_id=row['checkpoint_id'],grant_sha256=sha(grant),installed=True)
                if destination.exists():
                    if json.loads(destination.read_bytes())!=expected:raise ValueError('immutable install receipt')
                    continue
                pending.append((replica,config,grant,expected,destination))
                if len(pending)>=8:break
            if len(pending)>=8:break
        # One stalled route must not serialize other independent installations.
        # Keep the original outbox lock and immutable per-replica receipts.
        def deliver(item):
            replica,config,grant,expected,destination=item
            try:
                result=replicate(grant,config)
                if result!=expected:raise ValueError('full immutable metadata install required')
                publish_exclusive(destination,result)
                return dict(replica=replica,status='complete')
            except Exception as error:
                return dict(replica=replica,status='deferred',error_type=type(error).__name__)
        if pending:
            with ThreadPoolExecutor(max_workers=min(8,len(pending)))as pool:
                results.extend(pool.map(deliver,pending))
        return finish()
    finally:os.close(fd)

def current_published_models(*,publication_state,controller_state,authority,queue_path=None,now=None):
    """Pure read-only reconstruction from explicit genuine control paths."""
    publications=Path(publication_state);state_path=Path(controller_state)
    if not publications.is_absolute()or not state_path.is_absolute():raise ValueError('explicit absolute publication/controller paths')
    st=state_path.lstat()
    if state_path.is_symlink()or not stat.S_ISREG(st.st_mode)or st.st_uid!=os.getuid()or st.st_nlink!=1 or st.st_size>8_000_000:raise ValueError('owned current authority state')
    state=json.loads(state_path.read_bytes())
    if state.get('persistent_state_committed')is not True:raise ValueError('current genuinely committed authority checkpoint required')
    checkpoint=state['checkpoint']
    expected='public/checkpoints/'+checkpoint['id']+'/authorities/'+authority+'/checkpoint.json'
    if checkpoint.get('descriptor_key')!=expected:raise ValueError('current original authoritative model descriptor')
    required={checkpoint['id']:checkpoint}
    if queue_path is not None:
        queue=Path(queue_path)
        if not queue.is_absolute()or queue.is_symlink():raise ValueError('explicit readonly original queue')
        db=sqlite3.connect('file:'+str(queue)+'?mode=ro',uri=True,timeout=.25,isolation_level=None)
        try:
            for (raw,)in db.execute("SELECT envelope FROM jobs WHERE role='verify' AND status IN ('queued','leased') AND expires>?",(time.time()if now is None else now,)):
                job=authenticate(json.loads(raw),authority);manifest=authenticate(job['manifest'],authority)
                if job.get('role')!='verify':raise ValueError('authenticated original verifier queue record')
                cp=manifest['checkpoint']
                if cp['id']in required and required[cp['id']]['files']!=cp['files']:raise ValueError('immutable same checkpoint model map')
                required[cp['id']]=cp
        finally:db.close()
    journals=[]
    for path in sorted(publications.glob('*-checkpoint-publication.json')):
        st=path.lstat()
        if path.is_symlink()or not stat.S_ISREG(st.st_mode)or st.st_uid!=os.getuid()or st.st_nlink!=1 or st.st_size>1_000_000:raise ValueError('owned full publication journal')
        staged=json.loads(path.read_bytes())
        cp=required.get(staged.get('checkpoint'))
        if cp is None:continue
        if staged.get('operator_independent_hashes')is not True or {n:r['sha256']for n,r in staged['objects'].items()}!=cp['files']:raise ValueError('authoritative current/queued full-readback model map')
        journals.append((cp,staged))
    journals.sort(key=lambda row:row[0]['id']!=checkpoint['id'])
    yield from journals

def backfill(controller,policy,*,publication_state,controller_state,queue_path=None):
    """Current authority first; automatically follows each genuine closure."""
    if policy is None:return []
    return [enqueue(controller,policy,cp,staged)for cp,staged in current_published_models(publication_state=publication_state,controller_state=controller_state,authority=controller.authority.id,queue_path=queue_path)]

def install(RemoteJobs,policy):
    """Install only from a separately ROOT-scoped CPU launcher, default off."""
    if policy is None:return
    original=RemoteJobs.commit_remote_checkpoint
    def commit(self,manifest,staged):
        checkpoint=original(self,manifest,staged)
        try:enqueue(self.controller,policy,checkpoint,staged)
        except Exception as error:logging.warning('capacity metadata deferred (%s); independent observer retries genuine publication journal',type(error).__name__)
        return checkpoint
    RemoteJobs.commit_remote_checkpoint=commit

# Metadata-only remote install. No scientific imports, model reads or GPU probe.
INSTALL_SCRIPT='''import base64,hashlib,json,os,stat,sys,tempfile
from pathlib import Path
from nacl.signing import VerifyKey
v=json.load(sys.stdin);e=v['grant'];a=v['authority'];raw=json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
assert e['signer']==a
VerifyKey(bytes.fromhex(a)).verify(json.dumps(e['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode(),base64.b64decode(e['signature'],validate=True))
p=e['payload'];assert p['version']=='authenticated-model-byte-inventory-v1' and len(p['checkpoint_id'])==64 and all(c in '0123456789abcdef' for c in p['checkpoint_id'])
d=Path(v['directory']);assert d.is_absolute();d.mkdir(mode=0o700,parents=True,exist_ok=True);assert d.resolve()==d and d.stat().st_uid==os.getuid()
target=d/(p['checkpoint_id']+'.json')
if target.exists():
 s=target.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==os.getuid() and target.read_bytes()==raw
else:
 fd,name=tempfile.mkstemp(dir=d,prefix='inventory-')
 try:
  with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.link(name,target);fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 finally:os.unlink(name)
assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()
print(json.dumps(dict(version='owned-capacity-inventory-install-v1',checkpoint_id=p['checkpoint_id'],grant_sha256=hashlib.sha256(raw).hexdigest(),installed=True)))
'''

def ssh_replicate(grant,config):
    import shlex
    if set(config)!={'ssh_argv','python','directory','authority'}or not isinstance(config['ssh_argv'],list)or config['ssh_argv'][0]!='ssh':raise ValueError('ROOT-scoped exact metadata route')
    payload=canonical(dict(grant=grant,authority=config['authority'],directory=config['directory']))
    completed=subprocess.run(config['ssh_argv']+[shlex.quote(config['python'])+' -B -c '+shlex.quote(INSTALL_SCRIPT)],input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=True)
    return json.loads(completed.stdout)

def main(*,guard=None):
    import argparse,time
    from nacl.signing import SigningKey
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);parser.add_argument('--authority',required=True);parser.add_argument('--seed-file',required=True);parser.add_argument('--publication-state',required=True);parser.add_argument('--controller-state',required=True);parser.add_argument('--queue',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    # This operator must itself be launched under a reviewed trusted-service
    # scope that pins argv/seed path/roles/operator source. It grants no job.
    checked_policy(json.loads(Path(args.policy).read_bytes()),args.authority)
    seed=Path(args.seed_file)
    if seed.is_symlink()or seed.stat().st_uid!=os.getuid()or seed.stat().st_mode&0o077:raise ValueError('private existing controller authority seed')
    key=SigningKey(bytes.fromhex(seed.read_text().strip()))
    if key.verify_key.encode().hex()!=args.authority:raise ValueError('existing authority identity')
    def sign(value):return dict(payload=value,signer=args.authority,signature=base64.b64encode(key.sign(canonical(value)).signature).decode())
    controller=SimpleNamespace(authority=SimpleNamespace(id=args.authority),signed=sign)
    while True:
        if guard is not None:guard()
        try:
            policy=json.loads(Path(args.policy).read_bytes());grants=backfill(controller,policy,publication_state=args.publication_state,controller_state=args.controller_state,queue_path=args.queue);preferred=list(dict.fromkeys(authenticate(grant,args.authority)['checkpoint_id']for grant in grants));result=flush(controller,policy,replicate=ssh_replicate,preferred_checkpoint_ids=preferred)
            print(json.dumps(dict(metadata_only=True,installed=sum(x['status']=='complete'for x in result),deferred=sum(x['status']=='deferred'for x in result))),flush=True)
        except Exception as error:logging.warning('capacity metadata observer deferred (%s)',type(error).__name__)
        if args.once:return
        time.sleep(15)

if __name__=='__main__':main()
