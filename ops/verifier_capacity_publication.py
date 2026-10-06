"""Default-off durable delivery of genuine full-readback model size metadata."""
import base64,fcntl,hashlib,json,logging,os,re,stat,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
from subnet.distributed_roles import authenticate
from subnet.storage import canonical
from subnet.training_receipts import sha
from ops.retire_failed_training_candidate import publish_exclusive
VERSION='verifier-capacity-publication-v1'
INVENTORY_VERSION='authenticated-model-byte-inventory-v1'

def checked_policy(envelope,authority):
    v=authenticate(envelope,authority)
    if set(v)!={'version','outbox','replicas'}or v['version']!=VERSION or not isinstance(v['replicas'],dict)or set(v['replicas'])!={'1','2','3','4','5','6','8'}:raise ValueError('exact seven-replica size publication policy')
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

def flush(controller,policy,*,replicate):
    if policy is None:return []
    v=checked_policy(policy,controller.authority.id);root=Path(v['outbox']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if root.resolve()!=root or root.is_symlink()or root.stat().st_uid!=os.getuid():raise ValueError('owned outbox')
    fd=os.open(root/'lease',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or st.st_uid!=os.getuid():raise ValueError('owned observer lease')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return []
        results=[];dispatched=0
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
        for path in paths[offset:]+paths[:offset]:
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
                if dispatched>=8:return finish()
                dispatched+=1
                try:
                    result=replicate(grant,config)
                    if result!=expected:raise ValueError('full immutable metadata install required')
                    publish_exclusive(destination,result);results.append(dict(replica=replica,status='complete'))
                except Exception as error:results.append(dict(replica=replica,status='deferred',error_type=type(error).__name__))
        return finish()
    finally:os.close(fd)

def backfill(controller,policy,*,roles):
    """Read genuine controller-owned publication journals, never partial PUTs.

    A size grant is descriptive only. Existing signed job/manifest authorization
    remains mandatory, even if its checkpoint has a size record.
    """
    if policy is None:return []
    roles=Path(roles);result=[]
    for path in sorted(roles.glob('*-checkpoint-publication.json')):
        st=path.lstat()
        if path.is_symlink()or not stat.S_ISREG(st.st_mode)or st.st_uid!=os.getuid()or st.st_nlink!=1 or st.st_size>1_000_000:raise ValueError('owned full publication journal')
        staged=json.loads(path.read_bytes())
        cp=dict(id=staged['checkpoint'],files={n:r['sha256']for n,r in staged['objects'].items()})
        result.append(enqueue(controller,policy,cp,staged))
    return result

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

def main():
    import argparse,time
    from nacl.signing import SigningKey
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);parser.add_argument('--authority',required=True);parser.add_argument('--seed-file',required=True);parser.add_argument('--roles',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
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
        try:
            policy=json.loads(Path(args.policy).read_bytes());backfill(controller,policy,roles=args.roles);result=flush(controller,policy,replicate=ssh_replicate)
            print(json.dumps(dict(metadata_only=True,installed=sum(x['status']=='complete'for x in result),deferred=sum(x['status']=='deferred'for x in result))),flush=True)
        except Exception as error:logging.warning('capacity metadata observer deferred (%s)',type(error).__name__)
        if args.once:return
        time.sleep(15)

if __name__=='__main__':main()
