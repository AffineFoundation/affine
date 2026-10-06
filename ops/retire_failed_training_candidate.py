"""ROOT-scoped retirement of incomplete, durably preserved failed export bytes.

This CPU-only helper never repairs/promotes optimizer state. Call from the
recovery preflight with actual lease/process guards, not as manual cleanup.
"""
import fcntl,hashlib,json,os,re,secrets,stat,time
from pathlib import Path
from subnet.backend_jobs import signed
from subnet.training_receipts import sha
from subnet.storage import canonical

VERSION='failed-uncommitted-training-candidate-retirement-v1'

def publish_exclusive(path,value):
    """Publish complete journal bytes atomically, without replacing history."""
    temporary=path.with_name('.'+path.name+'.writing-'+secrets.token_hex(16))
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb')as stream:
            stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path,follow_symlinks=False)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

def _retire(envelope,authority,*,workspace,guard,now=None):
    value=signed(envelope,authority);now=time.time()if now is None else now
    fields={'version','execute_allowed','created_at','expires_at','original_signed_job','original_job_sha256','original_terminal','failure_evidence','durable_failure_evidence','transfer_directory','candidate_directory','pending_sha256','inventory','full_readbacks','journal','retire_failed_zero'}
    if set(value)!=fields or value['version']!=VERSION or value['execute_allowed']is not True or not value['created_at']<=now<value['expires_at'] or not 0<value['expires_at']-value['created_at']<=86400:raise ValueError('explicit bounded failed candidate retirement grant')
    original=signed(value['original_signed_job'],authority);jobid=original['job_id'];transport=original['persistent_training'];terminal=value['original_terminal'];w=value['failure_evidence']
    if sha(original)!=value['original_job_sha256'] or terminal.get('job_id')!=jobid or terminal.get('phase')!='failed' or terminal.get('exit_code')!=1 or type(terminal.get('exit_code'))is not int:raise ValueError('exact failed original job')
    if w!={'original_optimizer_updates':1,'candidate_committed':False,'original_report_absent':True,'complete_candidate_descriptor_absent':True,'original_processes_absent':True,'no_active_checkpoint_lease':True,'preserve_failure_history':True}:raise ValueError('post-update incomplete failure evidence')
    failure=signed(value['durable_failure_evidence'],authority)
    if set(failure)!={'version','original_job_sha256','key','sha256','size','bytes_read','complete'}or failure['version']!='full-object-readback-evidence-v1'or failure['original_job_sha256']!=sha(original)or failure['key']!='private/failed-training-evidence/'+jobid+'/failure-evidence.private.json'or re.fullmatch('[0-9a-f]{64}',failure['sha256']or '')is None or type(failure['size'])is not int or not 0<failure['size']<=8_000_000 or failure['bytes_read']!=failure['size']or failure['complete']is not True:raise ValueError('authenticated durable original failure evidence full readback')
    root=Path(workspace).resolve(strict=True);directory=Path(value['transfer_directory']);candidate=Path(value['candidate_directory']);journal=Path(value['journal'])
    from subnet.optimizer_state_cache import identifier
    if journal.parent!=root/'jobs'/jobid or journal.is_symlink():raise ValueError('exact immutable retirement journal path')
    resumed=journal.exists();saved=json.loads(journal.read_bytes())if resumed else None
    if resumed and (saved.get('version')!=VERSION or saved.get('grant')!=envelope):raise ValueError('immutable retirement grant changed')
    if type(value['retire_failed_zero'])is not bool:raise ValueError('explicit failed-zero preservation policy')
    completion=journal.with_name(journal.name+'.complete.json')
    if directory.is_symlink()or directory.resolve(strict=not resumed)!=directory or directory.parent!=root/'jobs'/jobid or not re.fullmatch(r'\.fp32-state-transfer-[0-9a-f]{32}',directory.name):raise ValueError('exact owned failed job transfer path')
    cache=root/'.optimizer-state-cache'
    if cache.is_symlink()or candidate!=cache/('candidate-'+identifier(jobid))or candidate.is_symlink()or candidate.resolve(strict=not resumed)!=candidate:raise ValueError('exact owned failed cache candidate path')
    pending=cache/'pending.json'
    catalog=saved['preserved_pending_catalogue']if resumed else json.loads(pending.read_bytes())
    raw=canonical(catalog)
    if pending.exists()and pending.read_bytes()!=raw:raise ValueError('original pending catalogue changed')
    if hashlib.sha256(raw).hexdigest()!=value['pending_sha256']or catalog['job_id']!=jobid or catalog['job_sha256']!=sha(original)or catalog['source_sha256']!=original['manifest']['payload']['source_bundle']['sha256']or catalog['descriptor_sha256']is not None:raise ValueError('authenticated incomplete owned pending catalogue')
    if not pending.exists()and not resumed:raise ValueError('original incomplete pending catalogue missing')
    rows=value['inventory'];receipts=value['full_readbacks']
    if not isinstance(rows,list)or not 1<=len(rows)<23 or not isinstance(receipts,list)or len(receipts)!=len(rows):raise ValueError('bounded incomplete shard inventory')
    names=[r['name']for r in rows]
    if len(set(names))!=len(names)or ('state-000000.safetensors'in names)!=value['retire_failed_zero']:raise ValueError('exact failed-zero retirement opt-in')
    present={p.name for folder in (directory,candidate)if folder.exists()for p in folder.iterdir()}
    allowed=set(names)|({'state-000000.safetensors'}if not value['retire_failed_zero']else set())
    if not present<=allowed or not resumed and not set(names)<=present:raise ValueError('unowned new or missing original partial shard')
    if set(catalog['files'])!=set(names)-{'state-000000.safetensors'}:raise ValueError('exact original pending owned shard catalogue')
    if guard(original,terminal,directory)is not True:raise ValueError('actual absent original processes and no active leases required')
    checked=[]
    for row,receipt in zip(rows,receipts):
        if set(row)!={'name','size','sha256','device','inode','uid','mtime_ns','ctime_ns','kind'}or not re.fullmatch(r'state-[0-9]{6}\.safetensors',row['name'])or row['name']not in transport['output_shards']:raise ValueError('exact failed shard allowlist')
        remote=signed(receipt,authority)
        key=('private/failed-training-evidence/'+jobid+'/'+row['name'])if row['name']=='state-000000.safetensors' else transport['output_namespace']+'/'+row['name']
        if remote!={'version':'full-object-readback-evidence-v1','original_job_sha256':sha(original),'key':key,'sha256':row['sha256'],'size':row['size'],'bytes_read':row['size'],'complete':True}:raise ValueError('genuine full R2 readback and failed-zero preservation required')
        if row['kind']not in ('original_transfer','pending_owned')or (row['name']=='state-000000.safetensors')!=(row['kind']=='original_transfer'):raise ValueError('failed zero versus owned successful PUT shard')
        if row['kind']=='pending_owned'and (catalog['files'].get(row['name'],{}).get('sha256')!=row['sha256']or catalog['files'][row['name']]['size']!=row['size']):raise ValueError('original successful PUT catalogue bytes')
        path=(directory if row['kind']=='original_transfer'else candidate)/row['name']
        if resumed and not path.exists()and not path.is_symlink():continue
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            st=os.fstat(fd)
            def matches(st):return stat.S_ISREG(st.st_mode)and st.st_nlink==1 and all(getattr(st,'st_ino'if k=='inode'else'st_'+k)==row[k]for k in ('size','inode','uid','mtime_ns','ctime_ns'))and st.st_dev==row['device']and st.st_uid==os.getuid()
            if not matches(st):raise ValueError('original shard inode/ownership changed')
            h=hashlib.sha256()
            with os.fdopen(os.dup(fd),'rb')as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=row['sha256']or not matches(os.fstat(fd)):raise ValueError('full original failed shard bytes changed')
            checked.append((path,row))
        finally:os.close(fd)
    if guard(original,terminal,directory)is not True:raise ValueError('actual absent original processes and no active leases required')
    # Persist the authenticated original bytes, receipts, failure and grant
    # BEFORE any unlink. Never remove the model, reports or original job.
    if not resumed:
        publish_exclusive(journal,dict(version=VERSION,grant=envelope,prepared_at=now,preserved_pending_catalogue=catalog))
    result=dict(version=VERSION,grant_sha256=sha(envelope),original_job_sha256=sha(original),retired_bytes=sum(row['size']for row in rows),retired_files=len(rows),durable_failure_history_preserved=True,optimizer_candidate_promoted=False)
    if completion.exists():
        if completion.is_symlink()or json.loads(completion.read_bytes())!=result or checked:raise ValueError('immutable completed retirement changed')
        return result
    freed=0
    for path,row in checked:
        if guard(original,terminal,directory)is not True:raise ValueError('process/lease became active during retirement')
        st=path.lstat()
        if not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or any(getattr(st,'st_ino'if k=='inode'else'st_'+k)!=row[k]for k in ('size','inode','uid','mtime_ns','ctime_ns'))or st.st_dev!=row['device']:raise ValueError('original shard changed before retirement')
        path.unlink();freed+=row['size']
    if value['retire_failed_zero']and directory.exists():directory.rmdir()
    if candidate.exists():candidate.rmdir()
    if pending.exists():
        if pending.read_bytes()!=raw:raise ValueError('pending catalogue changed during retirement')
        pending.unlink()
    publish_exclusive(completion,result)
    return result

def retire(envelope,authority,*,workspace,guard,now=None):
    root=Path(workspace).resolve(strict=True);lease=root/'.optimizer-state-cache'/'lease'
    fd=os.open(lease,os.O_RDWR|os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode)or os.fstat(fd).st_uid!=os.getuid():raise ValueError('owned original optimizer cache lease')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _retire(envelope,authority,workspace=workspace,guard=guard,now=now)
    finally:os.close(fd)
