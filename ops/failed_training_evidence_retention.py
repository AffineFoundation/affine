"""Default-off failed-export evidence archival after genuine recovery durability.

This operator-only lifecycle never repairs/promotes incomplete optimizer state.
Archive every scoped original byte and fully read it back before retiring it.
The immutable job, failed terminal, logs and recovery records stay untouched.
"""
import fcntl,hashlib,json,os,re,stat,time
from pathlib import Path
from subnet.backend_jobs import signed
from subnet.storage import canonical
from subnet.training_receipts import sha
from subnet import training_startup_recovery as recovery
from subnet.persistent_training_protocol import validate_output
from subnet.trainer_cache_lifecycle import live_original
from ops.retire_failed_training_candidate import publish_exclusive
from ops.training_retention import unreferenced
VERSION='failed-evidence-after-durable-recovery-retention-v1'

def identity(st):
    return st.st_mode,st.st_dev,st.st_ino,st.st_size,st.st_uid,st.st_nlink,st.st_mtime_ns,st.st_ctime_ns

def committed_replacement(value,authority,root,original):
    ack=signed(value['durable_recovery_ACK'],authority)
    if set(ack)!={'version','job_id','job_sha256','report_sha256','input_checkpoint','input_cache','new_checkpoint','trainer_state','authority_state_committed'}or ack.get('version')!='durable-original-trainer-cache-ACK-v1'or ack.get('authority_state_committed')is not True:raise ValueError('genuine recovery durability ACK required')
    job=signed(json.loads((root/(ack['job_id']+'.json')).read_bytes()),authority);m=signed(job['manifest'],authority)
    declaration=recovery.validate(job,m,authority)
    if declaration['version']!=recovery.POST_UPDATE_VERSION or declaration['original_job_sha256']!=sha(original):raise ValueError('same failed original explicit post-update recovery')
    report=json.loads((root/'jobs'/job['job_id']/'report.json').read_bytes());status=json.loads((root/'runner-status'/(job['job_id']+'.json')).read_bytes())
    state=report['persistent_training_state'];descriptor=validate_output(state['descriptor'],job,m)
    if (sha(job)!=ack['job_sha256']or sha(report)!=ack['report_sha256']or report.get('role')!='train'or report.get('operator')!=authority or report.get('epoch')!=m['epoch']or report.get('checkpoint')!=m['checkpoint']['id']or report.get('job_id')!=job['job_id']or report.get('job_sha256')!=sha(job)or report.get('success')is not True or report.get('new_checkpoint')!=ack['new_checkpoint']or ack['input_checkpoint']!=m['checkpoint']or state['descriptor_sha256']!=sha(descriptor)or state['descriptor_sha256']!=ack['trainer_state']['descriptor_sha256']or state['namespace']!=ack['trainer_state']['namespace']or descriptor['optimizer_steps']!=ack['trainer_state']['optimizer_steps']or descriptor['inference_checkpoint']!=ack['new_checkpoint']['id']):raise ValueError('exact committed replacement job/report/state lineage')
    if any(type(status.get(k))is not int or status[k]<=1 or not re.fullmatch('[0-9]+',str(status.get(k+'_ticks','')))for k in ('runner_pid','child_pid')):raise ValueError('actual replacement wait handles required')
    if status.get('job_id')!=job['job_id']or status.get('phase')!='complete'or status.get('exit_code')!=0 or live_original(status):raise ValueError('replacement actual terminal required')
    return ack

def retire(envelope,authority,*,workspace,archive,policy=None,now=None):
    if policy is None:return dict(status='disabled',retired_bytes=0)
    if policy!={'version':VERSION}:raise ValueError('explicit failed-evidence lifecycle opt-in')
    v=signed(envelope,authority);now=time.time()if now is None else now
    required={'version','execute_allowed','created_at','expires_at','original_signed_job','original_terminal','durable_recovery_ACK','directory','files','journal'}
    if set(v)!=required or v['version']!=VERSION or v['execute_allowed']is not True or not v['created_at']<=now<v['expires_at']or not 0<v['expires_at']-v['created_at']<=86400:raise ValueError('bounded explicit evidence retirement scope')
    root=Path(workspace).resolve(strict=True);original=signed(v['original_signed_job'],authority);jobid=original['job_id']
    if original.get('role')!='train'or not re.fullmatch('[A-Za-z0-9_-]+',jobid):raise ValueError('original failed train identity')
    terminal=json.loads((root/'runner-status'/(jobid+'.json')).read_bytes())
    if terminal!=v['original_terminal']or terminal.get('job_id')!=jobid or terminal.get('phase')!='failed'or terminal.get('exit_code')!=1 or live_original(terminal):raise ValueError('original failed terminal preserved')
    actual=signed(json.loads((root/(jobid+'.json')).read_bytes()),authority)
    if actual!=original:raise ValueError('immutable original signed job')
    committed_replacement(v,authority,root,original)
    for p in(root/'runner-status').glob('*.json'):
        if live_original(json.loads(p.read_bytes())):return dict(status='deferred',reason='workspace-role-in-flight',retired_bytes=0)
    directory=Path(v['directory']);journal=Path(v['journal']);owner=root/'jobs'/jobid
    if directory.parent!=owner or not re.fullmatch(r'\.fp32-state-transfer-[0-9a-f]{32}',directory.name)or directory.is_symlink()or directory.resolve()!=directory or journal.parent!=owner or journal.is_symlink():raise ValueError('exact original owned transfer/journal')
    rows=v['files'];names=[x.get('name')for x in rows]if isinstance(rows,list)else[]
    if not 1<=len(names)<=24 or len(set(names))!=len(names)or 'state-000000.safetensors'not in names:raise ValueError('bounded failed-zero and evidence allowlist')
    for row in rows:
        if set(row)!={'name','size','sha256','device','inode','uid','mtime_ns','ctime_ns'}or not re.fullmatch(r'(?:state-000000\.safetensors|(?:evidence|failure)-[0-9]{6}\.json)',row['name'])or type(row['size'])is not int or not 0<row['size']<=(5_000_000_000 if row['name']=='state-000000.safetensors'else 16384)or not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('exact scoped original evidence bytes')
    completion=journal.with_name(journal.name+'.complete.json');resume=journal.exists()
    if resume:
        if json.loads(journal.read_bytes())!={'version':VERSION,'grant':envelope}:raise ValueError('same immutable evidence retirement grant')
    elif any(not(directory/x).exists()for x in names):raise ValueError('missing original before retirement journal')
    if directory.exists()and not {p.name for p in directory.iterdir()}<=set(names):raise ValueError('unowned transfer member preserved')
    def verify(path,row):
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            st=os.fstat(fd)
            if not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or st.st_uid!=os.getuid()or st.st_dev!=row['device']or any(getattr(st,'st_ino'if k=='inode'else'st_'+k)!=row[k]for k in ('size','inode','uid','mtime_ns','ctime_ns')):raise ValueError('owned original evidence identity changed')
            h=hashlib.sha256()
            with os.fdopen(os.dup(fd),'rb')as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            if h.hexdigest()!=row['sha256']or identity(os.fstat(fd))!=identity(st):raise ValueError('full original evidence SHA changed')
            return identity(st)
        finally:os.close(fd)
    # Use the same workspace optimizer lease; never interrupt an active job.
    fd=os.open(root/'.optimizer-state-cache'/'lease',os.O_RDWR|os.O_NOFOLLOW)
    try:
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode)or st.st_uid!=os.getuid():raise ValueError('owned optimizer lease')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verified={}
        if not resume:
            verified={directory/row['name']:verify(directory/row['name'],row)for row in rows}
            unreferenced(verified)
            publish_exclusive(journal,dict(version=VERSION,grant=envelope))
        for row in rows:
            path=directory/row['name'];receipt_path=journal.with_name(journal.name+'.archive-'+row['name']+'.json');key='private/failed-training-evidence/'+jobid+'/'+row['name']
            expected=dict(version='full-object-readback-evidence-v1',key=key,sha256=row['sha256'],size=row['size'],bytes_read=row['size'],complete=True)
            if not path.exists()and not path.is_symlink():
                if not resume or not receipt_path.exists()or json.loads(receipt_path.read_bytes())!=expected:raise ValueError('missing evidence without full archive journal')
                continue
            before=verified.get(path)or verify(path,row);unreferenced([path])
            if identity(path.lstat())!=before:raise ValueError('original evidence changed before archive')
            receipt=archive(path,key,row)
            if receipt!=expected:raise ValueError('full private archive readback required before retirement')
            if receipt_path.exists():
                if json.loads(receipt_path.read_bytes())!=expected:raise ValueError('immutable full archive receipt changed')
            else:publish_exclusive(receipt_path,receipt)
            committed_replacement(v,authority,root,original);unreferenced([path])
            if not v['created_at']<=(time.time()if now is None else now)<v['expires_at']:raise ValueError('archive completed after retirement authority expiry')
            if identity(path.lstat())!=before:raise ValueError('evidence changed before unlink')
            path.unlink()
        result=dict(status='complete',version=VERSION,retired_bytes=sum(x['size']for x in rows),retired_files=len(rows),incomplete_optimizer_promoted=False,original_failure_jobs_logs_and_recovery_records_preserved=True,all_original_evidence_full_private_archive_readback=True)
        if directory.exists():directory.rmdir()
        if completion.exists():
            if json.loads(completion.read_bytes())!=result:raise ValueError('immutable completion changed')
        else:publish_exclusive(completion,result)
        return result
    finally:os.close(fd)

def bucket_archive(bucket):
    """Private immutable stream PUT + full stream GET, no large memory buffer."""
    def archive(path,key,row):
        from botocore.exceptions import ClientError
        with path.open('rb')as f:
            try:bucket.client.put_object(Bucket=bucket.name,Key=key,Body=f,ContentLength=row['size'],IfNoneMatch='*')
            except ClientError as error:
                if str(error.response.get('Error',{}).get('Code'))not in ('412','PreconditionFailed'):raise
        response=bucket.client.get_object(Bucket=bucket.name,Key=key);body=response['Body'];h=hashlib.sha256();size=0
        try:
            while True:
                b=body.read(8*1024**2)
                if not b:break
                h.update(b);size+=len(b)
        finally:body.close()
        return dict(version='full-object-readback-evidence-v1',key=key,sha256=h.hexdigest(),size=size,bytes_read=size,complete=response['ResponseMetadata']['HTTPStatusCode']==200)
    return archive

def schedule_after_durability(controller,ack,policy,*,dispatch):
    """Opt-in asynchronous CPU hook; the genuine ACK is supplied by completion.

    ROOT signs each exact failure blueprint once. The already trusted controller
    may then create per-invocation scopes after a real durable ACK, without manual
    renewal. The remote dispatch must pin this operator module and original
    workspace. Its independent journal/lease makes retries safe; failures remain
    deferred and cannot block the next training/mining epoch.
    """
    if policy is None:return []
    if set(policy)!={'version','approved_failures'}or policy['version']!=VERSION or not isinstance(policy['approved_failures'],list)or not 1<=len(policy['approved_failures'])<=4:raise ValueError('bounded explicit failed-evidence scheduling policy')
    confirmed=signed(ack,controller.authority.id)
    if confirmed.get('version')!='durable-original-trainer-cache-ACK-v1'or confirmed.get('authority_state_committed')is not True:raise ValueError('schedule only after actual durability ACK')
    import threading,logging
    threads=[]
    for blueprint in policy['approved_failures']:
        v=signed(blueprint,controller.authority.id)
        if set(v)!={'version','original_signed_job','original_terminal','directory','files','journal'}or v['version']!=VERSION:raise ValueError('exact ROOT-approved original failure blueprint')
        original=signed(v['original_signed_job'],controller.authority.id);m=signed(original['manifest'],controller.authority.id)
        if confirmed.get('input_checkpoint')!=m['checkpoint']:continue
        now=time.time();grant=dict(v,execute_allowed=True,created_at=now,expires_at=now+3600,durable_recovery_ACK=ack);envelope=controller.signed(grant)
        def work(document=envelope):
            try:dispatch(document)
            except Exception as error:logging.warning('failed-evidence archival deferred: %s',type(error).__name__)
        thread=threading.Thread(target=work,name='durable-failed-evidence-retention',daemon=True);thread.start();threads.append(thread)
    return threads
