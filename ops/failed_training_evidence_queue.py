"""Default-off durable CPU queue for failed-export evidence, never training work."""
import fcntl,json,os,stat,time
from pathlib import Path
from subnet.backend_jobs import signed
from subnet.training_receipts import sha
from ops.retire_failed_training_candidate import publish_exclusive
from ops.failed_training_evidence_retention import VERSION
RENEWAL_VERSION='failed-evidence-retention-validation-renewal-v1'
QUEUE_VERSION='durable-failed-evidence-retention-queue-v1'

def enqueue(controller,ack,policy,*,queue_path,now=None):
    if policy is None:return []
    if set(policy)!={'version','approved_failures'}or policy['version']!=VERSION or not isinstance(policy['approved_failures'],list)or not 1<=len(policy['approved_failures'])<=4:raise ValueError('explicit bounded approved failures')
    confirmed=signed(ack,controller.authority.id)
    if confirmed.get('version')!='durable-original-trainer-cache-ACK-v1'or confirmed.get('authority_state_committed')is not True:raise ValueError('genuine committed ACK required')
    root=Path(queue_path);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    if root.is_symlink()or root.stat().st_uid!=os.getuid():raise ValueError('owned durable queue')
    queued=[];clock=time.time()if now is None else now
    for blueprint in policy['approved_failures']:
        v=signed(blueprint,controller.authority.id)
        if set(v)!={'version','original_signed_job','original_terminal','directory','files','journal'}or v['version']!=VERSION:raise ValueError('exact failure blueprint')
        original=signed(v['original_signed_job'],controller.authority.id);manifest=signed(original['manifest'],controller.authority.id)
        key=sha(blueprint);path=root/(key+'.json')
        if path.exists():
            old=json.loads(path.read_bytes());grant=signed(old['grant'],controller.authority.id)
            if old.get('version')!=QUEUE_VERSION or old.get('blueprint')!=blueprint or {k:grant[k]for k in v}!=v:raise ValueError('immutable pending failure changed')
            # A later checkpoint ACK cannot replace the real recovery ACK.
            queued.append(key);continue
        if confirmed.get('input_checkpoint')!=manifest['checkpoint']:continue
        grant=controller.signed(dict(v,execute_allowed=True,created_at=clock,expires_at=clock+3600,durable_recovery_ACK=ack))
        publish_exclusive(path,dict(version=QUEUE_VERSION,blueprint=blueprint,grant=grant));queued.append(key)
    return queued

def drain(controller,policy,*,queue_path,idle,dispatch,now=None):
    """Call at post-ACK idle boundary AND future idle observer ticks.

    dispatch(grant, renewal) must be an independently timeout-bounded, pinned
    remote CPU adapter. Never run from, or wait inside, a training update.
    """
    if policy is None:return []
    if set(policy)!={'version','approved_failures'}or policy['version']!=VERSION:raise ValueError('explicit queue policy')
    root=Path(queue_path)
    if not root.exists():return []
    if root.is_symlink()or root.stat().st_uid!=os.getuid():raise ValueError('owned durable queue')
    fd=os.open(root/'lease',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode)or st.st_uid!=os.getuid()or st.st_nlink!=1:raise ValueError('owned queue lease')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return []
        approved={sha(x):x for x in policy['approved_failures']};results=[]
        for path in sorted(root.glob('*.json')):
            key=path.stem
            if key not in approved:raise ValueError('unapproved queued failure')
            if path.is_symlink():raise ValueError('unowned queue document')
            doc=json.loads(path.read_bytes());grant=signed(doc['grant'],controller.authority.id)
            if doc.get('version')!=QUEUE_VERSION or doc.get('blueprint')!=approved[key]:raise ValueError('pending blueprint changed')
            blueprint=signed(approved[key],controller.authority.id)
            if {k:grant[k]for k in blueprint}!=blueprint:raise ValueError('queued grant changed')
            attempts=root/key;attempts.mkdir(mode=0o700,exist_ok=True)
            if attempts.is_symlink():raise ValueError('owned attempts')
            done=attempts/'complete'
            if done.exists():continue
            receipts=sorted(attempts.glob('attempt-*.json'));clock=time.time()if now is None else now
            if receipts:
                last=json.loads(receipts[-1].read_bytes())
                if clock<last['retry_after']:continue
            if idle()is not True:continue
            renewal=None
            if not grant['created_at']<=clock<grant['expires_at']:
                renewal=controller.signed(dict(version=RENEWAL_VERSION,execute_allowed=True,grant_sha256=sha(doc['grant']),created_at=clock,expires_at=clock+3600))
                renewal_path=attempts/('renewal-'+sha(renewal)+'.json')
                if not renewal_path.exists():publish_exclusive(renewal_path,renewal)
                elif json.loads(renewal_path.read_bytes())!=renewal:raise ValueError('immutable renewal changed')
            try:
                result=dispatch(doc['grant'],renewal)
                if not isinstance(result,dict)or result.get('status')not in ('complete','deferred'):raise ValueError('explicit remote retention result')
                if result['status']=='complete'and (result.get('all_original_evidence_full_private_archive_readback')is not True or result.get('incomplete_optimizer_promoted')is not False):raise ValueError('full forensic completion required')
            except Exception as error:result=dict(status='deferred',error_type=type(error).__name__)
            receipt=dict(version=QUEUE_VERSION,grant_sha256=sha(doc['grant']),at=clock,retry_after=clock+min(300,30*2**min(len(receipts),4)),result=result)
            publish_exclusive(attempts/('attempt-%06d.json'%len(receipts)),receipt)
            if result['status']=='complete':publish_exclusive(done,receipt)
            results.append(result)
        return results
    finally:os.close(fd)
