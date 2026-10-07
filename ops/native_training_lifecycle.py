"""CPU-only no-update closure and durable retirement of owned input copies.

Original capture, population, selection and signed jobs are never removed.
"""
import json
import os
from pathlib import Path
import stat
import threading
import time
from .native_training_eligibility import NativeNoUpdate,_canonical,_create,_load,bind_subset
from .native_training_outcome_filter import digest

NO_UPDATE_VERSION='native-eligibility-no-update-v1'
ACK_VERSION='native-eligibility-full-R2-readback-ack-v1'
OWNERSHIP_VERSION='native-eligibility-owned-document-v1'
LIFECYCLE_POLICY=dict(version='native-eligibility-lifecycle-v1',no_update_policy='signed-parent-preserving-no-update-v1',indeterminate_retry_policy='new-unopened-epoch-same-parent-only',retirement_policy='full-R2-ACK-owned-copy-only-v1',max_document_gets_per_cycle=32,poll_seconds=30,no_cheating_penalties=True)

def stamp(s):
    return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns,uid=s.st_uid,mode=stat.S_IMODE(s.st_mode),nlink=s.st_nlink)

def directory(controller,epoch):
    import re
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{1,220}',epoch):raise ValueError('native lifecycle epoch')
    root=Path(controller.state)/'native-outcome-eligibility'/epoch
    if any(p.is_symlink() for p in (root,*root.parents)) or root.stat().st_uid!=os.getuid():
        raise ValueError('owned canonical native lifecycle namespace')
    return root


def document_path(root,sha):
    import re
    if not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('native document digest path')
    return root/('document-'+sha+'.owned')/'document.json'

def verify_owned_document(controller,root,sha):
    from subnet.distributed_roles import authenticate
    path=document_path(root,sha);bundle=path.parent
    if bundle.is_symlink()or bundle.stat().st_uid!=os.getuid()or stat.S_IMODE(bundle.stat().st_mode)!=0o700:
        raise ValueError('native paired copy owned namespace')
    if set(p.name for p in bundle.iterdir())!={'document.json','ownership.ROOT-SIGNED.json'}:
        raise ValueError('native paired copy exact membership')
    own=authenticate(json.loads(_load(bundle/'ownership.ROOT-SIGNED.json')),controller.authority.id)
    expected=dict(version=OWNERSHIP_VERSION,name='document-'+sha+'.json',sha256=sha,stamp=stamp(path.lstat()))
    if own!=expected or stamp(path.lstat())['mode']!=0o600 or stamp(path.lstat())['uid']!=os.getuid()or stamp(path.lstat())['nlink']!=1:
        raise ValueError('native copy ownership or inode drift')
    if __import__('hashlib').sha256(_load(path,2_000_000)).hexdigest()!=sha:raise ValueError('native owned copy SHA')
    return own

def install_document_bundle(controller,root,sha,data):
    """Atomically expose raw document and its signed inode ownership together.

    Incomplete private staging directories are preserved, never adopted. A
    retry creates a fresh pair instead of accepting an unowned visible copy.
    """
    import ctypes,errno
    final=document_path(root,sha).parent
    if __import__('hashlib').sha256(data).hexdigest()!=sha:raise ValueError('native installation original SHA')
    if final.exists():return verify_owned_document(controller,root,sha)
    staging=root/('.document-'+sha+'.install-'+os.urandom(12).hex());staging.mkdir(mode=0o700)
    _create(staging/'document.json',json.loads(data))
    own=dict(version=OWNERSHIP_VERSION,name='document-'+sha+'.json',sha256=sha,stamp=stamp((staging/'document.json').lstat()))
    _create(staging/'ownership.ROOT-SIGNED.json',controller.signed(own))
    # Linux renameat2 NOREPLACE prevents adopting/replacing an existing path.
    libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None)
    if rename is None:raise OSError(errno.ENOSYS,'atomic paired ownership publication unavailable')
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(staging),-100,os.fsencode(final),1)!=0:
        number=ctypes.get_errno()
        if number!=errno.EEXIST:raise OSError(number,'native paired ownership publication')
        # A signed complete staging pair is preserved; the winning visible
        # bundle must independently pass all original ownership checks.
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return verify_owned_document(controller,root,sha)

def original_receipts(controller,epoch):
    from subnet.distributed_roles import authenticate
    root=directory(controller,epoch);authority=controller.authority.id
    envelopes={k:json.loads(_load(root/(v+'.ROOT-SIGNED.json')))for k,v in
               (('context','context'),('grades','grades'),('subset','subset'))}
    context=authenticate(envelopes['context'],authority);grades=authenticate(envelopes['grades'],authority)
    subset=authenticate(envelopes['subset'],authority)
    accepted,expected=bind_subset(envelopes['context'],grades,context['submissions'],authority)
    if expected!=subset:raise ValueError('native original subset receipt')
    return root,envelopes,context,grades,subset,accepted

def close_no_update(controller,error,manifest,status):
    """Return unchanged parent to the existing after/completion path, never train."""
    if not isinstance(error,NativeNoUpdate):raise error
    root,envelopes,context,grades,subset,accepted=original_receipts(controller,manifest['epoch'])
    if accepted or subset['disposition']!='no_update':raise ValueError('native no-update requires exact empty authenticated subset')
    if (Path(controller.state)/'roles'/(manifest['epoch']+'-train.json')).exists():
        raise ValueError('native no-update cannot supersede any issued training')
    original_manifest=context['original_signed_manifest']['payload']
    binding=manifest['trainer_state_binding'];pointer=status['trainer_state'];checkpoint=status['checkpoint']
    if (original_manifest['checkpoint']['id']!=checkpoint['id'] or
        original_manifest['trainer_state_binding']!=binding or
        binding['parent']!=pointer or binding['global_step_before']!=pointer['optimizer_steps'] or
        pointer['inference_checkpoint']!=checkpoint['id'] or status.get('persistent_state_committed')is not True):
        raise ValueError('native no-update exact approved parent and optimizer pointer')
    latest=Path(controller.state)/'latest-trainer-state.json';before=_load(latest)
    if json.loads(before)!=pointer:raise ValueError('native no-update latest optimizer pointer mismatch')
    unknown=sum(row['status']=='excluded_indeterminate' for row in grades['rows'])
    excluded=sum(row['status']=='excluded_label_mismatch' for row in grades['rows'])
    payload=dict(version=NO_UPDATE_VERSION,epoch=manifest['epoch'],status='closed_native_indeterminate_no_update' if unknown else 'closed_native_label_exclusions_no_update',
                 reason='native_grading_indeterminate' if unknown else 'no_matching_native_pairs',
                 retry_policy='new_unopened_epoch_same_parent_only' if unknown else 'none_for_original_epoch',
                 indeterminate_pairs=unknown,excluded_pairs=excluded,context_sha256=digest(envelopes['context']),
                 grades_sha256=digest(envelopes['grades']),subset_sha256=digest(envelopes['subset']),
                 checkpoint=checkpoint['id'],optimizer_steps=pointer['optimizer_steps'],
                 optimizer_pointer_sha256=__import__('hashlib').sha256(before).hexdigest(),
                 training_steps_before=status['training_steps'],optimizer_updates=0,training_dispatched=False,
                 sampling_assurance='unaudited',claims_rewritten=False,cheating_penalties=False,payable=False)
    path=root/'no-update.ROOT-SIGNED.json'
    if path.exists():
        from subnet.distributed_roles import authenticate
        envelope=json.loads(_load(path))
        if authenticate(envelope,controller.authority.id)!=payload:raise ValueError('native immutable no-update collision')
    else:envelope=controller.signed(payload);_create(path,envelope)
    controller.bucket.json('public/'+manifest['epoch']+'/native-no-update.json',envelope)
    if _load(latest)!=before:raise ValueError('no-update modified optimizer pointer')
    metrics=dict(steps=0,checkpoint_path=status['checkpoint_path'],trainer_state=pointer,
                 native_no_update=dict(payload,no_update_receipt_sha256=digest(envelope)))
    return checkpoint,metrics

def completion_fields(controller,epoch):
    from subnet.distributed_roles import authenticate
    root=directory(controller,epoch);path=root/'no-update.ROOT-SIGNED.json'
    if not path.exists():return {}
    e=json.loads(_load(path));v=authenticate(e,controller.authority.id)
    return dict(native_no_update=v,no_update_receipt_sha256=digest(e),optimizer_updates=0,status=v['status'])

def _get(controller,key,size):
    if not 0<size<=16*1024**2:raise ValueError('native lifecycle bounded GET')
    if size<=2_000_000:data=controller.bucket.get_bounded(key,limit=size)
    else:
        response=controller.bucket.client.get_object(Bucket=controller.bucket.name,Key=key);body=response['Body']
        try:
            if response['ContentLength']!=size:raise ValueError('native receipt R2 metadata size')
            data=body.read(size+1)
        finally:body.close()
    if len(data)!=size:raise ValueError('native lifecycle full GET bytes')
    return data

def _immutable_archive(controller,key,data):
    try:previous=_get(controller,key,len(data))
    except KeyError:previous=None
    except Exception as error:
        from botocore.exceptions import ClientError
        if not isinstance(error,ClientError) or str(error.response.get('Error',{}).get('Code'))not in ('NoSuchKey','404','NotFound'):raise
        previous=None
    if previous is not None and previous!=data:raise ValueError('immutable native receipt R2 collision')
    if previous is None:controller.bucket.put(key,data)
    if _get(controller,key,len(data))!=data:raise ValueError('native receipt full R2 readback')
    return dict(key=key,sha256=__import__('hashlib').sha256(data).hexdigest(),size=len(data),full_get_verified=True)

def retire_completed(controller,epoch,*,max_documents=32):
    """Resumable bounded full-readback ACK, then guarded derived-copy unlink."""
    from subnet.distributed_roles import authenticate
    import fcntl
    if type(max_documents)is not int or not 1<=max_documents<=32:raise ValueError('bounded native retirement GET work')
    root,envelopes,context,grades,subset,_=original_receipts(controller,epoch)
    completion_path=Path(controller.state)/(epoch+'-signed-learner-completion.json')
    if not completion_path.exists():return {'status':'deferred_no_signed_completion','retired_bytes':0}
    completion_env=json.loads(_load(completion_path));completion=authenticate(completion_env,controller.authority.id)
    if completion['epoch']!=epoch or completion['input_assurance']!='unaudited':raise ValueError('native retirement completion binding')
    status_path=Path(controller.state)/'controller.json'
    if not status_path.exists():return {'status':'deferred_epoch_not_fully_closed','retired_bytes':0}
    state=json.loads(_load(status_path));active=state.get('active')
    if (type(completion.get('round'))is not int or type(state.get('round'))is not int or state['round']<=completion['round'] or
        (active is not None and active.get('epoch')==epoch)):
        return {'status':'deferred_epoch_not_fully_closed','retired_bytes':0}
    if subset['disposition']=='no_update':
        expected=completion_fields(controller,epoch)
        if any(completion.get(k)!=v for k,v in expected.items())or completion['checkpoint']!=completion['next_checkpoint']:
            raise ValueError('native no-update completion pointer preservation')
    # A signed immutable completion is admission; full R2 GET is required below.
    fd=os.open(root/'selector.lock',os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'deferred_active_selector_lease','retired_bytes':0}
        ackpath=root/'full-R2-ACK.ROOT-SIGNED.json';prefix='private/'+epoch+'/native-eligibility/'
        if not ackpath.exists():
            records=[];work=0
            for obj in context['submissions']:
                name='document-'+obj['sha256']+'.json';path=document_path(root,obj['sha256'])
                progress=root/(name+'.full-readback.ROOT-SIGNED.json')
                admission=authenticate(obj['learner_admission'],controller.authority.id)
                if (admission['epoch']!=epoch or admission['document_sha256']!=obj['sha256'] or admission['document_size']!=obj['size']):
                    raise ValueError('native original captured document binding')
                key='public/'+epoch+'/submissions/'+admission['miner_identity']+'/'+admission['commitment_sha256']+'/training/'+str(admission['slot'])+'.json'
                expected=dict(key=key,sha256=obj['sha256'],size=obj['size'],full_get_verified=True)
                if progress.exists():
                    if authenticate(json.loads(_load(progress)),controller.authority.id)!=expected:raise ValueError('native immutable document readback progress')
                else:
                    if work>=max_documents:return {'status':'full_readback_in_progress','retired_bytes':0}
                    data=_get(controller,key,obj['size']);work+=1
                    if __import__('hashlib').sha256(data).hexdigest()!=obj['sha256']:raise ValueError('native captured R2 document full SHA')
                    _create(progress,controller.signed(expected))
                records.append(expected)
            archives=[]
            for name,e in dict(envelopes,completion=completion_env).items():
                archives.append(_immutable_archive(controller,prefix+name+'.json',_canonical(e)))
            no_update=root/'no-update.ROOT-SIGNED.json'
            if no_update.exists():archives.append(_immutable_archive(controller,prefix+'no-update.json',_load(no_update)))
            ack=dict(version=ACK_VERSION,epoch=epoch,context_sha256=digest(envelopes['context']),grades_sha256=digest(envelopes['grades']),subset_sha256=digest(envelopes['subset']),completion_sha256=digest(completion_env),documents=records,archives=archives,full_get_verified=True,original_capture_preserved=True)
            ackenv=controller.signed(ack)
            _immutable_archive(controller,prefix+'ACK.json',_canonical(ackenv))
            _create(ackpath,ackenv)
        ackenv=json.loads(_load(ackpath));ack=authenticate(ackenv,controller.authority.id)
        if (ack['version']!=ACK_VERSION or ack['context_sha256']!=digest(envelopes['context'])or ack['completion_sha256']!=digest(completion_env) or
            ack['grades_sha256']!=digest(envelopes['grades'])or ack['subset_sha256']!=digest(envelopes['subset'])or ack['full_get_verified']is not True or len(ack['documents'])!=len(context['submissions'])):
            raise ValueError('native cached-copy retirement requires immutable full R2 ACK')
        expected_documents=[]
        for obj in context['submissions']:
            admission=authenticate(obj['learner_admission'],controller.authority.id)
            expected_documents.append(dict(key='public/'+epoch+'/submissions/'+admission['miner_identity']+'/'+admission['commitment_sha256']+'/training/'+str(admission['slot'])+'.json',sha256=obj['sha256'],size=obj['size'],full_get_verified=True))
        expected_archives=[]
        for name,e in dict(envelopes,completion=completion_env).items():
            data=_canonical(e);expected_archives.append(dict(key=prefix+name+'.json',sha256=__import__('hashlib').sha256(data).hexdigest(),size=len(data),full_get_verified=True))
        no_update=root/'no-update.ROOT-SIGNED.json'
        if no_update.exists():
            data=_load(no_update);expected_archives.append(dict(key=prefix+'no-update.json',sha256=__import__('hashlib').sha256(data).hexdigest(),size=len(data),full_get_verified=True))
        if ack['documents']!=expected_documents or ack['archives']!=expected_archives or ack.get('original_capture_preserved')is not True:
            raise ValueError('native full ACK exact original documents and receipt archives')
        retired=0;count=0
        for obj in context['submissions']:
            name='document-'+obj['sha256']+'.json';path=document_path(root,obj['sha256']);done=root/(name+'.retired.ROOT-SIGNED.json')
            if done.exists():
                v=authenticate(json.loads(_load(done)),controller.authority.id)
                own=authenticate(json.loads(_load(path.parent/'ownership.ROOT-SIGNED.json')),controller.authority.id)
                if v!=dict(ack_sha256=digest(ackenv),ownership_sha256=digest(own),name=name,stamp=own['stamp'],bytes=obj['size']) or path.exists():raise ValueError('native retirement replay mismatch')
                continue
            if not path.exists():
                intent=root/(name+'.retirement-intent.ROOT-SIGNED.json')
                if not intent.exists():raise ValueError('missing owned native copy without retirement receipt')
                pending=authenticate(json.loads(_load(intent)),controller.authority.id)
                own=authenticate(json.loads(_load(path.parent/'ownership.ROOT-SIGNED.json')),controller.authority.id)
                if pending!=dict(ack_sha256=digest(ackenv),ownership_sha256=digest(own),name=name,stamp=own['stamp'],bytes=obj['size']):
                    raise ValueError('native unlink intent restart binding')
                _create(done,controller.signed(pending));continue
            own=verify_owned_document(controller,root,obj['sha256'])
            if own['version']!=OWNERSHIP_VERSION or own['name']!=name or own['sha256']!=obj['sha256']or stamp(path.lstat())!=own['stamp']:
                raise ValueError('native copy ownership or inode drift')
            data=_load(path,2_000_000)
            if __import__('hashlib').sha256(data).hexdigest()!=obj['sha256'] or stamp(path.lstat())!=own['stamp']:
                raise ValueError('native owned copy full SHA/identity')
            # Receipt is written before unlink; pending intent allows SAME inode
            # recovery after a crash, without adopting a replacement file.
            intent=root/(name+'.retirement-intent.ROOT-SIGNED.json');value=dict(ack_sha256=digest(ackenv),ownership_sha256=digest(own),name=name,stamp=own['stamp'],bytes=obj['size'])
            _create(intent,controller.signed(value));path.unlink();directory_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory_fd)
            finally:os.close(directory_fd)
            _create(done,controller.signed(value));retired+=obj['size'];count+=1
        return dict(status='owned_documents_retired',retired_bytes=retired,retired_files=count,original_capture_preserved=True,ack_sha256=digest(ackenv))
    finally:os.close(fd)


def start_retirement_observer(controller):
    """One bounded daemon reconciles only owned signed-completed namespaces."""
    if getattr(controller,'_native_retirement_observer',None)is not None:raise ValueError('duplicate native retirement observer')
    root=Path(controller.state)/'native-outcome-eligibility'
    def observe():
        while True:
            try:
                if root.exists():
                    for path in sorted(root.iterdir()):
                        if not path.is_dir()or path.is_symlink():continue
                        if not (Path(controller.state)/(path.name+'-signed-learner-completion.json')).exists():continue
                        try:result=retire_completed(controller,path.name,max_documents=32)
                        except Exception as error:
                            result=dict(status='retirement_deferred',error_type=type(error).__name__,original_capture_preserved=True)
                        # Explicit error events; no fabricated ACK or partial deletion.
                        health=path/'retirement-health.json';temporary=path/'.retirement-health.tmp'
                        with temporary.open('w')as stream:
                            json.dump(dict(result,time=time.time()),stream);stream.flush();os.fsync(stream.fileno())
                        temporary.chmod(0o600);temporary.replace(health)
            except Exception:
                # A transient directory/read error must not permanently stop
                # automatic retirement; every per-epoch failure is recorded above.
                pass
            finally:time.sleep(30)
    thread=threading.Thread(target=observe,name='owned-native-document-retirement',daemon=True)
    controller._native_retirement_observer=thread;thread.start()
