"""Owned-copy retirement and truthful no-update for representative derivations."""
import hashlib
import json
import os
from pathlib import Path
import fcntl
from .native_training_eligibility import _load, _create, _canonical, NativeNoUpdate
from .native_training_lifecycle import (directory, document_path, stamp,
    verify_owned_document, _get)
from .native_task_representative_selection import read_documents
from subnet.training_receipts import authenticate, sha
from subnet.training_task_representatives import derivation_receipt


def evidence(controller,epoch):
    root=directory(controller,epoch);documents=read_documents(root)
    receipt=derivation_receipt(documents,controller.authority.id)
    return root,documents,receipt,authenticate(documents['result'],controller.authority.id)


def close_no_update(controller,error,manifest,status):
    if not isinstance(error,NativeNoUpdate):raise error
    epoch=manifest['epoch'];root,documents,receipt,result=evidence(controller,epoch)
    if result['accepted_submissions'] or result['disposition']!='no_update':raise ValueError('empty authenticated representative result')
    if (Path(controller.state)/'roles'/(epoch+'-train.json')).exists():raise ValueError('representative no-update cannot supersede issued job')
    original=authenticate(documents['pool']['payload']['original_signed_manifest'],controller.authority.id)
    pointer=status['trainer_state'];checkpoint=status['checkpoint'];binding=manifest['trainer_state_binding']
    latest=Path(controller.state)/'latest-trainer-state.json';before=_load(latest)
    if (original['checkpoint']['id']!=checkpoint['id'] or original['trainer_state_binding']!=binding
            or binding['parent']!=pointer or binding['global_step_before']!=pointer['optimizer_steps']
            or pointer['inference_checkpoint']!=checkpoint['id'] or status.get('persistent_state_committed') is not True
            or json.loads(before)!=pointer):raise ValueError('representative no-update preserves acknowledged parent')
    payload=dict(version='native-task-representative-no-update-v1',epoch=epoch,
        status='closed_native_representatives_no_update',reason=result['completion_reason'],
        representative_receipt=receipt,checkpoint=checkpoint['id'],optimizer_steps=pointer['optimizer_steps'],
        optimizer_pointer_sha256=hashlib.sha256(before).hexdigest(),training_steps_before=status['training_steps'],
        optimizer_updates=0,training_dispatched=False,sampling_assurance='unaudited',
        claims_rewritten=False,cheating_penalties=False,payable=False)
    envelope=controller.signed(payload);_create(root/'no-update.ROOT-SIGNED.json',envelope)
    controller.bucket.json('public/'+epoch+'/native-no-update.json',envelope)
    if _load(latest)!=before:raise ValueError('representative no-update pointer changed')
    return checkpoint,dict(steps=0,checkpoint_path=status['checkpoint_path'],trainer_state=pointer,
        native_no_update=dict(payload,no_update_receipt_sha256=sha(envelope)))


def _read_metadata(bucket,key,size):
    """Read exact bounded metadata without widening the small-token GET API."""
    if type(size) is not int or not 0<size<=64*1024**2:
        raise ValueError('representative durable metadata bound')
    response=bucket.client.get_object(Bucket=bucket.name,Key=key)
    body=response['Body']
    try:
        if type(response.get('ContentLength')) is not int or response['ContentLength']!=size:
            raise ValueError('representative metadata ContentLength')
        data=body.read(size+1)
        if len(data)!=size:raise ValueError('representative metadata complete byte count')
        return data
    finally:body.close()


def _archive(controller,key,data):
    """Bounded metadata PUT with full byte readback, never an optimizer export."""
    if type(data) is not bytes or not 0<len(data)<=64*1024**2:raise ValueError('representative durable metadata bound')
    try:previous=_read_metadata(controller.bucket,key,len(data))
    except Exception as error:
        from botocore.exceptions import ClientError
        if not isinstance(error,ClientError) or str(error.response.get('Error',{}).get('Code')) not in ('NoSuchKey','404','NotFound'):raise
        previous=None
    if previous is not None and previous!=data:raise ValueError('immutable representative durable metadata collision')
    if previous is None:controller.bucket.put(key,data)
    actual=_read_metadata(controller.bucket,key,len(data))
    if actual!=data:raise ValueError('representative full metadata readback')
    return dict(key=key,size=len(data),sha256=hashlib.sha256(data).hexdigest(),full_get_verified=True)


def retire_completed(controller,epoch,*,max_documents=32):
    if type(max_documents) is not int or not 1<=max_documents<=32:raise ValueError('bounded representative retirement')
    root,documents,receipt,result=evidence(controller,epoch);authority=controller.authority.id
    state=Path(controller.state);completion_path=state/(epoch+'-signed-learner-completion.json')
    if not completion_path.exists():return dict(status='deferred_no_signed_completion',retired_bytes=0)
    completion_env=json.loads(_load(completion_path));completion=authenticate(completion_env,authority)
    status=json.loads(_load(state/'controller.json'));active=status.get('active')
    if completion['epoch']!=epoch or completion.get('input_assurance')!='unaudited':raise ValueError('representative completion binding')
    if (type(completion.get('round')) is not int or status['round']<=completion['round']
            or active is not None and active['epoch']==epoch):return dict(status='deferred_epoch_not_closed',retired_bytes=0)
    if result['disposition']=='no_update':
        from .native_training_lifecycle import completion_fields
        if any(completion.get(k)!=v for k,v in completion_fields(controller,epoch).items()) or completion['checkpoint']!=completion['next_checkpoint']:
            raise ValueError('representative no-update completion original parent')
    fd=os.open(root/'selector.lock',os.O_RDWR|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='deferred_active_native_worker',retired_bytes=0)
        prefix='private/'+epoch+'/native-task-representatives/'
        # Completed and interrupted-wave contexts are retained in original form.
        records=[];owned=[];work=0
        pool={o['sha256']:o for o in documents['pool']['payload']['submissions']}
        for wave in sorted((root/'waves').iterdir()):
            if not (wave/'context.ROOT-SIGNED.json').exists():
                # A stopped parent may have created only the next empty wave
                # directory before the fixed deadline. No owned inputs exist.
                if any(wave.iterdir()):raise ValueError('unpublished native wave retains unknown files')
                continue
            context_env=json.loads(_load(wave/'context.ROOT-SIGNED.json'));context=authenticate(context_env,authority)
            for obj in context['submissions']:
                if pool.get(obj['sha256'])!=obj:raise ValueError('retirement original structural membership')
                path=document_path(wave,obj['sha256'])
                if not path.parent.exists():continue # Wave expired before this download.
                own_env=json.loads(_load(path.parent/'ownership.ROOT-SIGNED.json'));own=authenticate(own_env,authority)
                admission=authenticate(obj['learner_admission'],authority)
                key='public/'+epoch+'/submissions/'+admission['miner_identity']+'/'+admission['commitment_sha256']+'/training/'+str(admission['slot'])+'.json'
                expected=dict(key=key,sha256=obj['sha256'],size=obj['size'],full_get_verified=True)
                progress=wave/(obj['sha256']+'.full-readback.ROOT-SIGNED.json')
                if progress.exists():
                    if authenticate(json.loads(_load(progress)),authority)!=expected:raise ValueError('representative original full readback changed')
                else:
                    if work>=max_documents:return dict(status='full_readback_in_progress',retired_bytes=0)
                    raw=_get(controller,key,obj['size']);work+=1
                    if hashlib.sha256(raw).hexdigest()!=obj['sha256']:raise ValueError('representative original document R2 hash')
                    _create(progress,controller.signed(expected))
                records.append(expected);owned.append((wave,obj,path,own))
        archives=[]
        for name in ('pool','draw','authorization','result'):
            archives.append(_archive(controller,prefix+name+'.json',_canonical(documents[name])))
        archives.append(_archive(controller,prefix+'completion.json',_canonical(completion_env)))
        for wave in sorted((root/'waves').iterdir()):
            for name in ('context.ROOT-SIGNED.json','grades.ROOT-SIGNED.json','subset.ROOT-SIGNED.json'):
                path=wave/name
                if path.exists():archives.append(_archive(controller,prefix+'waves/'+wave.name+'/'+name,_load(path)))
        ack=controller.signed(dict(version='representative-owned-input-full-R2-ACK-v1',epoch=epoch,
            receipt=receipt,completion_sha256=sha(completion_env),documents=records,archives=archives,
            original_capture_preserved=True,full_get_verified=True))
        _archive(controller,prefix+'ACK.json',_canonical(ack));_create(root/'full-R2-ACK.ROOT-SIGNED.json',ack)
        retired=0
        for wave,obj,path,own in owned:
            intent=wave/(obj['sha256']+'.retirement-intent.ROOT-SIGNED.json');done=wave/(obj['sha256']+'.retired.ROOT-SIGNED.json')
            expected=dict(ack_sha256=sha(ack),ownership_sha256=sha(own),stamp=own['stamp'],bytes=obj['size'])
            if intent.exists():
                if authenticate(json.loads(_load(intent)),authority)!=expected:raise ValueError('representative retirement immutable intent')
            elif not path.exists():raise ValueError('missing representative owned input without intent')
            if path.exists():
                verify_owned_document(controller,wave,obj['sha256'])
                if stamp(path.lstat())!=own['stamp']:raise ValueError('representative copy replaced')
                _create(intent,controller.signed(expected));path.unlink()
                directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(directory_fd)
                finally:os.close(directory_fd)
                retired+=obj['size']
            _create(done,controller.signed(expected))
        return dict(status='owned_documents_retired',retired_bytes=retired,original_capture_preserved=True,ack_sha256=sha(ack))
    finally:os.close(fd)
