"""Explicit prospective collector-pool -> native waves -> final derivation.

No reward/audit writes. Each native wave is an original bounded context; there
is no fabricated aggregate native grade/context. The final receipt is versioned.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import urllib.request

from .native_training_eligibility import _create, _load, NativeNoUpdate, bind_subset
from .native_training_outcome_filter import CONTEXT_VERSION
from .native_training_lifecycle import (document_path, install_document_bundle,
                                       verify_owned_document)
from subnet.training_receipts import authenticate, sha, computation_binding
from subnet import training_task_representatives as rep
from subnet.committed_training_inputs import coverage_manifest, receipt_inventory


def read_documents(root):
    root=Path(root)
    values={name:json.loads(_load(root/(name+'.ROOT-SIGNED.json')))
            for name in ('pool','draw','authorization','result')}
    waves=[]
    wave_root=root/'waves'
    for path in sorted(wave_root.iterdir()) if wave_root.exists() else []:
        if not path.name.isdigit() or path.is_symlink() or not path.is_dir():
            raise ValueError('exact immutable native wave namespace')
        if int(path.name)!=len(waves):raise ValueError('contiguous native wave journal')
        grades=path/'grades.ROOT-SIGNED.json'
        if not grades.exists():
            if path!=sorted(wave_root.iterdir())[-1]:raise ValueError('partial nonterminal native wave')
            break
        waves.append({name:json.loads(_load(path/(name+'.ROOT-SIGNED.json')))
                      for name in ('context','grades')})
    values['waves']=waves
    return values


def prepare_documents(selector, root, objects, expires):
    paths=[]
    # Serial bounded reads keep decoded input residency out of the parent.
    for obj in objects:
        if time.time()>=expires:raise TimeoutError('representative fixed native deadline')
        path=document_path(root,obj['sha256'])
        if not path.exists():
            with urllib.request.urlopen(obj['url'],timeout=max(.1,min(30,expires-time.time()))) as response:
                raw=response.read(obj['size']+1)
            if len(raw)!=obj['size'] or hashlib.sha256(raw).hexdigest()!=obj['sha256']:
                raise ValueError('original representative GET size/hash')
            install_document_bundle(selector.controller,root,obj['sha256'],raw)
        verify_owned_document(selector.controller,root,obj['sha256'])
        paths.append(path)
    return paths


def group_members(pgid):
    """Grader subprocesses inherit the one group created by grade_wave."""
    members=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z':members.append(int(directory.name))
        except (FileNotFoundError,ProcessLookupError):pass
    return members


def drain_group(child):
    """Confirm the whole CPU group is gone before the selector lease can close.

    The original grader uses subprocess.run without a new session; the wrapper
    exiting is not proof that those descendants finished. Never accept their
    late output or adopt a result until the complete group is terminal.
    """
    for sig,seconds in ((signal.SIGTERM,2),(signal.SIGKILL,5)):
        if not group_members(child.pid):break
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:pass
        until=time.monotonic()+seconds
        while group_members(child.pid) and time.monotonic()<until:time.sleep(.02)
    child.wait(timeout=1)
    remaining=group_members(child.pid)
    if remaining:raise RuntimeError('owned native CPU group still live: '+str(remaining))
    return dict(process_group=child.pid,living_members=[],confirmed_at=time.time())


def grade_wave(selector, root, paths, context, expires, lease_fd):
    """Fresh CPU process, deadline includes tokenizer/hash/decode and grading."""
    remaining=expires-time.time()
    if remaining<=0:raise TimeoutError('representative fixed native deadline')
    request=dict(context=context,authorization=selector.authorization,
        authority=selector.authority,source_root=selector.policy['source_root'],
        tokenizer_root=str(selector.tokenizer_root),interpreter=str(selector.interpreter),
        paths=[str(p) for p in paths])
    _create(root/'request.json',request)
    output=root/'unsigned-grades.json'
    if output.exists():
        grades=json.loads(_load(output));bind_subset(context,grades,context['payload']['submissions'],selector.authority)
        return grades
    child=selector.wave_child
    if (type(child) is not dict or set(child)!={'path','sha256','execution_root'}
            or child['execution_root']!=selector.policy.get('execution_root')
            or child['path']!=str(Path(child['execution_root'])/'ops/native_task_representative_wave.py')):
        raise ValueError('explicit authorized installed native child path')
    program=Path(child['path'])
    if (not program.is_absolute() or program.resolve()!=program or program.is_symlink()
            or hashlib.sha256(program.read_bytes()).hexdigest()!=child['sha256']):
        raise ValueError('installed native child source hash')
    command=['/usr/bin/timeout','--signal=TERM','--kill-after=2s',str(max(.1,remaining))+'s',
             sys.executable,'-I','-B',str(program),
             '--request',str(root/'request.json'),'--output',str(output)]
    with (root/'worker.stderr').open('ab') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                               stderr=log,start_new_session=True,pass_fds=(lease_fd,))
        try:
            status=child.wait(timeout=remaining)
            if status in (124,137):raise TimeoutError('representative fixed native deadline')
            if status:raise RuntimeError('native representative CPU wave failed: '+str(status))
        finally:
            terminal=drain_group(child)
            _create(root/'child-terminal.json',terminal)
    grades=json.loads(_load(output))
    bind_subset(context,grades,context['payload']['submissions'],selector.authority)
    return grades


def select(selector, manifest, submissions):
    policy=rep._policy(manifest)
    if policy is None:raise ValueError('explicit representative selector policy')
    controller=selector.controller;authority=selector.authority;state=Path(controller.state)
    epoch=manifest['epoch'];pool_path,_=rep.collection_paths(state,epoch)
    original_path=state/(epoch+'-learner-population.json')
    population_raw=_load(original_path);population=json.loads(population_raw)
    selection_path=state/(epoch+'-learner-training-selection.json');selection_raw=_load(selection_path)
    pool=json.loads(_load(pool_path));original,*_=rep.admit_pool(pool,authority)
    if (sha(pool['payload']['original_signed_manifest']['payload']) != sha(original)
            or computation_binding(original)!=computation_binding(manifest)
            or original['trainer_state_binding']!=manifest['trainer_state_binding']
            or pool['payload']['original_population_file_sha256']!=hashlib.sha256(population_raw).hexdigest()
            or population['submissions']!=submissions):
        raise ValueError('exact original representative collector population')
    parent=state/'native-outcome-eligibility';parent.mkdir(mode=0o700,exist_ok=True)
    root=parent/epoch;root.mkdir(mode=0o700,exist_ok=True)
    for p in (parent,root):
        if p.is_symlink() or p.stat().st_uid!=os.getuid():raise ValueError('owned representative namespace')
    fd=os.open(root/'selector.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=os.fstat(fd)
        if not stat.S_ISREG(lock.st_mode) or lock.st_nlink!=1 or lock.st_uid!=os.getuid():raise ValueError('owned representative lease')
        result_path=root/'result.ROOT-SIGNED.json'
        if (state/'roles'/(epoch+'-train.json')).exists() and not result_path.exists():
            raise ValueError('representatives cannot change any original issued job')
        _create(root/'pool.ROOT-SIGNED.json',pool)
        _create(root/'authorization.ROOT-SIGNED.json',selector.authorization)
        draw=rep.freeze_draw(pool,authority,root/'draw.json',now=time.time())
        _create(root/'draw.ROOT-SIGNED.json',controller.signed(draw))
        (root/'waves').mkdir(mode=0o700,exist_ok=True)
        waves=[]
        # Replay only signed complete waves; an interrupted wave retains its rank.
        for wave_dir in sorted((root/'waves').iterdir()):
            if wave_dir.is_symlink() or wave_dir.name!=str(len(waves)).zfill(4):raise ValueError('contiguous native representative waves')
            if not (wave_dir/'grades.ROOT-SIGNED.json').exists():break
            waves.append({name:json.loads(_load(wave_dir/(name+'.ROOT-SIGNED.json')))for name in ('context','grades')})
        expires=draw['captured_at']+policy['max_native_wall_seconds']
        while not result_path.exists():
            progress=rep.replay(pool,authority,draw,selector.authorization,waves)
            if progress['complete'] or time.time()>=expires:
                result=rep.finalize(progress,draw,policy,time.time())
                _create(result_path,controller.signed(result));break
            wave_root=root/'waves'/str(len(waves)).zfill(4);wave_root.mkdir(mode=0o700,exist_ok=True)
            if wave_root.is_symlink() or wave_root.resolve()!=wave_root:raise ValueError('owned exact wave directory')
            objects=progress['next_submissions']
            context=controller.signed(dict(version=CONTEXT_VERSION,
                original_signed_manifest=pool['payload']['original_signed_manifest'],submissions=objects,
                source_files=selector.policy['source_files'],authorization_sha256=sha(selector.authorization),
                original_population_file_sha256=sha(pool),original_selection_file_sha256=sha(draw),
                parent_binding_sha256=sha(original['trainer_state_binding'])))
            from subnet.storage import canonical
            if len(canonical(context))>32_000_000:raise ValueError('bounded original native wave envelope')
            _create(wave_root/'context.ROOT-SIGNED.json',context)
            try:
                paths=prepare_documents(selector,wave_root,objects,expires)
                grades=grade_wave(selector,wave_root,paths,context,expires,fd)
            except (TimeoutError,subprocess.TimeoutExpired):
                if time.time()<expires:raise # Transport timeout is never native invalidity.
                continue
            accepted,subset=bind_subset(context,grades,objects,authority)
            _create(wave_root/'grades.ROOT-SIGNED.json',controller.signed(grades))
            _create(wave_root/'subset.ROOT-SIGNED.json',controller.signed(subset))
            waves.append(dict(context=context,grades=controller.signed(grades)))
        documents=read_documents(root);receipt=rep.derivation_receipt(documents,authority)
        result=authenticate(documents['result'],authority)
        if _load(original_path)!=population_raw or _load(selection_path)!=selection_raw:
            raise ValueError('original reward/selection population changed during native waves')
        if not result['accepted_submissions']:
            raise NativeNoUpdate('authenticated representative no_update; no optimizer dispatch')
        derived=coverage_manifest(manifest,result['accepted_submissions'],seed=draw['seed'],captured_at=draw['captured_at'])
        if draw['blacklist_selection'] is not None:
            # The old singleton-selection snapshot remains in its original file;
            # this separately derived manifest records the actual full-pool cut.
            derived['learner_blacklist_selection_snapshot']=draw['blacklist_selection']
        derived['native_training_eligibility_receipt']=receipt
        if computation_binding(derived)!=computation_binding(manifest):raise ValueError('representatives changed original computation')
        return derived,result['accepted_submissions']
    finally:os.close(fd)
