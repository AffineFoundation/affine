"""Default-off CPU controller selection hook; remote trainer source unchanged.

Receipts bind the full original population, and only new job construction may
consume the derived subset. Original capture/selection/audit files stay intact.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from .native_training_outcome_filter import (AUTHORIZATION_VERSION,CONTEXT_VERSION,
    VERSION,digest,filter_eligibility_context)

SUBSET_VERSION='native-outcome-accepted-subset-v1'

class NativeNoUpdate(ValueError):
    """Authenticated zero eligible pairs. Never issue an empty optimizer job."""


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def _load(path,maximum=64*1024**2):
    path=Path(path);before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>maximum:
        raise ValueError('owned native eligibility journal file')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        opened=os.fstat(stream.fileno())
        if (opened.st_dev,opened.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('native journal identity')
        data=stream.read(maximum+1)
        after=os.fstat(stream.fileno())
    current=path.lstat()
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=
        (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns) or
        (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns) or len(data)>maximum):
        raise ValueError('native journal changed')
    return data


def _create(path,value):
    path=Path(path);data=_canonical(value)
    # Hard-link install is atomic and never overwrites a historical context.
    temporary=path.parent/('.'+path.name+'.'+os.urandom(12).hex())
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            if _load(path)!=data:raise ValueError('immutable native journal collision')
    finally:temporary.unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _inventory(submissions):
    return [dict(sha256=o['sha256'],size=o['size'],learner_admission_sha256=digest(o['learner_admission']))for o in submissions]


def bind_subset(context_envelope,grade_receipt,submissions,authority):
    """Validate every original document decision; never drop/forge membership."""
    from subnet.distributed_roles import authenticate
    context=authenticate(context_envelope,authority)
    if (grade_receipt['context_sha256']!=digest(context_envelope) or
        grade_receipt.get('sampling_assurance')!='unaudited' or
        grade_receipt.get('proof_verification_performed') is not False or
        grade_receipt.get('claims_rewritten') is not False or grade_receipt.get('cheating_penalties') is not False or
        _inventory(context['submissions'])!=_inventory(submissions)):
        raise ValueError('native subset original context/assurance')
    decisions=grade_receipt.get('document_decisions')
    if not isinstance(decisions,list) or len(decisions)!=len(submissions):raise ValueError('complete native document disposition')
    grades=grade_receipt.get('rows');statuses={r['pair_sha256']:r for r in grades}
    if len(statuses)!=len(grades):raise ValueError('unique native pair dispositions')
    accepted=[];used=set()
    for decision,obj in zip(decisions,submissions):
        if (decision.get('document_sha256')!=obj['sha256'] or
            decision.get('learner_admission_sha256')!=digest(obj['learner_admission']) or
            type(decision.get('accepted')) is not bool or not decision.get('pair_sha256')):
            raise ValueError('native original document decision binding')
        identifiers=decision['pair_sha256']
        if len(set(identifiers))!=len(identifiers) or any(i not in statuses or i in used for i in identifiers):
            raise ValueError('native complete unique pair/document binding')
        used.update(identifiers)
        matching=True
        for identity in identifiers:
            row=statuses[identity];grades=row['grades']
            if len(grades)!=2 or [g['claim']for g in grades]!=['positive','negative']:
                raise ValueError('native pair original class order')
            for grade in grades:
                score=grade['native_score']
                if score is not None and (type(score) is not int or score not in (0,1)):raise ValueError('native exact score')
                expected=None if score is None else (score==1)==(grade['claim']=='positive')
                if grade['label_matches'] is not expected:raise ValueError('native score/claim disposition')
            status=('excluded_indeterminate' if any(g['label_matches'] is None for g in grades)
                else 'accepted_native_labels' if all(g['label_matches']for g in grades) else 'excluded_label_mismatch')
            if row['status']!=status:raise ValueError('native derived status')
            matching=matching and status=='accepted_native_labels'
        if decision['accepted'] is not matching:raise ValueError('native document completeness verdict')
        if matching:accepted.append(obj)
    if used!=set(statuses):raise ValueError('unbound native pair disposition')
    receipt=dict(version=SUBSET_VERSION,context_sha256=digest(context_envelope),
                 grade_receipt_sha256=digest(grade_receipt),original_inventory_sha256=digest(_inventory(submissions)),
                 accepted_inventory_sha256=digest(_inventory(accepted)),accepted_submissions=accepted,
                 accepted_count=len(accepted),excluded_count=len(submissions)-len(accepted),
                 disposition='no_update' if not accepted else 'all_accepted' if len(accepted)==len(submissions) else 'subset',
                 sampling_assurance='unaudited',cheating_penalties=False,claims_rewritten=False)
    return accepted,receipt


class NativeEligibilitySelector:
    """Installed only by a separately approved/pinned CPU operator runner.

    No execution callback or miner-supplied source is accepted by the public
    constructor. Tests may patch the module's trusted grading boundary.
    """
    def __init__(self,controller,authorization_envelope,tokenizer_root,interpreter):
        from subnet.distributed_roles import authenticate
        self.controller=controller;self.authorization=authorization_envelope
        self.authority=controller.authority.id
        self.policy=authenticate(authorization_envelope,self.authority)
        if self.policy.get('version')!=AUTHORIZATION_VERSION:raise ValueError('native preselection opt-in')
        self.tokenizer_root=tokenizer_root;self.interpreter=interpreter

    def select(self,manifest,submissions):
        from subnet.distributed_roles import authenticate
        from subnet.committed_training_inputs import coverage_manifest
        from subnet.training_receipts import computation_binding
        epoch=manifest['epoch']
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{1,220}',epoch):raise ValueError('native eligibility epoch path')
        state=Path(self.controller.state);population=state/(epoch+'-learner-population.json');selection=state/(epoch+'-learner-training-selection.json')
        population_data=_load(population);selection_data=_load(selection)
        original=json.loads(population_data)
        if (original['version']!='committed-unaudited-training-v1' or
            computation_binding(original['manifest'])!=computation_binding(manifest) or
            original['manifest'].get('trainer_state_binding')!=manifest.get('trainer_state_binding') or
            _inventory(original['submissions'])!=_inventory(submissions)):
            raise ValueError('native selector original frozen population')
        root=state/'native-outcome-eligibility';root.mkdir(mode=0o700,exist_ok=True)
        directory=root/epoch;directory.mkdir(mode=0o700,exist_ok=True)
        for parent in (root,directory):
            if parent.is_symlink() or parent.stat().st_uid!=os.getuid():raise ValueError('owned native eligibility namespace')
        fd=os.open(directory/'selector.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            lock=os.fstat(fd)
            if not stat.S_ISREG(lock.st_mode) or lock.st_nlink!=1 or lock.st_uid!=os.getuid() or stat.S_IMODE(lock.st_mode)!=0o600:raise ValueError('native selector lock ownership')
            result_path=directory/'subset.ROOT-SIGNED.json'
            role_record=state/'roles'/(epoch+'-train.json')
            if role_record.exists() and not result_path.exists():
                raise ValueError('native eligibility cannot adopt or subset an issued training job')
            context=dict(version=CONTEXT_VERSION,original_signed_manifest=self.controller.signed(manifest),
                         submissions=submissions,source_files=self.policy['source_files'],authorization_sha256=digest(self.authorization),
                         original_population_file_sha256=hashlib.sha256(population_data).hexdigest(),
                         original_selection_file_sha256=hashlib.sha256(selection_data).hexdigest(),
                         parent_binding_sha256=digest(manifest['trainer_state_binding']))
            context_path=directory/'context.ROOT-SIGNED.json'
            if context_path.exists():
                envelope=json.loads(_load(context_path))
                if authenticate(envelope,self.authority)!=context:raise ValueError('native immutable context changed')
            else:
                envelope=self.controller.signed(context);_create(context_path,envelope)
            grade_path=directory/'grades.ROOT-SIGNED.json'
            if grade_path.exists():
                grades=authenticate(json.loads(_load(grade_path)),self.authority)
            else:
                def prepare_document(obj):
                    if type(obj.get('size')) is not int or not 0<obj['size']<=2_000_000:
                        raise ValueError('bounded original native document')
                    if type(obj.get('sha256')) is not str or not re.fullmatch('[0-9a-f]{64}',obj['sha256']):
                        raise ValueError('native original document digest path')
                    path=directory/('document-'+obj['sha256']+'.json')
                    if not path.exists():
                        try:
                            with urllib.request.urlopen(obj['url'],timeout=30) as response:data=response.read(obj['size']+1)
                        except Exception as error:
                            raise ConnectionError('native original document GET '+type(error).__name__)from None
                        if len(data)!=obj['size'] or hashlib.sha256(data).hexdigest()!=obj['sha256']:
                            raise ValueError('native original document GET SHA/size')
                        # Preserve exact canonical original bytes, not a new JSON encoding.
                        value=json.loads(data)
                        if _canonical(value)!=data:raise ValueError('native canonical original document')
                        _create(path,value)
                    data=_load(path,2_000_000)
                    if len(data)!=obj['size'] or hashlib.sha256(data).hexdigest()!=obj['sha256']:
                        raise ValueError('native retained original document identity')
                    return path
                with ThreadPoolExecutor(max_workers=4) as pool:
                    paths=list(pool.map(prepare_document,submissions))
                _,grades=filter_eligibility_context(paths,envelope,self.authorization,self.authority,
                               self.policy['source_root'],self.tokenizer_root,self.interpreter)
                _create(grade_path,self.controller.signed(grades))
            accepted,subset=bind_subset(envelope,grades,submissions,self.authority)
            if result_path.exists():
                old=authenticate(json.loads(_load(result_path)),self.authority)
                if old!=subset:raise ValueError('native immutable subset changed')
            else:_create(result_path,self.controller.signed(subset))
            # Verify original files remain byte-exact after the CPU selection.
            if _load(population)!=population_data or _load(selection)!=selection_data:
                raise ValueError('original learner capture changed during native grading')
            if not accepted:raise NativeNoUpdate('authenticated native eligibility no_update; no optimizer dispatch')
            coverage=manifest['training_coverage']
            derived=coverage_manifest(manifest,accepted,seed=coverage['seed'],captured_at=coverage['captured_at'])
            if computation_binding(derived)!=computation_binding(manifest):raise ValueError('native subset changed scientific computation')
            return derived,accepted
        finally:
            os.close(fd)


BOUNDARY_VERSION='future-native-eligibility-boundary-v1'


def validate_future_boundary(boundary):
    fields={'version','epoch_prefix','earliest_round','minimum_parent_step','contract_fields'}
    if (type(boundary) is not dict or set(boundary)!=fields or boundary['version']!=BOUNDARY_VERSION or
        type(boundary['earliest_round']) is not int or boundary['earliest_round']<0 or
        type(boundary['minimum_parent_step']) is not int or boundary['minimum_parent_step']<0 or
        not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{1,180}',boundary['epoch_prefix']) or
        type(boundary['contract_fields']) is not dict or
        not {'training_input_policy','K','L','training_policy'}<=set(boundary['contract_fields'])):
        raise ValueError('explicit future native eligibility boundary')
    return boundary

class FutureNativeEligibilitySelector(NativeEligibilitySelector):
    """Skip original issued jobs; apply only to explicitly approved future epochs."""
    def __init__(self,controller,authorization_envelope,tokenizer_root,interpreter,boundary):
        super().__init__(controller,authorization_envelope,tokenizer_root,interpreter)
        validate_future_boundary(boundary)
        self.boundary=boundary

    def applies_to(self,manifest):
        epoch=manifest.get('epoch','')
        match=re.fullmatch(re.escape(self.boundary['epoch_prefix'])+r'--([0-9]+)-([0-9]+)',epoch)
        if match is None:raise ValueError('native eligibility epoch contract prefix')
        if int(match[2])<self.boundary['earliest_round']:return False
        state=Path(self.controller.state)
        if (state/'roles'/(epoch+'-train.json')).exists() and not (state/'native-outcome-eligibility'/epoch/'subset.ROOT-SIGNED.json').exists():
            return False
        if (manifest.get('source_bundle',{}).get('sha256')!=self.policy['source_sha256'] or
            any(manifest.get(k)!=v for k,v in self.boundary['contract_fields'].items()) or
            type(manifest.get('trainer_state_binding',{}).get('global_step_before')) is not int or
            manifest['trainer_state_binding']['global_step_before']<self.boundary['minimum_parent_step']):
            raise ValueError('future native eligibility source/contract/optimizer parent')
        return True

    def select(self,manifest,submissions):
        if not self.applies_to(manifest):return manifest,submissions
        return super().select(manifest,submissions)
