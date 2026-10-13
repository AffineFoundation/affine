"""Explicit source-only release admission and restartable boundary migration.

This module is prospective infrastructure, not an automatic deployment hook.
Original policies, jobs and evidence remain immutable. Admission does not claim
that changed Python code is scientifically equivalent: both policies still need
all normal release qualifications and the ROOT-signed changed-file allowlist.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

from .backend_jobs import signed
from .mimo_release_policy import (ReleasePolicy, POLICY_KEY, CHECKPOINT_KEY,
    descriptor, digest, need, sign, _hash)
from .storage import sha
from .mimo_grpo.transaction import _write

REGISTRY = 'mimo-source-release-registry-v1'
TRANSITION = 'mimo-source-boundary-transition-v1'
ACK = 'mimo-source-boundary-ack-v1'


def _source(policy):
    return policy.scope['immutable_manifest']['source_bundle']['sha256']


def _closure(registry, policy):
    return digest(registry.payload['execution'][_source(policy)]['source_files'])


def _stable_scope(policy):
    value = copy.deepcopy(policy.scope)
    source = value['immutable_manifest']['source_bundle']
    source.pop('sha256', None)
    source.pop('size', None)
    return value


class SourceReleaseRegistry:
    """Allowlisted policies and exact executable inventories, including history."""
    def __init__(self, envelope, authority, *, allow_public=False):
        value = copy.deepcopy(signed(envelope, authority))
        need(set(value) == {'version', 'policies', 'active_policy_sha256', 'execution',
             'source_changes', 'trainer_config', 'trainer_identity'}, 'exact source registry fields')
        need(value['version'] == REGISTRY and _hash(value['trainer_identity']), 'source registry identity')
        need(type(value['policies']) is list and 1 <= len(value['policies']) <= 32,
             'bounded explicit policy allowlist')
        policies = [ReleasePolicy(row, authority, allow_public=allow_public) for row in value['policies']]
        self.policies = {p.policy_sha256:p for p in policies}
        need(len(self.policies) == len(policies) and value['active_policy_sha256'] in self.policies,
             'unique allowlisted active policy')
        reference = policies[0]
        for policy in policies:
            need(policy.mode == reference.mode and _stable_scope(policy) == _stable_scope(reference),
                 'source-only migration cannot change model runtime grader sampler tasks or initial model')
        execution = value['execution']
        need(type(execution) is dict and set(execution) == {_source(p) for p in policies},
             'exact approved executable source profiles')
        need(type(value['trainer_config']) is dict and bool(value['trainer_config']),
             'explicit unchanged optimizer objective configuration')
        for source, profile in execution.items():
            need(_hash(source) and type(profile) is dict and set(profile) == {'source_files','runtime_versions'},
                 'exact execution profile fields')
            need(type(profile['source_files']) is dict and bool(profile['source_files'])
                 and all(type(k) is str and _hash(v) for k,v in profile['source_files'].items()),
                 'complete executable source inventory')
            need(type(profile['runtime_versions']) is dict and bool(profile['runtime_versions']),
                 'concrete execution dependency versions')
        first = execution[_source(reference)]
        changes = value['source_changes']
        need(type(changes) is list and len(set(changes)) == len(changes)
             and all(type(p) is str for p in changes), 'explicit unique changed-file allowlist')
        actual = set()
        for profile in execution.values():
            need(profile['runtime_versions'] == first['runtime_versions'], 'unchanged execution runtime required')
            actual.update(k for k in set(first['source_files']) | set(profile['source_files'])
                          if first['source_files'].get(k) != profile['source_files'].get(k))
        need(actual == set(changes), 'exact changed executable files must be admitted')
        self.envelope, self.authority, self.payload = copy.deepcopy(envelope), authority, value
        self.active = self.policies[value['active_policy_sha256']]

    def resolve(self, manifest):
        checksum = digest(manifest.get(POLICY_KEY))
        need(checksum in self.policies, 'unlisted release policy')
        policy = self.policies[checksum]
        policy.validate_manifest(manifest)
        return policy

    def execution_for(self, manifest):
        policy = self.resolve(manifest)
        return copy.deepcopy(self.payload['execution'][_source(policy)])

    def validate_job_execution(self, job):
        manifest = signed(job['manifest'], self.authority)
        expected = self.execution_for(manifest)
        need(all(job.get(k) == v for k,v in expected.items()), 'job must retain original executable source profile')
        return self.resolve(manifest)


def _decode(raw):
    need(type(raw) is str and len(raw) < 4*1024*1024, 'bounded original CURRENT bytes')
    result = base64.b64decode(raw, validate=True)
    need(type(json.loads(result)) is dict, 'original CURRENT object')
    return result


def _transition_payload(registry, old_opening, finalized, *, host_bytes, trainer_bytes,
                        original_transaction):
    """Build a reviewable ROOT certificate; never mutate state or train.

    The operator must supply original records after authenticating their full
    publication/readback. This binds their exact bytes and the published local
    optimizer generation; qualified policy admission remains separate.
    """
    old_manifest = signed(old_opening, registry.authority)
    old = registry.resolve(old_manifest)
    new = registry.active
    need(old.policy_sha256 != new.policy_sha256 and _source(old) != _source(new), 'distinct source-only transition')
    final = signed(finalized, registry.authority)
    host, trainer = json.loads(host_bytes), json.loads(trainer_bytes)
    need(final['policy_sha256'] == old.policy_sha256 and final['epoch'] == old_manifest['epoch']
         and final['input_checkpoint'] == old_manifest['checkpoint']['id']
         and final['opening_sha256'] == digest(old_opening), 'original finalized boundary')
    result = final['result']; cp = descriptor(result['checkpoint'])
    need(result['status'] == 'updated' and type(result['optimizer_step']) is int
         and result['optimizer_step'] > 0, 'published nonzero optimizer boundary')
    authorization = signed(result['checkpoint_authorization'], registry.authority)
    need(authorization == dict(version='mimo-checkpoint-authorization-v1', policy_sha256=old.policy_sha256,
         run_id=old.run_id, checkpoint=cp, input_checkpoint=final['input_checkpoint'],
         optimizer_step=result['optimizer_step'], publication_receipt_sha256=result['publication_receipt_sha256']),
         'original authenticated checkpoint authorization')
    need(host['policy_sha256'] == old.policy_sha256 and host['checkpoint'] == cp
         and host['checkpoint_authorization'] == result['checkpoint_authorization']
         and host['optimizer_step'] == result['optimizer_step'] and host['round'] == final['round']+1,
         'host CURRENT must be exact finalized successor')
    need(trainer['release_policy_sha256'] == old.policy_sha256 and trainer['run'] == old.run_id
         and trainer['source_closure_sha256'] == _closure(registry, old) and trainer['checkpoint'] == cp['id']
         and trainer['descriptor'] == cp and trainer['optimizer_step'] == result['optimizer_step']
         and trainer['epoch'] == old_manifest['epoch'] and trainer['input_checkpoint'] == final['input_checkpoint']
         and trainer['publication_receipt_sha256'] == result['publication_receipt_sha256'],
         'trainer CURRENT must retain same original published optimizer generation')
    transaction = original_transaction
    need(transaction['state'] == 'published' and transaction['job']['job_id'] == trainer['job_id']
         and transaction['job']['source_closure_sha256'] == _closure(registry, old)
         and transaction['job']['epoch'] == old_manifest['epoch']
         and transaction['job']['optimizer_step_before'] + 1 == trainer['optimizer_step']
         and transaction['job']['input_checkpoint'] == trainer['input_checkpoint']
         and transaction['output_checkpoint'] == cp['id'] and digest(transaction['report']) == trainer['report_sha256']
         and transaction['report_sha256'] == trainer['report_sha256']
         and transaction['report']['config'] == registry.payload['trainer_config'],
         'original published transaction and unchanged optimizer objective')
    payload = dict(version=TRANSITION, registry_sha256=digest(registry.envelope),
        from_policy_sha256=old.policy_sha256, to_policy_sha256=new.policy_sha256,
        old_opening=old_opening, finalized=finalized,
        original_host_bytes=base64.b64encode(host_bytes).decode(),
        original_trainer_bytes=base64.b64encode(trainer_bytes).decode(),
        original_transaction=transaction)
    return payload


def prepare_transition(registry, old_opening, finalized, *, host_bytes, trainer_bytes,
                       original_transaction, identity):
    need(identity.id == registry.authority, 'ROOT transition authority')
    return sign(identity, _transition_payload(registry, old_opening, finalized,
        host_bytes=host_bytes, trainer_bytes=trainer_bytes, original_transaction=original_transaction))


def validate_transition(registry, envelope):
    value = signed(envelope, registry.authority)
    need(type(value) is dict and set(value) == {'version','registry_sha256','from_policy_sha256',
         'to_policy_sha256','old_opening','finalized','original_host_bytes','original_trainer_bytes',
         'original_transaction'} and value['version'] == TRANSITION,
         'exact source transition fields')
    need(value['registry_sha256'] == digest(registry.envelope)
         and value['to_policy_sha256'] == registry.active.policy_sha256,
         'transition must select this admitted active release')
    expected = _transition_payload(registry, value['old_opening'], value['finalized'],
        host_bytes=_decode(value['original_host_bytes']), trainer_bytes=_decode(value['original_trainer_bytes']),
        original_transaction=value['original_transaction'])
    need(value == expected, 'exact original transition bindings')
    return value


def planned_current(registry, envelope, role, *, identity):
    value = validate_transition(registry, envelope)
    need(role in ('host','trainer') and identity.id == registry.authority, 'ROOT planned CURRENT authority')
    current = json.loads(_decode(value['original_'+role+'_bytes']))
    if role == 'host':
        current['policy_sha256'] = registry.active.policy_sha256
        old = signed(current['checkpoint_authorization'], registry.authority)
        current['checkpoint_authorization'] = registry.active.authorize_checkpoint(current['checkpoint'],
            input_checkpoint=old['input_checkpoint'], optimizer_step=old['optimizer_step'],
            publication_receipt_sha256=old['publication_receipt_sha256'], identity=identity)
    else:
        current['release_policy_sha256'] = registry.active.policy_sha256
        current['source_closure_sha256'] = _closure(registry, registry.active)
    return current


def validate_ack(registry, transition, envelope, role, planned):
    signer = registry.authority if role == 'host' else registry.payload['trainer_identity']
    value = signed(envelope, signer)
    need(value == dict(version=ACK, transition_sha256=digest(transition), role=role,
                      current_sha256=digest(planned)), 'exact role migration acknowledgement')
    return value


def apply_role_current(registry, transition, role, planned, *, path, journal_directory,
                       identity, assert_quiescent, locked, trainer_ack=None):
    """CAS CURRENT using an immutable signed journal and deterministic role ACK.

    Caller must hold the existing lifecycle/trainer lock via ``locked`` and
    prove no issued successor or active trainer update via ``assert_quiescent``.
    A host transition needs the original trainer ACK first. The ROOT-signed
    planned record prevents a remote trainer from choosing its own new state.
    Crash before journal/CURRENT/ACK resumes without a second optimizer update.
    This routine never modifies model, optimizer, transaction or report files.
    """
    value = validate_transition(registry, transition)
    expected_signer = registry.authority if role == 'host' else registry.payload['trainer_identity']
    need(role in ('host','trainer') and identity.id == expected_signer, 'authorized migration role')
    approved = signed(planned, registry.authority)
    need(set(approved) == {'transition_sha256','role','current'}
         and approved['transition_sha256'] == digest(transition) and approved['role'] == role,
         'ROOT admitted exact target CURRENT')
    target = approved['current']; original = _decode(value['original_'+role+'_bytes'])
    # Independent restrictions even if operator signed an accidentally broad change.
    before = json.loads(original); after = copy.deepcopy(target)
    if role == 'trainer':
        need(after.pop('release_policy_sha256') == registry.active.policy_sha256
             and after.pop('source_closure_sha256') == _closure(registry, registry.active), 'new trainer source pins')
        before.pop('release_policy_sha256'); before.pop('source_closure_sha256')
    else:
        need(after.pop('policy_sha256') == registry.active.policy_sha256, 'new host policy pin')
        certificate = after.pop('checkpoint_authorization'); before.pop('policy_sha256'); before.pop('checkpoint_authorization')
        old_cert = signed(json.loads(original)['checkpoint_authorization'], registry.authority)
        need(signed(certificate, registry.authority) == dict(old_cert, policy_sha256=registry.active.policy_sha256),
             'source-only checkpoint reauthorization')
        need(trainer_ack is not None, 'trainer acknowledgement required before host opening')
        trainer_target = json.loads(_decode(value['original_trainer_bytes']))
        trainer_target.update(release_policy_sha256=registry.active.policy_sha256,
                              source_closure_sha256=_closure(registry, registry.active))
        validate_ack(registry, transition, trainer_ack, 'trainer', trainer_target)
    need(after == before, 'migration must preserve every model optimizer and cursor field')
    path, journal = Path(path).absolute(), Path(journal_directory).absolute()
    need(path.resolve() == path and journal.resolve() == journal, 'ordinary migration paths')
    need(callable(assert_quiescent) and callable(locked), 'exclusive quiescent migration callbacks')
    with locked():
        assert_quiescent()
        need(path.is_file() and not path.is_symlink(), 'original CURRENT file required')
        raw = path.read_bytes()
        journal.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = dict(transition=transition, planned=planned, original_sha256=sha(original))
        record_path = journal/'PREPARED.json'
        if record_path.exists():
            need(not record_path.is_symlink() and json.loads(record_path.read_bytes()) == record,
                 'same original prepared migration')
            need(raw == original or json.loads(raw) == target, 'CURRENT changed since migration preparation')
        else:
            need(raw == original, 'CURRENT bytes changed before migration')
            _write(record_path, record)
        if raw == original:
            assert_quiescent()
            _write(path, target)
        ack = sign(identity, dict(version=ACK, transition_sha256=digest(transition), role=role,
                                  current_sha256=digest(target)))
        ack_path = journal/'ACK.json'
        if ack_path.exists():
            need(not ack_path.is_symlink() and json.loads(ack_path.read_bytes()) == ack, 'same original migration acknowledgement')
        else:
            _write(ack_path, ack)
        return ack

HOLD = 'mimo-source-boundary-hold-v1'


def boundary_gate(root, policy, identity, *, now):
    """Called under lifecycle.lock before PLAN creation, including after restart.

    The signed request may be armed one epoch ahead. A timeout alerts the
    operator but NEVER releases a partially migrated optimizer automatically.
    Old executables must first receive this host-only hook before arming it.
    """
    root=Path(root);path=root/'SOURCE-MIGRATION-HOLD.json'
    if not path.exists():return None
    need(not path.is_symlink() and 0<path.stat().st_size<32*1024**2,'bounded ordinary hold request')
    envelope=json.loads(path.read_bytes());request=signed(envelope,identity.id)
    need(set(request)=={'version','registry','from_policy_sha256','epoch_index','created_at','deadline'}
         and request['version']==HOLD,'exact signed boundary hold')
    registry=SourceReleaseRegistry(request['registry'],identity.id,allow_public=policy.mode=='public')
    need(request['from_policy_sha256'] in registry.policies
         and request['from_policy_sha256']!=registry.active.policy_sha256
         and type(request['epoch_index'])is int and request['epoch_index']>0
         and type(request['created_at'])is int and type(request['deadline'])is int
         and 1<=request['deadline']-request['created_at']<=7200,'bounded source migration hold')
    current_path=root/'CURRENT.json'
    need(current_path.is_file() and not current_path.is_symlink(),'existing current for boundary hold')
    raw=current_path.read_bytes();current=json.loads(raw);target=request['epoch_index']
    need(policy.policy_sha256 in (request['from_policy_sha256'],registry.active.policy_sha256),
         'hold admits only exact before and after policy')
    if current['epoch_index']<target:
        need(current['epoch_index']==target-1 and current['policy_sha256']==request['from_policy_sha256'],
             'hold must target next exact boundary')
        return None
    journal=root/'source-migrations'/digest(envelope)
    complete=journal/'COMPLETED.json'
    if complete.exists():
        need(not complete.is_symlink(),'ordinary completed migration')
        completed=signed(json.loads(complete.read_bytes()),identity.id)
        need(set(completed)=={'hold_sha256','transition','trainer_ack','host_ack'}
             and completed['hold_sha256']==digest(envelope),'completed exact hold')
        transition=completed['transition'];t=validate_transition(registry,transition)
        need(json.loads(_decode(t['original_host_bytes']))['epoch_index']==target
             and t['from_policy_sha256']==request['from_policy_sha256'],'same held boundary transition')
        for role in ('trainer','host'):
            planned=planned_current(registry,transition,role,identity=identity)
            validate_ack(registry,transition,completed[role+'_ack'],role,planned)
        need(current['epoch_index']>=target and current['policy_sha256']==registry.active.policy_sha256,
             'completed migration current policy')
        if policy.policy_sha256!=registry.active.policy_sha256:
            return dict(status='source-migration-held',reason='new-executable-required',epoch_index=target)
        if current['epoch_index']==target:
            need(current==planned_current(registry,transition,'host',identity=identity),'exact migrated boundary current')
        return None
    need(current['epoch_index']==target,'cannot arm hold after boundary has passed')
    directory=root/'epochs'/str(target)
    need(not(directory/'PLAN.json').exists() and not(directory/'coordinator/OPENING.json').exists(),
         'source hold cannot interrupt an issued successor')
    journal.mkdir(parents=True,exist_ok=True,mode=0o700)
    captured=journal/'BOUNDARY.json'
    if not captured.exists():
        need(current['policy_sha256']==request['from_policy_sha256'],'original policy before boundary capture')
        prior=root/'epochs'/str(target-1)/'FINALIZED.json'
        need(prior.is_file() and not prior.is_symlink(),'original finalized predecessor required')
        final_envelope=json.loads(prior.read_bytes());final=signed(final_envelope,identity.id)
        need(final['policy_sha256']==request['from_policy_sha256'] and final['round']+1==current['round']
             and final['result']['checkpoint']==current['checkpoint']
             and final['result']['optimizer_step']==current['optimizer_step'],'finalized predecessor matches current')
        _write(captured,sign(identity,dict(hold_sha256=digest(envelope),original_current_bytes=base64.b64encode(raw).decode(),
                                         finalized=final_envelope,captured_at=now)))
    else:
        snapshot=signed(json.loads(captured.read_bytes()),identity.id)
        need(snapshot['hold_sha256']==digest(envelope),'original captured boundary')
        # CURRENT may already be migrated during a crash before COMPLETED; hold
        # remains in force and the operator resumes its existing signed journal.
        need(current['checkpoint']==json.loads(_decode(snapshot['original_current_bytes']))['checkpoint'],
             'no model update during source hold')
    return dict(status='source-migration-held',reason='operator-deadline-exceeded' if now>request['deadline'] else 'awaiting-role-acknowledgements',epoch_index=target)


def arm_boundary_hold(root, request, policy, identity, *, now):
    """Persist one ROOT-approved next-boundary hold under the real host lock."""
    import os,fcntl
    root=Path(root).absolute();need(root.resolve()==root,'ordinary lifecycle root')
    fd=os.open(root/'lifecycle.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=signed(request,identity.id);current=json.loads((root/'CURRENT.json').read_bytes())
        need(p['from_policy_sha256']==policy.policy_sha256==current['policy_sha256']
             and p['created_at']<=now<p['deadline'],'current qualified unexpired hold')
        target=p['epoch_index'];need(target in (current['epoch_index'],current['epoch_index']+1),'next exact epoch boundary')
        directory=root/'epochs'/str(target)
        need(not(directory/'PLAN.json').exists() and not(directory/'coordinator/OPENING.json').exists(),
             'cannot hold an issued epoch')
        # Validate the complete envelope before persisting, including registry.
        reg=SourceReleaseRegistry(p['registry'],identity.id,allow_public=policy.mode=='public')
        need(set(p)=={'version','registry','from_policy_sha256','epoch_index','created_at','deadline'}
             and p['version']==HOLD and type(target)is int and target>0
             and 1<=p['deadline']-p['created_at']<=7200
             and p['from_policy_sha256'] in reg.policies and reg.active.policy_sha256!=policy.policy_sha256,
             'explicit bounded source-only hold request')
        path=root/'SOURCE-MIGRATION-HOLD.json'
        if path.exists():need(not path.is_symlink() and json.loads(path.read_bytes())==request,'original hold cannot be replaced')
        else:_write(path,request)
        return boundary_gate(root,policy,identity,now=now)
    finally:os.close(fd)


def complete_boundary_hold(root, registry, transition, trainer_ack, host_ack, identity, *, now):
    """Release only after original captured state and both exact ACKs authenticate."""
    import os,fcntl
    root=Path(root).absolute();need(root.resolve()==root,'ordinary lifecycle root')
    fd=os.open(root/'lifecycle.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        request=json.loads((root/'SOURCE-MIGRATION-HOLD.json').read_bytes());p=signed(request,identity.id)
        need(p['registry']==registry.envelope,'original held registry')
        journal=root/'source-migrations'/digest(request)
        snapshot=signed(json.loads((journal/'BOUNDARY.json').read_bytes()),identity.id)
        t=validate_transition(registry,transition)
        need(snapshot['hold_sha256']==digest(request)
             and snapshot['original_current_bytes']==t['original_host_bytes']
             and snapshot['finalized']==t['finalized'],'transition uses captured original boundary bytes')
        for role,ack in (('trainer',trainer_ack),('host',host_ack)):
            target=planned_current(registry,transition,role,identity=identity)
            validate_ack(registry,transition,ack,role,target)
        record=sign(identity,dict(hold_sha256=digest(request),transition=transition,trainer_ack=trainer_ack,host_ack=host_ack))
        path=journal/'COMPLETED.json'
        if path.exists():need(not path.is_symlink() and json.loads(path.read_bytes())==record,'same original completed migration')
        else:
            need(json.loads((root/'CURRENT.json').read_bytes())==target,'original host acknowledgement applied')
            _write(path,record)
        need(boundary_gate(root,registry.active,identity,now=now)is None,'completed hold permits new release only')
        return record
    finally:os.close(fd)


def _role_lock(root, name):
    from contextlib import contextmanager
    import os,fcntl
    @contextmanager
    def locked():
        fd=os.open(Path(root)/name,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield
        finally:os.close(fd)
    return locked


def migrate_trainer_current(root, registry, transition, planned, identity):
    """Locked one-time cold inventory admission; no model loading or update."""
    from .mimo_grpo.transaction import _inventory
    root=Path(root).absolute();need(root.resolve()==root,'ordinary original trainer run root')
    value=validate_transition(registry,transition);original=value['original_transaction']
    checked=False
    def quiescent():
        nonlocal checked
        if checked:return
        matches=[]
        for path in (root/'history').glob('*/TRANSACTION.json'):
            need(not path.is_symlink(),'ordinary original transaction')
            row=json.loads(path.read_bytes())
            need(row['state']=='published','unfinished trainer transaction forbids source migration')
            if row['job']['job_id']==original['job']['job_id']:matches.append((path.parent,row))
        need(len(matches)==1 and matches[0][1]==original,'exact original published optimizer transaction')
        folder,row=matches[0]
        need(_inventory(folder,list(row['artifacts']))==row['artifacts'],
             'cold original model and optimizer inventory authentication')
        descriptor=json.loads((folder/'optimizer/descriptor.json').read_bytes())
        need(descriptor['optimizer_step']==row['job']['optimizer_step_before']+1,
             'original optimizer generation')
        checked=True
    return apply_role_current(registry,transition,'trainer',planned,path=root/'CURRENT.json',
        journal_directory=root/'source-migrations'/digest(transition),identity=identity,
        assert_quiescent=quiescent,locked=_role_lock(root,'.trainer.lock'))


def migrate_host_current(root, registry, transition, planned, trainer_ack, identity, *, now):
    """Require the persisted captured boundary under the real lifecycle lock."""
    root=Path(root).absolute();need(root.resolve()==root,'ordinary original lifecycle root')
    t=validate_transition(registry,transition)
    def quiescent():
        request=json.loads((root/'SOURCE-MIGRATION-HOLD.json').read_bytes())
        r=signed(request,identity.id)
        need(r['registry']==registry.envelope,'same held source registry')
        held=boundary_gate(root,registry.policies[t['from_policy_sha256']],identity,now=now)
        need(held is not None and held['status']=='source-migration-held','persistent host boundary required')
        snapshot=signed(json.loads((root/'source-migrations'/digest(request)/'BOUNDARY.json').read_bytes()),identity.id)
        need(snapshot['original_current_bytes']==t['original_host_bytes'] and snapshot['finalized']==t['finalized'],
             'exact held predecessor and original host bytes')
    return apply_role_current(registry,transition,'host',planned,path=root/'CURRENT.json',
        journal_directory=root/'source-migrations'/digest(transition),identity=identity,
        assert_quiescent=quiescent,locked=_role_lock(root,'lifecycle.lock'),trainer_ack=trainer_ack)
