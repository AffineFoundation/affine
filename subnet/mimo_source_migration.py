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
