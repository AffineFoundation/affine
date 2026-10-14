"""Explicit, signed admission for a prepared MiMo release.

This module has no network or chain side effects. Public execution requires both
an operator-signed qualified policy and an explicit local opt-in. A prepared
release, a private diagnostic, or a miner-supplied flag cannot enable it.
"""
from __future__ import annotations

import base64
import copy
import re

from .backend_jobs import signed
from .storage import canonical, sha

VERSION = 'mimo-qualified-release-policy-v1'
QUALIFICATION_VERSION = 'mimo-release-qualification-v1'
CHECKPOINT_VERSION = 'mimo-checkpoint-authorization-v1'
POLICY_KEY = 'mimo_release_policy'
CHECKPOINT_KEY = 'mimo_checkpoint_authorization'
QUALIFICATIONS = frozenset({'runtime', 'cross-device-sampling', 'mutation-controls',
    'native-grading', 'trainer-update', 'model-publication', 'heldout-evaluation'})
_DYNAMIC = frozenset({'epoch', 'start', 'deadline', 'checkpoint', 'capabilities',
    POLICY_KEY, CHECKPOINT_KEY})


def digest(value):
    return sha(canonical(value))


def need(value, reason):
    if not value:
        raise ValueError(reason)


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def sign(identity, payload):
    return dict(payload=payload, signer=identity.id,
        signature=base64.b64encode(identity.key.sign(canonical(payload)).signature).decode())


def immutable_manifest(manifest):
    """Everything except epoch handover and explicitly per-epoch randomness."""
    value = {k: copy.deepcopy(v) for k, v in manifest.items() if k not in _DYNAMIC}
    # Read capabilities expire; archive content and its complete inventory do
    # not. Renewing this signed GET alias must not change qualified code/draws.
    if isinstance(value.get('source_bundle'), dict):
        value['source_bundle'].pop('url', None)
        value['source_bundle'].pop('read_url', None)
    sampling = value.get('sampling_contract')
    need(type(sampling) is dict, 'signed sampling contract required')
    sampling.pop('randomness', None)
    # A calibration is pinned in full except for the checkpoint it is applied to.
    # New measurements/tolerances still require a new signed release policy.
    if isinstance(sampling.get('calibration'), dict):
        sampling['calibration'].pop('checkpoint', None)
    return value


def descriptor(checkpoint):
    from .mimo_runtime import validate_descriptor
    return validate_descriptor({k: checkpoint[k] for k in ('id', 'files')})


def policy_scope(manifest, task_assets):
    from .mimo_release import task_assets as validate_assets
    assets = validate_assets(dict(task_assets=task_assets))
    need(bool(assets), 'explicit pinned task assets required')
    return dict(run_id=manifest['run_id'], immutable_manifest=immutable_manifest(manifest),
        initial_checkpoint=descriptor(manifest['checkpoint']), task_assets=assets)


def build_policy(identity, manifest, *, task_assets, mode='private', enabled=False,
                 qualifications=()):
    """Prepare a signed policy. This does not create qualifications or activate it."""
    scope = policy_scope(manifest, task_assets)
    payload = dict(version=VERSION, mode=mode, enabled=enabled, scope=scope,
        scope_sha256=digest(scope), qualifications=list(qualifications))
    envelope = sign(identity, payload)
    # Disabled policies are useful review artifacts, but cannot authorize work.
    ReleasePolicy(envelope, identity.id, allow_public=(mode == 'public'), require_enabled=False)
    return envelope


class ReleasePolicy:
    def __init__(self, envelope, authority, *, allow_public=False, require_enabled=True):
        need(type(allow_public) is bool and type(require_enabled) is bool,
             'explicit local release execution booleans')
        payload = copy.deepcopy(signed(envelope, authority))
        need(type(payload) is dict and set(payload) ==
             {'version','mode','enabled','scope','scope_sha256','qualifications'}, 'exact release policy fields')
        need(payload['version'] == VERSION and payload['mode'] in ('private', 'public')
             and type(payload['enabled']) is bool, 'explicit release execution mode')
        need(not require_enabled or payload['enabled'], 'release execution is disabled')
        need(payload['mode'] != 'public' or allow_public, 'public MiMo execution is locally disabled')
        scope = payload['scope']
        need(type(scope) is dict and set(scope) ==
             {'run_id','immutable_manifest','initial_checkpoint','task_assets'}
             and payload['scope_sha256'] == digest(scope), 'release scope binding')
        need(type(scope['run_id']) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', scope['run_id']),
             'safe release run namespace')
        need(scope['immutable_manifest'].get('run_id') == scope['run_id'], 'release run binding')
        descriptor(scope['initial_checkpoint'])
        from .mimo_release import task_assets as validate_assets
        need(bool(validate_assets(dict(task_assets=scope['task_assets']))), 'pinned release task assets')
        qualifications = payload['qualifications']
        need(type(qualifications) is list and len(qualifications) <= 64, 'bounded qualification evidence')
        covered = set()
        for envelope_row in qualifications:
            row = signed(envelope_row, authority)
            need(type(row) is dict and set(row) ==
                 {'version','scope_sha256','kind','passed','evidence_sha256'}, 'exact qualification receipt')
            need(row['version'] == QUALIFICATION_VERSION and row['scope_sha256'] == payload['scope_sha256']
                 and row['kind'] in QUALIFICATIONS and row['passed'] is True and _hash(row['evidence_sha256']),
                 'successful exact-release qualification required')
            need(row['kind'] not in covered, 'duplicate qualification category')
            covered.add(row['kind'])
        if payload['mode'] == 'public':
            need(covered == QUALIFICATIONS, 'public release lacks required qualification evidence')
            self._validate_public_protocol(scope)
        else:
            need(scope['immutable_manifest'].get('payable') is False,
                 'private release cannot authorize payable epochs')
        self.envelope, self.authority = copy.deepcopy(envelope), authority
        self.payload, self.scope = payload, scope
        self.policy_sha256 = digest(envelope)
        self.mode, self.run_id = payload['mode'], scope['run_id']

    @staticmethod
    def _validate_public_protocol(scope):
        from .complete_group_distribution import MIMO_VERSION
        from .complete_group_contract import GROUP_VERSION, OUTCOME_POLICIES
        from .mimo_runtime import validate_runtime_spec
        from .mimo_answer_projection import POLICY, POLICY_KEY as ANSWER_KEY
        m = scope['immutable_manifest']
        spec = validate_runtime_spec(m['mimo_runtime'])
        need(m.get('model_runtime_revision') == spec.version and m.get('model_id') == spec.model_id,
             'qualified MiMo model/runtime identity')
        fingerprint = spec.runtime_fingerprint
        need(all(isinstance(fingerprint['versions'].get(k), str) and fingerprint['versions'][k]
                 for k in ('torch','transformers','toploc')) and bool(fingerprint['sources'])
             and all(_hash(v) for v in fingerprint['sources'].values()),
             'concrete dependency versions and native/source fingerprints required')
        need(type(m.get('group_contract')) is dict and m['group_contract'].get('outcome_policy') in OUTCOME_POLICIES
             and m['group_contract'] == dict(version=GROUP_VERSION, group_size=8,
             min_success=1, min_completed_failure=1, outcome_policy=m['group_contract']['outcome_policy'])
             and m.get('max_batches') == 9, 'qualified N8 mixed complete groups and nine-group cap')
        need(m['sampling_contract'].get('version') == MIMO_VERSION and m.get(ANSWER_KEY) == POLICY,
             'qualified sampler and final-channel grading projection')
        need(_hash(m.get('source_bundle', {}).get('sha256')), 'exact qualified source archive')
        need(m.get('payable') is True, 'public policy must explicitly preserve reward eligibility')
        # Assets must be the exact snapshots named by the native task definitions.
        known_hashes = {row['sha256'] for row in scope['task_assets'].values()}
        need(all(_hash(x) for x in known_hashes), 'exact native task asset hashes')

    def validate_manifest(self, manifest):
        need(manifest.get(POLICY_KEY) == self.envelope, 'manifest must carry exact signed release policy')
        need(immutable_manifest(manifest) == self.scope['immutable_manifest'],
             'manifest differs from qualified source/runtime/task/sampler contract')
        source=manifest.get('source_bundle',{})
        aliases=[source[k]for k in ('url','read_url')if k in source]
        need(not aliases or all(url==aliases[0]for url in aliases), 'conflicting source GET aliases')
        need(type(manifest.get('epoch')) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', manifest['epoch']),
             'safe epoch identifier')
        need(self.mode != 'private' or manifest['epoch'].startswith(('private-','test-','nonpayable-')),
             'private release requires a nonpayable epoch namespace')
        need(type(manifest.get('start')) is int and type(manifest.get('deadline')) is int
             and 1 <= manifest['deadline'] - manifest['start'] <= 3600, 'bounded signed collection window')
        randomness = manifest['sampling_contract'].get('randomness')
        need(_hash(randomness), 'fresh explicit epoch randomness')
        cp = descriptor(manifest['checkpoint'])
        certificate = manifest.get(CHECKPOINT_KEY)
        if cp != self.scope['initial_checkpoint']:
            need(certificate is not None, 'successor checkpoint needs authenticated publication handover')
        if certificate is not None:
            value = signed(certificate, self.authority)
            need(type(value) is dict and set(value) == {'version','policy_sha256','run_id','checkpoint',
                'input_checkpoint','optimizer_step','publication_receipt_sha256'}, 'checkpoint authorization fields')
            need(value['version'] == CHECKPOINT_VERSION and value['policy_sha256'] == self.policy_sha256
                and value['run_id'] == self.run_id and value['checkpoint'] == cp
                and _hash(value['input_checkpoint']) and _hash(value['publication_receipt_sha256'])
                and type(value['optimizer_step']) is int and value['optimizer_step'] > 0,
                'successor checkpoint authorization binding')
        calibration = manifest['sampling_contract'].get('calibration', {})
        need(calibration.get('checkpoint') == cp['id'], 'calibration checkpoint must follow authenticated handover')
        return manifest

    def namespace(self, manifest=None):
        if manifest is not None:
            self.validate_manifest(manifest)
        return ('private/' if self.mode == 'private' else '') + 'mimo/' + self.run_id

    def authorize_checkpoint(self, checkpoint, *, input_checkpoint, optimizer_step,
                             publication_receipt_sha256, identity):
        """Called only after lifecycle authenticates a successful trainer receipt."""
        need(identity.id == self.authority, 'release authority required for checkpoint handover')
        need(_hash(input_checkpoint) and _hash(publication_receipt_sha256)
             and type(optimizer_step) is int and optimizer_step > 0, 'checkpoint publication evidence')
        return sign(identity, dict(version=CHECKPOINT_VERSION, policy_sha256=self.policy_sha256,
            run_id=self.run_id, checkpoint=descriptor(checkpoint), input_checkpoint=input_checkpoint,
            optimizer_step=optimizer_step, publication_receipt_sha256=publication_receipt_sha256))


def policy_for_manifest(manifest, authority, *, allow_public=False):
    """Old private diagnostics need no invented policy; public MiMo always does."""
    envelope = manifest.get(POLICY_KEY)
    if envelope is None:
        need(manifest.get('payable') is False and
             str(manifest.get('epoch', '')).startswith(('private-', 'test-', 'nonpayable-')),
             'public MiMo requires an enabled qualified release policy')
        return None
    policy = ReleasePolicy(envelope, authority, allow_public=allow_public)
    policy.validate_manifest(manifest)
    return policy
