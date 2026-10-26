"""Prospective task cooling around the exact admitted capture64 learner.

No trainer, verifier, reward writer or public miner implementation is replaced.
Unsigned policies can only construct and check the candidate without live I/O.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
VERSION = 'prospective-task-cooling-capture64-successor-v1'
C = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(), 'ordinary exact source file')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def load(reference, name):
    need(pin(reference['path']) == reference, 'pinned operator source ' + name)
    spec = importlib.util.spec_from_file_location(name, reference['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authenticated(document):
    """Authenticate before using any document-selected Python import path."""
    from nacl.signing import VerifyKey
    need(type(document) is dict and set(document) == {'payload', 'signer', 'signature'}
         and document['signer'] == AUTHORITY, 'known bootstrap authority envelope')
    VerifyKey(bytes.fromhex(AUTHORITY)).verify(C(document['payload']),
                                             base64.b64decode(document['signature'], validate=True))
    return document['payload']


def validate(value, guards):
    fields = {'version', 'authority', 'execute_allowed', 'kind', 'config', 'baseline_runtime',
              'source_sha256', 'runner_sha256', 'runner', 'parent_policy', 'parent_entry',
              'selector', 'provider', 'cooling_policy',
              'historical_rules_unchanged', 'trainer_lineage_unchanged',
              'hourly_writer_unchanged', 'public_source_unchanged'}
    need(type(value) is dict and set(value) == fields and value['version'] == VERSION, 'exact prospective operator policy')
    need(value['authority'] == AUTHORITY == guards.AUTHORITY, 'known operator authority')
    need(type(value['execute_allowed']) is bool, 'explicit execution authorization')
    for field in ('runner', 'parent_policy', 'parent_entry', 'selector', 'provider'):
        need(pin(value[field]['path']) == value[field], 'exact pinned ' + field)
    need(value['runner'] == pin(__file__) and value['runner_sha256'] == value['runner']['sha256'], 'physical operator entry')
    parent = guards.signed(json.loads(Path(value['parent_policy']['path']).read_bytes()))
    need(parent['version'] == 'prospective-capture64-H11-successor-v1' and parent['execute_allowed'] is True, 'actual active capture64 parent')
    need(parent['runner'] == value['parent_entry'], 'same admitted parent entry')
    for field in ('kind', 'config', 'baseline_runtime', 'source_sha256'):
        need(value[field] == parent[field], 'unchanged inherited ' + field)
    for field in ('historical_rules_unchanged', 'trainer_lineage_unchanged', 'hourly_writer_unchanged', 'public_source_unchanged'):
        need(value[field] is True, 'explicit unchanged scope ' + field)
    parent_module = load(value['parent_entry'], 'cooling_capture64_parent')
    inherited, config = parent_module.validate(parent, guards)
    selector = load(value['selector'], 'cooling_selector')
    p = selector.policy(value['cooling_policy'])
    need(p['first_round'] >= 154 and p['cooldown_count'] == 256, 'reviewed first broad cooling trial')
    need(p['epoch_prefix'] == config['epoch_prefix'] and p['calibration_anchor_indices'] == [1553, 7121], 'original stream and calibration anchors')
    need(p['approved_indices_sha256'] == hashlib.sha256(C(config['environments'][0]['indices'])).hexdigest(), 'actual original approved task pool')
    return parent_module, parent, inherited, config, selector


def signing(seed):
    from nacl.signing import SigningKey, VerifyKey
    key = SigningKey(bytes.fromhex(Path(seed).read_text().strip()))
    need(key.verify_key.encode().hex() == AUTHORITY, 'existing authority signing key')
    def sign(value):
        return {'payload': value, 'signer': AUTHORITY,
                'signature': base64.b64encode(key.sign(C(value)).signature).decode()}
    def verify(document):
        need(set(document) == {'payload', 'signer', 'signature'} and document['signer'] == AUTHORITY, 'snapshot authority envelope')
        VerifyKey(bytes.fromhex(AUTHORITY)).verify(C(document['payload']), base64.b64decode(document['signature'], validate=True))
        return document['payload']
    return sign, verify


def construct(value, guards, baseline, *, snapshot_sign, snapshot_verify):
    parent_module, parent, inherited, config, selector = validate(value, guards)
    service = parent_module.learner(parent, inherited, guards, baseline)
    provider_module = load(value['provider'], 'cooling_exposure_provider')
    provider = provider_module.evidence_provider
    from subnet import sample_harness
    previous = selector.install(service, value['cooling_policy'], provider,
                                project_harness=sample_harness.project,
                                sign_snapshot=snapshot_sign, verify_snapshot=snapshot_verify)
    service._task_cooling = copy.deepcopy(value['cooling_policy'])
    return service, config, inherited, previous, provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--unsigned-draft', action='store_true')
    args = parser.parse_args()
    need(not args.unsigned_draft or args.check_only, 'unsigned draft is check-only')
    document = json.loads(Path(args.policy).read_bytes())
    proposed = document if args.unsigned_draft else authenticated(document)
    need(pin(proposed['parent_policy']['path']) == proposed['parent_policy'], 'bootstrap parent file pin')
    parent_document = json.loads(Path(proposed['parent_policy']['path']).read_bytes())
    authenticated_parent = authenticated(parent_document)
    need(authenticated_parent['baseline_runtime'] == proposed['baseline_runtime'], 'authenticated inherited import path')
    sys.path.insert(0, authenticated_parent['baseline_runtime'])
    from ops import durable_audit_services as guards, durable_learner_service as baseline
    value = proposed if args.unsigned_draft else guards.signed(document)
    if args.unsigned_draft:
        need(value['execute_allowed'] is False, 'draft cannot execute')
    if args.check_only:
        import socket, sqlite3, subprocess, requests, torch, transformers
        from unittest.mock import patch
        from nacl.signing import SigningKey
        key = SigningKey(bytes(range(32)))
        test_authority = key.verify_key.encode().hex()
        def sign(v):
            return dict(payload=v, signer=test_authority, signature=base64.b64encode(key.sign(C(v)).signature).decode())
        def verify(v):
            need(v['signer'] == test_authority, 'fixture snapshot authority')
            key.verify_key.verify(C(v['payload']), base64.b64decode(v['signature'], validate=True))
            return v['payload']
        def denied(*args, **kwargs):
            raise AssertionError('check-only denies network, database, subprocess and singleton')
        with patch.object(socket.socket, 'connect', side_effect=denied), patch.object(socket, 'create_connection', side_effect=denied), \
             patch.object(sqlite3, 'connect', side_effect=denied), patch.object(subprocess, 'run', side_effect=denied), \
             patch.object(subprocess, 'Popen', side_effect=denied), patch.object(requests.sessions.Session, 'request', side_effect=denied), \
             patch.object(guards, 'singleton', side_effect=denied):
            service, config, _, old, provider = construct(value, guards, baseline, snapshot_sign=sign, snapshot_verify=verify)
            first = value['cooling_policy']['first_round']
            selector = sys.modules['cooling_selector']
            need(selector.contract_digest(service.contract(config, first - 1)) ==
                 selector.contract_digest(old(config, first - 1)), 'current issued rules untouched')
            original = old(config, first)
            time.sleep(1.1)
            need(selector.contract_digest(old(config, first)) == selector.contract_digest(original),
                 'stable original contract across renewed download URLs')
            # Authentic history is read once from live files. Only the private
            # fixture state and ephemeral signer are used to exercise freezing.
            epoch = config['epoch_prefix'] + '-check-' + str(first)
            provider_round = first - 1
            provider_epoch = config['epoch_prefix'] + '-check-' + str(provider_round)
            exposure = provider(config, provider_round, provider_epoch)
            with tempfile.TemporaryDirectory(prefix='affine-cooling-operator-check-') as root:
                local = copy.deepcopy(config); local['state'] = root
                Path(root, 'controller.json').write_bytes(C(dict(round=first, active=dict(phase='opening', epoch=epoch))))
                # Existing contracts contain state-local paths. Compare both
                # sides against this same private configuration, not live paths.
                original = old(local, first)
                # No provider reads or writes against the private fixture state.
                original_provider = lambda *_: copy.deepcopy(exposure)
                selector = sys.modules['cooling_selector']
                service.contract = old
                from subnet import sample_harness
                selector.install(service, value['cooling_policy'], original_provider,
                                 project_harness=sample_harness.project, sign_snapshot=sign, verify_snapshot=verify)
                result = service.contract(local, first)
                time.sleep(1.1)
                repeated = service.contract(local, first)
                need(selector.contract_digest(result) == selector.contract_digest(repeated),
                     'immutable snapshot survives repeated opening and URL renewal')
                left = copy.deepcopy(original); right = copy.deepcopy(result)
                left['environments'][0].pop('indices'); left['environments'][0].pop('harness')
                right['environments'][0].pop('indices'); right['environments'][0].pop('harness')
                need(selector.contract_digest(left) == selector.contract_digest(right),
                     'only allowed task indices and projected harness change')
                need(len(result['environments'][0]['indices']) == 6490, 'broad 6490-task opening')
                need(result['environments'][0]['indices'][:2] == [1553, 7121], 'original calibration workload')
            need(service._independent_training_dispatch['historical_routes_preserved'] is True, 'original trainer dispatch retained')
        print(json.dumps(dict(passed=True, check_only=True, allowed_tasks=6490,
                              current_epoch_unchanged=True, jobs_dispatched=False, network_calls=False,
                              source_sha256=value['source_sha256']), sort_keys=True))
        return
    need(value['execute_allowed'] is True, 'signed explicit execution required')
    _, _, inherited, config, _ = validate(value, guards)
    with guards.singleton(inherited['singleton_lock']):
        guards.no_predecessors(inherited)
        sign, verify = signing(Path(config['state']) / 'authority.seed')
        service, config, _, _, _ = construct(value, guards, baseline, snapshot_sign=sign, snapshot_verify=verify)
        service.run(config)


if __name__ == '__main__':
    main()
