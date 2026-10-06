"""Resume the approved learner under a durable ROOT-pinned CPU policy.

Startup changes no epoch, original job, checkpoint, or optimizer evidence.
The scientific controller retains responsibility for original-job observation.
"""
import argparse
import base64
import re
import hashlib
import json
import os
import stat
from pathlib import Path
import sys
from ops import durable_audit_services as guards

VERSION = 'durable-pinned-learner-service-v1'
OVERLAY_VERSION = 'coordinator-capture-overlay-v1'
CPU_OVERRIDES = {'subnet/capture_journal.py', 'subnet/training_documents.py',
                 'subnet/commitment_transport.py', 'subnet/controller.py',
                 'subnet/gpu_service.py', 'subnet/storage.py',
                 'subnet/remote_backend.py', 'subnet/checkpoint_upload_recovery.py'}


def validate_operator_overlay(overlay, source, policy, cfg):
    fields = {'version', 'root', 'full_source_files', 'overrides',
              'baseline_source_sha256', 'baseline_inventory_sha256', 'learner_capture_policy'}
    if (type(overlay) is not dict or set(overlay) != fields or
        overlay['version'] != OVERLAY_VERSION or
        overlay['baseline_source_sha256'] != policy['source_sha256'] or
        overlay['baseline_inventory_sha256'] != guards.digest(source['full_source_files'])):
        raise ValueError('exact ROOT-bound CPU overlay baseline')
    root = Path(overlay['root'])
    if (not root.is_absolute() or root.resolve() != root or root == Path(policy['source_root']) or
        not stat.S_ISDIR(root.lstat().st_mode) or root.lstat().st_uid != os.getuid()):
        raise ValueError('distinct canonical owned CPU overlay root')
    changes = overlay['overrides']
    if type(changes) is not dict or not changes or not set(changes) <= CPU_OVERRIDES:
        raise ValueError('only explicit coordinator transport modules may differ')
    original = source['full_source_files']
    if any(k not in original and k not in {'subnet/capture_journal.py', 'subnet/checkpoint_upload_recovery.py'} for k in changes):
        raise ValueError('only explicit CPU sidecars may extend original membership')
    expected = dict(original, **changes)
    if overlay['full_source_files'] != expected:
        raise ValueError('complete CPU overlay inventory and exact overrides')
    if any(original.get(k) == value for k, value in changes.items()):
        raise ValueError('redundant CPU overlay override declaration')
    # Even the source admission's historical cache exception does not permit
    # unlisted files in this fresh overlay or its baseline tree.
    for directory, files in ((Path(policy['source_root']), original), (root, expected)):
        observed = {str(f.relative_to(directory)) for f in directory.rglob('*') if f.is_file()}
        if observed != set(files) or any(f.is_symlink() for f in directory.rglob('*')):
            raise ValueError('exact ordinary source/overlay membership')
        guards.pinned_files(directory, files)
    capture = overlay['learner_capture_policy']
    keys = {'version', 'workers', 'max_document_bytes', 'max_inflight_bytes',
            'completion_order', 'journal_version', 'state_checkpoint_documents'}
    if (type(capture) is not dict or set(capture) != keys or
        capture['version'] != 'bounded-parallel-token-capture-v2' or
        type(capture['workers']) is not int or capture['workers'] not in (4, 8, 16) or
        type(capture['max_document_bytes']) is not int or capture['max_document_bytes'] != 2_000_000 or
        type(capture['max_inflight_bytes']) is not int or capture['max_inflight_bytes'] != capture['workers'] * 2_000_000 or
        capture['completion_order'] != 'first-completed' or
        capture['journal_version'] != 'fsynced-per-epoch-capture-v1' or
        type(capture['state_checkpoint_documents']) is not int or not 1 <= capture['state_checkpoint_documents'] <= 16 or
        cfg.get('learner_capture_policy') != capture):
        raise ValueError('exact prospective signed capture policy')
    if (cfg.get('training_input_policy') != 'committed-unaudited-training-v1' or
        cfg.get('submission_transport_policy') != 'small-commitment-pairs-v2' or
        cfg.get('hourly_execution_policy') is None):
        raise ValueError('CPU overlay cannot change scientific transport')
    return root


def verify_admission(row, authority):
    # Historical learner approvals predate BASE64 deployment policies. Bind
    # their exact bytes and declared encoding; never rewrite signed evidence.
    if guards.file_hash(row['path']) != row['file_sha256']:
        raise ValueError('historical admission bytes changed')
    document = guards.read(row['path'])
    encoding = row.get('signature_encoding', 'base64')
    if encoding == 'hex':
        if not re.fullmatch('[0-9a-f]{128}', document.get('signature', '')):
            raise ValueError('exact historical HEX signature required')
        document = dict(document, signature=base64.b64encode(bytes.fromhex(document['signature'])).decode())
    elif encoding != 'base64':
        raise ValueError('explicit admission signature encoding required')
    payload = guards.signed(document, authority)
    if guards.digest(payload) != row['payload_sha256']:
        raise ValueError('historical admission payload changed')
    return payload


def validate_policy(document, authority=guards.AUTHORITY):
    p = guards.signed(document, authority)
    fields = {'version', 'execute_allowed', 'authority', 'identity', 'config',
              'source_root', 'source_sha256', 'source_approval', 'qualification_approval',
              'reward_activation', 'authority_seed', 'singleton_lock', 'excluded_units',
              'runner_file_sha256', 'guards_file_sha256'}
    if 'operator_overlay' in p:
        fields.add('operator_overlay')
    if set(p) != fields or p['version'] != VERSION or p['execute_allowed'] is not True or p['authority'] != authority:
        raise ValueError('exact durable learner policy required')
    for path, expected in ((Path(__file__).resolve(), p['runner_file_sha256']),
                           (Path(guards.__file__).resolve(), p['guards_file_sha256'])):
        if guards.file_hash(path) != expected:
            raise ValueError('durable learner operator drift')
    identity = p['identity']
    if set(identity) != {'uid', 'machine_id_sha256'} or identity['uid'] != os.getuid() or identity['machine_id_sha256'] != hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest():
        raise ValueError('approved physical CPU identity required')
    if not isinstance(p['excluded_units'], list) or not p['excluded_units'] or any(not isinstance(v, str) or not v.endswith('.service') for v in p['excluded_units']):
        raise ValueError('explicit predecessor exclusion required')
    if guards.file_hash(p['config']['path']) != p['config']['file_sha256']:
        raise ValueError('approved config drift')
    cfg = guards.read(p['config']['path'])
    source = verify_admission(p['source_approval'], authority)
    qualification = verify_admission(p['qualification_approval'], authority)
    reward = guards.verify_document(p['reward_activation'], authority)
    if source['version'] != 'ordinary-orchestration-only-source-approval-v1' or source['approved'] is not True or source['source_sha256'] != p['source_sha256']:
        raise ValueError('approved scientific source required')
    if source['optimizer_reset'] is not False or source['historical_relabel'] is not False:
        raise ValueError('optimizer lineage and historical evidence must remain unchanged')
    guards.pinned_files(p['source_root'], source['full_source_files'])
    root = Path(p['source_root'])
    inventory = {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    if inventory != set(source['full_source_files']):
        raise ValueError('complete scientific source inventory required')
    if len(source['runtime_source_files']) != 177 or any(source['full_source_files'].get(k) != v for k, v in source['runtime_source_files'].items()):
        raise ValueError('approved 177-file runtime closure required')
    for row in source['evidence'].values():
        if guards.file_hash(row['path']) != row['file_sha256']:
            raise ValueError('source qualification evidence drift')
    if qualification['version'] != 'ordinary-orchestration-only-training-qualification-approval-v1' or qualification['approved'] is not True or qualification['candidate_source_sha256'] != p['source_sha256']:
        raise ValueError('approved training qualification required')
    translation = cfg['persistent_training_qualification_translation']
    if (translation['path'] != qualification['translation_path'] or
        translation['sha256'] != qualification['translation_file_sha256'] or
        guards.file_hash(translation['path']) != translation['sha256'] or
        translation['approval_path'] != p['qualification_approval']['path'] or
        translation['approval_sha256'] != p['qualification_approval']['file_sha256']):
        raise ValueError('exact qualification translation required')
    if cfg['source_bundle']['sha256'] != p['source_sha256'] or cfg['persistent_training_admission']['source_sha256'] != p['source_sha256'] or cfg['persistent_training_admission']['gpu_qualification_sha256'] != translation['sha256']:
        raise ValueError('persistent training source binding changed')
    if cfg.get('preparation_only') is not False or cfg.get('activation_allowed') is not True or cfg.get('activation_approved') is not True or cfg['deployment_gate']['execution_allowed'] is not True:
        raise ValueError('actual activation approval required')
    if cfg.get('token_artifact_policy') is not None or cfg.get('native_source_validation_policy') is not None:
        raise ValueError('scientific proof contract cannot change at CPU restart')
    if cfg.get('training_input_policy') != 'committed-unaudited-training-v1':
        raise ValueError('existing unaudited training policy required')
    if guards.read(p['reward_activation']['path']) != cfg['continuous_reward_activation_document'] or p['source_sha256'] not in reward['approved_sources']:
        raise ValueError('exact approved reward activation required')
    seed = p['authority_seed']
    if guards.file_hash(seed['path']) != seed['file_sha256'] or guards.private_file(seed['path']).stat().st_mode & 0o077:
        raise ValueError('private original authority seed required')
    from nacl.signing import SigningKey
    if SigningKey(bytes.fromhex(Path(seed['path']).read_text().strip())).verify_key.encode().hex() != authority:
        raise ValueError('original authority identity changed')
    state = Path(cfg['state'])
    if state.resolve() / 'authority.seed' != Path(seed['path']).resolve():
        raise ValueError('original learner state path required')
    # Never initialize a fresh controller during recovery. Its authenticated
    # remote controller handles existing jobs and durable optimizer commits.
    status = guards.read(guards.private_file(state / 'controller.json'))
    if status.get('initial_published') is not True or status.get('persistent_state_committed') is not True or not status.get('trainer_state'):
        raise ValueError('existing committed persistent learner required')
    if 'operator_overlay' in p:
        validate_operator_overlay(p['operator_overlay'], source, p, cfg)
    return p


def prepare_runtime(p):
    for name in list(sys.modules):
        if name == 'subnet' or name.startswith('subnet.'):
            del sys.modules[name]
    root = p['operator_overlay']['root'] if 'operator_overlay' in p else p['source_root']
    sys.path.insert(0, root)
    from subnet import gpu_service
    if Path(gpu_service.__file__).resolve() != Path(root) / 'subnet/gpu_service.py':
        raise ValueError('unexpected pinned learner import')
    return gpu_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if sys.flags.optimize:
        raise ValueError('scientific guards require unoptimized Python')
    p = validate_policy(guards.read(args.policy))
    with guards.singleton(p['singleton_lock']):
        guards.no_predecessors(p)
        service = prepare_runtime(p)
        if args.check:
            print(json.dumps({'checked': True, 'source': p['source_sha256'],
                              'operator_overlay': 'operator_overlay' in p,
                              'existing_state_preserved': True, 'jobs_dispatched': False}))
            return
        service.run(guards.read(p['config']['path']))


if __name__ == '__main__':
    main()
