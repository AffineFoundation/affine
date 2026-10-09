"""Publish mining availability from an authenticated, physically live learner.

This observer never opens an epoch or invents a protocol default. The learner's
signed policy pins its config, and the signed opening must agree with that config
and the controller checkpoint. Any failed binding closes public discovery.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

from ops.live_reward_exporter import atomic
from ops.mining_discovery import process_live
from subnet.live_reward_bridge import signed
from subnet.source_bootstrap import r2_url

LEARNER_UNIT = "affine-ordinary-v3-native-eligibility-durable-learner-v1.service"


def base_document(authority, now):
    return dict(schema=2, authority=authority, network="Finney", netuid=120,
                payable=False, chain_weight_submission=False,
                chain_writer_enabled=False, accepting_submissions=False,
                status="discovery_binding_error", status_updated_at=now,
                notice="Mining availability could not be authenticated. Watch for the next signed opening.")


def actual_selector(authority, unit=LEARNER_UNIT, *, proc=Path('/proc'), inspect_unit=None):
    """Bind the physical service lifetime to its signed policy and config bytes."""
    if inspect_unit is None:
        inspect_unit = lambda: subprocess.check_output(
            ['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'ActiveState', '-p', 'ExecStart'],
            text=True, timeout=10)
    fields = dict(line.split('=', 1) for line in inspect_unit().splitlines())
    pid = int(fields['MainPID'])
    if fields['ActiveState'] != 'active' or pid <= 0:
        raise ValueError('current learner absent')
    match = re.search(r'--policy ([^ ;]+)', fields['ExecStart'])
    if match is None:
        raise ValueError('current learner policy selector')
    policy_path = Path(match.group(1))
    policy = signed(json.loads(policy_path.read_text()), authority)
    if policy.get('execute_allowed') is not True:
        raise ValueError('learner execution is not authorized')
    argv = [part.decode() for part in (proc/str(pid)/'cmdline').read_bytes().split(b'\0') if part]
    if argv.count('--policy') != 1 or argv[argv.index('--policy')+1] != str(policy_path):
        raise ValueError('physical learner policy mismatch')
    runner_hash = policy.get('runner_sha256')
    if runner_hash is not None:
        scripts = [Path(arg) for arg in argv[1:argv.index('--policy')] if arg.endswith('.py')]
        if len(scripts) != 1 or hashlib.sha256(scripts[0].read_bytes()).hexdigest() != runner_hash:
            raise ValueError('physical learner runner mismatch')
    elif 'ops.durable_learner_service' not in argv:
        raise ValueError('unknown learner entry point')
    stat = (proc/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    record = dict(child_pid=pid, child_ticks=stat[19])
    if not process_live(record, proc):
        raise ValueError('learner lifetime ended')
    descriptor = policy['config']
    expected = descriptor.get('sha256', descriptor.get('file_sha256'))
    if expected is None or ('sha256' in descriptor and 'file_sha256' in descriptor
                            and descriptor['sha256'] != descriptor['file_sha256']):
        raise ValueError('ambiguous learner config hash')
    raw = Path(descriptor['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('learner config drift')
    config = json.loads(raw)
    if config['source_bundle']['sha256'] != policy['source_sha256']:
        raise ValueError('learner source mismatch')
    return config, record


def project(config, controller, discovery, envelope, authority, *, live, now):
    """Only copy contract claims present in the authenticated current opening."""
    result = base_document(authority, now)
    if not live:
        result.update(status='controller_unavailable', notice='Mining is closed while the learner is unavailable.')
        return result
    active = controller.get('active')
    if not active or active.get('phase') not in ('mine', 'collect'):
        result.update(status='between_epochs', notice='The learner is preparing the next signed opening.')
        return result
    manifest = signed(envelope, authority)
    if (manifest['epoch'] != active['epoch']
            or manifest['source_bundle']['sha256'] != config['source_bundle']['sha256']
            or any(manifest['checkpoint'].get(key) != controller['checkpoint'].get(key) for key in ('id', 'files'))
            or discovery['authority'] != authority
            or type(discovery['expires_at']) not in (int, float)
            or not math.isfinite(discovery['expires_at']) or discovery['expires_at'] <= now):
        raise ValueError('opening, source, checkpoint or discovery mismatch')
    contract = manifest.get('sampling_contract')
    configured = config.get('sampling_policy')
    if not isinstance(contract, dict) or not isinstance(configured, dict):
        raise ValueError('explicit signed sampling contract required')
    for field in ('version', 'max_attempts', 'support_adjudication'):
        if contract.get(field) != configured.get(field):
            raise ValueError('sampling policy mismatch')
    attempts = contract.get('max_attempts')
    if type(attempts) is not int or attempts < 1:
        raise ValueError('invalid sampling attempt budget')
    if not contract.get('generation') or not contract.get('verification') or not contract.get('randomness'):
        raise ValueError('incomplete signed sampling contract')
    for field in ('K', 'L', 'max_batches'):
        if type(manifest[field]) is not int or manifest[field] < 1 or manifest[field] != config[field]:
            raise ValueError('sample quota mismatch')
    transport = manifest.get('submission_transport_policy')
    if not transport or transport != config.get('submission_transport_policy'):
        raise ValueError('explicit submission transport mismatch')
    start, deadline = manifest['start'], manifest['deadline']
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, deadline))
            or deadline <= start):
        raise ValueError('invalid signed opening interval')
    expected_envs = {row['spec']['id']: row for row in config['environments']}
    environments = manifest['environments']
    if {row['env_id'] for row in environments} != set(expected_envs):
        raise ValueError('environment population mismatch')
    token_budgets = {}
    for row in environments:
        expected = expected_envs[row['env_id']]
        if row['spec'] != expected['spec'] or row['harness'] != expected['harness']:
            raise ValueError('environment or harness mismatch')
        token_budgets[row['env_id']] = row['harness']['max_output_tokens']
    accepting = start <= now < deadline
    result.update(
        current_url=r2_url(discovery['current_url']), expires_at=discovery['expires_at'],
        epoch_id=manifest['epoch'], epoch=manifest['epoch'], start=start, deadline=deadline,
        checkpoint=manifest['checkpoint']['id'], source_bundle_sha256=manifest['source_bundle']['sha256'],
        K=manifest['K'], L=manifest['L'], samples_per_batch=manifest['K']+manifest['L'],
        max_batches=manifest['max_batches'], max_attempts=attempts, nonce_min=0, nonce_max=attempts-1,
        sampling_version=contract['version'], sampling_contract=contract,
        submission_transport_policy=transport, max_output_tokens_by_environment=token_budgets,
        accepting_submissions=accepting, status='mining' if accepting else 'epoch_closed',
        notice='Mining is open. Follow the signed manifest and approved source.' if accepting
        else 'The signed upload deadline has passed. Watch for the next epoch.')
    if len(set(token_budgets.values())) == 1:
        result['max_output_tokens'] = next(iter(token_budgets.values()))
    for field in ('probability_artifact_policy', 'artifact_compression_policy', 'training_input_policy'):
        if field in manifest:
            if manifest[field] != config.get(field):
                raise ValueError(field+' mismatch')
            result[field] = manifest[field]
    return result


def publish(state, target, authority, *, unit=LEARNER_UNIT):
    now = time.time()
    error = None
    try:
        config, record = actual_selector(authority, unit)
        configured_state = Path(config['state'])
        if state is not None and Path(state).resolve() != configured_state.resolve():
            raise ValueError('caller state differs from signed learner state')
        controller = json.loads((configured_state/'controller.json').read_text())
        discovery = json.loads((configured_state/'direct-discovery.json').read_text())
        active = controller.get('active') or {}
        envelope = None
        if active.get('phase') in ('mine', 'collect'):
            envelope = json.loads((configured_state/(active['epoch']+'-first-signed-manifest.json')).read_text())
        result = project(config, controller, discovery, envelope, authority,
                         live=process_live(record), now=now)
        if not process_live(record):
            raise ValueError('learner lifetime changed during projection')
    except Exception as exc:
        result = base_document(authority, now)
        error = type(exc).__name__
    atomic(Path(target), result)
    Path(target).chmod(0o644)
    return dict(status=result['status'], accepting_submissions=result['accepting_submissions'],
                epoch_id=result.get('epoch_id'), source_bundle_sha256=result.get('source_bundle_sha256'),
                error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, help='Optional assertion; the signed learner policy selects state.')
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--learner-unit', default=LEARNER_UNIT)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        print(json.dumps(publish(args.state, args.target, args.authority, unit=args.learner_unit), sort_keys=True), flush=True)
        if not args.watch:
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
