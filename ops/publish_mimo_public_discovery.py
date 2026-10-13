#!/usr/bin/env python3
"""Read-only public mining projection for one explicitly activated MiMo host.

Run as the existing sole mining-discovery service after an explicit handoff.
It cannot open epochs, sign documents, upload objects, or submit chain weights.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.mimo_lifecycle import private_json
from subnet.backend_jobs import signed, r2_url
from subnet.mimo_release_policy import ReleasePolicy, descriptor, digest


def base_document(authority, now):
    return dict(schema=2, authority=authority, network='Finney', netuid=120,
        payable=None, chain_weight_submission=None, chain_writer_enabled=None,
        accepting_submissions=False, status='discovery_binding_error', status_updated_at=now,
        notice='Mining availability could not be authenticated. Watch for the next signed opening.')


def _pinned_reference(ref):
    if type(ref) is not dict or set(ref) != {'path', 'sha256'}:
        raise ValueError('exact observer source/config reference required')
    path = Path(ref['path'])
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']):
        raise ValueError('pinned observer source/config differs')
    return path


def process_live(record, proc=Path('/proc')):
    try:
        fields = (proc/str(record['child_pid'])/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] not in ('Z', 'X') and fields[19] == str(record['child_ticks'])
    except (OSError, KeyError, IndexError, ValueError):
        return False


def atomic(path, value):
    from subnet.storage import canonical
    temporary = path.with_name(path.name + '.mimo-observer-' + str(os.getpid()) + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def selector(observer, *, proc=Path('/proc'), inspect_unit=None):
    if observer.get('enabled') is not True:
        raise ValueError('MiMo discovery observer is disabled')
    config_path = _pinned_reference(observer['lifecycle_config'])
    entry = _pinned_reference(observer['lifecycle_entry'])
    config = private_json(config_path)
    if config.get('enabled') is not True:
        raise ValueError('MiMo lifecycle is disabled')
    policy = ReleasePolicy(config['release_policy'], observer['authority'], allow_public=True)
    if policy.mode != 'public':
        raise ValueError('private qualification cannot become public mining discovery')
    if policy.scope['immutable_manifest']['source_bundle']['sha256'] != observer['source_sha256']:
        raise ValueError('explicit public mining source binding differs')
    from subnet.mimo_worker import source_inventory
    if entry.parent.name != 'ops' or source_inventory(entry.parent.parent) != config['adapters']['source_files']:
        raise ValueError('running host source inventory differs')
    if inspect_unit is None:
        inspect_unit = lambda: subprocess.check_output(['systemctl', '--user', 'show',
            observer['lifecycle_unit'], '-p', 'MainPID', '-p', 'ActiveState'], text=True, timeout=10)
    fields = dict(line.split('=', 1) for line in inspect_unit().splitlines())
    pid = int(fields['MainPID'])
    if fields['ActiveState'] != 'active' or pid <= 0:
        raise ValueError('activated MiMo lifecycle is not running')
    argv = [part.decode() for part in (proc/str(pid)/'cmdline').read_bytes().split(b'\0') if part]
    scripts = [Path(arg) for arg in argv[1:] if arg.endswith('.py')]
    if (scripts != [entry] or argv.count('--config') != 1 or
            argv[argv.index('--config')+1] != str(config_path) or
            argv.count('--allow-public-mimo') != 1):
        raise ValueError('physical MiMo lifecycle config/entry/activation differs')
    stat = (proc/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    record = dict(child_pid=pid, child_ticks=stat[19])
    if not process_live(record, proc):
        raise ValueError('MiMo lifecycle lifetime ended')
    return config, policy, record


def project(policy, current, opening, receipt, *, authority, live, now):
    result = base_document(authority, now)
    if policy.mode != 'public':
        raise ValueError('private qualification cannot publish public discovery')
    if not live:
        result.update(status='controller_unavailable', notice='Mining is closed while the learner is unavailable.')
        return result
    if current['policy_sha256'] != policy.policy_sha256:
        raise ValueError('current lifecycle belongs to another release')
    if opening is None or receipt is None:
        result.update(status='between_epochs', notice='The learner is preparing the next signed opening.')
        return result
    manifest = signed(opening, authority)
    policy.validate_manifest(manifest)
    discovery = signed(receipt['document'], authority)
    if (receipt['sha256'] != digest(opening) or receipt['discovery_sha256'] != digest(receipt['document'])
            or discovery['manifest_sha256'] != digest(opening)
            or discovery['epoch'] != manifest['epoch'] or discovery['run_id'] != manifest['run_id']
            or discovery['checkpoint'] != manifest['checkpoint']['id']
            or discovery['source_bundle'] != manifest['source_bundle']
            or descriptor(manifest['checkpoint']) != descriptor(current['checkpoint'])
            or discovery['transport_policy'] != 'direct-r2-v1'):
        raise ValueError('current MiMo opening/discovery/checkpoint binding differs')
    start, deadline, expires = manifest['start'], manifest['deadline'], discovery['current_url_expires_at']
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, deadline, expires))
            or deadline <= start or expires <= now):
        raise ValueError('invalid or expired signed mining window/read capability')
    current_url = r2_url(discovery['current_url'], 'GET')
    r2_url(discovery['manifest_url'], 'GET')
    group = manifest['group_contract']
    accepting = start <= now < deadline
    result.update(status='mining' if accepting else 'epoch_closed', accepting_submissions=accepting,
        payable=manifest.get('payable') is True,
        run_id=manifest['run_id'], epoch=manifest['epoch'], epoch_id=manifest['epoch'],
        checkpoint=manifest['checkpoint']['id'], source_bundle_sha256=manifest['source_bundle']['sha256'],
        current_url=current_url, expires_at=expires, start=start, deadline=deadline,
        samples_per_batch=group['group_size'], min_success=group['min_success'],
        min_completed_failure=group['min_completed_failure'], nonce_min=0,
        nonce_max=group['group_size']-1, max_attempts=group['group_size'],
        max_batches=manifest['max_batches'], group_contract=group,
        sampling_version=manifest['sampling_contract']['version'],
        sampling_contract=manifest['sampling_contract'], required_client_features=manifest['required_client_features'],
        submission_transport_policy=manifest['submission_transport_policy'],
        max_output_tokens_by_environment={row['env_id']:row['harness']['max_output_tokens']
            for row in manifest['environments']},
        prospective_reward_eligible=True,
        notice='Mining is open. Follow the signed complete-group manifest and approved source.' if accepting
            else 'The signed upload deadline has passed. Watch for the next epoch.')
    limits = set(result['max_output_tokens_by_environment'].values())
    if len(limits) == 1: result['max_output_tokens'] = next(iter(limits))
    # Never project private upload capabilities, checkpoint credential material,
    # or assert anything about whether the independent hourly writer submitted.
    return result


def publish(observer, *, now=None):
    now = time.time() if now is None else now
    error = None
    try:
        config, policy, record = selector(observer)
        root = Path(config['state_root'])
        current = json.loads((root/'CURRENT.json').read_bytes())
        directory = root/'epochs'/str(current['epoch_index'])
        opening_path, receipt_path = directory/'coordinator/OPENING.json', directory/'DISCOVERY.json'
        opening = json.loads(opening_path.read_bytes()) if opening_path.is_file() else None
        receipt = json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else None
        result = project(policy, current, opening, receipt, authority=observer['authority'],
            live=process_live(record), now=now)
        if not process_live(record): raise ValueError('MiMo lifecycle lifetime changed during projection')
        if json.loads((root/'CURRENT.json').read_bytes()) != current:
            raise ValueError('MiMo checkpoint cursor advanced during projection')
    except Exception as exc:
        result = base_document(observer['authority'], now); error = type(exc).__name__
    target = Path(observer['target']); atomic(target, result); target.chmod(0o644)
    return dict(status=result['status'], accepting_submissions=result['accepting_submissions'],
        epoch=result.get('epoch'), error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    observer = private_json(args.config)
    # A disabled staging configuration has no filesystem/network side effects,
    # including no replacement of the existing site's valid mining document.
    if observer.get('enabled') is not True: raise ValueError('MiMo discovery observer is disabled')
    while True:
        print(json.dumps(publish(observer), sort_keys=True), flush=True)
        if not args.watch: return
        time.sleep(5)


if __name__ == '__main__': main()
