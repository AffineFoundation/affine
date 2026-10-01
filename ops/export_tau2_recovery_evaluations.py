"""Project authenticated Tau2 recovery evidence without rewriting evaluations."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from subnet.backend_jobs import canonical, signed

VERSION = 'tau2-explicit-seeded-recovery-projection-v1'

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def project(repository, epoch_folder, authority):
    repository = repository.resolve()
    folder = epoch_folder.resolve()
    rollup_path = folder / 'signed-completion-with-explicit-recovery.json'
    rollup = signed(read(rollup_path), authority)
    if rollup.get('status') != 'completed_with_explicit_recovery':
        raise ValueError('explicit recovery completion required')
    if any(rollup.get(k) is not False for k in ('original_failure_reports_rewritten',
            'extra_optimizer_during_recovery_or_successor', 'quality_improvement_claimed',
            'payable', 'chain_transactions')):
        raise ValueError('preserved nonpayable recovery scope required')
    if re.fullmatch(r'[A-Za-z0-9_-]{1,200}', rollup['epoch']) is None:
        raise ValueError('safe public epoch identity required')
    inventory = rollup['evidence_file_inventory']
    for name, item in inventory.items():
        path = (repository / name).resolve()
        if not path.is_relative_to(repository) or not path.is_file():
            raise ValueError('bounded recovery evidence path')
        if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError('immutable recovery evidence changed')
        if item['signature_authenticated']:
            document = read(path)
            for envelope in document if isinstance(document, list) else [document]:
                signed(envelope, authority)
    approval = signed(read(folder / 'dashboard-projection-approval.json'), authority)
    if approval['rollup_sha256'] != sha(rollup_path) or approval.get('chain_transactions') is not False:
        raise ValueError('original completion projection binding')
    for name in ('before-heldout-contract.json', 'after-heldout-contract.json'):
        item = approval['contracts'][name]
        path = folder / name
        if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError('approved heldout metadata changed')
    names = ('before-evaluation.json', 'after-evaluation.json',
             'before-heldout-contract.json', 'after-heldout-contract.json')
    for name in names[:2]:
        relative = str((folder / name).relative_to(repository))
        if relative not in inventory:
            raise ValueError('historical evaluation binding missing')
    before, after, contract, after_contract = [read(folder / name) for name in names]
    if contract != after_contract or before['dataset_id'] != after['dataset_id']:
        raise ValueError('historical cohort changed')
    if before['dataset_id'] != contract['dataset_id']:
        raise ValueError('historical cohort identity')
    comparisons = rollup['heldout_task_seed_comparisons']
    if len(comparisons) != 16 or {r['index'] for r in comparisons} != set(range(16, 32)):
        raise ValueError('complete heldout population required')
    before_rows = {r['index']: r for r in before['records'] if r['verified'] is True}
    original_after = {r['index']: r for r in after['records'] if r['verified'] is True}
    before_rewards, after_rewards, task_ids, seed_rows = [], [], [], []
    recovered = 0
    for comparison in sorted(comparisons, key=lambda row: row['index']):
        index = comparison['index']
        episode = (repository / comparison['after_episode_path']).resolve()
        if not episode.is_relative_to(repository):
            raise ValueError('bounded heldout episode path')
        verification_path = episode / 'independent-full-verification.json'
        if str(verification_path.relative_to(repository)) not in inventory:
            raise ValueError('verified heldout evidence binding missing')
        verification = read(verification_path)
        baseline = before_rows[index]
        if (baseline['task_hash'] != comparison['task_hash'] or
                baseline['reward'] != comparison['before_reward'] or
                verification['task_hash'] != comparison['task_hash'] or
                verification['reward'] != comparison['after_reward']):
            raise ValueError('paired task outcome binding')
        if comparison['after_origin'] == 'original_verified':
            if original_after[index]['reward'] != verification['reward']:
                raise ValueError('original verified outcome changed')
        elif comparison['after_origin'] == 'explicit_recovery_attempt_1':
            if index in original_after:
                raise ValueError('recovery overwrites original successful task')
            recovered += 1
        else:
            raise ValueError('unknown recovery origin')
        if any(verification.get(k) is not True for k in (
                'all_model_roles_verified', 'derived_responses_verified', 'full_native_trajectory_verified')):
            raise ValueError('full independent verification required')
        if verification.get('trajectory_attempt') != comparison['trajectory_attempt']:
            raise ValueError('paired trajectory attempt binding')
        for value in (baseline['reward'], verification['reward']):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('finite heldout reward required')
        before_rewards.append(baseline['reward'])
        after_rewards.append(verification['reward'])
        task_ids.append(comparison['task_hash'])
        seed_rows.append({k: comparison[k] for k in ('index', 'task_hash', 'task_seed',
                         'trajectory_attempt', 'agent_seed_start', 'agent_seed_policy')})
    if (len(before_rows) != 16 or len(original_after) != 11 or recovered != 5 or
            after['error_count'] != 5 or rollup['optimizer_steps_total'] != 1):
        raise ValueError('original and separate recovery counts')
    if (after['checkpoint'] != rollup['checkpoint'] or before['checkpoint'] == after['checkpoint'] or
            sum(before_rewards)/16 != rollup['before_mean_reward'] or
            sum(after_rewards)/16 != rollup['effective_after_mean_reward']):
        raise ValueError('checkpoint and measured reward binding')
    dataset = hashlib.sha256(canonical(dict(version=VERSION,
        original_dataset_id=contract['dataset_id'], seeds=seed_rows))).hexdigest()
    geometry = contract['agent_geometry_and_policy']
    common = dict(epoch_id=rollup['epoch'], model='Affine/Tau2-agent', env_id='affine_tau2',
        environment_version=contract['environment']['version'], harness='native-tau2-fixed-auxiliary',
        dataset_id=dataset, seed=seed_rows[0]['agent_seed_start'], requested_count=16,
        fixed_task_ids=task_ids, taskset_hash=contract['taskset_sha256'],
        policy_kind='autoregressive', model_runtime_revision=geometry['model_runtime_revision'],
        harness_config={'max_output_tokens': geometry['max_output_tokens']})
    rows = []
    for phase, checkpoint, rewards, timestamp, status, steps, recovered_count in (
        ('before', before['checkpoint'], before_rewards, max(r['completed_at'] for r in before_rows.values()), 'complete', 0, 0),
        ('original-after', after['checkpoint'], [r['reward'] for r in original_after.values()],
         max(r['completed_at'] for r in original_after.values()), 'partial', 1, 0),
        ('recovered-after', rollup['checkpoint'], after_rewards, rollup['completed_at'], 'complete', 1, recovered)):
        rows.append(dict(common, run_id=rollup['epoch']+'-'+VERSION+'-'+phase,
            checkpoint=checkpoint, count=len(rewards), completed_count=len(rewards),
            successes=sum(reward == 1 for reward in rewards), mean_reward=sum(rewards)/len(rewards),
            timestamp=timestamp, training_steps=steps, status=status,
            original_error_count=5 if phase != 'before' else 0, recovered_count=recovered_count,
            status_detail='completed-with-explicit-recovery' if recovered_count else
                          ('original-partial' if status == 'partial' else 'original-complete')))
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path('.'))
    parser.add_argument('--epoch-folder', type=Path, required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--output', type=Path, default=Path('state/evaluations'))
    args = parser.parse_args()
    rows = project(args.repository, args.epoch_folder, args.authority)
    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = args.output / (row['run_id']+'.json')
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(canonical(row))
        os.replace(temporary, target)
    print(json.dumps(dict(projected_records=len(rows), original_errors_preserved=5,
                          separately_recovered=5, chain_transactions=False)))

if __name__ == '__main__':
    main()
