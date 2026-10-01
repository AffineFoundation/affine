"""Inspect operator-audited native role artifacts without model or chain execution.

The separately supplied authority is the trust anchor. Signatures bind operator
reports; this command does not independently execute inference or environment
replay, and does not establish unbiased sampling provenance.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from subnet.backend_jobs import canonical
from subnet.native_role_batch import describe_sample, describe_batch, preference_pair


def inspect(paths, authority, index, k=None, l=None):
    if len(bytes.fromhex(authority)) != 32:
        raise ValueError('operator authority length')
    if (k is None) != (l is None):
        raise ValueError('supply both K and L')
    samples = [describe_sample(path, authority, index) for path in paths]
    summaries = []
    for sample in samples:
        views = sample['training_view']
        if any(any(row['loss_mask']) for row in views if not row['training_eligible']):
            raise ValueError('auxiliary tokens in loss')
        summaries.append(dict(
            checkpoint=sample['checkpoint'], task_hash=sample['task_hash'],
            trajectory_hash=sample['trajectory_hash'], reward=sample['reward'],
            classification=sample['classification'], files_hashed=len(sample['files']),
            artifact_bytes=sum(entry['size'] for entry in sample['files'].values()),
            model_roles=len(views),
            agent_target_tokens=sum(sum(row['loss_mask']) for row in views),
            auxiliary_target_tokens=0))
    result = dict(success=True, checked_at=time.time(), authority=authority,
        samples=summaries, fresh_model_execution_in_this_check=False,
        native_replay_execution_in_this_check=False, reports_are_operator_authenticated=True,
        training_performed=False, production_admitted=False, chain_transactions=False,
        validation_module_sha256=hashlib.sha256(Path('subnet/native_role_batch.py').read_bytes()).hexdigest())
    if k is not None:
        batch = describe_batch(samples, k, l)
        positive = next(s for s in samples if s['classification'] == 'positive')
        negative = next(s for s in samples if s['classification'] == 'negative')
        pair = preference_pair(positive, negative)
        result['batch'] = dict(K=batch['K'], L=batch['L'], environment_index=index,
            checkpoint=batch['checkpoint'], task_hash=batch['task_hash'],
            descriptor_sha256=hashlib.sha256(canonical(batch)).hexdigest())
        result['preference_pair'] = dict(objective=pair['objective'],
            prompt_tokens=len(pair['prompt']), chosen_tokens=len(pair['chosen']),
            rejected_tokens=len(pair['rejected']), auxiliary_tokens_in_loss=False,
            training_performed=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, action='append', required=True)
    parser.add_argument('--authority', required=True, help='Separately trusted operator public key')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--k', type=int)
    parser.add_argument('--l', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inspect(args.artifact, args.authority, args.index, args.k, args.l)
    if args.output:
        args.output.write_bytes(canonical(result))
        args.output.chmod(0o600)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
