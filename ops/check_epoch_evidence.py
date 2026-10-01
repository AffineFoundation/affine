"""Read-only independent checks of completed nonpayable epoch evidence.

Assumes the controller's local report authority is an operator trust anchor.
Reads signed R2 objects and frozen submission bytes; never submits chain writes.
"""
import argparse
import base64
import hashlib
import json
import math
import time
from pathlib import Path

from nacl.signing import VerifyKey
from botocore.exceptions import ClientError
from subnet.chain import hourly_points
from subnet.storage import Bucket, canonical


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while part := stream.read(1024 * 1024):
            result.update(part)
    return result.hexdigest()


def checkpoint_descriptor(bucket, identifier, authority):
    key=f'public/checkpoints/{identifier}/authorities/{authority}/checkpoint.json'
    try:
        envelope=json.loads(bucket.get(key))
    except (KeyError, ClientError) as error:
        if isinstance(error, ClientError):
            require(str(error.response.get('Error',{}).get('Code')) in ('NoSuchKey','404','NotFound'),
                    'checkpoint descriptor fetch failed')
        # Only older epochs precede authority-scoped descriptors. Invalid new
        # signatures never fall back to a less specific object.
        envelope=json.loads(bucket.get(f'public/checkpoints/{identifier}/checkpoint.json'))
    require(envelope['signer']==authority,'wrong checkpoint descriptor authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),
        base64.b64decode(envelope['signature'],validate=True))
    require(envelope['payload']['id']==identifier,'checkpoint descriptor identity mismatch')
    return envelope['payload']


def check(state, bucket):
    report = json.loads((state/'report.json').read_text())
    authority = report['authority']
    def signed(key):
        value = json.loads(bucket.get(key))
        require(value['signer'] == authority, 'wrong public artifact authority')
        VerifyKey(bytes.fromhex(authority)).verify(canonical(value['payload']),
                                                 base64.b64decode(value['signature'], validate=True))
        return value['payload']
    checked, score_reports = [], []
    for counter, row in enumerate(report['history'], 1):
        epoch = row['epoch_id']
        require(epoch.startswith('nonpayable-') and row['payable'] is False
                and row['weight_submission'] is False, 'payable epoch in test history')
        manifest = signed(f'public/{epoch}/manifest.json')
        scores = signed(f'public/{epoch}/scores.json')
        challenge = signed(f'public/{epoch}/audit-challenge.json')
        require(manifest['epoch'] == epoch and scores['epoch_id'] == epoch, 'public epoch binding')
        require(manifest['payable'] is False and scores['payable'] is False, 'payable public evidence')
        require(scores['checkpoint'] == manifest['checkpoint']['id'], 'wrong audited checkpoint')
        require(scores['points'] == row['points'] and scores['weights'] == row['proposed_weights'], 'local/public score mismatch')
        require(challenge['receipts'] == scores['receipts'], 'audit receipt binding')
        weights = scores['weights']
        require(all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in weights.values()), 'invalid weights')
        require(math.isclose(sum(weights.values()), 1.0) if scores['total'] else not weights, 'weight normalization')
        accepted = 0
        for miner, receipt in scores['receipts'].items():
            audit = signed(f'public/{epoch}/audits/{miner}.json')
            require(audit['epoch'] == epoch, 'audit epoch binding')
            require(hashlib.sha256(bucket.get(receipt['frozen_key'])).hexdigest() == receipt['sha256'], 'frozen bytes mismatch')
            require(audit['submission_sha256'] == receipt['sha256'], 'audit submission binding')
            require(audit['audit_seed'] == challenge['seed'], 'audit seed binding')
            require(all(outcome.get('fully_audited', False) for outcome in audit['outcomes']
                        if outcome.get('valid') is True), 'unchecked trajectory accepted')
            accepted += len(audit['accepted'])
        require(accepted == row['accepted_batches'], 'accepted batch count mismatch')
        item = dict(epoch_id=epoch, signatures_verified=True, frozen_bytes_verified=True,
                    accepted_batches=accepted, weights_normalized=True, payable=False,
                    score_basis=scores['score_basis'], provisional=scores['provisional'])
        if row.get('training'):
            training = signed(f'public/{epoch}/training.json')
            require(training == row['training'], 'training report mismatch')
            require(training['weights_changed'] and training['steps'] > 0, 'no training update')
            checkpoint = checkpoint_descriptor(bucket, training['checkpoint'], authority)
            require(checkpoint['id'] == training['checkpoint'], 'trained checkpoint descriptor mismatch')
            files = {}
            folder = state/f'checkpoint-{counter}'
            relevant = {p.name for p in folder.iterdir() if p.is_file()
                        and p.suffix in ('.json','.safetensors','.txt','.model','.jinja','.bin','.pt','.tiktoken')}
            require(relevant == set(checkpoint['files']), 'unexpected checkpoint files')
            for name, expected in checkpoint['files'].items():
                require(Path(name).name == name, 'checkpoint file path')
                files[name] = digest(folder/name)
                require(files[name] == expected, 'trained checkpoint file digest mismatch')
            require(hashlib.sha256(canonical(files)).hexdigest() == training['checkpoint'], 'trained checkpoint bytes mismatch')
            old_weights = {name: value for name, value in manifest['checkpoint']['files'].items()
                           if name.endswith('.safetensors')}
            new_weights = {name: value for name, value in files.items() if name.endswith('.safetensors')}
            require(old_weights and new_weights and old_weights != new_weights, 'model weight bytes unchanged')
            before, after = row['heldout_before'], row['heldout_after']
            require(before['dataset_id'] == after['dataset_id'] and before['task_hashes'] == after['task_hashes']
                    and before['runtime_profile'] == after['runtime_profile'], 'incomparable evaluation tasks/profile')
            require(after['checkpoint'] == training['checkpoint'] != before['checkpoint'], 'evaluation checkpoint mismatch')
            item.update(training_checkpoint=training['checkpoint'], training_steps=training['steps'],
                        model_weight_bytes_changed=True,
                        before_reward=before['mean_reward'], after_reward=after['mean_reward'])
        checked.append(item)
        score_reports.append(scores)
    require(hourly_points(score_reports, (int(time.time())//3600+1)*3600) == {}, 'test scores reached payout aggregation')
    return dict(timestamp=time.time(), epochs=checked, payout_filter_result={},
                chain_write_operations=0, goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=Path('state/multi-environment'))
    parser.add_argument('--bucket-config', type=Path, default=Path('state/mock-r2.json'))
    parser.add_argument('--output', type=Path, default=Path('state/multi-environment/independent-epoch-evidence.json'))
    args = parser.parse_args()
    result = check(args.state, Bucket(json.loads(args.bucket_config.read_text())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    args.output.chmod(0o600)
    print(json.dumps(dict(epochs_verified=len(result['epochs']), output=str(args.output),
                         payout_filter_result=result['payout_filter_result'])))


if __name__ == '__main__':
    main()
