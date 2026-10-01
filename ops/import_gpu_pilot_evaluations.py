"""Import operator-owned, artifact-bound GPU pilot metrics; no inference or chain writes."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(report, artifacts, timestamps):
    """Validate copied evidence before constructing either comparison point.

    The local report is an operator trust anchor. Hash checks authenticate its
    copied artifacts, not a new GPU recomputation or hardware attestation.
    """
    artifacts = Path(artifacts)
    require(report.get('success') is True and report.get('stage') == 'complete', 'incomplete GPU pilot')
    require(report.get('after_independent_verified') is True
            and all(report.get('pair_audits', {}).get(k) is True for k in ('positive','negative')),
            'missing independent GPU audits')
    training = report['training']
    require(training['weights_changed'] is True and training['steps'] > 0, 'missing real training')
    checkpoints = [report['approved_checkpoint_id'], report['new_checkpoint']['id']]
    for files, checkpoint in zip([report['approved_checkpoint_files'], report['new_checkpoint']['files']], checkpoints):
        require(hashlib.sha256(canonical(files)).hexdigest() == checkpoint, 'checkpoint identity mismatch')
    require(report['approved_checkpoint_files']['model.safetensors'] != report['new_checkpoint']['files']['model.safetensors'],
            'unchanged checkpoint weights')
    indices = report['heldout_indices']
    require(len(indices) == len(set(indices)) and not set(indices)&set(report['training_indices']), 'heldout overlap')
    pairs = []
    for phase in ('before', 'after'):
        rows = report['heldout_'+phase]
        require([r['index'] for r in rows] == indices, 'incomplete heldout population')
        verified = []
        for row in rows:
            name = row['name']
            require(re.fullmatch(r'[a-z0-9-]+', name) is not None, 'unsafe artifact name')
            require(row['verified'] is True, 'unaudited evaluation')
            for suffix in ('.json', '.npz'):
                require(sha(artifacts/(name+suffix)) == row['sha256'][suffix], 'artifact hash mismatch')
            doc = json.loads((artifacts/(name+'.json')).read_text())
            require(doc['index'] == row['index'] and doc['task_hash'] == row['task_hash'], 'task identity mismatch')
            require(doc['reward'] == row['reward'] and doc['classification'] == row['classification'], 'reward mismatch')
            require(type(doc['reward']) in (int,float) and math.isfinite(doc['reward']), 'nonfinite reward')
            require(doc['classification'] in ('positive','negative'), 'invalid classification')
            with np.load(artifacts/(name+'.npz'), allow_pickle=False) as arrays:
                require(set(arrays.files) == {'turn_'+str(i) for i in range(len(doc['turns']))}, 'probability turn mismatch')
                for i, turn in enumerate(doc['turns']):
                    probabilities = arrays['turn_'+str(i)]
                    require(probabilities.ndim == 2 and probabilities.shape[0] == len(turn['output'])
                            and probabilities.shape[1] > max(turn['output']) and np.isfinite(probabilities).all(),
                            'invalid probability array')
                    require(bool(turn['proofs']) and all(base64.b64decode(p, validate=True) for p in turn['proofs']), 'invalid proof encoding')
            verified.append(doc)
        pairs.append(verified)
    require([(d['index'], d['task_hash'], d['seed']) for d in pairs[0]] ==
            [(d['index'], d['task_hash'], d['seed']) for d in pairs[1]], 'incomparable heldout tasks/seeds')
    harness = {**report['harness'], 'policy':'autoregressive', 'max_output_tokens':16, 'turn_overrides':{}}
    dataset = dict(environment=report['environment'], model=report['model'],
                   profile=report['profile'], harness=harness,
                   tasks=[dict(index=d['index'], hash=d['task_hash'], seed=d['seed']) for d in pairs[0]])
    dataset_id = hashlib.sha256(canonical(dataset)).hexdigest()
    output = []
    for phase, documents, checkpoint in zip(('before','after'), pairs, checkpoints):
        # Recorded timestamps are remote artifact completion, not scoring-call time.
        completed = max(timestamps[r['name']+'.npz'] for r in report['heldout_'+phase])
        require(type(completed) in (int,float) and math.isfinite(completed), 'missing artifact completion time')
        count = len(documents)
        output.append(dict(run_id='gpu-pilot-'+dataset_id+'-'+checkpoint,
            epoch_id='nonpayable-gpu-pilot-'+report['new_checkpoint']['id'],
            checkpoint=checkpoint, model=report['model'], env_id=report['environment']['id'],
            environment_version=report['environment']['version'],
            harness=harness['version']+':autoregressive', policy_kind='autoregressive',
            model_runtime_revision=report['profile']['version'], dataset_id=dataset_id,
            taskset_hash=dataset_id, fixed_task_ids=[report['environment']['id']+':'+d['task_hash'] for d in documents],
            status='complete', count=count, requested_count=count, completed_count=count,
            successes=sum(d['classification']=='positive' for d in documents),
            mean_reward=sum(d['reward'] for d in documents)/count, timestamp=completed,
            timestamp_source='remote-artifact-write-completion',
            training_steps=0 if phase=='before' else training['steps'],
            experiment_kind='isolated-same-GPU-training-pilot',
            training_objective=training['objective'], full_model_finetune=training['full_model_finetune'],
            payable=False, weight_submission=False))
    require(output[0]['timestamp'] < output[1]['timestamp'], 'inverted evaluation timeline')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('state/multi-environment/gpu-training-pilot'))
    parser.add_argument('--output', type=Path, default=Path('state/evaluations'))
    args=parser.parse_args()
    report=json.loads((args.artifacts/'report.json').read_text())
    raw=json.loads((args.artifacts/'source-remote-artifact-mtimes.json').read_text())
    timestamps={row['name']:row['source_remote_mtime_ns']/1_000_000_000 for row in raw['files']}
    rows=records(report,args.artifacts,timestamps)
    args.output.mkdir(parents=True,exist_ok=True)
    for row in rows:
        path=args.output/(row['run_id']+'.json')
        if path.exists(): require(json.loads(path.read_text()) == row, 'immutable evaluation changed')
        else:
            temporary=path.with_suffix('.tmp');temporary.write_bytes(canonical(row));temporary.replace(path)
    print(json.dumps({'records_imported':len(rows),'environment':rows[0]['env_id'],
                      'before_reward':rows[0]['mean_reward'],'after_reward':rows[1]['mean_reward']}))


if __name__ == '__main__': main()
