"""Evaluate fixed trusted task suites against completed immutable checkpoints.

This worker reads local controller evidence, never imports chain write APIs, and
does not influence miner scores. Errors remain evaluation records, not successes.
"""
import argparse
import hashlib
import json
import signal
import time
import math
from pathlib import Path


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def checkpoints(state):
    """Return only controller-journaled baseline and completed training outputs."""
    state = Path(state)
    first = read(state/'epoch-stage-0.json')
    result = []
    if first.get('checkpoint') and first.get('checkpoint_path'):
        result.append(dict(checkpoint=first['checkpoint']['id'], path=first['checkpoint_path'],
                           epoch_id=first['manifest']['epoch'], training_steps=0))
    steps = 0
    for counter, row in enumerate(read(state/'progress.json').get('history', []), 1):
        training = row.get('training') or {}
        if not training.get('weights_changed') or not row.get('accepted_batches'):
            continue
        steps += training['steps']
        result.append(dict(checkpoint=training['checkpoint'], path=str(state/f'checkpoint-{counter}'),
                           epoch_id=row['epoch_id'], training_steps=steps))
    return result


def run_id(checkpoint, definition, heldout, profile, runtime_revision=''):
    fingerprint = hashlib.sha256(canonical(dict(definition=definition, heldout=heldout,
                                                runtime_profile=profile,
                                                model_runtime_revision=runtime_revision))).hexdigest()
    return f'evalsuite-{checkpoint}-{fingerprint[:24]}'


def retry_due(record, now, interval, limit):
    """Completed evidence is immutable; errors get a bounded recovery budget."""
    attempts = record.get('evaluation_attempt', 1)
    timestamp = record.get('timestamp')
    return (record.get('status') == 'error' and type(attempts) is int and 0 < attempts < limit
            and type(timestamp) in (int, float) and math.isfinite(timestamp)
            and now - timestamp >= interval)


def evaluate_available(config):
    from .evaluation import evaluate
    from .model import model_files, check_runtime_profile, NUMERICAL_RUNTIME_REVISION
    output = Path(config.get('output', 'state/evaluations'))
    output.mkdir(parents=True, exist_ok=True)
    profile = config['runtime_profile']
    retry_interval = config.get('error_retry_interval', 300)
    retry_limit = config.get('max_evaluation_attempts', 3)
    if type(retry_interval) is not int or retry_interval < 30 or type(retry_limit) is not int or not 1 <= retry_limit <= 10:
        raise ValueError('evaluation retry budget')
    check_runtime_profile({'runtime_profile': profile})
    completed = []
    for checkpoint in checkpoints(config['checkpoint_state']):
        files = None
        for suite in config['suites']:
            definition = dict(env_id=suite['spec']['id'], spec=suite['spec'],
                              harness=suite['harness'], indices=suite['training_indices'])
            heldout = dict(indices=suite['heldout_indices'], seed=suite.get('seed', 20260930),
                           repeats=suite.get('repeats', 1))
            identifier = run_id(checkpoint['checkpoint'], definition, heldout, profile, NUMERICAL_RUNTIME_REVISION)
            frozen = dict(env_id=definition['env_id'],environment=definition['spec'],
                          harness=definition['harness'], indices=heldout['indices'],seed=heldout['seed'],
                          repeats=heldout['repeats'],runtime_profile=profile,
                          model_runtime_revision=NUMERICAL_RUNTIME_REVISION)
            destination = output/f'{identifier}.json'
            previous = None
            if destination.exists():
                previous = read(destination)
                if not retry_due(previous, time.time(), retry_interval, retry_limit):
                    continue
            attempt = previous.get('evaluation_attempt', 1) + 1 if previous else 1
            heldout.update(run_id=identifier, experiment_id=config.get('experiment_id', 'continuous-suite'),
                           training_steps=checkpoint['training_steps'])
            try:
                if files is None:
                    candidate_files = model_files(checkpoint['path'])
                    if not candidate_files or hashlib.sha256(canonical(candidate_files)).hexdigest() != checkpoint['checkpoint']:
                        raise ValueError('completed checkpoint does not match controller commitment')
                    files = candidate_files
                manifest = dict(epoch=checkpoint['epoch_id'], runtime_profile=profile,
                                model_runtime_revision=NUMERICAL_RUNTIME_REVISION,
                                checkpoint=dict(id=checkpoint['checkpoint'], files=files))
                record = evaluate(manifest, checkpoint['path'], definition, heldout, destination)
            except Exception as exc:
                record = dict(run_id=identifier, epoch_id=checkpoint['epoch_id'],
                              checkpoint=checkpoint['checkpoint'], env_id=definition['env_id'],
                              environment_version=suite['spec']['version'],
                              harness=suite['harness']['version']+':'+suite['harness']['policy'],
                              dataset_id=hashlib.sha256(canonical(frozen)).hexdigest(),
                              model_runtime_revision=NUMERICAL_RUNTIME_REVISION,
                              timestamp=time.time(), status='error', count=0, successes=0, mean_reward=None,
                              requested_count=len(heldout['indices'])*heldout['repeats'], completed_count=0,
                              training_steps=checkpoint['training_steps'], runtime_profile=profile,
                              error=type(exc).__name__+': '+str(exc), payable=False, weight_submission=False)
            record['evaluation_attempt'] = attempt
            # Preserve each exact attempt outside the dashboard's root-level
            # record scan; only the latest result occupies the logical run ID.
            attempts = output/'attempts'
            attempts.mkdir(exist_ok=True)
            if previous:
                prior = attempts/f'{identifier}-attempt-{attempt-1}.json'
                if not prior.exists():
                    prior.write_bytes(canonical(previous))
            archive = attempts/f'{identifier}-attempt-{attempt}.json'
            archive.write_bytes(canonical(record))
            temporary = destination.with_suffix('.tmp')
            temporary.write_bytes(canonical(record)); temporary.replace(destination)
            completed.append(dict(run_id=identifier, env_id=definition['env_id'],
                                  checkpoint=checkpoint['checkpoint'], status=record['status']))
            print(json.dumps(completed[-1]), flush=True)
    return completed


def main():
    # Let active environment sessions run their finally/close handlers during
    # a service stop, rather than leaving their isolated Docker sandbox alive.
    def stop(_signum, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='state/multi-environment/eval-suite.json')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 10:
        raise ValueError('poll interval must be at least ten seconds')
    while True:
        config = read(args.config)
        if config:
            evaluate_available(config)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
