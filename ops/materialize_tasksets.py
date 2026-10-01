"""Create a prospective, disjoint training/held-out taskset from original providers.

Run in a scoped process with sufficient dataset-cache storage. This does not edit
active challenge configs, infer answers, run a miner, or submit chain writes.
"""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from subnet.environments import EnvironmentSpec, build_spec, snapshot_spec
from subnet.service import definitions
from subnet.storage import canonical


def split(count, training_count):
    if type(count) is not int or type(training_count) is not int or not 1 <= training_count < count <= 1024:
        raise ValueError('taskset requires nonempty disjoint training and held-out indices')
    return list(range(training_count)), list(range(training_count, count))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_bytes(canonical(value)); temp.replace(path)


def materialize(source, previous, output, count, training_count, taskset_override=None):
    training, heldout = split(count, training_count)
    config = copy.deepcopy(previous.config)
    config.pop('task_snapshot', None)
    config.setdefault('taskset', {}).update(taskset_override or {})
    spec = build_spec(source, config, num_samples=count, max_turns=previous.max_turns,
                      max_output_tokens=512, success_reward=previous.success_reward)
    path = output / (source + '.tasks.json')
    spec = snapshot_spec(spec, path)
    write(output / (source + '.spec.json'), spec.to_dict())
    return dict(source=source, status='materialized', samples=spec.num_samples,
                source_hash=spec.source_hash, snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                snapshot_bytes=path.stat().st_size, training_indices=training,
                heldout_indices=heldout, prospective=True, inferred_answers=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--spec-directory', type=Path, default=Path('state/original-task-snapshots'))
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--training-count', type=int, default=16)
    parser.add_argument('--taskset-overrides', type=Path)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--min-free-bytes', type=int, default=2 * 1024**3)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    split(args.count, args.training_count)
    if not 1 <= args.timeout <= 3600 or args.min_free_bytes < 2 * 1024**3:
        raise ValueError('bounded materialization resources')
    specs = {row['spec']['id']: EnvironmentSpec.from_dict(row['spec'])
             for row in definitions(json.loads(args.config.read_text()))} if args.config else {}
    for source in args.sources:
        if source not in specs:
            specs[source] = EnvironmentSpec.from_dict(json.loads((args.spec_directory / ('fixed4-' + source + '.spec.json')).read_text()))
    overrides = json.loads(args.taskset_overrides.read_text()) if args.taskset_overrides else {}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(args.sources) != 1: raise ValueError('one worker source')
        source = args.sources[0]
        print(json.dumps(materialize(source, specs[source], args.output, args.count,
                                     args.training_count, overrides.get(source)), sort_keys=True))
        return
    results = []
    for source in args.sources:
        start = time.time()
        if shutil.disk_usage(args.output).free < args.min_free_bytes:
            results.append(dict(source=source, status='disk_headroom_blocker', timestamp=start))
            write(args.output / 'materialization-report.json', results); break
        command = [sys.executable, '-m', 'ops.materialize_tasksets', '--worker', '--sources', source,
                   '--output', str(args.output), '--count', str(args.count), '--training-count', str(args.training_count),
                   '--spec-directory', str(args.spec_directory), '--timeout', str(args.timeout),
                   '--min-free-bytes', str(args.min_free_bytes)]
        if args.config: command += ['--config', str(args.config)]
        if args.taskset_overrides: command += ['--taskset-overrides', str(args.taskset_overrides)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            row = json.loads(result.stdout) if result.returncode == 0 else dict(source=source, status='error')
            row.update(returncode=result.returncode, stderr=result.stderr[-4000:])
        except subprocess.TimeoutExpired:
            row = dict(source=source, status='bounded_materialization_timeout')
        row.update(timestamp=start, elapsed=time.time()-start)
        results.append(row); write(args.output / 'materialization-report.json', results)
        print(source, row['status'], flush=True)


if __name__ == '__main__': main()
