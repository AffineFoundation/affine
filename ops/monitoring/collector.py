"""Read-only monitor scheduler. Writes only derived panel telemetry."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from .common import ROOT, metric, panel, table

GROUPS = {
    'validator': (10, ('validator', 'weights')),
    'submissions': (20, ('queue', 'registrations')),
    'scoring': (10, ('duel', 'scores', 'dialects', 'history')),
    'evalpods': (15, ('eval', 'bench-engine', 'chat')),
    'swarm': (10, ('teacher', 'router')),
    'corpus': (120, ('corpus', 'corpus-fold')),
    'datagen': (120, ('datagen', 'traces')),
    'advisory': (60, ('benchmarks', 'audits')),
    'infrastructure': (120, ('services', 'fleet')),
}
LANES = [
    ('Control plane', ['registrations', 'queue', 'validator', 'weights']),
    ('Evaluation', ['eval', 'duel', 'scores', 'dialects', 'history']),
    ('Teacher serving', ['teacher', 'router']),
    ('Data pipeline', ['datagen', 'traces', 'corpus-fold', 'corpus']),
    ('Independent branches', ['bench-engine', 'benchmarks', 'audits', 'chat']),
    ('Infrastructure', ['services', 'fleet']),
]

def publish(directory, key, payload):
    content = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.chmod(0o600)
        os.replace(temporary, directory / f'{key}.json')
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def collect_group(name, interval):
    started = time.monotonic()
    try:
        values = importlib.import_module('.' + name, __package__).collect()
        if set(values) != set(GROUPS[name][1]):
            raise ValueError('Unexpected panel keys')
        json.dumps(values, allow_nan=False)
    except Exception:
        values = {key: panel(key.replace('-', ' ').title(), 'Read-only collector unavailable', status='error',
                            notes=['Collector failed. Raw diagnostics withheld; other sources continue independently.'])
                  for key in GROUPS[name][1]}
    for value in values.values():
        value['refresh_seconds'] = interval
        value['stale_after_seconds'] = max(45, interval * 3)
        value['collection_seconds'] = round(time.monotonic() - started, 3)
    return values


def overview(values):
    nodes = []
    for lane, keys in LANES:
        for key in keys:
            value = values.get(key, {})
            state = value.get('status', 'unknown')
            try:
                age = time.time() - datetime.fromisoformat(value['collected_at']).timestamp()
                if age < -60 or age > value.get('stale_after_seconds', 45):
                    state = 'unknown'
            except (KeyError, TypeError, ValueError, OverflowError):
                state = 'unknown'
            nodes.append(dict(id=key, lane=lane, title=value.get('title', key),
                              status=state,
                              collected_at=value.get('collected_at'),
                              stale_after_seconds=value.get('stale_after_seconds', 45),
                              metrics=value.get('metrics', [])[:2]))
    counts = {state: sum(n['status'] == state for n in nodes) for state in ('ok', 'warn', 'error', 'unknown')}
    result = panel('Affine · system map', 'SN120 · read-only operational observatory',
        status='error' if counts['error'] else 'warn' if counts['warn'] or counts['unknown'] else 'ok',
        metrics=[metric('Observed', counts['ok']), metric('Review', counts['warn']),
                 metric('Unavailable / error', counts['error']), metric('Awaiting', counts['unknown'])],
        notes=['Status summarizes collector checks, not a proof of end-to-end correctness. Independent sources are sampled at different times.',
               'Benchmarks are advisory. Crown/score does not establish coding capability (RT-7 remains open). Audits are a separate post-crown enforcement path.',
               'Live panels are read-only. No serving knobs, scoring rules, weights, processes or submissions are changed.'],
        sources=['ops/monitoring/ARCHITECTURE.md', 'affine/affine.toml'])
    result.update(nodes=nodes, refresh_seconds=5, stale_after_seconds=30)
    return result


def run(directory, once=False):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.collector.lock').open('w') as lockfile:
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A collector already owns this output directory.')
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        values, guard = {}, threading.Lock()
        def worker(name, interval):
            while not stop.is_set():
                result = collect_group(name, interval)
                for key, value in result.items():
                    publish(directory, key, value)
                with guard:
                    values.update(result)
                print(f'{name}: ' + ', '.join(f'{k}={v["status"]}' for k, v in result.items()), flush=True)
                if once or stop.wait(interval):
                    break
        threads = [threading.Thread(target=worker, args=(name, interval), daemon=True)
                   for name, (interval, _) in GROUPS.items()]
        for thread in threads:
            thread.start()
        while not stop.is_set():
            with guard:
                publish(directory, 'overview', overview(values))
            if once and not any(thread.is_alive() for thread in threads):
                break
            stop.wait(1 if once else 5)
        print('collector completed' if once else 'collector stopped', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'panels/data')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.once)
