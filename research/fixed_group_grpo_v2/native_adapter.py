"""Pure worker binding/admission around the reviewed private native CPU broker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from research.native_math_reuse_v2.broker import authenticate


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bind_capability(scope, plan, tasks, envelope, *, authority, native_runtime_binding,
                    current_interpreter=None):
    """Validate the same frozen TRAIN schedule before spawning any CPU process."""
    root = Path(scope['workspace'])
    payload = authenticate(envelope, authority, scope['run_id'], scope['native_environment_sha256'])
    require(payload['source_root'] == str(root/'source') and payload['source_files'] == scope['source_files'],
        'native broker exact executable source composition')
    require(payload['interpreter'] == scope['native_interpreter'], 'original native interpreter')
    require(Path(payload['broker_interpreter']).resolve() == Path(current_interpreter or sys.executable).resolve(),
        'native broker current prepared main interpreter')
    require(payload['native_runtime_binding'] == native_runtime_binding, 'exact native grader closure')
    require(scope['created_at'] <= payload['created_at'] < payload['expires_at'] <= scope['expires_at'],
        'native capability bounded by this original worker scope')
    require(len({t['index'] for t in tasks}) == len(tasks)
        and all(type(t['index']) is int and t['index'] >= 0 for t in tasks), 'unique original native TRAIN indices')
    expected = {str(t['index']):dict(task_hash=t['task_hash'], row_sha256=sha(t['native_snapshot_row'])) for t in tasks}
    require(payload['tasks'] == expected, 'native capability exactly matches frozen TRAIN rows')
    require(payload['max_requests'] == len(tasks)*(1+plan['group_size']), 'native exact complete schedule call ceiling')
    snapshot = Path(payload['snapshot_path'])
    require(snapshot.is_absolute() and snapshot.is_relative_to(root), 'native snapshot owned by worker')
    def worker_pin(path):
        if path.is_relative_to(root/'source'):
            return scope['source_files'].get(str(path.relative_to(root/'source')))
        return scope['files'].get(str(path.relative_to(root)))
    require(worker_pin(snapshot) == payload['snapshot_sha256']
        and payload['asset_files'].get(str(snapshot)) == payload['snapshot_sha256'], 'worker-pinned original native snapshot')
    # All non-source capability assets must have been authenticated by the worker.
    for name, digest in payload['asset_files'].items():
        path = Path(name)
        require(path.is_absolute() and path.is_relative_to(root)
            and worker_pin(path) == digest, 'native capability assets inside worker pins')
    return payload


class NativeAdapter:
    """No outcome cache, replacement attempts, or infrastructure-to-negative map."""
    def __init__(self, client, *, guard, record, final_box, eos_token_id):
        self.client = client
        self.guard = guard
        self.record = record
        self.final_box = final_box
        self.eos_token_id = eos_token_id
        self.logged = 0
        self.flush()

    def flush(self, **context):
        while self.logged < len(self.client.observations):
            row = self.client.observations[self.logged]
            self.record(dict(context, **row))
            self.logged += 1

    def preflight(self, task, *, update):
        self.guard()
        try:
            result = self.client.preflight(index=task['index'], task_hash=task['task_hash'],
                answer=task['native_snapshot_row']['data']['answer'])
            require(result in ('0.0', '1.0', 'reference_unparseable'), 'known-reference preflight schema')
            return result
        finally:
            self.flush(update=update, task_id=task['task_hash'])

    def grade_completed(self, task, tokens, tokenizer, *, update):
        self.guard()
        # Collector normally calls us only after EOS; protect the adapter too.
        if not tokens or tokens[-1] != self.eos_token_id:
            return 'unresolved'
        require(self.eos_token_id not in tokens[:-1], 'first EOS terminates native trajectory')
        reply = tokenizer.decode(tokens[:-1], skip_special_tokens=True)
        if self.final_box(reply) is None:
            return 'unresolved'
        try:
            scalar = self.client.grade(index=task['index'], task_hash=task['task_hash'], reply=reply)
            require(scalar in ('0.0', '1.0'), 'native binary verdict')
            return 'correct' if scalar == '1.0' else 'wrong'
        finally:
            self.flush(update=update, task_id=task['task_hash'])

    def close(self):
        try:
            self.client.close()
        finally:
            self.flush()
