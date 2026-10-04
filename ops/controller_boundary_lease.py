"""Hold one completed epoch boundary briefly; resume on expiry or interruption.

Only the original controller PID/start ticks and unchanged private config can
qualify. A changed epoch or missed boundary is observed, never cancelled. This
operator guard neither starts a controller nor submits chain weights.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import time
from pathlib import Path

from subnet.storage import canonical


def process(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return fields[0], fields[19]


def save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))
    Path(path).chmod(0o600)


def lease(config_path, record_path, epoch, output, wait_seconds=3600, hold_seconds=600, poll=.02):
    if (not isinstance(epoch, str) or re.fullmatch('[A-Za-z0-9_-]{1,200}', epoch) is None
            or type(wait_seconds) is not int or not 1 <= wait_seconds <= 21600
            or type(hold_seconds) is not int or not 1 <= hold_seconds <= 900
            or not .005 <= poll <= 1):
        raise ValueError('bounded exact epoch and lease budgets')
    config_path = Path(config_path); record_path = Path(record_path)
    for path in (config_path, record_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError('private original controller/config records')
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    record = json.loads(record_path.read_text()); config = json.loads(config_path.read_text())
    if record['config_sha256'] != config_hash:
        raise ValueError('original config digest changed')
    pid = record['child_pid']; ticks = record['child_ticks']
    def actual():
        p = process(pid)
        if p is None or p[1] != ticks or p[0] in ('Z', 'X'):
            return None
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_hash:
            raise ValueError('immutable controller config changed')
        if Path('/proc', str(pid), 'cwd').resolve() != Path(record['cwd']).resolve():
            raise ValueError('original controller workspace changed')
        return p[0]
    if actual() not in ('R', 'S', 'D'):
        raise ValueError('original running controller required')
    state_path = Path(config['state']) / 'controller.json'
    initial = json.loads(state_path.read_text()); active = initial.get('active') or {}
    if active.get('epoch') != epoch or type(initial['round']) is not int:
        raise ValueError('exact active epoch required')
    output = Path(output); output.mkdir(mode=0o700, parents=True, exist_ok=False)
    if output.is_symlink() or output.stat().st_mode & 0o077:
        raise ValueError('private unique boundary lease state')
    save(output / 'authorization.private.json', dict(at=time.time(), epoch=epoch, controller_pid=pid,
         controller_ticks=ticks, config_sha256=config_hash, original_round=initial['round'],
         original_training_steps=initial['training_steps'], wait_seconds=wait_seconds, hold_seconds=hold_seconds,
         lease_pid=os.getpid(), lease_ticks=process(os.getpid())[1]))
    owned = False; result = dict(held=False, reason='wait_elapsed_original_controller_preserved')
    deadline = time.monotonic() + wait_seconds
    try:
        while time.monotonic() < deadline:
            state = actual()
            if state is None:
                result = dict(held=False, reason='original_controller_exited'); return result
            if state == 'T':
                raise ValueError('another owner already holds the controller')
            before_bytes = state_path.read_bytes(); before = json.loads(before_bytes)
            if before['round'] > initial['round']:
                if before['round'] != initial['round'] + 1 or before.get('active') is not None:
                    result = dict(held=False, reason='next_epoch_already_active'); return result
                cp = before['checkpoint']
                if (not isinstance(cp.get('files'), dict) or not cp['files']
                        or hashlib.sha256(canonical(cp['files'])).hexdigest() != cp['id']
                        or before['training_steps'] < initial['training_steps']):
                    raise ValueError('completed checkpoint/step binding')
                os.kill(pid, signal.SIGSTOP); owned = True
                end = time.monotonic() + 2
                while actual() != 'T' and time.monotonic() < end:
                    time.sleep(.005)
                if actual() != 'T':
                    raise RuntimeError('original controller stop was not observed')
                if state_path.read_bytes() != before_bytes:
                    os.kill(pid, signal.SIGCONT); owned = False
                    continue
                held_at = time.time()
                receipt = dict(at=held_at, held=True, epoch=epoch, pid=pid, ticks=ticks,
                    actual_state='T', round=before['round'], training_steps=before['training_steps'],
                    checkpoint=cp['id'], active_epoch=None, config_sha256=config_hash,
                    status_sha256=hashlib.sha256(before_bytes).hexdigest(), lease_until=held_at + hold_seconds,
                    lease_pid=os.getpid(), lease_ticks=process(os.getpid())[1])
                save(output / 'actual-held-boundary.private.json', receipt)
                print(json.dumps(dict(held=True, checkpoint=cp['id'], lease_until=receipt['lease_until'])), flush=True)
                end = time.monotonic() + hold_seconds
                while time.monotonic() < end:
                    state = actual()
                    if state is None:
                        owned = False
                        result = dict(held=True, reason='original_controller_retired'); return result
                    if state != 'T':
                        owned = False
                        result = dict(held=True, reason='controller_already_resumed'); return result
                    time.sleep(min(poll, max(0, end - time.monotonic())))
                result = dict(held=True, reason='lease_expired'); return result
            if (before.get('active') or {}).get('epoch') != epoch:
                raise ValueError('original epoch changed before completion')
            time.sleep(poll)
        return result
    except BaseException as error:
        result = dict(held=owned, reason='lease_refused_or_interrupted', error_type=type(error).__name__)
        raise
    finally:
        # Match original ticks again even during error handling. Never signal a
        # reused PID, resume another owner's hold, or create a replacement job.
        current = process(pid)
        resumed = owned and current is not None and current[1] == ticks and current[0] == 'T'
        if resumed:
            os.kill(pid, signal.SIGCONT)
        save(output / 'release.private.json', dict(at=time.time(), result=result,
             resumed_original=resumed, original_pid=pid, original_ticks=ticks, jobs_restarted=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'controller-process', 'epoch', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--wait-seconds', type=int, default=3600)
    p.add_argument('--hold-seconds', type=int, default=600)
    a = p.parse_args(); os.umask(0o077)
    def interrupted(signum, frame):
        raise InterruptedError('boundary lease interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print(json.dumps(lease(a.config, a.controller_process, a.epoch, a.output,
                           a.wait_seconds, a.hold_seconds)), flush=True)


if __name__ == '__main__': main()
