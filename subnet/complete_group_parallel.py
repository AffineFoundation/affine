"""Experimental ordered N-attempt collection with bounded independent workers.

This module never changes the sampler, prompt, group size or outcome filter.
It is not enabled by Miner. Workers must each own a B1 model/cache or provide a
separately qualified serialization boundary. One coordinator owns this spool;
workers never write the attempt journal. Out-of-order completions survive a
restart and are emitted only in the prescribed nonce order.
"""
from concurrent.futures import FIRST_COMPLETED, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

VERSION = 'complete-group-parallel-spool-v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, raw):
    if path.is_symlink():
        raise ValueError('ordinary original attempt spool file')
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class NonceSpool:
    """Bounded original bytes; a changed binding cannot resume an older group."""
    def __init__(self, root, binding, group_size, *, max_attempt_bytes=16*1024**2):
        if (type(group_size) is not int or not 2 <= group_size <= 128 or
                type(binding) is not dict or not binding or
                type(max_attempt_bytes) is not int or not 1 <= max_attempt_bytes <= 128*1024**2):
            raise ValueError('explicit bounded original group and attempt binding')
        self.root = Path(root).absolute()
        if self.root.resolve() != self.root:
            raise ValueError('ordinary isolated nonce spool')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.stat().st_uid != os.getuid() or self.root.stat().st_mode & 0o077:
            raise ValueError('owner-only nonce spool')
        self.lock = os.open(self.root/'.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.n, self.limit = group_size, max_attempt_bytes
            header = dict(version=VERSION, binding=binding, group_size=group_size,
                          max_attempt_bytes=max_attempt_bytes)
            path = self.root/'STATE.json'
            if path.exists():
                if path.is_symlink() or path.stat().st_size > 1024**2:
                    raise ValueError('bounded ordinary original nonce state')
                self.state = json.loads(path.read_bytes())
                if set(self.state) != set(header) | {'attempts'} or any(self.state[k] != v for k,v in header.items()):
                    raise ValueError('original nonce group binding changed')
                rows = self.state['attempts']
                if type(rows) is not dict or not set(rows) <= {str(i) for i in range(self.n)}:
                    raise ValueError('original bounded nonce slots')
                for key in rows: self.read(int(key))
            else:
                self.state = dict(header, attempts={})
                self._persist()
        except BaseException:
            self.close(); raise
    def close(self):
        if getattr(self, 'lock', None) is not None:
            os.close(self.lock); self.lock = None
    def _persist(self): _write(self.root/'STATE.json', _canonical(self.state))
    def _nonce(self, nonce):
        if type(nonce) is not int or not 0 <= nonce < self.n:
            raise ValueError('one of the original ordered nonces required')
    def has(self, nonce):
        self._nonce(nonce)
        return str(nonce) in self.state['attempts']
    def put(self, nonce, raw):
        self._nonce(nonce)
        if type(raw) is not bytes or not 0 < len(raw) <= self.limit:
            raise ValueError('bounded original attempt bytes')
        if self.has(nonce):
            if self.read(nonce) != raw: raise ValueError('completed original attempt cannot change')
            return
        _write(self.root/f'{nonce}.bin', raw)
        self.state['attempts'][str(nonce)] = dict(sha256=_sha(raw), size=len(raw))
        self._persist()
    def read(self, nonce):
        self._nonce(nonce)
        row = self.state['attempts'][str(nonce)]; path = self.root/f'{nonce}.bin'
        if (type(row) is not dict or set(row) != {'sha256','size'} or
                type(row['size']) is not int or not 0 < row['size'] <= self.limit or
                path.is_symlink() or path.stat().st_size != row['size']):
            raise ValueError('original completed nonce file binding')
        raw = path.read_bytes()
        if _sha(raw) != row['sha256']: raise ValueError('original completed nonce digest changed')
        return raw
    def ordered(self):
        if len(self.state['attempts']) != self.n:
            raise ValueError('all N original attempts required; no cherry picking')
        return [self.read(nonce) for nonce in range(self.n)]


def collect(spool, submit, encode, *, workers=2, deadline=None, progress=None,
            abort=None, cancelled=lambda: False):
    """Submit only missing canonical nonces; drain and preserve in-flight results.

    submit(nonce) returns a Future. encode(nonce, result) must authenticate the
    returned nonce/context and produce original bytes. Outcome filtering belongs
    after all N results. Native errors must remain errors, never fabricated fails.
    Without abort the caller owns its hard execution timeout and pending calls
    drain. With abort, deadlines/cancellation stop the original workers promptly;
    only already completed original bytes are preserved, never replacements.
    """
    if type(workers) is not int or not 1 <= workers <= min(8, spool.n):
        raise ValueError('bounded independent inference workers')
    pending = [n for n in range(spool.n) if not spool.has(n)]
    active, errors = {}, []
    def expired(): return deadline is not None and time.time() >= deadline
    while pending or active:
        if (expired() or cancelled()) and not errors:
            errors.append(TimeoutError('original collection deadline passed'))
        if errors and abort is not None:
            for future in tuple(active):
                if future.done() and not future.cancelled():
                    nonce=active.pop(future)
                    try: spool.put(nonce,encode(nonce,future.result()))
                    except BaseException: pass  # Original first failure remains the cause.
            abort()
            break
        while pending and len(active) < workers and not errors:
            nonce = pending.pop(0)
            try: active[submit(nonce)] = nonce
            except BaseException as error: errors.append(error); break
        if not active: break
        ready, _ = wait(active, timeout=.5, return_when=FIRST_COMPLETED)
        for future in ready:
            nonce = active.pop(future)
            try:
                raw = encode(nonce, future.result())
                spool.put(nonce, raw)
                if progress is not None: progress(nonce, len(spool.state['attempts']))
            except BaseException as error: errors.append(error)
    if errors: raise errors[0]
    if expired(): raise TimeoutError('original collection deadline passed')
    return spool.ordered()


def append_original_prefix(spool, journal, task_key, decode):
    """Bridge completed bytes into the existing single-writer AttemptJournal.

    The normal miner remains responsible for native/proof shape validation,
    mixed-group filtering, transport persistence and final journal disposition.
    This helper never skips an unfinished nonce or replaces a prior prefix.
    """
    import numpy as np
    records = journal.begin(task_key)
    for nonce, (row, arrays) in enumerate(records):
        if not spool.has(nonce):
            raise ValueError('journal prefix missing original parallel attempt')
        original_row, original_arrays = decode(spool.read(nonce))
        if (_canonical(row) != _canonical(original_row) or len(arrays) != len(original_arrays) or
                any(not np.array_equal(a, b) for a,b in zip(arrays, original_arrays))):
            raise ValueError('journal prefix differs from original parallel attempt')
    while len(records) < spool.n and spool.has(len(records)):
        nonce = len(records); row, arrays = decode(spool.read(nonce))
        if type(row.get('seed')) is not int or row['seed'] != nonce:
            raise ValueError('original prescribed attempt nonce')
        journal.append(task_key, row, arrays)
        records.append((row, arrays))
    return records
