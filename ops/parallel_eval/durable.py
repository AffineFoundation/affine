"""Crash-durable staging storage. There is deliberately no production default."""
import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('symlink output')
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(canonical(value) + b'\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('symlink input')
    return json.loads(path.read_bytes())


class Stage:
    MARKER = {'version': 'pe-staged-root-v1', 'live': False}

    def __init__(self, root):
        root = Path(root).absolute()
        if root.resolve() != root or any(p in root.parts for p in ('mimo-mainnet-v1', 'trainer-state')):
            raise ValueError('not an isolated staging root')
        if read(root / 'PE-STAGED-ONLY.json') != self.MARKER:
            raise ValueError('explicit staging marker required')
        self.root = root

    @classmethod
    def create(cls, root):
        root = Path(root).absolute()
        if root.resolve() != root or any(p in root.parts for p in ('mimo-mainnet-v1', 'trainer-state')):
            raise ValueError('not an isolated staging root')
        root.mkdir(parents=True, exist_ok=True)
        marker = root / 'PE-STAGED-ONLY.json'
        if marker.exists():
            return cls(root)
        if list(root.iterdir()):
            raise ValueError('refuse nonempty unmarked staging root')
        atomic(marker, cls.MARKER)
        return cls(root)

    @contextlib.contextmanager
    def locked(self):
        lock = self.root / '.state.lock'
        if lock.is_symlink():
            raise ValueError('symlink lock')
        with lock.open('a') as f:
            os.chmod(lock, 0o600)
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def load(self):
        return read(self.root / 'STATE.json')

    def save(self, value):
        atomic(self.root / 'STATE.json', value)
