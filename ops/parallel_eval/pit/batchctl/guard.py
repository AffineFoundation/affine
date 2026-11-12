"""Shadow-mode hard denies.

This pull request ships shadow mode only. Each deny below is a hard error,
not a setting:

* R2: every key that the controller reads, writes, copies or deletes must be
  under ``pit-shadow/batchctl/<run_id>/``. Live keys are out of reach.
* Local files: the controller writes four named files in one marked state
  root. The root must be outside every live root.
* Boxes: the controller destroys a box only if its name is exactly a name
  that this run made. A shared name prefix is not enough.
* Mode: any mode other than ``shadow`` is refused.

The package has no code that talks to the chain, the website or Discord. A
test scans the imports of every module to keep it that way.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from . import BOX_PREFIX_SHADOW, MODE_SHADOW, R2_SHADOW_ROOT, VERSION
from .interfaces import Blob, Digest, ObjectInfo, ObjectStore
from ..fakepass_sandbox import FakepassError, assert_paths_outside_live, live_roots_from_env
from ..shadow import FORBIDDEN_BASENAMES, FORBIDDEN_SHADOW_WRITES, ShadowWriteError

# Used with ``fullmatch`` only: ``$`` would also accept a trailing newline.
RUN_ID = re.compile(r'[a-z0-9]{4,32}')
BOX_SUFFIX = re.compile(r'f([1-9][0-9]{0,11})-t(0|[1-9][0-9]{0,8})')

# Names of live artifacts. No key and no local file may end with one of them,
# even inside the shadow prefix or the shadow state root.
FORBIDDEN_NAMES = frozenset(
    name.lower() for name in (
        set(FORBIDDEN_BASENAMES) | set(FORBIDDEN_SHADOW_WRITES) | {
            'mining.json', 'llms.txt', 'llms-full.txt', 'ledger.json', 'gate-ledger.json',
            'weights.json', 'extrinsics.json', 'index.html',
        }
    )
)

MARKER = 'BATCHCTL-SHADOW-ROOT.json'
JOURNAL = 'journal.jsonl'
LOCK = 'LOCK'
STATUS = 'status.json'
STATE_FILES = frozenset({MARKER, JOURNAL, LOCK, STATUS})
# Temporary names that the journal and the status writer rename from.
TEMP_SUFFIXES = ('.create.tmp', '.repair.tmp', '.tmp')


class LiveModeRefused(ShadowWriteError):
    """Only shadow mode exists in this version."""


def check_mode(mode: str) -> None:
    if mode != MODE_SHADOW:
        raise LiveModeRefused('batchctl runs in shadow mode only (got %r)' % (mode,))


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
        raise ShadowWriteError('run id must match %s' % RUN_ID.pattern)
    return run_id


def shadow_prefix(run_id: str) -> str:
    """The only R2 prefix that a run may touch."""
    return '%s%s/' % (R2_SHADOW_ROOT, check_run_id(run_id))


def check_key(prefix: str, key: str) -> str:
    """Refuse every key that is not a plain key under the shadow prefix."""
    if not prefix.startswith(R2_SHADOW_ROOT) or not prefix.endswith('/'):
        raise ShadowWriteError('R2 prefix is not a shadow prefix')
    if not isinstance(key, str) or not key.startswith(prefix) or len(key) == len(prefix):
        raise ShadowWriteError('shadow refuses R2 key outside its prefix')
    parts = key.split('/')
    if any(part in ('', '.', '..') for part in parts[:-1]) or parts[-1] in ('.', '..'):
        raise ShadowWriteError('shadow refuses R2 key with an empty or dot segment')
    if '\\' in key or '\x00' in key:
        raise ShadowWriteError('shadow refuses R2 key with a backslash or NUL')
    if parts[-1].lower() in FORBIDDEN_NAMES:
        raise ShadowWriteError('shadow refuses R2 key named like a live artifact')
    return key


class GuardedStore:
    """An ``ObjectStore`` that can reach the shadow prefix of one run only.

    Reads are fenced as well as writes: the controller has no reason to read
    a live key, and a read of the wrong prefix would hide a wiring mistake.
    """

    def __init__(self, inner: ObjectStore, run_id: str):
        self._inner = inner
        self.prefix = shadow_prefix(run_id)

    def _key(self, key: str) -> str:
        return check_key(self.prefix, key)

    def get(self, key: str, *, max_bytes: int) -> Blob:
        return self._inner.get(self._key(key), max_bytes=max_bytes)

    def head(self, key: str) -> Optional[ObjectInfo]:
        return self._inner.head(self._key(key))

    def digest(self, key: str) -> Digest:
        return self._inner.digest(self._key(key))

    def put(
        self, key: str, data: bytes, *, if_none_match: bool = False, if_match: Optional[str] = None,
    ) -> str:
        return self._inner.put(self._key(key), data, if_none_match=if_none_match, if_match=if_match)

    def copy(self, src: str, dst: str, *, src_etag: str) -> None:
        self._inner.copy(self._key(src), self._key(dst), src_etag=src_etag)

    def delete(self, key: str) -> None:
        self._inner.delete(self._key(key))

    def list(self, prefix: str) -> List[ObjectInfo]:
        if not prefix.startswith(self.prefix) or '..' in prefix.split('/'):
            raise ShadowWriteError('shadow refuses R2 listing outside its prefix')
        found = self._inner.list(prefix)
        for info in found:
            if not info.key.startswith(prefix):
                raise ShadowWriteError('store listed a key outside the asked prefix')
        return found


# ----------------------------------------------------------------------------
# Local state root
# ----------------------------------------------------------------------------

def _under(path: Path, root: Path) -> bool:
    return path == root or str(path).startswith(str(root) + os.sep)


def check_outside_live(path: Path, env: Optional[Mapping[str, str]] = None) -> Path:
    """Refuse a state root that is inside a live root, or holds one."""
    resolved = Path(path).resolve()
    lives = live_roots_from_env(env)
    try:
        assert_paths_outside_live({'batchctl_state_root': resolved}, lives)
    except FakepassError as exc:
        raise ShadowWriteError('shadow state root is inside a live root') from exc
    for live in lives:
        try:
            live_resolved = Path(live).resolve()
        except (OSError, RuntimeError):
            continue
        if _under(live_resolved, resolved):
            raise ShadowWriteError('a live root is inside the shadow state root')
    return resolved


class StateRoot:
    """The one local directory that the controller may write.

    ``create`` makes the directory and its marker. ``open`` needs the marker
    of the same run. Both refuse a directory that is in or around a live
    root, or that holds any file the controller does not own.
    """

    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id

    @classmethod
    def create(cls, path: Path, run_id: str, env: Optional[Mapping[str, str]] = None) -> 'StateRoot':
        check_run_id(run_id)
        resolved = check_outside_live(path, env)
        if resolved.exists():
            if not resolved.is_dir() or any(resolved.iterdir()):
                raise ShadowWriteError('shadow state root must be a new or empty directory')
        else:
            resolved.mkdir(parents=True)
        root = cls(resolved, run_id)
        marker = root.file(MARKER)
        tmp = marker.with_name(MARKER + '.tmp')
        body = json.dumps({'version': VERSION, 'mode': MODE_SHADOW, 'run_id': run_id}, sort_keys=True)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, body.encode() + b'\n')
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, marker)
        return root

    @classmethod
    def open(cls, path: Path, run_id: str, env: Optional[Mapping[str, str]] = None) -> 'StateRoot':
        check_run_id(run_id)
        resolved = check_outside_live(path, env)
        root = cls(resolved, run_id)
        try:
            doc = json.loads((resolved / MARKER).read_text())
        except (OSError, ValueError):
            raise ShadowWriteError('not a batchctl shadow state root (no marker)') from None
        if not isinstance(doc, dict) or doc.get('mode') != MODE_SHADOW or doc.get('run_id') != run_id:
            raise ShadowWriteError('state root marker is for another run or mode')
        for entry in resolved.iterdir():
            if not root.owns(entry.name):
                raise ShadowWriteError('state root holds a file that the controller does not own')
        return root

    def owns(self, name: str) -> bool:
        if name in STATE_FILES:
            return True
        return any(
            name.endswith(suffix) and name[:-len(suffix)] in STATE_FILES for suffix in TEMP_SUFFIXES
        )

    def file(self, name: str) -> Path:
        """Path of one of the controller's files. Any other name is refused."""
        if name.lower() in FORBIDDEN_NAMES or not self.owns(name):
            raise ShadowWriteError('shadow refuses local file %r' % (name,))
        return self.path / name

    def check(self, path: Path) -> Path:
        """Refuse a path that is not one of the controller's files."""
        path = Path(path)
        if path.parent != self.path:
            raise ShadowWriteError('shadow refuses a write outside its state root')
        return self.file(path.name)


# ----------------------------------------------------------------------------
# Box names and ownership
# ----------------------------------------------------------------------------

def own_prefix(run_id: str) -> str:
    return '%s%s-' % (BOX_PREFIX_SHADOW, check_run_id(run_id))


def box_name(run_id: str, fence: int, try_no: int) -> str:
    """The exact name of the box of one rent try. Never reused."""
    if fence < 1 or try_no < 0:
        raise ValueError('fence and try out of range')
    return '%sf%d-t%d' % (own_prefix(run_id), fence, try_no)


def parse_box_name(run_id: str, name: str) -> Optional[Tuple[int, int]]:
    """Return (fence, try) if this run made the name, else None.

    The match is exact. A name that only starts like ours is not ours, so a
    box of another run or of the live system is never touched.
    """
    prefix = own_prefix(run_id)
    if not isinstance(name, str) or not name.startswith(prefix):
        return None
    match = BOX_SUFFIX.fullmatch(name[len(prefix):])
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
