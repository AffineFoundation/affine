"""Hard denies of the two modes, shadow and live.

A run has one mode. ``shadow`` is the default. ``live`` is refused unless the
caller gives its explicit flag (``check_mode``). The mode selects three
things: the R2 root, the box name prefix and the marker of the state root.

Each deny below is a hard error, not a setting. Each one holds in both modes:

* R2: every key that the controller reads, writes, copies or deletes must be
  under the prefix of its own run and mode: ``pit-shadow/batchctl/<run_id>/``
  for shadow, ``pit/batchctl/<run_id>/`` for live. A shadow run cannot reach
  a live key and a live run cannot reach a shadow key. Every other key of the
  bucket is out of reach of both.
* Names: no key and no local file has the name of a live artifact.
* Local files: the controller writes four named files in one marked state
  root. The marker names the run and the mode. The root must be outside every
  live root.
* Boxes: the controller destroys a box only if its name is exactly a name
  that this run made in its mode. A shared name prefix is not enough.
* Mode: a mode other than ``shadow`` and ``live`` is refused, and so is
  ``live`` without its flag.

What only a live run may do: nothing in this module. The side effects of a
live run are outside this module.

The package has no code that talks to the chain, the website or Discord. A
test scans the imports of every module to keep it that way.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import FrozenSet, List, Mapping, Optional, Tuple

from . import BOX_PREFIXES, MODE_LIVE, MODE_SHADOW, MODES, R2_ROOTS, VERSION
from .interfaces import Blob, Digest, ObjectInfo, ObjectStore
from ..fakepass_sandbox import FakepassError, assert_paths_outside_live, live_roots_from_env
from ..shadow import ShadowWriteError

# Used with ``fullmatch`` only: ``$`` would also accept a trailing newline.
RUN_ID = re.compile(r'[a-z0-9]{4,32}')
BOX_SUFFIX = re.compile(r'f([1-9][0-9]{0,11})-t(0|[1-9][0-9]{0,8})')

# Names of live artifacts. No key and no local file may end with one of them,
# even inside the prefix or the state root of a run, in both modes. The list
# is kept here: the shadow guard of the PIT package has no list of names.
FORBIDDEN_NAMES = frozenset(
    name.lower() for name in (
        'DECISION.json', 'HEAD-CHAIN.json', 'CURRENT.json', 'FINALIZED.json', 'WEIGHTS.json', 'weights.pt',
        'R2_HEAD_POINTER', 'r2-head-pointer',
        'mining.json', 'llms.txt', 'llms-full.txt', 'ledger.json', 'gate-ledger.json',
        'weights.json', 'extrinsics.json', 'index.html',
    )
)

MARKER = 'BATCHCTL-SHADOW-ROOT.json'
MARKERS = {MODE_SHADOW: MARKER, MODE_LIVE: 'BATCHCTL-LIVE-ROOT.json'}
JOURNAL = 'journal.jsonl'
LOCK = 'LOCK'
STATUS = 'status.json'
# The files of a shadow root. A live root has its own marker (``StateRoot.state_files``).
STATE_FILES = frozenset({MARKER, JOURNAL, LOCK, STATUS})
# Temporary names that the journal and the status writer rename from.
TEMP_SUFFIXES = ('.create.tmp', '.repair.tmp', '.tmp')


class LiveModeRefused(ShadowWriteError):
    """The mode is unknown, or it is live without its explicit flag."""


def _known_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in MODES:
        raise LiveModeRefused('batchctl runs in shadow or live mode only (got %r)' % (mode,))
    return mode


def check_mode(mode: str, live_armed: bool = False) -> None:
    """Refuse an unknown mode, and refuse live mode without its flag.

    Only ``True`` itself is the flag. A value that is merely true is not.
    """
    if _known_mode(mode) == MODE_LIVE and live_armed is not True:
        raise LiveModeRefused('live mode needs its explicit flag')


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
        raise ShadowWriteError('run id must match %s' % RUN_ID.pattern)
    return run_id


def run_prefix(run_id: str, mode: str = MODE_SHADOW) -> str:
    """The only R2 prefix that a run of this mode may touch."""
    return '%s%s/' % (R2_ROOTS[_known_mode(mode)], check_run_id(run_id))


def shadow_prefix(run_id: str) -> str:
    """The only R2 prefix that a shadow run may touch."""
    return run_prefix(run_id, MODE_SHADOW)


def check_key(prefix: str, key: str) -> str:
    """Refuse every key that is not a plain key under the prefix of one run."""
    if not prefix.startswith(tuple(R2_ROOTS.values())) or not prefix.endswith('/'):
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
    """An ``ObjectStore`` that can reach the prefix of one run in one mode only.

    Reads are fenced as well as writes: the controller has no reason to read
    a key of the live system or of the other mode, and a read of the wrong
    prefix would hide a wiring mistake.
    """

    def __init__(self, inner: ObjectStore, run_id: str, mode: str = MODE_SHADOW):
        self._inner = inner
        self.mode = mode
        self.prefix = run_prefix(run_id, mode)

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

    ``create`` makes the directory and the marker of its mode. ``open`` needs
    the marker of the same run and the same mode. Both refuse a directory that
    is in or around a live root, or that holds any file the controller does
    not own. The marker of the other mode is such a file.
    """

    def __init__(self, path: Path, run_id: str, mode: str = MODE_SHADOW):
        self.path = path
        self.run_id = run_id
        self.mode = _known_mode(mode)

    @property
    def marker(self) -> str:
        """The name of the marker file of this mode."""
        return MARKERS[_known_mode(self.mode)]

    @property
    def state_files(self) -> FrozenSet[str]:
        """The names of the four files of a root of this mode."""
        return frozenset({self.marker, JOURNAL, LOCK, STATUS})

    @classmethod
    def create(
        cls, path: Path, run_id: str, env: Optional[Mapping[str, str]] = None, mode: str = MODE_SHADOW,
    ) -> 'StateRoot':
        check_run_id(run_id)
        root = cls(check_outside_live(path, env), run_id, mode)
        resolved = root.path
        if resolved.exists():
            if not resolved.is_dir() or any(resolved.iterdir()):
                raise ShadowWriteError('shadow state root must be a new or empty directory')
        else:
            resolved.mkdir(parents=True)
        marker = root.file(root.marker)
        tmp = marker.with_name(root.marker + '.tmp')
        body = json.dumps({'version': VERSION, 'mode': root.mode, 'run_id': run_id}, sort_keys=True)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, body.encode() + b'\n')
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, marker)
        return root

    @classmethod
    def open(
        cls, path: Path, run_id: str, env: Optional[Mapping[str, str]] = None, mode: str = MODE_SHADOW,
    ) -> 'StateRoot':
        check_run_id(run_id)
        resolved = check_outside_live(path, env)
        root = cls(resolved, run_id, mode)
        try:
            doc = json.loads((resolved / root.marker).read_text())
        except (OSError, ValueError):
            raise ShadowWriteError('not a batchctl %s state root (no marker)' % root.mode) from None
        if not isinstance(doc, dict) or doc.get('mode') != root.mode or doc.get('run_id') != run_id:
            raise ShadowWriteError('state root marker is for another run or mode')
        for entry in resolved.iterdir():
            if not root.owns(entry.name):
                raise ShadowWriteError('state root holds a file that the controller does not own')
        return root

    def owns(self, name: str) -> bool:
        """True for the four files of this mode and their temporary names.

        The marker of the other mode is not one of them.
        """
        files = self.state_files
        if name in files:
            return True
        return any(
            name.endswith(suffix) and name[:-len(suffix)] in files for suffix in TEMP_SUFFIXES
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

def own_prefix(run_id: str, mode: str = MODE_SHADOW) -> str:
    return '%s%s-' % (BOX_PREFIXES[_known_mode(mode)], check_run_id(run_id))


def box_name(run_id: str, fence: int, try_no: int, mode: str = MODE_SHADOW) -> str:
    """The exact name of the box of one rent try. Never reused."""
    if fence < 1 or try_no < 0:
        raise ValueError('fence and try out of range')
    return '%sf%d-t%d' % (own_prefix(run_id, mode), fence, try_no)


def parse_box_name(run_id: str, name: str, mode: str = MODE_SHADOW) -> Optional[Tuple[int, int]]:
    """Return (fence, try) if this run made the name in this mode, else None.

    The match is exact. A name that only starts like ours is not ours, so a
    box of another run, of the other mode or of the live system is never
    touched.
    """
    prefix = own_prefix(run_id, mode)
    if not isinstance(name, str) or not name.startswith(prefix):
        return None
    match = BOX_SUFFIX.fullmatch(name[len(prefix):])
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
