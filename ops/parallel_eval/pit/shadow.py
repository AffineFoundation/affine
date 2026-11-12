"""Shadow-mode write guard: an allowlist.

While ``mode=shadow``, PIT writes two places only:

* R2 objects whose key starts with a prefix of the allowlist (``pit-shadow/``);
* files in the shadow state root (the directory in ``PIT_SHADOW_STATE``).

Each other write is refused. There is no list of file names: a new live file or
a new live R2 prefix is refused without a change here. Live mode
(``may_write_live_artifacts``) lifts the guard for the pe_apply / HEAD-CHAIN
path only after cutover.

This file is the rule. ``shadow_guard.py`` puts the same rule on every file
write and every S3 call of the shadow process.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, FrozenSet, Iterable, Optional, Sequence, Set, Tuple, Union

from .config import PitConfig

STATE_ENV = 'PIT_SHADOW_STATE'
R2_PREFIX = 'pit-shadow/'
SHADOW_R2_PREFIXES: Tuple[str, ...] = (R2_PREFIX,)
# Actions on live artifacts. Shadow has no such action.
LIVE_ACTIONS: FrozenSet[str] = frozenset({
    'write_decision', 'write_decision.json', 'write_head_chain', 'write_head-chain', 'write_weights',
    'publish_r2_head', 'update_r2_head_pointer', 'set_current', 'finalize_head', 'apply_green_live',
})


class ShadowWriteError(RuntimeError):
    """Raised when shadow mode is asked to write outside its allowlist."""


def inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip('/') + '/')


def r2_key_problem(key: Any, prefixes: Sequence[str] = SHADOW_R2_PREFIXES) -> Optional[str]:
    """Why shadow can not write this R2 key. None: the key is under the allowlist."""
    if not isinstance(key, str) or not key:
        return 'the key is empty'
    if '/../' in '/%s/' % key or '/./' in '/%s/' % key:
        return 'the key has a "." or ".." part'
    if not key.startswith(tuple(prefixes)):
        return 'the key is not under the R2 prefix allowlist %s' % list(prefixes)
    return None


def path_problem(path: Any, state_root: Optional[Union[str, Path]]) -> Optional[str]:
    """Why shadow can not write this file. None: the real path is in the shadow state root."""
    if not state_root:
        return 'shadow has no state root (%s is not set)' % STATE_ENV
    real = os.path.realpath(os.path.abspath(os.fsdecode(path)))
    root = os.path.realpath(str(state_root))
    if not inside(real, root):
        return '%s is not in the shadow state root %s' % (real, root)
    return None


class ShadowGuard:
    def __init__(
        self,
        config: PitConfig,
        *,
        r2_prefixes: Sequence[str] = SHADOW_R2_PREFIXES,
        state_root: Optional[Union[str, Path]] = None,
    ):
        self.config = config
        self.r2_prefixes = tuple(r2_prefixes)
        self._state_root = state_root
        self._attempted: Set[str] = set()

    @property
    def active(self) -> bool:
        return not self.config.may_write_live_artifacts()

    @property
    def state_root(self) -> Optional[str]:
        return str(self._state_root) if self._state_root else (os.environ.get(STATE_ENV) or None)

    def assert_shadow(self) -> None:
        if not self.active:
            return
        if self.config.enabled and self.config.mode not in ('shadow', 'live'):
            raise ShadowWriteError('unauthorized mode %r' % self.config.mode)

    def check_r2_key(self, key: Any, *, kind: Optional[str] = None) -> None:
        """Refuse an R2 write whose key is not under the prefix allowlist."""
        if not self.active:
            return
        self.assert_shadow()
        self._attempted.add(kind or str(key))
        problem = r2_key_problem(key, self.r2_prefixes)
        if problem:
            raise ShadowWriteError('shadow refuses the R2 write of %r: %s' % (key, problem))

    def check_path(self, path: Union[str, Path], *, kind: Optional[str] = None) -> None:
        """Refuse a file write that is not in the shadow state root."""
        if not self.active:
            return
        self.assert_shadow()
        label = kind or Path(path).name
        self._attempted.add(label)
        problem = path_problem(path, self.state_root)
        if problem:
            raise ShadowWriteError('shadow refuses write of %s: %s' % (label, problem))

    def check_action(self, action: str) -> None:
        if not self.active:
            return
        self.assert_shadow()
        self._attempted.add(action)
        if action.strip().lower().replace(' ', '_') in LIVE_ACTIONS:
            raise ShadowWriteError('shadow refuses action %s' % action)

    def record_only(self, action: str) -> str:
        self.assert_shadow()
        return 'shadow-no-write:%s' % action

    @property
    def attempted(self) -> FrozenSet[str]:
        return frozenset(self._attempted)


def assert_no_forbidden_writes(
    paths_touched: Iterable[Union[str, Path]],
    state_root: Optional[Union[str, Path]] = None,
) -> None:
    """Each file that a shadow run wrote must be in the shadow state root."""
    root = state_root or os.environ.get(STATE_ENV)
    for path in paths_touched:
        problem = path_problem(path, root)
        if problem:
            raise ShadowWriteError('forbidden shadow write detected: %s' % problem)
