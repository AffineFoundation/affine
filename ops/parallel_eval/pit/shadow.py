"""Shadow-mode write guard.

While ``mode=shadow``, PIT must never write DECISION.json, weights, HEAD-CHAIN,
or R2 head pointers. Live mode (``may_write_live_artifacts``) lifts the guard
for the pe_apply / HEAD-CHAIN path only after cutover.
"""
from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, Iterable, Optional, Set, Union

from .config import PitConfig


FORBIDDEN_SHADOW_WRITES: FrozenSet[str] = frozenset({
    'DECISION.json',
    'HEAD-CHAIN.json',
    'CURRENT.json',
    'FINALIZED.json',
    'WEIGHTS.json',
    'weights.pt',
    'R2_HEAD_POINTER',
    'r2-head-pointer',
})

FORBIDDEN_BASENAMES: FrozenSet[str] = frozenset({
    'DECISION.json',
    'HEAD-CHAIN.json',
    'CURRENT.json',
    'FINALIZED.json',
    'WEIGHTS.json',
})


class ShadowWriteError(RuntimeError):
    """Raised when shadow mode is asked to touch a live artifact."""


class ShadowGuard:
    def __init__(self, config: PitConfig):
        self.config = config
        self._attempted: Set[str] = set()

    @property
    def active(self) -> bool:
        return not self.config.may_write_live_artifacts()

    def assert_shadow(self) -> None:
        if not self.active:
            return
        if self.config.enabled and self.config.mode not in ('shadow', 'live'):
            raise ShadowWriteError('unauthorized mode %r' % self.config.mode)

    def check_path(self, path: Union[str, Path], *, kind: Optional[str] = None) -> None:
        if not self.active:
            return
        self.assert_shadow()
        p = Path(path)
        name = p.name
        label = kind or name
        self._attempted.add(label)
        if name in FORBIDDEN_BASENAMES or label in FORBIDDEN_SHADOW_WRITES:
            raise ShadowWriteError('shadow refuses write of %s' % label)
        lowered = str(p).lower()
        if 'r2-head' in lowered or 'r2_head' in lowered or '/heads/current' in lowered:
            raise ShadowWriteError('shadow refuses R2 head pointer path %s' % p)

    def check_action(self, action: str) -> None:
        if not self.active:
            return
        self.assert_shadow()
        self._attempted.add(action)
        normalized = action.strip().lower().replace(' ', '_')
        forbidden_actions = {
            'write_decision',
            'write_decision.json',
            'write_head_chain',
            'write_head-chain',
            'write_weights',
            'publish_r2_head',
            'update_r2_head_pointer',
            'set_current',
            'finalize_head',
            'apply_green_live',
        }
        if normalized in forbidden_actions or normalized in {f.lower() for f in FORBIDDEN_SHADOW_WRITES}:
            raise ShadowWriteError('shadow refuses action %s' % action)

    def record_only(self, action: str) -> str:
        self.assert_shadow()
        return 'shadow-no-write:%s' % action

    @property
    def attempted(self) -> FrozenSet[str]:
        return frozenset(self._attempted)


def assert_no_forbidden_writes(paths_touched: Iterable[Union[str, Path]]) -> None:
    for path in paths_touched:
        name = Path(path).name
        if name in FORBIDDEN_BASENAMES:
            raise ShadowWriteError('forbidden shadow write detected: %s' % name)
