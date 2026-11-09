"""Host adapter seams for pe_rent / pe_lane / pe_score / pe_live.

Those helpers are host-only today (validator machine paths). PIT talks to them
through narrow callables so unit tests can inject fakes without importing host
modules. Missing adapters stay inert stubs that raise ``AdapterUnavailable``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional


class AdapterUnavailable(RuntimeError):
    """Raised when a host PE helper is not injected and not importable."""


@dataclass
class HostAdapters:
    """Bundle of host-side PE helpers.

    Each field is a callable. Defaults refuse use so shadow unit tests never
    accidentally reach production rental / scoring paths.
    """

    rent: Optional[Callable[..., Any]] = None
    release: Optional[Callable[..., Any]] = None
    lane: Optional[Callable[..., Any]] = None
    score: Optional[Callable[..., Any]] = None
    live_tick: Optional[Callable[..., Any]] = None
    baseline_draw: Optional[Callable[..., Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def require(self, name: str) -> Callable[..., Any]:
        fn = getattr(self, name, None)
        if not callable(fn):
            raise AdapterUnavailable('host adapter %r unavailable' % name)
        return fn


def _try_import(module_path: Optional[str]) -> Any:
    if not module_path:
        return None
    return importlib.import_module(module_path)


def load_host_adapters(
    *,
    pe_rent_module: Optional[str] = None,
    pe_lane_module: Optional[str] = None,
    pe_score_module: Optional[str] = None,
    pe_live_module: Optional[str] = None,
    overrides: Optional[Mapping[str, Callable[..., Any]]] = None,
) -> HostAdapters:
    """Build adapters from optional module paths + explicit overrides.

    Overrides win. Module paths are best-effort; ImportError leaves that slot
    empty (caller hits AdapterUnavailable on use).
    """
    adapters = HostAdapters()
    rent_mod = None
    lane_mod = None
    score_mod = None
    live_mod = None
    try:
        rent_mod = _try_import(pe_rent_module)
    except ImportError:
        rent_mod = None
    try:
        lane_mod = _try_import(pe_lane_module)
    except ImportError:
        lane_mod = None
    try:
        score_mod = _try_import(pe_score_module)
    except ImportError:
        score_mod = None
    try:
        live_mod = _try_import(pe_live_module)
    except ImportError:
        live_mod = None

    if rent_mod is not None:
        adapters.rent = getattr(rent_mod, 'rent', None) or getattr(rent_mod, 'rent_box', None)
        adapters.release = getattr(rent_mod, 'release', None) or getattr(rent_mod, 'release_box', None)
    if lane_mod is not None:
        adapters.lane = getattr(lane_mod, 'lane', None) or getattr(lane_mod, 'assign_lane', None)
    if score_mod is not None:
        adapters.score = getattr(score_mod, 'score', None) or getattr(score_mod, 'score_probe', None)
    if live_mod is not None:
        adapters.live_tick = getattr(live_mod, 'tick', None) or getattr(live_mod, 'dispatch_tick', None)
        adapters.baseline_draw = getattr(live_mod, 'baseline_draw', None)

    if overrides:
        for key, fn in overrides.items():
            if not hasattr(adapters, key):
                raise ValueError('unknown adapter override %r' % key)
            setattr(adapters, key, fn)
    return adapters


def stub_adapters(**behavior: Callable[..., Any]) -> HostAdapters:
    """Test helper: adapters that return scripted results."""
    return load_host_adapters(overrides=behavior)
