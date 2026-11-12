"""The one place where the shadow controller takes code of the validator host.

The host helpers are not part of this package and cannot be loaded off the
validator host, so they are loaded by name when the shadow process starts.
Only the names in ``HOST_HELPERS`` can be loaded. Each one is a function that
this package calls through one of its interfaces; none of them writes live
state:

``lium``   the command-line helper of the ops code. It holds the provider
           key. ``lium.LiumProvider`` gets it as its ``cli`` and sends only
           the commands of that module, for boxes with a shadow name.

A helper that cannot be loaded stops the start: there is no second way.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Tuple

# name -> (module, attribute). A test holds the same table.
HOST_HELPERS: Dict[str, Tuple[str, str]] = {
    'lium': ('ops.parallel_eval.live.pe_common', 'lium'),
}


class HostLinkError(RuntimeError):
    """A helper of the validator host is not there. The process must not start."""


def load(name: str) -> Callable[..., Any]:
    if name not in HOST_HELPERS:
        raise HostLinkError('%r is not a host helper of the shadow controller' % (name,))
    module, attribute = HOST_HELPERS[name]
    try:
        found = getattr(importlib.import_module(module), attribute)
    except Exception as exc:
        raise HostLinkError('host helper %s cannot be loaded: %s' % (name, type(exc).__name__)) from None
    if not callable(found):
        raise HostLinkError('host helper %s is not a function' % name)
    return found
