"""The one place where the controller takes code of the validator host.

The host helpers are not part of this package and cannot be loaded off the
validator host, so they are loaded by name when the process starts. Only the
names in ``HOST_HELPERS`` and ``HOST_VALUES`` can be loaded. Each helper is a
function that this package calls through one of its interfaces:

``lium``      the command-line helper of the ops code. It holds the provider
              key. ``lium.LiumProvider`` gets it as its ``cli`` and sends
              only the commands of that module, for boxes of this run.
``pod_of``    the ssh handle of the live lane for one rented box.
``boot_helpers``, ``push_probe_bundle``, ``bundle_trainer``, ``ship_inputs``,
``ship_train_pit``, ``train_cmd``, ``put``, ``put_tar``
              the box helpers of the live lane (``pe_box``), unchanged: what
              the serial lane does to a box before it trains, and the train
              command itself. ``agent.py`` calls them.
``public_harness``
              the public harness of the live lane: it names the sampling of a
              probe.
``score``, ``blocks_baseline``
              the decision function of the live gate (``pe_score``) and its
              baseline builder. ``judge.py`` calls them. They read the round
              files of the live gate and write nothing.

``HOST_VALUES`` holds what is not a function: ``heldout``, the directory
where the live scorer reads its rounds.

None of the helpers writes live state. A helper that cannot be loaded stops
the start: there is no second way.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Tuple

PE_COMMON = 'ops.parallel_eval.live.pe_common'
PE_BOX = 'ops.parallel_eval.live.pe_box'
PE_SCORE = 'ops.parallel_eval.live.pe_score'

# name -> (module, attribute). A test holds the same table.
HOST_HELPERS: Dict[str, Tuple[str, str]] = {
    'lium': (PE_COMMON, 'lium'),
    'pod_of': (PE_COMMON, 'pod_of'),
    'boot_helpers': (PE_BOX, 'boot_helpers'),
    'push_probe_bundle': (PE_BOX, 'push_probe_bundle'),
    'bundle_trainer': (PE_BOX, 'bundle_trainer'),
    'ship_inputs': (PE_BOX, 'ship_inputs'),
    'ship_train_pit': (PE_BOX, '_ship_train_pit'),
    'train_cmd': (PE_BOX, '_train_pit_inner_cmd'),
    'put': (PE_BOX, 'put'),
    'put_tar': (PE_BOX, 'put_tar'),
    'public_harness': ('ops.parallel_eval.pit.box', '_load_public_harness'),
    'score': (PE_SCORE, 'score'),
    'blocks_baseline': (PE_SCORE, 'blocks_baseline'),
}
HOST_VALUES: Dict[str, Tuple[str, str]] = {
    'heldout': (PE_SCORE, 'HELDOUT'),
}


class HostLinkError(RuntimeError):
    """A helper of the validator host is not there. The process must not start."""


def _find(table: Dict[str, Tuple[str, str]], name: str) -> Any:
    if name not in table:
        raise HostLinkError('%r is not a host helper of the controller' % (name,))
    module, attribute = table[name]
    try:
        return getattr(importlib.import_module(module), attribute)
    except Exception as exc:
        raise HostLinkError('host helper %s cannot be loaded: %s' % (name, type(exc).__name__)) from None


def load(name: str) -> Callable[..., Any]:
    found = _find(HOST_HELPERS, name)
    if not callable(found):
        raise HostLinkError('host helper %s is not a function' % name)
    return found


def value(name: str) -> Any:
    return _find(HOST_VALUES, name)
