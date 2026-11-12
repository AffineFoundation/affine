"""Names of probed tasks outside the validator host.

Which tasks a round probes is private to the validator host. A box gets the
task ids of its batch and a key for that batch, over ssh, in a directory that
the job removes. The result that the box uploads names every task by
``mask(key, env, id)``, and holds no id, no seed and no run name
(``box/steps.py``). The judge turns the names back into ids with the key,
which it makes again from the same host-private text.

A name says nothing about its id without the key, and the key of one batch
says nothing about the names of another batch.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any, Dict, Mapping, Sequence

MASK_HEX = 32
KEY_DOMAIN = 'affine-batchctl-mask-v1'


class MaskError(ValueError):
    """A result names a task that the round does not hold."""


def mask_key(private: str, run_id: str, batch_id: str) -> str:
    """The key of one batch. ``private`` is text that only the validator host
    holds (the plan seed of the round)."""
    if not isinstance(private, str) or not private:
        raise MaskError('no private text for the key')
    return hmac.new(private.encode(), ('%s|%s|%s' % (KEY_DOMAIN, run_id, batch_id)).encode(),
                    hashlib.sha256).hexdigest()


def mask(key: str, env_id: str, index: Any) -> str:
    return hmac.new(key.encode(), ('%s:%s' % (env_id, index)).encode(), hashlib.sha256).hexdigest()[:MASK_HEX]


def unmask(public: Mapping[str, Any], key: str, indices: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
    """The probe result of a box with task ids again.

    ``indices`` maps an environment to the ids that the round probes there. A
    name that belongs to none of them, and an environment that the round does
    not hold, raise ``MaskError``: such a result is not a result of this round.
    """
    envs = public.get('envs') if isinstance(public, Mapping) else None
    if not isinstance(envs, Mapping) or not envs:
        raise MaskError('the result has no environments')
    out = dict(public)
    out['envs'] = {}
    for env_id, env in envs.items():
        if env_id not in indices or not isinstance(env, Mapping) or not isinstance(env.get('per_task'), Mapping):
            raise MaskError('environment %r is not an environment of the round' % (env_id,))
        names = {mask(key, env_id, index): str(index) for index in indices[env_id]}
        cells = {}
        for name, cell in env['per_task'].items():
            if name not in names:
                raise MaskError('a task of the result is no task of the round')
            cells[names[name]] = cell
        out['envs'][env_id] = dict(env, per_task=cells)
    return out
