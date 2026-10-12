"""Prospective v3 native selector history; original v2 never enters this path.

This helper reuses the pinned bounded closed-history provider. It freezes either
ROOT-attested report attribution or an explicit original-order fallback before
any native outcome is seen. Reentry authenticates the original snapshot without
reading newer reports. Its signed envelope is embedded in the archived draw.
"""
import copy
import hashlib
import json
import re
import time
from pathlib import Path
from subnet import training_task_representatives as rep
from . import task_cooling_provider as exposure
from .native_training_eligibility import _create, _load


def freeze(selector, manifest, root, *, now):
    authority = selector.authority
    path = Path(root)/'priority-history.ROOT-SIGNED.json'
    if path.exists():
        envelope = json.loads(_load(path))
        rep.validate_history_snapshot(envelope, manifest, authority, at=now)
        return envelope
    binding = rep.history_binding(manifest)
    deadline = time.monotonic()+exposure.MAX_SECONDS
    try:
        data = exposure.evidence_provider({'state':str(selector.controller.state)}, binding['round'], binding['epoch'], authority=authority)
        if (set(data) != {'version','assurance','records'} or data['version'] != 'finalized-trained-task-exposure-input-v1'
                or data['assurance'] != 'signed-closure-and-job-linked-unsigned-trainer-report'):
            raise exposure.ExposureUnavailable('unexpected_history_provider_contract')
        rows = copy.deepcopy(data['records'])
        state = Path(selector.controller.state)
        for row in rows:
            ref = row['provenance']['closure']
            raw, actual = exposure._read(Path(ref['path']),state,64*1024,deadline)
            if actual != ref: raise exposure.ExposureUnavailable('closed_history_changed_before_freeze')
            row['signed_closure'] = json.loads(raw)
        payload = dict(binding, created_at=now, outcome='priority',reason_code=None,records=rows)
        envelope = selector.controller.signed(payload)
        rep.validate_history_snapshot(envelope,manifest,authority,at=now)
    except (exposure.ExposureUnavailable, OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        reason = getattr(error,'reason_code','history_binding_validation')
        if not isinstance(reason,str) or re.fullmatch('[a-z0-9_]{1,96}',reason) is None:
            reason = 'history_binding_validation'
        envelope = selector.controller.signed(dict(binding,created_at=now,
            outcome='original_order_fallback',reason_code=reason,records=[]))
        rep.validate_history_snapshot(envelope,manifest,authority,at=now)
    _create(path,envelope)
    original = json.loads(_load(path))
    rep.validate_history_snapshot(original,manifest,authority,at=now)
    return original
