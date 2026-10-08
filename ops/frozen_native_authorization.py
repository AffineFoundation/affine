"""Reuse completed native journals across an explicitly approved path move.

Only execution_root may differ. Source, grader, tokenizer, dataset, limits and
assurance remain identical. Never rewrite a context or generate replacement
grades under a historical authorization.
"""
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def install(module, history, authority):
    from subnet.distributed_roles import authenticate
    if type(history)is not list or not 1<=len(history)<=64:
        raise ValueError('bounded explicitly approved native authorization history')
    approved={}
    for envelope in history:
        policy=authenticate(envelope,authority)
        if type(policy.get('execution_root'))is not str or not Path(policy['execution_root']).is_absolute():
            raise ValueError('explicit historical native execution directory')
        identity=digest(envelope)
        if identity in approved:raise ValueError('duplicate native authorization history')
        approved[identity]=(envelope,policy)
    original=module.NativeEligibilitySelector.select
    def select(instance,manifest,submissions):
        epoch=manifest['epoch']
        # Original selector owns path/schema/manifest/input checks. Do not open
        # a journal path on its behalf until the epoch component is bounded.
        import re
        if type(epoch)is not str or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{1,220}',epoch)is None:
            raise ValueError('exact frozen native epoch namespace')
        root=Path(instance.controller.state)/'native-outcome-eligibility'/epoch
        context_path=root/'context.ROOT-SIGNED.json'
        if not context_path.exists():return original(instance,manifest,submissions)
        envelope=json.loads(module._load(context_path));context=authenticate(envelope,authority)
        identity=context['authorization_sha256']
        if identity==digest(instance.authorization):return original(instance,manifest,submissions)
        if identity not in approved:raise ValueError('frozen native authorization not explicitly approved')
        old_envelope,old_policy=approved[identity]
        current_policy=authenticate(instance.authorization,authority)
        if instance.policy!=current_policy:raise ValueError('current native policy drift')
        differences={k for k in set(old_policy)|set(current_policy)if old_policy.get(k)!=current_policy.get(k)}
        if differences!={'execution_root'}:raise ValueError('frozen native continuation changed grading semantics')
        if root.is_symlink()or root.stat().st_uid!=os.getuid():raise ValueError('owned frozen native journal directory')
        for name in ('grades','subset'):
            path=root/(name+'.ROOT-SIGNED.json')
            if not path.is_file()or path.is_symlink()or path.stat().st_uid!=os.getuid():
                raise ValueError('completed original native receipts required')
            authenticate(json.loads(module._load(path)),authority)
        previous_authorization,previous_policy=instance.authorization,instance.policy
        try:
            instance.authorization,instance.policy=old_envelope,old_policy
            # Original select reauthenticates complete original membership,
            # parent/checkpoint, manifest, grades and subset. No journal changes.
            return original(instance,manifest,submissions)
        finally:
            instance.authorization,instance.policy=previous_authorization,previous_policy
    module.NativeEligibilitySelector.select=select

