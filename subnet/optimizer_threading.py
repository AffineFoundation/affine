"""Candidate trainer-only threading; current production admission stays closed.

The caller reads this opt-in only from an authenticated execution declaration.
The current production release rejects that added field and this module until
a separately reviewed prospective trainer release admits the source and policy.
No SFT authorization, environment variable, or miner input grants permission.
"""
import copy

POLICY = dict(version='optimizer-only-cpu-threads-v1', steady_intraop=2,
              optimizer_intraop=16, interop=2)


def configuration(value):
    if (type(value) is not dict or set(value) != set(POLICY) or
            any(type(value[k]) is not type(v) or value[k] != v for k, v in POLICY.items())):
        raise ValueError('exact qualified optimizer-only CPU threading policy')
    return copy.deepcopy(value)


def declared_policy(declaration):
    """Signature and complete declaration admission belong to the caller."""
    if type(declaration) is not dict:
        raise ValueError('authenticated execution declaration required')
    if 'optimizer_threading' not in declaration:
        return None
    return configuration(declaration['optimizer_threading'])


def snapshot(torch):
    return dict(intraop=torch.get_num_threads(), interop=torch.get_num_interop_threads())


def expected_evidence(policy):
    p = configuration(policy)
    return dict(policy=p, before=dict(intraop=p['steady_intraop'], interop=p['interop']),
        during=dict(intraop=p['optimizer_intraop'], interop=p['interop']),
        after=dict(intraop=p['steady_intraop'], interop=p['interop']))


def require_steady(torch, policy):
    before = snapshot(torch)
    if before != expected_evidence(policy)['before']:
        raise ValueError('optimizer entry must use the admitted steady thread profile')
    return before


def step(torch, optimizer, gradients, policy):
    """One exclusive worker; propagate failure and restore entry intra-op threads.

An exception can leave Adam partly updated: do not publish or retry that state.
This wrapper neither changes inter-op threads nor rolls back optimizer tensors.
    """
    p = configuration(policy); expected = expected_evidence(p)
    before = require_steady(torch, p)
    try:
        torch.set_num_threads(p['optimizer_intraop'])
        during = snapshot(torch)
        if during != expected['during']:
            raise ValueError('optimizer thread profile did not take effect')
        result = optimizer.step(gradients=gradients)
    finally:
        torch.set_num_threads(before['intraop'])
    after = snapshot(torch)
    if after != expected['after']:
        raise ValueError('optimizer failed to restore steady runtime')
    return result, dict(policy=p, before=before, during=during, after=after)


def validate_report(report, policy, steps):
    """Called after independent signature/source/job provenance authentication."""
    updates=report.get('training',{}).get('updates',[])
    if policy is None:
        if any(type(row) is dict and 'optimizer_threading' in row for row in updates):
            raise ValueError('historical report cannot assert unrequested optimizer threading')
        return
    expected=expected_evidence(policy)
    if type(steps) is not int or steps<1 or type(updates) is not list or len(updates)!=steps:
        raise ValueError('exact per-update optimizer threading report count')
    for row in updates:
        if type(row) is not dict or row.get('optimizer_threading')!=expected:
            raise ValueError('every update needs exact before/during/restored thread evidence')
        actual=row['optimizer_threading']
        if any(type(actual[stage][key]) is not int for stage in ('before','during','after')
               for key in ('intraop','interop')):
            raise ValueError('integer thread evidence, not Boolean or floating aliases')
        configuration(actual['policy'])
