"""Explicit optimizer-only CPU threading; model execution keeps its pinned profile.

This mechanical policy requires a separately qualified signed SFT release.
It changes neither Adam equations, gradient accumulation, nor sampling runtime.
"""
import copy

POLICY = dict(version='optimizer-only-cpu-threads-v1', steady_intraop=2,
              optimizer_intraop=16, interop=2)


def configuration(value):
    if (type(value) is not dict or set(value) != set(POLICY) or
            any(type(value[k]) is not type(v) or value[k] != v for k, v in POLICY.items())):
        raise ValueError('exact qualified optimizer-only CPU threading policy')
    return copy.deepcopy(value)


def snapshot(torch):
    return dict(intraop=torch.get_num_threads(), interop=torch.get_num_interop_threads())


def expected_evidence(policy):
    p = configuration(policy)
    return dict(policy=p, before=dict(intraop=p['steady_intraop'], interop=p['interop']),
        during=dict(intraop=p['optimizer_intraop'], interop=p['interop']),
        after=dict(intraop=p['steady_intraop'], interop=p['interop']))


def step(torch, optimizer, gradients, policy):
    """Restore the admitted runtime even if the optimizer raises mid-update.

A failure is propagated; the caller must not export a partially updated state.
This is an exclusive worker, not a process with concurrent model forwards.
    """
    p = configuration(policy); expected = expected_evidence(p)
    before = snapshot(torch)
    if before != expected['before']:
        raise ValueError('optimizer entry must use the admitted steady thread profile')
    try:
        torch.set_num_threads(p['optimizer_intraop'])
        during = snapshot(torch)
        if during != expected['during']:
            raise ValueError('optimizer thread profile did not take effect')
        result = optimizer.step(gradients=gradients)
    finally:
        torch.set_num_threads(p['steady_intraop'])
    after = snapshot(torch)
    if after != expected['after']:
        raise ValueError('optimizer failed to restore steady runtime')
    return result, dict(policy=p, before=before, during=during, after=after)
