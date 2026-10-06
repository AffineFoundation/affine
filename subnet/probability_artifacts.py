"""Explicit selected-token transport; verification still computes every logit.

The legacy full-vocabulary artifact contract remains the default. This changes
the submitted probability claims, never TOPLOC or the prescribed sampler replay.
"""
VERSION = 'selected-token-logprobs-v1'


def validate_policy(value):
    if type(value) is not dict or value != {'version': VERSION}:
        raise ValueError('exact selected-token probability artifact policy')
    return dict(value)


def for_manifest(manifest):
    if 'probability_artifact_policy' not in manifest:
        return None
    return validate_policy(manifest['probability_artifact_policy'])


def bind_runtime(runtime, manifest):
    selected = for_manifest(manifest)
    if selected is not None and getattr(runtime, 'sampling_context', None) is None:
        raise ValueError('compact probability artifacts require the authenticated forced sampler')
    runtime.probability_artifact_policy = selected
    return runtime


def selected_values(full, tokens):
    import numpy as np
    if (not isinstance(full, np.ndarray) or full.dtype != np.float32 or full.ndim != 2
            or full.shape[0] != len(tokens) or full.shape[1] < 2
            or not np.isfinite(full).all() or not tokens
            or any(type(t) is not int or not 0 <= t < full.shape[1] for t in tokens)):
        raise ValueError('complete finite reference distribution and every selected token required')
    return np.ascontiguousarray(full[np.arange(len(tokens)), tokens][:, None])


def encode(full, tokens, policy):
    if policy is None:
        return full
    validate_policy(policy)
    return selected_values(full, tokens)


def verify_claim(claimed, full, tokens, policy, *, atol):
    import numpy as np
    from .audit_policy import InvalidSample
    if policy is None:
        # Preserve the existing full-vocabulary contract and tolerance exactly.
        if claimed.shape != full.shape or not np.isfinite(claimed).all() or not np.allclose(claimed, full, atol=atol, rtol=0):
            raise InvalidSample('probabilities')
        return
    validate_policy(policy)
    try:
        reference = selected_values(full, tokens)
    except (ValueError, TypeError) as error:
        raise InvalidSample('complete compact probability reference') from error
    if (not isinstance(claimed, np.ndarray) or claimed.dtype != np.float32
            or claimed.shape != reference.shape or not np.isfinite(claimed).all()
            or not np.allclose(claimed, reference, atol=atol, rtol=0)):
        raise InvalidSample('selected-token probabilities')
