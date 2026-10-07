"""Shared token-trace digest; inputs must already pass their caller's admission.

This is the existing learner duplicate key, not proof of execution, task binding,
classification or optimizer application. Metadata and labels do not enter it.
"""
import hashlib
from .storage import canonical


def token_trace_sha256(turns):
    """Preserve the existing ordered prompt/output projection and digest bytes."""
    trace = [dict(prompt=turn['prompt'], output=turn['output']) for turn in turns]
    return hashlib.sha256(canonical(trace)).hexdigest()
