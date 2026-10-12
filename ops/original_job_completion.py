"""Narrow completion admission for one already-running, authenticated job.

This module never signs a scope, launches a job, rewrites a deadline, or promotes
an optimizer. Its caller must authenticate the operator scope and timely-start
evidence first. The ordinary report validator still validates the full report.
"""
import copy
import hashlib
import json
import math
import types


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def finite(*values):
    return all(type(value) in (int, float) and math.isfinite(value)
               for value in values)


def exact_job_completion_allowed(scope, *, authority, job, report, prior,
                                 manifest, original_job_file_sha256):
    """Extra time predicate; ``scope`` must already be ROOT-authenticated.

    Job identity includes its entire immutable payload, original dispatch
    journal, manifest, authority and original file hash. Bounds are half-open.
    Nothing in this predicate establishes that the rest of a report is valid.
    """
    created, expires = job.get("created_at"), job.get("expires_at")
    started, before = scope["actual_started_at"], scope["completion_before"]
    completed = report.get("completed_at")
    if not finite(created, expires, started, before, completed):
        return False
    if not (created <= started < expires and started <= completed < before
            and 0 < before - started <= 3600):
        return False
    if (authority != scope["authority"] or job.get("role") != "train"
            or job.get("job_id") != scope["job_id"]
            or report.get("job_id") != scope["job_id"]):
        return False
    if (created != scope["original_created_at"]
            or expires != scope["original_expires_at"]
            or original_job_file_sha256 != scope["original_job_file"]["sha256"]):
        return False
    return (prior == scope["original_prior"]
            and digest(job) == scope["original_job_payload_sha256"]
            and digest(manifest) == scope["manifest_sha256"])


def extend_checked_predicate(checked, *, expected_code, additional_predicate):
    """Retain the exact checked body and its existing time predicate.

    The operator authenticates ``expected_code`` from its pinned original
    source. Only this function's private globals copy gains the extra predicate;
    the original function and its module globals remain unchanged.
    """
    if not isinstance(checked, types.FunctionType) or checked.__code__ != expected_code:
        raise ValueError("exact original report-checking code required")
    previous = checked.__globals__.get("_completion_time_allowed")
    if not callable(previous) or not callable(additional_predicate):
        raise ValueError("both original and additional predicates required")

    def allowed(*args, **kwargs):
        return previous(*args, **kwargs) or additional_predicate(*args, **kwargs)

    namespace = dict(checked.__globals__, _completion_time_allowed=allowed)
    result = types.FunctionType(checked.__code__, namespace, checked.__name__,
                                checked.__defaults__, checked.__closure__)
    result.__kwdefaults__ = copy.copy(checked.__kwdefaults__)
    result.__dict__.update(checked.__dict__)
    return result
