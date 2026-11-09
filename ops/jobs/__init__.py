"""Shared serverless job runner (SPEC §4).

Four kinds: item (PIT), baseline, bench, prior. Only the validator host is
always-on; every GPU box is rented per job and released after VERIFIED+WIPING.
"""
from .kinds import JOB_KINDS, JobKind, kind_cap, counts_toward_item_cap
from .runner import JobRecord, JobRunner, JobState

__all__ = [
    'JOB_KINDS', 'JobKind', 'kind_cap', 'counts_toward_item_cap',
    'JobRecord', 'JobRunner', 'JobState',
]
