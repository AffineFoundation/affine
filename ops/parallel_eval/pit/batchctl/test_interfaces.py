"""The job that a box gets is complete and fail-closed."""
from __future__ import annotations

import dataclasses

import pytest

from . import interfaces as I

H1, H2 = ('%064x' % n for n in (1, 2))


def spec(**over):
    fields = dict(
        run_id='run0001', batch_id='b1', batch_ref='ref', fence=7, attempt=1,
        base_state=H1, base_seq=0, base_prefix='shadow/batchctl/run0001/states/%s/' % H1,
        expected_fp32_master_digest=H2,
        staging_prefix='shadow/batchctl/run0001/staging/b1/000000000007/',
        heartbeat_every_s=60.0, job_ttl_s=3600.0,
    )
    fields.update(over)
    return I.JobSpec(**fields)


def test_job_spec_carries_everything_a_box_needs():
    job = spec()
    assert job.doc() == dataclasses.asdict(job)
    assert set(job.doc()) == {
        'run_id', 'batch_id', 'batch_ref', 'fence', 'attempt', 'base_state', 'base_seq',
        'base_prefix', 'expected_fp32_master_digest', 'staging_prefix', 'heartbeat_every_s',
        'job_ttl_s',
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        job.fence = 8


@pytest.mark.parametrize('digest', [None, '', 'bootstrap', 'skip', '0' * 63, '0' * 65, 'G' * 64, 'AB' * 32, H2 + '\n'])
def test_job_spec_without_a_real_fp32_master_digest_cannot_be_built(digest):
    # Fail closed: there is no value that means "do not check the master".
    with pytest.raises(ValueError):
        spec(expected_fp32_master_digest=digest)
    assert not any('override' in name or 'bootstrap' in name for name in spec().doc())


@pytest.mark.parametrize('over', [
    {'base_state': 'x'},
    {'fence': 0},
    {'attempt': 0},
    {'base_seq': -1},
    {'fence': 8},
    {'staging_prefix': 'shadow/batchctl/run0001/staging/b1/'},
    {'staging_prefix': 'shadow/batchctl/run0001/staging/b1/000000000007'},
    {'base_prefix': 'shadow/batchctl/run0001/states/%s/' % H2},
    {'base_prefix': 'shadow/batchctl/run0001/states/'},
])
def test_job_spec_must_be_consistent(over):
    with pytest.raises(ValueError):
        spec(**over)


def test_definite_errors_are_store_errors():
    for kind in (I.NotFound, I.PreconditionFailed, I.TooLarge):
        assert issubclass(kind, I.StoreError)
    assert issubclass(I.RentUnavailable, I.ProviderError)
    assert issubclass(I.JudgeInvalid, ValueError)
