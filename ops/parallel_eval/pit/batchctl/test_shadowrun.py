"""The shadow entry point: its settings, what it refuses, and what one run
with the parts that are not wired yet writes (R2 against moto, a fake Lium
command line)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import boto3
import pytest
from moto import mock_aws

from .. import ALLOW_SHARDED_GPUS_DEFAULT, PRICE_CEILING_USD_H
from ..shadow import ShadowWriteError
from . import MODE_SHADOW
from . import hostlink
from . import shadowrun as S
from .guard import STATE_FILES, shadow_prefix
from .interfaces import BoxError, JobSpec, JudgeNotReady
from .r2 import R2Store

RUN = 'shadow01'
GENESIS = 'a' * 64
BUCKET = 'batchctl-shadowrun'


def args(tmp_path: Path, command: str = 'check', *more: str):
    return S.options([
        command, '--run-id', RUN, '--state-root', str(tmp_path / 'own' / 'root'),
        '--bucket-config', str(tmp_path / 'none.json'), '--genesis-state', GENESIS, *more,
    ])


class Cli:
    """A Lium command line with no pods and no offers. It records each call."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, ...]] = []

    def __call__(self, *words: str, timeout: float) -> Tuple[int, str, str]:
        self.calls.append(words)
        return 0, '[]', ''


@pytest.fixture
def world(monkeypatch):
    with mock_aws():
        client = boto3.client('s3', region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test')
        client.create_bucket(Bucket=BUCKET)
        cli = Cli()
        monkeypatch.setattr(S, 'store', lambda _args: R2Store(client, BUCKET))
        monkeypatch.setattr(hostlink, 'load', lambda name: cli)
        yield client, cli


def keys(client: Any) -> List[str]:
    return [row['Key'] for row in client.list_objects_v2(Bucket=BUCKET).get('Contents', [])]


def test_the_default_run_is_shadow_on_rtx5090_with_at_most_eight_boxes(tmp_path):
    config, lium = S.settings(args(tmp_path))
    assert config.mode == MODE_SHADOW and config.gpu_allow_list == ('RTX5090',)
    assert [(r.provider, r.gpu, r.tier) for r in config.core.ladder] == [
        ('lium', 'RTX5090', 'secure'), ('lium', 'RTX5090', 'spot')]
    assert config.core.max_boxes == lium.max_boxes == 8 and lium.gpu_count == config.gpu_count == 8
    assert lium.min_ram_gb == 300.0 and lium.gpus == ('RTX5090',) and lium.run_id == RUN
    # A 5090 run opens the sharded gate for itself. The default that the live lane reads stays closed.
    assert config.allow_sharded_gpus is True and ALLOW_SHARDED_GPUS_DEFAULT is False
    assert config.lease_ttl_s == 6 * 3600.0 and config.box_ttl_s == 8 * 3600.0
    assert config.price_ceiling_usd_h == PRICE_CEILING_USD_H == 56.0
    assert set(lium.never_names) >= {
        'brave-shark-ec', 'noble-wolf-26', 'calm-matrix-56', 'noble-fox-5c', 'gentle-wolf-f6', 'noble-shark-d6'}
    assert S.plan(config, lium)['r2_prefix'] == shadow_prefix(RUN) == 'pit-shadow/batchctl/%s/' % RUN


def test_the_options_of_a_run(tmp_path):
    config, lium = S.settings(args(
        tmp_path, 'check', '--gpus', 'B300', '--price-ceiling-usd-h', '20', '--never-name', 'my-node-01',
        '--lease-hours', '5', '--box-hours', '7'))
    # A run without a sharded class leaves the gate closed.
    assert config.allow_sharded_gpus is False and config.gpu_allow_list == ('B300',)
    assert config.price_ceiling_usd_h == 20.0 and (config.lease_ttl_s, config.box_ttl_s) == (5 * 3600.0, 7 * 3600.0)
    assert lium.never_names[-1] == 'my-node-01' and 'noble-fox-5c' in lium.never_names


@pytest.mark.parametrize('more', [
    ('--max-boxes', '9'), ('--max-boxes', '0'), ('--gpus', 'RTX4090'), ('--gpus', ''), ('--gpus', 'RTX5090,RTX5090'),
    ('--box-hours', '1'), ('--price-ceiling-usd-h', '56.01'), ('--price-ceiling-usd-h', '0'),
])
def test_bad_settings_are_refused(tmp_path, more):
    with pytest.raises(ValueError):
        S.settings(args(tmp_path, 'check', *more))


def test_there_is_no_mode_but_shadow():
    assert set(S.options(['check', '--run-id', RUN, '--state-root', 'x', '--bucket-config', 'y',
                          '--genesis-state', GENESIS]).__dict__) == {
        'command', 'run_id', 'state_root', 'bucket_config', 'genesis_state', 'gpus', 'max_boxes',
        'max_infra_failures', 'max_cost_usd', 'price_ceiling_usd_h', 'never_name', 'lease_hours', 'box_hours',
        'survey'}
    with pytest.raises(SystemExit):
        S.options(['live', '--run-id', RUN, '--state-root', 'x', '--bucket-config', 'y', '--genesis-state', GENESIS])


def test_a_state_root_in_or_around_a_live_root_is_refused(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    inside = S.options(['run', '--run-id', RUN, '--state-root', str(live / 'shadow'), '--bucket-config', 'y',
                        '--genesis-state', GENESIS])
    around = S.options(['run', '--run-id', RUN, '--state-root', str(tmp_path), '--bucket-config', 'y',
                        '--genesis-state', GENESIS])
    for bad in (inside, around):
        for create in (False, True):
            with pytest.raises(ShadowWriteError):
                S.state_root(bad, create=create, env={'PE_STATE': str(live)})
    assert not (live / 'shadow').exists()


def test_the_parts_that_are_not_wired_fail_closed():
    assert S.NoBatches().list() == ()
    spec = JobSpec(
        run_id=RUN, batch_id='b', batch_ref='', fence=1, attempt=1, base_state=GENESIS, base_seq=0,
        base_prefix='p/%s/' % GENESIS, expected_fp32_master_digest=GENESIS, staging_prefix='s/%012d/' % 1,
        heartbeat_every_s=60.0,
        job_ttl_s=3600.0,
    )
    with pytest.raises(BoxError):
        S.NoLaunch().launch('box', 'lium', spec)
    with pytest.raises(JudgeNotReady):
        S.NoVerdict().verdict(b'{}', batch_id='b', base_state=GENESIS)
    assert set(S.STUBS) == {'batch_feed', 'box_agent', 'judge', 'genesis_import'}


def test_only_the_named_host_helper_can_be_loaded(monkeypatch):
    assert set(hostlink.HOST_HELPERS) == {'lium'}
    with pytest.raises(hostlink.HostLinkError):
        hostlink.load('other')
    monkeypatch.setitem(hostlink.HOST_HELPERS, 'lium', ('no_such_module_of_the_host', 'lium'))
    with pytest.raises(hostlink.HostLinkError):
        hostlink.load('lium')
    monkeypatch.setitem(hostlink.HOST_HELPERS, 'lium', ('json', 'decoder'))
    with pytest.raises(hostlink.HostLinkError):
        hostlink.load('lium')
    monkeypatch.setitem(hostlink.HOST_HELPERS, 'lium', ('json', 'dumps'))
    assert hostlink.load('lium') is json.dumps


def test_check_writes_nothing_and_rents_nothing(tmp_path, world):
    client, cli = world
    report, wrong = S.check(args(tmp_path, 'check', '--survey'))
    assert wrong == [] and report['bucket'] == 'ok' and report['state_root'] == 'will be created'
    assert report['stubs'] == S.STUBS and [row['tier'] for row in report['offers']] == ['secure', 'spot']
    assert keys(client) == [] and not (tmp_path / 'own').exists()
    assert all(words[0] == 'ls' for words in cli.calls)


def test_a_run_writes_only_its_own_prefix_and_its_own_root_and_rents_nothing(tmp_path, world):
    client, cli = world
    client.put_object(Bucket=BUCKET, Key='HEAD.json', Body=b'live')
    controller = S.build(args(tmp_path, 'run')).open()
    try:
        for _ in range(5):
            controller.step()
    finally:
        controller.close()
    written = [key for key in keys(client) if key != 'HEAD.json']
    assert written and all(key.startswith(shadow_prefix(RUN)) for key in written)
    assert client.get_object(Bucket=BUCKET, Key='HEAD.json')['Body'].read() == b'live'
    own = tmp_path / 'own'
    assert sorted(path.name for path in own.iterdir()) == ['root']
    assert {path.name for path in (own / 'root').iterdir()} <= STATE_FILES
    # No batch comes in, so no box is rented and none is removed.
    assert not any(words[0] in ('up', 'rm') for words in cli.calls)
    # The same root opens again: the run recovers from its journal.
    again = S.build(args(tmp_path, 'run')).open()
    again.close()
