"""The runner: its two modes, what each command writes, and the parts that it wires.

The validator host is a set of fakes: the helpers that ``hostlink`` loads (a
fake Lium command line, a fake pod, the fixture of the live scorer), a bucket
in moto, a live queue and a gate state in a temp directory. The controller,
the box agent, the judge, the seed and the pay hook are the real ones.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pytest
from moto import mock_aws

from .. import ALLOW_SHARDED_GPUS_DEFAULT, PRICE_CEILING_USD_H
from .. import shadow_guard
from ..gate_writer import HostGateWriter
from ..shadow import ShadowWriteError
from . import MODE_LIVE, MODE_SHADOW
from . import agent as A
from . import hostlink
from . import lium as L
from . import run as R
from .guard import MARKERS, box_name
from .layout import Layout
from .r2 import R2Presigner, R2Store
from .test_agent import HARNESS, Calls, FakePod
from .test_fakes import FakeLium, offer
from .test_feed import dispatch, turn
from .test_genesis import H, documents, job, put_live, write_json
from .test_judge import EPOCH, RUN, World
from .test_settle import FakeRounds

BUCKET = 'batchctl-run'
GENESIS = 'a' * 64
SIX = {'brave-shark-ec', 'noble-wolf-26', 'calm-matrix-56', 'noble-fox-5c', 'gentle-wolf-f6', 'noble-shark-d6'}
HOST_NEVER = ('swift-wolf-bd', 'noble-shark-d6')


class Clock:
    """The time of the controller: the wall time, plus what a test moves it ahead."""

    def __init__(self) -> None:
        self.ahead = 0.0

    def now(self) -> float:
        return time.time() + self.ahead


class Host:
    """The validator host of a test."""

    def __init__(self, tmp: Path, monkeypatch: Any, client: Any) -> None:
        self.tmp, self.client = tmp, client
        self.world = World(tmp / 'round')
        self.gate_state = self.world.gate
        self.life, self.masters, self.shadow = tmp / 'life', tmp / 'fp32-index', tmp / 'shadow'
        self.live_dir = tmp / 'live-run'
        self.sharded = tmp / 'trainer-host-v834f'
        self.sharded.mkdir()
        (self.sharded / A.TRAINER_LAUNCHER).write_text('# launcher\n')
        (self.sharded / A.ROOT_ADMISSION).write_text('{}')
        self.shadow.mkdir()
        # The live HEAD: its files in the bucket, its descriptor in the queue, its master index.
        self.descriptor, index = documents()
        put_live(client, BUCKET)
        write_json(self.life / 'CURRENT.json', {'epoch_index': EPOCH, 'checkpoint': self.descriptor})
        write_json(self.masters / (H + '.json'), index)
        (self.life / 'epochs').mkdir()
        self.lium = FakeLium().add(offer('node-a'), offer('node-b', tier='spot'))
        self.pod = FakePod()
        self.calls = Calls(self.pod)
        tools = self.calls.tools()
        scorer = self.world.live.scorer
        self.helpers: Dict[str, Any] = {
            'lium': self.lium, 'pod_of': tools.pod_of, 'boot_helpers': tools.boot,
            'push_probe_bundle': tools.probe_bundle, 'bundle_trainer': tools.trainer_bundle,
            'ship_inputs': tools.ship_inputs, 'ship_train_pit': tools.ship_trainer, 'train_cmd': tools.train_cmd,
            'put': tools.put, 'put_tar': tools.put_tar, 'public_harness': lambda: dict(HARNESS),
            'score': scorer.score, 'blocks_baseline': scorer.blocks_baseline,
        }
        self.values: Dict[str, Any] = {'heldout': scorer.HELDOUT, 'never_names': HOST_NEVER}
        self.rounds = FakeRounds()
        self.grants: List[Tuple[int, str]] = []
        self.writers = 0
        self.clock = Clock()
        monkeypatch.setattr(R, 'SystemClock', lambda: self.clock)
        monkeypatch.setattr(hostlink, 'load', self.load)
        monkeypatch.setattr(hostlink, 'value', self.value)
        monkeypatch.setattr(R, 'raw_store', lambda args: (R2Store(client, BUCKET), R2Presigner(client, BUCKET)))
        monkeypatch.setattr(R, 'gate_writer', self.gate_writer)
        # The provider waits on the clock of the fake command line, not on real time.
        monkeypatch.setattr(R, 'LiumProvider', lambda cli, config: L.LiumProvider(
            cli, config, monotonic=self.lium.clock.monotonic, sleep=self.lium.clock.sleep))

    def load(self, name: str) -> Any:
        if name not in self.helpers:
            raise hostlink.HostLinkError('host helper %s cannot be loaded' % name)
        return self.helpers[name]

    def value(self, name: str) -> Any:
        if name not in self.values:
            raise hostlink.HostLinkError('host value %s cannot be loaded' % name)
        return self.values[name]

    def gate_writer(self, gate_state: Path) -> HostGateWriter:
        """The real writer of the gate, with the round store and the K1 grant of the host as fakes."""
        self.writers += 1
        return HostGateWriter(
            epochs_root=Path(gate_state) / 'epochs', rounds=self.rounds, baseline_v1=self.world.live.BV,
            k1_grant_fn=lambda epoch, why: self.grants.append((epoch, why)) or {'granted': epoch},
        )

    def argv(self, command: str, *more: str, live: bool = False, arm: Optional[str] = RUN) -> List[str]:
        words = [command, '--live' if live else '--shadow', '--run-id', RUN,
                 '--bucket-config', str(self.tmp / 'none.json'), '--life-root', str(self.life),
                 '--gate-state', str(self.gate_state), '--sharded-host', str(self.sharded)]
        if live:
            words += ['--state-dir', str(self.live_dir)]
            if arm is not None:
                words += ['--arm-live', arm]
        return words + list(more)

    def args(self, command: str, *more: str, **how: Any) -> Any:
        return R.options(self.argv(command, *more, **how))

    def env(self, **more: str) -> Dict[str, str]:
        return dict({'PIT_SHADOW_STATE': str(self.shadow)}, **more)

    def seed(self, **how: Any) -> str:
        args = self.args('seed', '--master-index-dir', str(self.masters), **how)
        assert R.seed(args, self.env()) == 0
        return json.loads(R.folders(args, self.env()).genesis.read_text())['genesis_state']

    def epoch(self, number: int, *, sample: bool = True) -> None:
        """One turn of a miner in the live queue, made for the live HEAD."""
        folder = self.life / 'epochs' / str(number)
        write_json(folder / 'K1-TURN.json', turn(number))
        write_json(folder / 'TRAINING-JOB.json', job(self.descriptor))
        write_json(folder / 'TRAIN-DISPATCH-FF.json', dispatch())
        if sample:
            self.world.sample(number)

    def keys(self, prefix: str = '') -> Dict[str, str]:
        return {info.key: info.etag for info in R2Store(self.client, BUCKET).list(prefix)}

    def files(self, top: Optional[Path] = None) -> Dict[str, bytes]:
        top = top or self.tmp
        return {str(path.relative_to(top)): path.read_bytes() for path in sorted(top.rglob('*')) if path.is_file()}

    def seen(self) -> Tuple[Dict[str, str], Dict[str, bytes], int, int, int]:
        """Everything that a command could change."""
        return self.keys(), self.files(), len(self.lium.calls), len(self.pod.commands), self.writers


@pytest.fixture
def host(tmp_path: Path, monkeypatch: Any):
    try:
        with mock_aws():
            client = boto3.client('s3', region_name='us-east-1', aws_access_key_id='test',
                                  aws_secret_access_key='test')
            client.create_bucket(Bucket=BUCKET)
            yield Host(tmp_path / 'own', monkeypatch, client)
    finally:
        shadow_guard.leave()


def last_line(capsys: Any) -> Dict[str, Any]:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# ----------------------------------------------------------------------------
# The two modes
# ----------------------------------------------------------------------------

def test_a_command_names_one_mode_and_live_needs_the_run_id_a_second_time(host: Host) -> None:
    plain = [word for word in host.argv('run') if word != '--shadow']
    for words in (plain, plain + ['--shadow', '--live']):
        with pytest.raises(SystemExit):
            R.options(words)
    shadow = host.args('run')
    assert R.mode_of(shadow) == MODE_SHADOW and not R.armed(shadow)
    # The flag of the live mode does not make a shadow run live.
    shadow = host.args('run', '--arm-live', RUN)
    assert R.mode_of(shadow) == MODE_SHADOW and not R.armed(shadow)
    for arm in (None, 'other1', '', RUN + ' ', RUN.upper()):
        live = host.args('run', live=True, arm=arm)
        assert R.mode_of(live) == MODE_LIVE and not R.armed(live)
        with pytest.raises(R.RunnerError, match='arm-live'):
            R.enter_mode(live, host.env())
    live = host.args('run', live=True)
    assert R.armed(live)
    R.enter_mode(live, host.env())
    # A live run does not enter the shadow guard.
    assert shadow_guard.active() is None


@pytest.mark.parametrize('command', ['check', 'seed', 'run'])
def test_live_without_its_flag_starts_nothing(host: Host, command: str, capsys: Any) -> None:
    host.epoch(EPOCH)
    before = host.seen()
    for arm in (None, 'other1'):
        words = host.argv(command, '--master-index-dir', str(host.masters), '--genesis-state', GENESIS, live=True,
                          arm=arm)
        assert R.main(words, host.env()) == 2
        assert 'arm-live' in last_line(capsys)['error']
    assert host.seen() == before and not host.live_dir.exists()


def test_a_shadow_command_enters_the_shadow_guard_before_any_work(host: Host, monkeypatch: Any, capsys: Any) -> None:
    scratch = host.tmp / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(scratch))
    assert shadow_guard.active() is None
    # No first state: the command stops. The guard is on all the same.
    assert R.main(host.argv('check'), host.env()) == 2
    assert 'no first state' in last_line(capsys)['error']
    guard = shadow_guard.active()
    assert guard is not None and guard.state_root == host.shadow.resolve() and guard.r2_prefixes == ('pit-shadow/',)
    assert R.run_prefix(RUN, MODE_SHADOW).startswith(guard.r2_prefixes)
    # A file outside the shadow state root, and an R2 key outside the shadow prefix: both are refused.
    with pytest.raises(ShadowWriteError):
        (host.life / 'CURRENT.json').write_text('{}')
    with pytest.raises(ShadowWriteError):
        host.client.put_object(Bucket=BUCKET, Key='pit/batchctl/%s/HEAD.json' % RUN, Body=b'x')
    with pytest.raises(ShadowWriteError):
        host.client.delete_object(Bucket=BUCKET, Key=sorted(host.keys())[0])
    (host.shadow / 'note').write_text('inside')
    host.client.put_object(Bucket=BUCKET, Key=R.run_prefix(RUN, MODE_SHADOW) + 'note', Body=b'x')


def test_shadow_needs_its_own_state_root(host: Host, capsys: Any) -> None:
    before = host.seen()
    assert R.main(host.argv('run', '--genesis-state', GENESIS), {}) == 2
    assert 'PIT_SHADOW_STATE' in last_line(capsys)['error']
    # A shadow state root that is a live root, or holds one.
    for name, live in (('PE_STATE', host.shadow), ('HG_GATE_STATE', host.shadow / 'gate')):
        assert R.main(host.argv('run', '--genesis-state', GENESIS), host.env(**{name: str(live)})) == 2
        assert 'overlaps' in last_line(capsys)['error']
    assert host.seen() == before and shadow_guard.active() is None


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

def test_the_default_run_is_rtx5090_with_at_most_eight_boxes(host: Host) -> None:
    config, lium = R.settings(host.args('check'), GENESIS, HOST_NEVER)
    assert config.mode == MODE_SHADOW and not config.live_armed and config.gpu_allow_list == ('RTX5090',)
    assert [(r.provider, r.gpu, r.tier) for r in config.core.ladder] == [
        ('lium', 'RTX5090', 'secure'), ('lium', 'RTX5090', 'spot')]
    assert config.core.max_boxes == lium.max_boxes == 8 and lium.gpu_count == config.gpu_count == 8
    assert lium.min_ram_gb == 300.0 and lium.gpus == ('RTX5090',) and lium.run_id == RUN and lium.mode == MODE_SHADOW
    # A 5090 run opens the sharded gate for itself. The default that the serial lane reads stays closed.
    assert config.allow_sharded_gpus is True and ALLOW_SHARDED_GPUS_DEFAULT is False
    assert config.lease_ttl_s == 6 * 3600.0 and config.box_ttl_s == 8 * 3600.0
    assert config.price_ceiling_usd_h == PRICE_CEILING_USD_H == 56.0
    # The six machines of other work, then the never-list of the live lane, each name one time.
    assert lium.never_names == L.NEVER_NAMES_DEFAULT + ('swift-wolf-bd',) and set(lium.never_names) >= SIX
    live, live_lium = R.settings(host.args('check', live=True), GENESIS)
    assert live.mode == live_lium.mode == MODE_LIVE and live.live_armed is True
    assert (live.gpu_allow_list, live.lease_ttl_s, live.allow_sharded_gpus) == (('RTX5090',), 6 * 3600.0, True)


def test_the_options_of_a_run(host: Host) -> None:
    config, lium = R.settings(host.args(
        'check', '--price-ceiling-usd-h', '20', '--never-name', 'my-node-01', '--lease-hours', '5', '--box-hours', '7',
        '--max-boxes', '3', '--max-cost-usd', '250', '--max-infra-failures', '2'), GENESIS)
    assert config.price_ceiling_usd_h == 20.0 and (config.lease_ttl_s, config.box_ttl_s) == (5 * 3600.0, 7 * 3600.0)
    assert (config.core.max_boxes, config.core.max_infra_failures, config.max_cost_usd) == (3, 2, 250.0)
    assert lium.never_names == L.NEVER_NAMES_DEFAULT + ('my-node-01',) and lium.max_boxes == 3


@pytest.mark.parametrize('more', [
    ('--max-boxes', '9'), ('--max-boxes', '0'), ('--gpus', 'RTX4090'), ('--gpus', ''), ('--gpus', 'RTX5090,RTX5090'),
    ('--gpus', 'RTX5090,B300'), ('--gpus', 'B300'), ('--box-hours', '1'), ('--price-ceiling-usd-h', '56.01'),
    ('--price-ceiling-usd-h', '0'), ('--never-name', ' '),
])
def test_bad_settings_are_refused(host: Host, more: Tuple[str, ...]) -> None:
    with pytest.raises(R.RunnerError):
        R.settings(host.args('check', *more), GENESIS)


def test_the_first_state_is_a_state_id(host: Host) -> None:
    where = R.folders(host.args('check'), host.env())
    assert R.genesis_state(host.args('check', '--genesis-state', GENESIS), where) == GENESIS
    for bad in ('a' * 63, 'A' * 64, 'head'):
        with pytest.raises(R.RunnerError):
            R.genesis_state(host.args('check', '--genesis-state', bad), where)
    with pytest.raises(R.RunnerError, match='seed'):
        R.genesis_state(host.args('check'), where)


def test_the_directory_of_a_run(host: Host) -> None:
    env = host.env()
    where = R.folders(host.args('run'), env)
    assert where.state_dir == (host.shadow / 'batchctl' / RUN).resolve() and not where.state_dir.exists()
    assert (where.controller, where.private) == (where.state_dir / 'controller', where.state_dir / 'private')
    with pytest.raises(R.RunnerError, match='PIT_SHADOW_STATE'):
        R.folders(host.args('run'), {})
    # A shadow run has no directory outside the shadow state root, and not the root itself.
    for bad in (host.tmp / 'elsewhere', host.shadow, host.shadow / '..' / 'elsewhere'):
        with pytest.raises(R.RunnerError, match='under PIT_SHADOW_STATE'):
            R.folders(host.args('run', '--state-dir', str(bad)), env)
    assert R.folders(host.args('run', live=True), env).state_dir == host.live_dir.resolve()
    with pytest.raises(R.RunnerError, match='--state-dir'):
        R.folders(R.options([word for word in host.argv('run', live=True) if word not in (
            '--state-dir', str(host.live_dir))]), env)
    # In both modes: not in a live root, and not around one.
    lives = host.tmp / 'lives'
    for inside, live_root in ((lives / 'state' / 'run', lives / 'state'), (lives / 'state', lives / 'state' / 'gate')):
        with pytest.raises(ShadowWriteError):
            R.folders(host.args('run', '--state-dir', str(inside), live=True), host.env(PE_STATE=str(live_root)))
        with pytest.raises(ShadowWriteError):
            R.folders(host.args('run', '--state-dir', str(inside)),
                      {'PIT_SHADOW_STATE': str(lives), 'HG_GATE_STATE': str(live_root)})
    assert not lives.exists()


def test_only_the_named_host_helper_can_be_loaded(monkeypatch: Any) -> None:
    common, box, score = ('ops.parallel_eval.live.' + name for name in ('pe_common', 'pe_box', 'pe_score'))
    assert hostlink.HOST_HELPERS == {
        'lium': (common, 'lium'), 'pod_of': (common, 'pod_of'),
        'boot_helpers': (box, 'boot_helpers'), 'push_probe_bundle': (box, 'push_probe_bundle'),
        'bundle_trainer': (box, 'bundle_trainer'), 'ship_inputs': (box, 'ship_inputs'),
        'ship_train_pit': (box, '_ship_train_pit'), 'train_cmd': (box, '_train_pit_inner_cmd'),
        'put': (box, 'put'), 'put_tar': (box, 'put_tar'),
        'public_harness': ('ops.parallel_eval.pit.box', '_load_public_harness'),
        'score': (score, 'score'), 'blocks_baseline': (score, 'blocks_baseline'),
    }
    assert hostlink.HOST_VALUES == {'heldout': (score, 'HELDOUT'), 'never_names': (common, 'NEVER_NAMES')}
    # The box helpers are in this repository. The others need the code of the validator host.
    for name, (module, _) in hostlink.HOST_HELPERS.items():
        if module in (box, 'ops.parallel_eval.pit.box'):
            assert callable(hostlink.load(name)), name
    with pytest.raises(hostlink.HostLinkError):
        hostlink.load('other')
    with pytest.raises(hostlink.HostLinkError):
        hostlink.load('heldout')
    with pytest.raises(hostlink.HostLinkError):
        hostlink.value('lium')
    monkeypatch.setitem(hostlink.HOST_HELPERS, 'lium', ('no_such_module_of_the_host', 'lium'))
    with pytest.raises(hostlink.HostLinkError):
        hostlink.load('lium')
    monkeypatch.setitem(hostlink.HOST_HELPERS, 'lium', ('json', 'decoder'))
    with pytest.raises(hostlink.HostLinkError):
        hostlink.load('lium')
    monkeypatch.setitem(hostlink.HOST_HELPERS, 'lium', ('json', 'dumps'))
    assert hostlink.load('lium') is json.dumps


# ----------------------------------------------------------------------------
# seed and check
# ----------------------------------------------------------------------------

def test_seed_copies_the_live_head_under_the_prefix_of_the_run_and_keeps_the_state_id(host: Host, capsys: Any) -> None:
    live_keys, live_files = host.keys(), host.files()
    words = host.argv('seed', '--master-index-dir', str(host.masters))
    assert R.main(words, host.env()) == 0
    said = last_line(capsys)
    state_id = said['genesis_state']
    assert (said['event'], said['mode'], said['run_id'], said['checkpoint']) == ('seeded', MODE_SHADOW, RUN, H)
    prefix = 'pit-shadow/batchctl/%s/' % RUN
    made = set(host.keys()) - set(live_keys)
    assert made and all(key.startswith(prefix) for key in made)
    assert Layout(prefix).state_manifest(state_id) in made
    # The live side is as it was: each key, each file.
    assert {key: etag for key, etag in host.keys().items() if key in live_keys} == live_keys
    where = R.folders(host.args('run'), host.env())
    assert {name: data for name, data in host.files().items() if name in live_files} == live_files
    assert set(host.files()) - set(live_files) == {str(where.genesis.relative_to(host.tmp))}
    assert R.genesis_state(host.args('run'), where) == state_id
    # The same seed again gives the same state and writes no new key.
    assert R.main(words, host.env()) == 0 and last_line(capsys)['genesis_state'] == state_id
    assert set(host.keys()) - set(live_keys) == made
    # What a shadow seed kept is not the first state of a live run, also in the same directory.
    live = host.args('run', '--state-dir', str(where.state_dir), live=True)
    with pytest.raises(R.RunnerError, match='seed'):
        R.genesis_state(live, R.folders(live, host.env()))


def test_a_live_seed_goes_under_the_live_prefix(host: Host) -> None:
    live_keys = host.keys()
    state_id = host.seed(live=True)
    made = set(host.keys()) - set(live_keys)
    assert made and all(key.startswith('pit/batchctl/%s/' % RUN) for key in made)
    assert state_id == host.seed() and not host.live_dir.joinpath('controller').exists()


def test_seed_needs_the_master_index(host: Host, capsys: Any) -> None:
    before = host.seen()
    assert R.main(host.argv('seed'), host.env()) == 2
    assert 'master-index-dir' in last_line(capsys)['error']
    assert R.main(host.argv('seed', '--master-index-dir', str(host.tmp / 'none')), host.env()) == 2
    assert 'GenesisError' in last_line(capsys)['error']
    assert host.seen() == before


def test_check_writes_nothing_and_rents_nothing(host: Host, capsys: Any) -> None:
    state_id = host.seed()
    host.epoch(EPOCH)
    host.epoch(EPOCH + 1, sample=False)
    before = host.seen()
    assert R.main(host.argv('check', '--survey'), host.env()) == 0
    report = last_line(capsys)
    assert report['wrong'] == [] and report['mode'] == MODE_SHADOW and report['genesis_state'] == state_id
    assert report['r2_prefix'] == 'pit-shadow/batchctl/%s/' % RUN
    assert report['writes'] == {
        'r2': 'pit-shadow/batchctl/%s/' % RUN, 'files': str((host.shadow / 'batchctl' / RUN).resolve()),
        'gate_decisions': None, 'weights': None, 'score_file': None, 'turn_file': None,
    }
    for part in ('bucket', 'first_state', 'live_scorer', 'box_helpers', 'box_agent', 'never_names_of_the_live_lane'):
        assert report[part] == 'ok', part
    assert 'gate_writer' not in report and report['state_dir'] == 'will be created'
    assert [row['tier'] for row in report['lium']] == ['secure', 'spot']
    assert set(report['never_names']) >= SIX | set(HOST_NEVER)
    queue = report['live_queue']
    assert queue['batches_ready'] == ['e%d' % EPOCH] and list(queue['wait_for_round']) == ['e%d' % (EPOCH + 1)]
    keys, files, _, pod_commands, writers = host.seen()
    assert (keys, files, pod_commands, writers) == (before[0], before[1], before[3], 0)
    assert {words[0] for words in host.lium.calls} == {'ls'}


def test_check_names_each_part_that_is_missing(host: Host, capsys: Any) -> None:
    del host.helpers['score'], host.helpers['put_tar'], host.values['never_names']
    host.helpers['public_harness'] = lambda: {}
    (host.life / 'CURRENT.json').unlink()
    before = host.seen()
    assert R.main(host.argv('check', '--genesis-state', GENESIS), host.env()) == 1
    report = last_line(capsys)
    failed = sorted(part for part, said in report.items() if said == 'failed')
    assert failed == ['box_agent', 'box_helpers', 'first_state', 'live_queue', 'live_scorer',
                      'never_names_of_the_live_lane']
    assert sorted(line.split(':')[0] for line in report['wrong']) == failed
    assert any('sampling contract' in line for line in report['wrong'])
    assert any('seed' in line for line in report['wrong'])
    assert host.seen()[:2] == before[:2]


def test_the_scorer_must_read_the_rounds_of_the_gate_state_of_the_run(host: Host) -> None:
    assert R.scorer(host.gate_state) == (host.helpers['score'], host.helpers['blocks_baseline'])
    with pytest.raises(R.RunnerError, match='another directory'):
        R.scorer(host.tmp / 'other-gate')
    state_id = host.seed()
    before = host.seen()
    with pytest.raises(R.RunnerError, match='another directory'):
        R.build(host.args('run', '--gate-state', str(host.tmp / 'other-gate')), host.env())
    assert host.seen() == before and state_id


def test_a_live_check_tests_the_gate_writer_and_writes_nothing(host: Host, capsys: Any) -> None:
    state_id = host.seed(live=True)
    before = host.seen()
    assert R.main(host.argv('check', live=True), host.env()) == 0
    report = last_line(capsys)
    assert report['wrong'] == [] and report['gate_writer'] == 'ok' and report['genesis_state'] == state_id
    assert (report['mode'], report['r2_prefix']) == (MODE_LIVE, 'pit/batchctl/%s/' % RUN)
    assert report['writes'] == {
        'r2': 'pit/batchctl/%s/' % RUN, 'files': str(host.live_dir.resolve()),
        'gate_decisions': str(host.gate_state / 'epochs'), 'weights': None, 'score_file': None, 'turn_file': None,
    }
    assert host.seen()[:4] == before[:4] and shadow_guard.active() is None


def test_the_gate_writer_needs_the_host_parts_of_a_later_write(host: Host, monkeypatch: Any) -> None:
    monkeypatch.undo()
    # Off the validator host there is no code of the live gate: nothing to write a decision with.
    with pytest.raises(R.RunnerError, match='gate writer'):
        R.gate_writer(host.gate_state)
    writer = host.gate_writer(host.gate_state)
    writer.preflight()
    for broken in (HostGateWriter(epochs_root=host.gate_state / 'epochs', rounds=host.rounds,
                                  baseline_v1=host.world.live.BV),
                   HostGateWriter(epochs_root=host.gate_state / 'epochs', rounds=host.rounds, shadow=True,
                                  baseline_v1=host.world.live.BV, k1_grant_fn=lambda epoch, why: None)):
        with pytest.raises(R.GateWriteError):
            broken.preflight()
    assert not (host.gate_state / 'epochs' / str(EPOCH) / 'DECISION.json').exists()


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

def step(controller: Any, times: int = 6) -> None:
    for _ in range(times):
        controller.step()


def test_a_shadow_run_installs_its_first_head_and_writes_only_its_own_places(host: Host) -> None:
    state_id = host.seed()
    keys, files = host.keys(), host.files()
    parts = R.build(host.args('run'), host.env())
    assert parts.mode == MODE_SHADOW and parts.settler is None and host.writers == 0
    controller = parts.controller.open()
    try:
        step(controller)
        status = controller.status()
        controller.write_status()
    finally:
        controller.close()
    assert status['mode'] == MODE_SHADOW and status['head']['state_id'] == state_id and not status['halted']
    made = set(host.keys()) - set(keys)
    assert made and all(key.startswith('pit-shadow/batchctl/%s/' % RUN) for key in made)
    assert {key: etag for key, etag in host.keys().items() if key in keys} == keys
    where = parts.where
    new = set(host.files()) - set(files)
    assert new and all((host.tmp / name).is_relative_to(where.state_dir) for name in new)
    assert {name: data for name, data in host.files().items() if name in files} == files
    assert {path.name for path in where.controller.iterdir()} == {
        MARKERS[MODE_SHADOW], 'journal.jsonl', 'LOCK', 'status.json'}
    # No batch came in: no box is rented and none is removed.
    assert not host.lium.verbs('up') and not host.lium.verbs('rm') and not host.pod.commands
    # The same directory opens again: the run recovers from its journal.
    again = R.build(host.args('run'), host.env()).controller.open()
    try:
        assert again.status()['head']['state_id'] == state_id
    finally:
        again.close()
    # A live run cannot take the directory of a shadow run.
    with pytest.raises(ShadowWriteError):
        R.build(host.args('run', '--state-dir', str(where.state_dir), '--genesis-state', state_id, live=True),
                host.env())


def test_a_batch_gets_a_box_when_its_round_is_drawn_and_the_job_of_the_lease_starts(host: Host) -> None:
    live_keys = set(host.keys())
    state_id = host.seed()
    host.epoch(EPOCH, sample=False)
    (host.gate_state / 'epochs' / str(EPOCH) / 'SAMPLE.private.json').unlink()
    parts = R.build(host.args('run'), host.env())
    said: List[str] = []
    parts.source.log = said.append
    controller = parts.controller.open()
    try:
        step(controller)
        # The gate has not drawn the round of the epoch: the batch waits and costs nothing.
        assert parts.source.list() == [] and list(parts.source.waiting) == ['e%d' % EPOCH]
        assert len(said) == 1 and 'waits for its round' in said[0]
        assert controller.status()['batches']['total'] == 0 and not host.lium.verbs('up')
        host.world.sample(EPOCH)
        assert [ref.batch_id for ref in parts.source.list()] == ['e%d' % EPOCH] and not parts.source.waiting
        for _ in range(40):
            host.clock.ahead += 15.0
            controller.step()
            if host.pod.claimed:
                break
        status = controller.status()
    finally:
        controller.close()
    # One box of the run, on the secure tier, with the name of the shadow mode.
    ups = host.lium.verbs('up')
    assert len(ups) == 1 and ups[0][1] == 'node-a' and host.lium.names() == [box_name(RUN, 1, 0, MODE_SHADOW)]
    assert host.lium.names()[0].startswith('sn120-shadow-bctl-') and not host.lium.verbs('rm')
    # The box is prepared as the serial lane does it, and one job of the lease starts on it.
    assert host.calls.names() == ['pod_of', 'boot', 'probe_bundle', 'trainer_bundle', 'put_tar', 'ship_inputs',
                                  'ship_trainer']
    assert host.calls.seen[4][1] == str(host.sharded)
    assert host.calls.seen[5] == ('ship_inputs', EPOCH, str(host.life / 'epochs' / str(EPOCH)))
    starts = [cmd for cmd in host.pod.commands if 'nohup setsid python3' in cmd]
    assert len(starts) == 1 and host.pod.claimed
    job_doc = json.loads(next(data for path, data in host.pod.files.items() if path.endswith('/job.json')))
    assert job_doc['base_state'] == state_id and job_doc['batch_id'] == 'e%d' % EPOCH and job_doc['fence'] == 1
    assert status['batches']['total'] == 1 and status['boxes']['running'] == 1 and not status['halted']
    assert status['alerts']['counts'] == {}
    # Every key of the run is under its prefix. The private data of the host is in its own directory only.
    assert all(key.startswith('pit-shadow/batchctl/%s/' % RUN) for key in set(host.keys()) - live_keys)
    assert not any(path.is_file() for path in (host.life / 'epochs' / str(EPOCH)).iterdir()
                   if path.name not in ('K1-TURN.json', 'TRAINING-JOB.json', 'TRAIN-DISPATCH-FF.json'))


def test_a_live_run_has_the_pay_hook_and_a_shadow_run_has_none(host: Host) -> None:
    state_id = host.seed(live=True)
    parts = R.build(host.args('run', live=True), host.env())
    assert parts.mode == MODE_LIVE and parts.config.live_armed and host.writers == 1
    settler = parts.settler
    assert settler is not None and settler.store.mode == MODE_LIVE
    assert settler.store.prefix == 'pit/batchctl/%s/' % RUN and settler.epochs == host.gate_state / 'epochs'
    assert settler.life_epochs == host.life / 'epochs' and settler.marks == parts.where.private / 'settled'
    assert {path.name for path in parts.where.state_dir.iterdir()} == {'controller', 'private'}
    assert (parts.where.controller / MARKERS[MODE_LIVE]).is_file()
    controller = parts.controller.open()
    try:
        step(controller)
        assert controller.status()['head']['state_id'] == state_id and controller.status()['mode'] == MODE_LIVE
    finally:
        controller.close()
    assert box_name(RUN, 1, 0, MODE_LIVE).startswith('affine-bctl-live-')
    # An unarmed live run cannot be built, whatever calls the builder.
    with pytest.raises(ShadowWriteError):
        R.build(host.args('run', '--genesis-state', state_id, live=True, arm=None), host.env())


def test_run_stops_on_its_event_and_settles_what_is_left(host: Host, capsys: Any) -> None:
    host.seed(live=True)
    capsys.readouterr()
    stop = threading.Event()
    done: List[int] = []
    worker = threading.Thread(target=lambda: done.append(R.run(
        host.args('run', '--settle-every-s', '0.05', live=True), host.env(), stop=stop)))
    worker.start()
    try:
        status = host.live_dir / 'controller' / 'status.json'
        for _ in range(400):
            if status.is_file():
                break
            stop.wait(0.05)
    finally:
        stop.set()
        worker.join(timeout=60.0)
    assert done == [0] and not worker.is_alive()
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    events = [line['event'] for line in lines]
    assert events[0] == 'started' and events[-1] == 'stopped'
    assert lines[0] == {'event': 'started', 'mode': MODE_LIVE, 'run_id': RUN, 'pay_hook': True,
                        'r2_prefix': 'pit/batchctl/%s/' % RUN}
    # The lock is free again.
    R.build(host.args('run', live=True), host.env()).controller.open().close()
