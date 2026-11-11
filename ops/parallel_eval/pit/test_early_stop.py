"""The RED-only early stop of the probe.

What these tests show:

* the flag is off wherever it is not set by hand, in shadow mode and in live mode;
* a stop is the official estimate of the finished tasks at the fixed check points, and nothing else;
* an early stop gives RED or no decision, never GREEN;
* without a stop, the decision is the one the full sample gives, flag on or off.

The scorer is the live host scorer ``ops/parallel_eval/live/pe_score.py`` with the official
``baseline_v1.estimate``, on a round in a temp directory. The box is the local box of ``probe_fixture``: the
pinned ``podctl.py`` and ``probe_worker.py`` write the completion stream and end the run.

This file is the only place that turns the flag on.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import random
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from unittest import mock

from ops.parallel_eval.live.pe_probe_plan import build_probe_plan
from ops.parallel_eval.pit import DEFAULT_PIT, PitConfig, load_pit_config
from ops.parallel_eval.pit import early_stop as ES
from ops.parallel_eval.pit.apply_live import LiveApplicator
from ops.parallel_eval.pit.box import ScriptedBoxIO
from ops.parallel_eval.pit.controller import Controller
from ops.parallel_eval.pit.decision_emit import build_decision_doc
from ops.parallel_eval.pit.pe_lane import LaneService
from ops.parallel_eval.pit.pe_rent import RentService
from ops.parallel_eval.pit.pe_score import call_live_host_score
from ops.parallel_eval.pit.probe_fixture import ENV_ID, BoxCase
from ops.parallel_eval.pit.signer import FixtureRootSigner
from ops.parallel_eval.pit.state import ItemRecord, ItemState

REPO = Path(__file__).resolve().parents[3]
LIVE = REPO / 'ops' / 'parallel_eval' / 'live'
GATE = REPO / 'miner' / 'validator' / 'gate'
LIVE_IMPORTS = ('pe_common', 'replace_pod', 'baseline_v1', 'prior_gate', 'probe_error_guard', 'heldout')
H0, M0 = 'a' * 64, 'b' * 64
EPOCH, RID, SEED = 10, 'round-1', 'p' * 64
ON = ES.EarlyStop(enabled=True)
# The files of the PIT package that may name the early stop. A config file, an install script or a test fixture
# that starts to name it fails ``test_nothing_else_names_the_flag``.
MAY_NAME_THE_FLAG = {'early_stop.py', 'test_early_stop.py', 'config.py', 'box.py', 'controller.py', 'decision_emit.py'}


def load_live_scorer(tmp: Path) -> Any:
    """The live host scorer as a private module copy that reads the round files under ``tmp``.

    Its imports are the repository files: the official ``baseline_v1``, ``prior_gate`` and the probe-error guard.
    ``HOME`` is a temp directory while it loads, so no path of a real host is on the import path.
    """
    home = tmp / 'home'
    home.mkdir(parents=True)
    paths = [str(LIVE), str(GATE), str(REPO / 'ops' / 'gate_probe_error_guard'), str(REPO / 'ops' / 'pods')]
    env = {'HOME': str(home), 'PODS_ROLES': str(REPO / 'ops' / 'pods' / 'roles.json')}
    with mock.patch.dict(os.environ, env), mock.patch.dict(sys.modules), \
            mock.patch.object(sys, 'path', paths + sys.path), \
            mock.patch.object(sys, 'dont_write_bytecode', sys.dont_write_bytecode):
        for name in LIVE_IMPORTS:
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location('pit_early_stop_live_pe_score', LIVE / 'pe_score.py')
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
    for module, source in ((scorer.BV, GATE / 'baseline_v1.py'), (scorer.PG, GATE / 'prior_gate.py')):
        assert Path(module.__file__).resolve() == source, module.__file__
    scorer.HELDOUT, scorer.FLAG = tmp / 'heldout', tmp / 'flag.json'
    return scorer


def _write(path: Path, doc: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(doc))


def stream_line(index: int, cell: Mapping[str, Any]) -> str:
    """The line ``probe_worker.py`` appends to ``grades.jsonl`` when one task is finished."""
    return json.dumps(dict(env_id=ENV_ID, index=index, **cell), sort_keys=True) + '\n'


def probe_result(cells: Mapping[int, Mapping[str, Any]], *, planned_tasks: int, **extra: Any) -> Dict[str, Any]:
    """The result ``probe_worker.py`` writes for these finished cells. It is partial when tasks are missing."""
    samples = sum(cell['n'] for cell in cells.values())
    return dict(
        version='sn120-env-probe-worker-v1', k=8, dtype='bfloat16', samples=samples, planned_samples=planned_tasks * 8,
        partial=len(cells) < planned_tasks, envs={ENV_ID: dict(per_task={str(i): dict(c) for i, c in cells.items()})},
        **extra)


class Round:
    """One gate round in the files the live scorer reads: the secret draw, the priors and the full head baseline."""

    def __init__(self, tmp: Path, *, math: int = 160, dapo: int = 80, seed: int = 7,
                 head_passes: Optional[int] = None) -> None:
        tmp.mkdir(parents=True)
        self.scorer = load_live_scorer(tmp)
        self.BV = self.scorer.BV
        self.rng = random.Random(seed)
        self.cfg = dict(self.BV.DEFAULTS, max_tokens=65536)
        self.ids = ['math:%d' % i for i in range(math)] + ['dapo:%d' % i for i in range(dapo)]
        self.index = {t: self.BV.probe_index(t, self.cfg) for t in self.ids}
        self.rate = {t: self.rng.uniform(0.2, 0.8) for t in self.ids}
        self.drawn = {t: dict(env='affine_math' if t.startswith('math') else 'dapo', pi=self.rng.uniform(0.02, 0.1))
                      for t in self.ids}
        self.pool = dict(affine_math=7496, dapo=4000)
        if head_passes is None:
            self.head = {t: self.cell(self.rate[t], 12) for t in self.ids}
        else:
            self.head = {t: dict(n=12, c=head_passes, w=12 - head_passes, u=0, b=0, x=0, tok=1200) for t in self.ids}
        self.dir = tmp / 'heldout' / RID
        self.dir.mkdir(parents=True)
        (tmp / 'flag.json').write_text('{}')
        _write(self.dir / 'ROUND.private.json', dict(id=RID, cfg=self.cfg, ids=self.ids, pool=dict(per_env=self.pool),
                                                     seed='ab' * 32, n=len(self.ids)))
        _write(self.dir / 'PI.private.json', dict(env={t: d['env'] for t, d in self.drawn.items()},
                                                  pi={t: d['pi'] for t, d in self.drawn.items()}))
        self.baseline = dict(per_task=self.head, max_tokens=65536, rollouts_per_task=12, settings_hash='s' * 64,
                             done_at_cr='now')
        _write(self.dir / 'BASELINE.private.json', self.baseline)

    @property
    def indices(self) -> List[int]:
        return [self.index[t] for t in self.ids]

    def cell(self, rate: float, rollouts: int, *, errors: int = 0, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        rng = rng or self.rng
        passes = sum(rng.random() < rate for _ in range(rollouts - errors))
        return dict(n=rollouts, c=passes, w=rollouts - errors - passes, u=0, b=0, x=errors, tok=100 * rollouts)

    def candidate(self, shift: float, *, seed: int = 0, errors: int = 0) -> Dict[int, Dict[str, Any]]:
        """Candidate cells {probe index: cell}. The pass rate of every task is the head's rate plus ``shift``."""
        rng = random.Random('candidate:%r:%d' % (shift, seed))
        return {self.index[t]: self.cell(min(1.0, max(0.0, self.rate[t] + shift)), 8, errors=errors, rng=rng)
                for t in self.ids}

    def completion_order(self, seed: int = 0) -> List[int]:
        """The order in which a probe finishes the tasks."""
        order = self.indices
        random.Random('order:%d' % seed).shuffle(order)
        return order

    def score(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        """The scorer the controller uses: the live ``pe_score.score`` through the PIT adapter."""
        return call_live_host_score(self.scorer.score, epoch=EPOCH, round_id=RID, result=result)

    def official(self, cells: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
        """``baseline_v1.estimate`` of these candidate cells against the full head baseline, called directly."""
        cfg, BV = self.cfg, self.BV
        return BV.estimate(
            self.drawn, self.head, {BV.task_of(i, cfg): c for i, c in cells.items()}, self.pool,
            shares=cfg['shares'], tau=float(cfg['tau_run_separate']), sigma_run_floor=float(cfg['sigma_run_floor']),
            bar_z=float(cfg['bar_z']), z_inclusive=bool(cfg['z_inclusive']), cap_max=float(cfg['cap_max']),
            min_scored=int(cfg['min_scored']), min_cover=float(cfg['min_cover']))

    def first_stop(self, cells: Mapping[int, Mapping[str, Any]], order: List[int],
                   settings: ES.EarlyStop = ON) -> Optional[int]:
        """The rule, written out: the first check point where the official z is below the bar minus the margin."""
        for tasks in range(settings.min_tasks, len(order) + 1, settings.check_every):
            z = self.official({i: cells[i] for i in order[:tasks]})['evaluated']['z']
            if z < float(self.cfg['bar_z']) - settings.margin:
                return tasks
        return None


class RoundCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.round = Round(self.tmp / 'gate')

    def watch(self, cells: Mapping[int, Mapping[str, Any]], order: List[int], settings: ES.EarlyStop = ON,
              *, shown: Optional[int] = None) -> ES.Monitor:
        """A monitor on the stream of a probe that finishes ``cells`` in ``order``. ``self.shown`` lines are written."""
        self.lines = [stream_line(i, cells[i]) for i in order]
        self.shown = len(order) if shown is None else shown
        self.scored: List[int] = []          # tasks in each result that reached the scorer
        self.cancels = 0
        self.now = 0.0

        def score(result: Dict[str, Any]) -> Any:
            self.scored.append(sum(len(env['per_task']) for env in result['envs'].values()))
            return self.round.score(result)

        def cancel() -> None:
            self.cancels += 1

        return ES.Monitor(settings, read=lambda done: ''.join(self.lines[done:self.shown]), score=score,
                          cancel=cancel, tasks_total=len(order), clock=lambda: self.now)

    def look(self, monitor: ES.Monitor, shown: Optional[int] = None) -> None:
        """The probe has written ``shown`` lines, and the host looks at the stream."""
        if shown is not None:
            self.shown = shown
        self.now += monitor.settings.poll_s
        monitor.poll()

    def stopped_probe(self, cells: Mapping[int, Mapping[str, Any]], order: List[int], settings: ES.EarlyStop = ON,
                      *, extra_tasks: int = 13) -> Dict[str, Any]:
        """The partial result of a probe that the early stop ended, with its stop record.

        A real monitor watched the stream. When the stop ended the run, the worker had ``extra_tasks`` more tasks
        finished than the stop looked at.
        """
        monitor = self.watch(cells, order, settings)
        self.look(monitor)
        self.assertIsNotNone(monitor.stop, 'this probe was expected to stop')
        finished = order[:monitor.stop['tasks'] + extra_tasks]
        result = probe_result({i: cells[i] for i in finished}, planned_tasks=len(order), stop_reason='finalize')
        monitor.settle(result)
        return result


class ControllerMixin:
    """One item through a real controller. Needs ``self.tmp`` and ``self.round``."""

    def controller(self, config: PitConfig, **kwargs: Any) -> Controller:
        root = self.tmp / ('host-%d' % len(list(self.tmp.glob('host-*'))))
        pe, life = root / 'pe', root / 'life'
        (pe / 'pit').mkdir(parents=True)
        (life / 'pending').mkdir(parents=True)
        (root / 'fp32').mkdir()
        (root / 'K1-TURNS.jsonl').write_text('')
        epoch_dir = life / 'epochs' / str(EPOCH)
        epoch_dir.mkdir(parents=True)
        # The live training job of the epoch names the same HEAD, so the item is not skipped as stale.
        _write(epoch_dir / 'TRAINING-JOB.json', {'payload': {'manifest': {'payload': {'checkpoint': {'id': H0}}}}})
        _write(epoch_dir / 'TRAIN-DISPATCH-FF.json', {'signature': 's', 'signer': 'ROOT', 'payload': {}})
        env = mock.patch.dict(os.environ, {
            'PE_STATE': str(pe), 'PIT_STATE': str(pe / 'pit'), 'LIFE_ROOT': str(life), 'PE_LIFE': str(life),
            'PE_LIFE_STATE': str(life), 'K1_TURNS': str(root / 'K1-TURNS.jsonl'),
            'PIT_FP32_MASTER_INDEX': str(root / 'fp32'),
        })
        env.start()
        self.addCleanup(env.stop)
        clock = {'t': 100.0}

        def now() -> float:
            clock['t'] += 1.0
            return clock['t']

        def train(**_: Any) -> Dict[str, Any]:
            return {'candidate_id': 'c' * 64, 'candidate_path': '/tmp/c', 'candidate_master_path': '/tmp/m',
                    'candidate_master_digest': 'd' * 64, 'candidate_sha256': 'c' * 64,
                    'used_fp32_bootstrap_override': False}

        self.score_calls = 0

        def score(epoch: int, rid: str, result: Mapping[str, Any], flag: Any = None) -> Any:
            self.score_calls += 1
            return self.round.scorer.score(epoch, rid, result, flag)

        ctl = Controller(
            config=config, rent=RentService(max_boxes=config.max_boxes), lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(on_train=train), now_fn=now, score_fn=score,
            round_id_fn=lambda epoch: RID, plan_seed_fn=lambda epoch: SEED, **kwargs)
        ctl.set_head(H0, epoch=5, master_digest=M0)
        return ctl

    def scored_item(self, ctl: Controller) -> ItemRecord:
        """A new item of ``ctl`` whose box has returned a probe result. Its decision is the next step."""
        ctl.enqueue(epoch=EPOCH, registration_index=len(ctl.items) + 1, hotkey_prefix='5One')
        for _ in range(10):
            ctl.tick()
            if ctl.items[-1].state is ItemState.SCORED:
                return ctl.items[-1]
        raise AssertionError('the item did not reach its decision: %s' % ctl.items[-1].state)

    def decide(self, config: PitConfig, box_result: Mapping[str, Any], **kwargs: Any) -> Any:
        """-> (controller, item) after the item's box returned ``box_result`` and the item was released."""
        ctl = self.controller(config, **kwargs)
        item = self.scored_item(ctl)
        item.meta['probe_bundle']['probe_result'] = copy.deepcopy(dict(box_result))
        for _ in range(20):
            ctl.tick()
            if ctl.items[0].state is ItemState.RELEASED:
                return ctl, ctl.items[0]
        raise AssertionError('the item was not released: %s' % ctl.items[0].state)


def shadow(**pit: Any) -> PitConfig:
    return PitConfig(enabled=True, mode='shadow', max_boxes=1, **pit)


def outcome(ctl: Controller, item: ItemRecord) -> Dict[str, Any]:
    """Everything a decision leaves behind, without the random item id."""
    return dict(decision=item.decision, z=item.z, paid=item.paid, fresh=item.fresh, merge_blocked=item.merge_blocked,
                scored=item.meta.get('scored'), head=ctl.current_head,
                log=[entry.as_dict() for entry in ctl.decisions.entries])


class TestSettings(unittest.TestCase):
    def test_the_flag_is_off_wherever_it_is_not_set(self):
        configs = {
            'EarlyStop()': ES.EarlyStop(),
            'PitConfig()': PitConfig().early_stop,
            'DEFAULT_PIT': DEFAULT_PIT.early_stop,
            'no pit section': load_pit_config({}).early_stop,
            'shadow': load_pit_config({'pit': {'enabled': True, 'mode': 'shadow'}}).early_stop,
            'live': load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}}).early_stop,
            'null': load_pit_config({'pit': {'early_stop': None}}).early_stop,
            'empty': load_pit_config({'pit': {'early_stop': {}}}).early_stop,
            'other settings only': load_pit_config({'pit': {'early_stop': {'margin': 3, 'min_tasks': 150}}}).early_stop,
        }
        for name, settings in configs.items():
            self.assertIs(settings.enabled, False, name)
        self.assertEqual((ON.margin, ON.min_tasks, ON.check_every), (2.0, 100, 20))
        self.assertIs(json.loads(json.dumps(PitConfig().as_dict()))['early_stop']['enabled'], False)

    def test_only_the_explicit_flag_turns_it_on_in_shadow_and_in_live(self):
        for mode in ({'mode': 'shadow'}, {'mode': 'live', 'cutover': True}):
            cfg = load_pit_config({'pit': dict(mode, enabled=True, early_stop={'enabled': True})})
            self.assertEqual(cfg.early_stop, ON, mode)
        cfg = load_pit_config({'pit': {'early_stop': {
            'enabled': True, 'margin': 3, 'min_tasks': 150, 'check_every': 10, 'poll_s': 5}}})
        self.assertEqual(json.loads(json.dumps(cfg.as_dict()))['early_stop'],
                         {'enabled': True, 'margin': 3, 'min_tasks': 150, 'check_every': 10, 'poll_s': 5})
        self.assertEqual([n for n in range(140, 200) if cfg.early_stop.is_due(n)], [150, 160, 170, 180, 190])
        # A config that is built by hand goes through the same check.
        self.assertEqual(PitConfig(early_stop={'enabled': True}).early_stop, ON)

    def test_settings_that_are_not_valid_are_refused(self):
        bad = (
            True, 'on', 1, ['enabled'], {'enable': True},
            {'enabled': 'true'}, {'enabled': 1}, {'enabled': None},      # only the JSON value true is the flag
            {'enabled': True, 'margin': -0.5},                           # would stop a candidate above the bar
            {'enabled': True, 'margin': float('nan')}, {'enabled': True, 'margin': '2'}, {'enabled': True, 'margin': True},
            {'enabled': True, 'min_tasks': 0}, {'enabled': True, 'min_tasks': 99.5}, {'enabled': True, 'min_tasks': True},
            {'enabled': True, 'check_every': 0}, {'enabled': True, 'poll_s': 0}, {'enabled': True, 'poll_s': 'fast'},
        )
        for raw in bad:
            with self.assertRaises(ValueError, msg=repr(raw)):
                load_pit_config({'pit': {'early_stop': raw}})
            with self.assertRaises(ValueError, msg=repr(raw)):
                PitConfig(early_stop=raw)

    def test_nothing_else_names_the_flag(self):
        """No default config, install script or other test fixture can turn the flag on, because none names it."""
        config_suffixes = {'.json', '.sh', '.service', '.timer', '.toml', '.yaml', '.yml', '.env', '.cfg', '.ini',
                           '.conf', '.txt'}
        package = REPO / 'ops' / 'parallel_eval'
        named = set()
        for top in ('ops', 'systemd', 'operator_profiles', 'tools', 'miner', 'bench'):
            for path in (REPO / top).rglob('*'):
                parts = path.relative_to(REPO).parts
                if not path.is_file() or '__pycache__' in parts or 'vendor' in parts:
                    continue
                if package not in path.parents and path.suffix not in config_suffixes:
                    continue
                try:
                    text = path.read_text()
                except (UnicodeDecodeError, OSError):
                    continue
                if 'early_stop' in text:
                    named.add('/'.join(parts))
        self.assertEqual(named, {'ops/parallel_eval/pit/' + name for name in MAY_NAME_THE_FLAG})


class TestStopVerdict(unittest.TestCase):
    @staticmethod
    def estimate(decision: Any = 'red', z: Any = -1.5, n: Any = 120, **top: Any) -> Dict[str, Any]:
        return dict(dict(decision=decision, evaluated=dict(z=z, n=n), z_min=1.0), **top)

    def test_a_stop_is_a_red_estimate_below_the_bar_minus_the_margin(self):
        self.assertEqual(ES.stop_verdict(self.estimate(), ON),
                         dict(z=-1.5, bar_z=1.0, margin=2.0, threshold=-1.0, scored_tasks=120))
        self.assertIsNone(ES.stop_verdict(self.estimate(z=-1.0), ON))            # -1.0 is not below -1.0
        self.assertIsNotNone(ES.stop_verdict(self.estimate(z=-1.001), ON))
        wider = ES.EarlyStop(enabled=True, margin=3)                              # m = 3 stops below -2
        self.assertIsNone(ES.stop_verdict(self.estimate(z=-1.9), wider))
        self.assertEqual(ES.stop_verdict(self.estimate(z=-2.1), wider)['threshold'], -2.0)
        # The threshold follows the bar the estimator used.
        self.assertEqual(ES.stop_verdict(self.estimate(z=-0.6, z_min=1.5), ON)['threshold'], -0.5)
        self.assertIsNone(ES.stop_verdict(self.estimate(z=-0.4, z_min=1.5), ON))

    def test_every_doubt_is_no_stop(self):
        doubts = (
            None, {}, 'red', ['red'],
            self.estimate(decision='green', z=-9.0), self.estimate(decision='GREEN', z=-9.0),
            self.estimate(decision=None, z=-9.0, guard='candidate-probe-errors'),
            self.estimate(decision='pending', z=-9.0), self.estimate(decision='no_decision', z=-9.0),
            self.estimate(z=None), self.estimate(z=float('nan')), self.estimate(z=float('-inf')),
            self.estimate(z='-9'), self.estimate(z=True),
            self.estimate(n=99), self.estimate(n=None), self.estimate(n='120'),
            self.estimate(z_min=None), self.estimate(z_min='1.0'), self.estimate(z_min=float('nan')),
            dict(decision='red', z_min=1.0), dict(decision='red', z_min=1.0, evaluated='z=-9'),
        )
        for scored in doubts:
            self.assertIsNone(ES.stop_verdict(scored, ON), repr(scored))


class TestMonitor(RoundCase):
    def test_checks_are_at_100_tasks_and_then_every_20(self):
        cells, order = self.round.candidate(0.2), self.round.completion_order()
        monitor = self.watch(cells, order, shown=99)
        self.look(monitor)
        self.assertEqual(self.scored, [])
        self.look(monitor, 100)
        self.assertEqual(self.scored, [100])
        self.look(monitor, 119)
        self.assertEqual(self.scored, [100])
        self.look(monitor, 120)
        self.assertEqual(self.scored, [100, 120])
        self.look(monitor, 187)                               # many lines in one look: every check point is checked
        self.assertEqual(self.scored, [100, 120, 140, 160, 180])
        self.shown = 240
        monitor.poll()                                        # less than poll_s after the last look: nothing is read
        self.assertEqual(self.scored, [100, 120, 140, 160, 180])
        self.look(monitor)
        self.assertEqual(self.scored, [100, 120, 140, 160, 180, 200, 220, 240])
        self.assertIsNone(monitor.stop)
        self.assertEqual((self.cancels, monitor.checks, monitor.failures), (0, 8, 0))

    def test_the_stop_is_the_first_check_point_where_the_official_z_is_below_minus_one(self):
        cells, order = self.round.candidate(-0.015), self.round.completion_order()
        self.assertEqual(self.round.first_stop(cells, order), 160)      # z is above -1 at 100, 120 and 140 tasks
        official = self.round.official({i: cells[i] for i in order[:160]})

        monitor = self.watch(cells, order)
        self.look(monitor)
        self.assertEqual(self.scored, [100, 120, 140, 160])
        self.assertEqual(self.cancels, 1)
        self.assertEqual(monitor.stop, dict(
            version=ES.VERSION, triggered=True, tasks=160, tasks_total=240, fraction=160 / 240,
            z=official['evaluated']['z'], bar_z=1.0, margin=2.0, threshold=-1.0, scored_tasks=160,
            min_tasks=100, check_every=20, checks=4, order=[[ENV_ID, i] for i in order[:160]]))
        self.assertLess(official['evaluated']['z'], -1.0)
        self.look(monitor)                                    # after the stop: no more checks, no second cancel
        self.assertEqual((self.scored, self.cancels), ([100, 120, 140, 160], 1))

        # When the host looks, and how many lines one look returns, does not change the stop.
        for step in (1, 7, 64):
            other = self.watch(cells, order, shown=0)
            for shown in range(step, 241, step):
                self.look(other, shown)
            self.assertEqual(other.stop, monitor.stop, step)
            self.assertEqual(self.cancels, 1)

    def test_a_candidate_that_is_far_below_the_head_stops_at_the_first_check(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        monitor = self.watch(cells, order, shown=130)
        self.look(monitor)
        self.assertEqual((monitor.stop['tasks'], monitor.stop['fraction'], self.scored), (100, 100 / 240, [100]))

    def test_no_stop_when_the_candidate_is_not_below_the_threshold(self):
        order = self.round.completion_order()
        for shift, decision in ((0.2, 'green'), (0.0, 'red')):
            cells = self.round.candidate(shift)
            self.assertIsNone(self.round.first_stop(cells, order))
            self.assertEqual(self.round.official(cells)['decision'], decision)
            monitor = self.watch(cells, order)
            self.look(monitor)
            self.assertIsNone(monitor.stop)
            self.assertEqual((self.cancels, monitor.checks), (0, 8))
            full = probe_result(cells, planned_tasks=240)
            before = copy.deepcopy(full)
            monitor.settle(full)
            self.assertEqual(full, before)

    def test_the_margin_is_a_setting(self):
        order = self.round.completion_order()
        no_margin = ES.EarlyStop(enabled=True, margin=0.0)
        even = self.round.candidate(0.0)                      # z is between -1 and 1: a stop only without a margin
        self.assertEqual((self.round.first_stop(even, order), self.round.first_stop(even, order, no_margin)), (None, 100))
        monitor = self.watch(even, order, no_margin)
        self.look(monitor)
        self.assertEqual((monitor.stop['tasks'], monitor.stop['threshold']), (100, 1.0))
        better = self.round.candidate(0.2)                    # above the bar at every check: no stop, margin or not
        monitor = self.watch(better, order, no_margin)
        self.look(monitor)
        self.assertIsNone(monitor.stop)

    def test_no_stop_without_an_official_red_estimate(self):
        order = self.round.completion_order()
        # More than 1% of the rollouts failed in the grader: the live guard gives no decision, whatever the passes.
        monitor = self.watch(self.round.candidate(-0.2, errors=1), order)
        self.look(monitor)
        self.assertEqual(self.round.score(probe_result(self.round.candidate(-0.2, errors=1), planned_tasks=240))['guard'],
                         'candidate-probe-errors')
        self.assertIsNone(monitor.stop)
        self.assertEqual((self.cancels, monitor.checks), (0, 8))

    def test_a_scorer_that_fails_or_waits_never_stops_and_never_raises(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        baseline = self.round.dir / 'BASELINE.private.json'
        # The head baseline of the round is not ready: the check waits.
        _write(baseline, {})
        monitor = self.watch(cells, order)
        self.look(monitor)
        self.assertEqual((monitor.stop, monitor.checks, monitor.failures, self.cancels), (None, 0, 0, 0))
        # The scorer fails: the monitor keeps the failure and the probe goes on.
        _write(baseline, dict(self.round.baseline, max_tokens=1))
        self.look(monitor)
        self.assertEqual((monitor.stop, monitor.checks, monitor.failures, self.cancels), (None, 0, 1, 0))
        self.assertIn('baseline cap', monitor.last_failure)
        # The same check runs again when the scorer answers.
        _write(baseline, self.round.baseline)
        self.look(monitor)
        self.assertEqual((monitor.stop['tasks'], monitor.checks, self.cancels), (100, 1, 1))

    def test_a_cancel_that_fails_is_tried_again(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        monitor = self.watch(cells, order)
        answers = [RuntimeError('the box did not answer'), None]

        def cancel() -> None:
            self.cancels += 1
            answer = answers.pop(0)
            if answer is not None:
                raise answer

        monitor._cancel = cancel
        self.look(monitor)
        self.assertEqual((monitor.stop['tasks'], monitor.cancelled, self.cancels), (100, False, 1))
        self.look(monitor)
        self.assertEqual((monitor.cancelled, self.cancels, self.scored), (True, 2, [100]))
        self.look(monitor)
        self.assertEqual(self.cancels, 2)

    def test_only_whole_lines_of_finished_tasks_count(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        monitor = self.watch(cells, order, shown=99)
        whole = self.lines[99]
        self.lines[99] = whole[:25]                           # the worker is in the middle of line 100
        self.look(monitor, 100)
        self.assertEqual((self.scored, monitor.state()['lines']), ([], 99))
        self.lines[99] = whole
        self.look(monitor)
        self.assertEqual((self.scored, monitor.stop['tasks']), ([100], 100))

        # A second line for a task is not a new task.
        monitor = self.watch(cells, order, shown=100)
        self.lines[50] = self.lines[10]
        self.look(monitor)
        self.assertEqual((self.scored, monitor.state()['tasks']), ([], 99))

        # A line that is not a finished task: the stream cannot be trusted, so this probe is never stopped.
        for junk in ('{"env_id": "affine_gatepool", "index": 3}\n', 'not json\n', '[1, 2]\n',
                     json.dumps(dict(env_id=ENV_ID, index='3', n=8, c=0)) + '\n'):
            monitor = self.watch(cells, order)
            self.lines[40] = junk
            self.look(monitor)
            self.assertEqual((monitor.stop, self.scored, self.cancels), (None, [], 0), junk)
            self.assertIn('line 41', monitor.broken)

    def test_settle_gives_only_a_partial_result_its_stop_record(self):
        cells, order = self.round.candidate(-0.015), self.round.completion_order()
        watching = self.watch(cells, order)
        self.look(watching)
        partial = probe_result({i: cells[i] for i in order[:171]}, planned_tasks=240)

        # The host that watched: its record.
        seen = copy.deepcopy(partial)
        watching.settle(seen)
        self.assertEqual(seen.pop('early_stop'), watching.stop)
        self.assertEqual(seen, partial)

        # A host that restarted reads the stream again and finds the same stop. It does not cancel anything.
        restarted = self.watch(cells, order, shown=171)
        again = copy.deepcopy(partial)
        restarted.settle(again)
        self.assertEqual(again['early_stop'], watching.stop)
        self.assertEqual(self.cancels, 0)

        # A partial result without a stop in its stream (a stalled probe) gets no record.
        stalled = self.watch(cells, order, shown=150)
        short = probe_result({i: cells[i] for i in order[:150]}, planned_tasks=240)
        stalled.settle(short)
        self.assertNotIn('early_stop', short)

        # The probe finished before the stop could end it: the result is complete and stays as it is.
        full = probe_result(cells, planned_tasks=240)
        watching.settle(full)
        self.assertNotIn('early_stop', full)


class TestDecision(ControllerMixin, RoundCase):
    def test_an_early_stop_is_red_and_the_fraction_is_recorded(self):
        cells, order = self.round.candidate(-0.015), self.round.completion_order()
        box_result = self.stopped_probe(cells, order)
        self.assertEqual((box_result['early_stop']['tasks'], len(box_result['envs'][ENV_ID]['per_task'])), (160, 173))
        ctl, item = self.decide(shadow(early_stop=ON), box_result)

        official = self.round.official({i: cells[i] for i in order[:160]})
        self.assertEqual((item.decision, item.paid, item.z), ('red', False, official['evaluated']['z']))
        scored = item.meta['scored']
        # The estimate is of the 160 tasks of the stop, not of the 173 that were finished when the run ended.
        self.assertEqual(scored['evaluated'], official['evaluated'])
        self.assertIs(scored['early_stop'], True)
        self.assertEqual(scored['early_stop_fraction'], 160 / 240)
        self.assertEqual(scored['early_stop_detail'], dict(
            version=ES.VERSION, tasks=160, tasks_total=240, fraction=160 / 240, z=official['evaluated']['z'],
            bar_z=1.0, margin=2.0, threshold=-1.0, min_tasks=100, check_every=20))
        self.assertEqual(scored['reasons'][:-1], official['reasons'])
        self.assertRegex(scored['reasons'][-1], r'^early_stop: z -1\.\d+ < -1\.00 after 160 of 240 tasks$')
        self.assertEqual([(e.decision, e.paid, e.shadow, e.extra) for e in ctl.decisions.entries],
                         [('red', False, True, {'early_stop': True, 'early_stop_fraction': 160 / 240})])
        self.assertEqual(ctl.current_head, H0)

    def test_with_the_flag_off_a_stopped_probe_is_an_incomplete_probe_as_before(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        box_result = self.stopped_probe(cells, order)
        ctl, item = self.decide(shadow(), box_result)
        self.assertEqual((item.decision, item.paid, self.score_calls), ('no_decision', False, 0))
        self.assertEqual(ctl.decisions.entries, [])
        self.assertNotIn('scored', item.meta)

    def test_a_partial_probe_without_a_stop_is_an_incomplete_probe_flag_on_or_off(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        stalled = probe_result({i: cells[i] for i in order[:150]}, planned_tasks=240, stop_reason='finalize')
        for config in (shadow(), shadow(early_stop=ON)):
            ctl, item = self.decide(config, stalled)
            self.assertEqual((item.decision, self.score_calls, ctl.decisions.entries), ('no_decision', 0, []))

    def test_a_full_probe_decides_the_same_with_the_flag_on_or_off(self):
        order = self.round.completion_order()
        for shift, want in ((0.2, 'green_fresh'), (0.0, 'red'), (-0.2, 'red')):
            cells = self.round.candidate(shift)
            full = probe_result(cells, planned_tasks=240)
            # A stop record on a complete result is not read.
            marked = dict(full, early_stop=dict(self.stopped_probe(self.round.candidate(-0.2), order)['early_stop']))
            off = outcome(*self.decide(shadow(), full))
            self.assertEqual((off['decision'], off['z']), (want, self.round.official(cells)['evaluated']['z']), shift)
            self.assertEqual(len(off['log']), 1)
            self.assertNotIn('early_stop', off['scored'])
            self.assertEqual(off['log'][0]['extra'], {})
            self.assertEqual(outcome(*self.decide(shadow(early_stop=ON), full)), off, shift)
            self.assertEqual(outcome(*self.decide(shadow(early_stop=ON), marked)), off, shift)
            self.assertEqual(outcome(*self.decide(shadow(), marked)), off, shift)

    def test_an_early_stop_never_gives_green(self):
        """Partial results with a stop record, true or forged, under several settings: RED or no decision."""
        order = self.round.completion_order()
        rng = random.Random(11)
        seen: Dict[str, int] = {}
        for settings in (ON, ES.EarlyStop(enabled=True, margin=0.0), ES.EarlyStop(enabled=True, min_tasks=20, check_every=5)):
            ctl = self.controller(shadow(early_stop=settings))
            template = self.scored_item(ctl)
            check_points = [n for n in range(1, 241) if settings.is_due(n)]
            for case in range(60):
                shift = rng.choice((-0.3, -0.05, -0.02, 0.0, 0.02, 0.05, 0.3))
                cells = self.round.candidate(shift, seed=case)
                tasks = rng.choice(check_points + [240, 240])            # 240: the record names every task
                finished = min(240, tasks + rng.randrange(0, 40))
                record = dict(version=ES.VERSION, triggered=True, tasks=tasks, tasks_total=240, fraction=tasks / 240,
                              z=-99.0, bar_z=1.0, margin=settings.margin, threshold=1.0 - settings.margin,
                              order=[[ENV_ID, i] for i in order[:tasks]])
                box_result = probe_result({i: cells[i] for i in order[:finished]}, planned_tasks=240, early_stop=record)
                box_result['partial'] = True
                item = copy.deepcopy(template)
                item.meta['probe_bundle']['probe_result'] = box_result
                decided, _ = ctl._decide(item)

                official = self.round.official({i: cells[i] for i in order[:tasks]})
                below = official['evaluated']['z'] < 1.0 - settings.margin and tasks >= settings.min_tasks
                self.assertEqual(decided.decision, 'red' if below else 'no_decision', (settings, case))
                self.assertFalse(decided.paid)
                if below:
                    self.assertEqual(official['decision'], 'red')
                seen[decided.decision] = seen.get(decided.decision, 0) + 1
                if official['decision'] == 'green':
                    seen['official green'] = seen.get('official green', 0) + 1
            self.assertEqual(ctl.current_head, H0)
            self.assertTrue(all(entry.decision == 'red' and not entry.paid for entry in ctl.decisions.entries))
        # Both outcomes were reached, and some records named cells whose official estimate is GREEN.
        self.assertEqual(sorted(seen), ['no_decision', 'official green', 'red'])
        self.assertGreater(min(seen.values()), 5, seen)

    def test_a_stop_record_that_does_not_fit_is_an_incomplete_probe(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        good = self.stopped_probe(cells, order)
        self.assertEqual(self.decide(shadow(early_stop=ON), good)[1].decision, 'red')

        def changed(**fields: Any) -> Dict[str, Any]:
            return dict(good, early_stop=dict(good['early_stop'], **fields))

        missing_cell = copy.deepcopy(good)
        del missing_cell['envs'][ENV_ID]['per_task'][str(order[5])]
        records = dict(
            not_triggered=changed(triggered=False), triggered_is_not_true=changed(triggered='yes'),
            no_order=changed(order=None), order_not_at_a_check_point=changed(order=good['early_stop']['order'][:99]),
            task_named_twice=changed(order=good['early_stop']['order'][:99] + good['early_stop']['order'][:1]),
            total_too_small=changed(tasks_total=50), no_total=changed(tasks_total=None),
            bad_entry=changed(order=[[ENV_ID, str(i)] for i in order[:100]]),
            cell_not_in_the_result=missing_cell, record_is_not_a_mapping=dict(good, early_stop=True),
        )
        for name, box_result in records.items():
            ctl, item = self.decide(shadow(early_stop=ON), box_result)
            self.assertEqual((item.decision, ctl.decisions.entries), ('no_decision', []), name)
        # Settings that changed after the stop: 100 tasks is no check point of these settings.
        late = shadow(early_stop=ES.EarlyStop(enabled=True, min_tasks=150))
        self.assertEqual(self.decide(late, good)[1].decision, 'no_decision')

    def test_the_shadow_comparison_path_takes_the_same_stop(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        box_result = self.stopped_probe(cells, order)
        live = {EPOCH: {'decision': 'red', 'candidate': 'f' * 64, 'z': -3.0}}
        ctl, item = self.decide(shadow(early_stop=ON), box_result, live_decision_lookup=live.get)
        self.assertEqual((item.decision, item.meta['scored']['early_stop_fraction']), ('red', 100 / 240))
        self.assertEqual(item.meta['shadow_comparison']['box_candidate'], 'c' * 64)

    def test_live_mode_takes_the_stop_only_with_the_flag_and_writes_it_on_the_decision(self):
        cells, order = self.round.candidate(-0.2), self.round.completion_order()
        box_result = self.stopped_probe(cells, order)

        def live(**pit: Any) -> Any:
            config = PitConfig(enabled=True, mode='live', max_boxes=5, cutover=True, master_upload='after_green', **pit)
            applied: List[Any] = []
            applicator = LiveApplicator(
                config=config, accept_pit_decision=lambda item: applied.append(item) or True,
                verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
                destroy_state=lambda **k: {'ok': True}, upload_master=lambda **k: {'ok': True, 'complete': True},
                upload_candidate=lambda **k: {'ok': True}, checkpoint_signer=FixtureRootSigner())
            ctl, item = self.decide(config, box_result, live_applicator=applicator,
                                    destroy_box_state_fn=lambda **k: {'ok': True})
            self.assertEqual(applied, [])
            return ctl, item

        ctl, item = live()
        self.assertEqual((item.decision, self.score_calls), ('no_decision', 0))

        ctl, item = live(early_stop=ON)
        self.assertEqual((item.decision, item.paid, ctl.current_head), ('red', False, H0))
        self.assertEqual([(e.decision, e.paid, e.shadow, e.extra) for e in ctl.decisions.entries],
                         [('red', False, False, {'early_stop': True, 'early_stop_fraction': 100 / 240})])
        doc = build_decision_doc(item, scored=item.meta['scored'], mode='live')
        self.assertEqual((doc['decision'], doc['paid'], doc['red_credit']), ('red', False, 'zero'))
        self.assertIs(doc['parallel_eval']['early_stop'], True)
        self.assertEqual(doc['parallel_eval']['early_stop_fraction'], 100 / 240)
        self.assertEqual(doc['parallel_eval']['early_stop_detail'], item.meta['scored']['early_stop_detail'])
        self.assertIn('early_stop: z ', doc['reasons'][-1])
        # What the gate writer checks before it writes a decision of an open round: no task of the draw is named.
        self.round.BV.assert_public(dict(evaluated=doc['evaluated'], reasons=doc['reasons'], heldout=doc['heldout']), True)
        self.round.BV.assert_public(doc['parallel_eval'], True)
        # A decision that is not an early stop has no such fields.
        plain = build_decision_doc(item, scored={k: v for k, v in item.meta['scored'].items() if 'early_stop' not in k},
                                   mode='live')
        self.assertFalse([key for key in plain['parallel_eval'] if 'early_stop' in key])


class TestOnTheBox(ControllerMixin, BoxCase):
    """The eval stage of an item on the local box, through ``BoxRunner``.

    The pinned ``probe_worker.py`` writes the completion stream, the host reads it, and the stop ends the run
    with the pinned ``podctl.py finalize-run``.
    """

    GPUS = 4
    FAST = ES.EarlyStop(enabled=True, min_tasks=10, check_every=5, poll_s=0.2)

    def setUp(self) -> None:
        super().setUp()
        self.commands: List[str] = []       # every command the host sends to the box
        self.host_pod = self.pod()
        send = self.host_pod.run

        def run(cmd: str, *args: Any, **kwargs: Any) -> str:
            self.commands.append(cmd)
            return send(cmd, *args, **kwargs)

        self.host_pod.run = run

    def ended_runs(self) -> int:
        """How many times the host ran ``podctl.py finalize-run`` on the box."""
        podctl = shlex.quote(str(self.fb.root / 'podctl.py'))
        return sum(command.startswith('python3 %s finalize-run ' % podctl) for command in self.commands)

    def stream_reads(self) -> int:
        """How many times the host read the completion stream of the run."""
        stream = shlex.quote(str(self.run_dir() / ES.STREAM_NAME))
        return sum(command.startswith('tail ') and stream in command for command in self.commands)

    def run_dir(self) -> Path:
        return self.fb.root / 'runs' / build_probe_plan(self.cid, [], SEED)['run_name']

    def new_round(self, head_passes: int, tasks: int = 60) -> None:
        """A round of ``tasks`` tasks whose head passed ``head_passes`` of 12 rollouts of every task."""
        self.round = Round(self.tmp / 'gate', math=tasks * 2 // 3, dapo=tasks - tasks * 2 // 3, head_passes=head_passes)
        self.scored: List[int] = []

    def score(self, result: Dict[str, Any]) -> Any:
        self.scored.append(len(result['envs'][ENV_ID]['per_task']))
        return self.round.score(result)

    def host(self, settings: Optional[ES.EarlyStop]) -> Dict[str, Any]:
        """What a host with these early-stop settings gives the box runner. ``None`` is a host without the flag.

        The scorer is always there, so a host without the flag shows that the flag alone decides.
        """
        return dict(config=shadow() if settings is None else shadow(early_stop=settings), indices=self.round.indices,
                    pod=self.host_pod, early_stop_score=self.score)

    def probe(self, settings: Optional[ES.EarlyStop]) -> Dict[str, Any]:
        """-> the probe result of the eval stage of an item whose candidate is on the box."""
        probe, record = self.eval_stage(**self.host(settings))
        self.assertIs(record.state, ItemState.RESULT)
        return probe

    def test_the_stop_ends_the_run_on_the_box_and_the_item_is_red(self):
        self.new_round(head_passes=11)                        # the head passes 92%: this candidate is far below
        self.fb.control(delay_ms=4000)                        # slow answers: the probe is still running at the check
        probe = self.probe(self.FAST)

        stream = [json.loads(line) for line in (self.run_dir() / 'grades.jsonl').read_text().splitlines()]
        stop = probe['early_stop']
        self.assertEqual((stop['triggered'], stop['tasks'], stop['tasks_total'], stop['fraction']), (True, 10, 60, 10 / 60))
        self.assertEqual(stop['order'], [[line['env_id'], line['index']] for line in stream[:10]])
        self.assertLess(stop['z'], stop['threshold'])
        self.assertEqual(self.scored, [10])                   # one check, at the first check point
        # The run ended early: the worker wrote a partial result, with fewer rollouts than the plan has.
        on_box = json.loads((self.run_dir() / 'result.json').read_text())
        self.assertIs(on_box['partial'], True)
        self.assertIn(on_box['stop_reason'], ('finalize', 'signal:15'))
        self.assertIs(probe['partial'], True)
        self.assertLess(probe['samples'], 60 * 8)
        self.assertGreaterEqual(len(probe['envs'][ENV_ID]['per_task']), 10)
        self.assertEqual(self.ended_runs(), 1)
        self.assertEqual(probe['serve_layout']['layout']['name'], 'rtx5090-tp2')

        # The controller decides RED on this result.
        ctl, item = self.decide(shadow(early_stop=self.FAST), probe)
        self.assertEqual((item.decision, item.paid, item.meta['scored']['early_stop_fraction']), ('red', False, 10 / 60))
        self.assertEqual(ctl.decisions.entries[0].extra, {'early_stop': True, 'early_stop_fraction': 10 / 60})
        # With the flag off, the same result is an incomplete probe.
        self.assertEqual(self.decide(shadow(), probe)[1].decision, 'no_decision')

        # The host restarts after the box finished and before the host stored anything: the box has only the
        # result document of its own eval job. The host reads the stream again and finds the same stop.
        (self.job_dir() / 'RESULT.json').write_text((self.job_dir() / 'markers' / 'eval.done').read_text())
        self.assertNotIn('early_stop', json.loads((self.job_dir() / 'RESULT.json').read_text())['probe_result'])
        self.scored = []
        record, bundle = self.restarted_host(**self.host(self.FAST))
        self.assertIs(record.state, ItemState.RESULT)
        self.assertEqual(bundle.probe_result['early_stop'], stop)
        self.assertEqual((self.scored, self.ended_runs()), ([10], 1))
        self.assertEqual(self.decide(shadow(early_stop=self.FAST), bundle.probe_result)[1].decision, 'red')
        # The same when the box finishes between the first look of the restarted host and its eval stage.
        self.assertEqual(self.probe(self.FAST)['early_stop'], stop)
        self.assertEqual(self.ended_runs(), 1)
        # A host that restarts without the flag does not look at the stream.
        reads = self.stream_reads()
        self.assertGreater(reads, 0)
        self.assertNotIn('early_stop', self.restarted_host(**self.host(None))[1].probe_result)
        self.assertNotIn('early_stop', self.probe(None))
        self.assertEqual(self.stream_reads(), reads)

    def test_a_candidate_above_the_threshold_runs_to_the_end(self):
        # The head passes 8%, so this candidate is far above. The gate needs 100 scored tasks for a GREEN.
        self.new_round(head_passes=1, tasks=120)
        self.fb.control(delay_ms=1500)
        probe = self.probe(self.FAST)
        self.assertIs(probe['partial'], False)
        self.assertEqual((probe['samples'], len(probe['envs'][ENV_ID]['per_task'])), (120 * 8, 120))
        self.assertNotIn('early_stop', probe)
        self.assertEqual(self.ended_runs(), 0)
        self.assertGreater(self.stream_reads(), 0)                        # the host watched
        self.assertTrue(self.scored and set(self.scored) <= set(range(10, 121, 5)), self.scored)
        ctl, item = self.decide(shadow(early_stop=self.FAST), probe)
        self.assertEqual(item.decision, 'green_fresh')
        self.assertEqual(outcome(ctl, item), outcome(*self.decide(shadow(), probe)))

    def test_with_the_flag_off_the_host_never_reads_the_stream_or_ends_the_run(self):
        self.new_round(head_passes=11)
        probe = self.probe(None)
        self.assertIs(probe['partial'], False)
        self.assertEqual(probe['samples'], 60 * 8)
        self.assertNotIn('early_stop', probe)
        self.assertEqual((self.stream_reads(), self.ended_runs(), self.scored), (0, 0, []))
        self.assertEqual(self.decide(shadow(), probe)[1].decision, 'red')
