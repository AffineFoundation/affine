"""Test fixture: one gate round in the files that the live host scorer reads, and that scorer.

The scorer is the real ``ops/parallel_eval/live/pe_score.py`` with the official ``baseline_v1`` and
``prior_gate`` of this repository. It is loaded as a private module copy that reads its round files from a
temp directory. ``Round`` writes the secret draw, the priors and the full head baseline of one round.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from unittest import mock

from ops.parallel_eval.pit.pe_score import call_live_host_score
from ops.parallel_eval.pit.probe_fixture import ENV_ID

REPO = Path(__file__).resolve().parents[3]
LIVE = REPO / 'ops' / 'parallel_eval' / 'live'
GATE = REPO / 'miner' / 'validator' / 'gate'
LIVE_IMPORTS = ('pe_common', 'replace_pod', 'baseline_v1', 'prior_gate', 'probe_error_guard', 'heldout')
EPOCH, RID = 10, 'round-1'


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
        spec = importlib.util.spec_from_file_location('pit_round_fixture_live_pe_score', LIVE / 'pe_score.py')
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
    for module, source in ((scorer.BV, GATE / 'baseline_v1.py'), (scorer.PG, GATE / 'prior_gate.py')):
        assert Path(module.__file__).resolve() == source, module.__file__
    scorer.HELDOUT, scorer.FLAG = tmp / 'heldout', tmp / 'flag.json'
    return scorer


def write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(doc))


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
        write_json(self.dir / 'ROUND.private.json', dict(id=RID, cfg=self.cfg, ids=self.ids, pool=dict(per_env=self.pool),
                                                         seed='ab' * 32, n=len(self.ids)))
        write_json(self.dir / 'PI.private.json', dict(env={t: d['env'] for t, d in self.drawn.items()},
                                                      pi={t: d['pi'] for t, d in self.drawn.items()}))
        self.baseline = dict(per_task=self.head, max_tokens=65536, rollouts_per_task=12, settings_hash='s' * 64,
                             done_at_cr='now')
        write_json(self.dir / 'BASELINE.private.json', self.baseline)

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
