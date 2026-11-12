"""The judge: the verdict of the real live scorer against a baseline of HEAD blocks.

The scorer is ``ops/parallel_eval/live/pe_score.py`` with the official
``baseline_v1`` and ``prior_gate`` (``round_fixture``). The box results are made
with the real box code (``box/steps.public_probe``) and the real probe layouts.
"""
from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pytest

from ops.parallel_eval.pit import probe_layout as PL
from ops.parallel_eval.pit.batchctl import baseline as B
from ops.parallel_eval.pit.batchctl import judge as J
from ops.parallel_eval.pit.batchctl import rounds as R
from ops.parallel_eval.pit.batchctl.box import steps
from ops.parallel_eval.pit.batchctl.interfaces import VERDICT_GREEN, VERDICT_RED, JudgeInvalid, JudgeNotReady
from ops.parallel_eval.pit.batchctl.layout import canonical, sha256_hex
from ops.parallel_eval.pit.batchctl.masking import mask_key
from ops.parallel_eval.pit.round_fixture import RID
from ops.parallel_eval.pit.round_fixture import Round as LiveRound
from ops.parallel_eval.pit.round_fixture import probe_result

RUN = 'run1'
EPOCH = 1200
BASE_STATE, OTHER_STATE = 'a' * 64, 'b' * 64
HEAD, CAND = 'c' * 64, 'd' * 64
SAMPLING = {'harness': 'live-math-v3', 'sampling_contract': {'temperature': 1.0, 'top_p': 1.0}}
GPU = 'NVIDIA GeForce RTX 5090'
TP2 = PL.RTX5090_TP2
ONE_GPU = PL.RTX5090_1GPU


def serve_record(layout: PL.ProbeLayout = TP2, sampling: Any = SAMPLING, gpu: str = GPU) -> Dict[str, Any]:
    groups = layout.groups(8)
    return dict(
        layout=layout.as_dict(), launcher=layout.launcher(), gpu_groups=[list(g) for g in groups], serve_s=100.0,
        replicas=[dict(replica=i, pid=100 + i, gpus=list(g), port=PL.BASE_PORT + i, vllm_version=layout.vllm_version)
                  for i, g in enumerate(groups)],
        concurrency_per_replica=layout.concurrency(), runtime=dict(vllm_version=layout.vllm_version),
        gpus=[gpu] * 8, sampling=copy.deepcopy(sampling),
    )


class FakeFleet(J.Fleet):
    def __init__(self) -> None:
        self.fences: set = set()
        self.head: Optional[str] = None

    def live(self):
        return set(self.fences)

    def head_state(self):
        return self.head


class World:
    def __init__(self, tmp: Path) -> None:
        self.live = LiveRound(tmp / 'gate')
        round_file = self.live.dir / 'ROUND.private.json'
        doc = json.loads(round_file.read_text())
        doc['indices'] = self.live.indices
        round_file.write_text(json.dumps(doc))
        self.secret = doc['seed']
        self.gate = tmp / 'gate'
        self.sample(EPOCH)
        self.rounds = R.HostRounds(self.gate)
        self.rnd = self.rounds.of(EPOCH)
        self.baselines = B.Baselines(tmp / 'private' / 'baseline')
        self.records = J.Records(tmp / 'private' / 'judged')
        self.fleet = FakeFleet()
        self.logs: list = []
        self.judge = J.GateJudge(
            run_id=RUN, rounds=self.rounds, baselines=self.baselines, records=self.records,
            score=self.live.scorer.score, blocks_baseline=self.live.scorer.blocks_baseline, fleet=self.fleet,
            log=self.logs.append,
        )

    def sample(self, epoch: int, **extra: Any) -> None:
        folder = self.gate / 'epochs' / str(epoch)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'SAMPLE.private.json').write_text(json.dumps(dict(round_id=RID, **extra)))

    def head_cells(self, seed: int, *, drop: int = 0) -> Dict[int, Dict[str, Any]]:
        """One HEAD block: four rollouts for each task at the pass rate of the head."""
        rng = random.Random('head:%d' % seed)
        ids = self.live.ids[drop:]
        return {self.live.index[t]: self.live.cell(self.live.rate[t], 4, rng=rng) for t in ids}

    def masked(self, cells: Mapping[int, Mapping[str, Any]], batch_id: str, *, serve: Any = 'tp2',
               key: Optional[str] = None) -> Dict[str, Any]:
        result = probe_result(cells, planned_tasks=len(self.rnd.indices))
        if serve is not None:
            result['serve_layout'] = serve_record() if serve == 'tp2' else serve
        if key is None:
            epoch = J.epoch_of(batch_id)
            seed = self.rounds.of(epoch).plan_seed if epoch is not None else self.rnd.plan_seed
            key = mask_key(seed, RUN, batch_id)
        return steps.public_probe(result, key)

    def result(self, batch_id: str, cells: Mapping[int, Mapping[str, Any]], *, fence: int = 1,
               blocks: Sequence[Mapping[str, Any]] = (), base_state: str = BASE_STATE, serve: Any = 'tp2',
               key: Optional[str] = None, outcome: str = 'probed', reason: str = '') -> bytes:
        payload: Dict[str, Any] = {}
        candidate = None
        if outcome == 'probed':
            payload = {
                'candidate_id': CAND, 'train': {},
                'probe': {'status': 'ok', 'result': self.masked(cells, batch_id, serve=serve, key=key)},
            }
            if blocks:
                payload['head'] = {'checkpoint_id': HEAD, 'blocks': list(blocks)}
            candidate = {'state_id': 'e' * 64, 'checkpoint_id': CAND, 'fp32_master_digest': 'f' * 64}
        return canonical({
            'schema': J.RESULT_SCHEMA, 'run_id': RUN, 'batch_id': batch_id, 'fence': fence, 'attempt': 1,
            'base_state': base_state, 'base_seq': 0, 'outcome': outcome, 'reason': reason,
            'base': {'checkpoint_id': HEAD, 'fp32_master_digest': '1' * 64}, 'binding': {'checkpoint_id': HEAD},
            'candidate': candidate, 'payload': payload, 'box': {'gpus': 8}, 'seconds': {},
        })

    def block(self, number: int, batch_id: str, *, serve: Any = 'tp2', drop: int = 0, status: str = 'ok',
              k: int = 4) -> Dict[str, Any]:
        return {'block': number, 'k': k, 'max_tokens': self.rnd.max_tokens, 'status': status,
                'result': self.masked(self.head_cells(number, drop=drop), batch_id, serve=serve)}

    def fill(self, numbers: Sequence[int] = (0, 1, 2)) -> None:
        """Store these HEAD blocks, as the judge does from the results of other boxes."""
        fingerprint = J.probe_fingerprint.from_serve_layout(serve_record())
        for number in numbers:
            cells = {str(i): c for i, c in self.head_cells(number).items()}
            assert self.baselines.add(self.rnd, HEAD, number, k=4, max_tokens=self.rnd.max_tokens, cells=cells,
                                      fingerprint=fingerprint, source={'batch_id': 'e1'}) is None

    def verdict(self, batch_id: str, body: bytes, base_state: str = BASE_STATE) -> str:
        return self.judge.verdict(body, batch_id=batch_id, base_state=base_state)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path / 'own')


def test_green_and_red_are_the_decisions_of_the_live_estimator(world: World) -> None:
    world.fill()
    BV, cfg, live = world.live.BV, world.live.cfg, world.live
    merged = {}
    for task in live.ids:
        cells = [world.head_cells(n)[live.index[task]] for n in range(3)]
        merged[task] = {key: sum(cell[key] for cell in cells) for key in ('n', 'c', 'w', 'u', 'b', 'x', 'tok')}
    for shift, want in ((0.25, VERDICT_GREEN), (-0.25, VERDICT_RED)):
        batch_id = 'e%d' % EPOCH
        cells = live.candidate(shift, seed=3)
        body = world.result(batch_id, cells, fence=7 if shift > 0 else 8)
        assert world.verdict(batch_id, body) == want
        official = BV.estimate(
            live.drawn, merged, {BV.task_of(i, cfg): c for i, c in cells.items()}, live.pool, shares=cfg['shares'],
            tau=float(cfg['tau_run_separate']), sigma_run_floor=float(cfg['sigma_run_floor']),
            bar_z=float(cfg['bar_z']), z_inclusive=bool(cfg['z_inclusive']), cap_max=float(cfg['cap_max']),
            min_scored=int(cfg['min_scored']), min_cover=float(cfg['min_cover']))
        assert official['decision'] == want
        record = world.records.get(batch_id, sha256_hex(body))
        assert record['decision'] == want and record['candidate'] == CAND and record['head'] == HEAD
        assert record['evaluated']['z'] == pytest.approx(official['evaluated']['z'])
        assert record['heldout']['head_rollouts'] == 12 and record['heldout']['round_id'] == RID
        assert record['heldout']['baseline_done_at_cr'] and record['heldout']['settings_hash']
        assert record['plan_seed'] == world.rnd.plan_seed


def test_the_blocks_of_three_boxes_make_the_baseline(world: World) -> None:
    cells = world.live.candidate(0.25)
    bodies = {}
    for number, epoch in enumerate((EPOCH, EPOCH + 1, EPOCH + 2)):
        world.sample(epoch)
        batch_id = 'e%d' % epoch
        bodies[batch_id] = world.result(batch_id, cells, fence=number + 1, blocks=[world.block(number, batch_id)])
        world.fleet.fences.add(number + 1)
    first, second, third = sorted(bodies)
    with pytest.raises(JudgeNotReady):
        world.verdict(first, bodies[first])
    with pytest.raises(JudgeNotReady):
        world.verdict(second, bodies[second])
    assert world.baselines.missing(world.rnd, HEAD) == [2]
    assert world.verdict(third, bodies[third]) == VERDICT_GREEN
    assert world.verdict(first, bodies[first]) == VERDICT_GREEN
    assert world.baselines.missing(world.rnd, HEAD) == []


def test_fingerprint_mismatch_gives_no_verdict(world: World) -> None:
    world.fill()
    batch_id = 'e%d' % EPOCH
    for serve in (serve_record(ONE_GPU), serve_record(gpu='NVIDIA B300'), None,
                  serve_record(sampling={'harness': 'other', 'sampling_contract': {}}), serve_record(sampling=None)):
        body = world.result(batch_id, world.live.candidate(0.25), serve=serve)
        with pytest.raises(JudgeInvalid, match='fingerprint'):
            world.verdict(batch_id, body)
        assert world.records.get(batch_id, sha256_hex(body)) is None


def test_a_block_with_another_fingerprint_is_not_stored(world: World) -> None:
    world.fill((0, 1))
    batch_id = 'e%d' % EPOCH
    body = world.result(batch_id, world.live.candidate(0.25),
                        blocks=[world.block(2, batch_id, serve=serve_record(ONE_GPU))])
    with pytest.raises(JudgeNotReady):
        world.verdict(batch_id, body)
    assert world.baselines.missing(world.rnd, HEAD) == [2]
    assert any('another hardware and layout fingerprint' in text for text in world.logs)


def test_a_block_under_the_cover_or_with_other_settings_is_not_stored(world: World) -> None:
    world.fill((0, 1))
    batch_id = 'e%d' % EPOCH
    thin = len(world.live.ids) // 5
    for block in (world.block(2, batch_id, drop=thin), world.block(2, batch_id, k=8),
                  world.block(2, batch_id, status='failed'), world.block(9, batch_id)):
        body = world.result(batch_id, world.live.candidate(0.25), blocks=[block], fence=random.randrange(10, 99))
        with pytest.raises(JudgeNotReady):
            world.verdict(batch_id, body)
        assert world.baselines.missing(world.rnd, HEAD) == [2]


def test_a_result_of_another_round_or_run_is_invalid(world: World) -> None:
    world.fill()
    batch_id = 'e%d' % EPOCH
    cells = world.live.candidate(0.25)
    with pytest.raises(JudgeInvalid, match='not a probe of round'):
        world.verdict(batch_id, world.result(batch_id, cells, key='k' * 64))
    with pytest.raises(JudgeInvalid, match='another run, batch or base state'):
        world.verdict(batch_id, world.result(batch_id, cells), base_state=OTHER_STATE)
    with pytest.raises(JudgeInvalid, match='another run, batch or base state'):
        world.verdict('e%d' % (EPOCH + 1), world.result(batch_id, cells))
    with pytest.raises(JudgeInvalid, match='not an epoch'):
        world.verdict('bl-1', world.result('bl-1', cells))
    with pytest.raises(JudgeInvalid, match='not JSON'):
        world.verdict(batch_id, b'{')


def test_too_many_probe_errors_give_no_verdict(world: World) -> None:
    world.fill()
    batch_id = 'e%d' % EPOCH
    body = world.result(batch_id, world.live.candidate(0.25, errors=4))
    with pytest.raises(JudgeInvalid, match='no decision'):
        world.verdict(batch_id, body)
    assert world.records.get(batch_id, sha256_hex(body)) is None


def test_not_trained_is_red_with_a_no_decision_record(world: World) -> None:
    batch_id = 'e%d' % EPOCH
    body = world.result(batch_id, {}, outcome='not_trained', reason='bound_to_other_checkpoint')
    assert world.verdict(batch_id, body) == VERDICT_RED
    record = world.records.get(batch_id, sha256_hex(body))
    assert record['decision'] == J.DECISION_NONE and record['candidate'] is None
    assert record['reasons'] == ['not_trained: bound_to_other_checkpoint']


def test_no_round_yet_is_not_ready(world: World) -> None:
    batch_id = 'e%d' % (EPOCH + 5)
    body = world.result(batch_id, world.live.candidate(0.25), key='k' * 64)
    with pytest.raises(JudgeNotReady, match='no round yet'):
        world.verdict(batch_id, body)


def test_a_result_keeps_its_first_verdict(world: World) -> None:
    world.fill()
    batch_id = 'e%d' % EPOCH
    body = world.result(batch_id, world.live.candidate(0.25))
    assert world.verdict(batch_id, body) == VERDICT_GREEN

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('the scorer must not run again')
    world.judge.score = boom
    assert world.verdict(batch_id, body) == VERDICT_GREEN


def test_baseline_of_a_replaced_head_that_nobody_completes_is_a_loss(world: World) -> None:
    world.fill((0, 1))
    batch_id = 'e%d' % EPOCH
    body = world.result(batch_id, world.live.candidate(0.25), fence=4)
    world.fleet.fences = {4}
    world.fleet.head = BASE_STATE
    with pytest.raises(JudgeNotReady):
        world.verdict(batch_id, body)
    world.fleet.head = OTHER_STATE
    with pytest.raises(JudgeInvalid, match='replaced HEAD'):
        world.verdict(batch_id, body)


def test_a_live_lease_that_holds_the_missing_block_keeps_the_wait(world: World) -> None:
    world.fill((0, 1))
    batch_id = 'e%d' % EPOCH
    world.fleet.fences = {4, 5}
    assert world.baselines.claim(world.rnd, HEAD, 5, live=world.fleet.live(), others=0) == (2,)
    world.fleet.head = OTHER_STATE
    with pytest.raises(JudgeNotReady):
        world.verdict(batch_id, world.result(batch_id, world.live.candidate(0.25), fence=4))


def test_scorer_refuses_another_token_cap(world: World) -> None:
    world.fill()
    held = world.baselines.complete(world.rnd, HEAD)
    doc = world.live.scorer.blocks_baseline(RID, held['blocks'], rollouts=12, max_tokens=4096,
                                            fingerprint=held['fingerprint'], done_at=held['done_at'])
    with pytest.raises(RuntimeError, match='baseline cap'):
        world.live.scorer.score(EPOCH, RID, {}, baseline=doc)


# -- the baseline store --------------------------------------------------------

def test_claims_split_the_blocks_across_boxes(world: World) -> None:
    store, rnd = world.baselines, world.rnd
    live = {1, 2, 3, 4}
    assert store.claim(rnd, HEAD, 1, live=live, others=5) == (0,)
    assert store.claim(rnd, HEAD, 2, live=live, others=5) == (1,)
    assert store.claim(rnd, HEAD, 1, live=live, others=0) == (0,)
    assert store.claim(rnd, HEAD, 3, live=live, others=5) == (2,)
    assert store.claim(rnd, HEAD, 4, live=live, others=5) == ()
    assert store.free(rnd, HEAD, live) == []
    assert store.free(rnd, HEAD, {1, 3, 4}) == [1]
    assert store.claim(rnd, HEAD, 9, live={1, 3, 9}, others=5) == (1,)


def test_a_box_alone_takes_each_free_block(world: World) -> None:
    store, rnd = world.baselines, world.rnd
    assert store.claim(rnd, HEAD, 1, live={1}, others=0) == (0, 1, 2)
    assert store.claim(rnd, OTHER_STATE, 2, live={1, 2}, others=1) == (0, 1)
    assert store.claim(rnd, OTHER_STATE, 3, live={1, 2, 3}, others=0) == (2,)


def test_a_released_lease_frees_what_it_did_not_bring(world: World) -> None:
    store, rnd = world.baselines, world.rnd
    assert store.claim(rnd, HEAD, 1, live={1}, others=0) == (0, 1, 2)
    world.fill((0,))
    assert store.free(rnd, HEAD, {1}) == []
    store.release(rnd, HEAD, 1)
    assert store.free(rnd, HEAD, {1}) == [1, 2]


def test_a_stored_block_is_never_replaced(world: World) -> None:
    world.fill((0,))
    before = world.baselines.complete(world.rnd, HEAD)
    fingerprint = J.probe_fingerprint.from_serve_layout(serve_record())
    other = {str(i): dict(c, c=0) for i, c in world.head_cells(0).items()}
    assert world.baselines.add(world.rnd, HEAD, 0, k=4, max_tokens=world.rnd.max_tokens, cells=other,
                               fingerprint=fingerprint, source={}) is None
    world.fill((1, 2))
    after = world.baselines.complete(world.rnd, HEAD)
    assert before is None and after['blocks'][0] == {str(i): c for i, c in world.head_cells(0).items()}
    assert after['rollouts'] == 12 and len(after['sources']) == 3


# -- the rounds ----------------------------------------------------------------

def test_round_seeds_are_the_seeds_of_the_live_gate(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    BV = world.live.BV
    assert world.rnd.plan_seed == BV.cand_seed(world.secret, EPOCH)
    assert list(world.rnd.block_seeds) == [BV.plan_seed(world.secret, block) for block in range(3)]
    assert (world.rnd.k_cand, world.rnd.k_block, world.rnd.blocks) == (8, 4, 3)
    assert list(world.rnd.indices) == world.live.indices and world.rnd.env_id == world.live.cfg['probe_env']
    monkeypatch.setenv('HG_GATE_STATE', str(world.gate))
    monkeypatch.delenv('PE_GATE_EP', raising=False)
    monkeypatch.delenv('PE_HELDOUT', raising=False)
    monkeypatch.setitem(sys.modules, 'baseline_v1', BV)
    seeds = importlib.import_module('ops.parallel_eval.live.pe_round_seed')
    assert seeds.plan_seed_for_epoch(EPOCH) == world.rnd.plan_seed
    assert seeds.indices_for_epoch(EPOCH) == list(world.rnd.indices)
    world.sample(EPOCH + 1, attempt=2)
    assert world.rounds.of(EPOCH + 1).plan_seed == seeds.plan_seed_for_epoch(EPOCH + 1) != world.rnd.plan_seed
    assert 'seed' not in repr(world.rnd) and str(world.rnd.indices[0]) not in repr(world.rnd).split(',')[0]


def test_rounds_fail_closed(world: World) -> None:
    with pytest.raises(R.RoundNotReady):
        world.rounds.of(EPOCH + 7)
    world.sample(EPOCH + 1, indices=[1, 2, 3])
    with pytest.raises(R.RoundError, match='not the draw'):
        world.rounds.of(EPOCH + 1)
    round_file = world.live.dir / 'ROUND.private.json'
    doc = json.loads(round_file.read_text())
    round_file.write_text(json.dumps(dict(doc, cfg=dict(doc['cfg'], scoring_mode='same-run'))))
    with pytest.raises(R.RoundError, match='own HEAD run'):
        world.rounds.of(EPOCH)
    cfg = {key: value for key, value in doc['cfg'].items() if key != 'max_tokens'}
    round_file.write_text(json.dumps(dict(doc, cfg=cfg)))
    with pytest.raises(R.RoundError, match='settings'):
        world.rounds.of(EPOCH)
    assert R.HostRounds(world.gate, max_tokens=65536).of(EPOCH).max_tokens == 65536
    round_file.write_text(json.dumps(dict(doc, seed=None)))
    with pytest.raises(R.RoundNotReady):
        world.rounds.of(EPOCH)
