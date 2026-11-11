"""The controller object: settings, start-up checks, status file, and the
threaded loop.

The world around the controller is the fake world of ``sim.py``. In the
threaded tests one lock makes the fakes safe for many threads.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import threading
import time
from pathlib import Path

import pytest

from . import MODE_SHADOW
from . import controller as K
from . import core as C
from . import guard as G
from . import journal as J
from .guard import GuardedStore, StateRoot, parse_box_name, shadow_prefix
from .sim import ControllerStore, Harness, MemOps, Scenario
from .. import PRICE_CEILING_USD_H
from ..shadow import ShadowWriteError

RUN = 'run0001'
H1 = 'a' * 64
PACKAGE = Path(__file__).parent
B300 = (C.Rung('p', 'B300', 'secure'),)


def config(**changes) -> K.Config:
    settings = dict(run_id=RUN, genesis_state=H1, core=C.CoreConfig(2, 2, B300))
    settings.update(changes)
    return K.Config(**settings)


def controller_on(harness: Harness, root: StateRoot, **changes) -> K.Controller:
    """One more controller object in the world of this harness."""
    world = harness.world
    parts = dict(
        root=root, config=harness.config, store=ControllerStore(world), providers=world.providers,
        agent=world.agent, source=world.source, judge=world.judge, clock=world.clock,
        file_ops=world.files,
    )
    parts.update(changes)
    return K.Controller(**parts)


def journal_records(harness: Harness):
    world = harness.world
    return J.scan(bytes(world.files.files[world.journal_path].data))[0]


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

def test_the_default_settings_are_shadow_with_the_promotion_gate_closed():
    cfg = config()
    assert cfg.mode == MODE_SHADOW
    assert cfg.allow_sharded_gpus is False
    assert cfg.gpu_count == 8 and cfg.price_ceiling_usd_h == PRICE_CEILING_USD_H
    assert cfg.max_cost_usd is None and cfg.breaker_failures >= 1


@pytest.mark.parametrize('mode', ['live', 'LIVE', 'prod', 'mainnet', '', None, 'shadow ', 'Shadow', 7])
def test_settings_refuse_every_mode_but_shadow(mode):
    with pytest.raises(ShadowWriteError):
        config(mode=mode)


@pytest.mark.parametrize('run_id', ['', 'ab', 'Run0001', 'run 0001', 'run0001\n', '../x', 'a' * 33, None])
def test_settings_refuse_a_bad_run_id(run_id):
    with pytest.raises(ShadowWriteError):
        config(run_id=run_id)


@pytest.mark.parametrize('gpu', ['RTX5090', 'H100'])
def test_a_sharded_gpu_class_needs_the_promotion_gate(gpu):
    ladder = (C.Rung('p', gpu, 'secure'), C.Rung('p', 'B300', 'secure'))
    with pytest.raises(ValueError, match='promotion gate'):
        config(core=C.CoreConfig(2, 2, ladder))
    assert config(core=C.CoreConfig(2, 2, ladder), allow_sharded_gpus=True).core.ladder == ladder


@pytest.mark.parametrize('changes', [
    {'core': C.CoreConfig(2, 2, (C.Rung('p', 'A100', 'secure'),))},
    {'core': C.CoreConfig(2, 2, (C.Rung('p', 'RTX4090', 'secure'),)), 'allow_sharded_gpus': True},
    {'genesis_state': 'a' * 63},
    {'genesis_state': 'A' * 64},
    {'genesis_state': None},
    {'price_ceiling_usd_h': 0.0},
    {'price_ceiling_usd_h': -1.0},
    {'price_ceiling_usd_h': PRICE_CEILING_USD_H + 0.01},
    {'gpu_count': 4},
    {'gpu_count': 16},
    {'box_ttl_s': 100.0, 'lease_ttl_s': 200.0},
    {'breaker_failures': 0},
    {'effect_error_limit': 0},
    {'effect_timeout_s': 0.0},
    {'rent_timeout_s': 0.0},
    {'max_workers': 11},
    {'core': C.CoreConfig(40, 2, B300)},
], ids=lambda changes: '+'.join(sorted(changes)))
def test_settings_out_of_range_are_refused(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_the_price_ceiling_can_only_go_down():
    assert config(price_ceiling_usd_h=20.0).price_ceiling_usd_h == 20.0


def test_default_ladder_is_small_gpu_first_and_secure_before_spot():
    full = K.default_ladder('lium', allow_sharded_gpus=True)
    assert [(r.provider, r.gpu, r.tier) for r in full] == [
        ('lium', 'RTX5090', 'secure'), ('lium', 'RTX5090', 'spot'),
        ('lium', 'H100', 'secure'), ('lium', 'H100', 'spot'),
        ('lium', 'B300', 'secure'), ('lium', 'B300', 'spot'),
    ]
    secure_first = K.default_ladder('lium', allow_sharded_gpus=True, secure_first=True)
    assert [(r.gpu, r.tier) for r in secure_first] == [
        ('RTX5090', 'secure'), ('H100', 'secure'), ('B300', 'secure'),
        ('RTX5090', 'spot'), ('H100', 'spot'), ('B300', 'spot'),
    ]
    assert set(secure_first) == set(full) and len(secure_first) == len(full)
    # Both are accepted settings.
    for ladder in (full, secure_first):
        config(core=C.CoreConfig(2, 2, ladder), allow_sharded_gpus=True)


def test_default_ladder_respects_the_promotion_gate():
    # The gate is closed by default: only the class that is not sharded.
    gated = K.default_ladder('lium')
    assert [(r.gpu, r.tier) for r in gated] == [('B300', 'secure'), ('B300', 'spot')]
    config(core=C.CoreConfig(2, 2, gated))


def test_a_ladder_needs_its_providers(tmp_path):
    harness = Harness(tmp_path, 1, Scenario(batches=1))
    root = StateRoot.open(harness.root_path, harness.run_id)
    with pytest.raises(ValueError, match='unknown provider'):
        controller_on(harness, root, providers={'p0': harness.world.providers['p0']})


def test_a_state_root_of_another_run_is_refused(tmp_path):
    harness = Harness(tmp_path, 1, Scenario(batches=1))
    other = StateRoot.create(tmp_path / 'other-root', 'other0001')
    with pytest.raises(ShadowWriteError):
        controller_on(harness, other)


# ----------------------------------------------------------------------------
# Shadow mode, end to end
# ----------------------------------------------------------------------------

def test_a_full_run_stays_inside_its_prefix_and_its_state_root(tmp_path):
    before = sorted(p.name for p in tmp_path.iterdir())
    harness = Harness(tmp_path, 5, Scenario(
        batches=5, fault_rate=0.05, fault_budget=40, crash_rate=0.01, crash_budget=5, real_disk=True,
    ))
    harness.run()
    harness.ctl.write_status()
    world = harness.world

    # R2: every object is under the shadow prefix of this run.
    assert world.raw.objs and all(key.startswith(world.prefix) for key in world.raw.objs)
    assert world.prefix == shadow_prefix(harness.run_id)
    names = {key.rsplit('/', 1)[-1].lower() for key in world.raw.objs}
    assert names.isdisjoint(G.FORBIDDEN_NAMES)

    # Local: one new directory, and in it only files that the controller owns.
    after = sorted(p.name for p in tmp_path.iterdir())
    assert sorted(set(after) - set(before)) == [harness.root_path.name]
    assert {p.name for p in harness.root_path.iterdir()} == G.STATE_FILES

    # Boxes: every box that this run rented has a name of this run.
    for provider in world.providers.values():
        for box in provider.boxes.values():
            assert box.ours == (parse_box_name(harness.run_id, box.name) is not None)
            assert box.alive != box.ours


class ListOnlySource:
    """A source that fails on every call but ``list``."""

    def __init__(self, inner):
        self._inner = inner
        # The harness, not the controller, releases batches through this.
        self.released = inner.released

    def list(self):
        return self._inner.list()

    def __getattr__(self, name):
        raise AssertionError('the controller called source.%s' % name)


def test_the_batch_source_is_read_only(tmp_path):
    harness = Harness(tmp_path, 9, Scenario(batches=4))
    harness.world.source = ListOnlySource(harness.world.source)
    report = harness.run()
    assert sum(report.outcomes.values()) == 4


def test_the_guard_wraps_every_store_the_controller_gets(tmp_path):
    # The controller cannot be built around a store that is not fenced.
    source = inspect.getsource(K.Controller.__init__)
    assert 'GuardedStore(store, config.run_id)' in source
    harness = Harness(tmp_path, 1, Scenario(batches=1))
    harness._boot()
    ctl = harness.ctl
    assert type(ctl.store) is GuardedStore
    assert ctl.effects.store is ctl.store
    for call in (
        lambda: ctl.store.put('HEAD.json', b'{}'),
        lambda: ctl.store.get('HEAD.json', max_bytes=10),
        lambda: ctl.store.delete('pit/HEAD-CHAIN.json'),
        lambda: ctl.store.list(''),
        lambda: ctl.store.copy(ctl.layout.head, 'HEAD.json', src_etag='x'),
        lambda: ctl.root.file('mining.json'),
        lambda: ctl.root.file('llms.txt'),
    ):
        with pytest.raises(ShadowWriteError):
            call()
    ctl.close()


def test_runtime_modules_exist():
    names = {path.name for path in PACKAGE.glob('*.py')}
    assert {'controller.py', 'effects.py', 'core.py', 'journal.py', 'guard.py', 'layout.py',
            'timers.py', 'interfaces.py'} <= names


# ----------------------------------------------------------------------------
# Start-up checks
# ----------------------------------------------------------------------------

def test_one_controller_per_state_root(tmp_path):
    harness = Harness(tmp_path, 2, Scenario(batches=2, chaos=False))
    harness._boot()
    second = controller_on(harness, StateRoot.open(harness.root_path, harness.run_id))
    with pytest.raises(J.JournalLocked):
        second.open()
    # The refused start changed nothing, and the first controller works on.
    assert second.journal is None
    harness.finish()
    # ``finish`` closed the first controller. Now the root is free.
    second.open().close()


def test_a_new_journal_cannot_take_over_the_r2_prefix(tmp_path):
    harness = Harness(tmp_path, 3, Scenario(batches=3, chaos=False))
    harness.run()
    world = harness.world
    before = dict(world.raw.objs)
    world.clock.advance(60.0)

    # The journal is lost and someone starts again with an empty state root.
    fresh = StateRoot.create(tmp_path / 'fresh-root', harness.run_id)
    amnesiac = controller_on(harness, fresh, file_ops=MemOps(world))
    with pytest.raises(K.ControllerError, match='bound to another journal'):
        amnesiac.open()
    # It did not move the pointer, and it wrote nothing at all.
    assert world.raw.objs == before
    assert amnesiac.journal is None
    with pytest.raises(K.ControllerError):
        amnesiac.commit(C.Tick())

    # The real journal still opens.
    harness._boot()
    harness.finish()


def test_a_journal_of_other_settings_is_refused(tmp_path):
    harness = Harness(tmp_path, 4, Scenario(batches=2, chaos=False))
    harness.run()
    root = StateRoot.open(harness.root_path, harness.run_id)
    other = dataclasses.replace(harness.config, genesis_state='b' * 64)
    with pytest.raises(K.ControllerError, match='another run'):
        controller_on(harness, root, config=other).open()
    controller_on(harness, root).open().close()


def test_an_open_controller_cannot_be_opened_again(tmp_path):
    harness = Harness(tmp_path, 4, Scenario(batches=1, chaos=False))
    harness._boot()
    with pytest.raises((K.ControllerError, J.JournalLocked)):
        harness.ctl.open()


class FullDisk:
    """The file system of the harness, with a disk that can become full."""

    def __init__(self, inner):
        self._inner = inner
        self.full = False

    def write(self, fd, data):
        if self.full:
            raise OSError(28, 'No space left on device')
        return self._inner.write(fd, data)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_failed_journal_write_stops_the_controller(tmp_path):
    harness = Harness(tmp_path, 6, Scenario(batches=3, chaos=False))
    disks = []
    harness.wrap = lambda role, part: disks.append(FullDisk(part)) or disks[-1] if role == 'files' else part
    harness._boot()
    ctl = harness.ctl
    harness.fair(2)
    count = ctl.journal.count
    durable = [rec['h'] for rec in journal_records(harness)]

    disks[-1].full = True
    with pytest.raises(OSError):
        for _ in range(50):
            harness.fair(1)
    # Dead: no record, no effect, and no view of a state that is not durable.
    for call in (
        lambda: ctl.commit(C.Tick()),
        ctl.due_ops,
        ctl.tick,
        ctl.step,
        ctl._fence_view,
        lambda: ctl._lease_view(1),
        lambda: ctl._done_batch('b000'),
        lambda: ctl.execute(K.IngestOp()),
        lambda: ctl.attempt(K.SweepOp()),
    ):
        with pytest.raises(K.ControllerError):
            call()
    assert [rec['h'] for rec in journal_records(harness)][:count] == durable

    # A new process with a working disk finishes every batch.
    disks[-1].full = False
    harness._restart()
    report = harness.finish()
    assert sum(report.outcomes.values()) == 3 and report.halted is None


# ----------------------------------------------------------------------------
# Status file
# ----------------------------------------------------------------------------

def test_the_status_file_is_a_view_and_never_an_input(tmp_path):
    harness = Harness(tmp_path, 7, Scenario(batches=4, max_boxes=2, chaos=False))
    harness._boot()
    harness.fair(4)
    ctl = harness.ctl
    ctl.write_status()
    path = harness.root_path / G.STATUS
    doc = json.loads(path.read_text())
    assert doc == json.loads(json.dumps(ctl.status()))
    assert doc['mode'] == MODE_SHADOW and doc['run_id'] == harness.run_id
    assert {'head', 'pending_apply', 'batches', 'boxes', 'cost', 'infra', 'halted', 'journal', 'alerts'} <= set(doc)
    assert {'usd_spent_estimate', 'usd_per_hour_now', 'boxes_alive', 'boxes_total', 'max_cost_usd'} <= set(doc['cost'])
    assert {'max_boxes', 'slots_in_use', 'renting', 'running', 'to_destroy'} <= set(doc['boxes'])
    assert not (harness.root_path / (G.STATUS + '.tmp')).exists()

    # Damage the file. The next start does not read it, and it is rewritten.
    path.write_text('{"head": {"seq": 999}, "halted": null')
    harness._restart()
    report = harness.finish()
    assert sum(report.outcomes.values()) == 4
    harness.ctl.write_status()
    assert json.loads(path.read_text())['head']['seq'] == report.head_seq


# ----------------------------------------------------------------------------
# The threaded loop
# ----------------------------------------------------------------------------

class Locked:
    """One fake, safe for many threads: one call at a time, world-wide."""

    def __init__(self, inner, lock, threads):
        self._inner = inner
        self._lock = lock
        self._threads = threads

    def __getattr__(self, name):
        value = getattr(self._inner, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            with self._lock:
                self._threads.add(threading.current_thread().name)
                return value(*args, **kwargs)

        return call


class Threaded:
    """Runs ``Controller.run`` on its own thread next to a world thread."""

    def __init__(self, tmp_path, seed: int, sc: Scenario, **settings):
        self.harness = Harness(tmp_path, seed, sc)
        self.lock = threading.RLock()
        self.threads = set()
        self.layers = {}
        timing = dict(
            retry_s=0.1, poll_every_s=0.2, ingest_every_s=0.2, sweep_every_s=1.0, status_every_s=5.0,
        )
        timing.update(settings)
        self.harness.config = dataclasses.replace(self.harness.config, **timing)
        self.harness.wrap = self._wrap
        self.failure = []
        self.stop = threading.Event()
        self.loop = None

    def _wrap(self, role, part):
        part = Locked(part, self.lock, self.threads)
        layer = self.layers.get(role)
        return layer(part) if layer is not None else part

    def start(self) -> K.Controller:
        self.harness._boot()
        ctl = self.harness.ctl

        def loop():
            try:
                ctl.run(self.stop, idle_s=0.001)
            except BaseException as exc:
                self.failure.append(exc)

        self.loop = threading.Thread(target=loop, name='controller-loop')
        self.loop.start()
        return ctl

    def world_step(self) -> None:
        """Every part of the world gets its turn. Call with both locks held."""
        harness, world = self.harness, self.harness.world
        while harness.todo:
            harness._arrive()
        while world.late:
            world.land(0)
        for box in harness._boxes():
            if box.job is not None:
                box.job.step()
        world.clock.advance(0.1)

    def until(self, done, timeout_s: float = 60.0) -> bool:
        ctl = self.harness.ctl
        end = time.monotonic() + timeout_s
        while self.loop.is_alive() and time.monotonic() < end:
            # Lock order: the controller's lock, then the world's lock.
            with ctl._lock, self.lock:
                self.world_step()
                if done():
                    return True
            time.sleep(0.002)
        return False

    def halt(self) -> None:
        self.stop.set()
        self.loop.join(30)
        assert not self.loop.is_alive()
        for thread in threading.enumerate():
            if thread.name.startswith('batchctl-'):
                thread.join(30)

    def finish(self):
        """Stop the threads, then let the harness end the run and audit it."""
        self.halt()
        assert not self.failure, self.failure
        return self.harness.finish()


def all_done(harness: Harness, batches: int):
    return lambda: len(harness.ctl.state.done) == batches


def test_the_threaded_loop_decides_every_batch_in_a_world_that_works(tmp_path):
    run = Threaded(tmp_path, 11, Scenario(batches=8, max_boxes=4, chaos=False))
    ctl = run.start()
    try:
        assert run.until(run.harness._quiescent)
    finally:
        run.halt()
    assert (run.harness.root_path / G.STATUS).exists()
    report = run.finish()
    assert report.halted is None
    assert sum(report.outcomes.values()) == 8
    assert report.head_seq == report.outcomes.get('applied', 0) >= 1
    # The effects ran on worker threads; the journal has one writer thread.
    assert {'batchctl-rent', 'batchctl-launch', 'batchctl-poll', 'batchctl-settle', 'batchctl-apply',
            'batchctl-destroy', 'batchctl-ingest', 'batchctl-sweep'} <= run.threads
    assert ctl.status()['alerts']['counts'] == {}
    # More than one box was in use at the same time.
    slots, most = 0, 0
    for rec in journal_records(run.harness):
        if rec['t'] == 'lease_issued':
            slots += 1
        elif rec['t'] in ('decided', 'attempt_failed'):
            slots -= 1
        most = max(most, slots)
    assert most > 1


@pytest.mark.parametrize('seed', range(4))
def test_the_threaded_loop_keeps_every_invariant_in_a_hostile_world(tmp_path, seed):
    sc = Scenario(batches=6, max_boxes=3, fault_rate=0.1, fault_budget=60, chaos=False)
    run = Threaded(tmp_path, 20 + seed, sc)
    run.start()
    try:
        # Kill a box now and then, while the faults of the fake world fire.
        harness = run.harness
        kills = iter(range(3))

        def done():
            boxes = harness._boxes()
            if boxes and harness.world.rng.random() < 0.01 and next(kills, None) is not None:
                boxes[0].alive = False
            return harness._quiescent()

        run.until(done, timeout_s=20.0)
    finally:
        run.halt()
    # The harness ends the run in a fair world and audits every invariant.
    report = run.finish()
    assert sum(report.outcomes.values()) == 6
    assert sum(report.faults.values()) > 0


class Stuck:
    """A provider whose rent call does not come back until the gate opens.

    The request is not lost: when the gate opens, the provider makes the box.
    """

    def __init__(self, inner, world, gate):
        self._inner = inner
        self._world = world
        self._gate = gate
        self.name = inner.name
        self.calls = 0

    def rent(self, *, name, gpu, tier, **_rest):
        with self._inner._lock:
            self._world.audit.controller_rent(self.name, name)
            self.calls += 1
        self._gate.wait(60)
        with self._inner._lock:
            return self._world.providers[self.name].make(name, gpu, tier).info()

    def list(self):
        return self._inner.list()

    def destroy(self, box_id):
        return self._inner.destroy(box_id)


def test_a_provider_that_hangs_does_not_block_the_queue(tmp_path):
    # Rungs 1 and 2 of the ladder are provider p0, and p0 never answers.
    gate = threading.Event()
    run = Threaded(tmp_path, 31, Scenario(batches=3, max_boxes=2, chaos=False), rent_timeout_s=1.0)
    stuck = {}

    def layer(part):
        if part.name != 'p0':
            return part
        stuck['p0'] = Stuck(part, run.harness.world, gate)
        return stuck['p0']

    run.layers['provider'] = layer
    ctl = run.start()
    harness = run.harness
    try:
        assert run.until(all_done(harness, 3))
        # Every batch was decided on a box of the other provider.
        with ctl._lock:
            records = journal_records(harness)
        bound = [rec['b'] for rec in records if rec['t'] == 'box_bound']
        assert bound and {body['provider'] for body in bound} == {'p1'}
        tries = [rec['b']['why'] for rec in records if rec['t'] == 'rent_try']
        assert tries.count('watchdog_timeout') >= 2 * len(bound)
        # A rental that does not come back is not an infra loss of the batch.
        assert not [rec for rec in records if rec['t'] == 'attempt_failed']
        assert stuck['p0'].calls == tries.count('watchdog_timeout')
        assert ctl.status()['alerts']['counts']['effect_watchdog'] == stuck['p0'].calls

        # Now the calls come back, each with a box that nobody waits for.
        gate.set()
        late = lambda: [box for box in harness.world.providers['p0'].boxes.values() if box.ours]
        assert run.until(lambda: len(late()) == stuck['p0'].calls and harness._quiescent())
        assert late() and not [box for box in late() if box.alive]
    finally:
        gate.set()
        run.halt()
    report = run.finish()
    assert sum(report.outcomes.values()) == 3 and report.halted is None
    assert report.kinds['box_doomed'] >= stuck['p0'].calls


class Faulty:
    """An agent whose launch call raises what the test tells it to."""

    def __init__(self, inner, errors, world):
        self._inner = inner
        self._errors = errors
        self._world = world

    def launch(self, box_id, provider, spec):
        if self._errors:
            self._world.bugs['launch'] += 1
            raise self._errors.pop(0)
        return self._inner.launch(box_id, provider, spec)


def test_an_unexpected_error_of_an_effect_is_shown_and_the_effect_runs_again(tmp_path):
    errors = [RuntimeError('boom'), KeyError('boom'), ZeroDivisionError('boom')]
    run = Threaded(tmp_path, 41, Scenario(batches=3, max_boxes=1, chaos=False))
    run.layers['agent'] = lambda part: Faulty(part, errors, run.harness.world)
    ctl = run.start()
    try:
        assert run.until(run.harness._quiescent)
    finally:
        run.halt()
    report = run.finish()
    assert not errors
    assert sum(report.outcomes.values()) == 3 and report.halted is None
    assert 'attempt_failed' not in report.kinds
    status = ctl.status()
    assert status['alerts']['counts'] == {'effect_error': 3}
    assert 'boom' in status['alerts']['recent'][0]['detail']


def test_the_same_unexpected_error_again_and_again_halts_new_work(tmp_path):
    errors = [RuntimeError('boom')] * 5
    run = Threaded(tmp_path, 42, Scenario(batches=3, max_boxes=1, chaos=False))
    run.layers['agent'] = lambda part: Faulty(part, errors, run.harness.world)
    ctl = run.start()
    harness = run.harness
    try:
        assert run.until(lambda: ctl.state.halted is not None)
        assert ctl.state.halted == 'effect_error:launch'
        # The lease that ran into the bug still gets its decision. No new
        # lease starts while the halt stands.
        assert run.until(lambda: not ctl.state.by_fence)
        with ctl._lock:
            kinds = [rec['t'] for rec in journal_records(harness)]
            assert kinds.count('lease_issued') == kinds.count('decided') == 1
            assert 'attempt_failed' not in kinds
            assert len(ctl.state.active) + len(ctl.state.done) == 3
        # An operator looks, and resumes.
        ctl.resume('checked the launch path')
        assert run.until(harness._quiescent)
    finally:
        run.halt()
    report = run.finish()
    assert sum(report.outcomes.values()) == 3 and report.halted is None
    assert report.kinds['halted'] == report.kinds['resumed'] == 1


@pytest.mark.parametrize('error', [
    ShadowWriteError('an effect tried to leave the shadow'),
    C.CoreError('an event that the core cannot place'),
    K.ControllerError('stop'),
    J.JournalError('stop'),
], ids=lambda error: type(error).__name__)
def test_a_fatal_error_of_an_effect_stops_the_loop(tmp_path, error):
    run = Threaded(tmp_path, 43, Scenario(batches=2, max_boxes=1, chaos=False))
    errors = [error]
    run.layers['agent'] = lambda part: Faulty(part, errors, run.harness.world)
    run.start()
    try:
        run.until(lambda: False, timeout_s=20.0)
    finally:
        run.halt()
    # The loop ended by itself, with the error. It did not swallow it.
    assert run.failure == [error]
    count = len(journal_records(run.harness))
    assert 'halted' not in [rec['t'] for rec in journal_records(run.harness)]
    # A new process recovers from the journal and finishes.
    run.harness._restart()
    report = run.harness.finish()
    assert sum(report.outcomes.values()) == 2
    assert report.records > count


def test_too_many_hung_effects_stop_the_loop(tmp_path):
    gate = threading.Event()
    run = Threaded(
        tmp_path, 44, Scenario(batches=12, max_boxes=4, chaos=False), rent_timeout_s=0.5, max_workers=16,
    )
    world = run.harness.world

    def layer(part):
        return Stuck(part, world, gate)

    # Every provider hangs, so every rental ends as a hung worker thread.
    run.layers['provider'] = layer
    run.start()
    try:
        run.until(lambda: False, timeout_s=30.0)
    finally:
        gate.set()
        run.halt()
    assert len(run.failure) == 1 and isinstance(run.failure[0], K.ControllerError)
    assert 'hung' in str(run.failure[0])
    # A new process (new threads) goes on. No batch lost its place.
    run.harness.wrap = lambda role, part: part
    run.harness._restart()
    report = run.harness.finish()
    assert sum(report.outcomes.values()) == 12
