"""Exhaustive state-space checker for the batch controller.

The checker runs the real ``core.decide`` / ``core.resolve`` / ``core.evolve``
/ ``core.plan`` functions. Nothing of the controller logic is copied here.
Only the outside world is a model: the provider, the boxes, R2, and the
results of side effects that are on their way back to the controller.

The world model
---------------
An effect is one step that looks at the world, maybe changes it, and puts one
result "in flight". The result reaches the core in a later step, so any other
step can happen in between. This is how a stale observation is modelled. One
effect key has at most one result in flight, as in the runtime (one worker per
effect key).

What the world may do, in any order
-----------------------------------
* A rent try returns a box, returns "no offer", or times out. After a timeout
  the box may exist anyway (late rent). The provider may also create a second
  box with the same name.
* A box dies at any step. A box whose lease is gone keeps working (zombie) and
  may upload again after its staging prefix was cleaned.
* A box finishes with a GREEN package, a RED package, or a corrupt package.
* A lease expires while its box is healthy or slow (spurious expiry).
* A result in flight is lost. A controller crash is the loss of every result
  in flight; the journal state stays. A result is delivered twice.
* Another writer moves the HEAD pointer (optional; checks the fail-stop).

What is proven
--------------
Safety is checked on every reachable state and every transition. Liveness is
checked on the full graph. Fault steps, lost results and duplicate deliveries
are the only steps that are not "fair". The checker proves that the graph of
fair steps has no cycle and that every state with no fair step left is fully
done: every batch is terminal, every box is destroyed, every outcome is
published. So every run with a finite number of faults ends with every batch
terminal.

The proof is exhaustive for the configuration that was run, and only for it:
the number of batches, boxes and faults is bounded.

Two reductions keep the state space small
-----------------------------------------
1. Results that can never matter again are removed from the in-flight set. The
   checker asserts that each removed result is a no-op for the core, and it
   asserts the facts that keep it a no-op (fences, rent tries and the HEAD
   sequence only grow).
2. ``coarse`` mode is for runs with more than one batch. The result of a
   box-local effect (rent, launch, poll, settle) reaches the core in the same
   step, and clean-up runs at once. The results of the apply step stay in
   flight. This loses nothing between batches: the content of a box-local
   result does not depend on any other batch, and the steps of different
   batches still interleave in every order. What it does not explore is a
   stale box-local result. The full mode explores that, with one batch (and
   with two batches and one fault).

Run ``python -m ops.parallel_eval.pit.batchctl.modelcheck --help``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from array import array
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from . import core as C

# Kinds of complete package. ``GREEN_NO_STATE`` has a valid GREEN result, but
# its state files are missing or damaged.
GOOD_GREEN, GOOD_RED, BAD, GREEN_NO_STATE = 1, 2, 3, 4
PROVIDER = 'p'
GENESIS_STATE = 'state-0'
PHASE_RANK = {C.Phase.RENTING: 0, C.Phase.RUNNING: 1, C.Phase.SETTLING: 2}

INVARIANTS = (
    'HEAD chain is linear: seq grows by one, each link extends the tip',
    'no double apply: a batch is in the chain at most once',
    'no apply of unverified state: every applied state was verified in R2',
    'an applied state was trained on the HEAD it replaced',
    'the applied batch is the first GREEN against its HEAD',
    'a GREEN with a verified state is CREDITED only when an earlier GREEN on the same base won the apply',
    'a GREEN result without a verified state is credited and never applied',
    'no lost GREEN: a GREEN decision always ends APPLIED or CREDITED',
    'RED only from a verified RED result; LOST_INFRA only after N infra failures',
    'a decision uses only the package of the live fence (fencing)',
    'a package with a valid result is never rejected as invalid',
    'terminal outcomes never change; published outcomes match the journal',
    'queue order: the oldest queued batch gets the next lease',
    'a batch keeps its queue position; its infra counter never goes down',
    'a box that a running lease needs is never destroyed by the controller',
    'the controller never writes the pointer after another writer moved it',
    'liveness: no cycle of fair steps; every fair sink is fully done',
)


@dataclass(frozen=True)
class ModelConfig:
    batches: int = 2
    max_boxes: int = 2
    max_infra_failures: int = 2
    rungs: int = 2
    # ``faults`` is the total number of fault steps in one run. Each kind of
    # fault also has its own cap. A fault step needs both to be above zero.
    faults: int = 2
    rent_faults: int = 2
    box_deaths: int = 2
    corrupt_uploads: int = 2
    spurious_expiries: int = 2
    duplicates: int = 2
    foreign_writes: int = 0
    fault_reports: int = 0
    green: bool = True
    red: bool = True
    lose_results: bool = True
    atomic_housekeeping: bool = True
    # ``coarse`` delivers the result of a box-local effect at once and cleans
    # up at once. Only the results of the apply step stay in flight. See the
    # module text for why this keeps every cross-batch behaviour.
    coarse: bool = False
    max_states: int = 20_000_000
    progress: int = 0


class Budget(NamedTuple):
    total: int
    rent: int
    death: int
    corrupt: int
    expiry: int
    dup: int
    foreign: int
    fault: int

    def has(self, kind: str) -> bool:
        return self.total > 0 and getattr(self, kind) > 0

    def spend(self, kind: str) -> 'Budget':
        return self._replace(total=self.total - 1, **{kind: getattr(self, kind) - 1})


def state_of(fence: int) -> str:
    return 'state-%d' % fence


def result_of(fence: int) -> str:
    return 'result-%d' % fence


def box_fence_try(box_id: str) -> Tuple[int, int]:
    fence, try_no, _gen = box_id[1:].split('.')
    return int(fence), int(try_no)


def op_key(msg: Any) -> Tuple:
    """The effect that made this result. One result in flight per effect."""
    kind = type(msg)
    if kind is C.RentOk or kind is C.RentFailed:
        return ('rent', msg.fence, msg.try_no)
    if kind is C.Launched:
        return ('launch', msg.fence)
    if kind is C.Polled:
        return ('poll', msg.fence)
    if kind is C.Settled or kind is C.SettleInvalid:
        return ('settle', msg.fence)
    if kind is C.ApplyDone or kind is C.ApplyDiverged:
        return ('apply', msg.seq)
    if kind is C.BoxGone:
        return ('destroy', msg.box_id)
    if kind is C.OrphanSeen:
        return ('sweep', msg.box_id)
    if kind is C.StagingCleared:
        return ('gc', msg.fence)
    if kind is C.Published:
        return ('publish', msg.batch_id)
    raise AssertionError('no effect key for %r' % (msg,))


def never_again(core: C.State, msg: Any) -> bool:
    """True when the result can never produce a record again.

    Each rule rests on a fact that only grows: a fence is issued once, a rent
    try number only grows, a lease phase only moves forward, the HEAD sequence
    only grows. ``check_step`` asserts these facts on every transition.
    """
    kind = type(msg)
    if kind in (C.RentFailed, C.Launched, C.Polled, C.Settled, C.SettleInvalid):
        batch = core.leased(msg.fence)
        if batch is None:
            return True
        if kind is C.RentFailed:
            return batch.phase is not C.Phase.RENTING or batch.lease.try_no > msg.try_no
        if kind is C.Launched:
            return batch.phase is C.Phase.SETTLING or batch.lease.launched
        if kind is C.Polled:
            return batch.phase is C.Phase.SETTLING
        return False
    if kind is C.ApplyDone or kind is C.ApplyDiverged:
        if core.head is not None and core.head.seq >= msg.seq:
            return True
        # The model has no operator, so a halt is permanent.
        return kind is C.ApplyDiverged and core.halted is not None
    if kind is C.StagingCleared:
        return (msg.batch_id, msg.fence) not in core.dirty
    if kind is C.Published:
        return msg.batch_id not in core.unpublished
    # RentOk, OrphanSeen and BoxGone can matter again; they stay in flight.
    return False


class World:
    __slots__ = (
        'core', 'boxes', 'staging', 'pointer', 'chain', 'canon', 'published', 'msgs',
        'arrived', 'budget', 'gens', 'diverged', 'faulted',
    )

    def __init__(self) -> None:
        self.core = C.State()
        self.boxes: Dict[str, int] = {}          # alive boxes: box_id -> job (0 none, 1 running, 2 done)
        self.staging: Dict[int, int] = {}        # fence -> complete package kind
        self.pointer: Optional[Tuple[int, str, str]] = None
        self.chain: Tuple[Tuple[int, Optional[str], str, str], ...] = ()
        self.canon: frozenset = frozenset()
        self.published: frozenset = frozenset()
        self.msgs: frozenset = frozenset()
        self.arrived = 0
        self.budget = Budget(0, 0, 0, 0, 0, 0, 0, 0)
        self.gens: Dict[Tuple[int, int], int] = {}
        self.diverged = False
        self.faulted = False

    def clone(self) -> 'World':
        other = World.__new__(World)
        other.core = self.core.clone()
        other.boxes = dict(self.boxes)
        other.staging = dict(self.staging)
        other.pointer = self.pointer
        other.chain = self.chain
        other.canon = self.canon
        other.published = self.published
        other.msgs = self.msgs
        other.arrived = self.arrived
        other.budget = self.budget
        other.gens = dict(self.gens)
        other.diverged = self.diverged
        other.faulted = self.faulted
        return other

    def key(self) -> Tuple:
        return (
            self.core.key(),
            tuple(sorted(self.boxes.items())),
            tuple(sorted(self.staging.items())),
            self.pointer, self.chain,
            tuple(sorted(self.canon)),
            tuple(sorted(self.published)),
            tuple(sorted(repr(m) for m in self.msgs)),
            self.arrived, tuple(self.budget),
            tuple(sorted(self.gens.items())),
            self.diverged, self.faulted,
        )


def digest(key: Tuple) -> bytes:
    return hashlib.blake2b(repr(key).encode(), digest_size=16).digest()


Step = Tuple[str, bool, World, List[C.Record]]


class ModelError(AssertionError):
    """The model itself is wrong (not the controller)."""


class Model:
    def __init__(self, mc: ModelConfig):
        self.mc = mc
        self.cfg = C.CoreConfig(
            max_boxes=mc.max_boxes,
            max_infra_failures=mc.max_infra_failures,
            ladder=tuple(C.Rung(PROVIDER, 'gpu%d' % i, 'tier') for i in range(mc.rungs)),
        )
        self.batch_ids = tuple('b%d' % i for i in range(mc.batches))

    def initial(self) -> World:
        w = World()
        w.budget = Budget(
            self.mc.faults, self.mc.rent_faults, self.mc.box_deaths, self.mc.corrupt_uploads,
            self.mc.spurious_expiries, self.mc.duplicates, self.mc.foreign_writes,
            self.mc.fault_reports,
        )
        return w

    # -- helpers -------------------------------------------------------------

    def _settled_world(self, w: World) -> World:
        """Drop the results in flight that can never matter again."""
        if w.msgs:
            dead = [m for m in w.msgs if never_again(w.core, m)]
            if dead:
                for msg in dead:
                    if C.decide(self.cfg, w.core, msg) is not None:
                        raise ModelError('a result was dropped as dead but the core still acts on it: %r' % (msg,))
                w.msgs = w.msgs - frozenset(dead)
        return w

    def _emit(self, w: World, msg: Any) -> World:
        w2 = w.clone()
        w2.msgs = w2.msgs | {msg}
        return w2

    def _named(self, w: World, fence: int, try_no: int) -> Optional[str]:
        prefix = 'x%d.%d.' % (fence, try_no)
        found = sorted(box_id for box_id in w.boxes if box_id.startswith(prefix))
        return found[0] if found else None

    def _new_box(self, w2: World, fence: int, try_no: int) -> str:
        name = (fence, try_no)
        gen = w2.gens.get(name, 0)
        box_id = 'x%d.%d.%d' % (fence, try_no, gen)
        w2.gens[name] = gen + 1
        w2.boxes[box_id] = 0
        return box_id

    # -- transitions ---------------------------------------------------------

    def _post(self, w2: World, msg: Any, label: str, fair: bool, out: List[Step], local: bool) -> None:
        """Hand the result of an effect to the core, at once or in flight."""
        at_once = self.mc.coarse if local else self.mc.atomic_housekeeping
        if at_once:
            recs = C.step(self.cfg, w2.core, msg)
            out.append((label, fair, w2, recs))
        else:
            w2.msgs = w2.msgs | {msg}
            out.append((label, fair, w2, []))

    def successors(self, w: World) -> List[Step]:
        out: List[Step] = []
        core = w.core
        cfg = self.cfg
        busy = {op_key(m) for m in w.msgs}

        if core.head is None and core.pending is None:
            w2 = w.clone()
            w2.canon = w2.canon | {GENESIS_STATE}
            recs = C.step(cfg, w2.core, C.GenesisVerified(GENESIS_STATE))
            out.append(('genesis', True, w2, recs))

        if w.arrived < len(self.batch_ids):
            w2 = w.clone()
            recs = C.step(cfg, w2.core, C.Arrived(self.batch_ids[w.arrived], 'miner'))
            w2.arrived += 1
            out.append(('arrive', True, w2, recs))

        w2 = w.clone()
        recs = C.step(cfg, w2.core, C.Tick())
        if recs:
            out.append(('tick', True, w2, recs))

        for op in C.plan(cfg, core):
            kind = type(op)
            if kind is C.RentOp:
                if ('rent', op.fence, op.try_no) not in busy:
                    self._rent(w, op, out)
            elif kind is C.LaunchOp:
                if ('launch', op.fence) not in busy and op.box_id in w.boxes:
                    w2 = w.clone()
                    if w2.boxes[op.box_id] == 0:
                        w2.boxes[op.box_id] = 1
                    self._post(w2, C.Launched(op.fence), 'launch', True, out, local=True)
            elif kind is C.PollOp:
                if ('poll', op.fence) not in busy:
                    self._poll(w, op, out)
            elif kind is C.SettleOp:
                if ('settle', op.fence) not in busy:
                    self._settle(w, op, out)
            elif kind is C.ApplyOp:
                if ('apply', op.seq) not in busy:
                    self._apply(w, op, out)
            elif self.mc.coarse:
                continue
            elif kind is C.DestroyOp:
                if ('destroy', op.box_id) not in busy:
                    if op.box_id in core.live_box_ids():
                        raise ModelError('the plan destroys a box that a running lease needs')
                    w2 = w.clone()
                    w2.boxes.pop(op.box_id, None)
                    self._post(w2, C.BoxGone(op.box_id), 'destroy', True, out, local=False)
            elif kind is C.GcOp:
                if ('gc', op.fence) not in busy:
                    w2 = w.clone()
                    w2.staging.pop(op.fence, None)
                    self._post(w2, C.StagingCleared(op.batch_id, op.fence), 'gc', True, out, local=False)
            elif kind is C.PublishOp:
                if ('publish', op.batch_id) not in busy:
                    w2 = w.clone()
                    batch = core.done[op.batch_id]
                    w2.published = w2.published | {(op.batch_id, batch.phase.value)}
                    self._post(w2, C.Published(op.batch_id), 'publish', True, out, local=False)

        if not self.mc.coarse:
            # Reconcile sweep: boxes with our prefix that no lease needs.
            for box_id in self._orphans(w):
                if ('sweep', box_id) not in busy:
                    fence, try_no = box_fence_try(box_id)
                    self._post(w.clone(), C.OrphanSeen(box_id, PROVIDER, fence, try_no), 'sweep-box', True, out, local=False)
            # Reconcile sweep: staging prefixes of fences that are not live.
            for fence in self._stale_staging(w):
                w2 = w.clone()
                del w2.staging[fence]
                out.append(('sweep-staging', True, w2, []))

        # Boxes work on their own, with or without a lease.
        for box_id, job in w.boxes.items():
            fence, try_no = box_fence_try(box_id)
            matters = fence in core.by_fence
            if job == 1 and fence not in w.staging:
                kinds = []
                if self.mc.green:
                    kinds.append((GOOD_GREEN, True, 'box-green'))
                if self.mc.red and (matters or not kinds):
                    kinds.append((GOOD_RED, True, 'box-red'))
                # The core never reads the staging prefix of a dead fence, so
                # one kind of package is enough for a zombie box.
                if matters and w.budget.has('corrupt'):
                    kinds.append((BAD, False, 'box-corrupt'))
                    if self.mc.green:
                        kinds.append((GREEN_NO_STATE, False, 'box-green-no-state'))
                for status, fair, label in kinds:
                    w2 = w.clone()
                    w2.staging[fence] = status
                    w2.boxes[box_id] = 2
                    if not fair:
                        w2.budget = w.budget.spend('corrupt')
                    out.append((label, fair, w2, []))
            # A dead zombie does less than a live one, so only the death of a
            # box that a lease still needs is a new behaviour.
            if matters and w.budget.has('death'):
                w2 = w.clone()
                del w2.boxes[box_id]
                w2.budget = w.budget.spend('death')
                out.append(('box-dies', False, w2, []))

        for msg in w.msgs:
            name = type(msg).__name__
            w2 = w.clone()
            w2.msgs = w2.msgs - {msg}
            recs = C.step(cfg, w2.core, msg)
            out.append(('deliver:%s' % name, True, w2, recs))
            if recs and w.budget.has('dup'):
                w3 = w.clone()
                recs3 = C.step(cfg, w3.core, msg)
                w3.budget = w.budget.spend('dup')
                out.append(('duplicate:%s' % name, False, w3, recs3))
            if self.mc.lose_results:
                w4 = w.clone()
                w4.msgs = w4.msgs - {msg}
                out.append(('lose:%s' % name, False, w4, []))

        if w.budget.has('foreign') and w.pointer is not None:
            w2 = w.clone()
            w2.pointer = (w.pointer[0] + 1, 'foreign-link', 'foreign-state')
            w2.diverged = True
            w2.budget = w.budget.spend('foreign')
            out.append(('foreign-pointer-write', False, w2, []))

        # An effect finds durable data that is wrong and reports a fault.
        if w.budget.has('fault') and core.halted is None and core.head is not None:
            w2 = w.clone()
            recs = C.step(cfg, w2.core, C.Fault('injected'))
            w2.faulted = True
            w2.budget = w.budget.spend('fault')
            out.append(('fault-report', False, w2, recs))

        if self.mc.coarse:
            for _label, _fair, nxt, recs in out:
                self._clean_up(nxt, recs)
        return [(label, fair, self._settled_world(nxt), recs) for label, fair, nxt, recs in out]

    @staticmethod
    def _open_try(core: C.State, fence: int, try_no: int) -> bool:
        batch = core.leased(fence)
        return batch is not None and batch.phase is C.Phase.RENTING and batch.lease.try_no == try_no

    def _orphans(self, w: World) -> List[str]:
        core = w.core
        live = core.live_box_ids()
        found = []
        for box_id in w.boxes:
            if box_id in live or box_id in core.doomed:
                continue
            if not self._open_try(core, *box_fence_try(box_id)):
                found.append(box_id)
        return found

    @staticmethod
    def _stale_staging(w: World) -> List[int]:
        dirty_fences = {f for _b, f in w.core.dirty}
        return [f for f in w.staging if f not in w.core.by_fence and f not in dirty_fences]

    def _clean_up(self, w: World, recs: List[C.Record]) -> None:
        """Coarse mode: run every clean-up effect to its end, in place."""
        core = w.core
        cfg = self.cfg
        while True:
            if core.doomed:
                box_id = min(core.doomed)
                if box_id in core.live_box_ids():
                    raise ModelError('the plan destroys a box that a running lease needs')
                w.boxes.pop(box_id, None)
                recs.extend(C.step(cfg, core, C.BoxGone(box_id)))
            elif core.dirty:
                batch_id, fence = min(core.dirty)
                w.staging.pop(fence, None)
                recs.extend(C.step(cfg, core, C.StagingCleared(batch_id, fence)))
            elif core.unpublished:
                batch_id = min(core.unpublished)
                w.published = w.published | {(batch_id, core.done[batch_id].phase.value)}
                recs.extend(C.step(cfg, core, C.Published(batch_id)))
            else:
                orphans = self._orphans(w)
                stale = self._stale_staging(w)
                if orphans:
                    fence, try_no = box_fence_try(orphans[0])
                    recs.extend(C.step(cfg, core, C.OrphanSeen(orphans[0], PROVIDER, fence, try_no)))
                elif stale:
                    del w.staging[stale[0]]
                else:
                    return

    def _rent(self, w: World, op: C.RentOp, out: List[Step]) -> None:
        existing = self._named(w, op.fence, op.try_no)
        w2 = w.clone()
        box_id = existing if existing is not None else self._new_box(w2, op.fence, op.try_no)
        self._post(w2, C.RentOk(op.fence, op.try_no, box_id, PROVIDER), 'rent-ok', True, out, local=True)
        if not w.budget.has('rent'):
            return
        spent = w.budget.spend('rent')
        if existing is None:
            w3 = w.clone()
            w3.budget = spent
            self._post(w3, C.RentFailed(op.fence, op.try_no, 'unavailable'), 'rent-unavailable', False, out, local=True)
            # The call timed out, but the provider made the box anyway.
            w4 = w.clone()
            w4.budget = spent
            self._new_box(w4, op.fence, op.try_no)
            self._post(w4, C.RentFailed(op.fence, op.try_no, 'timeout'), 'rent-timeout-late-box', False, out, local=True)
        else:
            # The provider made a second box with the same name.
            w5 = w.clone()
            w5.budget = spent
            twin = self._new_box(w5, op.fence, op.try_no)
            self._post(w5, C.RentOk(op.fence, op.try_no, twin, PROVIDER), 'rent-twin-box', False, out, local=True)
            w6 = w.clone()
            w6.budget = spent
            self._post(w6, C.RentFailed(op.fence, op.try_no, 'timeout'), 'rent-timeout', False, out, local=True)

    def _poll(self, w: World, op: C.PollOp, out: List[Step]) -> None:
        if op.fence in w.staging:
            self._post(w.clone(), C.Polled(op.fence, complete=True), 'poll-complete', True, out, local=True)
            return
        if op.box_id not in w.boxes:
            self._post(w.clone(), C.Polled(op.fence, gone=True), 'poll-gone', True, out, local=True)
            self._post(w.clone(), C.Polled(op.fence, expired=True), 'poll-expired', True, out, local=True)
        elif w.budget.has('expiry'):
            w2 = w.clone()
            w2.budget = w.budget.spend('expiry')
            self._post(w2, C.Polled(op.fence, expired=True), 'poll-spurious-expiry', False, out, local=True)

    def _settle(self, w: World, op: C.SettleOp, out: List[Step]) -> None:
        status = w.staging.get(op.fence)
        w2 = w.clone()
        if status == GOOD_GREEN:
            w2.canon = w2.canon | {state_of(op.fence)}
            msg: Any = C.Settled(op.fence, C.VERDICT_GREEN, result_of(op.fence), state_of(op.fence))
        elif status == GREEN_NO_STATE:
            msg = C.Settled(op.fence, C.VERDICT_GREEN, result_of(op.fence), None, (), 'state_invalid')
        elif status == GOOD_RED:
            msg = C.Settled(op.fence, C.VERDICT_RED, result_of(op.fence))
        else:
            msg = C.SettleInvalid(op.fence, 'corrupt' if status == BAD else 'missing')
        self._post(w2, msg, 'settle', True, out, local=True)

    def _apply(self, w: World, op: C.ApplyOp, out: List[Step]) -> None:
        done = C.ApplyDone(op.seq, op.link_id)
        if w.pointer is None:
            if op.parent_link is not None:
                out.append(('apply-diverged', True, self._emit(w, C.ApplyDiverged(op.seq, op.link_id, 'no pointer')), []))
                return
            w2 = self._emit(w, done)
        elif w.pointer[1] == op.link_id:
            out.append(('apply-confirm', True, self._emit(w, done), []))
            return
        elif w.pointer[1] == op.parent_link:
            w2 = self._emit(w, done)
        else:
            out.append(('apply-diverged', True, self._emit(w, C.ApplyDiverged(op.seq, op.link_id, 'pointer moved')), []))
            return
        if op.state_id not in w.canon:
            raise ModelError('the plan applies a state that was never verified')
        # Compare-and-swap succeeded: this is the only place the pointer moves.
        w2.pointer = (op.seq, op.link_id, op.state_id)
        w2.chain = w2.chain + ((op.seq, op.batch_id, op.state_id, op.link_id),)
        out.append(('apply-cas', True, w2, []))

    # -- invariants ----------------------------------------------------------

    def check_state(self, w: World) -> Optional[str]:
        core = w.core
        chain = w.chain
        for i, (seq, _bid, state_id, _link) in enumerate(chain):
            if seq != i:
                return 'HEAD chain is not linear'
            if state_id not in w.canon:
                return 'applied a state that was never verified'
        applied = [bid for (_s, bid, _st, _l) in chain if bid is not None]
        if len(applied) != len(set(applied)):
            return 'a batch was applied twice'
        if not w.diverged:
            tip = (chain[-1][0], chain[-1][3], chain[-1][2]) if chain else None
            if w.pointer != tip:
                return 'the pointer is not the tip of the chain'
        if core.head is not None:
            if core.head.seq >= len(chain) or chain[core.head.seq][3] != core.head.link_id:
                return 'the journal HEAD is not in the R2 chain'
            ahead = len(chain) - 1 - core.head.seq
            if ahead > 1 or (ahead == 1 and (core.pending is None or chain[-1][3] != core.pending.link_id)):
                return 'the R2 chain is ahead of the journal without a matching intent'
        elif chain and (len(chain) > 1 or core.pending is None or chain[0][3] != core.pending.link_id):
            return 'genesis in R2 without a matching intent'
        for i in range(1, len(chain)):
            batch = core.batch(chain[i][1])
            decision = batch.decision if batch is not None else None
            if (
                decision is None or decision.base_seq != i - 1
                or decision.base_state != chain[i - 1][2] or decision.state_id != chain[i][2]
            ):
                return 'an applied state was not trained on the HEAD it replaced'
        limit = self.cfg.max_infra_failures
        fences = []
        for batch in list(core.active.values()) + list(core.done.values()):
            phase = batch.phase
            decision = batch.decision
            green = decision is not None and decision.verdict == C.VERDICT_GREEN
            if green and phase not in (C.Phase.GREEN, C.Phase.APPLYING, C.Phase.APPLIED, C.Phase.CREDITED):
                return 'lost GREEN: %s' % batch.batch_id
            if phase in (C.Phase.GREEN, C.Phase.APPLYING, C.Phase.APPLIED, C.Phase.CREDITED) and not green:
                return 'paid as GREEN without a verified GREEN decision'
            if phase is C.Phase.RED and (decision is None or decision.verdict != C.VERDICT_RED):
                return 'RED without a verified RED decision'
            if phase is C.Phase.LOST_INFRA and (batch.infra_failures != limit or decision is not None):
                return 'LOST_INFRA without exactly N infra failures'
            if phase is not C.Phase.LOST_INFRA and batch.infra_failures >= limit:
                return 'a batch went on after N infra failures'
            if phase is C.Phase.APPLIED and (
                batch.head_seq is None or batch.head_seq >= len(chain)
                or chain[batch.head_seq][1] != batch.batch_id
            ):
                return 'APPLIED in the journal but not in the R2 chain'
            if phase is C.Phase.APPLYING and (core.pending is None or core.pending.batch_id != batch.batch_id):
                return 'APPLYING without the pending intent'
            if phase in (C.Phase.APPLYING, C.Phase.APPLIED) and decision.state_id is None:
                return 'applied a GREEN that has no verified state'
            if (phase in C.TERMINAL) != (batch.batch_id in core.done):
                return 'terminal bookkeeping is wrong'
            leased = phase in (C.Phase.RENTING, C.Phase.RUNNING, C.Phase.SETTLING)
            if leased != (batch.lease is not None):
                return 'lease bookkeeping is wrong'
            if batch.lease is not None:
                fences.append(batch.lease.fence)
            if phase is C.Phase.CREDITED and not self._beaten(core, batch):
                return 'CREDITED, but no earlier GREEN took the HEAD after its base'
        if len(fences) != len(set(fences)) or set(fences) != set(core.by_fence):
            return 'two leases share a fence'
        holding = sum(
            1 for b in core.active.values() if b.phase in (C.Phase.RENTING, C.Phase.RUNNING)
        )
        if holding > self.cfg.max_boxes:
            return 'more boxes in use than max_boxes'
        if core.live_box_ids() & set(core.doomed):
            return 'a box that a running lease needs is marked for destruction'
        seen = set()
        for batch_id, phase in w.published:
            batch = core.done.get(batch_id)
            if batch is None or batch.phase.value != phase or batch_id in seen:
                return 'a published outcome differs from the journal'
            seen.add(batch_id)
        return None

    @staticmethod
    def _beaten(core: C.State, batch: C.Batch) -> bool:
        """True when the batch has no verified state, or an earlier GREEN on
        the same base won the apply."""
        mine = batch.decision
        if mine.state_id is None:
            return True
        for other in list(core.active.values()) + list(core.done.values()):
            d = other.decision
            if (
                other is not batch and d is not None and d.verdict == C.VERDICT_GREEN
                and d.base_seq == mine.base_seq and d.order < mine.order
                and other.phase in (C.Phase.APPLYING, C.Phase.APPLIED)
            ):
                return True
        return False

    def check_step(self, w: World, recs: List[C.Record], w2: World) -> Optional[str]:
        before = w.core
        after = w2.core
        for kind, body in recs:
            if kind == 'lease_issued':
                queued = [b.seq for b in before.active.values() if b.phase is C.Phase.QUEUED]
                if before.active[body['batch_id']].seq != min(queued):
                    return 'queue order violated'
            elif kind == 'decided':
                fence = body['fence']
                batch = before.leased(fence)
                if batch is None or batch.phase is not C.Phase.SETTLING:
                    return 'decision from a fence that is not live'
                if body['verdict'] != C.VERDICT_GREEN:
                    want = GOOD_RED
                elif body['state_id'] is None:
                    want = GREEN_NO_STATE
                else:
                    want = GOOD_GREEN
                if w.staging.get(fence) != want:
                    return 'decision does not match the package in R2'
                if want == GOOD_GREEN and body['state_id'] not in w2.canon:
                    return 'GREEN decision without a verified state'
            elif kind == 'attempt_failed' and body['reason'].startswith('invalid_upload'):
                if w.staging.get(body['fence']) in (GOOD_GREEN, GOOD_RED, GREEN_NO_STATE):
                    return 'a package with a valid result was rejected as invalid'
            elif kind == 'apply_intent' and body['batch_id'] is not None:
                winner = after.batch(body['batch_id']).decision
                for other in list(after.active.values()) + list(after.done.values()):
                    d = other.decision
                    if (
                        d is not None and d.verdict == C.VERDICT_GREEN and d.state_id is not None
                        and d.base_seq == winner.base_seq and d.order < winner.order
                    ):
                        return 'the apply winner is not the first GREEN against its HEAD'
        for batch_id, batch in before.done.items():
            other = after.done.get(batch_id)
            if other is None or other.key() != batch.key():
                return 'a terminal batch changed'
        for batch_id, batch in before.active.items():
            other = after.batch(batch_id)
            if other is None or other.seq != batch.seq or other.infra_failures < batch.infra_failures:
                return 'a batch vanished or lost its queue position'
        if w2.chain[:len(w.chain)] != w.chain:
            return 'the chain was rewritten'
        if w.diverged and len(w2.chain) != len(w.chain):
            return 'the controller wrote the pointer after another writer moved it'
        return self._check_monotonic(before, after)

    def _check_monotonic(self, before: C.State, after: C.State) -> Optional[str]:
        """The facts that ``never_again`` rests on."""
        if after.next_fence < before.next_fence:
            return 'the fence counter went down'
        for fence in after.by_fence:
            if fence not in before.by_fence and fence < before.next_fence:
                return 'a fence was issued twice'
        for fence, batch_id in before.by_fence.items():
            if fence not in after.by_fence:
                continue
            old = before.active[batch_id]
            new = after.active.get(after.by_fence[fence])
            if new is None or new.batch_id != batch_id:
                return 'a fence moved to another batch'
            if PHASE_RANK[new.phase] < PHASE_RANK[old.phase]:
                return 'a lease phase moved backwards'
            if new.lease.try_no < old.lease.try_no or (old.lease.launched and not new.lease.launched):
                return 'a rent try or a launch flag moved backwards'
        if before.head is not None and (after.head is None or after.head.seq < before.head.seq):
            return 'the HEAD sequence went down'
        if before.halted is not None and after.halted is None:
            return 'a halt ended without an operator'
        return None

    def sink_problem(self, w: World) -> Optional[str]:
        core = w.core
        if core.halted is not None:
            if not (w.diverged or w.faulted):
                return 'halted without a foreign pointer write or a fault report'
            # A halt stops new work only. Every lease that ran went on to a
            # decision or a retry, and the boxes and staging are cleaned up.
            if core.by_fence or w.msgs or w.boxes or w.staging:
                return 'leases, results, boxes or staging left at a halted sink'
            if core.doomed or core.dirty or core.unpublished:
                return 'housekeeping left at a halted sink'
            return None
        if core.active:
            return 'batches not terminal: %s' % sorted(
                (b.batch_id, b.phase.value) for b in core.active.values()
            )
        if w.arrived < len(self.batch_ids) or core.pending is not None or core.head is None:
            return 'work left at a sink'
        if w.msgs or w.boxes or w.staging:
            return 'results, boxes or staging left at a sink'
        if core.doomed or core.dirty or core.unpublished:
            return 'housekeeping left at a sink'
        if len(w.published) != len(self.batch_ids):
            return 'not every outcome was published'
        return None


@dataclass
class Report:
    ok: bool
    states: int
    transitions: int
    fair_transitions: int
    sinks: int
    outcomes: Dict[str, int]
    seconds: float
    violation: Optional[str] = None
    trace: Optional[List[str]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _trace(parents: List[Tuple[int, str]], node: int) -> List[str]:
    labels = []
    while node > 0:
        node, label = parents[node][0], parents[node][1]
        labels.append(label)
    return list(reversed(labels))


def _find_fair_cycle(adj: List[array]) -> Optional[int]:
    """Return a state on a cycle of fair steps, or None."""
    color = bytearray(len(adj))
    for root in range(len(adj)):
        if color[root]:
            continue
        stack = [(root, 0)]
        color[root] = 1
        while stack:
            node, i = stack[-1]
            edges = adj[node]
            if i < len(edges):
                stack[-1] = (node, i + 1)
                nxt = edges[i]
                if color[nxt] == 1:
                    return nxt
                if color[nxt] == 0:
                    color[nxt] = 1
                    stack.append((nxt, 0))
            else:
                color[node] = 2
                stack.pop()
    return None


def run(mc: ModelConfig) -> Report:
    """Explore every reachable state. Stops at the first violation."""
    started = time.time()
    model = Model(mc)
    root = model.initial()
    ids: Dict[bytes, int] = {digest(root.key()): 0}
    parents: List[Tuple[int, str]] = [(-1, '')]
    adj: List[array] = [array('I')]
    stack: List[Tuple[int, World]] = [(0, root)]
    transitions = 0
    fair_transitions = 0
    sinks = 0
    outcomes: Dict[str, int] = {}

    def fail(reason: str, node: int, extra: Optional[str] = None) -> Report:
        trace = _trace(parents, node)
        if extra is not None:
            trace.append(extra)
        return Report(False, len(ids), transitions, fair_transitions, sinks, outcomes,
                      time.time() - started, reason, trace)

    while stack:
        node, world = stack.pop()
        key = world.key()
        fair_out = array('I')
        try:
            steps = model.successors(world)
        except C.CoreError as exc:
            return fail('core refused its own record: %s' % exc, node)
        except ModelError as exc:
            return fail(str(exc), node)
        for label, fair, nxt, recs in steps:
            nkey = nxt.key()
            if nkey == key:
                continue
            transitions += 1
            problem = model.check_step(world, recs, nxt)
            if problem is not None:
                return fail(problem, node, label)
            d = digest(nkey)
            nid = ids.get(d)
            if nid is None:
                nid = len(ids)
                ids[d] = nid
                parents.append((node, label))
                adj.append(array('I'))
                problem = model.check_state(nxt)
                if problem is not None:
                    return fail(problem, nid)
                if len(ids) > mc.max_states:
                    return fail('state limit reached; the result is not a proof', nid)
                if mc.progress and len(ids) % mc.progress == 0:
                    print('states=%d stack=%d seconds=%.0f' % (len(ids), len(stack), time.time() - started),
                          file=sys.stderr, flush=True)
                stack.append((nid, nxt))
            if fair:
                fair_transitions += 1
                fair_out.append(nid)
        adj[node] = fair_out
        if not fair_out:
            sinks += 1
            problem = model.sink_problem(world)
            if problem is not None:
                return fail('fair sink is not done: %s' % problem, node)
            tally = sorted(b.phase.value for b in world.core.done.values())
            name = 'halted' if world.core.halted is not None else ','.join(tally)
            outcomes[name] = outcomes.get(name, 0) + 1
    cycle = _find_fair_cycle(adj)
    if cycle is not None:
        return fail('cycle of fair steps: a run without faults can loop forever', cycle)
    return Report(True, len(ids), transitions, fair_transitions, sinks, outcomes, time.time() - started)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Exhaustive checker for the batch controller core.')
    defaults = ModelConfig()
    for name, value in asdict(defaults).items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ('1', 'true', 'yes'), default=value)
        else:
            parser.add_argument(flag, type=int, default=value)
    args = parser.parse_args(argv)
    mc = ModelConfig(**vars(args))
    report = run(mc)
    print(json.dumps({'config': asdict(mc), 'report': report.as_dict(), 'invariants': list(INVARIANTS)}, indent=1))
    return 0 if report.ok else 1


if __name__ == '__main__':
    sys.exit(main())
