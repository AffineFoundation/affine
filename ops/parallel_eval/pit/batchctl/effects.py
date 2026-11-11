"""Side effects of the batch controller.

One method per kind of work. Each method reads the world, changes it in an
idempotent way, and returns events for the core. No method touches the
journal or the state.

Rules that every effect keeps
-----------------------------
* An error with an unknown outcome returns no event. The effect runs again
  and reads the world first, so a repeated effect never does harm.
* An event says only what was read back from R2 or from the provider after
  the change. A request that "probably worked" is not evidence.
* A finding (an object is missing, is too large, or has wrong bytes) is
  reported only after a second read gave the same answer as the first. One
  bad read is a retry, not a finding.
* A staging object is hashed first and then copied only while it still has
  the etag that was hashed. So a canonical key can only ever hold bytes that
  matched the manifest.
* The HEAD pointer moves by compare-and-swap on the etag that was read, and
  only after the full state and the result were read back and verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, List, Mapping, Optional, Set, Tuple

from . import core as C
from .guard import box_name, parse_box_name
from .interfaces import (
    BatchSource, Blob, BoxAgent, BoxError, BoxInfo, Clock, JobSpec, Judge, JudgeInvalid, NotFound,
    ObjectStore, Provider, ProviderError, RentUnavailable, SourceError, StoreError, TooLarge,
    VERDICT_GREEN, VERDICT_RED,
)
from .layout import (
    GENESIS_BATCH, MAX_COMPLETE_BYTES, MAX_HEAD_BYTES, MAX_HEARTBEAT_BYTES, HeadPointer, Layout,
    Package, StateManifest, head_body, is_safe_batch_id, link_body, load_json, outcome_body,
    parse_complete, parse_head, parse_heartbeat, parse_manifest, sha256_hex,
)
from .timers import Timers

APPLY_VERIFY_SHA256 = 'sha256'
APPLY_VERIFY_ETAG = 'etag'

OK, RETRY, INVALID, FAULT = 'ok', 'retry', 'invalid', 'fault'
MISSING, TOO_LARGE = 'missing', 'too_large'


@dataclass(frozen=True)
class EffectConfig:
    run_id: str
    gpu_count: int
    price_ceiling_usd_h: float
    rent_timeout_s: float
    box_ttl_s: float
    job_ttl_s: float
    heartbeat_every_s: float
    apply_verify: str = APPLY_VERIFY_SHA256

    def __post_init__(self) -> None:
        if self.apply_verify not in (APPLY_VERIFY_SHA256, APPLY_VERIFY_ETAG):
            raise ValueError('apply_verify must be sha256 or etag')


class FenceView:
    """What the staging sweep needs to know about fences."""

    def __init__(self, live: FrozenSet[int], dirty: FrozenSet[Tuple[str, int]], next_fence: int):
        self.live = live
        self.dirty = dirty
        self.next_fence = next_fence


class LeaseView:
    """The lease of one fence as the journal has it now."""

    def __init__(self, phase: C.Phase, try_no: int, box_id: Optional[str]):
        self.phase = phase
        self.try_no = try_no
        self.box_id = box_id


class Effects:
    def __init__(
        self,
        cfg: EffectConfig,
        *,
        store: ObjectStore,
        providers: Mapping[str, Provider],
        agent: BoxAgent,
        source: BatchSource,
        judge: Judge,
        clock: Clock,
        timers: Timers,
        layout: Layout,
        fences: Callable[[], FenceView],
        lease: Callable[[int], Optional[LeaseView]],
        done_batch: Callable[[str], Optional[C.Batch]],
        known_refs: Callable[[], Mapping[str, Tuple[str, str]]],
        alert: Callable[[str, str], None],
    ):
        self.cfg = cfg
        self.store = store
        self.providers = providers
        self.agent = agent
        self.source = source
        self.judge = judge
        self.clock = clock
        self.timers = timers
        self.layout = layout
        self._fences = fences
        self._lease = lease
        self._done_batch = done_batch
        self._known_refs = known_refs
        self._alert = alert

    def run(self, op: Any) -> List[Any]:
        return self._DISPATCH[type(op)](self, op)

    # -- small helpers -------------------------------------------------------

    def _digest(self, key: str) -> Any:
        """A ``Digest``, None when the key does not exist, or RETRY."""
        try:
            return self.store.digest(key)
        except NotFound:
            return None
        except StoreError:
            return RETRY

    def _same_again(self, key: str, first: Any) -> bool:
        """True when a second ``_digest`` gives the same answer as the first.

        Only two equal answers make a finding: that the object is missing,
        or that it has other bytes.
        """
        again = self._digest(key)
        return again is not RETRY and again == first

    def _fetch(self, key: str, max_bytes: int) -> Tuple[str, Optional[Blob]]:
        """One read of a small object: (OK, blob), or (MISSING, None),
        (TOO_LARGE, None) or (RETRY, None)."""
        try:
            return OK, self.store.get(key, max_bytes=max_bytes)
        except NotFound:
            return MISSING, None
        except TooLarge:
            return TOO_LARGE, None
        except StoreError:
            return RETRY, None

    def _doc(
        self, key: str, max_bytes: int, parse: Callable[[Blob], Any], *, root: bool = False,
    ) -> Tuple[str, Any]:
        """Read a small object and parse it.

        Returns (OK, value), (RETRY, None) or (INVALID, reason). ``parse``
        raises ``ValueError`` with the reason when the bytes are wrong.
        INVALID needs two reads that gave the same bytes, or the same
        absence.

        ``root``: nothing that the controller knows already (a hash, the
        journal) says what the bytes must be. Then OK needs two equal reads
        too, so that one bad read can never become a durable decision.
        """
        first = self._fetch(key, max_bytes)
        status, blob = first
        if status == RETRY:
            return RETRY, None
        value, reason = None, None
        if status == OK:
            try:
                value = parse(blob)
            except ValueError as exc:
                reason = str(exc)
        else:
            reason = status
        if reason is None and not root:
            return OK, value
        if self._fetch(key, max_bytes) != first:
            return RETRY, None
        return (OK, value) if reason is None else (INVALID, reason)

    def _still(self, fence: int, phase: C.Phase, *, try_no: Optional[int] = None,
               box_id: Optional[str] = None) -> bool:
        """True while the journal still wants this piece of work.

        An effect can start late, after the lease moved on. The fence makes a
        late effect harmless; this check only saves the wasted work.
        """
        lease = self._lease(fence)
        if lease is None or lease.phase is not phase:
            return False
        if try_no is not None and lease.try_no != try_no:
            return False
        return box_id is None or lease.box_id == box_id

    def _store_once(self, key: str, data: bytes) -> str:
        """Write a document under a key that is derived from its content.

        Returns OK only after the bytes were read back. FAULT means the key
        holds other bytes: someone else wrote into the prefix.
        """
        try:
            self.store.put(key, data, if_none_match=True)
        except StoreError:
            # Refused because the key exists, or outcome unknown. Read it.
            pass
        got = self._digest(key)
        if got is RETRY or got is None:
            return RETRY
        if got.size == len(data) and got.sha256 == sha256_hex(data):
            return OK
        return FAULT if self._same_again(key, got) else RETRY

    # -- ingest --------------------------------------------------------------

    def ingest(self) -> List[Any]:
        """Read the public batch listing. Never writes to the source."""
        try:
            refs = list(self.source.list())
        except SourceError:
            return []
        # batch id -> (miner, reference) of the batch that has this id.
        known = dict(self._known_refs())
        events: List[Any] = []
        for ref in refs:
            ok = (
                is_safe_batch_id(ref.batch_id)
                and isinstance(ref.miner, str) and 0 < len(ref.miner) <= 128
                and isinstance(ref.ref, str) and len(ref.ref) <= 1024
            )
            if ok and known.setdefault(ref.batch_id, (ref.miner, ref.ref)) == (ref.miner, ref.ref):
                events.append(C.Arrived(ref.batch_id, ref.miner, ref.ref))
                continue
            # A batch that cannot be admitted is recorded, never dropped. An
            # id names one batch for ever: the same id with another miner or
            # another reference is a second batch, and it is refused.
            reason = 'batch_id_reused' if ok else 'bad_batch_ref'
            shown = repr((ref.batch_id, ref.miner, ref.ref))
            events.append(C.Rejected(sha256_hex(shown.encode()), '%s:%s' % (reason, shown[:120])))
        return events

    # -- genesis -------------------------------------------------------------

    def genesis(self, state_id: str) -> List[Any]:
        """Verify the first HEAD state, which an operator put in R2."""
        status, detail = self._verify_state(
            state_id, batch_id=GENESIS_BATCH, parent=None, evidence=(), full=True,
        )
        if status == OK:
            return [C.GenesisVerified(state_id, detail)]
        if status != RETRY:
            self._alert('genesis_state_not_ready', str(detail))
        return []

    # -- rent ----------------------------------------------------------------

    def _named(self, provider: Provider, name: str) -> List[BoxInfo]:
        return [box for box in provider.list() if box.name == name]

    def _box_problem(self, box: BoxInfo, name: str, rung: C.Rung) -> Optional[str]:
        if box.name != name:
            return 'name'
        if box.gpu != rung.gpu:
            return 'gpu'
        if box.gpu_count != self.cfg.gpu_count:
            return 'gpu_count'
        if box.tier != rung.tier:
            return 'tier'
        if not 0 <= box.price_usd_h <= self.cfg.price_ceiling_usd_h:
            return 'price'
        return None

    def rent(self, op: C.RentOp) -> List[Any]:
        provider = self.providers.get(op.rung.provider)
        if provider is None:
            return [C.RentFailed(op.fence, op.try_no, 'unknown_provider')]
        name = box_name(self.cfg.run_id, op.fence, op.try_no)
        if not self._still(op.fence, C.Phase.RENTING, try_no=op.try_no):
            return []
        try:
            # A box of this exact name exists if an earlier run of this effect
            # rented it and then died. Adopt it; never rent a second one.
            found = self._named(provider, name)
        except ProviderError:
            return []
        if not found:
            if not self._still(op.fence, C.Phase.RENTING, try_no=op.try_no):
                return []
            try:
                found = [provider.rent(
                    name=name, gpu=op.rung.gpu, tier=op.rung.tier, gpu_count=self.cfg.gpu_count,
                    max_price_usd_h=self.cfg.price_ceiling_usd_h, ttl_s=self.cfg.box_ttl_s,
                    timeout_s=self.cfg.rent_timeout_s,
                )]
            except RentUnavailable:
                return [C.RentFailed(op.fence, op.try_no, 'unavailable')]
            except ProviderError:
                # Outcome unknown. Look once. If no box shows up, this try is
                # over; a box that shows up later has the name of a dead try,
                # and the reconcile sweep destroys it.
                try:
                    found = self._named(provider, name)
                except ProviderError:
                    found = []
                if not found:
                    return [C.RentFailed(op.fence, op.try_no, 'provider_error')]
        events: List[Any] = []
        spare: List[BoxInfo] = []
        problem = None
        chosen = None
        for box in sorted(found, key=lambda b: b.box_id):
            why = self._box_problem(box, name, op.rung)
            if chosen is None and why is None:
                chosen = box
            else:
                spare.append(box)
                problem = problem or why
        if chosen is not None:
            events.append(C.RentOk(
                op.fence, op.try_no, chosen.box_id, provider.name, chosen.gpu, chosen.tier,
                chosen.price_usd_h, chosen.name,
            ))
        else:
            events.append(C.RentFailed(op.fence, op.try_no, 'box_rejected:%s' % problem))
        # After the event above the try is closed, so the core dooms these.
        for box in spare:
            events.append(C.OrphanSeen(box.box_id, provider.name, op.fence, op.try_no))
        return events

    # -- launch --------------------------------------------------------------

    def launch(self, op: C.LaunchOp) -> List[Any]:
        if not self._still(op.fence, C.Phase.RUNNING, box_id=op.box_id):
            return []

        def base_manifest(blob: Blob) -> StateManifest:
            if sha256_hex(blob.data) != op.base_state:
                raise ValueError('mismatch')
            try:
                doc = load_json(blob.data)
                return parse_manifest(doc, batch_id=doc.get('batch_id'), parent=doc.get('parent'))
            except (ValueError, AttributeError):
                raise ValueError('invalid') from None

        status, manifest = self._doc(
            self.layout.state_manifest(op.base_state), MAX_COMPLETE_BYTES, base_manifest,
        )
        if status == RETRY:
            return []
        if status != OK:
            # The base was the HEAD (or the pending HEAD) when this box was
            # bound. Its manifest was verified before that, and no state is
            # ever changed or deleted, so this must never happen.
            return [C.Fault('base_manifest_%s' % manifest)]
        spec = JobSpec(
            run_id=self.cfg.run_id,
            batch_id=op.batch_id,
            batch_ref=op.ref,
            fence=op.fence,
            attempt=op.attempt,
            base_state=op.base_state,
            base_seq=op.base_seq,
            base_prefix=self.layout.state_prefix(op.base_state),
            expected_fp32_master_digest=manifest.fp32_master_digest,
            staging_prefix=self.layout.staging_prefix(op.batch_id, op.fence),
            heartbeat_every_s=self.cfg.heartbeat_every_s,
            job_ttl_s=self.cfg.job_ttl_s,
        )
        try:
            self.agent.launch(op.box_id, op.provider, spec)
        except BoxError:
            return []
        return [C.Launched(op.fence)]

    # -- poll ----------------------------------------------------------------

    def _heartbeat_seq(self, op: C.PollOp) -> Optional[int]:
        try:
            blob = self.store.get(
                self.layout.staging_heartbeat(op.batch_id, op.fence), max_bytes=MAX_HEARTBEAT_BYTES,
            )
            return parse_heartbeat(blob.data, fence=op.fence)
        except (StoreError, ValueError):
            return None

    def poll(self, op: C.PollOp) -> List[Any]:
        complete_key = self.layout.staging_complete(op.batch_id, op.fence)
        # The time is read first. "Expired" below then means: the upload was
        # not complete at a time after the deadline.
        now = self.clock.now()
        try:
            if self.store.head(complete_key) is not None:
                return [C.Polled(op.fence, complete=True)]
        except StoreError:
            return []
        events: List[Any] = []
        seq = self._heartbeat_seq(op)
        if seq is not None:
            until = self.timers.heartbeat(op.fence, seq, now)
            if until is not None:
                events.append(C.Renewal(op.fence, seq, until))
        gone = False
        provider = self.providers.get(op.provider)
        if provider is not None:
            try:
                listed = provider.list()
            except ProviderError:
                listed = None
            if listed is not None:
                present = any(box.box_id == op.box_id for box in listed)
                gone = self.timers.presence(('poll', op.fence), present)
        deadline = self.timers.deadline(op.fence)
        expired = deadline is not None and now > deadline
        if gone or expired:
            # Look again: a box can finish its upload and die between the two
            # reads. Work that is in R2 is adopted, not thrown away.
            try:
                if self.store.head(complete_key) is not None:
                    return [C.Polled(op.fence, complete=True)]
            except StoreError:
                return events
            events.append(C.Polled(op.fence, gone=gone, expired=expired))
        return events

    # -- settle --------------------------------------------------------------

    def settle(self, op: C.SettleOp) -> List[Any]:
        lay = self.layout
        if not self._still(op.fence, C.Phase.SETTLING):
            return []

        def package(blob: Blob) -> Package:
            return parse_complete(
                blob.data, batch_id=op.batch_id, fence=op.fence, base_state=op.base_state,
            )

        # ``COMPLETE.json`` names the hashes of everything else in the
        # package, and nothing names the hash of ``COMPLETE.json``.
        status, pkg = self._doc(
            lay.staging_complete(op.batch_id, op.fence), MAX_COMPLETE_BYTES, package, root=True,
        )
        if status == RETRY:
            return []
        if status != OK:
            reason = 'complete_%s' % pkg if pkg in (MISSING, TOO_LARGE) else pkg
            return [C.SettleInvalid(op.fence, reason)]

        def exact(blob: Blob) -> bytes:
            if len(blob.data) != pkg.result_size or sha256_hex(blob.data) != pkg.result_sha:
                raise ValueError('mismatch')
            return blob.data

        status, result = self._doc(lay.staging_result(op.batch_id, op.fence), pkg.result_size, exact)
        if status == RETRY:
            return []
        if status != OK:
            reason = 'result_missing' if result == MISSING else 'result_mismatch'
            return [C.SettleInvalid(op.fence, reason)]

        # The result becomes durable under its own hash before any verdict.
        stored = self._store_once(lay.result(pkg.result_sha), result)
        if stored == RETRY:
            return []
        if stored == FAULT:
            # The key of this result holds other bytes, so the result cannot
            # become durable and there is no verdict. Stop new work, and
            # count the attempt as lost: the batch keeps its place.
            return [
                C.Fault('result_object_conflict'),
                C.SettleInvalid(op.fence, 'result_object_conflict'),
            ]

        try:
            verdict = self.judge.verdict(result, batch_id=op.batch_id, base_state=op.base_state)
        except JudgeInvalid:
            return [C.SettleInvalid(op.fence, 'result_unreadable')]
        if verdict == VERDICT_RED:
            return [C.Settled(op.fence, C.VERDICT_RED, pkg.result_sha)]
        if verdict != VERDICT_GREEN:
            return [C.SettleInvalid(op.fence, 'verdict_unknown')]

        # GREEN. The miner is credited from here on, with or without a state.
        if pkg.manifest is None:
            return [C.Settled(op.fence, C.VERDICT_GREEN, pkg.result_sha, None, (), 'no_state_in_package')]
        status, detail = self._adopt_state(op, pkg.manifest)
        if status == OK:
            return [C.Settled(op.fence, C.VERDICT_GREEN, pkg.result_sha, pkg.manifest.state_id, detail)]
        if status == INVALID:
            return [C.Settled(
                op.fence, C.VERDICT_GREEN, pkg.result_sha, None, (), 'state_invalid:%s' % detail,
            )]
        if status == FAULT:
            # Someone else wrote into the keys of this state. The verdict
            # stands and the miner is paid, but this state is never applied.
            # New work stops until an operator has looked.
            return [
                C.Fault(str(detail)),
                C.Settled(op.fence, C.VERDICT_GREEN, pkg.result_sha, None, (), 'state_invalid:%s' % detail),
            ]
        return []

    def _adopt_state(self, op: C.SettleOp, manifest: StateManifest) -> Tuple[str, Any]:
        """Copy the state of a package to its canonical keys and verify it."""
        lay = self.layout
        state_id = manifest.state_id
        evidence = []
        for ref in manifest.objects:
            src = lay.staging_object(op.batch_id, op.fence, ref.name)
            dst = lay.state_object(state_id, ref.name)
            status, detail = self._adopt_object(src, dst, ref.size, ref.sha256)
            if status in (INVALID, FAULT):
                return status, '%s:%s' % (detail, ref.name)
            if status != OK:
                return status, detail
            evidence.append((ref.name, ref.size, ref.sha256, detail))
        # The manifest is written last: a canonical state with a manifest is
        # complete. Its key is the hash of its bytes.
        stored = self._store_once(lay.state_manifest(state_id), manifest.body())
        if stored == RETRY:
            return RETRY, None
        if stored == FAULT:
            return FAULT, 'state_manifest_conflict'
        return OK, tuple(evidence)

    def _adopt_object(self, src: str, dst: str, size: int, sha: str) -> Tuple[str, Any]:
        want = (size, sha)
        have = self._digest(dst)
        if have is RETRY:
            return RETRY, None
        if have is not None:
            if (have.size, have.sha256) == want:
                return OK, have.etag
            # A canonical key is written once, and only with verified bytes.
            # Other bytes mean that someone else wrote into the prefix. An
            # immutable key is never overwritten: stop and show it.
            if self._same_again(dst, have):
                return FAULT, 'canonical_object_conflict'
            return RETRY, None

        staged = self._digest(src)
        if staged is RETRY:
            return RETRY, None
        if staged is None or (staged.size, staged.sha256) != want:
            if not self._same_again(src, staged):
                return RETRY, None
            return INVALID, 'object_missing' if staged is None else 'object_mismatch'
        try:
            # Copy exactly the bytes that were hashed, or nothing.
            self.store.copy(src, dst, src_etag=staged.etag)
        except StoreError:
            return RETRY, None
        have = self._digest(dst)
        if have is RETRY or have is None:
            return RETRY, None
        if (have.size, have.sha256) != want:
            if self._same_again(dst, have):
                return FAULT, 'canonical_object_conflict'
            return RETRY, None
        return OK, have.etag

    # -- verify a canonical state -------------------------------------------

    def _verify_state(
        self, state_id: str, *, batch_id: str, parent: Optional[str], evidence: Tuple, full: bool,
    ) -> Tuple[str, Any]:
        """Check that R2 holds the full state ``state_id`` right now.

        Returns (OK, evidence), (RETRY, None) or (INVALID, reason).
        """
        lay = self.layout

        def state_manifest(blob: Blob) -> StateManifest:
            if sha256_hex(blob.data) != state_id:
                raise ValueError('mismatch')
            try:
                return parse_manifest(load_json(blob.data), batch_id=batch_id, parent=parent)
            except ValueError as exc:
                raise ValueError('invalid:%s' % exc) from None

        status, manifest = self._doc(lay.state_manifest(state_id), MAX_COMPLETE_BYTES, state_manifest)
        if status == RETRY:
            return RETRY, None
        if status != OK:
            return INVALID, 'manifest_%s' % manifest
        known = {item[0]: item for item in evidence}
        out = []
        for ref in manifest.objects:
            key = lay.state_object(state_id, ref.name)
            seen = known.get(ref.name)
            if not full and seen is not None and (seen[1], seen[2]) == (ref.size, ref.sha256):
                try:
                    info = self.store.head(key)
                except StoreError:
                    return RETRY, None
                if info is not None and info.size == ref.size and info.etag == seen[3]:
                    out.append((ref.name, ref.size, ref.sha256, info.etag))
                    continue
            have = self._digest(key)
            if have is RETRY:
                return RETRY, None
            if have is None or (have.size, have.sha256) != (ref.size, ref.sha256):
                if not self._same_again(key, have):
                    return RETRY, None
                why = 'object_missing' if have is None else 'object_mismatch'
                return INVALID, '%s:%s' % (why, ref.name)
            out.append((ref.name, ref.size, ref.sha256, have.etag))
        return OK, tuple(out)

    # -- apply ---------------------------------------------------------------

    def apply(self, op: C.ApplyOp) -> List[Any]:
        lay = self.layout

        def diverged(why: str) -> List[Any]:
            return [C.ApplyDiverged(op.seq, op.link_id, why)]

        def pointer_of(blob: Blob) -> Tuple[HeadPointer, Blob]:
            try:
                pointer = parse_head(blob.data)
            except ValueError:
                raise ValueError('pointer_unreadable') from None
            if pointer.link == op.link_id:
                if pointer.seq != op.seq or pointer.state_id != op.state_id:
                    raise ValueError('pointer_inconsistent')
            elif op.parent_link is None or pointer.link != op.parent_link or pointer.seq != op.seq - 1:
                raise ValueError('pointer_moved')
            elif pointer.state_id != op.parent_state:
                raise ValueError('pointer_inconsistent')
            return pointer, blob

        # A pointer that does not fit halts the controller. Like every other
        # finding, that answer needs two equal reads (``_doc``).
        status, found = self._doc(lay.head, MAX_HEAD_BYTES, pointer_of)
        if status == RETRY:
            return []
        blob = None
        if status == OK:
            pointer, blob = found
            if pointer.link == op.link_id:
                return [C.ApplyDone(op.seq, op.link_id)]
        elif found != MISSING or op.parent_link is not None:
            return diverged('pointer_%s' % found if found in (MISSING, TOO_LARGE) else found)

        # Verify the new HEAD in R2 right before the swap: the manifest, every
        # state file by size and sha256, and the batch result.
        status, detail = self._verify_state(
            op.state_id,
            batch_id=op.batch_id if op.batch_id is not None else GENESIS_BATCH,
            parent=op.parent_state,
            evidence=op.evidence,
            full=self.cfg.apply_verify == APPLY_VERIFY_SHA256,
        )
        if status == RETRY:
            return []
        if status != OK:
            return diverged('state_unverified:%s' % detail)
        if op.result_sha is not None:
            key = lay.result(op.result_sha)
            have = self._digest(key)
            if have is RETRY:
                return []
            if have is None or have.sha256 != op.result_sha:
                if not self._same_again(key, have):
                    return []
                return diverged('result_unverified')

        body = link_body(
            seq=op.seq, state_id=op.state_id, parent=op.parent_link, batch_id=op.batch_id,
            result_sha=op.result_sha, fence=op.fence,
        )
        if sha256_hex(body) != op.link_id:
            return diverged('link_id_mismatch')
        stored = self._store_once(lay.link(op.link_id), body)
        if stored == RETRY:
            return []
        if stored == FAULT:
            return diverged('link_conflict')

        new = head_body(seq=op.seq, link=op.link_id, state_id=op.state_id)
        try:
            if blob is None:
                self.store.put(lay.head, new, if_none_match=True)
            else:
                self.store.put(lay.head, new, if_match=blob.etag)
        except StoreError:
            # Refused or unknown. The next run reads the pointer and decides.
            return []
        try:
            after = self.store.get(lay.head, max_bytes=MAX_HEAD_BYTES)
        except StoreError:
            return []
        if after.data == new:
            return [C.ApplyDone(op.seq, op.link_id)]
        return []

    # -- destroy -------------------------------------------------------------

    def destroy(self, op: C.DestroyOp) -> List[Any]:
        provider = self.providers.get(op.provider)
        if provider is None:
            self._alert('destroy_unknown_provider', op.provider)
            return []
        try:
            listed = provider.list()
        except ProviderError:
            return []
        box = next((b for b in listed if b.box_id == op.box_id), None)
        if box is None:
            return [C.BoxGone(op.box_id, 'absent')]
        if parse_box_name(self.cfg.run_id, box.name) is None:
            # The id now names a box that this run did not make. Hands off.
            self._alert('destroy_refused_not_ours', op.box_id)
            return [C.BoxGone(op.box_id, 'not_ours')]
        try:
            provider.destroy(op.box_id)
        except ProviderError:
            pass
        # The next run confirms through the listing that the box is gone.
        return []

    # -- staging clean-up and outcomes --------------------------------------

    def gc(self, op: C.GcOp) -> List[Any]:
        prefix = self.layout.staging_prefix(op.batch_id, op.fence)
        try:
            for info in self.store.list(prefix):
                self.store.delete(info.key)
            if self.store.list(prefix):
                return []
        except StoreError:
            return []
        return [C.StagingCleared(op.batch_id, op.fence)]

    def publish(self, op: C.PublishOp) -> List[Any]:
        batch = self._done_batch(op.batch_id)
        if batch is None:
            return []
        stored = self._store_once(
            self.layout.outcome(op.batch_id), outcome_body(C.outcome_fields(batch)),
        )
        if stored == OK:
            return [C.Published(op.batch_id)]
        if stored == FAULT:
            return [C.Fault('outcome_conflict')]
        return []

    # -- reconcile sweep -----------------------------------------------------

    def sweep(self) -> List[Any]:
        """Find boxes and staging uploads that no lease owns."""
        events: List[Any] = []
        for name in sorted(self.providers):
            try:
                listed = self.providers[name].list()
            except ProviderError:
                continue
            for box in listed:
                parsed = parse_box_name(self.cfg.run_id, box.name)
                if parsed is not None:
                    events.append(C.OrphanSeen(box.box_id, name, parsed[0], parsed[1]))
        try:
            found = self.store.list(self.layout.staging_root)
        except StoreError:
            return events
        # The fences are read after the listing. A fence that is not live in
        # that view and is below ``next_fence`` ended, and a fence never comes
        # back to life. So every key under it is garbage.
        view = self._fences()
        strange: Set[str] = set()
        for info in found:
            parsed = self.layout.parse_staging_key(info.key)
            if parsed is None or parsed[1] >= view.next_fence:
                strange.add(info.key)
                continue
            if parsed[1] in view.live or parsed in view.dirty:
                continue
            try:
                self.store.delete(info.key)
            except StoreError:
                continue
        if strange:
            self._alert('staging_key_without_fence', sorted(strange)[0])
        return events

    _DISPATCH = {
        C.RentOp: rent,
        C.LaunchOp: launch,
        C.PollOp: poll,
        C.SettleOp: settle,
        C.ApplyOp: apply,
        C.DestroyOp: destroy,
        C.GcOp: gc,
        C.PublishOp: publish,
    }
