"""Affine validator — root-machine main loop.

Single async process, one outstanding duel at a time. Each tick:

  1. watchdog heartbeat (a stuck tick self-exits; pm2 restarts us)
  2. ensure the eval machine is alive (provision / reprovision via Lium→Targon)
  3. refresh the metagraph, seed the genesis king when state is empty
  4. scan on-chain commit-reveals → intake-filter → enqueue
  5. start the queue head as a BACKGROUND task: validate repo, copy-check,
     dispatch the duel, stream to verdict, crown on win. Ticks keep running
     while a duel is in flight, so weight-setting, dashboard flushes, and the
     provisioner are never starved by a multi-hour eval.
  6. pump the benchmark queue on the dedicated bench machine
  7. reassert rolling-king weights on the weight interval
  8. flush state + dashboard

Failure taxonomy: miner-caused failures burn the submission (recorded in
history with an error code); infra-caused failures (pod died, stream idle,
server busy, 5xx, chain hiccups) requeue at the front with a bounded retry
count and never penalize the miner. A popped challenge is NEVER silently
dropped: any unexpected exception requeues it (bounded) or records an
`internal_error` failure — the slot was burned at enqueue, so losing the
entry would erase the submission without a trace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

import bittensor as bt

from . import chain, model_store
from .bench import BenchOrchestrator
from .chain import BlockHashUnavailable
from .config import Config, load_config
from .dashboard import Dashboard
from .eval_client import (EvalBusyError, EvalClient, InfraFaultError,
                          TransientEvalError)
from .hippius import Hippius
from .provisioner import BenchMachineManager, EvalMachineManager
from .state import QueueEntry, State, now_iso

log = logging.getLogger("affine.validator")

# Periodic one-line status summary so anyone tailing the log (or an automated
# monitor) can see the whole system state without assembling it from
# state.json + scattered log lines. Silence is ambiguous; this is not.
HEARTBEAT_INTERVAL_S = 600


class TickWatchdog:
    """Self-exit when the main loop wedges; the supervisor restarts us and
    State.load() + history reconciliation recover cleanly."""

    def __init__(self, warn_after_s: int, restart_after_s: int):
        self.warn_after = warn_after_s
        self.restart_after = restart_after_s
        self._beat = time.monotonic()
        self._warned = False
        threading.Thread(target=self._loop, daemon=True, name="tick-watchdog").start()

    def beat(self) -> None:
        self._beat = time.monotonic()
        self._warned = False

    def _loop(self) -> None:
        while True:
            time.sleep(15)
            age = time.monotonic() - self._beat
            if age > self.restart_after:
                log.critical("tick stuck for %.0fs > %ds; self-exiting for restart",
                             age, self.restart_after)
                os._exit(71)
            if age > self.warn_after and not self._warned:
                log.warning("tick slow: %.0fs since last heartbeat", age)
                self._warned = True


class Validator:
    def __init__(self, cfg: Config):
        if cfg.min_submission_block < 0:
            raise SystemExit(
                "subnet.min_submission_block is unset (-1). Set it to a recent "
                "block in affine.toml before starting, or every pre-launch "
                "test reveal becomes an eligible submission.")
        self.cfg = cfg
        self.state = State(cfg.state_dir)
        self.state.load()
        self.hippius = Hippius(
            cfg.hippius["endpoint"], cfg.hippius["bucket"],
            cfg.secrets.hippius_access_key, cfg.secrets.hippius_secret_key)
        self.dashboard = Dashboard(cfg, self.state, self.hippius)
        repo_root = Path(__file__).resolve().parents[1]
        self.machine = EvalMachineManager(cfg, self.state, repo_root)
        self.bench_machine = BenchMachineManager(cfg, self.state, repo_root)
        self.eval_client = EvalClient(
            self.machine.local_url,
            duel_timeout_s=cfg.duel.timeout_s,
            stream_idle_timeout_s=cfg.validator.stream_idle_timeout_s,
            eval_token=cfg.secrets.eval_token)
        self.bench_client = EvalClient(
            self.bench_machine.local_url,
            duel_timeout_s=cfg.duel.timeout_s,
            stream_idle_timeout_s=cfg.validator.stream_idle_timeout_s,
            eval_token=cfg.secrets.eval_token)
        self.bench = BenchOrchestrator(cfg, self.state, self.bench_client)
        dropped = self.state.drop_unknown_bench_suites(list(cfg.bench.suites))
        if dropped:
            log.info("dropped %d bench jobs not in current suites=%s",
                     dropped, list(cfg.bench.suites))
        self._enqueue_king_benches_if_needed()
        self.metagraph = chain.Metagraph()
        self.watchdog = TickWatchdog(
            cfg.validator.tick_warn_after_s,
            cfg.validator.tick_restart_after_s)
        self.subtensor = bt.subtensor(network=cfg.network)
        self.wallet = bt.Wallet(name=cfg.wallet_name, hotkey=cfg.wallet_hotkey)
        self._last_weights = 0.0
        # (repo, revision) -> (status, monotonic ts) from the payout
        # accessibility sweep, so force-triggered weight sets (crowns/reverts)
        # don't re-probe HF within one weight interval.
        self._access_cache: dict[tuple[str, str], tuple[str, float]] = {}
        self._consecutive_tick_errors = 0
        self._duel_task: asyncio.Task | None = None
        # One slot: at most one prefetch fires per duel. Held so the event
        # loop (which references tasks weakly) can never GC it mid-flight.
        self._prefetch_task: asyncio.Task | None = None
        self._last_heartbeat = 0.0

    # A suite that failed this many recorded runs for the same revision is
    # deterministic, not flaky: stop re-burning a bench pod-day on it at
    # every validator restart.
    _BENCH_BACKFILL_MAX_FAILURES = 3

    def _enqueue_king_benches_if_needed(self) -> None:
        """Backfill advisory suites for the reigning king (e.g. after a suite
        contract change). Skips suites already queued or successfully scored.
        Failed runs are retriable (load/infra flakes) but only up to
        _BENCH_BACKFILL_MAX_FAILURES per revision+suite."""
        king = self.state.king
        if king is None or not self.cfg.bench.enabled:
            return
        pending = {(j["revision"], j["suite"]) for j in self.state.bench_jobs}
        done: set[tuple[str, str]] = set()
        failures: dict[tuple[str, str], int] = {}
        path = self.state.bench_history_path
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("revision") != king.revision:
                        continue
                    key = (king.revision, row.get("suite", ""))
                    result = row.get("result") or {}
                    if result.get("ok"):
                        done.add(key)
                    else:
                        failures[key] = failures.get(key, 0) + 1
        exhausted = {k for k, n in failures.items()
                     if n >= self._BENCH_BACKFILL_MAX_FAILURES}
        if exhausted:
            log.warning("bench backfill: giving up on %s after %d+ failures",
                        sorted(s for _, s in exhausted),
                        self._BENCH_BACKFILL_MAX_FAILURES)
        missing = [s for s in self.cfg.bench.suites
                   if (king.revision, s) not in pending
                   and (king.revision, s) not in done
                   and (king.revision, s) not in exhausted]
        if not missing:
            return
        self.state.enqueue_bench(king.repo, king.revision, king.hotkey,
                                 missing, label=f"reign-{king.reign_number}")
        log.info("enqueued %d missing bench suites for king %s",
                 len(missing), king.repo)

    # -- king lifecycle ---------------------------------------------------------
    def _seed_king_if_needed(self) -> None:
        if self.state.king is not None:
            return
        seed = self.cfg.seed_king
        repo = seed["repo"]
        revision = seed.get("revision", "")
        if not revision:
            revision = model_store.resolve_head_revision(
                repo, self.cfg.secrets.hf_token)
        log.info("seeding genesis king: %s@%s", repo, revision[:12])
        self.state.set_king(hotkey="", repo=repo, revision=revision,
                            block=chain.safe_block(self.subtensor),
                            challenge_id="seed")

    def _sweep_payout_accessibility(self) -> None:
        """Contract: no incentive while your model is not downloadable at the
        exact crowned commit. The pre-dispatch probe only covers the *current*
        king; prior kings in the payout window were never re-checked, so one
        could gate its HF repo after crowning and keep earning (observed:
        reign #1 gated post-crown, 403 on the bench pod, still paid 20%).
        Probe every window candidate here, right before weights are set:
        provably gone/gated ⇒ forfeit the slot (deeper kings backfill); the
        slot returns if the repo comes back. "unknown" probes keep the
        member's previous status — an HF hiccup must never flip payouts."""
        depth = self.cfg.king_chain_size
        # 2x headroom so backfill candidates behind excluded members are
        # probed too; beyond that the lineage tail is dashboard-only.
        candidates = self.state.king_lineage_members(depth)[:depth * 2]
        ttl = self.cfg.validator.weight_interval_s
        gone = set(self.state.inaccessible_hotkeys)
        for m in candidates:
            if not m["hotkey"] or not m["repo"]:
                continue  # genesis seed rows can't earn via the metagraph
            key = (m["repo"], m["revision"])
            status, ts = self._access_cache.get(key, ("", 0.0))
            if not status or time.monotonic() - ts >= ttl:
                ref = model_store.ModelRef(m["repo"], m["revision"])
                status, _ = model_store.fetch_repo_info_or_status(
                    ref, self.cfg.secrets.hf_token)
                self._access_cache[key] = (status, time.monotonic())
            if status == "gone":
                gone.add(m["hotkey"])
            elif status == "ok":
                gone.discard(m["hotkey"])
        if gone != self.state.inaccessible_hotkeys:
            log.warning("payout accessibility changed: inaccessible %s -> %s",
                        sorted(self.state.inaccessible_hotkeys), sorted(gone))
        self.state.inaccessible_hotkeys = gone

    async def _maybe_set_weights(self, force: bool = False) -> None:
        if not force and (time.monotonic() - self._last_weights
                          < self.cfg.validator.weight_interval_s):
            return
        await asyncio.to_thread(self._sweep_payout_accessibility)
        hotkeys = self.state.king_chain_hotkeys(self.cfg.king_chain_size)
        ok = chain.set_rolling_weights(
            self.subtensor, self.wallet, self.cfg.netuid,
            hotkeys, self.metagraph, self.cfg.burn_uid,
            max_metagraph_age_s=self.cfg.validator.metagraph_max_age_s,
            version_key=self.cfg.weight_version_key)
        if ok:
            self._last_weights = time.monotonic()
            self.state.record_weights_set()

    # -- intake -------------------------------------------------------------------
    def _scan_and_enqueue(self) -> None:
        reveals, bad = chain.scan_reveals(
            self.subtensor, self.cfg.netuid,
            self.cfg.submission.reveal_prefix,
            self.cfg.min_submission_block, self.state.seen_hotkeys)
        for b in bad:
            self.state.record_intake(
                hotkey=b["hotkey"], block=b["block"],
                decision="rejected_bad_payload",
                detail=b.get("detail") or "unparseable reveal payload")
        for r in reveals:
            entry = self.state.enqueue(r.hotkey, r.repo, r.revision, r.block,
                                       self.cfg.min_submission_block)
            if entry:
                log.info("enqueued %s: %s@%s from %s",
                         entry.challenge_id, r.repo, r.revision[:12], r.hotkey[:16])

    # -- challenge processing --------------------------------------------------------
    async def _process_challenge_safely(self, entry: QueueEntry) -> None:
        """Catch-all wrapper: whatever happens, the popped entry lands back in
        the queue or in history — never in the void."""
        try:
            await self._process_challenge(entry)
        except (TransientEvalError, BlockHashUnavailable) as e:
            self._requeue_or_exhaust(entry, e)
        except Exception as e:
            log.exception("unexpected error processing %s", entry.challenge_id)
            max_retries = self.cfg.validator.max_transient_eval_retries
            if entry.retry_count < max_retries:
                self.state.requeue_front(entry, f"internal_error: {e}")
            else:
                self.state.record_failure(
                    entry, "internal_error",
                    f"{type(e).__name__}: {e} (after {entry.retry_count} retries)",
                    uid=self.metagraph.uid_of.get(entry.hotkey))
        except BaseException as e:
            # Shutdown path (KeyboardInterrupt / CancelledError): without this
            # the popped entry dies with the task and the submission vanishes
            # (observed limbo: chal-00075, 2026-08-04). Requeue uncounted and
            # flush NOW — the process is going down and the periodic tick
            # flush will not run again. But NOT when the duel already landed:
            # record_verdict / record_failure clear in_flight when they write
            # the terminal history row, and requeuing after that duplicates
            # the duel (observed: chal-00308 re-dueled after a post-verdict
            # SIGINT, 2026-08-07).
            if (self.state.in_flight is not None
                    and self.state.in_flight.challenge_id == entry.challenge_id):
                self.state.requeue_front(
                    entry, f"interrupted: {type(e).__name__}", count_retry=False)
            else:
                log.info("not requeuing %s on %s: verdict already recorded",
                         entry.challenge_id, type(e).__name__)
            self.state.flush()
            raise
        finally:
            self.state.current_eval = None

    async def _revert_dead_king_and_requeue(self, entry: QueueEntry,
                                            reason: str) -> None:
        """The reigning king's repo is provably gone/gated. Revert to the
        previous king, re-assert weights so a dead model stops drawing
        emissions, and requeue the blameless challenge (no burn) to duel the
        restored king next tick. With no prior king to fall back to, nothing
        can be dueled until an operator restores one: defer to the tail (so we
        don't hot-spin the head) and page loudly."""
        reverted = self.state.revert_king(reason)
        if reverted is not None:
            log.error("reverted dead king → %s@%s (reign #%d): %s",
                      reverted.repo, reverted.revision[:12],
                      reverted.reign_number, reason)
            await self._maybe_set_weights(force=True)
            self.state.requeue_front(entry, f"king_reverted: {reason}",
                                     count_retry=False)
        else:
            log.critical("king gone and NO previous king to fall back to "
                         "(genesis/seed unservable?): %s — operator must "
                         "intervene", reason)
            self.state.requeue_back(entry, f"king_unservable_no_fallback: {reason}")

    def _requeue_or_exhaust(self, entry: QueueEntry, e: Exception) -> None:
        # Scheduling contention (bench/duel holding the server), self-diagnosed
        # pod trouble (low disk, dead teacher), and a dead machine are OUR
        # faults: requeue without spending the miner's retry budget. Other
        # transients (including chain hiccups fetching the seed block hash)
        # count against the bounded budget so a permanent failure cannot wedge
        # the queue forever.
        machine_fault = (isinstance(e, (EvalBusyError, InfraFaultError))
                         or not self.machine.is_healthy_now())
        max_retries = self.cfg.validator.max_transient_eval_retries
        if machine_fault:
            # Infra never burns the miner, but a *persistent* infra fault (a
            # repo too large for the pod, a wedged tunnel) must not pin the
            # head forever. Past a bound, rotate to the tail so every other
            # submission still gets its turn while this one keeps its slot.
            entry.infra_retry_count += 1
            if entry.infra_retry_count > self.cfg.validator.max_infra_front_requeues:
                self.state.requeue_back(entry, str(e))
            else:
                self.state.requeue_front(entry, str(e), count_retry=False)
        elif entry.retry_count < max_retries:
            self.state.requeue_front(entry, str(e), count_retry=True)
        else:
            self.state.record_failure(
                entry, "eval_infra_exhausted",
                f"{e} (after {entry.retry_count} transient retries)",
                uid=self.metagraph.uid_of.get(entry.hotkey))

    def _history_meta(self, entry: QueueEntry,
                      t0: float | None = None) -> dict:
        meta: dict = {"uid": self.metagraph.uid_of.get(entry.hotkey)}
        if t0 is not None:
            meta["duration_s"] = max(0.0, time.monotonic() - t0)
        return meta

    def _hygiene_reason(self, info: model_store.RepoInfo) -> str | None:
        """Contract hygiene gates on a repo's metadata; one definition so the
        dispatch gate and the prefetch precheck can never drift."""
        sub = self.cfg.submission
        return model_store.validate_repo_hygiene(
            info, max_size_gb=sub.max_model_size_gb,
            max_total_repo_gb=sub.max_total_repo_gb,
            allow_python_files=sub.allow_python_files,
            allow_auto_map=sub.allow_auto_map,
            max_repo_files=sub.max_repo_files,
            max_config_bytes=sub.max_config_bytes)

    async def _prefetch_next(self, nxt: QueueEntry) -> None:
        """Warm the next queued challenger's weights on the pod while the
        current duel scores. Runs the same cheap gates that entry's own
        dispatch will run (name + metadata hygiene) so bandwidth is never
        spent on a repo that would be rejected anyway, and ships the weight
        size so the engine needs no HF-metadata access of its own.
        Best-effort: every failure is swallowed — a missed prefetch only
        means that duel pays its own download."""
        sub = self.cfg.submission
        try:
            pairs = self.metagraph.identity_token_pairs(
                nxt.hotkey, sub.coldkey_prefix_len, sub.coldkey_suffix_len)
            if model_store.validate_repo_name(nxt.repo, sub.repo_pattern, pairs):
                return
            ref = model_store.ModelRef(nxt.repo, nxt.revision)
            info = await asyncio.to_thread(
                model_store.fetch_repo_info, ref, self.cfg.secrets.hf_token)
            if self._hygiene_reason(info):
                return
            await self.eval_client.prefetch(nxt.repo, nxt.revision,
                                            info.total_safetensors_bytes)
        except Exception as e:
            log.debug("prefetch precheck failed for %s (ignored): %s",
                      nxt.repo, e)

    async def _process_challenge(self, entry: QueueEntry) -> None:
        cid = entry.challenge_id
        sub = self.cfg.submission
        king = self.state.king
        assert king is not None
        t0 = time.monotonic()
        self.state.set_phase("process_challenge", challenge_id=cid, repo=entry.repo)
        log.info("processing %s: %s@%s", cid, entry.repo, entry.revision[:12])

        # Repo-name policy: pattern + anti-impersonation identity binding
        # (coldkey OR hotkey prefix+suffix). The hotkey pair is always
        # checkable, so a deregistered/unknown coldkey no longer wedges the
        # queue head in a retry loop.
        pairs = self.metagraph.identity_token_pairs(
            entry.hotkey, sub.coldkey_prefix_len, sub.coldkey_suffix_len)
        reason = model_store.validate_repo_name(entry.repo, sub.repo_pattern, pairs)
        if reason:
            self.state.record_failure(entry, "repo_name_rejected", reason,
                                      **self._history_meta(entry, t0))
            return

        # Pin + hygiene (metadata only, no weight download).
        ref = model_store.ModelRef(entry.repo, entry.revision)
        try:
            info = model_store.fetch_repo_info(ref, self.cfg.secrets.hf_token)
        except Exception as e:
            self.state.record_failure(entry, "revision_not_found",
                                      f"cannot read {ref.immutable_ref}: {e}",
                                      **self._history_meta(entry, t0))
            return
        reason = self._hygiene_reason(info)
        if reason:
            self.state.record_failure(entry, "repo_hygiene_rejected", reason,
                                      **self._history_meta(entry, t0))
            return

        # Proactive king liveness (the fix for a king that takes its model off
        # HF): probe the king repo before spending a duel on it. A *proven*
        # gone/gated king is reverted here — pre-dispatch, so a launch blip on
        # the pod can never masquerade as a dead king and dethrone a healthy
        # one. An inconclusive probe ("unknown") never dethrones; we proceed
        # and let the copy check fall back to open. This same call supplies the
        # king metadata for copy detection when the repo is live.
        king_ref = model_store.ModelRef(king.repo, king.revision)
        king_status, king_info = model_store.fetch_repo_info_or_status(
            king_ref, self.cfg.secrets.hf_token)
        if king_status == "gone":
            await self._revert_dead_king_and_requeue(
                entry, f"king repo {king_ref.immutable_ref} is gone/gated")
            return
        copy = model_store.check_model_copy(ref, info, king_ref, king_info)
        if copy is not None:
            if copy.action == "reject":
                self.state.record_failure(entry, "model_copy", copy.reason,
                                          **self._history_meta(entry, t0))
                return
            if copy.action == "crown_earlier":
                log.warning("%s: identical weights, earlier commit — crowning original: %s",
                            cid, copy.reason)
                # record_verdict crowns inline (one history row per duel).
                # No duel S* for copy-arbitration crowns.
                self.state.record_verdict(entry, {
                    "challenger_wins": True, "verdict": "crown_earlier",
                    "reason": copy.reason}, **self._history_meta(entry, t0))
                await self._maybe_set_weights(force=True)
                self.bench.enqueue_for(entry.repo, entry.revision, entry.hotkey,
                                       accepted=True, label=f"reign-{self.state.king.reign_number}")
                return
            log.error("unknown copy action %r; failing safe (reject)", copy.action)
            self.state.record_failure(entry, "model_copy", copy.reason,
                                      **self._history_meta(entry, t0))
            return

        # Duel. Seeded by the reveal-block hash (external auditability).
        # BlockHashUnavailable propagates to the safety wrapper: fail CLOSED
        # and requeue rather than duel on a predictable fallback slice.
        block_hash = chain.block_hash_at(self.subtensor, entry.block)
        self.state.current_eval = {
            "challenge_id": cid, "repo": entry.repo, "hotkey": entry.hotkey,
            "stage": "dispatching", "progress": {},
            "started_at": now_iso(),
        }
        self.state.set_phase("duel", challenge_id=cid)
        self.dashboard.flush(force=True)

        prefetch_sent = False

        def on_progress(data: dict) -> None:
            nonlocal prefetch_sent
            self.watchdog.beat()
            if self.state.current_eval is not None:
                self.state.current_eval["stage"] = data.get("phase", "scoring")
                self.state.current_eval["progress"] = data
            self.dashboard.flush()
            # Scoring is GPU-bound and the pod's network is idle: warm the next
            # queued challenger's weights so its duel skips the download. Fired
            # at the load→scoring transition, never earlier, so it cannot steal
            # bandwidth from the current challenger's own download.
            if not prefetch_sent and data.get("phase") == "scoring":
                prefetch_sent = True
                nxt = self.state.peek_next()
                if nxt is not None:
                    self._prefetch_task = asyncio.create_task(
                        self._prefetch_next(nxt),
                        name=f"prefetch-{nxt.challenge_id}")

        verdict = await self.eval_client.run_duel(
            king_repo=king.repo, king_revision=king.revision,
            challenger_repo=entry.repo, challenger_revision=entry.revision,
            challenger_hotkey=entry.hotkey, block_hash=block_hash,
            challenger_weight_bytes=info.total_safetensors_bytes,
            on_progress=on_progress)
        self.state.current_eval = None

        verdict["block_hash"] = block_hash
        # One history row per duel: a winning verdict crowns inside
        # record_verdict, so the crowned row carries the full verdict payload.
        self.state.record_verdict(entry, verdict, **self._history_meta(entry, t0))
        accepted = bool(verdict.get("challenger_wins"))
        log.info("verdict %s: challenger_wins=%s z=%s reason=%s", cid, accepted,
                 verdict.get("z"), verdict.get("rejection_reason"))
        await self._publish_eval_artifact(entry, verdict)

        if accepted:
            await self._maybe_set_weights(force=True)
        self.bench.enqueue_for(
            entry.repo, entry.revision, entry.hotkey, accepted=accepted,
            label=(f"reign-{self.state.king.reign_number}" if accepted else cid))
        self.dashboard.flush(force=True)

    async def _publish_eval_artifact(self, entry: QueueEntry,
                                     verdict: dict) -> None:
        """Fetch the full duel record from the pod and publish it for miners.
        Best-effort: training-data publishing never blocks the verdict path."""
        job_id = verdict.get("job_id")
        if not job_id:
            return
        try:
            gz = await self.eval_client.fetch_artifact(job_id)
            if gz is None:
                log.warning("no eval artifact for %s (job %s)",
                            entry.challenge_id, job_id)
                return
            self.state.save_eval_artifact(entry.challenge_id, gz, {
                "challenge_id": entry.challenge_id,
                "repo": entry.repo, "revision": entry.revision,
                "hotkey": entry.hotkey,
                "challenger_wins": bool(verdict.get("challenger_wins")),
                "z": verdict.get("z"), "margin": verdict.get("margin"),
                "rejection_reason": verdict.get("rejection_reason"),
            })
            self.dashboard.publish_evals()
        except Exception:
            log.warning("eval artifact publish failed for %s",
                        entry.challenge_id, exc_info=True)

    # -- tick -------------------------------------------------------------------------
    def _duel_running(self) -> bool:
        return self._duel_task is not None and not self._duel_task.done()

    async def tick(self) -> None:
        self.watchdog.beat()
        if not self._duel_running():
            self.state.set_phase("tick")
        self.metagraph.refresh(self.subtensor, self.cfg.netuid)
        self.dashboard.uid_of = dict(self.metagraph.uid_of)
        self._seed_king_if_needed()

        machine_healthy = self.machine.ensure()
        bench_healthy = self.bench_machine.ensure()
        self._scan_and_enqueue()

        if self._duel_task is not None and self._duel_task.done():
            # The safety wrapper never raises, but surface a cancelled/broken
            # task loudly rather than dropping it.
            exc = (None if self._duel_task.cancelled()
                   else self._duel_task.exception())
            if exc is not None:
                log.error("duel task died outside the safety wrapper: %s", exc)
            self._duel_task = None

        if machine_healthy and not self._duel_running() and self.state.queue:
            entry = self.state.pop_next()
            if entry is not None:
                self._duel_task = asyncio.create_task(
                    self._process_challenge_safely(entry),
                    name=f"duel-{entry.challenge_id}")

        if self.bench.pump(machine_ready=bench_healthy):
            # A bench result just landed: publish its rollout artifact (and
            # any backlog) without waiting for the next duel verdict.
            try:
                self.dashboard.publish_evals()
            except Exception:
                log.warning("bench artifact publish failed", exc_info=True)
        await self._maybe_set_weights()
        self._maybe_heartbeat()
        self.state.flush()
        self.dashboard.flush()

    def _maybe_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        self._last_heartbeat = now
        king = self.state.king
        duel = (self._duel_task.get_name() if self._duel_running() else "none")
        weights_age = f"{now - self._last_weights:.0f}s" if self._last_weights else "never"
        log.info("heartbeat: eval=%s bench=%s queue=%d duel=%s bench_jobs=%d "
                 "king=%s@%s weights_set=%s ago",
                 self.machine.status_word(), self.bench_machine.status_word(),
                 len(self.state.queue), duel, len(self.state.bench_jobs),
                 king.repo if king else "-",
                 king.revision[:8] if king else "-", weights_age)

    async def run(self) -> None:
        self.dashboard.push_website()
        poll = self.cfg.validator.poll_interval_s
        max_errors = 10
        while True:
            try:
                await self.tick()
                self._consecutive_tick_errors = 0
            except Exception:
                self._consecutive_tick_errors += 1
                log.exception("tick failed (%d consecutive)",
                              self._consecutive_tick_errors)
                if self._consecutive_tick_errors >= max_errors:
                    log.critical("%d consecutive tick errors; exiting for restart",
                                 max_errors)
                    os._exit(72)
            await asyncio.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(description="Affine SN120 validator")
    ap.add_argument("--config", default=None, help="path to affine.toml")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    validator = Validator(cfg)
    asyncio.run(validator.run())


if __name__ == "__main__":
    main()
