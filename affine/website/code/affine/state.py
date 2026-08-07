"""Validator state: king lineage, submission queue, verdict history.

Persistence model: local JSON/JSONL under `state/` on the root machine (the
durable home of the validator), with the presentation subset mirrored to
Hippius by dashboard.py. On startup the king chain is reconciled from
history.jsonl so a dethrone lost to a crash between the `crowned` history
append and the state.json flush is recovered (idempotent).

History invariant: one row per duel. A winning verdict crowns inline and the
single `crowned` row carries the full verdict payload — never a `verdict` row
plus a bare `crowned` row for the same challenge.

Policy invariants enforced here:
  * 1-hotkey-1-eval: the hotkey slot is burned at ENQUEUE, not at verdict.
    A crash between enqueue and verdict costs the miner their slot; they
    must re-register. This makes grinding expensive by construction.
  * A content revision that was ever submitted can never be resubmitted,
    by anyone.
  * Transient infra failures requeue at the front with a bounded retry
    count and do NOT burn a failure record.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("affine.state")

# Recent reveal→intake decisions for the dashboard (commit ≠ queue row).
INTAKE_MAX = 50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class QueueEntry:
    challenge_id: str
    hotkey: str
    repo: str
    revision: str
    block: int
    queued_at: str
    retry_count: int = 0
    # Infra faults never burn the miner's retry budget, but a *persistent* one
    # (dead king, a repo too large for the pod's disk) must not pin the queue
    # head forever: past a bound the entry is rotated to the tail so other
    # submissions still get evaluated.
    infra_retry_count: int = 0


@dataclass
class King:
    hotkey: str
    repo: str
    revision: str
    block: int
    challenge_id: str
    reign_number: int
    crowned_at: str
    # Absolute S* of this model at coronation (challenger.S from the winning
    # duel). Used by the reign-evolution chart; None for genesis / unknown.
    score: float | None = None
    # Prior kings newest-first. Each entry: hotkey, repo, revision, score,
    # and when known reign_number / crowned_at / block. Deduped by hotkey when
    # building the rolling payout chain (Albedo / Teutonic last-N design).
    previous: list[dict] = field(default_factory=list)


class State:
    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.king: King | None = None
        self.queue: list[QueueEntry] = []
        self.seen_hotkeys: set[str] = set()
        self.completed_revisions: set[str] = set()
        self.bench_jobs: list[dict] = []
        self.stats = {"queued": 0, "accepted": 0, "rejected": 0, "failed": 0}
        self._id_counter = 0
        self.eval_machine: dict = {}   # duel provisioner {provider, id, host, url, ...}
        self.bench_machine: dict = {}  # dedicated SWE bench provisioner record
        # Newest-last ring of reveal intake decisions (dashboard "intake" panel).
        self.intake: list[dict] = []
        # Permanent dedupe of (hotkey:block) so trimmed ring rows are not
        # re-emitted every scan tick.
        self.intake_decided: set[str] = set()
        # The popped-but-unresolved queue entry, persisted so a HARD exit
        # (watchdog os._exit, SIGKILL) can never vanish a submission: normal
        # shutdowns requeue via the BaseException handler, but nothing that
        # bypasses `finally` would — load() reconciles this back into the
        # queue instead (observed limbo: chal-00075/chal-00080, 2026-08-04).
        self.in_flight: QueueEntry | None = None
        # When this validator last successfully set weights on-chain (ISO).
        # Persisted so the dashboard can show ground truth instead of a
        # third-party indexer's view of it (TMC's last_commit_at was observed
        # frozen for 16h while the chain kept accepting our set_weights).
        self.last_weights_at: str = ""
        # Payout-window members whose repo@revision is provably gone/gated on
        # HF right now (validator accessibility sweep). Not persisted: it is
        # re-derived before every weight set, and a stale carry-over across a
        # restart could wrongly withhold a restored model's share.
        self.inaccessible_hotkeys: set[str] = set()
        self.current_eval: dict | None = None
        self.phase: dict = {"name": "boot", "since": now_iso()}
        self._last_flush = 0.0
        # State is touched from the main loop AND the provisioner thread
        # (flush / eval_machine): every structural mutation and the flush
        # snapshot hold this lock. Reentrant because set_king flushes inline.
        self._lock = threading.RLock()

    # -- files ---------------------------------------------------------------
    @property
    def _state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def history_path(self) -> Path:
        return self.dir / "history.jsonl"

    @property
    def bench_history_path(self) -> Path:
        return self.dir / "bench_history.jsonl"

    @property
    def evals_dir(self) -> Path:
        return self.dir / "evals"

    @property
    def evals_index_path(self) -> Path:
        return self.evals_dir / "index.jsonl"

    def save_eval_artifact(self, challenge_id: str, gz_bytes: bytes,
                           meta: dict) -> None:
        """Persist a full duel record locally (durable home before Hippius
        upload) and append its manifest line."""
        with self._lock:
            self.evals_dir.mkdir(parents=True, exist_ok=True)
            name = f"{challenge_id}.json.gz"
            (self.evals_dir / name).write_bytes(gz_bytes)
            line = {"key": f"evals/{name}", "bytes": len(gz_bytes),
                    "at": now_iso(), **meta}
            with open(self.evals_index_path, "a") as f:
                f.write(json.dumps(line, default=str) + "\n")

    # -- load / save ----------------------------------------------------------
    def load(self) -> None:
        if self._state_path.exists():
            d = json.loads(self._state_path.read_text())
            self.king = King(**d["king"]) if d.get("king") else None
            self.queue = [QueueEntry(**e) for e in d.get("queue", [])]
            self.seen_hotkeys = set(d.get("seen_hotkeys", []))
            self.completed_revisions = set(d.get("completed_revisions", []))
            self.last_weights_at = d.get("last_weights_at", "")
            self.bench_jobs = d.get("bench_jobs", [])
            self.stats = d.get("stats", self.stats)
            self._id_counter = int(d.get("id_counter", 0))
            self.eval_machine = d.get("eval_machine", {})
            self.bench_machine = d.get("bench_machine", {})
            self.intake = list(d.get("intake") or [])[-INTAKE_MAX:]
            decided = d.get("intake_decided")
            if isinstance(decided, list) and decided:
                self.intake_decided = set(str(x) for x in decided)
            else:
                self.intake_decided = {
                    f"{e.get('hotkey')}:{e.get('block')}"
                    for e in self.intake
                    if e.get("hotkey") is not None and e.get("block") is not None
                }
            # Validator restart loses in-memory bench dispatch; requeue.
            for j in self.bench_jobs:
                if j.get("state") == "RUNNING":
                    j["state"] = "QUEUED"
                    j.pop("started_at", None)
            # A persisted in-flight entry means the previous process died a
            # hard death mid-duel (os._exit / SIGKILL): put it back at the
            # front, uncounted — infra deaths never burn the miner.
            inflight = d.get("in_flight")
            if inflight:
                entry = QueueEntry(**inflight)
                if not any(e.challenge_id == entry.challenge_id
                           for e in self.queue):
                    log.warning("recovered in-flight %s (%s) from a hard "
                                "shutdown; requeued at front",
                                entry.challenge_id, entry.repo)
                    self.queue.insert(0, entry)
        self._reconcile_from_history()

    def drop_unknown_bench_suites(self, suites: list[str]) -> int:
        """Remove queued jobs whose suite is no longer in the contract."""
        allow = set(suites)
        with self._lock:
            before = len(self.bench_jobs)
            self.bench_jobs = [j for j in self.bench_jobs
                               if j.get("suite") in allow]
            return before - len(self.bench_jobs)

    def flush(self) -> None:
        with self._lock:
            d = {
                "king": asdict(self.king) if self.king else None,
                "queue": [asdict(e) for e in self.queue],
                "in_flight": asdict(self.in_flight) if self.in_flight else None,
                "seen_hotkeys": sorted(self.seen_hotkeys),
                "completed_revisions": sorted(self.completed_revisions),
                "last_weights_at": self.last_weights_at,
                "bench_jobs": self.bench_jobs,
                "stats": self.stats,
                "intake": self.intake,
                "intake_decided": sorted(self.intake_decided),
                "id_counter": self._id_counter,
                "eval_machine": self.eval_machine,
                "bench_machine": self.bench_machine,
                "flushed_at": now_iso(),
            }
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(d, indent=1))
            tmp.replace(self._state_path)
            self._last_flush = time.monotonic()

    def _append_history(self, record: dict) -> None:
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _reconcile_from_history(self) -> None:
        """Rebuild the king from history if a crash lost the set_king write.
        The last `crowned` event wins; idempotent when state is current."""
        if not self.history_path.exists():
            return
        last_crown: dict | None = None
        with open(self.history_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "crowned":
                    last_crown = r
        if last_crown is None:
            return
        if self.king and self.king.challenge_id == last_crown.get("challenge_id"):
            return
        if self.king and self.king.reign_number >= int(last_crown.get("reign_number", -1)):
            return
        log.warning("reconciling king from history: reign #%s (%s) was lost to a crash",
                    last_crown.get("reign_number"), last_crown.get("repo"))
        prev = ([self._king_lineage_entry(self.king)] + self.king.previous) if self.king else []
        score = last_crown.get("score")
        self.king = King(
            hotkey=last_crown["hotkey"], repo=last_crown["repo"],
            revision=last_crown["revision"], block=int(last_crown.get("block", 0)),
            challenge_id=last_crown.get("challenge_id", "recovered"),
            reign_number=int(last_crown.get("reign_number", 0)),
            crowned_at=last_crown.get("at", now_iso()),
            score=float(score) if score is not None else None,
            previous=prev)

    # -- intake / queue ------------------------------------------------------
    def next_id(self) -> str:
        self._id_counter += 1
        return f"chal-{self._id_counter:05d}"

    def record_intake(self, *, hotkey: str, block: int, decision: str,
                      repo: str = "", revision: str = "", detail: str = "",
                      challenge_id: str | None = None) -> None:
        """Record a one-shot reveal intake decision for the dashboard.

        Deduped by (hotkey, block) so a reveal is not re-logged every tick.
        """
        key = f"{hotkey}:{block}"
        with self._lock:
            if key in self.intake_decided:
                return
            event = {
                "at": now_iso(),
                "hotkey": hotkey,
                "repo": repo,
                "revision": revision,
                "block": int(block),
                "decision": decision,
                "detail": (detail or "")[:500],
                "challenge_id": challenge_id,
            }
            self.intake.append(event)
            self.intake_decided.add(key)
            while len(self.intake) > INTAKE_MAX:
                self.intake.pop(0)

    def enqueue(self, hotkey: str, repo: str, revision: str, block: int,
                min_submission_block: int) -> QueueEntry | None:
        """Try to place a reveal on the duel queue. Records an intake decision.

        Returns the new QueueEntry on success, else None. Slot is burned at
        successful enqueue (and on revision_already_submitted).
        """
        with self._lock:
            # Already decided this reveal (prior tick) — do not re-run gates.
            if f"{hotkey}:{block}" in self.intake_decided:
                return None
            if block <= min_submission_block:
                self.record_intake(
                    hotkey=hotkey, block=block, repo=repo, revision=revision,
                    decision="skipped_min_block",
                    detail=(f"reveal block {block} ≤ min_submission_block "
                            f"{min_submission_block}"))
                return None
            if self.king and hotkey == self.king.hotkey:
                log.info("skip enqueue: %s is the current king", hotkey[:16])
                self.record_intake(
                    hotkey=hotkey, block=block, repo=repo, revision=revision,
                    decision="skipped_king",
                    detail="hotkey is the reigning king")
                return None
            if hotkey in self.seen_hotkeys:
                log.info("skip enqueue: hotkey %s already used its eval slot",
                         hotkey[:16])
                self.record_intake(
                    hotkey=hotkey, block=block, repo=repo, revision=revision,
                    decision="skipped_slot_burned",
                    detail="hotkey already used its one eval slot")
                return None
            if revision in self.completed_revisions:
                # Burn the slot anyway: resubmitting known content is not free.
                self.seen_hotkeys.add(hotkey)
                detail = (f"revision {revision[:12]} was already submitted "
                          f"before")
                self.record_failure_raw(
                    hotkey, repo, revision, "revision_already_submitted",
                    detail)
                self.record_intake(
                    hotkey=hotkey, block=block, repo=repo, revision=revision,
                    decision="rejected_revision_already_submitted",
                    detail=detail)
                return None
            if any(e.repo == repo for e in self.queue):
                log.info("skip enqueue: repo %s already queued", repo)
                self.record_intake(
                    hotkey=hotkey, block=block, repo=repo, revision=revision,
                    decision="skipped_repo_queued",
                    detail=f"repo {repo} is already on the duel queue")
                return None
            entry = QueueEntry(challenge_id=self.next_id(), hotkey=hotkey,
                               repo=repo, revision=revision, block=block,
                               queued_at=now_iso())
            self.queue.append(entry)
            # Slot burned at enqueue: one reveal = one shot (policy, module doc).
            self.seen_hotkeys.add(hotkey)
            self.completed_revisions.add(revision)
            self.stats["queued"] += 1
            self.record_intake(
                hotkey=hotkey, block=block, repo=repo, revision=revision,
                decision="enqueued", challenge_id=entry.challenge_id,
                detail=f"duel queue ← {entry.challenge_id}")
            return entry

    def pop_next(self) -> QueueEntry | None:
        with self._lock:
            entry = self.queue.pop(0) if self.queue else None
            if entry is not None:
                # Durable BEFORE dispatch: from here until the entry lands in
                # history or back in the queue, state.json carries it as
                # in_flight so no exit path can lose it.
                self.in_flight = entry
                self.flush()
            return entry

    def _clear_in_flight(self, entry: QueueEntry) -> None:
        """The entry landed (history or queue); stop tracking it as popped."""
        with self._lock:
            if (self.in_flight is not None
                    and self.in_flight.challenge_id == entry.challenge_id):
                self.in_flight = None

    def record_weights_set(self) -> None:
        """Stamp a successful on-chain set_weights (flushed immediately so
        the dashboard sees it without waiting for the next periodic flush)."""
        with self._lock:
            self.last_weights_at = now_iso()
            self.flush()

    def peek_next(self) -> QueueEntry | None:
        """Head of the queue without popping (prefetch hinting)."""
        with self._lock:
            return self.queue[0] if self.queue else None

    def push_front(self, entry: QueueEntry) -> None:
        """Put an entry back untouched (e.g. metagraph not ready yet)."""
        with self._lock:
            self.queue.insert(0, entry)
            self._clear_in_flight(entry)

    def requeue_front(self, entry: QueueEntry, reason: str,
                      count_retry: bool = True) -> int:
        """Put a challenge back at the head of the queue after a transient
        failure. `count_retry=False` when the failure was our infrastructure
        (eval machine down): the miner's bounded retry budget only pays for
        failures that happened while the machine was healthy."""
        with self._lock:
            if count_retry:
                entry.retry_count += 1
            entry.queued_at = now_iso()
            self.queue = [entry] + [e for e in self.queue if e.repo != entry.repo]
            self.current_eval = None
            self._clear_in_flight(entry)
        log.warning("requeued %s at front (retry %d, counted=%s) due to %s",
                    entry.challenge_id, entry.retry_count, count_retry, reason)
        return entry.retry_count

    def requeue_back(self, entry: QueueEntry, reason: str) -> None:
        """Rotate an entry to the TAIL of the queue. Used when a persistent
        infra fault (e.g. the pod cannot fit this repo) would otherwise wedge
        the head forever: deferring lets every other submission progress while
        the head entry keeps its slot for a later retry."""
        with self._lock:
            entry.queued_at = now_iso()
            self.queue = [e for e in self.queue if e.repo != entry.repo] + [entry]
            self.current_eval = None
            self._clear_in_flight(entry)
        log.warning("deferred %s to queue tail (infra retry %d) due to %s",
                    entry.challenge_id, entry.infra_retry_count, reason)

    # -- verdicts / king lifecycle --------------------------------------------
    def record_failure(self, entry: QueueEntry, code: str, detail: str = "",
                       *, uid: int | None = None,
                       duration_s: float | None = None) -> None:
        log.info("failed %s: %s — %s@%s: %s", entry.challenge_id, code,
                 entry.repo, entry.revision[:12], detail[:200])
        self.stats["failed"] += 1
        row = {
            "event": "failed", "at": now_iso(), "challenge_id": entry.challenge_id,
            "hotkey": entry.hotkey, "repo": entry.repo, "revision": entry.revision,
            "error_code": code, "error_detail": detail[:2000],
        }
        if uid is not None:
            row["uid"] = int(uid)
        if duration_s is not None:
            row["duration_s"] = round(float(duration_s), 1)
        self._append_history(row)
        self._clear_in_flight(entry)

    def record_failure_raw(self, hotkey: str, repo: str, revision: str,
                           code: str, detail: str, *,
                           uid: int | None = None) -> None:
        log.info("failed (raw): %s — %s@%s: %s", code, repo, revision[:12],
                 detail[:200])
        self.stats["failed"] += 1
        row = {
            "event": "failed", "at": now_iso(), "challenge_id": self.next_id(),
            "hotkey": hotkey, "repo": repo, "revision": revision,
            "error_code": code, "error_detail": detail[:2000],
        }
        if uid is not None:
            row["uid"] = int(uid)
        self._append_history(row)

    def record_verdict(self, entry: QueueEntry, verdict: dict, *,
                       uid: int | None = None,
                       duration_s: float | None = None) -> "King | None":
        """Terminal duel outcome — exactly ONE history row per duel.

        A winning verdict crowns inline: the single `crowned` row carries the
        full verdict payload (z, margin, gates, per-side scores). The old
        shape wrote a `verdict` row then a bare `crowned` row, which doubled
        the duel on every chart and left the crown row without values.
        Returns the new King when the challenger won, else None.
        """
        accepted = bool(verdict.get("challenger_wins"))
        self.stats["accepted" if accepted else "rejected"] += 1
        extra: dict = {"accepted": accepted, "verdict": verdict}
        if uid is not None:
            extra["uid"] = int(uid)
        if duration_s is not None:
            extra["duration_s"] = round(float(duration_s), 1)
        if accepted:
            score = (verdict.get("challenger") or {}).get("S")
            king = self.set_king(
                entry.hotkey, entry.repo, entry.revision, entry.block,
                entry.challenge_id,
                score=float(score) if score is not None else None,
                history_extra=extra)
            self._clear_in_flight(entry)
            return king
        row = {
            "event": "verdict", "at": now_iso(), "challenge_id": entry.challenge_id,
            "hotkey": entry.hotkey, "repo": entry.repo, "revision": entry.revision,
            **extra,
        }
        self._append_history(row)
        self._clear_in_flight(entry)
        return None

    @staticmethod
    def _king_lineage_entry(king: King) -> dict:
        return {
            "hotkey": king.hotkey, "repo": king.repo, "revision": king.revision,
            "reign_number": king.reign_number, "crowned_at": king.crowned_at,
            "block": king.block, "score": king.score,
        }

    def set_king(self, hotkey: str, repo: str, revision: str, block: int,
                 challenge_id: str, score: float | None = None,
                 history_extra: dict | None = None) -> King:
        """Crown a king. `history_extra` merges duel context (verdict payload,
        uid, duration) into the single `crowned` history row — duels must not
        write a second row for the same challenge."""
        with self._lock:
            prev = []
            reign = 0
            if self.king:
                prev = ([self._king_lineage_entry(self.king)]
                        + self.king.previous)[:64]
                reign = self.king.reign_number + 1
            self.king = King(hotkey=hotkey, repo=repo, revision=revision,
                             block=block, challenge_id=challenge_id,
                             reign_number=reign, crowned_at=now_iso(),
                             score=score, previous=prev)
            log.info("CROWNED reign #%d: %s@%s (challenge %s, score=%s)",
                     reign, repo, revision[:12], challenge_id, score)
            row = {
                "event": "crowned", "at": now_iso(), "challenge_id": challenge_id,
                "hotkey": hotkey, "repo": repo, "revision": revision,
                "block": block, "reign_number": reign, "score": score,
            }
            if history_extra:
                row.update(history_extra)
            self._append_history(row)
            self.flush()
            return self.king

    def revert_king(self, reason: str) -> King | None:
        """Drop the reigning king and promote the most recent prior king.

        Used when the sitting king can no longer be served (repo deleted or
        gated): a dead king that stays crowned wedges every duel and keeps
        draining emissions to a model that no longer exists. Returns the
        restored king, or None when there is no prior king to fall back to
        (genesis/seed unservable — an operator-level outage).

        The dead king is dropped entirely (not kept in the payout chain). The
        transition is recorded as a `crowned` event with a fresh, strictly
        higher reign number (carrying `reverted_from_*` for forensics) so that
        `_reconcile_from_history` — which replays the latest crown — restores
        the SAME king after a crash restart instead of resurrecting the dead
        one. Reign numbers stay monotonic across crowns and reverts alike.
        """
        with self._lock:
            if not self.king or not self.king.previous:
                return None
            dead = self.king
            entry = self.king.previous[0]
            rest = self.king.previous[1:]
            reign = dead.reign_number + 1
            challenge_id = f"revert-{reign}"
            self.king = King(
                hotkey=entry.get("hotkey", ""),
                repo=entry["repo"], revision=entry["revision"],
                block=int(entry.get("block", 0)),
                challenge_id=challenge_id,
                reign_number=reign,
                crowned_at=now_iso(),
                score=entry.get("score"),
                previous=rest)
            self._append_history({
                "event": "crowned", "at": now_iso(), "challenge_id": challenge_id,
                "hotkey": self.king.hotkey, "repo": self.king.repo,
                "revision": self.king.revision, "block": self.king.block,
                "reign_number": reign, "score": self.king.score,
                "via": "revert", "reason": reason[:2000],
                "reverted_from_repo": dead.repo,
                "reverted_from_revision": dead.revision,
            })
            self.flush()
            return self.king

    def king_lineage_members(self, payout_depth: int) -> list[dict]:
        """Full stored king lineage (current first) for the dashboard.

        The rolling payout window is the first `payout_depth` distinct hotkeys
        whose model is still accessible on HF (`earning=True`, equal-share
        `weight_bps`). Members in `inaccessible_hotkeys` (repo@revision gone or
        gated — validator sweep) forfeit their slot for as long as the repo
        stays dark; deeper accessible kings backfill the window. Older kings
        stay listed with `earning=False` / zero weight so miners can see the
        full reign history, not only the last-N earners.
        """
        if not self.king:
            return []
        members = [{
            "reign_number": self.king.reign_number,
            "repo": self.king.repo,
            "revision": self.king.revision,
            "hotkey": self.king.hotkey,
            "crowned_at": self.king.crowned_at,
            "block": self.king.block,
            "score": self.king.score,
            "current": True,
        }]
        seen = {self.king.hotkey}
        for p in self.king.previous:
            hk = p.get("hotkey", "")
            if not hk or hk in seen:
                continue
            seen.add(hk)
            row = {
                "reign_number": p.get("reign_number"),
                "repo": p.get("repo", ""),
                "revision": p.get("revision", ""),
                "hotkey": hk,
                "crowned_at": p.get("crowned_at"),
                "block": p.get("block"),
                "score": p.get("score"),
                "current": False,
            }
            if p.get("uid") is not None:
                row["uid"] = int(p["uid"])
            members.append(row)
        depth = max(int(payout_depth), 0)
        n_earners = 0
        for m in members:
            m["inaccessible"] = m["hotkey"] in self.inaccessible_hotkeys
            m["earning"] = not m["inaccessible"] and n_earners < depth
            n_earners += m["earning"]
        weight_bps = (10000 // n_earners) if n_earners else 0
        for m in members:
            m["weight_bps"] = weight_bps if m["earning"] else 0
        return members

    def king_chain_members(self, depth: int) -> list[dict]:
        """Rolling last-N king chain (current first), equal-share weights.

        Same payout shape as Albedo / Teutonic: up to `depth` distinct
        hotkeys, each receiving 1/N of emissions. Used for weight-setting.
        """
        return [m for m in self.king_lineage_members(depth) if m.get("earning")]

    def king_chain_hotkeys(self, depth: int) -> list[str]:
        """Current king first, then prior distinct kings, deduped by hotkey."""
        return [m["hotkey"] for m in self.king_chain_members(depth)]

    # -- bench jobs ------------------------------------------------------------
    def enqueue_bench(self, repo: str, revision: str, hotkey: str,
                      suites: list[str], label: str) -> None:
        with self._lock:
            for suite in suites:
                self._id_counter += 1
                self.bench_jobs.append({
                    "job_id": f"bench-{self._id_counter:05d}-{suite}",
                    "repo": repo, "revision": revision, "hotkey": hotkey,
                    "suite": suite, "label": label,
                    "state": "QUEUED", "queued_at": now_iso(),
                })

    def record_bench_result(self, job: dict, result: dict) -> None:
        log.info("bench %s %s: %s", job["job_id"],
                 "done" if result.get("ok") else "FAILED",
                 json.dumps(result, default=str)[:300])
        with self._lock:
            job["state"] = "DONE" if result.get("ok") else "FAILED"
            job["finished_at"] = now_iso()
            job["result"] = result
            with open(self.bench_history_path, "a") as f:
                f.write(json.dumps({**job}, default=str) + "\n")
            self.bench_jobs = [j for j in self.bench_jobs
                               if j["job_id"] != job["job_id"]]

    # -- phase / progress (dashboard + watchdog) --------------------------------
    def set_phase(self, name: str, **extra) -> None:
        self.phase = {"name": name, "since": now_iso(), **extra}
