"""Benchmark orchestration on the root machine.

Policy (affine.toml [bench]): every valid submission gets the configured
suites enqueued after its duel completes ("all"), or only new kings
("accepted"). Jobs run on the dedicated always-on bench machine
(independent of the duel eval pod). Results land in bench_history.jsonl
and the dashboard.

Benchmarks are ADVISORY — the isomorphism display for miners and operators.
They never feed S*, gates, or the duel rule.
"""

from __future__ import annotations

import logging

from .eval_client import EvalClient
from .state import State, now_iso

log = logging.getLogger("affine.bench")


class BenchOrchestrator:
    def __init__(self, cfg, state: State, client: EvalClient):
        self.cfg = cfg.bench
        self.state = state
        self.client = client
        self._running: dict | None = None  # {job(state entry), remote_id}

    def enqueue_for(self, repo: str, revision: str, hotkey: str,
                    accepted: bool, label: str) -> None:
        if not self.cfg.enabled:
            return
        if self.cfg.policy == "accepted" and not accepted:
            return
        if self.cfg.policy not in ("accepted", "all"):
            log.error("unknown bench policy %r; skipping", self.cfg.policy)
            return
        self.state.enqueue_bench(repo, revision, hotkey,
                                 list(self.cfg.suites), label)
        log.info("bench enqueued for %s (%d suites)", repo, len(self.cfg.suites))

    def pump(self, machine_ready: bool = True) -> None:
        """Called every tick. Polls the running job; dispatches the next one
        when the dedicated bench machine is healthy."""
        if self._running:
            self._poll_running()
            return
        if not machine_ready or not self.state.bench_jobs:
            return
        job = next((j for j in self.state.bench_jobs if j["state"] == "QUEUED"), None)
        if job is None:
            return
        remote_id = self.client.try_start_bench(
            repo=job["repo"], revision=job["revision"], suite=job["suite"],
            num_trials=self.cfg.num_trials,
            max_concurrency=self.cfg.max_concurrency,
            user_llm=self.cfg.user_llm)
        if remote_id is None:
            return
        job["state"] = "RUNNING"
        job["started_at"] = now_iso()
        self._running = {"job": job, "remote_id": remote_id}
        log.info("bench %s dispatched as %s", job["job_id"], remote_id)

    def _poll_running(self) -> None:
        assert self._running is not None
        job = self._running["job"]
        remote = self.client.poll_bench(self._running["remote_id"])
        if remote is None:
            # Bench server unreachable; provisioner will deal with the pod.
            # Requeue the job so it reruns on the replacement machine.
            log.warning("bench %s: bench server unreachable; requeueing", job["job_id"])
            job["state"] = "QUEUED"
            self._running = None
            return
        state = remote.get("state", "")
        if state in ("queued", "LOADING_MODEL", "RUNNING"):
            return
        if state == "aborted":
            log.info("bench %s aborted; requeueing", job["job_id"])
            job["state"] = "QUEUED"
        elif state == "completed":
            self.state.record_bench_result(job, remote.get("result", {"ok": False}))
        else:
            self.state.record_bench_result(
                job, remote.get("result", {"ok": False, "error": f"state={state}"}))
        self._running = None
