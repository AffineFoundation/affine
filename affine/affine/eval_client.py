"""Root-side client for the eval server (through the provisioner's tunnel).

Duels stream over SSE with an idle watchdog: the server heartbeats every
30s, so a silent gap longer than `stream_idle_timeout_s` means the pod (or
tunnel) is wedged — we classify that as a transient infra failure so the
validator can requeue the challenge at the front without burning it.

If the stream breaks or ends without a verdict, the job record is fetched
directly (`GET /duel/{id}`) before declaring the eval transient: the duel may
have finished server-side and only the stream was lost — dropping that
verdict would waste GPU-hours and a retry.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

log = logging.getLogger("affine.eval_client")


class TransientEvalError(Exception):
    """Infra-side failure (pod down, stream idle, 5xx). Requeue, don't burn."""


class EvalBusyError(TransientEvalError):
    """Server stayed busy (bench/other duel). Scheduling contention is never
    the miner's fault: requeue without spending the retry budget."""


class InfraFaultError(TransientEvalError):
    """The eval server diagnosed its own infrastructure as the cause (low
    disk, dead teacher/king launch, pod too small). Requeue without spending
    the miner's retry budget."""


class Fault:
    """Structured error codes the eval server stamps on failure events, so the
    root client classifies faults by an explicit contract field rather than by
    sniffing free-form exception text. A king that will not *launch* is a pod
    fault (KING_LAUNCH), never a "king is gone" signal — the root proves the
    king actually vanished from HF with a metadata probe before dethroning it."""

    TEACHER = "teacher_unservable"        # our reference model would not serve
    KING_LAUNCH = "king_launch_failed"    # king would not launch (transient/pod)
    POD_CAPACITY = "pod_capacity"         # challenger cannot fit the pod's disk
    CHALLENGER_INFRA = "challenger_infra"  # load failed for a pod-side reason
    # Prompt (+ max_tokens) exceeded vLLM --max-model-len. Our serving config /
    # corpus length mismatch — never the checkpoint's fault.
    CONTEXT_LIMIT = "context_limit"


# Every code above is OUR infrastructure, never the miner's fault: the duel is
# requeued without spending the miner's bounded retry budget. An error event
# with no known code is treated as a generic transient (bounded retries).
INFRA_FAULT_CODES = frozenset({
    Fault.TEACHER, Fault.KING_LAUNCH, Fault.POD_CAPACITY, Fault.CHALLENGER_INFRA,
    Fault.CONTEXT_LIMIT,
})


class EvalClient:
    def __init__(self, base_url: str, *, duel_timeout_s: int,
                 stream_idle_timeout_s: int, eval_token: str = ""):
        self.base = base_url.rstrip("/")
        self.duel_timeout_s = duel_timeout_s
        self.stream_idle_timeout_s = stream_idle_timeout_s
        self._headers = {"X-Affine-Token": eval_token} if eval_token else {}

    # -- prefetch --------------------------------------------------------------
    async def prefetch(self, repo: str, revision: str,
                       weight_bytes: int = 0) -> None:
        """Best-effort hint: warm the next challenger's weights on the pod
        while the current duel scores. Every failure is swallowed — a missed
        prefetch only means the next duel pays the download as before."""
        try:
            async with httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    headers=self._headers) as client:
                r = await client.post(f"{self.base}/prefetch",
                                      json={"repo": repo, "revision": revision,
                                            "weight_bytes": weight_bytes})
                body = r.json() if r.status_code == 200 else {}
                if body.get("accepted"):
                    log.info("prefetch accepted for %s@%s (%s)",
                             repo, revision[:12], body.get("reason"))
                else:
                    log.info("prefetch declined for %s: %s", repo,
                             body.get("reason") or f"HTTP {r.status_code}")
        except Exception as e:
            log.debug("prefetch call failed for %s (ignored): %s", repo, e)

    # -- duels ---------------------------------------------------------------
    async def run_duel(self, *, king_repo: str, king_revision: str,
                       challenger_repo: str, challenger_revision: str,
                       challenger_hotkey: str, block_hash: str,
                       challenger_weight_bytes: int = 0,
                       on_progress=None) -> dict:
        """Dispatch a duel and stream to verdict. Raises TransientEvalError on
        infra failure; returns the verdict dict on completion."""
        payload = {
            "king_repo": king_repo, "king_revision": king_revision,
            "challenger_repo": challenger_repo,
            "challenger_revision": challenger_revision,
            "challenger_hotkey": challenger_hotkey, "block_hash": block_hash,
            "challenger_weight_bytes": challenger_weight_bytes,
        }
        timeout = httpx.Timeout(self.duel_timeout_s, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout,
                                     headers=self._headers) as client:
            # 409 = busy. A bench is aborted server-side on the first attempt;
            # bounded backoff while it winds down, then EvalBusyError (which
            # the validator treats as machine contention — no burn).
            for attempt in range(30):
                try:
                    resp = await client.post(f"{self.base}/duel", json=payload)
                except httpx.HTTPError as e:
                    raise TransientEvalError(f"duel dispatch failed: {e}") from e
                if resp.status_code == 409:
                    log.info("eval server busy (attempt %d/30): %s; waiting 30s",
                             attempt + 1, resp.text[:120])
                    await asyncio.sleep(30)
                    continue
                if resp.status_code == 503:
                    raise TransientEvalError(f"eval server not ready: {resp.text[:200]}")
                resp.raise_for_status()
                break
            else:
                raise EvalBusyError("eval server busy for 15 minutes")

            job_id = resp.json()["job_id"]
            log.info("duel dispatched as %s", job_id)

            verdict = None
            stream_error: Exception | None = None
            try:
                # read timeout = idle timeout: the server heartbeats every
                # 30s, so a silent connection longer than this means the pod
                # or tunnel is wedged and httpx raises ReadTimeout for us.
                stream_timeout = httpx.Timeout(
                    self.duel_timeout_s, connect=30.0,
                    read=self.stream_idle_timeout_s)
                async with client.stream(
                        "GET", f"{self.base}/duel/{job_id}/stream",
                        timeout=stream_timeout) as stream:
                    async for line in stream.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        if event["type"] in ("progress", "heartbeat"):
                            if on_progress:
                                on_progress(event["data"])
                        elif event["type"] == "verdict":
                            verdict = event["data"]
                            break
                        elif event["type"] == "error":
                            err = event["data"].get("error", "?")
                            code = event["data"].get("code")
                            if code in INFRA_FAULT_CODES:
                                raise InfraFaultError(
                                    f"eval server infra fault [{code}]: {err}")
                            raise TransientEvalError(f"eval server error: {err}")
                        else:
                            log.warning("unknown SSE event type %r", event["type"])
            except TransientEvalError:
                raise
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                stream_error = e

            if verdict is None:
                verdict = await self._fetch_verdict(client, job_id)
            if verdict is None:
                if stream_error is not None:
                    raise TransientEvalError(
                        f"duel stream broke: {stream_error}") from stream_error
                raise TransientEvalError("duel stream ended without verdict")
        return verdict

    async def fetch_artifact(self, job_id: str) -> bytes | None:
        """Gzipped full duel record (rollouts + logprobs) for a finished job.
        Best-effort: publishing training data must never affect the verdict."""
        try:
            async with httpx.AsyncClient(headers=self._headers,
                                         timeout=httpx.Timeout(120.0)) as client:
                r = await client.get(f"{self.base}/duel/{job_id}/artifact")
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
        except httpx.HTTPError:
            log.warning("artifact fetch failed for %s", job_id, exc_info=True)
            return None

    async def _fetch_verdict(self, client: httpx.AsyncClient,
                             job_id: str) -> dict | None:
        """SSE race / broken-stream fallback: the duel may have completed
        server-side even though we lost the stream."""
        try:
            r = await client.get(f"{self.base}/duel/{job_id}",
                                 timeout=httpx.Timeout(30.0))
            r.raise_for_status()
            job = r.json()
            if job.get("state") == "completed" and job.get("verdict"):
                log.info("recovered verdict for %s via job poll", job_id)
                return job["verdict"]
        except httpx.HTTPError:
            log.warning("verdict fallback poll failed for %s", job_id,
                        exc_info=True)
        return None

    # -- benchmarks -------------------------------------------------------------
    def try_start_bench(self, *, repo: str, revision: str, suite: str,
                        num_trials: int, max_concurrency: int,
                        user_llm: str) -> str | None:
        """Non-blocking dispatch; returns job_id or None when busy/down."""
        try:
            r = httpx.post(f"{self.base}/bench", json={
                "repo": repo, "revision": revision, "suite": suite,
                "num_trials": num_trials, "max_concurrency": max_concurrency,
                "user_llm": user_llm,
            }, timeout=30, headers=self._headers)
            if r.status_code == 409:
                return None
            r.raise_for_status()
            return r.json()["job_id"]
        except httpx.HTTPError:
            log.warning("bench dispatch failed", exc_info=True)
            return None

    def fetch_bench_artifact(self, job_id: str) -> bytes | None:
        """Gzipped per-task bench rollouts for a finished job. Sync (the
        bench orchestrator pumps synchronously); best-effort like the duel
        artifact — publishing must never affect the recorded result."""
        try:
            r = httpx.get(f"{self.base}/bench/{job_id}/artifact",
                          timeout=120, headers=self._headers)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except httpx.HTTPError:
            log.warning("bench artifact fetch failed for %s", job_id,
                        exc_info=True)
            return None

    def poll_bench(self, job_id: str) -> dict | None:
        try:
            r = httpx.get(f"{self.base}/bench/{job_id}", timeout=30,
                          headers=self._headers)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            return None
