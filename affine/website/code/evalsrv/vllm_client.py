"""Async vLLM clients: sample rollouts, teacher-force logprobs of an action span."""

from __future__ import annotations

import asyncio
import zlib
from dataclasses import dataclass

import httpx

from .chat import (
    extract_action,
    force_text,
    gen_prompt,
    get_tokenizer,
    inject_prompt,
    split_rollout,
)


class ContextLengthError(RuntimeError):
    """vLLM rejected the request: prompt (+ max_tokens) > max_model_len.

    This is a serving-config / corpus-length mismatch, not a model fault —
    the duel server maps it to Fault.CONTEXT_LIMIT (infra, requeue).
    """


class EngineUnreachableError(RuntimeError):
    """vLLM HTTP endpoint unreachable after retries — pod/infra, not checkpoint.

    Typical after an evalsrv bounce while a slot's engine is still coming up,
    or when an engine dies mid-duel. The duel server maps this to an infra
    Fault (teacher / king_launch / challenger_infra) so the validator requeues
    without burning the miner.
    """

    def __init__(self, name: str, detail: str):
        self.name = name
        super().__init__(f"{name}: {detail}")


# Transport failures that mean "engine not answering", not "bad checkpoint".
_UNREACHABLE_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


@dataclass(frozen=True)
class Served:
    name: str            # short id used in result rows ("king", "challenger", "teacher")
    repo: str            # HF repo (also the vLLM served-model name)
    revision: str | None
    port: int
    # Optional OpenAI-compatible base including /v1 (e.g. remote teacher box).
    # When set, clients hit this URL instead of localhost:{port}/v1.
    base_url: str | None = None


class ModelPool:
    """Load-balanced pool over identical replicas of one model.

    Used for the teacher. Sticky routing by ``sticky_key`` (turn_id) pins
    every call for one turn — ref sample + king echo + challenger echo — to
    the same replica so vLLM automatic prefix caching can reuse the shared
    turn prefix ``x``. Without a key, pick least-in-flight (tie-break
    round-robin) so replica[0] is not preferred whenever it happens to be
    idle. Replicas serve identical weights at temperature-0 echo scoring, so
    which replica answers is score-invariant.
    """

    def __init__(self, replicas: list[VllmModel]):
        assert replicas, "ModelPool needs at least one replica"
        self.replicas = replicas
        self.cfg = replicas[0].cfg
        self._rr = 0

    def _pick(self, sticky_key: str | None = None) -> VllmModel:
        n = len(self.replicas)
        if sticky_key is not None and n > 1:
            # Stable, uniform: adler32 is fast and dependency-free.
            idx = zlib.adler32(sticky_key.encode("utf-8")) % n
            return self.replicas[idx]
        # Least in-flight; among ties advance RR so we do not stick on [0].
        self._rr = (self._rr + 1) % n
        best_i = min(
            range(n),
            key=lambda i: (self.replicas[i].in_flight, (i - self._rr) % n),
        )
        return self.replicas[best_i]

    async def sample(self, prefix_messages: list[dict], temperature: float,
                     max_tokens: int, *, sticky_key: str | None = None
                     ) -> tuple[str, str]:
        return await self._pick(sticky_key).sample(
            prefix_messages, temperature, max_tokens, sticky_key=sticky_key)

    async def sample_injected(self, prefix_messages: list[dict], thoughts: str,
                              temperature: float, max_tokens: int, *,
                              sticky_key: str | None = None) -> str:
        return await self._pick(sticky_key).sample_injected(
            prefix_messages, thoughts, temperature, max_tokens,
            sticky_key=sticky_key)

    async def score_action(self, prefix_messages: list[dict], thoughts: str,
                           action: str, *, sticky_key: str | None = None
                           ) -> dict:
        return await self._pick(sticky_key).score_action(
            prefix_messages, thoughts, action, sticky_key=sticky_key)


class VllmModel:
    def __init__(self, cfg: Served, client: httpx.AsyncClient, sem: asyncio.Semaphore):
        self.cfg = cfg
        if cfg.base_url:
            self.base = cfg.base_url.rstrip("/")
        else:
            self.base = f"http://localhost:{cfg.port}/v1"
        self.http = client
        self.sem = sem
        self.in_flight = 0

    async def _post(self, payload: dict) -> dict:
        # Keep per-request timeout well under vLLM hang windows. 180s covers
        # echo+logprobs on 32k ctx with prompt_logprobs; retries absorb
        # transient engine stalls without wedging the whole asyncio gather.
        timeout = httpx.Timeout(180.0, connect=10.0)
        async with self.sem:
            self.in_flight += 1
            try:
                for attempt in range(3):
                    try:
                        r = await self.http.post(
                            f"{self.base}/completions", json=payload, timeout=timeout
                        )
                        r.raise_for_status()
                        return r.json()
                    except httpx.HTTPStatusError as e:
                        body = (e.response.text or "")[:500]
                        # Context-length 400s are deterministic for this prompt —
                        # retrying burns minutes and used to exhaust the miner's
                        # transient budget as eval_infra_exhausted.
                        if e.response.status_code == 400 and (
                                "maximum context length" in body
                                or "max_model_len" in body
                                or "input_tokens" in body):
                            raise ContextLengthError(
                                f"{self.cfg.name} prompt exceeds max_model_len: "
                                f"{body}") from e
                        if attempt == 2:
                            raise httpx.HTTPStatusError(
                                f"{e}; body={body}",
                                request=e.request, response=e.response) from e
                        await asyncio.sleep(2 * (attempt + 1))
                    except _UNREACHABLE_EXC as e:
                        if attempt == 2:
                            raise EngineUnreachableError(
                                self.cfg.name,
                                f"{type(e).__name__}:{e}",
                            ) from e
                        await asyncio.sleep(2 * (attempt + 1))
                    except (httpx.HTTPError, httpx.TimeoutException):
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 * (attempt + 1))
            finally:
                self.in_flight -= 1
        raise RuntimeError("unreachable")

    async def sample(self, prefix_messages: list[dict], temperature: float,
                     max_tokens: int, *, sticky_key: str | None = None
                     ) -> tuple[str, str]:
        """Natural rollout -> (thoughts, action)."""
        del sticky_key  # only ModelPool uses this; accepted for API symmetry
        # add_special_tokens=False must match score_action so vLLM automatic
        # prefix caching can reuse the shared turn-prefix token blocks.
        d = await self._post({
            "model": self.cfg.repo,
            "prompt": gen_prompt(self.cfg.repo, self.cfg.revision, prefix_messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "add_special_tokens": False,
        })
        return split_rollout(d["choices"][0]["text"])

    async def sample_injected(self, prefix_messages: list[dict], thoughts: str,
                              temperature: float, max_tokens: int, *,
                              sticky_key: str | None = None) -> str:
        """Rollout with planted thoughts -> action only."""
        del sticky_key
        d = await self._post({
            "model": self.cfg.repo,
            "prompt": inject_prompt(self.cfg.repo, self.cfg.revision,
                                    prefix_messages, thoughts),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "add_special_tokens": False,
        })
        return extract_action(d["choices"][0]["text"])

    async def score_action(self, prefix_messages: list[dict], thoughts: str,
                           action: str, *, sticky_key: str | None = None
                           ) -> dict:
        """Teacher-force: mean logprob per byte of `action` given (prefix, thoughts).

        echo=True + logprobs, action span located with the tokenizer's offset
        mapping on the full text (robust to BPE merges across the injection
        boundary); add_special_tokens=False keeps vLLM's tokenization aligned.
        """
        del sticky_key
        tok = get_tokenizer(self.cfg.repo, self.cfg.revision)
        full = force_text(self.cfg.repo, self.cfg.revision, prefix_messages,
                          thoughts, action)
        action_start = len(full) - len(action)
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        n_prompt = sum(1 for s, _ in enc["offset_mapping"] if s < action_start)
        d = await self._post({
            "model": self.cfg.repo,
            "prompt": full,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 0,
            "add_special_tokens": False,
        })
        lp = d["choices"][0]["logprobs"]["token_logprobs"]
        # Action-span logprobs: everything after the prompt tokens (last
        # generated token excluded: echo returns prompt tokens + 1 generated).
        span = [x for x in lp[n_prompt:-1] if x is not None]
        n_bytes = max(len(action.encode()), 1)
        return {
            "sum_lp": sum(span),
            "n_tokens": len(span),
            "n_bytes": n_bytes,
            "lp_per_byte": sum(span) / n_bytes if span else 0.0,
        }
