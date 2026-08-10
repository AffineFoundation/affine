"""Async vLLM clients: sample rollouts, teacher-force logprobs of an action span."""

from __future__ import annotations

import asyncio
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
    """Round-robin over identical replicas of one model.

    Used for the teacher, which does ~2/3 of all forced-logprob work in a
    duel: a second GLM replica on otherwise-idle GPUs doubles teacher
    throughput. Each replica keeps its own semaphore (its own queue-depth
    budget); calls are spread least-loaded-first so a slow replica does not
    stall the pool. Replicas serve identical weights at temperature-0 echo
    scoring, so which replica answers is score-invariant.
    """

    def __init__(self, replicas: list[VllmModel]):
        assert replicas, "ModelPool needs at least one replica"
        self.replicas = replicas
        self.cfg = replicas[0].cfg
        self._rr = 0

    def _pick(self) -> VllmModel:
        # Prefer an idle replica; fall back to round-robin when all busy.
        for m in self.replicas:
            if not m.sem.locked():
                return m
        self._rr = (self._rr + 1) % len(self.replicas)
        return self.replicas[self._rr]

    async def sample(self, prefix_messages: list[dict], temperature: float,
                     max_tokens: int) -> tuple[str, str]:
        return await self._pick().sample(prefix_messages, temperature,
                                         max_tokens)

    async def sample_injected(self, prefix_messages: list[dict], thoughts: str,
                              temperature: float, max_tokens: int) -> str:
        return await self._pick().sample_injected(prefix_messages, thoughts,
                                                  temperature, max_tokens)

    async def score_action(self, prefix_messages: list[dict], thoughts: str,
                           action: str) -> dict:
        return await self._pick().score_action(prefix_messages, thoughts,
                                               action)


class VllmModel:
    def __init__(self, cfg: Served, client: httpx.AsyncClient, sem: asyncio.Semaphore):
        self.cfg = cfg
        if cfg.base_url:
            self.base = cfg.base_url.rstrip("/")
        else:
            self.base = f"http://localhost:{cfg.port}/v1"
        self.http = client
        self.sem = sem

    async def _post(self, payload: dict) -> dict:
        # Keep per-request timeout well under vLLM hang windows. 180s covers
        # echo+logprobs on 32k ctx with prompt_logprobs; retries absorb
        # transient engine stalls without wedging the whole asyncio gather.
        timeout = httpx.Timeout(180.0, connect=10.0)
        async with self.sem:
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
                except (httpx.HTTPError, httpx.TimeoutException):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    async def sample(self, prefix_messages: list[dict], temperature: float,
                     max_tokens: int) -> tuple[str, str]:
        """Natural rollout -> (thoughts, action)."""
        d = await self._post({
            "model": self.cfg.repo,
            "prompt": gen_prompt(self.cfg.repo, self.cfg.revision, prefix_messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return split_rollout(d["choices"][0]["text"])

    async def sample_injected(self, prefix_messages: list[dict], thoughts: str,
                              temperature: float, max_tokens: int) -> str:
        """Rollout with planted thoughts -> action only."""
        d = await self._post({
            "model": self.cfg.repo,
            "prompt": inject_prompt(self.cfg.repo, self.cfg.revision,
                                    prefix_messages, thoughts),
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return extract_action(d["choices"][0]["text"])

    async def score_action(self, prefix_messages: list[dict], thoughts: str,
                           action: str) -> dict:
        """Teacher-force: mean logprob per byte of `action` given (prefix, thoughts).

        echo=True + logprobs, action span located with the tokenizer's offset
        mapping on the full text (robust to BPE merges across the injection
        boundary); add_special_tokens=False keeps vLLM's tokenization aligned.
        """
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
