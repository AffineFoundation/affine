"""Async vLLM clients: sample rollouts, teacher-force logprobs of an action span."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import orjson

from affine import dialects

from .chat import (
    TEXT_FALLBACK_KINDS,
    chat_prompt,
    extract_action,
    force_text,
    gen_prompt,
    get_tokenizer,
    inject_prompt,
    split_rollout,
    think_closed,
    thought_text,
)

log = logging.getLogger("evalsrv.vllm_client")

# Request extra-arg read by the teacher-side vLLM plugin
# (ops/teacher-swarm/echo_cache_plugin/affine_vllm_echo_cache.py): number of
# trailing prompt tokens that must be recomputed (not served from the
# prefix cache) so their logprobs exist.
ECHO_TAIL_XARG = "affine_echo_tail"

# Echo cost accounting (sd-meter shadow, 2026-09-18). Every echo request is
# booked under the tag current in this contextvar — "base" for the live
# min(R,G) echoes, "sd_meter" for the ones only the shadow score needs — so a
# verdict can say what the extra teacher work measured: requests, prompt
# tokens sent, tail tokens the engine had to recompute, and request wall
# seconds (summed over concurrent requests, i.e. engine occupancy, not duel
# wall). Set with ``echo_tag("sd_meter")`` around the shadow calls.
ECHO_TAG: contextvars.ContextVar[str] = contextvars.ContextVar("affine_echo_tag",
                                                              default="base")

# Prefix used for the unconditioned thought echo lpC(z|∅) of the content
# mask: one empty user turn, so the template renders exactly its fixed
# header and the thought is scored with no task in front of it. Rendered
# once per model; ~50 tokens, shared by every such echo via the prefix cache.
UNCOND_PREFIX: list[dict] = [{"role": "user", "content": ""}]


class echo_tag:
    """``with echo_tag("sd_meter"): ...`` — book the echoes inside under tag."""

    def __init__(self, tag: str):
        self.tag = tag
        self._token = None

    def __enter__(self):
        self._token = ECHO_TAG.set(self.tag)
        return self

    def __exit__(self, *exc):
        ECHO_TAG.reset(self._token)
        return False


def _new_echo_stat() -> dict:
    return {"requests": 0, "prompt_tokens": 0, "tail_tokens": 0,
            "span_tokens": 0, "computed_tokens": 0, "seconds": 0.0,
            "uncached_retries": 0}

# Echo bookkeeping off the event loop (2026-09-07 py-spy of a live duel:
# the single duel thread sat at 99% CPU, 47% in the HF tokenizer call that
# locates the span and 32% in json.loads of the echo response, while 16
# teacher replicas idled at <1 request each). The Rust tokenizer releases
# the GIL on batch encodes, so a thread pool runs them in parallel on the
# pod's spare cores; orjson takes the JSON share down ~5x.
_TOK_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="tok")


def _encode_offsets(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """(input_ids, offset_mapping) for one text via the batch path (GIL-free)."""
    enc = tok([text], add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"][0], enc["offset_mapping"][0]


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
    # Name to put in the request's "model" field when it is not the repo:
    # warm-swapped challenger engines serve successive checkpoints under a
    # fixed alias (engine.CHALLENGER_ALIAS). Tokenizer/rendering still use
    # `repo`/`revision` (the checkpoint's own files).
    model_name: str | None = None

    @property
    def request_model(self) -> str:
        return self.model_name or self.repo


class ModelPool:
    """Load-balanced pool over identical replicas of one model.

    Used for the teacher and the dual miner copies. Sticky routing by
    ``sticky_key`` (turn_id) pins every call for one turn — teacher ref
    sample + king echo + challenger echo — to the same replica so vLLM
    automatic prefix caching can reuse the shared turn prefix ``x``.
    Pass a key only on the teacher: miners sample once per turn, so
    pinning cannot hit the cache and hash-splits leave one copy idle.
    Without a key, pick least-in-flight (tie-break round-robin) so
    replica[0] is not preferred whenever it happens to be idle.
    Replicas serve identical weights; which replica answers a sample
    or a temperature-0 echo is score-invariant.
    """

    def __init__(self, replicas: list[VllmModel]):
        assert replicas, "ModelPool needs at least one replica"
        self.replicas = replicas
        self.cfg = replicas[0].cfg
        self._rr = 0

    @property
    def n_samples(self) -> int:
        return sum(r.n_samples for r in self.replicas)

    @property
    def n_think_closed(self) -> int:
        return sum(r.n_think_closed for r in self.replicas)

    @property
    def n_text_fallback(self) -> int:
        """Samples split as a prose `text` action at a tool_call turn (wvk 18)."""
        return sum(r.n_text_fallback for r in self.replicas)

    @property
    def think_close_rate(self) -> float | None:
        """Fraction of natural samples that emitted </think> (all replicas)."""
        n = self.n_samples
        return self.n_think_closed / n if n else None

    def echo_stats(self) -> dict[str, dict]:
        """Echo cost per tag, summed over the replicas (see ECHO_TAG)."""
        out: dict[str, dict] = {}
        for r in self.replicas:
            for tag, s in r.echo_stats().items():
                acc = out.setdefault(tag, _new_echo_stat())
                for k, v in s.items():
                    acc[k] += v
        return out

    async def complete(self, messages: list[dict], temperature: float,
                       max_tokens: int, *, tools: list[dict] | None = None
                       ) -> str:
        return await self._pick().complete(
            messages, temperature, max_tokens, tools=tools)

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
                     max_tokens: int, *, sticky_key: str | None = None,
                     action_kind: str | None = None) -> tuple[str, str]:
        return await self._pick(sticky_key).sample(
            prefix_messages, temperature, max_tokens, sticky_key=sticky_key,
            action_kind=action_kind)

    async def sample_injected(self, prefix_messages: list[dict], thoughts: str,
                              temperature: float, max_tokens: int, *,
                              sticky_key: str | None = None,
                              action_kind: str | None = None) -> str:
        return await self._pick(sticky_key).sample_injected(
            prefix_messages, thoughts, temperature, max_tokens,
            sticky_key=sticky_key, action_kind=action_kind)

    async def score_action(self, prefix_messages: list[dict], thoughts: str,
                           action: str, *, sticky_key: str | None = None
                           ) -> dict:
        return await self._pick(sticky_key).score_action(
            prefix_messages, thoughts, action, sticky_key=sticky_key)

    async def score_thought(self, prefix_messages: list[dict], thoughts: str,
                            *, sticky_key: str | None = None,
                            tokens: bool = False) -> dict:
        return await self._pick(sticky_key).score_thought(
            prefix_messages, thoughts, sticky_key=sticky_key, tokens=tokens)

    async def score_thought_uncond(self, thoughts: str, *,
                                   sticky_key: str | None = None) -> dict:
        return await self._pick(sticky_key).score_thought_uncond(
            thoughts, sticky_key=sticky_key)


class VllmModel:
    def __init__(self, cfg: Served, client: httpx.AsyncClient, sem: asyncio.Semaphore,
                 require_think_close: bool = False,
                 text_fallback_at_tool_turns: bool = False):
        self.cfg = cfg
        if cfg.base_url:
            self.base = cfg.base_url.rstrip("/")
        else:
            self.base = f"http://localhost:{cfg.port}/v1"
        self.http = client
        self.sem = sem
        self.in_flight = 0
        # [duel].require_think_close: a natural sample without </think>
        # splits to ("", "") — a forfeit. Set per side by run_duel (miner
        # sides only; teacher refs keep their pre-knob semantics).
        self.require_think_close = require_think_close
        # Well-formedness telemetry, counted on every natural sample whether
        # or not the knob is on, so the live rate is known before any flip.
        self.n_samples = 0
        self.n_think_closed = 0
        # [duel].text_fallback_at_tool_turns (wvk 18): a closed-think prose
        # reply at a tool_call turn is a `text` action. Counted per sample.
        self.text_fallback_at_tool_turns = text_fallback_at_tool_turns
        self.n_text_fallback = 0
        # Echo cost per ECHO_TAG (requests / prompt tokens / recomputed tail
        # tokens / request seconds); read by ModelPool.echo_stats.
        self._echo_stats: dict[str, dict] = defaultdict(_new_echo_stat)

    def echo_stats(self) -> dict[str, dict]:
        return {k: dict(v) for k, v in self._echo_stats.items()}

    async def _post(self, payload: dict) -> dict:
        # Keep per-request timeout under vLLM hang windows but above worst-case
        # queue wait. 180s was sized for 32k-ctx echo on the B300 pod; on the
        # H200 pod with turn_conc=concurrency a tail echo request can sit in
        # the teacher queue past 180s while the engine is healthy (observed
        # 2026-08-13: ReadTimeout x3 killed an hour-long duel mid-scoring).
        # Retries absorb transient engine stalls without wedging the gather.
        timeout = httpx.Timeout(480.0, connect=10.0)
        async with self.sem:
            self.in_flight += 1
            try:
                for attempt in range(3):
                    try:
                        r = await self.http.post(
                            f"{self.base}/completions", json=payload, timeout=timeout
                        )
                        r.raise_for_status()
                        return orjson.loads(r.content)
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
                     max_tokens: int, *, sticky_key: str | None = None,
                     action_kind: str | None = None) -> tuple[str, str]:
        """Natural rollout -> (thoughts, action) under the turn's dialect."""
        del sticky_key  # only ModelPool uses this; accepted for API symmetry
        # add_special_tokens=False must match score_action so vLLM automatic
        # prefix caching can reuse the shared turn-prefix token blocks.
        d = await self._post({
            "model": self.cfg.request_model,
            "prompt": gen_prompt(self.cfg.repo, self.cfg.revision, prefix_messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "add_special_tokens": False,
        })
        text = d["choices"][0]["text"]
        self.n_samples += 1
        self.n_think_closed += int(think_closed(text))
        z, y = split_rollout(text, action_kind,
                             require_think_close=self.require_think_close,
                             text_fallback_at_tool_turns=self.text_fallback_at_tool_turns)
        if (y and self.text_fallback_at_tool_turns
                and (action_kind or dialects.DEFAULT_KIND) in TEXT_FALLBACK_KINDS
                and dialects.count_actions(y, action_kind) == 0):
            self.n_text_fallback += 1
        return z, y

    async def complete(self, messages: list[dict], temperature: float,
                       max_tokens: int, *, tools: list[dict] | None = None
                       ) -> str:
        """Raw completion text for a chat rendered the way an OpenAI client
        would send it (own template, thinking on, optional tool schemas).
        Not split, not counted in the duel's well-formedness telemetry —
        the protocol probe evaluates the text itself."""
        d = await self._post({
            "model": self.cfg.request_model,
            "prompt": chat_prompt(self.cfg.repo, self.cfg.revision, messages,
                                  tools=tools),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "add_special_tokens": False,
        })
        return d["choices"][0]["text"]

    async def sample_injected(self, prefix_messages: list[dict], thoughts: str,
                              temperature: float, max_tokens: int, *,
                              sticky_key: str | None = None,
                              action_kind: str | None = None) -> str:
        """Rollout with planted thoughts -> action only."""
        del sticky_key
        d = await self._post({
            "model": self.cfg.request_model,
            "prompt": inject_prompt(self.cfg.repo, self.cfg.revision,
                                    prefix_messages, thoughts),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "add_special_tokens": False,
        })
        return extract_action(d["choices"][0]["text"], action_kind)

    async def _echo_span(self, full: str, span_start: int,
                         span_bytes: int, *, tokens: bool = False) -> dict:
        """Teacher-force echo: mean logprob per byte of full[span_start:].

        echo=True + logprobs, span located with the tokenizer's offset
        mapping on the full text (robust to BPE merges across the injection
        boundary); add_special_tokens=False keeps vLLM's tokenization aligned.

        tokens=True additionally returns ``tokens``: one (start, end, lp) per
        span token with character offsets relative to span_start — the
        per-token view the sd-meter content mask aligns across the
        conditioned and unconditioned echoes of the same thought. The
        request is identical either way (the engine always returns
        per-token logprobs; only what is kept changes).
        """
        stat = self._echo_stats[ECHO_TAG.get()]
        t0 = time.monotonic()
        tok = get_tokenizer(self.cfg.repo, self.cfg.revision)
        input_ids, offsets = await asyncio.get_running_loop().run_in_executor(
            _TOK_POOL, _encode_offsets, tok, full)
        n_prompt = sum(1 for s, _ in offsets if s < span_start)
        n_total = len(input_ids)
        tail = n_total - n_prompt + 1 + 8
        stat["requests"] += 1
        stat["prompt_tokens"] += n_total
        stat["tail_tokens"] += min(tail, n_total)
        stat["span_tokens"] += n_total - n_prompt
        payload = {
            "model": self.cfg.request_model,
            "prompt": full,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 0,
            "add_special_tokens": False,
            # Echo prefix caching (ops/teacher-swarm/echo_cache_plugin): let
            # the engine reuse the cached prefix KV and recompute only the
            # tail we read. T = span tokens + 1 (token n_prompt's logprob
            # comes from hidden state n_prompt-1) + slack for tokenizer
            # drift at the injection boundary. Stock vLLM ignores the xarg
            # and echoes uncached, exactly as before.
            "vllm_xargs": {ECHO_TAIL_XARG: tail},
        }
        d = await self._post(payload)
        lp = d["choices"][0]["logprobs"]["token_logprobs"]
        # Span logprobs: everything after the prompt tokens (last generated
        # token excluded: echo returns prompt tokens + 1 generated).
        raw_span = lp[n_prompt:-1]
        if any(x is None or x > 0 for x in raw_span):
            # A cached block reached into the span (the plugin marks cached
            # positions with an impossible positive logprob) or the engine
            # withheld a position. Score-bearing bytes must be computed:
            # redo the echo with the cache lookup off (stock behaviour).
            log.warning("%s echo span touched the prefix cache (%d/%d "
                        "positions); retrying uncached", self.cfg.name,
                        sum(1 for x in raw_span if x is None or x > 0),
                        len(raw_span))
            payload.pop("vllm_xargs")
            stat["uncached_retries"] += 1
            stat["requests"] += 1
            stat["prompt_tokens"] += n_total
            stat["tail_tokens"] += n_total
            d = await self._post(payload)
            lp = d["choices"][0]["logprobs"]["token_logprobs"]
            raw_span = lp[n_prompt:-1]
        span = [x for x in raw_span if x is not None]
        # Positions the engine actually computed (cached prefix rows come
        # back as the plugin's +1.0 marker or None): the measured recompute.
        stat["computed_tokens"] += sum(1 for x in lp if x is not None and x <= 0)
        n_bytes = max(span_bytes, 1)
        out = {
            "sum_lp": sum(span),
            "n_tokens": len(span),
            "n_bytes": n_bytes,
            "lp_per_byte": sum(span) / n_bytes if span else 0.0,
        }
        if tokens:
            span_offsets = offsets[n_prompt:]
            out["tokens"] = [
                (s - span_start, e - span_start, x)
                for (s, e), x in zip(span_offsets, raw_span) if x is not None]
        stat["seconds"] += time.monotonic() - t0
        return out

    async def score_action(self, prefix_messages: list[dict], thoughts: str,
                           action: str, *, sticky_key: str | None = None
                           ) -> dict:
        """Mean logprob per byte of `action` given (prefix, thoughts)."""
        del sticky_key
        full = force_text(self.cfg.repo, self.cfg.revision, prefix_messages,
                          thoughts, action)
        return await self._echo_span(full, len(full) - len(action),
                                     len(action.encode()))

    async def score_thought(self, prefix_messages: list[dict], thoughts: str,
                            *, sticky_key: str | None = None,
                            tokens: bool = False) -> dict:
        """Mean logprob per byte of `thoughts` given the turn prefix x.

        Grounding-leg echo for min(R, G): m = lpC(z_A|x) for the miner's
        thought, t_i = lpC(z_C^i|x) for each teacher reference thought. Uses
        the same canonical injected rendering as score_action minus the
        action, so the scored span is exactly the thought bytes. tokens=True
        keeps the per-token logprobs (sd-meter content mask); same request.
        """
        del sticky_key
        full = thought_text(self.cfg.repo, self.cfg.revision, prefix_messages,
                            thoughts)
        return await self._echo_span(full, len(full) - len(thoughts),
                                     len(thoughts.encode()), tokens=tokens)

    async def score_thought_uncond(self, thoughts: str, *,
                                   sticky_key: str | None = None) -> dict:
        """Per-token logprobs of `thoughts` with NO task in front of it:
        lpC(z|∅), the same canonical THOUGHT rendering after UNCOND_PREFIX
        (one empty user turn). The sd-meter content mask keeps the tokens
        whose logprob the task moves by more than θ nats between this echo
        and score_thought's. Short prompt (header + thought), no turn prefix
        to cache — the cost is the thought's own tokens once."""
        del sticky_key
        full = thought_text(self.cfg.repo, self.cfg.revision, UNCOND_PREFIX,
                            thoughts)
        return await self._echo_span(full, len(full) - len(thoughts),
                                     len(thoughts.encode()), tokens=True)
