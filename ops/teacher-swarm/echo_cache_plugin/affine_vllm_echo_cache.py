"""vLLM general plugin: prefix-cache reuse for teacher-forcing echo requests.

Why: every duel score term is a teacher echo (``echo=True, logprobs=0``)
over ``prefix x + thought + action``. vLLM 0.28 sets
``SamplingParams.skip_reading_prefix_cache = True`` whenever prompt
logprobs are requested, because cached tokens produce no logprobs. The
duel only reads the logprobs of the *span* (the last few hundred tokens),
so with ~21 echoes per turn the teacher re-prefilled the same 5k-60k
token prefix ~21 times and ran the fp32 248k-vocab log-softmax over every
prefix token each time. That was the whole teacher bottleneck (2026-09-07:
16 B200 replicas at 100%, miner engines idle at 1-2 in flight).

Contract with the client (``affine/evalsrv/vllm_client.py::_echo_span``):
the request carries ``vllm_xargs: {"affine_echo_tail": T}`` = number of
trailing prompt tokens whose logprobs the caller needs. This plugin then

1. lets the request read the prefix cache (``Request.get_skip_reading_
   prefix_cache`` -> False),
2. caps the cache hit at ``num_tokens - 1 - T`` so the tail is always
   recomputed (``KVCacheManager.get_computed_blocks``), and
3. pre-fills the cached positions of the prompt-logprobs tensor with the
   real token ids and logprob ``+1.0`` (``GPUModelRunner._get_prompt_
   logprobs_dict``). vLLM leaves them as ``torch.empty`` garbage otherwise,
   which crashes the detokenizer on out-of-vocab ids. A positive logprob
   is impossible, so the client can detect a cache hit that reached into
   its span and fall back to an uncached echo.

Requests without the xarg are untouched: stock vLLM behaviour.

Loaded through the ``vllm.general_plugins`` entry point, so it runs in the
API server, EngineCore (scheduler) and every TP worker. Written against
vLLM 0.28.0; ``register()`` refuses to patch (and says so) when the hooked
attributes are missing, so a vLLM bump fails loud instead of silently
losing the speedup or corrupting logprobs.
"""

from __future__ import annotations

import logging

import torch

# Under vLLM's logger namespace so the "enabled" line reaches the engine
# log (vLLM configures handlers for "vllm.*" only).
logger = logging.getLogger("vllm.affine_echo_cache")

XARG = "affine_echo_tail"
# Impossible logprob; marks positions whose KV came from the prefix cache.
CACHED_SENTINEL = 1.0
_PATCHED = "_affine_echo_cache_patched"


def echo_tail(sampling_params) -> int | None:
    """T from the request's extra_args, or None when the request opted out."""
    extra = getattr(sampling_params, "extra_args", None)
    if not extra:
        return None
    raw = extra.get(XARG)
    if raw is None:
        return None
    try:
        return max(int(raw), 1)
    except (TypeError, ValueError):
        return None


def _patch_request(Request) -> None:
    orig = Request.get_skip_reading_prefix_cache

    def get_skip_reading_prefix_cache(self):
        sp = self.sampling_params
        if sp is not None and echo_tail(sp) is not None:
            return False
        return orig(self)

    Request.get_skip_reading_prefix_cache = get_skip_reading_prefix_cache


def _patch_kv_cache_manager(KVCacheManager) -> None:
    orig = KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request):
        tail = echo_tail(request.sampling_params)
        if tail is None:
            return orig(self, request)
        # Stock code already caps the hit at num_tokens - 1 (the last token
        # must be recomputed for logits). Tighten it to leave the whole tail
        # uncached. The coordinator floors to block / retention boundaries.
        cap = max(request.num_tokens - 1 - tail, 0)
        coord = self.coordinator
        orig_find = coord.find_longest_cache_hit

        def find_longest_cache_hit(block_hashes, max_cache_hit_length):
            return orig_find(block_hashes, min(max_cache_hit_length, cap))

        # The scheduler is single-threaded; a temporary instance attribute is
        # invisible to anyone else and restored before we return.
        coord.find_longest_cache_hit = find_longest_cache_hit
        try:
            return orig(self, request)
        finally:
            del coord.find_longest_cache_hit

    KVCacheManager.get_computed_blocks = get_computed_blocks


def _patch_model_runner(GPUModelRunner, LogprobsTensors) -> None:
    orig = GPUModelRunner._get_prompt_logprobs_dict

    def _get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
        pending = self.num_prompt_logprobs
        if pending:
            for req_id, num_prompt_logprobs in pending.items():
                if req_id not in num_scheduled_tokens:
                    continue
                req = self.requests.get(req_id)
                if req is None or req.prompt_token_ids is None:
                    continue
                if req.in_progress_prompt_logprobs_cpu is not None:
                    continue  # not the first chunk; tensor already exists
                n_prompt = len(req.prompt_token_ids)
                cached = min(int(req.num_computed_tokens), n_prompt - 1)
                if cached <= 0:
                    continue  # no cache hit: stock path allocates the tensor
                tensors = LogprobsTensors.empty_cpu(
                    n_prompt - 1, num_prompt_logprobs + 1)
                # Tensor row i holds the logprob of prompt token i+1. Rows
                # [0, cached) never get written by the model runner because
                # their hidden states came from the cache.
                ids = torch.tensor(req.prompt_token_ids[1:cached + 1],
                                   dtype=torch.int32).unsqueeze(1)
                tensors.logprob_token_ids[:cached] = ids
                tensors.logprobs[:cached] = CACHED_SENTINEL
                tensors.selected_token_ranks[:cached] = 1
                req.in_progress_prompt_logprobs_cpu = tensors
        return orig(self, hidden_states, num_scheduled_tokens)

    GPUModelRunner._get_prompt_logprobs_dict = _get_prompt_logprobs_dict


def register() -> None:
    """Entry point for ``vllm.general_plugins``. Idempotent per process."""
    import vllm
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(vllm, _PATCHED, False):
        return
    needed = (
        (Request, "get_skip_reading_prefix_cache"),
        (KVCacheManager, "get_computed_blocks"),
        (GPUModelRunner, "_get_prompt_logprobs_dict"),
        (LogprobsTensors, "empty_cpu"),
    )
    missing = [f"{c.__name__}.{a}" for c, a in needed if not hasattr(c, a)]
    if missing:
        logger.error("affine_vllm_echo_cache: vLLM %s lacks %s; NOT patching "
                     "(echo requests will run uncached)",
                     vllm.__version__, ", ".join(missing))
        return
    _patch_request(Request)
    _patch_kv_cache_manager(KVCacheManager)
    _patch_model_runner(GPUModelRunner, LogprobsTensors)
    setattr(vllm, _PATCHED, True)
    logger.info("affine_vllm_echo_cache: echo prefix caching enabled "
                "(vLLM %s, xarg %r)", vllm.__version__, XARG)
