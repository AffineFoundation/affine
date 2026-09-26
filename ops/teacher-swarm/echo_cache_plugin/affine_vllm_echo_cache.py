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
   real token ids and logprob ``+1.0``. vLLM leaves them out otherwise
   (0.28: ``torch.empty`` garbage that crashes the detokenizer on
   out-of-vocab ids; 0.30: a tensor ``cached`` rows too short, which
   misaligns every logprob with its prompt token and 500s in the API
   server's ``_create_completion_logprobs``). A positive logprob is
   impossible, so the client can detect a cache hit that reached into its
   span and fall back to an uncached echo.

Requests without the xarg are untouched: stock vLLM behaviour.

Loaded through the ``vllm.general_plugins`` entry point, so it runs in the
API server, EngineCore (scheduler) and every TP worker. Hook 3 exists for
both model runners: the legacy ``vllm.v1.worker.gpu_model_runner.
GPUModelRunner._get_prompt_logprobs_dict`` (vLLM 0.28/0.29, the Qwen3.8
swarm) and the v2 runner's ``vllm.v1.worker.gpu.sample.prompt_logprob.
PromptLogprobsWorker.compute_prompt_logprobs`` (vLLM 0.30, the default
runner and the one GLM-5.3-Flash needs). ``register()`` refuses to patch
(and says so) when the hooked attributes are missing, so a vLLM bump
fails loud instead of silently losing the speedup or corrupting logprobs.
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


_TAILS = "_affine_echo_tails"
_first_hits_logged = 0


def _sentinel_rows(like, all_token_ids, state_idx: int, cached: int):
    """``cached`` prompt-logprob rows for tokens 1..cached, shaped and typed
    like ``like`` (a LogprobsTensors the runner just produced) so
    ``LogprobsTensors.cat`` accepts them. Row i holds prompt token i+1 with
    logprob CACHED_SENTINEL and rank 1."""
    dev = like.logprobs.device
    width = int(like.logprob_token_ids.shape[1])
    ids = all_token_ids[state_idx, 1:cached + 1].to(
        device=dev, dtype=like.logprob_token_ids.dtype)
    token_ids = ids.unsqueeze(1).expand(cached, width).contiguous()
    logprobs = torch.full((cached, width), CACHED_SENTINEL,
                          dtype=like.logprobs.dtype, device=dev)
    ranks = torch.ones(cached, dtype=like.selected_token_ranks.dtype,
                       device=dev)
    return type(like)(token_ids, logprobs, ranks)


def _patch_prompt_logprobs_worker(PromptLogprobsWorker, LogprobsTensors) -> None:
    """vLLM >= 0.30 (v2 GPU model runner). ``compute_prompt_logprobs`` only
    emits rows for the query tokens of the current step, so a request whose
    first prefill chunk starts at a cache hit of ``cached`` tokens comes back
    ``cached`` rows short. Prepend the sentinel rows once, on that first
    chunk: to the returned tensor when the prompt fits one step, else to
    the head of the worker's in-progress list so the final ``cat`` carries
    them."""
    orig_add = PromptLogprobsWorker.add_request
    orig_remove = PromptLogprobsWorker.remove_request
    orig_compute = PromptLogprobsWorker.compute_prompt_logprobs

    def add_request(self, req_id, req_idx, sampling_params):
        orig_add(self, req_id, req_idx, sampling_params)
        tails = self.__dict__.setdefault(_TAILS, {})
        tail = None
        if getattr(sampling_params, "prompt_logprobs", None) is not None:
            tail = echo_tail(sampling_params)
        if tail is None:
            tails.pop(req_id, None)
        else:
            tails[req_id] = tail

    def remove_request(self, req_id):
        self.__dict__.get(_TAILS, {}).pop(req_id, None)
        return orig_remove(self, req_id)

    def compute_prompt_logprobs(self, logits_fn, hidden_states, input_batch,
                                all_token_ids, num_computed_tokens, prompt_lens):
        global _first_hits_logged
        tails = self.__dict__.get(_TAILS)
        firsts = []  # (req_id, req_state_idx, cached) for first chunks on a cache hit
        if tails:
            idx_mapping_np = input_batch.idx_mapping_np
            computed = input_batch.num_computed_prefill_tokens_np
            plens = prompt_lens[idx_mapping_np]
            prefill = input_batch.prefill_len_np
            for i, req_id in enumerate(input_batch.req_ids):
                if req_id not in tails:
                    continue
                if self.in_progress_prompt_logprobs.get(req_id):
                    continue  # a later chunk; the first one already prepended
                cached, n_prompt = int(computed[i]), int(plens[i])
                # Same guards as stock: still inside the prompt, and not a
                # request resumed after preemption (its prompt logprobs were
                # already emitted before the preemption).
                if cached <= 0 or cached >= n_prompt or n_prompt < int(prefill[i]):
                    continue
                firsts.append((req_id, int(idx_mapping_np[i]), min(cached, n_prompt - 1)))
        result = orig_compute(self, logits_fn, hidden_states, input_batch,
                              all_token_ids, num_computed_tokens, prompt_lens)
        for req_id, state_idx, cached in firsts:
            done = result.get(req_id)
            if done is not None:
                head = _sentinel_rows(done, all_token_ids, state_idx, cached)
                result[req_id] = LogprobsTensors.cat([head, done])
            else:
                pending = self.in_progress_prompt_logprobs.get(req_id)
                if not pending:
                    continue  # nothing was produced for it this step
                pending.insert(0, _sentinel_rows(pending[0], all_token_ids, state_idx, cached))
            if _first_hits_logged < 3:
                _first_hits_logged += 1
                logger.info("affine_vllm_echo_cache: cached echo req=%s cached=%d "
                            "rows prefilled (%s)", req_id, cached,
                            "single step" if done is not None else "chunked prompt")
        return result

    PromptLogprobsWorker.add_request = add_request
    PromptLogprobsWorker.remove_request = remove_request
    PromptLogprobsWorker.compute_prompt_logprobs = compute_prompt_logprobs


def _runner_hooks(LogprobsTensors):
    """(name, patch-callable) for every model runner this vLLM ships whose
    prompt-logprobs path we know how to hook."""
    hooks = []
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        GPUModelRunner = None
    if (GPUModelRunner is not None
            and hasattr(GPUModelRunner, "_get_prompt_logprobs_dict")
            and hasattr(LogprobsTensors, "empty_cpu")):
        hooks.append(("GPUModelRunner._get_prompt_logprobs_dict",
                      lambda: _patch_model_runner(GPUModelRunner, LogprobsTensors)))
    try:
        from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
    except ImportError:
        PromptLogprobsWorker = None
    if (PromptLogprobsWorker is not None
            and all(hasattr(PromptLogprobsWorker, a)
                    for a in ("add_request", "remove_request", "compute_prompt_logprobs"))
            and hasattr(LogprobsTensors, "cat")):
        hooks.append(("PromptLogprobsWorker.compute_prompt_logprobs",
                      lambda: _patch_prompt_logprobs_worker(PromptLogprobsWorker, LogprobsTensors)))
    return hooks


def register() -> None:
    """Entry point for ``vllm.general_plugins``. Idempotent per process."""
    import vllm
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.request import Request

    if getattr(vllm, _PATCHED, False):
        return
    needed = (
        (Request, "get_skip_reading_prefix_cache"),
        (KVCacheManager, "get_computed_blocks"),
    )
    missing = [f"{c.__name__}.{a}" for c, a in needed if not hasattr(c, a)]
    hooks = _runner_hooks(LogprobsTensors)
    if not hooks:
        missing.append("a hookable prompt-logprobs path (GPUModelRunner."
                       "_get_prompt_logprobs_dict or PromptLogprobsWorker."
                       "compute_prompt_logprobs)")
    if missing:
        logger.error("affine_vllm_echo_cache: vLLM %s lacks %s; NOT patching "
                     "(echo requests will run uncached)",
                     vllm.__version__, ", ".join(missing))
        return
    _patch_request(Request)
    _patch_kv_cache_manager(KVCacheManager)
    for _, patch in hooks:
        patch()
    setattr(vllm, _PATCHED, True)
    logger.info("affine_vllm_echo_cache: echo prefix caching enabled "
                "(vLLM %s, xarg %r, runner hooks: %s)", vllm.__version__, XARG,
                ", ".join(name for name, _ in hooks))
