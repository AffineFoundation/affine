"""Shadow trainer v8.3 candidate: training/scoring-only fast kernels for the pinned HF Qwen3.5 hybrid model.

Same math as the pinned transformers 5.14.1 modeling_qwen3_5 forward (cache-free path), different kernels:
  * GatedDeltaNet token mixer: fla causal_conv1d (varlen, triton) + fla chunk_gated_delta_rule (varlen) instead of
    the torch fallbacks (F.conv1d + torch_chunk_gated_delta_rule). The gated RMSNorm keeps the exact HF torch formula.
  * Full attention: torch.nn.attention.varlen.varlen_attn (FlashAttention-2, causal, GQA) instead of eager
    matmul-softmax-matmul. Packed sequences are separated by cu_seqlens, never attend across rollouts.
Installed per layer instance (layer.linear_attn.forward / layer.self_attn.forward); module source files,
parameters, names and the model config are untouched. Any call WITH a generation cache falls back to the
original forward (sampling/inference is never changed)."""
import torch
import torch.nn.functional as F

_FLA = None


def _fla():
    global _FLA
    if _FLA is None:
        from fla.modules.convolution import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        _FLA = (causal_conv1d, chunk_gated_delta_rule)
    return _FLA


def _cu(kwargs, hidden_states):
    cu = kwargs.get('cu_seq_lens_q')
    if cu is None:
        if hidden_states.shape[0] != 1:
            raise ValueError('fast kernels need packed (batch 1) inputs with cu_seqlens')
        T = hidden_states.shape[1]
        cu = torch.tensor([0, T], dtype=torch.int32, device=hidden_states.device)
        return cu, T
    return cu, int(kwargs.get('max_length_q') or int((cu[1:] - cu[:-1]).max()))


def gated_rmsnorm_hf(norm, hidden_states, gate):
    """Exact Qwen3_5RMSNormGated.forward math (torch), whatever class the module is."""
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    eps = getattr(norm, 'variance_epsilon', None)
    if eps is None:
        eps = getattr(norm, 'eps')
    h = h * torch.rsqrt(variance + eps)
    h = norm.weight * h.to(input_dtype)
    h = h * F.silu(gate.to(torch.float32))
    return h.to(input_dtype)


def make_gdn_forward(module, original, fused_norm=False):
    causal_conv1d, chunk_gdr = _fla()

    def forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None:
            return original(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        cu, _ = _cu(kwargs, hidden_states)
        m = module
        bsz, T, _ = hidden_states.shape
        mixed_qkv = m.in_proj_qkv(hidden_states)                      # [1, T, C]
        z = m.in_proj_z(hidden_states).reshape(bsz, T, -1, m.head_v_dim)
        b = m.in_proj_b(hidden_states)
        a = m.in_proj_a(hidden_states)
        mixed_qkv, _ = causal_conv1d(mixed_qkv, weight=m.conv1d.weight.squeeze(1), bias=m.conv1d.bias,
                                     activation=m.activation, cu_seqlens=cu)
        query, key, value = torch.split(mixed_qkv, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        query = query.reshape(bsz, T, -1, m.head_k_dim)
        key = key.reshape(bsz, T, -1, m.head_k_dim)
        value = value.reshape(bsz, T, -1, m.head_v_dim)
        beta = b.sigmoid()
        g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
        if m.num_v_heads // m.num_k_heads > 1:
            query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
            key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        core, _ = chunk_gdr(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False,
                            use_qk_l2norm_in_kernel=True, cu_seqlens=cu.to(torch.long))
        core = core.reshape(-1, m.head_v_dim)
        z = z.reshape(-1, m.head_v_dim)
        core = m.norm(core, z) if fused_norm else gated_rmsnorm_hf(m.norm, core, z)
        core = core.reshape(bsz, T, -1)
        return m.out_proj(core)
    return forward


def make_attn_forward(module, original):
    from torch.nn.attention.varlen import varlen_attn
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    def forward(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None:
            return original(hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs)
        m = module
        cu, maxlen = _cu(kwargs, hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, m.head_dim)
        query_states, gate = torch.chunk(m.q_proj(hidden_states).view(*input_shape, -1, m.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = m.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = m.k_norm(m.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = m.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        q = query_states[0].transpose(0, 1).contiguous()   # [T, Hq, D]
        k = key_states[0].transpose(0, 1).contiguous()     # [T, Hkv, D]
        v = value_states[0].transpose(0, 1).contiguous()
        out = varlen_attn(q, k, v, cu, cu, maxlen, maxlen, scale=m.scaling, window_size=(-1, 0), enable_gqa=True)
        attn_output = out.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return m.o_proj(attn_output), None
    return forward


def install(model, *, fused_norm=False):
    """Patch every text decoder layer of a loaded Qwen3_5ForConditionalGeneration. Returns layer counts."""
    layers = model.model.language_model.layers
    n_lin = n_full = 0
    for layer in layers:
        if layer.block_type == 'linear_attention':
            mod = layer.linear_attn
            if not hasattr(mod, '_shadow_original_forward'):
                mod._shadow_original_forward = mod.forward
            mod.forward = make_gdn_forward(mod, mod._shadow_original_forward, fused_norm=fused_norm)
            n_lin += 1
        else:
            mod = layer.self_attn
            if not hasattr(mod, '_shadow_original_forward'):
                mod._shadow_original_forward = mod.forward
            mod.forward = make_attn_forward(mod, mod._shadow_original_forward)
            n_full += 1
    return dict(linear=n_lin, full=n_full)


def packed_log_probs(runtime, items, *, gradients, logit_chunk=4096, checkpoint_head=True):
    """Selected raw log-probs for each (prompt, output) pair, one packed varlen forward.
    Same selected-token math as v8211_batched.batched_log_probs (bf16 lm_head -> fp32 logits -> gather - logsumexp)."""
    from torch.utils.checkpoint import checkpoint
    model, device = runtime.model, runtime.device
    seqs = []
    for prompt, output in items:
        prompt, output = runtime._tokens(prompt), runtime._tokens(output)
        if len(prompt) + len(output) > runtime.max_context_tokens:
            raise ValueError('response context limit')
        seqs.append(prompt + output)
    lens = [len(s) for s in seqs]
    with torch.set_grad_enabled(gradients):
        ids = torch.tensor([t for s in seqs for t in s], device=device)[None]
        pos = torch.cat([torch.arange(n) for n in lens])[None].to(device)
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = torch.tensor(lens).cumsum(0)
        offsets = cu.tolist()
        cu = cu.to(device)
        hidden = model.model(ids, position_ids=pos, use_cache=False, cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                             max_length_q=max(lens), max_length_k=max(lens),
                             attention_mask={'full_attention': None, 'linear_attention': None}).last_hidden_state[0]
        idx = torch.cat([torch.arange(offsets[r] + len(p) - 1, offsets[r] + len(p) + len(o) - 1)
                         for r, (p, o) in enumerate(items)]).to(device)
        states = hidden.index_select(0, idx)
        targets = torch.tensor([t for _, o in items for t in o], device=device)

        def selected(h, t):
            logits = model.lm_head(h).float()
            return logits.gather(-1, t[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        chunks = []
        for start in range(0, len(targets), logit_chunk):
            h, t = states[start:start + logit_chunk], targets[start:start + logit_chunk]
            chunks.append(checkpoint(selected, h, t, use_reentrant=False) if (gradients and checkpoint_head) else selected(h, t))
        flat = torch.cat(chunks)
    return list(torch.split(flat, [len(o) for _, o in items]))
