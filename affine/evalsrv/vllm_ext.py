"""vLLM worker extension for weight-swap verification.

Loaded into every TP worker with
    vllm serve ... --worker-extension-cls evalsrv.vllm_ext.WeightTools
and driven over the dev-mode RPC endpoint
    POST /collective_rpc {"method": "<name>", "kwargs": {...}}
(kwargs arrive as strings; results are returned per worker).

Purpose (2026-09-07): the challenger warm swap (`reload_weights` into a
live engine) logged "weights were not loaded" for a per-rank subset of
fused MoE expert tensors. `affine_param_digest` lets a test harness compare EVERY
tensor of a swapped engine against a freshly loaded one (done 2026-09-07 on
TP1 and TP2: bit-identical). `affine_direct_load` is the production swap
path used by engine._swap_weights; its loaded/missing report is checked
against pinned-architecture constants there.
"""

from __future__ import annotations

import copy
import hashlib
import json

import torch


class WeightTools:
    """Mixed into vllm.v1.worker.gpu_worker.Worker; `self.model_runner` is
    the GPUModelRunner."""

    def affine_param_digest(self) -> str:
        """sha256 of every parameter's and buffer's raw bytes (rank-local
        shard). JSON {name: "shape|dtype|sha256"}."""
        model = self.model_runner.get_model()
        out: dict[str, str] = {}
        items = list(model.named_parameters()) + list(model.named_buffers())
        for name, t in items:
            if t.numel() == 0:
                out[name] = f"{tuple(t.shape)}|{t.dtype}|empty"
                continue
            flat = t.detach().contiguous().view(-1)
            raw = flat.view(torch.uint8) if flat.dtype != torch.bool else flat.to(torch.uint8)
            h = hashlib.sha256()
            # Chunk the D2H copy so a 1.6 GB fused-expert tensor does not
            # need a second full-size host buffer.
            step = 256 << 20
            for i in range(0, raw.numel(), step):
                h.update(raw[i:i + step].cpu().numpy().tobytes())
            out[name] = f"{tuple(t.shape)}|{t.dtype}|{h.hexdigest()}"
        return json.dumps(out)

    def affine_direct_load(self, weights_path: str) -> str:
        """Raw `model.load_weights` from `weights_path` (checkpoint format),
        WITHOUT vLLM's layerwise reload wrappers. Reports parameters the
        model did not report as loaded."""
        from vllm.model_executor.model_loader import get_model_loader

        mr = self.model_runner
        loader = get_model_loader(mr.load_config)
        mc = copy.copy(mr.model_config)
        mc.model = weights_path
        mc.revision = None
        model = mr.get_model()
        weights = loader.get_all_weights(mc, model)
        loaded = model.load_weights(weights) or set()
        expected = {n for n, _ in model.named_parameters()}
        missing = sorted(expected - set(loaded))
        return json.dumps({"loaded": len(loaded), "n_missing": len(missing),
                           "missing": missing[:40]})

    def affine_process_weights(self) -> str:
        """Re-run process_weights_after_loading on every quantized/fused layer
        (what a fresh load does after load_weights)."""
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

        model = self.model_runner.get_model()
        n = 0
        for layer in model.modules():
            qm = getattr(layer, "quant_method", None)
            if isinstance(qm, QuantizeMethodBase):
                qm.process_weights_after_loading(layer)
                n += 1
        return json.dumps({"processed_layers": n})
