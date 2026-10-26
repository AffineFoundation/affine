#!/usr/bin/env python3
"""Merge a small LoRA adapter into its base model by direct shard surgery.

Runs on the fastbench box (CPU + disk only, never touches the GPUs), so it
can overlap a running bench. Avoids peft entirely: reads the adapter's
A/B pairs, applies  W += scale * (B @ A)  to the matching tensors while
streaming the base checkpoint shard by shard, and writes a complete,
vLLM-servable model dir.

    python merge_lora.py --adapter /root/kings/<id> --out /root/models/<id>

The base repo is taken from adapter_config.json (base_model_name_or_path)
and resolved through the local HF cache (downloads only if absent).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def resolve_base(repo: str) -> Path:
    if Path(repo).is_dir():
        return Path(repo)
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, local_files_only=True))
    except Exception:
        print(f"[merge] base {repo} not cached; downloading", flush=True)
        return Path(snapshot_download(repo))


def load_adapter(adapter_dir: Path) -> tuple[dict[str, dict], float]:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = cfg["r"]
    alpha = cfg.get("lora_alpha", r)
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    if cfg.get("use_dora"):
        sys.exit("[merge] DoRA adapters are not supported by this script")
    weights = load_file(adapter_dir / "adapter_model.safetensors")
    pairs: dict[str, dict] = {}
    for key, tensor in weights.items():
        for tag in (".lora_A.weight", ".lora_B.weight"):
            if key.endswith(tag):
                # base_model.model.<module>.lora_A.weight -> <module>.weight
                module = key[: -len(tag)]
                for prefix in ("base_model.model.", "base_model."):
                    if module.startswith(prefix):
                        module = module[len(prefix):]
                        break
                pairs.setdefault(module + ".weight", {})[tag[6:7]] = tensor
                break
        else:
            sys.exit(f"[merge] unrecognized adapter key: {key}")
    bad = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if bad:
        sys.exit(f"[merge] incomplete A/B pairs: {bad[:3]}")
    return pairs, scale


def remap_to_base(pairs: dict[str, dict], base_keys: set[str]) -> dict:
    """Match adapter module names to base tensor names.

    The training wrapper and the checkpoint disagree on nesting (observed:
    adapter 'language_model.model.layers.N...' vs checkpoint
    'model.language_model.layers.N...'), so try a few known rewrites per key
    and demand every one resolves uniquely."""
    def candidates(key: str):
        yield key
        if key.startswith("language_model.model."):
            yield "model.language_model." + key[len("language_model.model."):]
        if key.startswith("model."):
            yield key[len("model."):]
        yield "model." + key

    out: dict[str, dict] = {}
    unresolved = []
    for key, ab in pairs.items():
        hit = next((c for c in candidates(key) if c in base_keys), None)
        if hit is None:
            unresolved.append(key)
        else:
            out[hit] = ab
    if unresolved:
        sys.exit(f"[merge] cannot map adapter keys to base: "
                 f"{unresolved[:3]} (of {len(unresolved)})")
    if len(out) != len(pairs):
        sys.exit("[merge] adapter keys collided after remap")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    adapter_cfg = json.loads(
        (args.adapter / "adapter_config.json").read_text())
    base_repo = adapter_cfg["base_model_name_or_path"]
    base = resolve_base(base_repo)
    pairs, scale = load_adapter(args.adapter)
    print(f"[merge] base={base_repo} ({base}) targets={len(pairs)} "
          f"scale={scale}", flush=True)

    out = args.out
    tmp = out.parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_of = index["weight_map"]
        shards = sorted(set(shard_of.values()))
        pairs = remap_to_base(pairs, set(shard_of))
    else:
        shards = ["model.safetensors"]
        from safetensors import safe_open
        with safe_open(base / shards[0], framework="pt") as f:
            pairs = remap_to_base(pairs, set(f.keys()))

    applied = 0
    for shard in shards:
        tensors = load_file(base / shard)
        touched = False
        for key in list(tensors):
            if key not in pairs:
                continue
            w = tensors[key]
            ab = pairs[key]
            delta = (ab["B"].to(torch.float32)
                     @ ab["A"].to(torch.float32)) * scale
            if delta.shape != w.shape:
                sys.exit(f"[merge] shape mismatch {key}: "
                         f"{tuple(delta.shape)} vs {tuple(w.shape)}")
            tensors[key] = (w.to(torch.float32) + delta).to(w.dtype)
            applied += 1
            touched = True
        save_file(tensors, tmp / shard,
                  metadata={"format": "pt"})
        print(f"[merge] {shard}: {'merged' if touched else 'copied'}",
              flush=True)

    if applied != len(pairs):
        sys.exit(f"[merge] applied {applied} of {len(pairs)} deltas")

    for item in base.iterdir():
        if item.name.endswith(".safetensors") or item.name.startswith("."):
            continue
        if item.is_file():
            shutil.copy(item, tmp / item.name)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(f"[merge] OK: {applied} tensors merged -> {out}", flush=True)


if __name__ == "__main__":
    main()
