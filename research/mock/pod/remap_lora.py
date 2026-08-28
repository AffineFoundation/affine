#!/usr/bin/env python3
"""Remap PEFT LoRA keys to vLLM's module naming for *ForConditionalGeneration.

HF AutoModelForCausalLM trains the text model directly, so PEFT saves keys as
  base_model.model.model.layers.N...
but vLLM serves the checkpoint via its ConditionalGeneration wrapper, whose
text tower lives under `language_model`, so it silently matches NOTHING and
the adapter becomes a no-op (this actually happened: judge v0, 2026-08-22).

Usage: remap_lora.py <in_dir> <out_dir> [--prefix language_model.model.]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import safetensors.torch as st

OLD = "base_model.model.model."


def remap(in_dir, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    sd = st.load_file(os.path.join(in_dir, "adapter_model.safetensors"))
    new = {}
    n_hit = 0
    for k, v in sd.items():
        if k.startswith(OLD):
            k = "base_model.model." + prefix + k[len(OLD):]
            n_hit += 1
        new[k] = v
    st.save_file(new, os.path.join(out_dir, "adapter_model.safetensors"))
    for f in os.listdir(in_dir):
        if f not in ("adapter_model.safetensors",):
            src = os.path.join(in_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out_dir, f))
    return n_hit, len(sd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--prefix", default="language_model.model.")
    args = ap.parse_args()
    hit, tot = remap(args.in_dir, args.out_dir, args.prefix)
    print(f"remapped {hit}/{tot} keys with prefix {args.prefix!r} "
          f"-> {args.out_dir}")
    if not hit:
        raise SystemExit("no keys matched -- wrong input format?")
