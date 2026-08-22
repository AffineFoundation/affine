#!/usr/bin/env python3
"""Merge a G LoRA adapter into the base model for clean vLLM serving in SWE runs."""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, args.lora)
    m = m.merge_and_unload()
    m.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"MERGED -> {args.out}")


if __name__ == "__main__":
    main()
