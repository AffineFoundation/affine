#!/usr/bin/env python3
"""Build a tiny random model with the HEAD's tokenizer (CPU test fixture; no download). usage: make_tiny.py TOKENIZER_DIR OUT_HEAD OUT_REF"""
import json, shutil, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
src, head, ref = map(Path, sys.argv[1:4])
tok = AutoTokenizer.from_pretrained(str(src), local_files_only=True)
V = max(len(tok), 248320)
cfg = Qwen2Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  max_position_embeddings=12288, tie_word_embeddings=False, eos_token_id=[248046, 248044])
for out, seed in ((head, 0), (ref, 1)):
    torch.manual_seed(seed)
    m = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    if out == ref:   # reference = head + small noise, so the KL term is nonzero
        hm = Qwen2ForCausalLM.from_pretrained(str(head), dtype=torch.bfloat16)
        with torch.no_grad():
            for (n, p), (_, q) in zip(m.named_parameters(), hm.named_parameters()): p.copy_(q + 0.01 * torch.randn_like(q.float()).to(q.dtype))
    m.save_pretrained(str(out))
    for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        if (src / f).exists(): shutil.copy(src / f, out / f)
print('tiny models', head, ref, 'vocab', V)
