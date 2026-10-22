"""Interference proxy: the evaluator's decode path (HF Qwen3_5ForConditionalGeneration, eager, bf16, single stream,
KV/recurrent cache, logits_to_keep=1) timed for N tokens. Run alone vs while vLLM is loaded to measure slowdown."""
import sys, time, json, torch
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
ck = sys.argv[1]; N = int(sys.argv[2]); label = sys.argv[3]; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
tok = AutoTokenizer.from_pretrained(ck)
m = Qwen3_5ForConditionalGeneration.from_pretrained(ck, dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
ids = tok.apply_chat_template([dict(role='user', content='Prove that the sum of the first n odd numbers is n^2. Think step by step.')],
                              tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
if isinstance(ids, dict): ids = ids['input_ids']
g = torch.Generator(device='cuda').manual_seed(0)
for r in range(reps):
    t0 = time.time(); cache = None; cur = ids; n = 0
    with torch.inference_mode():
        for _ in range(N):
            out = m(torch.tensor([cur], device='cuda'), use_cache=True, past_key_values=cache, logits_to_keep=1)
            cache = out.past_key_values; p = torch.softmax(out.logits[0, -1].float(), -1)
            t = int(torch.multinomial(p, 1, generator=g)); cur = [t]; n += 1
    torch.cuda.synchronize(); dt = time.time() - t0
    print(json.dumps(dict(label=label, rep=r, tokens=n, seconds=round(dt, 2), tok_s=round(n / dt, 2))), flush=True)
