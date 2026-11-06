# Pinned judge (Qwen3-8B, non-thinking, fixed prompt) selects ~20% of tokens per rollout at segment level (span cap 32 tokens).
# Judge sees only prompt + response text; never student logprobs. Usage: judge_masks.py SHARD NSHARD FRAC
import zlib, sys, pickle, glob, json, re, random, numpy as np, os
from vllm import LLM, SamplingParams
shard, nsh, frac = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
HONEST = ["alpaca", "ultrachat", "gsm8k", "math", "numina", "pycode", "codecontests", "mmlu_aux", "science", "multiling", "writing"]
recs = []
for f in sorted(glob.glob("/root/kl/data/pool/shard*.pkl")):
    for r in pickle.load(open(f, "rb")):
        if r["src"] in HONEST and (zlib.crc32(r["id"].encode()) % 1000) < frac * 1000 and int(r["id"].split(":")[1]) % nsh == shard:
            recs.append(dict(id=r["id"], prompt=r["prompt"], rids=r["rids"]))
llm = LLM("Qwen/Qwen3-8B", dtype="bfloat16", gpu_memory_utilization=0.42, max_model_len=8192, enable_prefix_caching=True, seed=0)
tok = llm.get_tokenizer()
SPAN = 32
def segments(rids):
    segs, cur = [], []
    pieces = [tok.decode([t]) for t in rids]
    for k, (t, p) in enumerate(zip(rids, pieces)):
        cur.append(k)
        end = ("\n" in p) or (p.rstrip().endswith((".", "?", "!", ";", ":")) and k + 1 < len(rids) and pieces[k + 1].startswith(" ")) or len(cur) >= SPAN
        if end: segs.append(cur); cur = []
    if cur: segs.append(cur)
    return segs, pieces
prompts, meta = [], []
for r in recs:
    segs, pieces = segments(r["rids"].tolist())
    n = len(r["rids"]); k = max(1, round(0.2 * len(segs)))
    body = "\n".join(f"[{j+1}] " + "".join(pieces[t] for t in s).replace("\n", " ").strip() for j, s in enumerate(segs))
    q = ("You are selecting which parts of an answer a student model most needs to learn. Below is a task and an answer split into numbered segments. "
         "Pick the segments that carry the essential content: key reasoning steps, calculations, decisions, code logic, specific facts, and the final answer. "
         "Do not pick filler, restatements of the question, pleasantries, or pure formatting. "
         f"List about {k} segment numbers, most important first, as comma-separated integers only.\n\nTask:\n{r['prompt'][:3000]}\n\nAnswer segments:\n{body}")
    ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids, "input_ids"): ids = ids["input_ids"]
    if len(ids) > 7900: continue
    prompts.append({"prompt_token_ids": list(ids)}); meta.append((r["id"], segs, n))
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=200))
masks = {}
for (rid, segs, n), o in zip(meta, outs):
    nums = [int(x) - 1 for x in re.findall(r"\d+", o.outputs[0].text)]
    m = np.zeros(n, bool); budget = 0.2 * n; seen = set()
    for j in nums:
        if j in seen or j < 0 or j >= len(segs): continue
        seen.add(j)
        if m.sum() + len(segs[j]) > budget * 1.25 and m.sum() > 0: break
        m[segs[j]] = True
        if m.sum() >= budget: break
    masks[rid] = m
pickle.dump(masks, open(f"/root/kl/data/judge_masks.{shard}.pkl", "wb"))
print("DONE", len(masks), np.mean([m.mean() for m in masks.values()]), flush=True)
import psutil
for c in psutil.Process().children(recursive=True):
    try: c.kill()
    except Exception: pass
os._exit(0)
