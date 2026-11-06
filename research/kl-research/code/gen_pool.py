# Teacher sampling with top-20 raw logprobs. Usage: python gen_pool.py SHARD NSHARD
import sys, json, pickle, time, os, numpy as np
from vllm import LLM, SamplingParams
shard, nsh = int(sys.argv[1]), int(sys.argv[2])
TEACHER = os.environ.get("TEACHER", "Qwen/Qwen3-8B")
OUT = "/root/kl/data/pool"; os.makedirs(OUT, exist_ok=True)
pool = json.load(open("/root/kl/data/pool_prompts.json"))
held = json.load(open("/root/kl/data/heldout_prompts.json"))
items = []
for k, v in pool.items():
    for i, p in enumerate(v): items.append((f"{k}:{i}", k, p))
for k, v in held.items():
    for i, p in enumerate(v): items.append((f"held_{k}:{i}", "held_" + k, p))
items = items[shard::nsh]
llm = LLM(TEACHER, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=6144, logprobs_mode="raw_logprobs",
          enable_prefix_caching=True, max_num_seqs=512, seed=shard)
tok = llm.get_tokenizer()
def chat(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
groups = {}
for it in items:
    s = it[1]
    sp = dict(temperature=1.0, top_p=1.0, max_tokens=1024, logprobs=20)
    if s == "adv_long": sp["max_tokens"] = 4096
    if s == "adv_hot": sp["temperature"] = 1.5
    if s == "adv_greedy": sp["temperature"] = 0.0
    groups.setdefault(json.dumps(sp, sort_keys=True), []).append(it)
recs, ntok, t0 = [], 0, time.time()
for spk, its in groups.items():
    sp = SamplingParams(**json.loads(spk))
    prompts, keep = [], []
    for it in its:
        c = chat(it[2]); ids = tok(c, add_special_tokens=False).input_ids
        if len(ids) > 1536: continue
        prompts.append({"prompt_token_ids": ids}); keep.append((it, ids))
    for b in range(0, len(prompts), 4000):
        outs = llm.generate(prompts[b:b+4000], sp)
        for (it, pids), o in zip(keep[b:b+4000], outs):
            c = o.outputs[0]; n = len(c.token_ids)
            if n == 0: continue
            lp = np.zeros(n, np.float32); ti = np.zeros((n, 20), np.int32); tl = np.full((n, 20), -1e4, np.float32)
            for t, (tid, d) in enumerate(zip(c.token_ids, c.logprobs)):
                lp[t] = d[tid].logprob
                j = 0
                for k2, v in d.items():
                    if v.rank is not None and 1 <= v.rank <= 20:
                        ti[t, v.rank - 1] = k2; tl[t, v.rank - 1] = v.logprob
            recs.append(dict(id=it[0], src=it[1], prompt=it[2], pids=np.array(pids, np.int32), rids=np.array(c.token_ids, np.int32),
                             lp=lp, ti=ti, tl=tl.astype(np.float16), fin=c.finish_reason))
            ntok += n
        print(f"shard{shard} {spk} {b+len(outs)}/{len(prompts)} tok={ntok} tps={ntok/(time.time()-t0):.0f}", flush=True)
pickle.dump(recs, open(f"{OUT}/shard{shard}.pkl", "wb"), protocol=5)
json.dump(dict(ntok=ntok, sec=time.time() - t0, n=len(recs)), open(f"{OUT}/shard{shard}.stats.json", "w"))
print("DONE", shard, ntok, time.time() - t0, flush=True)
