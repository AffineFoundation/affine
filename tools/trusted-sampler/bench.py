"""Throughput benchmark of one generation phase under different vLLM settings."""
import json, sys, time, types, random, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def main():
    import sampler, fetch
    cfg = json.loads(sys.argv[1])
    idx = json.load(open('data/index.json')); tasks = json.load(open('data/original-math7496.tasks.json'))
    cp = sampler.find(idx, '19f21658'); d, _ = fetch.ensure(cp, '/root/ckpt-cache')
    a = types.SimpleNamespace(tp=1, gpu_mem=cfg.get('gpu_mem', 0.9), max_num_seqs=cfg.get('max_num_seqs', 1024),
                              max_batched_tokens=cfg.get('max_batched_tokens', 16384), enforce_eager=False, engine='swap', max_tokens=8192)
    extra = cfg.get('extra', {})
    e = sampler.Engine(a)
    if extra:
        from vllm import LLM
        orig = LLM.__init__
        def patched(self, *x, **k):
            k.update(extra); return orig(self, *x, **k)
        LLM.__init__ = patched
    t0 = time.time(); e.load(cp, d); tl = time.time() - t0
    rng = random.Random(1234); K = cfg.get('K', 128); m = cfg.get('m', 8)
    chosen = rng.sample(idx['pool']['indices'], K); seeds = [[rng.getrandbits(31) for _ in range(m)] for _ in chosen]
    tok = e.tokenizer; rt = idx['checkpoints'][-1]['mimo_runtime']
    P = [list(tok.apply_chat_template([dict(role='system', content=tasks[i]['data']['system_prompt']), dict(role='user', content=tasks[i]['data']['prompt'])],
          tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)) for i in chosen]
    t0 = time.time(); outs = e.generate(P, seeds, rt); tg = time.time() - t0
    n = sum(len(s) for o in outs for s in o)
    print('RESULT', json.dumps(dict(cfg=cfg, load_s=round(tl, 1), gen_s=round(tg, 1), tokens=n, tok_per_s=round(n / tg))), flush=True)
    e.close()
if __name__ == '__main__':
    main()
