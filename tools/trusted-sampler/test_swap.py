"""Check that the in-place weight swap equals a cold load (greedy, same prompts)."""
import json, sys, types, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampler, fetch
def main():
    idx = json.load(open('data/index.json')); tasks = json.load(open('data/original-math7496.tasks.json'))
    b = sampler.find(idx, sys.argv[1]); c = sampler.find(idx, sys.argv[2])
    a = types.SimpleNamespace(tp=1, gpu_mem=0.85, max_num_seqs=64, max_batched_tokens=16384, enforce_eager=False, engine='swap', max_tokens=256)
    rt = dict(idx['checkpoints'][-1]['mimo_runtime']); rt.update(temperature=0.0, top_p=1.0, top_k=-1)
    def prompts(tok):
        return [list(tok.apply_chat_template([dict(role='system', content=tasks[i]['data']['system_prompt']), dict(role='user', content=tasks[i]['data']['prompt'])],
                tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)) for i in range(0, 1600, 100)]
    res = {}
    for name, seq in (('cold', [c]), ('swap', [b, c])):
        e = sampler.Engine(a)
        for cp in seq:
            d, _ = fetch.ensure(cp, '/root/ckpt-cache'); how = e.load(cp, d)
        P = prompts(e.tokenizer)
        res[name] = [o[0] for o in e.generate(P, [[0]] * len(P), rt)]; print(name, how, flush=True); e.close()
    same = sum(x == y for x, y in zip(res['cold'], res['swap']))
    pref = [next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y))) for x, y in zip(res['cold'], res['swap'])]
    print(json.dumps(dict(identical=same, of=len(pref), common_prefix_tokens=pref)))

if __name__ == '__main__':
    main()