"""Parameter-level check: swapped engine (A->B) vs cold engine (B): every param/buffer must be bitwise equal."""
import json, sys, types, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def sums(worker):
    import torch, hashlib
    out = {}
    m = worker.model_runner.model
    for n, p in list(m.named_parameters()) + list(m.named_buffers()):
        t = p.detach().reshape(-1).contiguous().view(torch.uint8) if p.dtype != torch.bool else p.detach().to(torch.uint8)
        out[n] = hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]
    return out
def main():
    import sampler, fetch
    idx = json.load(open('data/index.json'))
    b = sampler.find(idx, sys.argv[1]); c = sampler.find(idx, sys.argv[2])
    a = types.SimpleNamespace(tp=1, gpu_mem=0.5, max_num_seqs=64, max_batched_tokens=16384, enforce_eager=False, engine='swap', max_tokens=256)
    res = {}
    for name, seq in (('cold', [c]), ('swap', [b, c])):
        e = sampler.Engine(a)
        for cp in seq:
            d, _ = fetch.ensure(cp, '/root/ckpt-cache'); e.load(cp, d)
        res[name] = e.llm.collective_rpc(sums)[0]; e.close()
    diff = [k for k in res['cold'] if res['cold'][k] != res['swap'].get(k)]
    print('RESULT', json.dumps(dict(tensors=len(res['cold']), differing=len(diff), examples=diff[:10])), flush=True)
if __name__ == '__main__':
    main()
