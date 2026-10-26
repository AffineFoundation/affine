"""Rank agreement: world 8 (gloo, CPU). Rank 0 owns the optimizer and runs the capped step; every rank receives the bf16 actor
by dist.broadcast(src=0) exactly as v83_dp does at each job; all 8 hashes must match rank 0's, and a 2nd run must reproduce."""
import sys, os, json, hashlib, importlib.util
import torch, torch.distributed as dist, torch.multiprocessing as mp
SRC, STAGE, SETTINGS = sys.argv[1], sys.argv[2], sys.argv[3]
def work(rank, world, out):
    sys.path.insert(0, SRC); sys.path.insert(0, STAGE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ.update(V832_SRC=SRC, V832_STAGE=STAGE, V832_SETTINGS=SETTINGS)
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT='29611')
    dist.init_process_group('gloo', rank=rank, world_size=world)
    import test_helpers as T
    params = T.model()
    if rank == 0:
        o = T.make(params)
        for k in range(4): T.L.controlled_step(o, T.grads(k), T.inner, 2e-5, 3, 0.5, emit=lambda r: None)
    for _, p in params: dist.broadcast(p.data, src=0)
    h = hashlib.sha256(b''.join(p.detach().view(torch.int16).numpy().tobytes() for _, p in params)).hexdigest()
    hs = [None] * world; dist.all_gather_object(hs, h)
    if rank == 0: out.put(hs)
    dist.destroy_process_group()
if __name__ == '__main__':
    q = mp.get_context('spawn').SimpleQueue(); runs = []
    for _ in range(2):
        mp.spawn(work, args=(8, q), nprocs=8, join=True); runs.append(q.get())
    ok = all(len(set(r)) == 1 for r in runs) and runs[0][0] == runs[1][0]
    print(json.dumps(dict(run1=[x[:12] for x in runs[0]], run2_rank0=runs[1][0][:12], all_ranks_equal=ok)))
    print('RANKS PASS' if ok else 'RANKS FAIL'); sys.exit(0 if ok else 1)
