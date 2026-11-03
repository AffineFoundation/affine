"""RA/20261102-12k-everywhere: per-rank GPU memory of one DP bin (dp_token_budget 32768, packed varlen, fast kernels, support renorm,
gradients) for today's 8K-shaped rows vs 12K rows, with the v8.3.3m host modules on ONE H200 of the trainer pod (same per-rank work
as each of the 8 DP ranks: full bf16 actor + one packed bin forward/backward). Synthetic token ids (no data). usage: memtest_pod.py MODEL_DIR"""
import json, os, sys, time, random
from pathlib import Path
W = Path("/workspace/affine-mimo-trainer-prod"); SRC = W/"source-final-v823u"; HOST = Path(os.environ.get("HOST", W/"host-v833m"))
os.chdir(SRC); sys.path.insert(0, str(SRC)); sys.path.insert(0, str(HOST))
import importlib.util
_s = importlib.util.spec_from_file_location("mimo_host_trainer_v83", HOST/"mimo_host_trainer_v83.py")
host = importlib.util.module_from_spec(_s); sys.modules["mimo_host_trainer_v83"] = host; _s.loader.exec_module(host)
print(json.dumps(host.install(host.checked(str(HOST/"TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json"))))[:200], flush=True)
import torch
from subnet.mimo_runtime import MimoRuntime
spec = json.loads(Path("/workspace/rec/rebuild-spec.json").read_bytes())
model_dir = Path(sys.argv[1])
files = {p.name: __import__("subnet.mimo_runtime", fromlist=["file_hash"]).file_hash(p) for p in sorted(model_dir.iterdir()) if p.is_file() and not p.name.startswith(".")}
from subnet.mimo_grpo.data import digest
desc = dict(id=digest(files), files=files)
rt = MimoRuntime.load(model_dir, desc, spec["mimo_runtime"], device="cuda", trainable=True)
FK = sys.modules["v83_fast_kernels"]
FK.install(rt.model)
rt.model.train(False)
support = dict(temperature=1.0, top_k=20, top_p=0.95)
V = rt.vocab_size; rng = random.Random(0)
def toks(n): return [rng.randrange(1000, 200000) for _ in range(n)]
cases = {"A_today_4x8192": [(toks(400), toks(7792)) for _ in range(4)],
         "B_12k_2x16384": [(toks(4096), toks(12288)) for _ in range(2)],
         "C_12k_1x16384+pad": [(toks(4096), toks(12288)), (toks(4096), toks(12288))][:1] + [(toks(400), toks(7792)) for _ in range(2)],
         "D_12k_2x12688": [(toks(400), toks(12288)) for _ in range(2)] + [(toks(200), toks(7192))]}
out = {}
for name, items in cases.items():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    res = FK.packed_log_probs(rt, items, gradients=True, logit_chunk=4096, checkpoint_head=True, support=support)
    loss = sum(r[0].sum() for r in res) * 1e-6
    loss.backward()
    torch.cuda.synchronize()
    out[name] = dict(tokens=sum(len(p) + len(o) for p, o in items), rows=len(items), max_row=max(len(p) + len(o) for p, o in items),
                     peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                     seconds=round(time.time() - t, 1))
    for p in rt.model.parameters(): p.grad = None
    del res, loss
    print(json.dumps({name: out[name]}), flush=True)
out["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
print(json.dumps(dict(done=True, **out)), flush=True)
