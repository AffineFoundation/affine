"""Run the prod train_job (sn120_v2.gpu.train_job) on a benchmark batch with chosen caps; print peak memory and time."""
import argparse, json, sys, time, hashlib, os
from pathlib import Path
import torch
sys.path.insert(0, os.environ.get("SN120_SRC", "/root/t64/v2"))  # sn120-standalone-source-path
from sn120_v2.params import Recipe
from sn120_v2.gpu.batch import training_rows
from sn120_v2.gpu import train_job

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", type=int, default=32768)
ap.add_argument("--ctx", type=int, default=36864)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=float, default=32.0)
ap.add_argument("--targets", default="")
ap.add_argument("--steps", type=int, default=1)
ap.add_argument("--minibatches", type=int, default=1)
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--base", default="/root/t64/model")
ap.add_argument("--tag", required=True)
ap.add_argument("--rows", type=int, default=0, help="keep only the first N groups")
a = ap.parse_args()

prod = json.load(open(os.environ.get("SN120_PARAMS", "/root/t64/v2/tools/dedup-deploy/new-params.json")))["recipe"]
rd = dict(prod)
rd.update(train_max_output_tokens=a.out, train_max_context_tokens=a.ctx, rank=a.rank, alpha=a.alpha, steps=a.steps, minibatches=a.minibatches)
if a.targets:
    rd["target_modules"] = a.targets.split(",")
recipe = Recipe.from_dict(rd)
groups = json.load(open(a.data))
if a.rows:
    groups = groups[: a.rows]
rows = training_rows([(p, [(o, r) for o, r in ro]) for p, ro in groups], recipe.group_size)
job = {"base_dir": a.base, "base_hash": "bench", "dtype": "bfloat16", "recipe": recipe.to_dict(), "seed": 1234567, "rows": rows}
out = Path(f"/root/t64/out/{a.tag}")
out.mkdir(parents=True, exist_ok=True)
devices = [f"cuda:{i}" for i in range(a.gpus)]
for d in devices:
    torch.ones(1, device=d)
    torch.cuda.reset_peak_memory_stats(d)
t0 = time.monotonic()
status, err = "ok", None
try:
    rep = train_job.train(job, out / "lora", devices)
except torch.OutOfMemoryError as e:
    status, err, rep = "OOM", str(e)[:400], {}
except Exception as e:  # noqa
    status, err, rep = "error", f"{type(e).__name__}: {str(e)[:400]}", {}
wall = time.monotonic() - t0
mem = {d: {"max_alloc_gib": round(torch.cuda.max_memory_allocated(d) / 2**30, 2), "max_reserved_gib": round(torch.cuda.max_memory_reserved(d) / 2**30, 2)} for d in devices}
lora_sha = None
f = out / "lora" / "adapter_model.safetensors"
if status == "ok" and f.exists():
    lora_sha = hashlib.sha256(f.read_bytes()).hexdigest()
res = {"tag": a.tag, "status": status, "err": err, "out": a.out, "ctx": a.ctx, "rank": a.rank, "targets": rd["target_modules"], "steps": a.steps,
       "gpus": a.gpus, "wall_s": round(wall, 1), "mem": mem, "lora_sha": lora_sha, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "report": {k: rep.get(k) for k in ("load_s", "step_s", "rows", "forward_tokens", "tokens", "truncated_rows", "dropped_rows", "loss", "step_norm", "step_norm_before_cap")}}
print("RESULT " + json.dumps(res), flush=True)
with open("/root/t64/results.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")
