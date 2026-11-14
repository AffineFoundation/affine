"""Train one made-up row of the longest length the trainer takes, on the GPUs this process sees. Writes only below $T."""

import dataclasses
import os
import time
from pathlib import Path

import torch
import transformers

from sn120_v2.gpu import train_job
from sn120_v2.params import load_params

temp, base = Path(os.environ["T"]), os.environ["BASE"]
params = load_params(os.environ["PARAMS"])
recipe = params.recipe
n = int(os.environ.get("TOKENS", recipe.train_max_context_tokens))
recipe = dataclasses.replace(recipe, steps=1, train_max_context_tokens=n)
tokenizer = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
WORDS = " the quick brown fox jumps over the lazy dog"
per_copy = len(tokenizer.encode(WORDS * 100, add_special_tokens=False)) / 100


def text(tokens: int) -> str:
    return WORDS * (int(tokens / per_copy) + 2)


rows = [{"prompt": text(max(n - recipe.train_max_output_tokens, n // 8)), "output": text(n), "advantage": 1.0}]
(row,), _ = train_job.encode(tokenizer, rows, recipe)
print(f"ROW: {len(row['prompt_ids'])} prompt tokens + {len(row['tokens'])} output tokens = {len(row['prompt_ids']) + len(row['tokens'])} (asked: {n})", flush=True)
job = {"base_dir": base, "base_hash": Path(base).name, "dtype": params.dtype, "recipe": recipe.to_dict(), "seed": 1, "rows": rows}
devices = [os.environ["DEVICE"]] if "DEVICE" in os.environ else None  # DEVICE=cpu is for a test without a GPU only
started = time.monotonic()
try:
    result = train_job.run(job, temp / "lora", devices)
    print(f"RESULT: trained. load_s={result['load_s']:.1f} step_s={result['step_s']:.1f} n_gpus={result['n_gpus']} status={result['status']}")
except Exception as exc:  # noqa: BLE001 - the class and the text of the failure are the answer
    print(f"RESULT: failed after {time.monotonic() - started:.1f} s. {type(exc).__module__}.{type(exc).__name__}: {str(exc)[:400]}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        gib = 2**30
        print(f"GPU {i} of this process: total {total / gib:.1f} GiB, torch peak in tensors {torch.cuda.max_memory_allocated(i) / gib:.1f} GiB, torch peak taken {torch.cuda.max_memory_reserved(i) / gib:.1f} GiB, free now {free / gib:.1f} GiB")
