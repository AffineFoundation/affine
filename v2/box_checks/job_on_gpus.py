"""Time one made-up training job on the GPUs this process sees, or compare the LoRA files of two such runs. Writes only below $T."""

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file

from sn120_v2.gpu import train_job
from sn120_v2.params import load_params

temp = Path(os.environ["T"])
WORDS = " the quick brown fox jumps over the lazy dog"


def timed(name: str):
    """Print how long each call of ``train_job.Replica.<name>`` takes. The call itself is not changed."""
    inner = getattr(train_job.Replica, name)

    def outer(self, *args, **kwargs):
        start = time.monotonic()
        try:
            return inner(self, *args, **kwargs)
        finally:
            what = "load" if name == "__init__" else "warm-up"
            print(f"COPY on {self.device if hasattr(self, 'device') else '?'}: {what} {time.monotonic() - start:.1f} s", flush=True)

    setattr(train_job.Replica, name, outer)


def run(name: str) -> None:
    base, params = os.environ["BASE"], load_params(os.environ["PARAMS"])
    n_rows, longest, shortest = (int(os.environ.get(key, default)) for key, default in (("ROWS", 96), ("LONGEST", 8192), ("SHORTEST", 1024)))
    recipe = dataclasses.replace(params.recipe, steps=int(os.environ.get("STEPS", 2)))
    tokenizer = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
    per_copy = len(tokenizer.encode(WORDS * 100, add_special_tokens=False)) / 100
    rows = []
    for i in range(n_rows):
        tokens = longest - (longest - shortest) * i // max(1, n_rows - 1)
        rows.append({"prompt": "Write the sentence many times.", "output": WORDS * max(1, int(tokens / per_copy)), "advantage": 1.0 if i % 2 == 0 else -1.0})
    job = {"base_dir": base, "base_hash": Path(base).name, "dtype": params.dtype, "recipe": recipe.to_dict(), "seed": 1, "rows": rows}
    devices = os.environ["DEVICES"].split(",") if "DEVICES" in os.environ else None  # DEVICES=cpu,cpu is for a test without a GPU only
    for method in ("__init__", "warm_up"):
        timed(method)
    started = time.monotonic()
    try:
        result = train_job.run(job, temp / name, devices)
    except Exception as exc:  # noqa: BLE001 - the class and the text of the failure are the answer
        print(f"RESULT {name}: failed after {time.monotonic() - started:.1f} s. {type(exc).__module__}.{type(exc).__name__}: {str(exc)[:400]}")
        return
    wall, tokens, gpus = time.monotonic() - started, result["forward_tokens"], result["n_gpus"]
    print(f"RESULT {name}: trained. n_gpus={gpus} rows={result['rows']} steps={recipe.steps} forward_tokens={tokens} wall_s={wall:.1f} load_s={result['load_s']:.1f} step_s={result['step_s']:.1f}")
    print(f"SPEED {name}: {tokens / result['step_s']:.0f} tokens per second in the steps, {tokens / result['step_s'] / gpus:.0f} per GPU")
    for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        print(f"GPU {i} of this process: torch peak taken {torch.cuda.max_memory_reserved(i) / 2**30:.1f} GiB of {torch.cuda.mem_get_info(i)[1] / 2**30:.1f} GiB")


def compare(first: str, second: str) -> None:
    one, two = (load_file(str(temp / name / "adapter_model.safetensors")) for name in (first, second))
    reports = [json.loads((temp / name / "train_report.json").read_text()) for name in (first, second)]
    same = (temp / first / "adapter_model.safetensors").read_bytes() == (temp / second / "adapter_model.safetensors").read_bytes()
    largest = max(float(tensor.abs().max()) for key, tensor in one.items() if "lora_B" in key)
    diff = max(float((one[key] - two[key]).abs().max()) for key in one)
    size = sum(float((tensor.double() ** 2).sum()) for key, tensor in one.items() if "lora_B" in key) ** 0.5
    apart = sum(float(((one[key] - two[key]).double() ** 2).sum()) for key in one) ** 0.5
    print(f"COMPARE {first} {second}: same bytes={same}; largest difference of a weight {diff:.3e} (largest weight of B {largest:.3e}); difference / size of the update {apart / size:.3e}")
    print(f"COMPARE losses: {reports[0]['losses']} against {reports[1]['losses']}")
    print(f"COMPARE gradient norms: {reports[0]['grad_norms']} against {reports[1]['grad_norms']}")


if __name__ == "__main__":
    compare(*sys.argv[2:4]) if sys.argv[1] == "compare" else run(sys.argv[2])
