"""The GPU part of training: one GRPO LoRA from prepared rows.

Run as ``python -m sn120_v2.gpu.train_job <job_dir>``. The parent
(``trainer.py``) writes ``job.json`` and sets ``CUDA_VISIBLE_DEVICES``. This
process writes the LoRA files to ``<job_dir>/lora`` and ``report.json``.

The step, for each rollout (one sampled answer) with advantage A:

  q      the head's probability of each sampled token, at the miner's sampling
         temperature, renormalised over the tokens the head could sample
         (top-k, then top-p). A token outside that set does not train.
  ratio  q_now / q_old. q_old is q before the first update.
  loss   PPO with the dual clip, token mean per rollout, mean over rollouts.

Only the LoRA matrices train. They are fp32; the base stays frozen. The
optimizer is a fresh Adam per LoRA (betas 0.9 and 0.999, eps 1e-8, no weight
decay). One step is one pass over the whole batch and one Adam update.
"""

from __future__ import annotations

import functools
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

try:  # Triton comes with the GPU install. A machine without it runs no Triton kernel, so it needs no lock.
    from triton.runtime.autotuner import Autotuner
except ImportError:
    Autotuner = None

from ..canonical import canonical_json
from ..errors import BatchRejected, InfraError
from ..params import Recipe
from .batch import END_OF_TURN, derail_keep, fit_tokens

LOGIT_CHUNK = 2048  # tokens per pass through the output layer: a memory knob, with no effect on the result
OPTIMIZER = "adam, fresh per LoRA, betas 0.9/0.999, eps 1e-8, weight decay 0"
# GPU memory that torch never takes. Triton loads a kernel, and the CUDA libraries make their handles, with memory
# of the CUDA driver. torch keeps every block it ever took, so after a long row the card can be full of blocks that
# torch does not use, and such a load fails with "Triton Error [CUDA]: out of memory". With this part left free it cannot.
DRIVER_RESERVE_BYTES = 8 * 2**30
WARM_UP_TOKENS = 2048  # the most tokens of the warm-up row: one length class of FLA's l2norm kernels, about 9 GiB at the head's size


def torch_share(total_bytes: int) -> float:
    """The share of a GPU's memory that torch may take: all but the driver's part, and never less than 15/16."""
    return 1.0 - min(DRIVER_RESERVE_BYTES, total_bytes // 16) / total_bytes


def leave_room_for_the_driver(device: str) -> None:
    """Cap torch's memory on this GPU. At the cap torch frees its unused blocks, and only then fails, with its own text."""
    if device != "cpu":
        torch.cuda.set_per_process_memory_fraction(torch_share(torch.cuda.get_device_properties(device).total_memory), device)


def one_thread_in_the_tuner(tuner: type | None = Autotuner) -> None:
    """Let one thread at a time into Triton's kernel tuner. Call it once, before the first kernel.

    The tuner keeps the arguments of the call in the tuner object and clears
    them when the call ends. The job runs one thread per GPU: without the
    lock a second thread in the same tuner finds no arguments (a TypeError)
    or the tensors of the other GPU. A kernel only starts inside the lock; it
    runs on its GPU after the lock is free, so the GPUs still work side by side.
    """
    if tuner is None or getattr(tuner.run, "one_thread", False):
        return
    lock, run = threading.RLock(), tuner.run

    @functools.wraps(run)
    def locked(self, *args, **kwargs):
        with lock:
            return run(self, *args, **kwargs)

    locked.one_thread = True
    tuner.run = locked


class LoraLinear(torch.nn.Module):
    """A frozen linear layer plus ``scale * B @ A``. B starts at zero, so the first forward equals the head."""

    def __init__(self, base: torch.nn.Linear, a: torch.Tensor, scale: float):
        super().__init__()
        self.base, self.scale = base, scale
        self.lora_A = torch.nn.Parameter(a.to(base.weight.device))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, a.shape[0], device=base.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + low * self.scale


def add_lora(model: torch.nn.Module, recipe: Recipe, seed: int) -> dict[str, LoraLinear]:
    """Wrap every target projection. The same seed gives the same A matrices on every replica."""
    generator = torch.Generator().manual_seed(seed)
    layers = {}
    for path, module in list(model.named_modules()):
        for name, child in list(module.named_children()):
            if name in recipe.target_modules and isinstance(child, torch.nn.Linear):
                bound = 1.0 / math.sqrt(child.in_features)
                a = (torch.rand(recipe.rank, child.in_features, generator=generator) * 2 - 1) * bound
                layers[f"{path}.{name}"] = LoraLinear(child, a, recipe.alpha / recipe.rank)
                setattr(module, name, layers[f"{path}.{name}"])
    if not layers:
        raise InfraError("the model has none of the recipe's target modules")
    return layers


def load_head(base_dir: str, dtype: str) -> torch.nn.Module:
    """The head in the class its own config names, so every module path equals a tensor name of the weight files."""
    config = json.loads((Path(base_dir) / "config.json").read_text(encoding="utf-8"))
    cls = getattr(transformers, str((config.get("architectures") or [""])[0]), None)
    if cls is None:
        raise InfraError("this transformers version does not have the model class of the head")
    return cls.from_pretrained(base_dir, dtype=getattr(torch, dtype), local_files_only=True)


def check_names(base_dir: str, layers: Sequence[str]) -> None:
    """The fold adds a LoRA to the tensor ``<path>.weight``. A LoRA with another name could never be folded."""
    stored: set[str] = set()
    for shard in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            stored.update(handle.keys())
    missing = [path for path in layers if f"{path}.weight" not in stored]
    if missing:
        raise InfraError(f"{len(missing)} LoRA layers match no tensor of the head's weight files, for example {missing[0]}")


def token_scores(logits: torch.Tensor, targets: torch.Tensor, recipe: Recipe) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per token: plain log-probability, support log-probability ``log q``, and whether it is inside the support."""
    logits = logits.float()
    picked = logits.gather(-1, targets[:, None])[:, 0]
    top, index = logits.topk(min(recipe.support_top_k, logits.shape[-1]), dim=-1)
    top = top / recipe.sample_temperature
    prob = top.softmax(-1)
    keep = prob.cumsum(-1) - prob < recipe.support_top_p
    log_q = picked / recipe.sample_temperature - top.masked_fill(~keep, -math.inf).logsumexp(-1)
    inside = ((index == targets[:, None]) & keep).any(-1)
    return picked - logits.logsumexp(-1), log_q, inside


def ppo_loss(log_q: torch.Tensor, old: torch.Tensor, advantage: float, recipe: Recipe) -> torch.Tensor:
    ratio = (log_q - old).exp()
    loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1 - recipe.ppo_clip, 1 + recipe.ppo_clip))
    if advantage < 0:  # dual clip: a below-average rollout cannot push without limit
        loss = loss.clamp(max=-advantage * recipe.dual_clip)
    return loss


class Replica:
    """One copy of the head with LoRA layers, on one device."""

    def __init__(self, base_dir: str, device: str, dtype: str, recipe: Recipe, seed: int):
        leave_room_for_the_driver(device)
        self.model, self.device, self.recipe = load_head(base_dir, dtype).to(device).eval().requires_grad_(False), device, recipe
        self.layers = add_lora(self.model, recipe, seed)
        self.weights = [w for layer in self.layers.values() for w in (layer.lora_A, layer.lora_B)]

    def warm_up(self, ids: Sequence[int]) -> None:
        """One short row forward and backward through the layers. It leaves no gradient, changes no weight and draws no random number.

        A Triton kernel is tuned, compiled and loaded at its first call, and
        the CUDA libraries make their handles at their first use. Without
        this pass that happens in the first backward of the longest row: the
        memory peak of the whole job. Here the GPU holds only the head.
        """
        hidden = self.model.base_model(input_ids=torch.tensor([list(ids)], device=self.device)).last_hidden_state
        hidden.float().square().sum().backward()
        for weight in self.weights:
            weight.grad = None

    def train_rows(self, rows: Sequence[dict], n_rows: int) -> tuple[float, int]:
        """Add the gradient of these rows. Returns their share of the loss and the tokens that trained."""
        head, loss, trained = self.model.get_output_embeddings(), 0.0, 0
        for row in rows:
            targets = torch.tensor(row["tokens"], device=self.device)
            ids = torch.tensor([row["prompt_ids"] + row["tokens"]], device=self.device)
            hidden = self.model.base_model(input_ids=ids).last_hidden_state[0, len(row["prompt_ids"]) - 1 : -1]
            leaf = hidden.detach().requires_grad_(True)  # the output layer runs in chunks on this copy
            chunks = range(0, len(targets), LOGIT_CHUNK)
            if "old" not in row:  # first pass: the head itself is the old policy
                with torch.no_grad():
                    parts = [token_scores(head(leaf[i : i + LOGIT_CHUNK]), targets[i : i + LOGIT_CHUNK], self.recipe) for i in chunks]
                log_p, row["old"], mask = (torch.cat(part) for part in zip(*parts))
                if row["capped"] and row["advantage"] < 0:
                    recipe = self.recipe
                    mask[derail_keep(log_p.tolist(), recipe.derail_window, recipe.derail_threshold, recipe.derail_tail) :] = False
                row["mask"], row["count"] = mask, int(mask.sum())
            if not row["count"]:
                continue
            for i in chunks:
                _, log_q, inside = token_scores(head(leaf[i : i + LOGIT_CHUNK]), targets[i : i + LOGIT_CHUNK], self.recipe)
                token_loss = ppo_loss(log_q, row["old"][i : i + LOGIT_CHUNK], row["advantage"], self.recipe)
                part = (token_loss * (row["mask"][i : i + LOGIT_CHUNK] & inside)).sum() / (row["count"] * n_rows)
                part.backward()
                loss += float(part.detach())
            hidden.backward(leaf.grad)
            trained += row["count"]
        return loss, trained


def encode(tokenizer, rows: Sequence[dict], recipe: Recipe) -> tuple[list[dict], int]:
    """Token ids of every row that trains, and the count of dropped rows. The prompt is one user message with the generation prompt."""
    eos = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if not isinstance(eos, int) or eos < 0 or eos == tokenizer.unk_token_id:
        raise InfraError(f"the tokenizer has no {END_OF_TURN} token")
    prompts: dict[str, list[int]] = {}
    out, dropped = [], 0
    for row in rows:
        if row["prompt"] not in prompts:
            text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], add_generation_prompt=True, tokenize=False)
            prompts[row["prompt"]] = tokenizer.encode(text, add_special_tokens=False)
        prompt_ids = prompts[row["prompt"]]
        output = tokenizer.encode(row["output"], add_special_tokens=False)
        tokens, capped, cut = fit_tokens(len(prompt_ids), output, eos, recipe.train_max_output_tokens, recipe.train_max_context_tokens)
        if not tokens:  # no output token is left inside the training caps
            dropped += 1
            continue
        out.append({"prompt_ids": prompt_ids, "tokens": tokens, "capped": capped, "truncated": cut, "advantage": row["advantage"]})
    return out, dropped


def run(job: dict, out_dir: Path, devices: Sequence[str] | None = None) -> dict:
    recipe, started = Recipe.from_dict(job["recipe"]), time.monotonic()
    torch.manual_seed(job["seed"])
    rows, dropped = encode(transformers.AutoTokenizer.from_pretrained(job["base_dir"], local_files_only=True), job["rows"], recipe)
    active = sorted((row for row in rows if row["advantage"] != 0), key=lambda row: -len(row["tokens"]))
    if not active:
        raise BatchRejected("no rollout is left with an advantage other than zero: the batch gives no gradient")
    devices = list(devices or [f"cuda:{i}" for i in range(torch.cuda.device_count())])[: len(active)]
    if not devices:  # never train on the CPU by accident: it would run for hours and then pass the lease
        raise InfraError("the training job sees no GPU")
    one_thread_in_the_tuner()
    replicas = [Replica(job["base_dir"], device, job["dtype"], recipe, job["seed"]) for device in devices]
    first = replicas[0]
    check_names(job["base_dir"], list(first.layers))
    for replica in replicas:  # one GPU after another, before any long row: see Replica.warm_up
        replica.warm_up((active[0]["prompt_ids"] + active[0]["tokens"])[:WARM_UP_TOKENS])
    optimizer = torch.optim.Adam(first.weights, lr=recipe.lora_lr)
    loaded = time.monotonic()
    losses, grad_norms, trained = [], [], 0
    with ThreadPoolExecutor(len(replicas)) as pool:
        for _ in range(recipe.steps):
            shares = [active[i :: len(replicas)] for i in range(len(replicas))]
            results = list(pool.map(lambda pair: pair[0].train_rows(pair[1], len(rows)), zip(replicas, shares)))
            for other in replicas[1:]:
                for mine, theirs in zip(first.weights, other.weights):
                    if theirs.grad is not None:
                        mine.grad = theirs.grad.to(mine.device) if mine.grad is None else mine.grad + theirs.grad.to(mine.device)
                    theirs.grad = None
            norm = float(torch.nn.utils.clip_grad_norm_(first.weights, recipe.max_grad_norm))
            if not math.isfinite(norm):
                raise BatchRejected("training gave a gradient that is not finite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for other in replicas[1:]:
                for mine, theirs in zip(first.weights, other.weights):
                    theirs.data.copy_(mine.data)
            losses.append(sum(loss for loss, _ in results))
            grad_norms.append(norm)
            trained = sum(count for _, count in results)
    facts = {"losses": losses, "grad_norms": grad_norms, "trained_tokens": trained, "rows": len(rows)}
    # Rows cut to a training token cap, and rows with no output token left inside the caps.
    facts.update(truncated_rows=sum(row["truncated"] for row in rows), dropped_rows=dropped)
    result = save(first, recipe, job, out_dir, facts)
    # Times and sizes for the box log. They are not part of the LoRA files: a LoRA hash never depends on a clock.
    result.update(load_s=loaded - started, step_s=time.monotonic() - loaded, n_gpus=len(replicas), rows=len(rows))
    result.update(forward_tokens=sum(len(row["prompt_ids"]) + len(row["tokens"]) for row in active) * recipe.steps)
    return result


def save(replica: Replica, recipe: Recipe, job: dict, out_dir: Path, facts: dict) -> dict:
    """Apply the update cap, check the weights and write the LoRA in the PEFT file layout."""
    scale = recipe.alpha / recipe.rank
    squares = [((layer.lora_B.T @ layer.lora_B) * (layer.lora_A @ layer.lora_A.T)).sum().double() for layer in replica.layers.values()]
    norm = uncapped = scale * math.sqrt(max(0.0, float(torch.stack(squares).sum().detach())))
    if recipe.global_norm_cap is not None and norm > recipe.global_norm_cap:
        for layer in replica.layers.values():
            layer.lora_B.data.mul_(recipe.global_norm_cap / norm)
        norm = recipe.global_norm_cap
    tensors = {}
    for path, layer in replica.layers.items():
        for name, weight in (("lora_A", layer.lora_A), ("lora_B", layer.lora_B)):
            tensors[f"base_model.model.{path}.{name}.weight"] = weight.detach().to("cpu", torch.float32).contiguous()
    if not math.isfinite(norm) or not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise BatchRejected("training gave a weight that is not finite")
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": job["base_hash"],
        "r": recipe.rank,
        "lora_alpha": recipe.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(recipe.target_modules),
        "inference_mode": True,
    }
    used = {key: getattr(recipe, key) for key in ("lora_lr", "steps", "alpha", "group_size", "sample_temperature", "rank", "max_grad_norm", "global_norm_cap")}
    # step_norm_before_cap is the norm that training gave; step_norm is the norm of the stored LoRA.
    report = {"optimizer": OPTIMIZER, "seed": job["seed"], "step_norm": norm, "step_norm_before_cap": uncapped, "recipe_values": used, **facts}
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_bytes(canonical_json(config))
    (out_dir / "train_report.json").write_bytes(canonical_json(report))
    gpu_class = torch.cuda.get_device_name(0) if replica.device != "cpu" else "cpu"
    result = {"status": "ok", "loss": facts["losses"][-1], "step_norm": norm, "step_norm_before_cap": uncapped}
    return {**result, "tokens": facts["trained_tokens"], "gpu_class": gpu_class, "truncated_rows": facts["truncated_rows"], "dropped_rows": facts["dropped_rows"]}


def main(job_dir: Path) -> None:
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    try:
        report = run(job, job_dir / "lora")
    except BatchRejected as exc:
        report = {"status": "rejected", "reason": str(exc)}
    (job_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
