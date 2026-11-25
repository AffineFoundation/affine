"""Ordered affine-batch-v4 GRPO/SFT stages on one LoRA.

The legacy run/run_minibatches functions remain the v1-v3 implementation.
Every stage gets fresh Adam and a deterministic seed derived from the job seed
and stage index. GRPO q_old/support masks are measured again at each boundary.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import torch
import transformers

from ..errors import BatchRejected, InfraError
from ..params import Recipe
from .batch import staged_rows
from .train_job import (
    LOGIT_CHUNK, WARM_UP_TOKENS, Replica, bring, check_names, encode,
    minibatches_of, one_thread_in_the_tuner, ppo_loss, save, score_old, token_scores,
)


def stage_seed(seed: int, index: int) -> int:
    """The deterministic independent stream of one stage, including stage 0."""
    return int.from_bytes(hashlib.sha256(f"{seed}|stage|{index}".encode()).digest()[:8], "big")


def sft_loss(replica: Replica, row: dict, n_rows: int) -> tuple[torch.Tensor, int]:
    """Token-mean negative full-vocabulary log probability at temperature 1.

    ``token_scores``' first return is deliberately used, not support q. Every
    encoded output token participates, including the appended end token.
    """
    targets = torch.tensor(row["tokens"], device=replica.device)
    hidden = replica.hidden(row["prompt_ids"] + row["tokens"])[0, len(row["prompt_ids"]) - 1 : -1]
    head = replica.model.get_output_embeddings()
    # Keep the output-vocabulary graph to one chunk, just as PPO does. Each
    # chunk contributes sum(NLL) / total_tokens, so their sum is the exact
    # token-mean gradient even when the last chunk is shorter.
    leaf = hidden.detach().requires_grad_(True)
    values = []
    for i in range(0, len(targets), LOGIT_CHUNK):
        log_p, _, _ = token_scores(head(leaf[i : i + LOGIT_CHUNK]), targets[i : i + LOGIT_CHUNK], replica.recipe)
        values.append(log_p.detach())
        part = -log_p.sum() * replica.recipe.sft_weight / (len(targets) * n_rows)
        part.backward()
    hidden.backward(leaf.grad)
    loss = -torch.cat(values).mean() * replica.recipe.sft_weight / n_rows
    return loss, len(targets)


class StageReplica(Replica):
    """One LoRA replica whose optimization recipe changes between stages."""

    def set_recipe(self, recipe: Recipe) -> None:
        self.recipe = recipe

    def mixed_rows(self, rows: Sequence[dict], n_rows: int) -> tuple[float, int]:
        """Accumulate PPO and SFT gradients with the same row normalizer."""
        head, loss, trained = self.model.get_output_embeddings(), 0.0, 0
        for row in rows:
            if row.get("kind") == "sft":
                if row["tokens"]:
                    part, count = sft_loss(self, row, n_rows)
                    loss += float(part)
                    trained += count
                continue
            targets = torch.tensor(row["tokens"], device=self.device)
            bring(self, (row,))
            hidden = self.hidden(row["prompt_ids"] + row["tokens"])[0, len(row["prompt_ids"]) - 1 : -1]
            leaf = hidden.detach().requires_grad_(True)
            chunks = range(0, len(targets), LOGIT_CHUNK)
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


def _step(replicas: Sequence[StageReplica], optimizer: torch.optim.Optimizer, recipe: Recipe) -> float:
    first = replicas[0]
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
    return norm


def run_stages(job: dict, out_dir: Path, devices: Sequence[str] | None = None) -> dict:
    top, started = Recipe.from_dict(job["recipe"]), time.monotonic()
    stages = job["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 5:
        raise BatchRejected("a staged job needs one to five stages")
    devices = list(devices or [f"cuda:{i}" for i in range(torch.cuda.device_count())])
    if not devices:
        raise InfraError("the training job sees no GPU")
    if any(device != "cpu" for device in devices):
        torch.cuda.reset_peak_memory_stats()
    one_thread_in_the_tuner()
    seed = job["seed"]
    torch.manual_seed(seed)
    replicas = [StageReplica(job["base_dir"], device, job["dtype"], top, seed) for device in devices]
    first = replicas[0]
    check_names(job["base_dir"], list(first.layers))
    tokenizer = transformers.AutoTokenizer.from_pretrained(job["base_dir"], local_files_only=True)
    losses, grad_norms, stage_reports = [], [], []
    all_rows = dropped = truncated = forward_tokens = 0
    loaded = time.monotonic()
    with ThreadPoolExecutor(len(replicas)) as pool:
        for index, stage in enumerate(stages):
            seed = stage_seed(job["seed"], index)
            torch.manual_seed(seed)
            recipe = Recipe.from_dict(stage["recipe"])
            for replica in replicas:
                replica.set_recipe(recipe)
            raw_rows = stage.pop("rows")
            rows, cut = encode(tokenizer, raw_rows, recipe)
            for row in raw_rows:
                row.pop("prompt", None)
                row.pop("output", None)
            raw_rows.clear()
            dropped += cut
            truncated += sum(row["truncated"] for row in rows)
            all_rows += len(rows)
            active = sorted((row for row in rows if row["kind"] == "sft" or row["advantage"] != 0), key=lambda row: -len(row["tokens"]))
            if not active:
                raise BatchRejected(f"stage {index} has no trainable row")
            if len(replicas) > len(active):
                replicas = replicas[:len(active)]
                first = replicas[0]
            if index == 0:
                for replica in replicas:
                    replica.warm_up((active[0]["prompt_ids"] + active[0]["tokens"])[:WARM_UP_TOKENS])
            optimizer = torch.optim.Adam(first.weights, lr=recipe.lora_lr, eps=recipe.adam_eps)
            order = random.Random(seed)
            shares = [active[i::len(replicas)] for i in range(len(replicas))]
            # q_old, masks and counts always include all earlier stage updates.
            list(pool.map(lambda pair: score_old(pair[0], pair[1]), zip(replicas, [[row for row in share if row["kind"] == "grpo"] for share in shares])))
            # A later minibatch may place a row on another replica. mixed_rows
            # brings each GRPO row's old-policy tensors to its current device.
            # SFT needs no old-policy pass.
            grpo_tokens = sum(len(row["prompt_ids"]) + len(row["tokens"]) for row in active if row["kind"] == "grpo")
            sft_tokens = sum(len(row["prompt_ids"]) + len(row["tokens"]) for row in active if row["kind"] == "sft")
            forward_tokens += grpo_tokens * (recipe.steps + 1) + sft_tokens * recipe.steps
            trained = 0
            stage_losses = 0
            for _ in range(recipe.steps):
                for group in minibatches_of(active, recipe.minibatches, order):
                    shares = [group[i::len(replicas)] for i in range(len(replicas))]
                    results = list(pool.map(lambda pair: pair[0].mixed_rows(pair[1], len(group)), zip(replicas, shares)))
                    norm = _step(replicas, optimizer, recipe)
                    losses.append(sum(loss for loss, _ in results))
                    grad_norms.append(norm)
                    trained += sum(count for _, count in results)
                    stage_losses += 1
            stage_reports.append({"index": index, "seed": seed, "rows": len(rows), "steps": recipe.steps, "updates": stage_losses, "trained_tokens": trained})
    facts = {"losses": losses, "grad_norms": grad_norms, "trained_tokens": stage_reports[-1]["trained_tokens"], "rows": all_rows,
             "truncated_rows": truncated, "dropped_rows": dropped, "stages": stage_reports}
    result = save(first, top, job, out_dir, facts)
    result.update(load_s=loaded - started, step_s=time.monotonic() - loaded, n_gpus=len(replicas), rows=all_rows, updates=len(losses), forward_tokens=forward_tokens)
    return result
