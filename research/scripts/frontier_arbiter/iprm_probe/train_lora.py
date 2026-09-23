"""IPRM probe — stage 2 (runs ON THE POD): LoRA-SFT the frozen teacher on the
assistant turns of solved trajectories -> C+.

Input  train.jsonl  {ids: [int], labels: [0/1]}   (built by build_data.py)
Output <out>/adapter (peft), <out>/train_log.jsonl (per-step loss / tok/s)

Loss = mean token CE over label positions (assistant content + <|im_end|>);
logits are computed only at label positions (248k vocab x 12k tokens in
fp32 would be 12 GB otherwise). bf16 weights, LoRA on every linear of the
language model (attention, GDN in/out projections, MLP), gradient
checkpointing, AdamW + cosine, one window per micro-step, grad accumulation.

    python train_lora.py --data /root/iprm/train.jsonl --out /root/iprm/cplus \
        --epochs 1 --accum 8 --lr 1e-4 --rank 32 --max-minutes 120
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL = "Qwen/Qwen3.8-27B"


def load_windows(path: Path, max_tokens: int) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if len(d["ids"]) <= max_tokens and sum(d["labels"]) > 0:
                out.append(d)
    return out


def lm_parts(model):
    """(backbone returning hidden states, lm_head) for the ConditionalGeneration
    or CausalLM wrapper; peft wraps one more level."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.get_output_embeddings()
    inner = getattr(base, "model", None)
    if inner is None:
        raise RuntimeError("cannot find backbone")
    # Qwen3_5ForConditionalGeneration.model = Qwen3_5Model(language_model, visual);
    # calling it with input_ids only skips the vision tower.
    return inner, head


def step_loss(model, ids: list[int], labels: list[int], device) -> tuple[torch.Tensor, int]:
    x = torch.tensor(ids, device=device).unsqueeze(0)
    lab = torch.tensor(labels, device=device)
    # position t predicts token t+1: loss positions = t where labels[t+1] == 1
    tgt_pos = torch.nonzero(lab[1:], as_tuple=False).squeeze(1)      # indices into ids[1:]
    backbone, head = lm_parts(model)
    out = backbone(input_ids=x, use_cache=False)
    h = out.last_hidden_state[0]                                        # [T, H]
    hs = h[tgt_pos]                                                     # [n, H]
    logits = head(hs).float()
    targets = x[0, 1:][tgt_pos]
    loss = F.cross_entropy(logits, targets, reduction="mean")
    return loss, int(tgt_pos.numel())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=12400)
    ap.add_argument("--max-minutes", type=float, default=150.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N windows")
    ap.add_argument("--filter", default="", help="'failed' -> only windows with mixed==... (unused)")
    ap.add_argument("--save-every", type=int, default=50)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = torch.device("cuda")

    wins = load_windows(a.data, a.max_tokens)
    if a.limit:
        wins = wins[: a.limit]
    random.shuffle(wins)
    n_tok = sum(len(w["ids"]) for w in wins)
    n_loss = sum(sum(w["labels"]) for w in wins)
    print(f"windows {len(wins)}  tokens {n_tok/1e6:.2f}M  loss tokens {n_loss/1e6:.2f}M", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    print(f"loaded {type(model).__name__} in {time.time()-t0:.0f}s", flush=True)
    model.config.use_cache = False
    # every linear of the LANGUAGE model (skip the vision tower and lm_head)
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "visual" not in name and "lm_head" not in name \
                and "mtp" not in name:
            targets.append(name)
    print(f"lora targets: {len(targets)} linears, e.g. {targets[:3]} ... {targets[-3:]}", flush=True)
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                      target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_micro = int(len(wins) * a.epochs)
    total_steps = max(1, total_micro // a.accum)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, total_steps)
    print(f"optimizer steps {total_steps} (micro {total_micro}, accum {a.accum})", flush=True)

    log = open(a.out / "train_log.jsonl", "a")
    t_start = time.time()
    micro = step = 0
    acc_loss = acc_tok = 0.0
    seen_tok = 0
    ema = None
    order = list(range(len(wins)))
    while micro < total_micro:
        i = order[micro % len(wins)]
        if micro % len(wins) == 0 and micro > 0:
            random.shuffle(order)
        w = wins[i]
        loss, n = step_loss(model, w["ids"], w["labels"], device)
        (loss / a.accum).backward()
        acc_loss += loss.item() * n
        acc_tok += n
        seen_tok += len(w["ids"])
        micro += 1
        if micro % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            mean_loss = acc_loss / max(acc_tok, 1)
            ema = mean_loss if ema is None else 0.9 * ema + 0.1 * mean_loss
            el = time.time() - t_start
            rec = {"step": step, "micro": micro, "loss": mean_loss, "ema": ema, "lr": sched.get_last_lr()[0],
                   "tok_s": seen_tok / el, "elapsed_min": el / 60,
                   "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            acc_loss = acc_tok = 0.0
            if step % a.save_every == 0:
                model.save_pretrained(a.out / "adapter")
            if el / 60 > a.max_minutes:
                print("time budget reached", flush=True)
                break
    model.save_pretrained(a.out / "adapter")
    json.dump({"steps": step, "micro": micro, "windows": len(wins), "tokens_seen": seen_tok,
               "minutes": (time.time() - t_start) / 60, "final_ema": ema, "args": vars(a) | {"data": str(a.data), "out": str(a.out)}},
              open(a.out / "train_summary.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
