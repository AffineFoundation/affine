#!/usr/bin/env python3
"""H138/F43: Offline DPO on duel thought pairs ranked by teacher Λ2.

Each row: messages, chosen z (higher Λ2), rejected z (lower Λ2).
No live teacher, no sampling — pure preference on thought tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

_AFFINE = Path("/root/mining_src/affine_pkg")
if _AFFINE.is_dir() and str(_AFFINE) not in sys.path:
    sys.path.insert(0, str(_AFFINE))
from evalsrv.chat import THINK_OPEN  # noqa: E402


def _msg_chars(row: dict) -> int:
    return sum(len(m.get("content") or "") for m in row.get("messages") or [])


def _normalize_z(z_text: str) -> str:
    z = (z_text or "").strip()
    if z.startswith("<think>"):
        z = z[len("<think>") :].lstrip()
    if "</think>" in z:
        z = z.split("</think>", 1)[0].strip()
    return z


def _logp_continuation(
    model,
    tok,
    prompt: str,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """Mean token logp of `text` given `prompt` (thought continuation)."""
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt")
    forced = tok(prompt + text, add_special_tokens=False, return_tensors="pt")
    ids = forced["input_ids"].to(device)
    n_p = enc["input_ids"].shape[1]
    logits = model(input_ids=ids).logits
    logp = F.log_softmax(logits[0, n_p - 1 : -1], dim=-1)
    targets = ids[0, n_p:]
    if targets.numel() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    tok_lp = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    return tok_lp.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    raw = [
        json.loads(l)
        for l in args.data.read_text().splitlines()
        if l.strip()
    ]
    budget = int(args.max_len * 2.0)
    rows = []
    for r in raw:
        ch = _normalize_z(r.get("chosen") or "")
        rj = _normalize_z(r.get("rejected") or "")
        if not ch or not rj or ch == rj:
            continue
        if not r.get("messages") or not r.get("y"):
            continue
        if _msg_chars(r) + len(ch) + len(rj) > budget:
            continue
        rows.append({**r, "chosen": ch, "rejected": rj})
    rows.sort(key=_msg_chars)
    print(f"[dpo] kept={len(rows)}/{len(raw)} budget_chars={budget}", flush=True)
    if len(rows) < 50:
        raise SystemExit(f"too few rows after fit-filter: {len(rows)}")

    meta = {
        "experiment": "s4-h138-f43-tok-dpo-l2",
        "family": "F43",
        "method": "offline_dpo_l2",
        "base": args.base,
        "data": str(args.data),
        "n_rows": len(rows),
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "beta": args.beta,
        "max_steps": args.max_steps,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out_dir / "train_config.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[dpo] loading base {args.base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    # Reference = frozen initial policy (LoRA at init ≈ base). Capture once.
    # For LoRA-DPO we use the common simplification: ref logp from disabled adapters.
    steps = 0
    epoch = 0
    hist: list[dict] = []
    while steps < args.max_steps and epoch < math.ceil(args.epochs):
        epoch += 1
        for row in rows:
            if steps >= args.max_steps:
                break
            messages = row["messages"]
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if not prompt.rstrip().endswith(THINK_OPEN):
                prompt = prompt + THINK_OPEN
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            if len(p_ids) > args.max_len - 256:
                continue

            chosen = row["chosen"]
            rejected = row["rejected"]

            model.train()
            lp_c = _logp_continuation(model, tok, prompt, chosen, device)
            lp_r = _logp_continuation(model, tok, prompt, rejected, device)

            # Reference logps with adapters disabled (base policy).
            with torch.no_grad():
                with model.disable_adapter():
                    model.eval()
                    ref_c = _logp_continuation(model, tok, prompt, chosen, device)
                    ref_r = _logp_continuation(model, tok, prompt, rejected, device)
            model.train()

            # DPO: -log σ(β * ((lp_c - ref_c) - (lp_r - ref_r)))
            logits = args.beta * ((lp_c - ref_c) - (lp_r - ref_r))
            loss = -F.logsigmoid(logits)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            opt.step()

            steps += 1
            rec = {
                "step": steps,
                "loss": float(loss.detach().item()),
                "logit": float(logits.detach().item()),
                "gap": row.get("gap"),
                "z0": chosen[:80],
            }
            hist.append(rec)
            if steps % 5 == 0 or steps <= 3:
                print(f"[dpo-log] {json.dumps(rec)}", flush=True)
            if steps % 50 == 0:
                ckpt = args.out_dir / "checkpoints" / f"step-{steps}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt)
                tok.save_pretrained(ckpt)

    adapter = args.out_dir / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    result = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": steps,
        "elapsed_s": time.time() - t0,
        "mean_loss_last20": (
            sum(h["loss"] for h in hist[-20:]) / max(1, len(hist[-20:]))
        ),
        "adapter": str(adapter),
    }
    (args.out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    (args.out_dir / "train_history.jsonl").write_text(
        "\n".join(json.dumps(h) for h in hist) + "\n"
    )
    (args.out_dir / "train.done").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
    )
    print(f"[dpo] DONE {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
