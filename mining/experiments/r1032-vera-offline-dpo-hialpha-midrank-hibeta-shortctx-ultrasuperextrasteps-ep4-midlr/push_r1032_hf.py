#!/usr/bin/env python3
"""Push /tmp/r1032_merged → public HF with detailed Reason-v4 training card. Run on mine-r924."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "unconst/Affine-5czsc2fc98-r1032-vera-odpo-midrank-hibeta-shortctx-ultraextra-ep4-midlr-merged"
MERGE = Path("/tmp/r1032_merged")
BASE = "vera6/affine-5g4yy75zuz-t6"
BASE_REV = "8e3f1695e058837ed80fec3238ff439fdc2d0f0e"
EXP = (
    "mining/experiments/"
    "r1032-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr"
)

README = f"""---
base_model: {BASE}
base_model_revision: {BASE_REV}
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
tags:
  - affine
  - sn120
  - reason-v4
  - offline-dpo
  - r1032
---

# R1032 — ShortCtx × MidRank × HiBeta Ultra MidLR (offline DPO on vera king)

Affine SN120 challenger for **Reason v4** (`weight_version_key=7`): tempered
multi-sample log-mean-exp over k=3 teacher refs (τ=0.03).

Per turn: `a_i = lpC(y_i|z_A) − lpC(y_i|∅)`;
`Reason = τ·log(mean_i exp(a_i/τ))`. Crown also needs median stripped `|z|≥80`
and B pass ≥0.30.

## How this checkpoint was trained

- **Base / parent:** `{BASE}@{BASE_REV}` (live king reign36)
- **Method:** offline DPO on Reason-ranked duel pairs (not SFT / not online GRPO)
- **What was optimized:** preference for thoughts that raise teacher-side Reason
  (commit to a teacher next-action mode; filler loses under LME)
- **Data:** Soft Mid Mid Soft → ShortCtx filtered duel preference pairs
  (`dpo_duel_reason.jsonl`, 604 lines) under `{EXP}` / pod `/root/r1032/`
- **Key hyperparameters:**
  - LoRA r=**32** (MidRank), α=**128** (HiAlpha)
  - β=**0.3** (HiBeta)
  - lr=**1e-6** (MidLR)
  - max_len=**6144** (ShortCtx)
  - max_steps=**28800** (UltraSuperExtra)
  - epochs=**4**
- **Hardware:** Lium `mine-r924-vera-midctx-hibeta-1` (cosmic-orbit-55)
  8×H200 GPUs **1,3** train+merge; TKC warm; chall :8003 GPUs **1,3** for
  v4 n80 → `/tmp/r1032_merged` (~16 safetensor shards)
- **Local n80 vs live king reign36** (`{BASE}@{BASE_REV}`) under **wvk=7**:
  - margin **+0.005461**, SE **0.002603**, z=**2.098**, n=**80**
  - bar `max(2·SE, δ=0.002)` = **0.005206** (~**1.049×**)
  - thought median **175.5** (≥80 ✓), B pass **0.475** (≥0.30 ✓)
  - k=**3**, τ=**0.03** (fail-closed if stamp ≠ v4)
  - decision: **WIN / Stage-5 licensed** (`r1032_sim_result_reign36_wvk7.json`, p4177)
- **Lineage:** R1015 ShortCtx MidRank Hiβ Mega MidLR REFUTE m=−0.002353 ~−0.94×
  → ShortCtx MidRank Hiβ Ultra MidLR isolate (Mega→Ultra);
  ≠ Mega MidLR R1015 / ≠ Mega UltraLoLR R1001 / ≠ Ultra HiLR R984 /
  ≠ SoftCtx MidRank Hiβ Mega MidLR R1027 / ≠ MidCtx MidRank Hiβ Ultra MidLR R1017 /
  ≠ Online / ≠ GRPO
- **Experiment path:** `{EXP}`

## Intended use

SN120 Affine miner submission / evalsrv Reason v4 duel. Not a general chat model.

## License

Follows base model + Affine mining artifacts policy.
"""


def main() -> None:
    token = os.environ.get("HF_TOKEN") or ""
    if not token:
        raise SystemExit("HF_TOKEN missing")
    if not MERGE.is_dir() or not (MERGE / "config.json").is_file():
        raise SystemExit(f"missing merge dir {MERGE}")

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    readme = MERGE / "README.md"
    readme.write_text(README)
    print(f"[r1032-push] wrote model card → {readme}", flush=True)

    api = HfApi(token=token)
    who = api.whoami()
    print(f"[r1032-push] whoami={who.get('name')}", flush=True)

    api.create_repo(REPO, private=False, exist_ok=True, repo_type="model")
    print(f"[r1032-push] uploading {MERGE} → {REPO} …", flush=True)
    t0 = time.time()
    info = api.upload_folder(
        folder_path=str(MERGE),
        repo_id=REPO,
        repo_type="model",
        commit_message=(
            "R1032 ShortCtx×MidRank×HiBeta Ultra MidLR offline-DPO "
            "(v4 n80 +0.005461 ~1.049× bar vs reign36 wvk7)"
        ),
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.model",
            "tokenizer*",
            "vocab*",
            "merges.txt",
            "special_tokens_map.json",
            "chat_template*",
            "README.md",
            "*.jinja",
        ],
        ignore_patterns=[
            "*.py",
            "*.bin",
            "optimizer*",
            "rng*",
            "scheduler*",
            "*_result.json",
            "adapter*",
            "MERGE_DONE",
        ],
    )
    elapsed = time.time() - t0
    sha = getattr(info, "commit_id", None) or getattr(info, "oid", None)
    if not sha:
        try:
            sha = api.repo_info(REPO, repo_type="model").sha
        except Exception:
            sha = None
    meta = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": REPO,
        "revision": sha,
        "elapsed_s": elapsed,
        "merge_dir": str(MERGE),
        "whoami": who.get("name"),
        "n80_margin": 0.005461317130130356,
        "n80_se": 0.0026027643111423433,
        "n80_z": 2.098275708926332,
        "n80_bar": 0.0052055286222846865,
        "n80_n": 80,
        "n80_thought_median": 175.5,
        "n80_b_pass": 0.475,
        "contract": "wvk7",
        "n_teacher_samples": 3,
        "tau": 0.03,
        "king": "reign36",
        "pass": "p4177",
    }
    out = Path("/root/affine_data/r1032_hf_pushed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2) + "\n")
    Path("/root/logs/r1032_hf_pushed.done").write_text(
        (sha or "ok") + "\n" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
    )
    print(f"[r1032-push] DONE {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
