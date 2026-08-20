#!/usr/bin/env python3
"""Push /tmp/r959_merged → public HF with detailed Reason-v4 training card. Run on mine-r338."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "unconst/Affine-5czsc2fc98-r959-vera-odpo-hirank-midbeta-softctx-megaextra-ep4-ultralolr-merged"
MERGE = Path("/tmp/r959_merged")
BASE = "vera6/affine-5g4yy75zuz-t6"
BASE_REV = "8e3f1695e058837ed80fec3238ff439fdc2d0f0e"
EXP = (
    "mining/experiments/"
    "r959-vera-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr"
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
  - r959
---

# R959 — SoftCtx × HiRank × MidBeta Mega UltraLoLR (offline DPO on vera king)

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
- **Data:** Soft Mid Mid Soft × SoftCtx filtered duel preference pairs from
  `dpo_duel_reason.jsonl` under `{EXP}` / pod `/root/r959/`
- **Key hyperparameters:**
  - LoRA r=**64** (HiRank), α=**128** (HiAlpha)
  - β=**0.1** (MidBeta)
  - lr=**5e-7** (UltraLoLR)
  - max_len=**12288** (SoftCtx)
  - max_steps=**19200** (MegaSuperExtra)
  - epochs=**4**
- **Hardware:** Lium `mine-r338-marsplan-online-dpo-bigg-hilr-1` (calm-fox-6a)
  8×B200 GPUs **4,5** train+merge; TKC warm; chall :8002 GPUs **4,5** for
  v4 n80 → `/tmp/r959_merged` (~66G / 16 safetensor shards)
- **Local n80 vs live king reign36** (`{BASE}@{BASE_REV}`) under **wvk=7**:
  - margin **+0.006384**, SE **0.002605**, z=**2.451**, n=**79**
  - bar `max(2·SE, δ=0.002)` = **0.005210** (~**1.23×**)
  - thought median **173** (≥80 ✓), B pass **0.406** (≥0.30 ✓)
  - k=**3**, τ=**0.03** (fail-closed if stamp ≠ v4)
  - decision: **WIN / Stage-5 licensed** (`r959_sim_result_reign36_wvk7.json`, p4101)
- **Lineage:** R947 SoftCtx HiRank Midβ Ultra REFUTE ~0.31× B✗0.289 → SoftCtx
  HiRank Midβ Mega UltraLoLR isolate; ≠ Ultra R947 / ≠ SoftCtx HiRank Midβ
  Mega R937 REFUTE / ≠ Online / ≠ GRPO
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
    print(f"[r959-push] wrote model card → {readme}", flush=True)

    api = HfApi(token=token)
    who = api.whoami()
    print(f"[r959-push] whoami={who.get('name')}", flush=True)

    api.create_repo(REPO, private=False, exist_ok=True, repo_type="model")
    print(f"[r959-push] uploading {MERGE} → {REPO} …", flush=True)
    t0 = time.time()
    info = api.upload_folder(
        folder_path=str(MERGE),
        repo_id=REPO,
        repo_type="model",
        commit_message=(
            "R959 SoftCtx×HiRank×MidBeta Mega UltraLoLR offline-DPO "
            "(v4 n80 +0.006384 ~1.23× bar vs reign36 wvk7)"
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
        "n80_margin": 0.0063841132784964475,
        "n80_se": 0.0026051572232085067,
        "n80_z": 2.450567367536377,
        "n80_bar": 0.005210314446417013,
        "n80_n": 79,
        "n80_thought_median": 173.0,
        "n80_b_pass": 0.405982905982906,
        "contract": "wvk7",
        "n_teacher_samples": 3,
        "tau": 0.03,
        "king": "reign36",
        "pass": "p4101",
    }
    out = Path("/root/affine_data/r959_hf_pushed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2) + "\n")
    Path("/root/logs/r959_hf_pushed.done").write_text(
        (sha or "ok") + "\n" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
    )
    print(f"[r959-push] DONE {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
