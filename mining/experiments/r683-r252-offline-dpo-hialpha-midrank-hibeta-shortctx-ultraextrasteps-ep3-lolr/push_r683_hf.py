#!/usr/bin/env python3
"""Push /tmp/r683_merged → public HF with detailed Reason-v4 training card. Run on golden."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "unconst/Affine-5czsc2fc98-r683-r252-odpo-midrank-hibeta-shortctx-ultraextra-ep3-lolr-merged"
MERGE = Path("/tmp/r683_merged")
BASE = "unconst/Affine-5czsc2fc98-r252-merged"
BASE_REV = "b42d6245d77fe30885ea8a90387771e1bc465e0f"
EXP = (
    "mining/experiments/"
    "r683-r252-offline-dpo-hialpha-midrank-hibeta-shortctx-ultraextrasteps-ep3-lolr"
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
  - r683
---

# R683 — ShortCtx × MidRank × HiBeta UltraExtra ep3 × LoLR (offline DPO)

Affine SN120 challenger for **Reason v4** (`weight_version_key=7`): tempered
multi-sample log-mean-exp over k=3 teacher refs (τ=0.03).

Per turn: `a_i = lpC(y_i|z_A) − lpC(y_i|∅)`;  
`Reason = τ·log(mean_i exp(a_i/τ))`. Crown also needs median stripped `|z|≥80`
and B pass ≥0.30.

## How this checkpoint was trained

- **Base / parent:** `{BASE}@{BASE_REV}` (our crowned r252, reign 33)
- **Method:** offline DPO on Reason-ranked duel pairs (not SFT / not online GRPO)
- **What was optimized:** preference for thoughts that raise teacher-side Reason
  (commit to a teacher next-action mode; filler loses under LME)
- **Data:** ShortCtx × MidRank × HiBeta filtered duel preference pairs from
  `dpo_duel_reason.jsonl` under `{EXP}` (kept ≥200 examples at launch)
- **Key hyperparameters:**
  - LoRA r=**32** (MidRank), α=**128** (HiAlpha)
  - β=**0.3** (HiBeta)
  - lr=**1e-6** (LoLR)
  - max_len=**6144** (ShortCtx)
  - max_steps=**7200** (UltraExtra; 2× Mega 3600)
  - epochs=**3**
- **Hardware:** Lium `mine-r226-marsplan-fullft-1` (brave-raven-a9) 8×B200
  GPUs **6,7** train+merge; SCP → `mine-r262-kevin-v5-nonking-grpo-1`
  (golden-comet-78) GPUs **4,5** challenger serve + v4 n80 → `/tmp/r683_merged`
  (~66G / 16 safetensor shards)
- **Local n80 vs live king reign34**
  (`cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0…`) under **wvk=7**:
  - margin **+0.002137**, SE **0.000943**, z=**2.27**, n=**79**
  - bar `max(2·SE, δ=0.002)` = **0.002** (~**1.07×**)
  - thought median **172** (≥80 ✓), B pass **0.304** (≥0.30 ✓, knife-edge)
  - k=**3**, τ=**0.03** (fail-closed if stamp ≠ v4)
  - decision: **WIN / Stage-5 licensed** (`r683_decision_reign34_wvk7.json`, p3738)
- **Lineage:** amplify R659 Short MidRank HiBeta Mega / R622 ~0.85× with 2×
  steps; ≠ R679 Short HiRank HiBeta UltraExtra / ≠ R653 Short MidRank LoBeta /
  ≠ R658 Short MidRank MidBeta / ≠ Online / ≠ GRPO
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

    # Force online hub for this push (pod mine.env may set OFFLINE=1 for sims).
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    readme = MERGE / "README.md"
    readme.write_text(README)
    print(f"[r683-push] wrote model card → {readme}", flush=True)

    api = HfApi(token=token)
    who = api.whoami()
    print(f"[r683-push] whoami={who.get('name')}", flush=True)

    api.create_repo(REPO, private=False, exist_ok=True, repo_type="model")
    print(f"[r683-push] uploading {MERGE} → {REPO} …", flush=True)
    t0 = time.time()
    info = api.upload_folder(
        folder_path=str(MERGE),
        repo_id=REPO,
        repo_type="model",
        commit_message=(
            "R683 ShortCtx×MidRank×HiBeta UltraExtra ep3×LoLR offline-DPO "
            "(v4 n80 +0.002137 ~1.07× bar vs reign34 wvk7)"
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
        "n80_margin": 0.002137002820357482,
        "n80_se": 0.0009433265826316925,
        "n80_z": 2.265390226145935,
        "n80_bar": 0.002,
        "n80_thought_median": 172.0,
        "n80_b_pass": 0.3037974683544304,
        "contract": "wvk7",
        "n_teacher_samples": 3,
        "tau": 0.03,
    }
    out = Path("/root/affine_data/r683_hf_pushed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2) + "\n")
    Path("/root/affine_data/r683_hf_pushed.done").write_text(sha or "ok")
    print(f"[r683-push] DONE {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
