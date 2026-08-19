#!/usr/bin/env bash
# R888: fresh mine-r888-grpo-reason-1 (gentle-orbit-0d) — pip + HF vera king + teacher.
# Axis: vera6 reign36 × GRPO-on-Reason HiAlpha (≠ Offline-DPO fleet; ≠ R583 r252 base).
# Pod reports 7×B200 visible (config 8×) — teacher TP1 + train 2-GPU.
# p3979: huggingface-cli is dead on hub 1.28 — use `hf download` (venv PATH).
set -euo pipefail
LOG=/root/logs/bootstrap_r888.log
mkdir -p /root/logs /root/hf /root/affine_data /root/r888 /root/mining_src/r3-reason-grpo
exec > >(tee -a "$LOG") 2>&1
echo "[r888-boot] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname) pass=${PASS:-3979}"

if [[ -f /root/mine.env ]]; then
  set -a; # shellcheck disable=SC1091
  source /root/mine.env; set +a
fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
unset HF_HUB_ENABLE_HF_TRANSFER || true
export PATH="/root/venv/bin:$HOME/.local/bin:$PATH"
export HF_TOKEN
test -n "${HF_TOKEN:-}"

NGPU=$(nvidia-smi -L | wc -l)
echo "[r888-boot] GPU_COUNT=$NGPU"
test "$NGPU" -ge 4
test -s /root/r888/winner_za_high_l1.jsonl
test -f /root/mining_src/r3-reason-grpo/train_reason_grpo.py

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -d /root/venv ]]; then
  uv venv /root/venv --python 3.12
fi
# shellcheck disable=SC1091
source /root/venv/bin/activate
export PATH="/root/venv/bin:$PATH"

NEED_PIP=1
if python -c 'import torch,transformers,vllm,peft,accelerate,httpx; assert torch.__version__.startswith("2.11"); assert transformers.__version__.startswith("5.14"); assert vllm.__version__.startswith("0.22.1"); print("[r888-boot] VERSIONS_OK", torch.__version__, transformers.__version__, vllm.__version__)' 2>/dev/null; then
  NEED_PIP=0
fi

if [[ "$NEED_PIP" -eq 1 ]]; then
  uv pip install \
    "torch==2.11.0" \
    "transformers==5.14.1" \
    "vllm==0.22.1" \
    "peft" "accelerate" "httpx" \
    "huggingface_hub" "hf_transfer" \
    "safetensors" "numpy" "scipy" "pandas" "pyarrow" \
    2>&1 | tee /root/logs/pip_r888.log | tail -40
  python -c 'import torch,transformers,vllm,peft,accelerate,httpx; print("[r888-boot] VERSIONS", torch.__version__, transformers.__version__, vllm.__version__); assert vllm.__version__.startswith("0.22.1")'
else
  echo "[r888-boot] skip pip (versions already OK)"
fi

hf_download() {
  local repo="$1" rev="$2" tag="$3"
  echo "[r888-boot] HF download $tag $repo@$rev"
  # hub≥1.28: huggingface-cli download is a hard no-op; use `hf download`.
  if ! command -v hf >/dev/null 2>&1; then
    echo "FATAL: hf CLI missing in PATH=$PATH" >&2
    exit 1
  fi
  hf download "$repo" --revision "$rev" --cache-dir "$HF_HOME" \
    2>&1 | tee "/root/logs/hf_${tag}_r888.log" | tail -40
}

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
TEACHER_REV=f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc

KING_SNAP=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/$KING_REV
TEACHER_SNAP=/root/hf/hub/models--zai-org--GLM-4.5-Air-FP8/snapshots/$TEACHER_REV

if [[ -e "$KING_SNAP/config.json" ]]; then
  echo "[r888-boot] king already present $KING_SNAP"
else
  hf_download "$KING_REPO" "$KING_REV" king
fi
test -e "$KING_SNAP/config.json"

if [[ -e "$TEACHER_SNAP/config.json" ]] || [[ -d "$TEACHER_SNAP" && -n "$(ls -A "$TEACHER_SNAP" 2>/dev/null || true)" ]]; then
  echo "[r888-boot] teacher already present $TEACHER_SNAP"
else
  hf_download "$TEACHER_REPO" "$TEACHER_REV" teacher
fi
test -e "$TEACHER_SNAP/config.json" || test -d "$TEACHER_SNAP"

python3 - <<PY
import json, time
from pathlib import Path
meta = {
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo": "R888",
  "pass": 3979,
  "stage": "BOOT_HF_DONE",
  "king_repo": "$KING_REPO",
  "king_rev": "$KING_REV",
  "teacher_repo": "$TEACHER_REPO",
  "teacher_rev": "$TEACHER_REV",
  "ngpu": int("$NGPU"),
  "note": "p3979 bootstrap HF done via hf download (cli dead); next=serve teacher TP1 + GRPO train",
}
Path("/root/affine_data/r888_bootstrap_done.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
PY
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r888_bootstrap.done
echo "[r888-boot] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
