#!/usr/bin/env bash
# R888 p3978: fresh mine-r888-grpo-reason-1 (gentle-orbit-0d) — pip + HF vera king + teacher.
# Axis: vera6 reign36 × GRPO-on-Reason HiAlpha (≠ Offline-DPO fleet; ≠ R583 r252 base).
# Pod reports 7×B200 visible (config 8×) — teacher TP1 + train 2-GPU.
set -euo pipefail
LOG=/root/logs/bootstrap_r888.log
mkdir -p /root/logs /root/hf /root/affine_data /root/r888 /root/mining_src/r3-reason-grpo
exec > >(tee -a "$LOG") 2>&1
echo "[r888-boot] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"

if [[ -f /root/mine.env ]]; then
  set -a; # shellcheck disable=SC1091
  source /root/mine.env; set +a
fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export PATH="$HOME/.local/bin:$PATH"
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

uv pip install \
  "torch==2.11.0" \
  "transformers==5.14.1" \
  "vllm==0.22.1" \
  "peft" "accelerate" "httpx" \
  "huggingface_hub[hf_transfer]" "hf_transfer" \
  "safetensors" "numpy" "scipy" "pandas" "pyarrow" \
  2>&1 | tee /root/logs/pip_r888.log | tail -40

python - <<'PY'
import torch, transformers, vllm, peft, accelerate, httpx
print("[r888-boot] VERSIONS", torch.__version__, transformers.__version__, vllm.__version__)
assert vllm.__version__.startswith("0.22.1"), vllm.__version__
PY

KING_REPO=vera6/affine-5g4yy75zuz-t6
KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REPO=zai-org/GLM-4.5-Air-FP8
TEACHER_REV=f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc

echo "[r888-boot] HF download king $KING_REPO@$KING_REV"
huggingface-cli download "$KING_REPO" --revision "$KING_REV" --local-dir-use-symlinks False \
  2>&1 | tee /root/logs/hf_king_r888.log | tail -20
echo "[r888-boot] HF download teacher $TEACHER_REPO@$TEACHER_REV"
huggingface-cli download "$TEACHER_REPO" --revision "$TEACHER_REV" --local-dir-use-symlinks False \
  2>&1 | tee /root/logs/hf_teacher_r888.log | tail -20

KING_SNAP=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/$KING_REV
TEACHER_SNAP=/root/hf/hub/models--zai-org--GLM-4.5-Air-FP8/snapshots/$TEACHER_REV
test -e "$KING_SNAP/config.json"
test -e "$TEACHER_SNAP/config.json" || test -d "$TEACHER_SNAP"

python3 - <<PY
import json, time
from pathlib import Path
meta = {
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo": "R888",
  "pass": 3978,
  "stage": "BOOT_HF_DONE",
  "king_repo": "$KING_REPO",
  "king_rev": "$KING_REV",
  "teacher_repo": "$TEACHER_REPO",
  "teacher_rev": "$TEACHER_REV",
  "ngpu": int("$NGPU"),
  "note": "p3978 bootstrap HF done; next=serve teacher TP1 + GRPO train on free GPUs",
}
Path("/root/affine_data/r888_bootstrap_done.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
PY
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r888_bootstrap.done
echo "[r888-boot] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
