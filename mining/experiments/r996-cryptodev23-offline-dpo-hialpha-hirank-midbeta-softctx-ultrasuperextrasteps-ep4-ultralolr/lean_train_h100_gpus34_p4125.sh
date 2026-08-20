#!/usr/bin/env bash
# p4125: R973 SoftCtx Midβ Ultra HiLR REFUTE m=-0.008977 ~-1.36× → SoftCtx HiRank Midβ Ultra UltraLoLR
# ≠ MidRank Midβ Ultra HiLR R973 / ≠ MidRank Midβ Ultra UltraLoLR R944 / ≠ MidLoβ SoftCtx R926 /
# ≠ vera SoftCtx MidRank MidLoβ Mega UltraLoLR R994 / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r996_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3,4 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r996-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 3,4 SoftCtx HiRank Midβ Ultra UltraLoLR"
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
EXP=r996-cryptodev23-offline-dpo-hialpha-hirank-midbeta-softctx-ultrasuperextrasteps-ep4-ultralolr
mkdir -p /root/r996 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r996/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r973/dpo_duel_reason.jsonl ]]; then cp -f /root/r973/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r926/dpo_duel_reason.jsonl ]]; then cp -f /root/r926/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r996-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r996_train.pid ]]; then
  old=$(cat /root/logs/r996_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r996 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[r996-lean] wait VRAM3+4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 3,4 busy"; exit 1; }
rm -rf /root/r996/train; mkdir -p /root/r996/train
rm -f /root/logs/r996_train.done /root/logs/r996_merge.done /root/logs/r996_merge_ready
: >/root/logs/r996_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r996/train \
  --max-len 12288 --epochs 4 --lr 5e-7 \
  --lora-r 64 --lora-alpha 128 --beta 0.1 \
  --max-steps 28800 >/root/logs/r996_train.nohup 2>&1 &
echo $! | tee /root/logs/r996_train.pid >/root/r996/train.pid
python3 - <<'PY'
import json, time
from pathlib import Path
meta = {
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "axis": "r996-cryptodev23-offline-dpo-hialpha-hirank-midbeta-softctx-ultrasuperextrasteps-ep4-ultralolr",
  "base": "cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0",
  "lr": "5e-7",
  "lora_r": 64,
  "lora_alpha": 128,
  "beta": 0.1,
  "max_len": 12288,
  "epochs": 4,
  "max_steps": 28800,
  "gpus": "3,4",
  "parent_signal": "R973 SoftCtx Midβ Ultra HiLR REFUTE m=-0.008977 ~-1.36× (thought✓227 B✓0.377 k=3) → SoftCtx HiRank Midβ Ultra UltraLoLR isolate on cryptoDev; ≠ MidRank Midβ Ultra HiLR R973 / ≠ MidRank Midβ Ultra UltraLoLR R944 / ≠ MidLoβ SoftCtx R926 / ≠ vera SoftCtx MidLoβ Mega UltraLoLR R994 / ≠ Online / ≠ GRPO",
  "decision_rule": "Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36",
}
Path("/root/affine_data/r996_train_launched.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY
echo "[r996-lean] TRAIN launched pid=$(cat /root/logs/r996_train.pid) BASE=$BASE"
