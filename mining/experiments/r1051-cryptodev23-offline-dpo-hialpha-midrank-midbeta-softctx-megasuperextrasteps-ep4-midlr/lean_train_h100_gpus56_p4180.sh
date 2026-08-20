#!/usr/bin/env bash
# p4180: R1025 cryptoDev SoftCtx MidRank Midβ Ultra MidLR REFUTE m=+0.000156 ~0.028× → Mega MidLR isolate
# ≠ Ultra MidLR R1025 / ≠ MidRank Ultra UltraLoLR R944 / ≠ MidRank Ultra HiLR R973 /
# ≠ HiRank SoftCtx Midβ Ultra MidLR R1013 / ≠ vera SoftCtx MidRank Midβ Mega MidLR R1037 REFUTE / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r1051_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=5,6 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1051-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 5,6 SoftCtx MidRank Midβ Mega MidLR"
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
EXP=r1051-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-midlr
mkdir -p /root/r1051 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1051/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1025/dpo_duel_reason.jsonl ]]; then cp -f /root/r1025/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1013/dpo_duel_reason.jsonl ]]; then cp -f /root/r1013/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1051-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1051_train.pid ]]; then
  old=$(cat /root/logs/r1051_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1051 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
  echo "[r1051-lean] wait VRAM5+6 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 5,6 busy"; exit 1; }
rm -rf /root/r1051/train; mkdir -p /root/r1051/train
rm -f /root/logs/r1051_train.done /root/logs/r1051_merge.done /root/logs/r1051_merge_ready
: >/root/logs/r1051_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1051/train \
  --max-len 12288 --epochs 4 --lr 1e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.1 \
  --max-steps 19200 >/root/logs/r1051_train.nohup 2>&1 &
echo $! | tee /root/logs/r1051_train.pid >/root/r1051/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r1051-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-midlr",
 "base":"cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0",
 "lr":"1e-6","lora_r":32,"lora_alpha":128,"beta":0.1,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"5,6",
 "parent_signal":"R1025 SoftCtx MidRank Midβ Ultra MidLR REFUTE m=+0.000156 ~0.028× → Mega MidLR isolate",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1051_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1051-lean] TRAIN launched pid=$(cat /root/logs/r1051_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1051_train_then_merge_p4180.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1051_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1051_merge_then_n80_p4180.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1051_merge_then_n80.pid
echo "[r1051-lean] waiters armed"
