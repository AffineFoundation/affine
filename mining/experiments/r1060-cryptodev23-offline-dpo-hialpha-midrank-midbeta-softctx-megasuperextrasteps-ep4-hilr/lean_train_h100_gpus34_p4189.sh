#!/usr/bin/env bash
# p4189: R1051 SoftCtx MidRank Midβ Mega MidLR still TRAIN; free GPUs3,4 after teacher TP4→TP2
# → SoftCtx MidRank Midβ Mega HiLR isolate (lr=2e-6). ≠ MidLR R1051 / ≠ Ultra MidLR R1025 /
# ≠ Ultra HiLR R973 / ≠ Ultra UltraLoLR R944 / ≠ vera SoftCtx MidRank Midβ Mega HiLR R989 /
# ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r1060_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3,4 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1060-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 3,4 SoftCtx MidRank Midβ Mega HiLR"
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
EXP=r1060-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-hilr
mkdir -p /root/r1060 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1060/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1051/dpo_duel_reason.jsonl ]]; then cp -f /root/r1051/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1025/dpo_duel_reason.jsonl ]]; then cp -f /root/r1025/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1060-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1060_train.pid ]]; then
  old=$(cat /root/logs/r1060_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1060 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[r1060-lean] wait VRAM3+4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 3,4 busy"; exit 1; }
rm -rf /root/r1060/train; mkdir -p /root/r1060/train
rm -f /root/logs/r1060_train.done /root/logs/r1060_merge.done /root/logs/r1060_merge_ready
: >/root/logs/r1060_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1060/train \
  --max-len 12288 --epochs 4 --lr 2e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.1 \
  --max-steps 19200 >/root/logs/r1060_train.nohup 2>&1 &
echo $! | tee /root/logs/r1060_train.pid >/root/r1060/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r1060-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-hilr",
 "base":"cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0",
 "lr":"2e-6","lora_r":32,"lora_alpha":128,"beta":0.1,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"3,4","chall_port":8002,
 "parent_signal":"p4189 teacher TP4→TP2 freed GPUs3,4; R1051 Mega MidLR still TRAIN → Mega HiLR isolate",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1060_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1060-lean] TRAIN launched pid=$(cat /root/logs/r1060_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1060_train_then_merge_p4189.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1060_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1060_merge_then_n80_p4189.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1060_merge_then_n80.pid
echo "[r1060-lean] waiters armed"
