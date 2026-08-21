#!/usr/bin/env bash
# p4328: R1187 SoftCtx MidRank MidLoβ Hyper MidLR REFUTE m=-0.00715 SE=0.00548 ~-0.65x
# thought✓202 B✓0.438 k=3 → SoftCtx Mega MidLR isolate (≠ Hyper MidLR R1187 / ≠ Mega UltraLoLR R915/R926)
# Fill r926 GPUs 3,4 after exact-PID reap of R1187 chall :8002. Never pkill -f.
set -euo pipefail
exec >/root/logs/r1220_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3,4 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1220-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 3,4 SoftCtx MidRank MidLoβ Mega MidLR"
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
EXP=r1220-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-midlr
mkdir -p /root/r1220 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1220/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1187/dpo_duel_reason.jsonl ]]; then cp -f /root/r1187/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r926/dpo_duel_reason.jsonl ]]; then cp -f /root/r926/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing SoftCtx data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1220-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1220_train.pid ]]; then
  old=$(cat /root/logs/r1220_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1220 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[r1220-lean] wait VRAM3+4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 3,4 busy"; exit 1; }
rm -rf /root/r1220/train; mkdir -p /root/r1220/train
rm -f /root/logs/r1220_train.done /root/logs/r1220_merge.done /root/logs/r1220_merge_ready
: >/root/logs/r1220_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1220/train \
  --max-len 12288 --epochs 4 --lr 1e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r1220_train.nohup 2>&1 &
echo $! | tee /root/logs/r1220_train.pid >/root/r1220/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "pass": 4328,
 "axis":"r1220-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-midlr",
 "base":"cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0",
 "lr":"1e-6","lora_r":32,"lora_alpha":128,"beta":0.05,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"3,4","chall_port":8002,
 "parent_signal":"R1187 SoftCtx MidRank MidLoβ Hyper MidLR REFUTE m=-0.00715 SE=0.00548 ~-0.65x thought✓202 B✓0.438 k=3 → SoftCtx Mega MidLR isolate",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1220_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1220-lean] TRAIN launched pid=$(cat /root/logs/r1220_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1220_train_then_merge_p4328.sh >/root/logs/r1220_wait_merge.nohup 2>&1 &
echo $! >/root/logs/r1220_wait_merge.pid
echo "[r1220-lean] merge waiter armed (n80 needs free chall slot later)"
