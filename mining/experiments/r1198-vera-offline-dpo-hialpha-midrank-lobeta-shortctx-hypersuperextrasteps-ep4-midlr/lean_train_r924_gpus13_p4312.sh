#!/usr/bin/env bash
# p4312: R1169 ShortCtx LoRank Loβ MidLR REFUTE m=+0.000984 SE=0.001544 ~0.32×
# ShortCtx LoRank Loβ LR exhausted (Ultra R1154 / Mid R1169 / Hi R1132) → MidRank Loβ MidLR ShortCtx
# ≠ UltraLoLR R1148 / ≠ HiLR R1134 / ≠ LoRank MidLR R1169 / ≠ SoftCtx MidRank Loβ MidLR R1179
# Fill r924 GPUs 1,3 after exact-PID reap of R1169 chall :8003. Never pkill -f. Leave R1178 on 6,7.
set -euo pipefail
exec >/root/logs/r1198_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1,3 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1198-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 1,3 ShortCtx MidRank Loβ Hyper MidLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1198-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-midlr
mkdir -p /root/r1198 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/s4-h1-sft /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1198/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1169/dpo_duel_reason.jsonl ]]; then cp -f /root/r1169/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1148/dpo_duel_reason.jsonl ]]; then cp -f /root/r1148/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1198-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1198_train.pid ]]; then
  old=$(cat /root/logs/r1198_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then echo FATAL alive; exit 1; fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[r1198-lean] wait VRAM1+3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r1198/train; mkdir -p /root/r1198/train
rm -f /root/logs/r1198_train.done /root/logs/r1198_merge.done /root/logs/r1198_merge_ready
: >/root/logs/r1198_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1198/train \
  --max-len 6144 --epochs 4 --lr 1e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.02 \
  --max-steps 38400 >/root/logs/r1198_train.nohup 2>&1 &
echo $! | tee /root/logs/r1198_train.pid >/root/r1198/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "axis":"r1198-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-midlr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695","lr":"1e-6","lora_r":32,"lora_alpha":128,"beta":0.02,
 "max_len":6144,"epochs":4,"max_steps":38400,"gpus":"1,3","chall_port":8003,
 "parent_signal":"R1169 ShortCtx LoRank Loβ MidLR REFUTE m=+0.000984 SE=0.001544 ~0.32×; ShortCtx LoRank Loβ LR exhausted R1154/R1169/R1132 → MidRank Loβ MidLR isolate (≠ Ultra R1148 / ≠ Hi R1134)",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"}
Path("/root/affine_data/r1198_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1198-lean] TRAIN pid=$(cat /root/logs/r1198_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1198_train_then_merge_p4312.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1198_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1198_merge_then_n80_p4312.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1198_merge_then_n80.pid
echo "[r1198-lean] waiters armed"
