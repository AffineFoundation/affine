#!/usr/bin/env bash
# p4265: R1121 MidCtx HiRank MidLoβ Hyper HiLR REFUTE m=+0.002685 ~0.41× thought✓208.5 B✓0.4375 k=3
# → UltraLoLR isolate (lr 2e-6→5e-7); keep MidCtx HiRank MidLoβ HyperExtra steps=38400
# ≠ HiLR R1121 / ≠ MidLR R1097 / ≠ SoftCtx HiRank MidLoβ UltraLoLR R1138 / ≠ SoftCtx HiRank MidLoβ Hyper HiLR R1110
# ≠ MidCtx HiRank Midβ Hyper HiLR R1100 / ≠ ShortCtx HiRank MidLoβ Hyper HiLR R1123 / ≠ Online / ≠ GRPO
# Fill r340 GPUs 3,4 after exact-PID reap of R1121 chall :8003. Never pkill -f.
set -euo pipefail
exec >/root/logs/r1141_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3,4 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1141-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 3,4 MidCtx HiRank MidLoβ Hyper UltraLoLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1141-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr
mkdir -p /root/r1141 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/s4-h1-sft /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1141/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1121/dpo_duel_reason.jsonl ]]; then cp -f /root/r1121/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1120/dpo_duel_reason.jsonl ]]; then cp -f /root/r1120/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1141-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1141_train.pid ]]; then
  old=$(cat /root/logs/r1141_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then echo FATAL alive; exit 1; fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[r1141-lean] wait VRAM3+4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r1141/train; mkdir -p /root/r1141/train
rm -f /root/logs/r1141_train.done /root/logs/r1141_merge.done /root/logs/r1141_merge_ready
: >/root/logs/r1141_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1141/train \
  --max-len 8192 --epochs 4 --lr 5e-7 \
  --lora-r 64 --lora-alpha 128 --beta 0.05 \
  --max-steps 38400 >/root/logs/r1141_train.nohup 2>&1 &
echo $! | tee /root/logs/r1141_train.pid >/root/r1141/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "axis":"r1141-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695","lr":"5e-7","lora_r":64,"lora_alpha":128,"beta":0.05,
 "max_len":8192,"epochs":4,"max_steps":38400,"gpus":"3,4","chall_port":8003,
 "parent_signal":"R1121 MidCtx HiRank MidLoβ Hyper HiLR REFUTE m=+0.002685 ~0.41× thought✓208.5 B✓0.4375 k=3 → UltraLoLR isolate; ≠ HiLR R1121 / ≠ MidLR R1097 / ≠ SoftCtx UltraLoLR R1138 / ≠ SoftCtx HiLR R1110 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"}
Path("/root/affine_data/r1141_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1141-lean] TRAIN pid=$(cat /root/logs/r1141_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1141_train_then_merge_p4265.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1141_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1141_merge_then_n80_p4265.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1141_merge_then_n80.pid
echo "[r1141-lean] waiters armed"
