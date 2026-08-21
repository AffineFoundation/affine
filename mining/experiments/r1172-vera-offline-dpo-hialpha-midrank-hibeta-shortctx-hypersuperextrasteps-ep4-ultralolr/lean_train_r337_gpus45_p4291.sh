#!/usr/bin/env bash
# p4291: R1137 ShortCtx MidRank Hiβ Hyper MidLR REFUTE m=+0.001233 SE=0.001923 ~0.32× thought✓166 B✓0.354 k=3 → UltraLoLR isolate
# → UltraLoLR isolate (lr →5e-7); ShortCtx MidRank Hiβ Hyper UltraLoLR
# Fill r337 GPUs 4,5 after exact-PID reap. Never pkill -f. Do not touch teacher/king.
set -euo pipefail
exec >/root/logs/r1172_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1172-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5 ShortCtx MidRank Hiβ Hyper UltraLoLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1172-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-hypersuperextrasteps-ep4-ultralolr
mkdir -p /root/r1172 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/s4-h1-sft /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1172/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1137/dpo_duel_reason.jsonl ]]; then cp -f /root/r1137/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1172-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1172_train.pid ]]; then
  old=$(cat /root/logs/r1172_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then echo FATAL alive; exit 1; fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r1172-lean] wait VRAM used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r1172/train; mkdir -p /root/r1172/train
rm -f /root/logs/r1172_train.done /root/logs/r1172_merge.done /root/logs/r1172_merge_ready
: >/root/logs/r1172_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1172/train \
  --max-len 6144 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.3 \
  --max-steps 38400 >/root/logs/r1172_train.nohup 2>&1 &
echo $! | tee /root/logs/r1172_train.pid >/root/r1172/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "axis":"r1172-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-hypersuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695","lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.3,
 "max_len":6144,"epochs":4,"max_steps":38400,"gpus":"4,5","chall_port":8003,
 "parent_signal":"R1150 ShortCtx MidRank Midβ UltraLoLR REFUTE m=+0.001358 ~0.31×; MidLR R1137 already REFUTE → Hiβ isolate ShortCtx MidRank UltraLoLR",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"}
Path("/root/affine_data/r1172_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1172-lean] TRAIN pid=$(cat /root/logs/r1172_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1172_train_then_merge_p4291.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1172_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1172_merge_then_n80_p4291.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1172_merge_then_n80.pid
echo "[r1172-lean] waiters armed"
