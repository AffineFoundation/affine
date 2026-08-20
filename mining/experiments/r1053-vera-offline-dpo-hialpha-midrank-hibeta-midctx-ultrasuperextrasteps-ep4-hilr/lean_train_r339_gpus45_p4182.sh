#!/usr/bin/env bash
# p4182: R339 Online-DPO HiRank REFUTE m=+0.001071 ~0.48× → Offline MidCtx MidRank Hiβ Ultra HiLR
# Parent lane: R1017 MidCtx MidRank Hiβ Ultra MidLR REFUTE ~0.08× → HiLR isolate
# ≠ Ultra MidLR R1017 / ≠ Mega MidLR R1007 / ≠ SoftCtx Mega MidLR R1027 / ≠ SoftCtx MidRank Hiβ Ultra MidLR R1039 /
# ≠ SoftCtx MidRank Hiβ Ultra HiLR R1050 / ≠ ShortCtx HiRank Hiβ Ultra MidLR R1052 / ≠ Online R339 / ≠ GRPO
set -euo pipefail
exec >/root/logs/r1053_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1053-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5 MidCtx MidRank Hiβ Ultra HiLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1053-vera-offline-dpo-hialpha-midrank-hibeta-midctx-ultrasuperextrasteps-ep4-hilr
mkdir -p /root/r1053 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1053/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1052/dpo_duel_reason.jsonl ]]; then cp -f /root/r1052/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1053-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1053_train.pid ]]; then
  old=$(cat /root/logs/r1053_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1053 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r1053-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r1053/train; mkdir -p /root/r1053/train
rm -f /root/logs/r1053_train.done /root/logs/r1053_merge.done /root/logs/r1053_merge_ready
: >/root/logs/r1053_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1053/train \
  --max-len 8192 --epochs 4 --lr 2e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.3 \
  --max-steps 28800 >/root/logs/r1053_train.nohup 2>&1 &
echo $! | tee /root/logs/r1053_train.pid >/root/r1053/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r1053-vera-offline-dpo-hialpha-midrank-hibeta-midctx-ultrasuperextrasteps-ep4-hilr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"2e-6","lora_r":32,"lora_alpha":128,"beta":0.3,"max_len":8192,"epochs":4,"max_steps":28800,
 "gpus":"4,5","chall_port":8002,
 "parent_signal":"R339 Online-DPO REFUTE m=+0.001071 ~0.48× + R1017 MidCtx MidRank Hiβ Ultra MidLR ~0.08× → Ultra HiLR isolate",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1053_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1053-lean] TRAIN launched pid=$(cat /root/logs/r1053_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1053_train_then_merge_p4182.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1053_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1053_merge_then_n80_p4182.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1053_merge_then_n80.pid
echo "[r1053-lean] waiters armed"
