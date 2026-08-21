#!/usr/bin/env bash
# p4341: R1207 SoftCtx HiRank MidLoβ Mega MidLR REFUTE m=+0.001811 ~0.44× (HiRank MidLoβ Mega LR exhausted R1020/R1040/R1207)
# → SoftCtx LoRank MidLoβ Mega UltraLoLR isolate (R1202 MidLR REFUTE ~−0.50×; R1229 HiLR TRAIN; Ultra never).
# Never pkill -f. Leave T:8000 K:8001 and R1232 TRAIN on GPUs 4,5.
set -euo pipefail
exec >/root/logs/r1241_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1241-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7 SoftCtx LoRank MidLoβ Mega UltraLoLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1241-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r1241 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/s4-h1-sft /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1241/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1229/dpo_duel_reason.jsonl ]]; then cp -f /root/r1229/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1207/dpo_duel_reason.jsonl ]]; then cp -f /root/r1207/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1202/dpo_duel_reason.jsonl ]]; then cp -f /root/r1202/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1241-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1241_train.pid ]]; then
  old=$(cat /root/logs/r1241_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then echo FATAL alive; exit 1; fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r1241-lean] wait VRAM used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r1241/train; mkdir -p /root/r1241/train
rm -f /root/logs/r1241_train.done /root/logs/r1241_merge.done /root/logs/r1241_merge_ready
: >/root/logs/r1241_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1241/train \
  --max-len 12288 --epochs 4 --lr 5e-7 \
  --lora-r 16 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r1241_train.nohup 2>&1 &
echo $! | tee /root/logs/r1241_train.pid >/root/r1241/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "axis":"r1241-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695","lr":"5e-7","lora_r":16,"lora_alpha":128,"beta":0.05,
 "max_len":12288,"epochs":4,"max_steps":19200,"gpus":"6,7","chall_port":8003,
 "parent_signal":"R1207 SoftCtx HiRank MidLoβ Mega MidLR REFUTE m=+0.001811 ~0.44x (HiRank MidLoβ Mega LR exhausted) + R1202 Soft LoRank MidLoβ Mega MidLR ~-0.50x / R1229 HiLR TRAIN → SoftCtx LoRank MidLoβ Mega UltraLoLR isolate",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"}
Path("/root/affine_data/r1241_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1241-lean] TRAIN pid=$(cat /root/logs/r1241_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1241_train_then_merge_p4341.sh >/root/logs/r1241_wait_merge.nohup 2>&1 &
echo $! >/root/logs/r1241_wait_merge.pid
echo "[r1241-lean] merge waiter armed"
