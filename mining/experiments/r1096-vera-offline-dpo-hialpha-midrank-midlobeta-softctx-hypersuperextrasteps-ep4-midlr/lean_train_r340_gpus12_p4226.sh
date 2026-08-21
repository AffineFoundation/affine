#!/usr/bin/env bash
# p4226: fill idle r340 GPUs1,2 — SoftCtx MidRank MidLoβ Hyper MidLR
# ≠ SoftCtx HiRank MidLoβ Hyper R1090 / ≠ SoftCtx MidRank Midβ Hyper R1083 / ≠ MidCtx MidLoβ Hyper R1095 / ≠ Online R340 / ≠ GRPO
set -euo pipefail
exec >/root/logs/r1096_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1,2 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1096-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 1,2 SoftCtx MidRank MidLoβ HyperExtra MidLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1096-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr
mkdir -p /root/r1096 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1096/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1096-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1096_train.pid ]]; then
  old=$(cat /root/logs/r1096_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1096 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2 | awk '{s+=$1} END{print s+0}')
  echo "[r1096-lean] wait VRAM1+2 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 1,2 busy"; exit 1; }
rm -rf /root/r1096/train; mkdir -p /root/r1096/train
rm -f /root/logs/r1096_train.done /root/logs/r1096_merge.done /root/logs/r1096_merge_ready
: >/root/logs/r1096_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1096/train \
  --max-len 12288 --epochs 4 --lr 1e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.05 \
  --max-steps 38400 >/root/logs/r1096_train.nohup 2>&1 &
echo $! | tee /root/logs/r1096_train.pid >/root/r1096/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r1096-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"1e-6","lora_r":32,"lora_alpha":128,"beta":0.05,"max_len":12288,"epochs":4,"max_steps":38400,
 "gpus":"1,2",
 "parent_signal":"p4226 fill idle r340 GPUs1,2; SoftCtx MidRank MidLoβ Ultra MidLR R1021 REFUTE ~-0.16x → Hyper MidLR isolate; ≠ SoftCtx HiRank MidLoβ Hyper R1090 / ≠ SoftCtx MidRank Midβ Hyper R1083 / ≠ MidCtx MidLoβ Hyper R1095 / ≠ Online R340 / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1096_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1096-lean] TRAIN launched pid=$(cat /root/logs/r1096_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1096_train_then_merge_p4226.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1096_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1096_merge_then_n80_p4226.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1096_merge_then_n80.pid
echo "[r1096-lean] waiters armed"
