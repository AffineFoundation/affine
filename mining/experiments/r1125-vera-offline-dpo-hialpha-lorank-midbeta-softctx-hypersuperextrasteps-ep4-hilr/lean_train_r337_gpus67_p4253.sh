#!/usr/bin/env bash
# p4253: R1106 SoftCtx MidRank Midβ Hyper HiLR REFUTE m=+0.002112 ~0.29× thought✓195 B✓0.477 k=3
# → LoRank isolate (r 32→16); keep SoftCtx Midβ HyperExtra HiLR @12288 steps=38400
# ShortCtx MidRank Midβ Hyper HiLR R1119 already TRAIN on same pod GPUs4,5
# ≠ MidRank SoftCtx R1106 / ≠ HiRank SoftCtx Midβ Hyper HiLR R1118 / ≠ ShortCtx MidRank Midβ Hyper HiLR R1119 /
# ≠ SoftCtx LoRank Hiβ Hyper HiLR R1116 / ≠ MidCtx LoRank Midβ Ultra HiLR R1057 / ≠ Online / ≠ GRPO
# Fill r337 GPUs 6,7 after exact-PID reap of R1106 chall :8002. Never pkill -f. Do not touch R1119 on 4,5.
set -euo pipefail
exec >/root/logs/r1125_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1125-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7 SoftCtx LoRank Midβ Hyper HiLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1125-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-hilr
mkdir -p /root/r1125 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/s4-h1-sft /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1125/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1106/dpo_duel_reason.jsonl ]]; then cp -f /root/r1106/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1083/dpo_duel_reason.jsonl ]]; then cp -f /root/r1083/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1125-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1125_train.pid ]]; then
  old=$(cat /root/logs/r1125_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then echo FATAL alive; exit 1; fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r1125-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r1125/train; mkdir -p /root/r1125/train
rm -f /root/logs/r1125_train.done /root/logs/r1125_merge.done /root/logs/r1125_merge_ready
: >/root/logs/r1125_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1125/train \
  --max-len 12288 --epochs 4 --lr 2e-6 \
  --lora-r 16 --lora-alpha 128 --beta 0.1 \
  --max-steps 38400 >/root/logs/r1125_train.nohup 2>&1 &
echo $! | tee /root/logs/r1125_train.pid >/root/r1125/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "axis":"r1125-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-hilr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695","lr":"2e-6","lora_r":16,"lora_alpha":128,"beta":0.1,
 "max_len":12288,"epochs":4,"max_steps":38400,"gpus":"6,7","chall_port":8002,
 "parent_signal":"R1106 SoftCtx MidRank Midβ Hyper HiLR REFUTE m=+0.002112 ~0.29× thought✓195 B✓0.477 k=3 → LoRank isolate; ShortCtx MidRank Midβ Hyper HiLR R1119 already on GPUs4,5; ≠ MidRank SoftCtx R1106 / ≠ SoftCtx HiRank Midβ Hyper HiLR R1118 / ≠ SoftCtx LoRank Hiβ Hyper HiLR R1116 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"}
Path("/root/affine_data/r1125_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1125-lean] TRAIN pid=$(cat /root/logs/r1125_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1125_train_then_merge_p4253.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1125_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1125_merge_then_n80_p4253.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1125_merge_then_n80.pid
echo "[r1125-lean] waiters armed"
