#!/usr/bin/env bash
# p4258: R1115 ShortCtx MidRank Hiβ Hyper HiLR REFUTE m=-0.017658 ~-1.58× thought✓253 B✓0.4625 k=3
# → cryptoDev MidCtx MidRank MidLoβ Hyper HiLR isolate (ctx Short→Mid @8192; β 0.3→0.05; keep Hyper HiLR).
# ≠ ShortCtx Hiβ Hyper HiLR R1115 / ≠ ShortCtx Hiβ Hyper MidLR R1098 / ≠ SoftCtx Midβ Mega HiLR R1060 /
# ≠ SoftCtx Midβ Mega MidLR R1051 / ≠ MidCtx MidLoβ Mega UltraLoLR R927 / ≠ vera MidCtx MidLoβ Hyper HiLR R1112 /
# ≠ Online / ≠ GRPO.
# Fill r926 GPUs 3,4 after exact-PID reap of R1115 chall :8002. Never pkill -f.
# Do not touch teacher:8000 / king:8001.
set -euo pipefail
exec >/root/logs/r1130_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=3,4 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1130-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 3,4 MidCtx MidRank MidLoβ Hyper HiLR"
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
EXP=r1130-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr
mkdir -p /root/r1130 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1130/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r927/dpo_duel_reason.jsonl ]]; then cp -f /root/r927/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r927-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r927-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing MidCtx data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1130-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1130_train.pid ]]; then
  old=$(cat /root/logs/r1130_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1130 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[r1130-lean] wait VRAM3+4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 3,4 busy"; exit 1; }
rm -rf /root/r1130/train; mkdir -p /root/r1130/train
rm -f /root/logs/r1130_train.done /root/logs/r1130_merge.done /root/logs/r1130_merge_ready
: >/root/logs/r1130_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1130/train \
  --max-len 8192 --epochs 4 --lr 2e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.05 \
  --max-steps 38400 >/root/logs/r1130_train.nohup 2>&1 &
echo $! | tee /root/logs/r1130_train.pid >/root/r1130/train.pid
python3 - <<'PYMETA'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r1130-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-hilr",
 "base":"cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0",
 "lr":"2e-6","lora_r":32,"lora_alpha":128,"beta":0.05,"max_len":8192,"epochs":4,"max_steps":38400,
 "gpus":"3,4","chall_port":8002,
 "parent_signal":"R1115 ShortCtx MidRank Hiβ Hyper HiLR REFUTE m=-0.017658 ~-1.58× thought✓253 B✓0.4625 k=3 → MidCtx+MidLoβ isolate",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1130_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PYMETA
echo "[r1130-lean] TRAIN launched pid=$(cat /root/logs/r1130_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r1130_train_then_merge_p4258.sh >/root/logs/r1130_wait_merge.nohup 2>&1 &
echo $! >/root/logs/r1130_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1130_merge_then_n80_p4258.sh >/root/logs/p4258_r1130_merge_then_n80.nohup 2>&1 &
echo $! >/root/logs/r1130_merge_then_n80.pid
echo "[r1130-lean] waiters armed"
