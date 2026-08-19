#!/usr/bin/env bash
# p4030: R926 cryptoDev SoftCtx MidLoβ=0.05 Soft Mid Mid Soft UltraLoLR on H100 GPUs 0,1
set -euo pipefail
exec >/root/logs/r926_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r926-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 0,1 MidLoβ SoftCtx"
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
EXP=r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r926 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r926/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r926-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
test -f /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r926_train.pid ]]; then
  old=$(cat /root/logs/r926_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r926 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '{s+=$1} END{print s+0}')
  echo "[r926-lean] wait VRAM0+1 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 0,1 busy"; exit 1; }
rm -rf /root/r926/train; mkdir -p /root/r926/train
rm -f /root/logs/r926_train.done /root/logs/r926_merge.done
: >/root/logs/r926_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r926/train \
  --max-len 12288 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r926_train.nohup 2>&1 &
echo $! | tee /root/logs/r926_train.pid >/root/r926/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr",
 "base":"cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.05,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"0,1",
 "parent_signal":"R896 SoftCtx Loβ ~0.49× → MidLoβ SoftCtx isolate; R915 TTL-lost → H100 restart; ≠ Loβ R896 / ≠ Midβ R874 / ≠ Hiβ R897 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r926_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r926-lean] TRAIN launched pid=$(cat /root/logs/r926_train.pid) BASE=$BASE"
