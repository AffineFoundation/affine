#!/usr/bin/env bash
set -euo pipefail
exec >/root/logs/r916_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r916-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7 Loβ MidCtx"
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
mkdir -p /root/r916 /root/logs /root/affine_data
test -e "$BASE/config.json"
if [[ ! -s /root/r916/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/r896/dpo_duel_reason.jsonl ]]; then cp -f /root/r896/dpo_duel_reason.jsonl /root/r916/dpo_duel_reason.jsonl
  elif [[ -s /root/r874/dpo_duel_reason.jsonl ]]; then cp -f /root/r874/dpo_duel_reason.jsonl /root/r916/dpo_duel_reason.jsonl
  else echo FATAL missing Soft Mid Mid Soft data; exit 1; fi
fi
DATA=/root/r916/dpo_duel_reason.jsonl
test -s "$DATA"
n=$(wc -l <"$DATA"); echo "[r916-lean] data_lines=$n"; test "$n" -ge 200
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r916_train.pid ]]; then
  old=$(cat /root/logs/r916_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r916 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r916-lean] wait VRAM used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r916/train; mkdir -p /root/r916/train
rm -f /root/logs/r916_train.done /root/logs/r916_merge.done /root/logs/r916_merge_launched.p4018
: >/root/logs/r916_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r916/train \
  --max-len 8192 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.02 \
  --max-steps 19200 >/root/logs/r916_train.nohup 2>&1 &
echo $! | tee /root/logs/r916_train.pid >/root/r916/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r916-cryptodev23-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr",
 "base":"cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.02,"max_len":8192,"epochs":4,"max_steps":19200,
 "gpus":"6,7",
 "parent_signal":"R896 Loβ SoftCtx ~0.49× + R897 Hiβ SoftCtx ~-1.11× → Loβ MidCtx@8192 transfer; ≠ SoftCtx Loβ R896 / ≠ SoftCtx Hiβ R897 / ≠ Midβ SoftCtx R874 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r916_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r916-lean] TRAIN launched pid=$(cat /root/logs/r916_train.pid)"
