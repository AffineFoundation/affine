#!/usr/bin/env bash
# p4065: R925 HiRank MidLoβ MidCtx MegaExtra REFUTE m≈0 → SoftCtx HiRank MidLoβ UltraSuperExtra isolate
# ≠ MidCtx HiRank MidLoβ R925 / ≠ SoftCtx MidLoβ MidRank UltraExtra R942 / ≠ SoftCtx MidLoβ MidRank MegaExtra R941 / ≠ SoftCtx MidLoβ HiRank MegaExtra R931 / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r951_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r951-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7 SoftCtx HiRank MidLoβ UltraSuperExtra"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r951-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-ultrasuperextrasteps-ep4-ultralolr
mkdir -p /root/r951 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r951/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r942/dpo_duel_reason.jsonl ]]; then cp -f /root/r942/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r951-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r951_train.pid ]]; then
  old=$(cat /root/logs/r951_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r951 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r951-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r951/train; mkdir -p /root/r951/train
rm -f /root/logs/r951_train.done /root/logs/r951_merge.done
: >/root/logs/r951_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r951/train \
  --max-len 12288 --epochs 4 --lr 5e-7 \
  --lora-r 64 --lora-alpha 128 --beta 0.05 \
  --max-steps 28800 >/root/logs/r951_train.nohup 2>&1 &
echo $! | tee /root/logs/r951_train.pid >/root/r951/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r951-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-ultrasuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"5e-7","lora_r":64,"lora_alpha":128,"beta":0.05,"max_len":12288,"epochs":4,"max_steps":28800,
 "gpus":"6,7",
 "parent_signal":"R925 HiRank MidLoβ MidCtx MegaExtra REFUTE m=-2.1e-6 SE=0.000738 bar=0.002 ~-0.001× thought✓164 B✓0.483 → SoftCtx HiRank MidLoβ UltraSuperExtra(28800) isolate; ≠ MidCtx R925 / ≠ MidRank SoftCtx MidLoβ UltraExtra R942 / ≠ MidRank SoftCtx MidLoβ MegaExtra R941 / ≠ SoftCtx MidLoβ HiRank MegaExtra R931 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r951_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r951-lean] TRAIN launched pid=$(cat /root/logs/r951_train.pid) BASE=$BASE"
