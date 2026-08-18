#!/usr/bin/env bash
# p3760: R724 Short MidRank LoBeta ShortCtx SuperExtra on idle R252 GPUs 4,5 after R710 REFUTE
set -euo pipefail
exec >/root/logs/r724_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# hard-pin BASE after mine.env
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
echo "[r724-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
EXP=r724-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-superextrasteps-ep3-lolr
mkdir -p /root/r724 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r724/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r696-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-ultraextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r696-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-ultraextrasteps-ep3-lolr/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Short MidRank LoBeta ShortCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r724-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r724_train.pid ]]; then
  old=$(cat /root/logs/r724_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r724 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r724-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r724/train; mkdir -p /root/r724/train; : >/root/logs/r724_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R724_LR=1e-6 R724_MAX_STEPS=14400 R724_LORA_R=32 R724_LORA_ALPHA=128 R724_BETA=0.02 R724_MAX_LEN=6144 R724_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r724 DATA=/root/r724/dpo_duel_reason.jsonl LOG=/root/logs/r724_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r724.sh
echo "[r724-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r724_train.pid)"
