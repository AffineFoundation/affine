# p3840: R786 tammy Soft MidRank MidBeta SoftCtx Mega UltraLoLR on crown GPUs 4,5 after R775 REFUTE; leave R776 on 6,7
# p3840: R786 tammy Soft MidRank MidBeta SoftCtx Mega UltraLoLR on crown GPUs 4,5 after R775 REFUTE; leave R776 on 6,7
# p3840: R786 tammy Soft MidRank MidBeta SoftCtx Mega UltraLoLR on crown GPUs 4,5 after R775 REFUTE; leave R776 on 6,7
exec >/root/logs/r786_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r786-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r786-tammy-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r786 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r786/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r775/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r775/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r775/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r775/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx MidBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r786-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r786_train.pid ]]; then
  old=$(cat /root/logs/r786_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r786 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r786-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r786/train; mkdir -p /root/r786/train; : >/root/logs/r786_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R786_LR=5e-7 R786_MAX_STEPS=19200 R786_LORA_R=32 R786_LORA_ALPHA=128 R786_BETA=0.1 R786_MAX_LEN=12288 R786_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r786 DATA=/root/r786/dpo_duel_reason.jsonl LOG=/root/logs/r786_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r786.sh
echo "[r786-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r786_train.pid) BASE=$BASE"
