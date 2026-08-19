#!/usr/bin/env bash
# p4043: R924 TRAIN_DONE (900 steps) but wait_merge FATAL'd early — looked for
# adapter_model at /root/r924/train/ (flat) while peft writes …/train/adapter/.
# Also waited on /root/logs/r924_train.done while train wrote …/train/train.done.
# Re-merge on idle GPUs 0,1 with correct adapter path; n80 needs host-relay (no local TK).
set -euo pipefail
exec >>/root/logs/p4043_r924_merge.nohup 2>&1
source /root/venv/bin/activate
set -a; source /root/mine.env 2>/dev/null || true; set +a
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
ADAPTER=/root/r924/train/adapter
MERGED=/tmp/r924_merged
log() { echo "[p4043-r924] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
log "START merge"
test -f "$ADAPTER/adapter_model.safetensors" || { log "FATAL no adapter"; exit 1; }
test -f "$BASE/config.json" || { log "FATAL no base"; exit 1; }
# mark train.done in the log path waiters expect
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r924_train.done
cp -f /root/r924/train/train.done /root/logs/r924_train.done 2>/dev/null || true
for i in $(seq 1 60); do
  used=$(nvidia-smi -i 0,1 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  [[ "${used:-999999}" -lt 4096 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 0,1 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999999}" -lt 8192 ]] || { log "FATAL GPUs 0,1 busy used=$used"; exit 1; }
export CUDA_VISIBLE_DEVICES=0,1
rm -rf "$MERGED"; mkdir -p "$MERGED"
log "MERGE → $MERGED"
/root/venv/bin/python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter "$ADAPTER" --out "$MERGED" \
  >/root/logs/r924_merge.nohup 2>&1
nm=$(ls "$MERGED"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ -f "$MERGED/config.json" && "${nm:-0}" -ge 2 ]] || {
  log "FATAL merge incomplete shards=$nm"; tail -n 60 /root/logs/r924_merge.nohup; exit 1
}
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r924_merge.done
log "MERGE_DONE shards=$nm — next: host-relay n80 to crown TK"
