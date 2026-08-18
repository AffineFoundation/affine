#!/usr/bin/env bash
# p3911: R338 online-DPO TRAIN_DONE → merge only (no local n80). Same rationale as R337.
set -euo pipefail
log() { echo "[p3911-r338-merge] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a /root/logs/p3911_r338_merge_only.log; }
source /root/venv/bin/activate
[[ -f /root/mine.env ]] && { set -a; source /root/mine.env; set +a; }
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
ADAPTER=/root/r338/train/adapter
MERGED=/tmp/r338_merged
TRAIN_PID_FILE=/root/logs/r338_train.pid
LAUNCHED=/root/logs/r338_merge_only_launched.p3911
mkdir -p /root/logs /root/affine_data
[[ -f "$LAUNCHED" && -f /root/logs/r338_merge.done ]] && { log "already merged"; exit 0; }
log "armed wait TRAIN → merge-only (skip local n80)"
while true; do
  train_alive=0
  if [[ -f "$TRAIN_PID_FILE" ]]; then
    tpid=$(cat "$TRAIN_PID_FILE" 2>/dev/null || true)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then train_alive=1; fi
  fi
  adapter_ok=0
  [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] && adapter_ok=1
  if [[ "$train_alive" -eq 0 && "$adapter_ok" -eq 1 ]]; then
    log "TRAIN_DONE — merge LoRA → $MERGED"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r338_train.done
    rm -rf "$MERGED"
    export CUDA_VISIBLE_DEVICES=6,7
    python /root/mining_src/s4-h1-sft/merge_lora.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED"
    n=$(ls "$MERGED"/model-*-of-*.safetensors 2>/dev/null | wc -l)
    [[ -f "$MERGED/config.json" && "$n" -ge 16 ]] || { log "FATAL merge shards=$n"; exit 1; }
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r338_merge.done
    log "MERGE_DONE shards=$n — host-relay n80 vs reign35 (no local chall)"
    exit 0
  fi
  step=$(grep -o '"step": [0-9]*' /root/logs/r338_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${step:-?}"
  sleep 30
done
