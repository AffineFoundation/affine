#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4226_r1096_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4226-r1096-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready + king:8001"
for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1096_merge_ready ]] || [[ -f /root/logs/r1096_merge.done ]]; then
    if [[ -d /tmp/r1096_merged ]] && [[ -f /tmp/r1096_merged/config.json ]]; then
      n=$(ls /tmp/r1096_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ "${n:-0}" -ge 16 ]]; then
        echo "[p4226-r1096-n80] merge ready shards=$n iter=$i"; break
      fi
    fi
  fi
  sleep 30
done
[[ -f /tmp/r1096_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
# ensure king up on GPU5
if ! curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null; then
  echo "[p4226-r1096-n80] king down — start GPU5"
  bash /root/mining_src/r1096-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr/start_king_gpu5_p4226.sh || true
  for j in $(seq 1 180); do
    curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && break
    sleep 10
  done
fi
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4226-r1096-n80] TK warm — lean_chall armed when GPUs1,2 free"
for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1096_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then sleep 30; continue; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done
LEAN=/root/mining_src/r1096-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr/lean_chall_n80_r340_gpus12_p4226.sh
if [[ -x "$LEAN" ]]; then bash "$LEAN"; else echo "WARN no lean_chall yet — next pass"; fi
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4226_r1096_merge_then_n80_armed.done
