#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4219_r1085_merge_then_n80.nohup
mkdir -p /root/logs; exec > >(tee -a "$LOG") 2>&1
echo "[p4219-r1085-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"
for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1085_merge_ready || -f /root/logs/r1085_merge.done ]] && [[ -f /tmp/r1085_merged/config.json ]]; then
    n=$(ls /tmp/r1085_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ "${n:-0}" -ge 16 ]] && { echo "merge ready shards=$n"; break; }
  fi
  sleep 30
done
[[ -f /tmp/r1085_merged/config.json ]] || { echo FATAL no merge; exit 1; }
for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1085_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then sleep 30; continue; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 40960 ]] && break; sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
LEAN=/root/mining_src/r1085-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-midlr/lean_chall_n80_r339_gpus45_p4219.sh
chmod +x "$LEAN"; bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4219_r1085_merge_then_n80_armed.done
