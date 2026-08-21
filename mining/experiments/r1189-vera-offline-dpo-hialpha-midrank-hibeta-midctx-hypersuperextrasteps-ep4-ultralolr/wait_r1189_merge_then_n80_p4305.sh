#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4305_r1189_merge_then_n80.nohup
mkdir -p /root/logs; exec > >(tee -a "$LOG") 2>&1
EXP=r1189-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr
echo "[p4305-r1189-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"
for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1189_merge_ready ]] && [[ -f /tmp/r1189_merged/config.json ]]; then
    n=$(ls /tmp/r1189_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ "${n:-0}" -ge 16 ]] && break
  fi
  sleep 30
done
[[ -f /tmp/r1189_merged/config.json ]] || exit 1
for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1189_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then sleep 30; continue; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 40960 ]] && break; sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
LEAN=/root/mining_src/$EXP/lean_chall_n80_crown_gpus45_p4305.sh
chmod +x "$LEAN"; bash "$LEAN"
