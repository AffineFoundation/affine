#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4327_r1219_merge_then_n80.nohup
exec >>"$LOG" 2>&1
echo "[r1219-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"
while true; do
  if [[ -f /root/logs/r1219_merge_ready ]] && [[ -f /tmp/r1219_merged/config.json ]]; then
    n=$(ls /tmp/r1219_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ "${n:-0}" -ge 16 ]] && break
  fi
  sleep 20
done
echo "[r1219-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE_READY shards=$n (lean_chall next pass)"
exit 0
