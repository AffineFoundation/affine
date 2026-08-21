#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4321_r1207_merge_then_n80.nohup
exec >>"$LOG" 2>&1
echo "[r1207-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"
while true; do
  if [[ -f /root/logs/r1207_merge_ready ]] && [[ -f /tmp/r1207_merged/config.json ]]; then
    n=$(ls /tmp/r1207_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ "${n:-0}" -ge 16 ]] && break
  fi
  sleep 20
done
[[ -f /tmp/r1207_merged/config.json ]] || exit 1
echo "[r1207-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE_READY shards=$n"
LEAN=/root/mining_src/r1207-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-midlr/lean_chall_n80_r252_gpus6_p4321.sh
if [[ -x "$LEAN" ]]; then
  exec bash "$LEAN"
else
  echo "[r1207-n80] no lean_chall yet; merge ready for next pass"
  exit 0
fi
