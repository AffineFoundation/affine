#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r1184_merge_then_n80.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r1184-n80wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait merge_ready"
for i in $(seq 1 1200); do
  [[ -f /root/logs/r1184_merge_ready || -f /root/logs/r1184_merge.done ]] && break
  sleep 30
done
[[ -f /root/logs/r1184_merge_ready || -f /root/logs/r1184_merge.done ]] || { echo FATAL no merge; exit 1; }
echo "[r1184-n80wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) launch lean_chall"
bash /root/mining_src/r1184-vera-offline-dpo-hialpha-midrank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r339_gpus45_p4299.sh
