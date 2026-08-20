#!/usr/bin/env bash
# p4152: R998 already MERGE_DONE — launch lean chall+n80 immediately on GPUs 6,7.
set -euo pipefail
LOG=/root/logs/p4152_r998_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4152-r998-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START"
[[ -f /tmp/r998_merged/config.json ]] || { echo FATAL no merge; exit 1; }
n=$(ls /tmp/r998_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo "FATAL shards=$n"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
echo "[p4152-r998-n80] GPUs6+7 used_mib=$used shards=$n"
[[ "$used" -lt 40960 ]] || { echo "FATAL GPUs busy"; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
LEAN=/root/mining_src/r998-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus67_p4152.sh
[[ -x "$LEAN" ]] || chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4152_r998_merge_then_n80_armed.done
echo "[p4152-r998-n80] DONE lean finished"
