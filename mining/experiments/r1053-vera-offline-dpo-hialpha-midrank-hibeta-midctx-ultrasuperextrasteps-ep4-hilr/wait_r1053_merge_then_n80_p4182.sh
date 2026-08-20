#!/usr/bin/env bash
# p4182: R1053 wait → lean chall+v4 n80 on r339 GPUs 4,5 :8002. Never pkill -f. Leave R1052 on 6,7 :8003 alone.
set -euo pipefail
LOG=/root/logs/p4182_r1053_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4182-r1053-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"
for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1053_merge_ready ]] || [[ -f /root/logs/r1053_merge.done ]]; then
    if [[ -d /tmp/r1053_merged ]] && [[ -f /tmp/r1053_merged/config.json ]]; then
      n=$(ls /tmp/r1053_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ "${n:-0}" -ge 16 ]]; then echo "[p4182-r1053-n80] merge ready shards=$n"; break; fi
    fi
  fi
  sleep 30
done
[[ -f /tmp/r1053_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1053_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then echo train alive; sleep 30; continue; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
LEAN=/root/mining_src/r1053-vera-offline-dpo-hialpha-midrank-hibeta-midctx-ultrasuperextrasteps-ep4-hilr/lean_chall_n80_r339_gpus45_p4182.sh
chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4182_r1053_merge_then_n80_armed.done
