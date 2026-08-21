#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4232_r1101_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
EXP=r1101-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-hilr
echo "[p4232-r1101-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"
for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1101_merge_ready ]] || [[ -f /root/logs/r1101_merge.done ]]; then
    if [[ -d /tmp/r1101_merged ]] && [[ -f /tmp/r1101_merged/config.json ]]; then
      n=$(ls /tmp/r1101_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ "${n:-0}" -ge 16 ]]; then echo "[p4232-r1101-n80] merge ready shards=$n iter=$i"; break; fi
    fi
  fi
  sleep 30
done
[[ -f /tmp/r1101_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
n=$(ls /tmp/r1101_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo "FATAL shards=$n"; exit 1; }
for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1101_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then echo "[p4232-r1101-n80] train still alive pid=$tp iter=$i"; sleep 30; continue; fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4232-r1101-n80] wait free GPUs6,7 used_mib=$used iter=$i"
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
# p4256 fix: p4232 waiter pointed at missing r338_gpus45 script; real lean is crown GPUs6,7
LEAN=/root/mining_src/$EXP/lean_chall_n80_crown_r1101_gpus67_p4232.sh
chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4232_r1101_merge_then_n80_armed.done
echo "[p4232-r1101-n80] DONE lean launched"
