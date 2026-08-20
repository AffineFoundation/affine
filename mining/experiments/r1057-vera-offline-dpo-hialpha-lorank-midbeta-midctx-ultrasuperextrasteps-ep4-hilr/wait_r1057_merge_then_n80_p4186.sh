#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4186_r1057_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4186-r1057-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"
for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1057_merge_ready ]] || [[ -f /root/logs/r1057_merge.done ]]; then
    if [[ -d /tmp/r1057_merged ]] && [[ -f /tmp/r1057_merged/config.json ]]; then
      n=$(ls /tmp/r1057_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ "${n:-0}" -ge 16 ]]; then
        echo "[p4186-r1057-n80] merge ready shards=$n iter=$i"; break
      fi
    fi
  fi
  sleep 30
done
[[ -f /tmp/r1057_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
n=$(ls /tmp/r1057_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo "FATAL shards=$n"; exit 1; }
for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1057_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then
    echo "[p4186-r1057-n80] train still alive pid=$tp iter=$i"; sleep 30; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4186-r1057-n80] wait free GPUs1,3 used_mib=$used iter=$i"
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
LEAN="/root/mining_src/r1057-vera-offline-dpo-hialpha-lorank-midbeta-midctx-ultrasuperextrasteps-ep4-hilr/lean_chall_n80_crown_r1057_gpus13_p4186.sh"
[[ -x "$LEAN" ]] || chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4186_r1057_merge_then_n80_armed.done
echo "[p4186-r1057-n80] DONE lean launched"
