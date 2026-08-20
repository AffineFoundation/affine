#!/usr/bin/env bash
# p4189: R1060 auto-arm lean chall+v4 n80 when merge_ready. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4189_r1060_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4189-r1060-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"

for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1060_merge_ready ]] || [[ -f /root/logs/r1060_merge.done ]]; then
    if [[ -d /tmp/r1060_merged ]] && [[ -f /tmp/r1060_merged/config.json ]]; then
      n=$(ls /tmp/r1060_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ "${n:-0}" -ge 16 ]]; then
        echo "[p4189-r1060-n80] merge ready shards=$n iter=$i"
        break
      fi
    fi
  fi
  sleep 30
done

[[ -f /tmp/r1060_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
n=$(ls /tmp/r1060_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo "FATAL shards=$n"; exit 1; }

for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1060_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then
    echo "[p4189-r1060-n80] train still alive pid=$tp iter=$i"; sleep 30; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[p4189-r1060-n80] wait free GPUs3+4 used_mib=$used iter=$i"
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4189-r1060-n80] TK warm → lean_chall"

LEAN=/root/mining_src/r1060-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-hilr/lean_chall_n80_r926_gpus34_p4189.sh
[[ -x "$LEAN" ]] || chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4189_r1060_merge_then_n80_armed.done
echo "[p4189-r1060-n80] DONE lean launched"
