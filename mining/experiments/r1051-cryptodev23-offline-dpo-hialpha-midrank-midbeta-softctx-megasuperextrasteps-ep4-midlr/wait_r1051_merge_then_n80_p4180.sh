#!/usr/bin/env bash
# p4180: R1051 auto-arm lean chall+v4 n80 when merge_ready. Never pkill -f.
set -euo pipefail
LOG=/root/logs/p4180_r1051_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4180-r1051-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"

for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1051_merge_ready ]] || [[ -f /root/logs/r1051_merge.done ]]; then
    if [[ -d /tmp/r1051_merged ]] && [[ -f /tmp/r1051_merged/config.json ]]; then
      n=$(ls /tmp/r1051_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      # cryptoDev may have more shards
      if [[ "${n:-0}" -ge 16 ]]; then
        echo "[p4180-r1051-n80] merge ready shards=$n iter=$i"
        break
      fi
    fi
  fi
  sleep 30
done

[[ -f /tmp/r1051_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
n=$(ls /tmp/r1051_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo "FATAL shards=$n"; exit 1; }

for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1051_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then
    echo "[p4180-r1051-n80] train still alive pid=$tp iter=$i"; sleep 30; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
  echo "[p4180-r1051-n80] wait free GPUs5+6 used_mib=$used iter=$i"
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4180-r1051-n80] TK warm → lean_chall"

LEAN=/root/mining_src/r1051-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-midlr/lean_chall_n80_r926_gpus56_p4180.sh
[[ -x "$LEAN" ]] || chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4180_r1051_merge_then_n80_armed.done
echo "[p4180-r1051-n80] DONE lean launched"
