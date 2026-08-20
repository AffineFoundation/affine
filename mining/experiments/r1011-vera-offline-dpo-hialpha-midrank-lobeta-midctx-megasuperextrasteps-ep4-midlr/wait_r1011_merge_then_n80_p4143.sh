#!/usr/bin/env bash
# p4143: R1011 wait scripts stop at MERGE — auto-arm lean chall+v4 n80 when merge_ready.
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1004 6,7.
set -euo pipefail
LOG=/root/logs/p4143_r1011_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4143-r1011-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"

for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1011_merge_ready ]] || [[ -f /root/logs/r1011_merge.done ]]; then
    if [[ -d /tmp/r1011_merged ]] && [[ -f /tmp/r1011_merged/config.json ]]; then
      n=$(ls /tmp/r1011_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ "${n:-0}" -ge 16 ]]; then
        echo "[p4143-r1011-n80] merge ready shards=$n iter=$i"
        break
      fi
    fi
  fi
  sleep 30
done

[[ -f /tmp/r1011_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
n=$(ls /tmp/r1011_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo "FATAL shards=$n"; exit 1; }

for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1011_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then
    echo "[p4143-r1011-n80] train still alive pid=$tp iter=$i"; sleep 30; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4143-r1011-n80] wait free GPUs4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4143-r1011-n80] TK warm → lean_chall"

LEAN=/root/mining_src/r1011-vera-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-midlr/lean_chall_n80_r337_gpus45_p4143.sh
[[ -x "$LEAN" ]] || chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4143_r1011_merge_then_n80_armed.done
echo "[p4143-r1011-n80] DONE lean launched"
