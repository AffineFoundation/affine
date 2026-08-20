#!/usr/bin/env bash
# p4169: R1039 wait → lean chall+v4 n80 on r338 GPUs 6,7 :8002. Never pkill -f. Do not touch TK or R1038 on 4,5.
set -euo pipefail
LOG=/root/logs/p4169_r1039_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4169-r1039-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"

for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1039_merge_ready ]] || [[ -f /root/logs/r1039_merge.done ]]; then
    if [[ -d /tmp/r1039_merged ]] && [[ -f /tmp/r1039_merged/config.json ]]; then
      n=$(ls /tmp/r1039_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ "${n:-0}" -ge 16 ]]; then
        echo "[p4169-r1039-n80] merge ready shards=$n iter=$i"
        break
      fi
    fi
  fi
  sleep 30
done

[[ -f /tmp/r1039_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
n=$(ls /tmp/r1039_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo "FATAL shards=$n"; exit 1; }

for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1039_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then
    echo "[p4169-r1039-n80] train still alive pid=$tp iter=$i"; sleep 30; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4169-r1039-n80] wait free GPUs6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4169-r1039-n80] TK warm → lean_chall"

LEAN=/root/mining_src/r1039-vera-offline-dpo-hialpha-midrank-hibeta-softctx-ultrasuperextrasteps-ep4-midlr/lean_chall_n80_r338_gpus67_p4169.sh
[[ -x "$LEAN" ]] || chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4169_r1039_merge_then_n80_armed.done
echo "[p4169-r1039-n80] DONE lean launched"
