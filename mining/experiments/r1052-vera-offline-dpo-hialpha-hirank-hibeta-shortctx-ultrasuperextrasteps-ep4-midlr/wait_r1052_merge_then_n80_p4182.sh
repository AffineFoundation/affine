#!/usr/bin/env bash
# p4182: R1052 wait → lean chall+v4 n80 on r339 GPUs 6,7 :8003. Never pkill -f.
# Do not touch TK, R339 chall :8002 on GPUs 4,5, or R339 n80.
set -euo pipefail
LOG=/root/logs/p4182_r1052_merge_then_n80.nohup
mkdir -p /root/logs
exec > >(tee -a "$LOG") 2>&1
echo "[p4182-r1052-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) START wait merge_ready"

for i in $(seq 1 2000); do
  if [[ -f /root/logs/r1052_merge_ready ]] || [[ -f /root/logs/r1052_merge.done ]]; then
    if [[ -d /tmp/r1052_merged ]] && [[ -f /tmp/r1052_merged/config.json ]]; then
      n=$(ls /tmp/r1052_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
      if [[ "${n:-0}" -ge 16 ]]; then
        echo "[p4182-r1052-n80] merge ready shards=$n iter=$i"
        break
      fi
    fi
  fi
  sleep 30
done

[[ -f /tmp/r1052_merged/config.json ]] || { echo "FATAL no merge"; exit 1; }
n=$(ls /tmp/r1052_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo "FATAL shards=$n"; exit 1; }

for i in $(seq 1 120); do
  tp=$(cat /root/logs/r1052_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then
    echo "[p4182-r1052-n80] train still alive pid=$tp iter=$i"; sleep 30; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4182-r1052-n80] wait free GPUs6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 40960 ]] && break
  sleep 10
done

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
echo "[p4182-r1052-n80] TK warm → lean_chall :8003"

LEAN=/root/mining_src/r1052-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr/lean_chall_n80_r339_gpus67_p4182.sh
[[ -x "$LEAN" ]] || chmod +x "$LEAN"
bash "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4182_r1052_merge_then_n80_armed.done
echo "[p4182-r1052-n80] DONE lean launched"
