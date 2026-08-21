#!/usr/bin/env bash
# p4348: wait R1228 MERGE_DONE then arm lean chall+n80 on GPUs 6,7 :8003
set -euo pipefail
LOG=/root/logs/p4348_r1228_wait_arm.nohup
exec >>"$LOG" 2>&1
echo "[p4348-r1228-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait merge.done"
while [[ ! -f /root/logs/r1228_merge.done ]]; do sleep 15; done
# also wait merge hub complete
for i in $(seq 1 120); do
  n=$(ls /tmp/r1228_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f /tmp/r1228_merged/config.json && "${n:-0}" -ge 16 ]] && break
  echo "[p4348-r1228-wait] hub incomplete n=$n poll=$i"
  sleep 10
done
n=$(ls /tmp/r1228_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ -f /tmp/r1228_merged/config.json && "${n:-0}" -ge 16 ]] || { echo "FATAL merge incomplete"; exit 1; }
# wait GPUs 6,7 free after merge process exits
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  echo "[p4348-r1228-wait] gpu67 used=$used poll=$i"
  [[ "${used:-999999}" -lt 2000 ]] && break
  sleep 5
done
SCR=/root/mining_src/r1228-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-megasuperextrasteps-ep4-hilr/lean_chall_n80_r339_r1228_gpus67_p4348.sh
chmod +x "$SCR"
echo "[p4348-r1228-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) arm $SCR"
nohup bash "$SCR" >/root/logs/p4348_r1228_outer.nohup 2>&1 &
echo $! >/root/logs/p4348_r1228_outer.pid
echo "[p4348-r1228-wait] outer pid=$(cat /root/logs/p4348_r1228_outer.pid)"
