#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p4128_r924_reap_launch.nohup
: >"$LOG"
log(){ echo "[p4128] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
log "START R984 REFUTE→reap chall + R985 n80 + R1001 TRAIN"

# Exact-PID reap R984 chall (never pkill -f)
CHALL_PID=$(cat /root/logs/vllm_chall_r984.pid 2>/dev/null || true)
SIM_PID=$(cat /root/logs/r984_sim_wvk7.pid 2>/dev/null || true)
for p in $SIM_PID $CHALL_PID 52122; do
  [[ -n "${p:-}" && "$p" =~ ^[0-9]+$ ]] || continue
  if kill -0 "$p" 2>/dev/null; then
    log "kill pid=$p"
    kill "$p" 2>/dev/null || true
  fi
done
for p in $(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true); do
  if kill -0 "$p" 2>/dev/null; then
    log "kill :8002 pid=$p"
    kill "$p" 2>/dev/null || true
  fi
done
sleep 3
for p in $SIM_PID $CHALL_PID 52122; do
  [[ -n "${p:-}" && "$p" =~ ^[0-9]+$ ]] || continue
  kill -9 "$p" 2>/dev/null || true
done
for p in $(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true); do
  kill -9 "$p" 2>/dev/null || true
done
rm -f /root/logs/vllm_chall_r984.pid /root/logs/r984_sim_wvk7.pid
log "reaped"

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  log "wait GPUs1+3 used_mib=$used iter=$i"
  [[ "$used" -lt 2000 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 2000 ]] || { log "FATAL GPUs1+3 still busy"; exit 1; }

used45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
log "GPUs4+5 used_mib=$used45"
[[ "$used45" -lt 2000 ]] || { log "FATAL GPUs4+5 busy"; exit 1; }
test -f /tmp/r985_merged/config.json
n=$(ls /tmp/r985_merged/model-*-of-*.safetensors | wc -l)
log "r985 merge shards=$n"
[[ "$n" -ge 16 ]] || { log "FATAL merge incomplete"; exit 1; }

chmod +x /root/mining_src/fleet-rent/lean_chall_n80_r924_r985_gpus45_p4128.sh \
  /root/mining_src/r1001-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr/*.sh

nohup bash /root/mining_src/fleet-rent/lean_chall_n80_r924_r985_gpus45_p4128.sh \
  >/root/logs/p4128_r985_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p4128_r985_lean_outer.pid
log "R985 lean outer pid=$(cat /root/logs/p4128_r985_lean_outer.pid)"

nohup bash /root/mining_src/r1001-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr/lean_train_r924_gpus13_p4128.sh \
  >/root/logs/p4128_r1001_outer.nohup 2>&1 &
echo $! >/root/logs/p4128_r1001_outer.pid
log "R1001 outer pid=$(cat /root/logs/p4128_r1001_outer.pid)"

date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4128_r984_refute_r985_n80_r1001_armed.done
log "ARMED done"
