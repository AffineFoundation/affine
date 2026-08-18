#!/usr/bin/env bash
# p3848: wait R780 n80 decision + R781 SCP ready → chall+n80 on R252 GPUs 4,5.
# Never pkill -f. Leave R793 TRAIN on 6,7 alone. Leave teacher/king alone.
set -euo pipefail
LOG=/root/logs/p3848_r781_wait_after_r780.log
: >"$LOG"
log(){ echo "[p3848-r781-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
DEC780=/root/affine_data/r780_decision_reign35_wvk7.json
SIM780=/root/affine_data/r780_sim_result_reign35_wvk7.json
READY=/root/logs/r781_scp_ready.done
LEAN=/root/mining_src/r781-chall/lean_chall_n80_r252_gpus45_p3848.sh
LAUNCHED=/root/logs/r781_chall_n80_launched.p3848
mkdir -p /root/logs /root/affine_data

[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed: wait R780 decision + R781 SCP ready"

for i in $(seq 1 1440); do
  scp_ok=0; [[ -f "$READY" ]] && scp_ok=1
  n80_ok=0
  if [[ -f "$DEC780" ]] || [[ -f "$SIM780" ]]; then n80_ok=1; fi
  if [[ "$n80_ok" -eq 0 ]]; then
    if [[ -f /root/logs/r780_reign35_wvk7_pipeline.done ]]; then n80_ok=1; fi
  fi
  if [[ "$scp_ok" -eq 1 && "$n80_ok" -eq 1 ]]; then
    log "gates clear poll=$i scp=1 n80=1"
    break
  fi
  (( i % 20 == 0 )) && log "waiting poll=$i scp=$scp_ok n80=$n80_ok"
  sleep 15
done
[[ -f "$READY" ]] || { log "FATAL timeout SCP"; exit 2; }
n=$(ls /tmp/r781_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 && -f /tmp/r781_merged/config.json ]] || { log "FATAL incomplete merge shards=$n"; exit 3; }
[[ -f "$DEC780" || -f "$SIM780" || -f /root/logs/r780_reign35_wvk7_pipeline.done ]] \
  || { log "FATAL timeout R780 n80"; exit 4; }

stop_pid() {
  local pid=$1; [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  log "stop pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for pf in /root/logs/vllm_chall_r780.pid /root/logs/r780_sim_wvk7.pid; do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)"
  rm -f "$pf"
done
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 2000 ]]; then
    log "GPUs 4,5 free used_mib=$used"
    break
  fi
  sleep 2
done

chmod +x "$LEAN"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
nohup bash "$LEAN" >/root/logs/p3848_r781_lean.outer.log 2>&1 &
echo $! >/root/logs/p3848_r781_lean.outer.pid
log "armed R781 lean pid=$(cat /root/logs/p3848_r781_lean.outer.pid)"
