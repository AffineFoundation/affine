#!/usr/bin/env bash
# p3848: wait R768 n80 decision + R780 SCP ready → chall+n80 on R252 GPUs 4,5.
# Never pkill -f. Leave R793 TRAIN on 6,7 alone. Leave teacher/king alone.
set -euo pipefail
LOG=/root/logs/p3848_r780_wait_after_r768.log
: >"$LOG"
log(){ echo "[p3848-r780-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
DEC768=/root/affine_data/r768_decision_reign35_wvk7.json
SIM768=/root/affine_data/r768_sim_result_reign35_wvk7.json
READY=/root/logs/r780_scp_ready.done
LEAN=/root/mining_src/r780-chall/lean_chall_n80_r252_gpus45_p3848.sh
LAUNCHED=/root/logs/r780_chall_n80_launched.p3848
mkdir -p /root/logs /root/affine_data

[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed: wait R768 decision + R780 SCP ready"

for i in $(seq 1 1440); do
  scp_ok=0; [[ -f "$READY" ]] && scp_ok=1
  n80_ok=0
  if [[ -f "$DEC768" ]] || [[ -f "$SIM768" ]]; then n80_ok=1; fi
  if [[ "$n80_ok" -eq 0 ]]; then
    if [[ -f /root/logs/r768_reign35_wvk7_pipeline.done ]]; then n80_ok=1; fi
  fi
  if [[ "$scp_ok" -eq 1 && "$n80_ok" -eq 1 ]]; then
    log "gates clear poll=$i scp=1 n80=1"
    break
  fi
  (( i % 20 == 0 )) && log "waiting poll=$i scp=$scp_ok n80=$n80_ok"
  sleep 15
done
[[ -f "$READY" ]] || { log "FATAL timeout SCP"; exit 2; }
n=$(ls /tmp/r780_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 && -f /tmp/r780_merged/config.json ]] || { log "FATAL incomplete merge shards=$n"; exit 3; }
[[ -f "$DEC768" || -f "$SIM768" || -f /root/logs/r768_reign35_wvk7_pipeline.done ]] \
  || { log "FATAL timeout R768 n80"; exit 4; }

stop_pid() {
  local pid=$1; [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  log "stop pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for pf in /root/logs/vllm_chall_r768.pid /root/logs/r768_sim_wvk7.pid; do
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
nohup bash "$LEAN" >/root/logs/p3848_r780_lean.outer.log 2>&1 &
echo $! >/root/logs/p3848_r780_lean.outer.pid
log "armed R780 lean pid=$(cat /root/logs/p3848_r780_lean.outer.pid)"
