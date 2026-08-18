#!/usr/bin/env bash
# p3834: wait R761 n80 decision + R762 SCP ready → chall+n80 on R252 GPUs 4,5.
# Never pkill -f. Leave R782 TRAIN on 6,7 alone. Leave teacher/king alone.
set -euo pipefail
LOG=/root/logs/p3834_r762_wait_after_r761.log
: >"$LOG"
log(){ echo "[p3834-r762-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
DEC761=/root/affine_data/r761_decision_reign35_wvk7.json
SIM761=/root/affine_data/r761_sim_result_reign35_wvk7.json
READY=/root/logs/r762_scp_ready.done
LEAN=/root/mining_src/r762-chall/lean_chall_n80_r252_gpus45_p3834.sh
LAUNCHED=/root/logs/r762_chall_n80_launched.p3834
mkdir -p /root/logs /root/affine_data

[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed: wait R761 decision + R762 SCP ready"

for i in $(seq 1 720); do
  scp_ok=0; [[ -f "$READY" ]] && scp_ok=1
  n80_ok=0
  if [[ -f "$DEC761" ]] || [[ -f "$SIM761" ]]; then n80_ok=1; fi
  # Also accept if R761 lean exited
  if [[ "$n80_ok" -eq 0 ]]; then
    if [[ -f /root/logs/r761_reign35_wvk7_pipeline.done ]]; then n80_ok=1; fi
  fi
  if [[ "$scp_ok" -eq 1 && "$n80_ok" -eq 1 ]]; then
    log "gates clear poll=$i scp=1 n80=1"
    break
  fi
  (( i % 20 == 0 )) && log "waiting poll=$i scp=$scp_ok n80=$n80_ok"
  sleep 15
done
[[ -f "$READY" ]] || { log "FATAL timeout SCP"; exit 2; }
n=$(ls /tmp/r762_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 && -f /tmp/r762_merged/config.json ]] || { log "FATAL incomplete merge shards=$n"; exit 3; }
[[ -f "$DEC761" || -f "$SIM761" || -f /root/logs/r761_reign35_wvk7_pipeline.done ]] \
  || { log "FATAL timeout R761 n80"; exit 4; }

# Stop R761 chall by pidfile only
stop_pid() {
  local pid=$1; [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  log "stop pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for pf in /root/logs/vllm_chall_r761.pid /root/logs/r761_sim_wvk7.pid; do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)"
  rm -f "$pf"
done
# Wait GPUs 4,5 free (do not touch 6,7 / R782)
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
nohup bash "$LEAN" >/root/logs/p3834_r762_lean.outer.log 2>&1 &
echo $! >/root/logs/p3834_r762_lean.outer.pid
log "armed R762 lean pid=$(cat /root/logs/p3834_r762_lean.outer.pid)"
