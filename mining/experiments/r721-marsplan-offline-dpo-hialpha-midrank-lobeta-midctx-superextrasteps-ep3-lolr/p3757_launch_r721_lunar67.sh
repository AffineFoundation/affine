#!/usr/bin/env bash
# p3757 host→lunar: reap idle r537 chall on 6,7 by pidfile/pid, free old merges, launch R721 TRAIN+wait→merge.
# Never pkill -f.
set -euo pipefail
EXP=r721-marsplan-offline-dpo-hialpha-midrank-lobeta-midctx-superextrasteps-ep3-lolr
log() { echo "[p3757-r721-launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

log "reap leftover r537 chall on GPUs 6,7 (idle ~1.4d)"
CHALL_PID=""
if [[ -f /root/logs/vllm_chall_r537.pid ]]; then
  CHALL_PID=$(cat /root/logs/vllm_chall_r537.pid 2>/dev/null || true)
fi
if [[ -z "${CHALL_PID:-}" || ! "$CHALL_PID" =~ ^[0-9]+$ ]]; then
  CHALL_PID=$(ss -tlnp 2>/dev/null | grep ':8002 ' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true)
fi
if [[ -n "${CHALL_PID:-}" && "$CHALL_PID" =~ ^[0-9]+$ ]] && kill -0 "$CHALL_PID" 2>/dev/null; then
  cmd=$(tr '\0' ' ' </proc/"$CHALL_PID"/cmdline 2>/dev/null || true)
  case "$cmd" in
    *r537_merged*|*vllm*8002*)
      log "kill r537 chall pid=$CHALL_PID"
      kill "$CHALL_PID" 2>/dev/null || true
      for i in $(seq 1 30); do
        kill -0 "$CHALL_PID" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$CHALL_PID" 2>/dev/null; then
        log "SIGKILL r537 pid=$CHALL_PID"
        kill -9 "$CHALL_PID" 2>/dev/null || true
      fi
      ;;
    *)
      log "SKIP kill pid=$CHALL_PID cmd does not look like r537 chall: $cmd"
      ;;
  esac
else
  log "no live r537 chall pid found"
fi

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  log "VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy used=$used"; exit 1; }

for d in /tmp/r222_merged /tmp/r336_merged /tmp/r439_merged /tmp/r537_merged; do
  if [[ -d "$d" ]]; then
    log "free $d"
    rm -rf "$d"
  fi
done
df -h /tmp | tail -1 | tee /root/logs/p3757_r721_disk.txt || true

mkdir -p /root/mining_src/$EXP /root/logs /root/affine_data /root/r721
chmod +x /root/mining_src/$EXP/*.sh

rm -f /root/logs/r721_merge_launched.p3757
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus67_p3757.sh >/root/logs/r721_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r721_lean_outer.pid
log "lean outer pid=$(cat /root/logs/r721_lean_outer.pid)"

for i in $(seq 1 120); do
  if [[ -f /root/logs/r721_train.pid ]]; then
    tpid=$(cat /root/logs/r721_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "train alive pid=$tpid — arm wait→merge"
      nohup bash /root/mining_src/$EXP/wait_r721_train_then_merge_p3757.sh >/root/logs/r721_wait.nohup 2>&1 &
      echo $! >/root/logs/r721_wait.pid
      log "wait pid=$(cat /root/logs/r721_wait.pid)"
      python3 - <<'PY'
import json
from pathlib import Path
p=Path("/root/affine_data/r721_train_launched.json")
if p.exists():
  d=json.loads(p.read_text())
  print("META", {k:d.get(k) for k in ("axis","base","beta","max_len","max_steps","gpus","pid")})
  b=d.get("base","")
  assert "marsplan" in b or "5gedzafcvg" in b, b
else:
  print("META pending")
PY
      log "R721 TRAIN ARMED on lunar 6,7"
      exit 0
    fi
  fi
  sleep 2
done
log "FATAL train pid never appeared"
tail -n 40 /root/logs/r721_lean_warm.log || true
tail -n 40 /root/logs/r721_lean_outer.nohup || true
exit 1
