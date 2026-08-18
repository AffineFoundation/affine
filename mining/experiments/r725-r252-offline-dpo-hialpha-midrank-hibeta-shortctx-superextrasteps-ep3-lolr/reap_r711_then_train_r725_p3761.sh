#!/usr/bin/env bash
# p3761: R711 REFUTE → reap chall :8003 by pidfile → R725 TRAIN on GPUs 6,7. Never pkill -f.
set -euo pipefail
exec >/root/logs/p3761_r711_reap_r725_launch.log 2>&1
log() { echo "[p3761-r711→r725] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
source /root/venv/bin/activate
mkdir -p /root/logs /root/affine_data

stop_pid() {
  local pid=$1
  local why=${2:-}
  [[ -n "${pid:-}" ]] || return 0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_pidfile() {
  local pidf=$1
  local why=${2:-}
  [[ -f "$pidf" ]] || return 0
  local pid
  pid=$(cat "$pidf" 2>/dev/null || true)
  stop_pid "$pid" "$why"
  rm -f "$pidf"
}

log "R711 REFUTE harvest confirmed — reap chall then launch R725"
# decision already on disk from p3760 sim
if [[ -f /root/affine_data/r711_decision_reign34_wvk7.json ]]; then
  python3 - <<'PY'
import json
d=json.load(open("/root/affine_data/r711_decision_reign34_wvk7.json"))
print(f"R711 wins={d.get('wins')} m={d.get('margin'):.6f} bar={d.get('bar'):.6f} thought={d.get('thought_median')} B={d.get('b_pass'):.3f}")
PY
fi

# stop sim if somehow still alive
stop_pidfile /root/logs/r711_sim_wvk7.pid "r711 sim"
# stop chall vllm by pidfile
stop_pidfile /root/logs/vllm_chall_r711.pid "r711 chall :8003"
# also stop known parent if still listening on 8003
if ss -lntp 2>/dev/null | grep -q ':8003'; then
  pid=$(ss -lntp 2>/dev/null | grep ':8003' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
  stop_pid "$pid" "port 8003 leftover"
fi

# free a few old merges if disk tight — keep r711/r710/r724
kept=0
for d in /tmp/r6*_merged /tmp/r70*_merged /tmp/r707_merged /tmp/r706_merged /tmp/r705_merged /tmp/r675_merged; do
  [[ -d "$d" ]] || continue
  case "$d" in
    */r710_merged|*/r711_merged|*/r724_merged) continue ;;
  esac
  log "free old merge $d"
  rm -rf "$d"
  kept=$((kept+1))
  [[ "$kept" -ge 8 ]] && break
done
log "freed_old=$kept keep=/tmp/r711_merged"

# wait VRAM
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  log "VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 3
done

EXP=r725-r252-offline-dpo-hialpha-midrank-hibeta-shortctx-superextrasteps-ep3-lolr
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_r252_gpus67_p3761.sh >/root/logs/p3761_r725_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3761_r725_lean_outer.pid
log "lean_train outer pid=$(cat /root/logs/p3761_r725_lean_outer.pid)"
# arm wait→merge once train pid appears
for i in $(seq 1 90); do
  if [[ -f /root/logs/r725_train.pid ]]; then
    tpid=$(cat /root/logs/r725_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_LIVE pid=$tpid — arm wait→merge"
      nohup bash /root/mining_src/$EXP/wait_r725_train_then_merge_p3761.sh >/root/logs/p3761_r725_wait.nohup 2>&1 &
      echo $! >/root/logs/p3761_r725_wait.pid
      log "wait pid=$(cat /root/logs/p3761_r725_wait.pid)"
      date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p3761_r711_reap_r725_done
      exit 0
    fi
  fi
  sleep 2
done
log "FATAL train pid never appeared"
exit 1
