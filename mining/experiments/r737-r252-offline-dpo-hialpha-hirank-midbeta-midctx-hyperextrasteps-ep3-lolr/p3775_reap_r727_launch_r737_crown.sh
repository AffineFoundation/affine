#!/usr/bin/env bash
# p3775: R727 REFUTE → reap chall :8003 by pidfile → R737 TRAIN. Never pkill -f.
set -euo pipefail
exec >/root/logs/p3775_crown_reap_launch.log 2>&1
log() { echo "[p3775-crown] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
source /root/venv/bin/activate
mkdir -p /root/logs /root/affine_data

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    # kill process group if possible, else pid + children via /proc
    kill "$pid" 2>/dev/null || true
    # also stop direct children (vLLM workers)
    for c in $(ls /proc/$pid/task/$pid/children 2>/dev/null; pgrep -P "$pid" 2>/dev/null || true); do
      kill "$c" 2>/dev/null || true
    done
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
    for c in $(pgrep -P "$pid" 2>/dev/null || true); do kill -9 "$c" 2>/dev/null || true; done
  fi
}
stop_pidfile() {
  local pidf=$1 why=${2:-}
  [[ -f "$pidf" ]] || return 0
  stop_pid "$(cat "$pidf" 2>/dev/null || true)" "$why"
  rm -f "$pidf"
}

f=/root/affine_data/r727_decision_reign35_wvk7.json
if [[ -f "$f" ]]; then
  python3 -c "import json;d=json.load(open('$f'));print(f'R727 wins={d.get(\"wins\")} m={d.get(\"margin\"):.6f} bar={d.get(\"bar\"):.6f} thought={d.get(\"thought_median\")} B={d.get(\"b_pass\"):.3f}')"
fi

stop_pidfile /root/logs/r727_sim_wvk7.pid "r727 sim"
stop_pidfile /root/logs/vllm_chall_r727.pid "r727 chall :8003"
if ss -lntp 2>/dev/null | grep -q ":8003"; then
  pid=$(ss -lntp 2>/dev/null | grep ":8003" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
  stop_pid "$pid" "port 8003 leftover"
fi

# free old merges but keep r726/r727/r736 + active
kept=0
for d in /tmp/r6*_merged /tmp/r70*_merged /tmp/r71*_merged /tmp/r72[0-5]_merged; do
  [[ -d "$d" ]] || continue
  case "$d" in */r726_merged|*/r727_merged|*/r736_merged) continue ;; esac
  log "free old merge $d"; rm -rf "$d"; kept=$((kept+1)); [[ "$kept" -ge 8 ]] && break
done
log "freed_old=$kept keep=/tmp/r726_merged /tmp/r727_merged"

for i in $(seq 1 90); do
  used67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  log "VRAM6+7=$used67 iter=$i"
  [[ "$used67" -lt 8192 ]] && break
  sleep 3
done
used67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used67" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }

EXP737=r737-r252-offline-dpo-hialpha-hirank-midbeta-midctx-hyperextrasteps-ep3-lolr
chmod +x /root/mining_src/$EXP737/*.sh

nohup bash /root/mining_src/$EXP737/lean_train_crown_gpus67_p3775.sh >/root/logs/p3775_r737_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3775_r737_lean_outer.pid
log "r737 lean outer pid=$(cat /root/logs/p3775_r737_lean_outer.pid)"

for i in $(seq 1 120); do
  ok737=0
  if [[ -f /root/logs/r737_train.pid ]]; then
    t=$(cat /root/logs/r737_train.pid)
    if [[ "$t" =~ ^[0-9]+$ ]] && kill -0 "$t" 2>/dev/null; then ok737=1; fi
  fi
  log "poll train live r737=$ok737 iter=$i"
  if [[ "$ok737" -eq 1 ]]; then
    nohup bash /root/mining_src/$EXP737/wait_r737_train_then_merge_p3775.sh >/root/logs/p3775_r737_wait.nohup 2>&1 &
    echo $! >/root/logs/p3775_r737_wait.pid
    log "TRAIN_LIVE r737=$(cat /root/logs/r737_train.pid) wait armed"
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p3775_crown_reap_launch.done
    exit 0
  fi
  sleep 2
done
log "FATAL train pid never appeared"; exit 1
