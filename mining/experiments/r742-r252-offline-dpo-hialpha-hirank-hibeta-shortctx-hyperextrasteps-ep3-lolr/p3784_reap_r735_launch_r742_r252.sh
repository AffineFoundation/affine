#!/usr/bin/env bash
# p3784: R735 REFUTE → reap chall GPUs 6,7 → launch R742 TRAIN + wait→merge. Never pkill -f.
set -euo pipefail
log() { echo "[p3784-reap-r735] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=$1; local why=${2:-}
  [[ -f "$pidf" ]] || return 0
  local pid; pid=$(cat "$pidf" 2>/dev/null || true)
  stop_pid "$pid" "$why pidf=$pidf"
  rm -f "$pidf"
}
log "START reap R735 chall; keep /tmp/r735_merged"
stop_pidfile /root/logs/r735_sim_wvk7.pid "r735 sim"
stop_pidfile /root/logs/p3781_r735_lean_outer.pid "r735 outer"
stop_pidfile /root/logs/vllm_chall_r735.pid "r735 vllm"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r735 argv"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r735_merged/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free 6,7 used_mib=$used iter=$i"
  [[ "${used:-999999}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r735_refute_reaped.p3784
python3 - <<'PY'
import json, time
from pathlib import Path
v = json.load(open('/root/affine_data/r735_sim_result_reign35_wvk7.json'))['verdict']
m, se, z, n = v['margin'], v['se'], v['z'], v['n_paired_turns']
bar = max(2.0*se, 0.002)
dec = {
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'exp': 'R735',
  'verdict': 'REFUTE',
  'margin': m, 'se': se, 'z': z, 'n': n, 'bar': bar,
  'xbar': (m/bar) if bar else None,
  'thought_median': v['challenger']['median_len_z'],
  'b_pass': v['challenger']['b_gate_pass_rate'],
  'k': v['duel_params']['n_teacher_samples'],
  'tau': v['duel_params']['tau'],
  'king': 'tammyfritz/Affine-5hmwhnfbix-tammy2',
  'next': 'R742 TRAIN on GPUs 6,7',
}
Path('/root/affine_data/r735_decision_reign35_wvk7.json').write_text(json.dumps(dec, indent=2)+'\n')
print(json.dumps(dec, indent=2))
PY
log "launch R742 TRAIN"
nohup bash /root/mining_src/r742-r252-offline-dpo-hialpha-hirank-hibeta-shortctx-hyperextrasteps-ep3-lolr/lean_train_r252_gpus67_p3784.sh \
  >/root/logs/p3784_r742_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3784_r742_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r742-r252-offline-dpo-hialpha-hirank-hibeta-shortctx-hyperextrasteps-ep3-lolr/wait_r742_train_then_merge_p3784.sh \
  >/root/logs/p3784_r742_wait.nohup 2>&1 &
echo $! >/root/logs/p3784_r742_wait.pid
log "R742 lean outer=$(cat /root/logs/p3784_r742_lean_outer.pid) wait=$(cat /root/logs/p3784_r742_wait.pid)"
