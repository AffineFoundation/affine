#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p3784-reap-r730] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
stop_pid() {
  local pid=$1
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
stop_pidfile() {
  local pidf=$1
  [[ -f "$pidf" ]] || return 0
  stop_pid "$(cat "$pidf" 2>/dev/null || true)"
  rm -f "$pidf"
}
log START
stop_pidfile /root/logs/r730_sim_wvk7.pid
stop_pidfile /root/logs/p3783_r730_lean_outer.pid
stop_pidfile /root/logs/vllm_chall_r730.pid
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid"
done < <(ps -eo pid=,args= | awk '/vllm serve .*\/tmp\/r730_merged/ && !/awk/ {print $1}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  [[ "${used:-999}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999}" -lt 8192 ]] || { log FATAL; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r730_refute_reaped.p3784
python3 - <<'PY'
import json, time
from pathlib import Path
v = json.load(open('/root/affine_data/r730_sim_result_reign35_wvk7.json'))['verdict']
m, se, z, n = v['margin'], v['se'], v['z'], v['n_paired_turns']
bar = max(2.0*se, 0.002)
dec = {
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'exp': 'R730', 'verdict': 'REFUTE',
  'margin': m, 'se': se, 'z': z, 'n': n, 'bar': bar,
  'xbar': (m/bar) if bar else None,
  'thought_median': v['challenger']['median_len_z'],
  'b_pass': v['challenger']['b_gate_pass_rate'],
  'k': v['duel_params']['n_teacher_samples'],
  'tau': v['duel_params']['tau'],
  'king': 'tammyfritz/Affine-5hmwhnfbix-tammy2',
  'next': 'R746 TRAIN on GPUs 6,7',
}
Path('/root/affine_data/r730_decision_reign35_wvk7.json').write_text(json.dumps(dec, indent=2)+'\n')
print(json.dumps(dec, indent=2))
PY
nohup bash /root/mining_src/r746-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-superextrasteps-ep3-lolr/lean_train_golden_gpus67_p3784.sh >/root/logs/p3784_r746_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3784_r746_lean_outer.pid
sleep 2
nohup bash /root/mining_src/r746-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-superextrasteps-ep3-lolr/wait_r746_train_then_merge_p3784.sh >/root/logs/p3784_r746_wait.nohup 2>&1 &
echo $! >/root/logs/p3784_r746_wait.pid
log R746 outer=$(cat /root/logs/p3784_r746_lean_outer.pid) wait=$(cat /root/logs/p3784_r746_wait.pid)
