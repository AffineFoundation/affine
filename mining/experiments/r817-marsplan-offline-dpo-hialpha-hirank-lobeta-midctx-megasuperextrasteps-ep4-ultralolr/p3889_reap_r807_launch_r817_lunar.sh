#!/usr/bin/env bash
# p3889: R807 REFUTE ~−0.23× → reap chall :8002 GPUs 4,5 → R817 MidCtx Hi Lo UltraLoLR TRAIN.
# Leave R808 TRAIN on 6,7. Never pkill -f. Keep /tmp/r807_merged.
set -euo pipefail
log() { echo "[p3889-reap-r807] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
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
log "START reap R807 chall :8002 on 4,5; leave R808 on 6,7; keep /tmp/r807_merged"
stop_pid 787162 "r807 vllm :8002"
stop_pidfile /root/logs/r807_sim_wvk7.pid "r807 sim"
stop_pidfile /root/logs/vllm_chall_r807.pid "r807 vllm pidf"
stop_pidfile /root/logs/p3876_r807_chall_n80.pid "r807 lean outer"
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r807 vllm/sim/lean argv"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r807_merged|run_sim_duel.py.*r807|lean_chall_n80_lunar_gpus45_p3876/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
  echo "$cmd" | grep -qE 'r807_merged|port 8002' || continue
  echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001|r808|r817|train_dpo' && continue
  stop_pid "$pid" "r807 :8002"
done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
for i in $(seq 1 90); do
  used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used45=$used45 iter=$i"
  [[ "${used45:-999999}" -lt 8192 ]] && break
  sleep 2
done
used45=$(nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used45:-999999}" -lt 8192 ]] || { log "FATAL GPUs 4,5 still busy"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r807_refute_reaped.p3889

EXP=r817-marsplan-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r817 /root/mining_src/$EXP
if [[ ! -s /root/r817/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r817/dpo_duel_reason.jsonl
  elif [[ -s /root/r807/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r807/dpo_duel_reason.jsonl /root/r817/dpo_duel_reason.jsonl
  elif [[ -s /root/mining_src/r807-marsplan-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r807-marsplan-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl /root/r817/dpo_duel_reason.jsonl
  fi
fi
cp -f /root/r817/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl
# ensure lean_chall exists
if [[ ! -f /root/mining_src/$EXP/lean_chall_n80_lunar_gpus45_p3889.sh ]]; then
  src=/root/mining_src/r807-marsplan-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_lunar_gpus45_p3876.sh
  sed -e 's/r807/r817/g' -e 's/R807/R817/g' -e 's/p3876/p3889/g' \
      -e 's/midbeta/lobeta/g' -e 's/MidBeta/LoBeta/g' \
      "$src" > /root/mining_src/$EXP/lean_chall_n80_lunar_gpus45_p3889.sh
  chmod +x /root/mining_src/$EXP/lean_chall_n80_lunar_gpus45_p3889.sh
fi
log "launch R817 TRAIN 4,5 + wait→merge + wait→n80 (:8002); leave R808"
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus45_p3889.sh >/root/logs/p3889_r817_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3889_r817_lean_outer.pid
sleep 1
nohup bash /root/mining_src/$EXP/wait_r817_train_then_merge_p3889.sh >/root/logs/p3889_r817_wait.nohup 2>&1 &
echo $! >/root/logs/p3889_r817_wait.pid
nohup bash /root/mining_src/$EXP/wait_r817_merge_then_n80_p3889.sh >/root/logs/p3889_r817_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3889_r817_wait_n80.pid
log "R817 lean=$(cat /root/logs/p3889_r817_lean_outer.pid) wait=$(cat /root/logs/p3889_r817_wait.pid) n80w=$(cat /root/logs/p3889_r817_wait_n80.pid)"
# verify train started within ~90s
for i in $(seq 1 45); do
  if [[ -f /root/logs/r817_train.pid ]]; then
    tpid=$(cat /root/logs/r817_train.pid)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then
      log "TRAIN_OK pid=$tpid"
      # confirm beta in cmdline
      tr '\0' ' ' </proc/$tpid/cmdline; echo
      exit 0
    fi
  fi
  sleep 2
done
log "WARN train pid not yet visible; check lean outer log"
tail -40 /root/logs/p3889_r817_lean_outer.nohup || true
tail -40 /root/logs/r817_lean_warm.log || true
exit 1
