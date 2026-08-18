#!/usr/bin/env bash
set -euo pipefail
log(){ echo "[p3844-onpod-r789] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
stop_pid(){ local pid=$1; [[ "$pid" =~ ^[0-9]+$ ]] || return 0; kill -0 "$pid" 2>/dev/null || return 0; log "stop $pid"; kill "$pid" 2>/dev/null || true; for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -9 "$pid" 2>/dev/null || true; }
log "reap R778 chall :8003; keep R779 TRAIN 4,5; keep /tmp/r778_merged"
for pf in /root/logs/vllm_chall_r778.pid /root/logs/r778_sim_wvk7.pid /root/logs/p3828_r778_lean.outer.pid; do
  [[ -f "$pf" ]] || continue; stop_pid "$(cat "$pf" 2>/dev/null || true)"; rm -f "$pf"
done
# also kill known chall pid if still up
stop_pid 748677
while read -r pid; do [[ "$pid" =~ ^[0-9]+$ ]] || continue; stop_pid "$pid"; done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r778_merged|run_sim_duel.py.*r778|lean_chall_n80_lunar_gpus67_p3828/ && !/awk/{print $1}')
for i in $(seq 1 90); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  [[ "${used:-999999}" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
[[ "${used:-999999}" -lt 8192 ]] || { log "FATAL GPUs 6,7 busy used=$used"; exit 1; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r778_refute_reaped.p3844
# seed data from r778
mkdir -p /root/r789 /root/logs /root/affine_data
if [[ ! -s /root/r789/dpo_duel_reason.jsonl ]]; then
  cp -f /root/r778/dpo_duel_reason.jsonl /root/r789/dpo_duel_reason.jsonl
fi
EXP=r789-marsplan-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
# ensure train_dpo present
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
chmod +x /root/mining_src/$EXP/*.sh
# Fix any residual 1e-6 in lean env to 5e-7
sed -i 's/R789_LR=5e-7/R789_LR=5e-7/; s/R778_LR=1e-6/R789_LR=5e-7/; s/R789_LR=1e-6/R789_LR=5e-7/' /root/mining_src/$EXP/lean_train_lunar_gpus67_p3844.sh || true
grep -n 'LR=' /root/mining_src/$EXP/lean_train_lunar_gpus67_p3844.sh | head -5
grep -n 'LR=' /root/mining_src/$EXP/start_r789.sh | head -5
nohup bash /root/mining_src/$EXP/lean_train_lunar_gpus67_p3844.sh >/root/logs/p3844_r789_lean_outer.nohup 2>&1 &
echo $! >/root/logs/p3844_r789_lean_outer.pid
sleep 2
nohup bash /root/mining_src/$EXP/wait_r789_train_then_merge_p3844.sh >/root/logs/p3844_r789_wait.nohup 2>&1 &
echo $! >/root/logs/p3844_r789_wait.pid
nohup bash /root/mining_src/$EXP/wait_r789_merge_then_n80_p3844.sh >/root/logs/p3844_r789_wait_n80.nohup 2>&1 &
echo $! >/root/logs/p3844_r789_wait_n80.pid
sleep 8
# confirm train alive
pid=$(cat /root/logs/r789_train.pid 2>/dev/null || true)
log "R789 lean=$(cat /root/logs/p3844_r789_lean_outer.pid) wait=$(cat /root/logs/p3844_r789_wait.pid) n80w=$(cat /root/logs/p3844_r789_wait_n80.pid) train_pid=${pid:-none}"
if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
  log "TRAIN_OK pid=$pid"
  ps -p "$pid" -o pid,etime,cmd | head -2
else
  log "WARN train not yet; lean log:";
  tail -40 /root/logs/r789_lean_warm.log || true
  tail -40 /root/logs/p3844_r789_lean_outer.nohup || true
fi
