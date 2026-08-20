#!/usr/bin/env bash
set -euo pipefail
exec >/root/logs/p4072_reap_r947_launch_r959.nohup 2>&1
kill_tree(){ local pid="$1"; [[ "$pid" =~ ^[0-9]+$ ]] || return 0; kill -0 "$pid" 2>/dev/null || return 0
  for k in $(pgrep -P "$pid" 2>/dev/null||true); do kill_tree "$k"; done
  kill "$pid" 2>/dev/null||true; sleep 1; kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null||true; }
for f in /root/logs/vllm_chall_r947.pid /root/logs/r947_sim_wvk7.pid; do
  [[ -f $f ]] && kill_tree "$(cat $f)"
done
# also match live serve if pidfile missing
p=$(ss -lptn | awk -F'pid=' '/:8003/ {print $2}' | cut -d, -f1 | head -1 || true)
[[ -n "${p:-}" ]] && kill_tree "$p"
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "wait u45=$used"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
chmod +x /root/mining_src/r959-vera-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r959-vera-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_train_r338_gpus45_p4072.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4072_r959_lean_outer.pid
sleep 10
tail -30 /root/logs/r959_lean_warm.log
ps -p "$(cat /root/logs/r959_train.pid 2>/dev/null)" -o pid,cmd= || true
