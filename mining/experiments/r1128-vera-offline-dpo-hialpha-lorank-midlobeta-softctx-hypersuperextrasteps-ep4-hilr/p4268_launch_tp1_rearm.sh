#!/usr/bin/env bash
# p4268 host: SCP TP1 rearm to r340 and nohup it; wait briefly for VRAM climb / CHALL_READY.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
EXP=r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=18.118.83.97
PORT=40127
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")

"${SCP[@]}" "$ROOT/$EXP/p4268_reap_rearm_tp1.sh" "root@${HOST}:/root/mining_src/$EXP/p4268_reap_rearm_tp1.sh"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
chmod +x /root/mining_src/r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/p4268_reap_rearm_tp1.sh
nohup bash /root/mining_src/r1128-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-hilr/p4268_reap_rearm_tp1.sh \
  >/root/logs/p4268_r1128_rearm_tp1_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4268_r1128_rearm_tp1_outer.pid
# Wait up to ~100s for VRAM climb past stall threshold
for i in $(seq 1 20); do
  sleep 5
  u=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1+0}')
  echo "poll=$i gpu6_mib=$u"
  if curl -sf -m 2 http://127.0.0.1:8004/v1/models >/dev/null 2>&1; then
    echo CHALL_READY_EARLY
    break
  fi
  if [[ "${u:-0}" -gt 8000 ]]; then
    echo VRAM_CLIMBING
    break
  fi
done
tail -n 30 /root/logs/p4268_r1128_rearm_tp1.log 2>/dev/null || true
ps -eo pid,etime,args | grep -E 'p4268_reap_rearm_tp1|vllm serve /tmp/r1128' | grep -v grep | head -8
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 6,7
REMOTE
echo "[p4268] TP1 rearm launched"
