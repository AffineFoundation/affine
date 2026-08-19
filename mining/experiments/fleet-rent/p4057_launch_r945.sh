#!/usr/bin/env bash
# p4057: extract+launch R945 SoftCtx Hiβ UltraExtra on crown GPUs 1,3 (tar already at /tmp/r945_p4057.tar.gz)
set -uo pipefail
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=40 -o BatchMode=yes -o ServerAliveInterval=10)
DST=95.133.252.28
DPORT=40298
EXP=r945-vera-offline-dpo-hialpha-midrank-hibeta-softctx-ultrasuperextrasteps-ep4-ultralolr

for attempt in $(seq 1 12); do
  echo "attempt $attempt $(date -u +%H:%M:%S)"
  if ssh "${SSH_OPTS[@]}" -p "$DPORT" "root@$DST" bash -s <<REMOTE
set -euo pipefail
mkdir -p /root/mining_src /root/logs
tar -C /root/mining_src -xzf /tmp/r945_p4057.tar.gz
chmod +x /root/mining_src/${EXP}/*.sh
if [[ -f /root/logs/r945_train.pid ]]; then
  old=\$(cat /root/logs/r945_train.pid)
  if [[ "\$old" =~ ^[0-9]+\$ ]] && kill -0 "\$old" 2>/dev/null; then
    echo ALREADY:\$old
    exit 0
  fi
fi
nohup bash /root/mining_src/${EXP}/lean_train_crown_gpus13_p4057.sh >/root/logs/r945_lean_outer.nohup 2>&1 &
echo \$! >/root/logs/r945_lean_outer.pid
sleep 8
nohup bash /root/mining_src/${EXP}/wait_r945_train_then_merge_p4057.sh >/root/logs/r945_wait_outer.nohup 2>&1 &
echo \$! >/root/logs/r945_wait_outer.pid
sleep 4
echo TRAIN=\$(cat /root/logs/r945_train.pid 2>/dev/null || echo none)
tail -12 /root/logs/r945_lean_warm.log || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
curl -sS --max-time 4 http://127.0.0.1:8002/v1/models 2>&1 | head -c 160 || true
echo
REMOTE
  then
    echo SUCCESS
    exit 0
  fi
  sleep 12
done
echo GIVE_UP
exit 1
