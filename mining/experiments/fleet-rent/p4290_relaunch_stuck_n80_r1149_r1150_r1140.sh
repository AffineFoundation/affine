#!/usr/bin/env bash
# p4290: R1149/R1150/R1140 MERGE_DONE sat idle — wait scripts had wrong EXP paths.
# Relaunch lean_chall from correct dirs. Never pkill -f. Never touch non-mine pods.
set -euo pipefail
ROOT=/home/const/subnet120/mining
KH="$ROOT/.ralph/known_hosts"
KEY=/home/const/.ssh/id_ed25519
SSH_BASE=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)
SCP_BASE=(scp -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)

EXP1149=r1149-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-ultralolr
EXP1150=r1150-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-hypersuperextrasteps-ep4-ultralolr
EXP1140=r1140-vera-offline-dpo-hialpha-hirank-midbeta-softctx-hypersuperextrasteps-ep4-ultralolr

echo "[p4290] $(date -u +%Y-%m-%dT%H:%M:%SZ) sync+launch stuck n80s"

# --- mine-r337: R1150 GPUs4,5 :8003 + R1149 GPUs6,7 :8002 ---
HOST=150.136.46.118; PORT=20300
"${SCP_BASE[@]}" -P "$PORT" \
  "$ROOT/experiments/$EXP1149/lean_chall_n80_r337_gpus67_p4273.sh" \
  "$ROOT/experiments/$EXP1149/wait_r1149_merge_then_n80_p4273.sh" \
  "root@${HOST}:/root/mining_src/$EXP1149/"
"${SCP_BASE[@]}" -P "$PORT" \
  "$ROOT/experiments/$EXP1150/lean_chall_n80_r337_gpus45_p4273.sh" \
  "$ROOT/experiments/$EXP1150/wait_r1150_merge_then_n80_p4273.sh" \
  "root@${HOST}:/root/mining_src/$EXP1150/"

"${SSH_BASE[@]}" -p "$PORT" root@"$HOST" 'bash -s' <<REMOTE
set -euo pipefail
[[ -f /tmp/r1149_merged/config.json ]] || { echo FATAL no r1149 merge; exit 1; }
[[ -f /tmp/r1150_merged/config.json ]] || { echo FATAL no r1150 merge; exit 1; }
n49=\$(ls /tmp/r1149_merged/model-*-of-*.safetensors | wc -l)
n50=\$(ls /tmp/r1150_merged/model-*-of-*.safetensors | wc -l)
echo "shards r1149=\$n49 r1150=\$n50"
[[ "\$n49" -ge 16 && "\$n50" -ge 16 ]] || exit 1
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
# free VRAM on 4-7
for i in 4 5 6 7; do
  used=\$(nvidia-smi -i \$i --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "GPU\$i used=\$used"
  [[ "\$used" -lt 2048 ]] || { echo FATAL GPU\$i busy; exit 1; }
done
chmod +x /root/mining_src/$EXP1150/lean_chall_n80_r337_gpus45_p4273.sh
chmod +x /root/mining_src/$EXP1149/lean_chall_n80_r337_gpus67_p4273.sh
nohup bash /root/mining_src/$EXP1150/lean_chall_n80_r337_gpus45_p4273.sh >/root/logs/p4290_r1150_chall_n80_relaunch.nohup 2>&1 &
echo \$! | tee /root/logs/p4290_r1150_chall_n80_relaunch.pid
nohup bash /root/mining_src/$EXP1149/lean_chall_n80_r337_gpus67_p4273.sh >/root/logs/p4290_r1149_chall_n80_relaunch.nohup 2>&1 &
echo \$! | tee /root/logs/p4290_r1149_chall_n80_relaunch.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4290_r1149_r1150_n80_relaunch_armed.done
echo R337_RELAUNCH_ARMED
REMOTE

# --- mine-r938: R1140 GPUs2,3 :8002 ---
HOST=38.255.28.21; PORT=20100
"${SCP_BASE[@]}" -P "$PORT" \
  "$ROOT/experiments/$EXP1140/lean_chall_n80_r938_gpus23_p4264.sh" \
  "$ROOT/experiments/$EXP1140/wait_r1140_merge_then_n80_p4264.sh" \
  "root@${HOST}:/root/mining_src/$EXP1140/"

"${SSH_BASE[@]}" -p "$PORT" root@"$HOST" 'bash -s' <<REMOTE
set -euo pipefail
[[ -f /tmp/r1140_merged/config.json ]] || { echo FATAL no r1140 merge; exit 1; }
n=\$(ls /tmp/r1140_merged/model-*-of-*.safetensors | wc -l)
echo "shards r1140=\$n"
[[ "\$n" -ge 16 ]] || exit 1
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
for i in 2 3; do
  used=\$(nvidia-smi -i \$i --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "GPU\$i used=\$used"
  [[ "\$used" -lt 2048 ]] || { echo FATAL GPU\$i busy; exit 1; }
done
chmod +x /root/mining_src/$EXP1140/lean_chall_n80_r938_gpus23_p4264.sh
nohup bash /root/mining_src/$EXP1140/lean_chall_n80_r938_gpus23_p4264.sh >/root/logs/p4290_r1140_chall_n80_relaunch.nohup 2>&1 &
echo \$! | tee /root/logs/p4290_r1140_chall_n80_relaunch.pid
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4290_r1140_n80_relaunch_armed.done
echo R938_RELAUNCH_ARMED
REMOTE

echo "[p4290] $(date -u +%Y-%m-%dT%H:%M:%SZ) armed — poll chall/n80 readiness"
