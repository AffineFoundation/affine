#!/usr/bin/env bash
# p4279: R1139 merge ready but n80 waiter died (softctx path typo) → arm lean chall n80 on GPUs6,7.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
EXP=r1139-vera-offline-dpo-hialpha-hirank-hibeta-midctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=31.22.104.113
PORT=40300
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/logs /root/affine_data"
"${SCP[@]}" "$ROOT/$EXP/lean_chall_n80_r924_gpus67_p4264.sh" "$ROOT/$EXP/wait_r1139_merge_then_n80_p4264.sh" "root@${HOST}:/root/mining_src/$EXP/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
EXP=r1139-vera-offline-dpo-hialpha-hirank-hibeta-midctx-hypersuperextrasteps-ep4-hilr
n=$(ls /tmp/r1139_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
echo "shards=$n"
[[ "$n" -ge 16 ]] || { echo FATAL merge incomplete; exit 1; }
[[ -f /tmp/r1139_merged/config.json ]] || exit 1
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
echo "gpus67 used=$used"
[[ "$used" -lt 8192 ]] || { echo FATAL gpus busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
# clear stale :8002 if any leftover
if ss -lptn 'sport = :8002' 2>/dev/null | grep -q 8002; then
  pid=$(ss -lptn 'sport = :8002' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
  if [[ -n "${pid:-}" ]]; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "stale :8002 pid=$pid cmd=$cmd"
    if echo "$cmd" | grep -qE 'r1139|8002'; then
      kill "$pid" 2>/dev/null || true; sleep 2; kill -9 "$pid" 2>/dev/null || true
    fi
  fi
fi
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_chall_n80_r924_gpus67_p4264.sh >/root/logs/p4279_r1139_n80_outer.nohup 2>&1 &
echo $! >/root/logs/p4279_r1139_n80_outer.pid
sleep 8
echo "outer=$(cat /root/logs/p4279_r1139_n80_outer.pid)"
tail -30 /root/logs/p4279_r1139_n80_outer.nohup 2>/dev/null || true
tail -20 /root/logs/p4264_r1139_chall_n80_wvk7.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4279_r1139_n80_armed.done
echo "[p4279] R1139 n80 armed"
REMOTE
