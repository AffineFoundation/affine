#!/usr/bin/env bash
# p4276: R1132 REFUTE → exact-PID reap r924 :8003 → R1154 UltraLoLR TRAIN GPUs1,3
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
EXP=r1154-vera-offline-dpo-hialpha-lorank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=31.22.104.113
PORT=40300
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1154 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid tok=$tok cmd=$cmd"
    if echo "$cmd" | grep -q "$tok"; then
      pids="$pid"
      kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do
        alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break; sleep 1
      done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid; exit 2; fi
  else echo already gone pid=$pid; fi
}
# stop lean outer / sim if still around
for pf in /root/logs/vllm_chall_r1132.pid /root/logs/r1132_sim_wvk7.pid /root/logs/r1132_merge_then_n80.pid; do
  [[ -f "$pf" ]] || continue
  p=$(cat "$pf" 2>/dev/null || true)
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r1132|8003'; then
    if echo "$cmd" | grep -q r1132_merged; then reap "$p" r1132_merged
    elif echo "$cmd" | grep -q run_sim_duel; then reap "$p" r1132
    elif echo "$cmd" | grep -q lean_chall; then reap "$p" r1132
    elif echo "$cmd" | grep -q wait_r1132; then reap "$p" r1132
    fi
  fi
done
# port 8003 listeners that are r1132
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1132_merged; then reap "$p" r1132_merged; fi
done < <(ss -lptn 'sport = :8003' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
# also kill known chall pid 153997 if still r1132
if kill -0 153997 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/153997/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1132_merged; then reap 153997 r1132_merged; fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4276] wait GPUs1,3 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi -i 1,3; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
# do not touch R1131 on :8004 / GPUs 4,5 or R1139 on 6,7
chmod +x /root/mining_src/r1154-vera-offline-dpo-hialpha-lorank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1154-vera-offline-dpo-hialpha-lorank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r924_gpus13_p4276.sh \
  >/root/logs/p4276_r1154_outer.nohup 2>&1 &
echo $! >/root/logs/p4276_r1154_outer.pid
sleep 15
echo "=== verify R1154 ==="
cat /root/logs/r1154_train.pid 2>/dev/null || true
head -20 /root/logs/r1154_lean_warm.log 2>/dev/null || true
ps -p "$(cat /root/logs/r1154_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null \
  || { echo FATAL train not up; tail -60 /root/logs/r1154_lean_warm.log; exit 1; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4276_r1132_refute_r1154_armed.done
echo "[p4276] R1154 UltraLoLR TRAIN armed"
REMOTE
