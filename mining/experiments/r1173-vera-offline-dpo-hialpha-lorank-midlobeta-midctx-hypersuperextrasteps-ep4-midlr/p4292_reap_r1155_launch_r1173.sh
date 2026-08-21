#!/usr/bin/env bash
# p4292: r924 R1155 REFUTE → exact-PID reap :8004 → R1173 MidLR TRAIN on GPUs 4,5
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1169 TRAIN GPUs1,3 / R1159 TRAIN GPUs6,7.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1173-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=31.22.104.113
PORT=40300
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
echo "[p4292] sync R1173 → mine-r924"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1173 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=$cmd"
    if echo "$cmd" | grep -q "$tok"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid; exit 2; fi
  else echo already gone pid=$pid; fi
}
# stop leftover R1155 waiters / lean / sim (token-gated)
for f in /root/logs/r1155_merge_then_n80.pid /root/logs/r1155_wait_merge.pid /root/logs/r1155_sim_wvk7.pid /root/logs/vllm_chall_r1155.pid; do
  if [[ -f "$f" ]]; then
    sp=$(cat "$f" 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -Eq 'r1155'; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
    fi
  fi
done
for sp in $(pgrep -af 'lean_chall_n80_r924_gpus45_p4277|run_sim_duel.py.*r1155|wait_r1155' 2>/dev/null | awk '{print $1}' || true); do
  [[ "$sp" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r1155'; then
    echo "stop leftover r1155 pid=$sp"
    kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
  fi
done
reap "$(cat /root/logs/vllm_chall_r1155.pid 2>/dev/null || echo 170202)" r1155_merged
for port in 8004; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1155_merged'; then
      reap "$p" r1155_merged
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear leftover compute apps on GPUs 4,5 only
python3 - <<'PY'
import os, signal, subprocess, time
want={4,5}
out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    parts=[p.strip() for p in line.split(",")]
    if len(parts)>=2: idx_to_uuid[int(parts[0])]=parts[1]
uuids={idx_to_uuid[i] for i in want if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],text=True)
kill=set()
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in uuids: continue
    try: pid=int(parts[1])
    except ValueError: continue
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd: continue
    if "r1169" in cmd or "r1159" in cmd or "r1173" in cmd: continue
    kill.add(pid)
    print(f"gpu4,5 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 4,5 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4292] wait free GPUs4,5 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
ps -p "$(cat /root/logs/r1159_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3 || true
ps -p "$(cat /root/logs/r1169_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3 || true
chmod +x /root/mining_src/r1173-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/*.sh
nohup bash /root/mining_src/r1173-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/lean_train_r924_gpus45_p4292.sh >/root/logs/p4292_r1173_outer.nohup 2>&1 &
echo $! >/root/logs/p4292_r1173_outer.pid
sleep 25
echo "=== verify ==="
ps -p "$(cat /root/logs/r1173_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
ps -p "$(cat /root/logs/r1159_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
ps -p "$(cat /root/logs/r1169_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
cat /root/affine_data/r1173_train_launched.json 2>/dev/null || true
tail -40 /root/logs/r1173_lean_warm.log 2>/dev/null || true
REMOTE
