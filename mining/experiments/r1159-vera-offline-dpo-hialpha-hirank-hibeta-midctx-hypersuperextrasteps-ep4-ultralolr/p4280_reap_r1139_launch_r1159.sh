#!/usr/bin/env bash
# p4280: R1139 REFUTE → exact-PID reap r924 :8002 → R1159 MidCtx HiRank Hiβ Hyper UltraLoLR TRAIN GPUs6,7
# Never pkill -f. Do not touch teacher/king, R1154 on 1,3, or R1155 on 4,5.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
EXP=r1159-vera-offline-dpo-hialpha-hirank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=31.22.104.113
PORT=40300
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1159 /root/logs /root/affine_data"
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
# stop leftover r1139 waiters / sim / lean outer
for pf in /root/logs/vllm_chall_r1139.pid /root/logs/r1139_sim_wvk7.pid \
          /root/logs/r1139_merge_then_n80.pid /root/logs/p4279_r1139_n80_outer.pid \
          /root/logs/p4264_r1139_lean_outer.pid; do
  [[ -f "$pf" ]] || continue
  p=$(cat "$pf" 2>/dev/null || true)
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r1139|8002|lean_chall_n80_r924'; then
    if echo "$cmd" | grep -q r1139_merged; then reap "$p" r1139_merged
    elif echo "$cmd" | grep -q run_sim_duel; then reap "$p" r1139
    elif echo "$cmd" | grep -q lean_chall; then reap "$p" r1139
    elif echo "$cmd" | grep -qE 'wait_r1139|r1139_n80'; then reap "$p" r1139
    fi
  fi
done
# known chall APIServer pid 162026 if still r1139
if kill -0 162026 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/162026/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1139_merged; then reap 162026 r1139_merged; fi
fi
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1139_merged; then reap "$p" r1139_merged; fi
done < <(ss -lptn 'sport = :8002' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
# clear leftover EngineCore/Worker on GPUs 6,7 that belong to r1139 only
python3 - <<'PY'
import os, signal, subprocess, time
want={6,7}
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
    if "train_dpo" in cmd or "train_online" in cmd: continue
    if "r1154" in cmd or "r1155" in cmd: continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1139_merged" not in cmd: continue
    kill.add(pid)
    print(f"gpu67 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpu67 apps cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4280] wait GPUs6,7 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi -i 6,7; exit 1; }
# protect siblings
if ! kill -0 "$(cat /root/logs/r1154_train.pid 2>/dev/null)" 2>/dev/null; then echo WARN R1154 train not alive; fi
if ! kill -0 "$(cat /root/logs/r1155_train.pid 2>/dev/null)" 2>/dev/null; then echo WARN R1155 train not alive; fi
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
chmod +x /root/mining_src/r1159-vera-offline-dpo-hialpha-hirank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1159-vera-offline-dpo-hialpha-hirank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r924_gpus67_p4280.sh \
  >/root/logs/p4280_r1159_outer.nohup 2>&1 &
echo $! >/root/logs/p4280_r1159_outer.pid
sleep 15
echo "=== verify R1159 ==="
cat /root/logs/r1159_train.pid 2>/dev/null || true
head -30 /root/logs/r1159_lean_warm.log 2>/dev/null || true
ps -p "$(cat /root/logs/r1159_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null \
  || { echo FATAL train not up; tail -60 /root/logs/r1159_lean_warm.log; exit 1; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4280_r1139_refute_r1159_armed.done
echo "[p4280] R1159 UltraLoLR TRAIN armed"
REMOTE
