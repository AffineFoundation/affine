#!/usr/bin/env bash
# p4267: R1120 REFUTE m=−0.013799 ~−1.52× → exact-PID reap r340 :8002 GPU1 → R1142 UltraLoLR TRAIN
# Do not touch R1141 TRAIN GPUs3,4 / R1128 TRAIN GPUs6,7 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1142-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=18.118.83.97
PORT=40127
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
echo "[p4267] sync $EXP → mine-r340"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1142 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2" port="$3"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=$cmd"
    if echo "$cmd" | grep -q "$tok" && echo "$cmd" | grep -q -- "--port $port"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid; exit 2; fi
  else echo already gone; fi
}
# stop leftover sim if any
if [[ -f /root/logs/r1120_sim_wvk7.pid ]]; then
  sp=$(cat /root/logs/r1120_sim_wvk7.pid 2>/dev/null || true)
  if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -q r1120; then kill "$sp" 2>/dev/null || true; sleep 2; kill -9 "$sp" 2>/dev/null || true; fi
  fi
fi
pid=$(cat /root/logs/vllm_chall_r1120.pid 2>/dev/null || echo 58072)
reap "$pid" r1120_merged 8002
# clear any leftover :8002 listeners matching r1120
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1120_merged; then reap "$p" r1120_merged 8002; fi
done < <(ss -lptn 'sport = :8002' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
# also reap EngineCore workers still on GPU1 only (exact via nvidia-smi)
python3 - <<'PY'
import os, signal, subprocess, time
want={1}
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
    name=parts[2] if len(parts)>2 else ""
    if "train_dpo" in cmd or "train_online" in cmd: continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1120_merged" not in cmd: continue
    kill.add(pid)
    print(f"gpu1 app pid={pid} name={name} cmd={cmd[:120]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpu1 reaped", sorted(kill))
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2 | awk '{s+=$1} END{print s+0}')
  echo "VRAM1+2=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; nvidia-smi; exit 1; }
# confirm siblings still training
for id in r1141 r1128; do
  tp=$(cat /root/logs/${id}_train.pid 2>/dev/null || true)
  if [[ -n "${tp:-}" ]] && kill -0 "$tp" 2>/dev/null; then echo ${id}_OK pid=$tp; else echo WARN $id; fi
done
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo TEACHER_OK
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null && echo KING_OK
chmod +x /root/mining_src/r1142-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1142-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r340_gpus12_p4267.sh >/root/logs/p4267_r1142_lean_outer.nohup 2>&1 &
echo $! | tee /root/logs/p4267_r1142_lean_outer.pid
ok=0
for i in $(seq 1 60); do
  if [[ -f /root/logs/r1142_train.pid ]]; then
    tp=$(cat /root/logs/r1142_train.pid)
    if kill -0 "$tp" 2>/dev/null; then
      echo TRAIN_r1142_PID=$tp
      head -40 /root/logs/r1142_lean_warm.log || true
      cat /root/affine_data/r1142_train_launched.json || true
      ok=1
      break
    fi
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { echo FATAL train not started; tail -80 /root/logs/p4267_r1142_lean_outer.nohup /root/logs/r1142_lean_warm.log 2>/dev/null || true; exit 1; }
# after train owns GPUs, drop idle r1120 merge to free /tmp
rm -rf /tmp/r1120_merged
df -h /tmp | head -2
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4267_r1120_refute_r1142_armed.done
echo R1142_TRAIN_OK
REMOTE
echo "[p4267] R1142 armed"
