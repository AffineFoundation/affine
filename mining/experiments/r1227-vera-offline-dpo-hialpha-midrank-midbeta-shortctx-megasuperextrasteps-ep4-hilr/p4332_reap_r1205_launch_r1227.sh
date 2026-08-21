#!/usr/bin/env bash
# p4332: r339 R1205 REFUTE idle chall :8002 → exact-PID reap → Mega R1227 ShortCtx MidRank Midβ HiLR GPUs4,5.
# Leave teacher:8000 / king:8001 / R1206 train GPUs6,7. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1227=r1227-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/known_hosts
HOST=23.153.44.20
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$E1227"/*.sh
echo "[p4332] sync R1227 → mine-r339"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1227 /root/r1227 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1227"/. "root@${HOST}:/root/mining_src/$E1227/"
dir=$(ls -d "$ROOT"/r1205-vera-* 2>/dev/null | head -1 || true)
if [[ -n "${dir:-}" ]]; then
  mkdir -p "$dir/results"
  "${SCP[@]}" "root@${HOST}:/root/affine_data/r1205_decision_reign36_wvk7.json" "$dir/results/" 2>/dev/null || true
  "${SCP[@]}" "root@${HOST}:/root/affine_data/r1205_sim_result_reign36_wvk7.json" "$dir/results/" 2>/dev/null || true
fi
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=${cmd:0:200}"
    if [[ -z "$cmd" ]] || echo "$cmd" | grep -q "$tok"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do
        alive=0
        for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break
        sleep 1
      done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo "skip wrong-token pid=$pid"; fi
  else echo already gone pid=$pid; fi
}
# stop R1205 waiters / lean_chall parent (exact pidfile)
for f in /root/logs/r1205_merge_then_n80.pid /root/logs/r1205_sim_wvk7.pid /root/logs/vllm_chall_r1205.pid; do
  if [[ -f "$f" ]]; then
    sp=$(cat "$f" 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if [[ -z "$cmd" ]] || echo "$cmd" | grep -Eq 'r1205|lean_chall|run_sim_duel'; then
        # do not kill lean_chall until after chall reap if it is the parent — kill sim first
        if echo "$cmd" | grep -q 'run_sim_duel'; then
          kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
        fi
      fi
    fi
  fi
done
# kill lean_chall wrapper 93951 if still waiting (exact)
if kill -0 93951 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/93951/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q 'r1205'; then
    kill 93951 2>/dev/null || true; sleep 1; kill -9 93951 2>/dev/null || true
  fi
fi
reap 96406 r1205_merged
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
    if "train_dpo" in cmd or "r1206" in cmd: continue
    if "GLM-4.5-Air-FP8" in cmd or ("affine-5g4yy75zuz-t6" in cmd and "--port 8001" in cmd):
        continue
    if any(x in cmd for x in ("r1205_merged","Worker_TP","EngineCore")):
        kill.add(pid)
for pid in sorted(kill):
    print(f"gpu-clear pid={pid}")
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpu-clear done")
PY
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4332] VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 12000 ]] && break
  sleep 3
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 12000 ]] || { echo FATAL VRAM still busy used=$used; nvidia-smi; exit 1; }
# R1206 must still be alive
kill -0 93945 2>/dev/null || echo WARN r1206 train pid gone
ss -lptn | grep -E ':8000|:8001' || { echo FATAL teacher/king down; exit 1; }
echo "[p4332] launching R1227"
nohup bash /root/mining_src/r1227-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-hilr/lean_train_r339_gpus45_p4332.sh >/root/logs/r1227_lean_launch.nohup 2>&1 &
echo $! >/root/logs/r1227_lean_launch.pid
sleep 12
echo -n "r1227 train.pid="; cat /root/logs/r1227_train.pid 2>/dev/null || echo MISSING
tail -5 /root/logs/r1227_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4332_r1227_launched.done
echo "[p4332] DONE r1227"
REMOTE
echo "[p4332] r339 host script finished"
