#!/usr/bin/env bash
# p4330: r924 R1198/99/1200 REFUTE idle challs → exact-PID reap :8002/:8003/:8004
# → Mega R1222 ShortCtx HiRank Loβ MidLR (GPUs1,3) + R1223 ShortCtx MidRank Loβ HiLR (4,5)
# + R1224 MidCtx HiRank Loβ HiLR (6,7). Leave teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1222=r1222-vera-offline-dpo-hialpha-hirank-lobeta-shortctx-megasuperextrasteps-ep4-midlr
E1223=r1223-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-hilr
E1224=r1224-vera-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/known_hosts
HOST=31.22.104.113
PORT=40300
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$E1222"/*.sh "$ROOT/$E1223"/*.sh "$ROOT/$E1224"/*.sh
echo "[p4330] sync R1222+R1223+R1224 → mine-r924"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1222 /root/mining_src/$E1223 /root/mining_src/$E1224 /root/r1222 /root/r1223 /root/r1224 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1222"/. "root@${HOST}:/root/mining_src/$E1222/"
"${SCP[@]}" -r "$ROOT/$E1223"/. "root@${HOST}:/root/mining_src/$E1223/"
"${SCP[@]}" -r "$ROOT/$E1224"/. "root@${HOST}:/root/mining_src/$E1224/"
# pull decisions for archive
for tag in r1198 r1199 r1200; do
  mkdir -p "$ROOT/${tag}"*/results 2>/dev/null || true
done
for tag in r1198 r1199 r1200; do
  dir=$(ls -d "$ROOT"/${tag}-vera-* 2>/dev/null | head -1 || true)
  [[ -n "${dir:-}" ]] || continue
  mkdir -p "$dir/results"
  "${SCP[@]}" "root@${HOST}:/root/affine_data/${tag}_decision_reign36_wvk7.json" "$dir/results/" 2>/dev/null || true
  "${SCP[@]}" "root@${HOST}:/root/affine_data/${tag}_sim_result_reign36_wvk7.json" "$dir/results/" 2>/dev/null || true
done
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
# stop merge/sim waiters for resolved axes
for tag in r1198 r1199 r1200; do
  for f in /root/logs/${tag}_merge_then_n80.pid /root/logs/${tag}_wait_merge.pid /root/logs/${tag}_sim_wvk7.pid /root/logs/vllm_chall_${tag}.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if [[ -z "$cmd" ]] || echo "$cmd" | grep -Eq "$tag|r1198|r1199|r1200"; then
          kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
        fi
      fi
    fi
  done
done
# exact-PID chall servers
for pair in "194356:r1198_merged" "193322:r1199_merged" "198630:r1200_merged"; do
  pid=${pair%%:*}; tok=${pair##*:}
  reap "$pid" "$tok"
done
# clear leftover workers on GPUs 1,3,4,5,6,7 only (leave 0 teacher, 2 king)
python3 - <<'PY'
import os, signal, subprocess, time
want={1,3,4,5,6,7}
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
    # never touch teacher/king serve lines
    if "GLM-4.5-Air-FP8" in cmd or "affine-5g4yy75zuz-t6" in cmd and "--port 8001" in cmd:
        continue
    if any(x in cmd for x in ("r1198_merged","r1199_merged","r1200_merged","Worker_TP","EngineCore")):
        # EngineCore on GPU0/2 is teacher/king — skip if not in want uuid already filtered
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
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3,4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4330] VRAM1+3+4+5+6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 24000 ]] && break
  sleep 3
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3,4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 24000 ]] || { echo FATAL VRAM still busy used=$used; nvidia-smi; exit 1; }
# confirm T/K still up
ss -lptn | grep -E ':8000|:8001' || { echo FATAL teacher/king down; exit 1; }
echo "[p4330] launching R1222/23/24"
nohup bash /root/mining_src/r1222-vera-offline-dpo-hialpha-hirank-lobeta-shortctx-megasuperextrasteps-ep4-midlr/lean_train_r924_gpus13_p4330.sh >/root/logs/r1222_lean_launch.nohup 2>&1 &
echo $! >/root/logs/r1222_lean_launch.pid
nohup bash /root/mining_src/r1223-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-hilr/lean_train_r924_gpus45_p4330.sh >/root/logs/r1223_lean_launch.nohup 2>&1 &
echo $! >/root/logs/r1223_lean_launch.pid
nohup bash /root/mining_src/r1224-vera-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-hilr/lean_train_r924_gpus67_p4330.sh >/root/logs/r1224_lean_launch.nohup 2>&1 &
echo $! >/root/logs/r1224_lean_launch.pid
sleep 8
echo "=== launch status ==="
for t in r1222 r1223 r1224; do
  echo -n "$t train.pid="; cat /root/logs/${t}_train.pid 2>/dev/null || echo MISSING
  tail -2 /root/logs/${t}_lean_warm.log 2>/dev/null || tail -5 /root/logs/${t}_lean_launch.nohup 2>/dev/null || true
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/p4330_r1222_23_24_launched.done
echo "[p4330] DONE"
REMOTE
echo "[p4330] host script finished"
