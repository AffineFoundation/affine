#!/usr/bin/env bash
# p4304: r926 R1170 REFUTE ~0.57× → exact-PID reap :8002 r1170_merged → R1187 SoftCtx MidRank MidLoβ MidLR TRAIN
# Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E187=r1187-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr
KH=/home/const/subnet120/mining/known_hosts
HOST=93.120.231.186
PORT=32301
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$E187"/*.sh
echo "[p4304] sync R1187 → mine-r926"
"${SSH[@]}" "mkdir -p /root/mining_src/$E187 /root/r1187 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E187"/. "root@${HOST}:/root/mining_src/$E187/"
# also pull decision/result for local archive
mkdir -p "$ROOT/r1170-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/artifacts"
"${SCP[@]}" "root@${HOST}:/root/affine_data/r1170_decision_reign36_wvk7.json" \
  "$ROOT/r1170-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/artifacts/" || true
"${SCP[@]}" "root@${HOST}:/root/affine_data/r1170_sim_result_reign36_wvk7.json" \
  "$ROOT/r1170-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/artifacts/" || true
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=${cmd:0:180}"
    if [[ -z "$cmd" ]] || echo "$cmd" | grep -q "$tok"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo "skip wrong-token pid=$pid"; fi
  else echo already gone pid=$pid; fi
}
for tag in r1170; do
  for f in /root/logs/${tag}_merge_then_n80.pid /root/logs/${tag}_wait_merge.pid /root/logs/${tag}_sim_wvk7.pid /root/logs/vllm_chall_${tag}.pid /root/logs/p4291_r1170_outer.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if [[ -z "$cmd" ]] || echo "$cmd" | grep -Eq "$tag|r1170"; then
          kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
        fi
      fi
    fi
  done
done
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'wait_r1170|lean_chall_n80_r926.*r1170|r1170_sim|r1170_merged|p4291_r1170'; then
    reap "$p" "r1170"
  fi
done < <(ps -eo pid=,args= | awk '/r1170/ && !/awk/ {print $1}')
for port in 8002; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1170_merged'; then
      reap "$p" "r1170_merged"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
python3 - <<'PY'
import os, signal, subprocess, time
want={3,4}
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
    if "r1187" in cmd: continue
    kill.add(pid)
    print(f"gpu3-4 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 3-4 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[p4304] wait free GPUs3,4 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs3,4 still busy used=$used; exit 1; }
mkdir -p /root/r1187 /root/r1170 /root/r926
if [[ ! -s /root/r1187/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/r1187-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r1187-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl /root/r1187/dpo_duel_reason.jsonl
  elif [[ -s /root/r926/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r926/dpo_duel_reason.jsonl /root/r1187/dpo_duel_reason.jsonl
  fi
fi
chmod +x /root/mining_src/r1187-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr/*.sh
bash /root/mining_src/r1187-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-midlr/lean_train_h100_gpus34_p4304.sh
echo "[p4304] R1187 lean launched"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4304_r1170_refute_r1187_armed.done
tail -n 30 /root/logs/r1187_lean_warm.log
ps -eo pid,etime,cmd | awk '/[t]rain_dpo.*r1187|[r]1187_train/{print}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0,1,2,3,4,5,6,7
REMOTE
echo "[p4304] DONE"
