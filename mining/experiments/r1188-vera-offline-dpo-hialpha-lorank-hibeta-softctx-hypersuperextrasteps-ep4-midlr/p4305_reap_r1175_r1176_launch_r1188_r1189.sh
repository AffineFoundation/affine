#!/usr/bin/env bash
# p4305: crown R1175+R1176 REFUTE → exact-PID reap :8002/:8003 → R1188 MidLR + R1189 Hiβ UltraLoLR TRAIN
# Never pkill -f. Do not touch teacher:8000 / king:8001 / r1179 train on GPUs1,3.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E188=r1188-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-midlr
E189=r1189-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/known_hosts
HOST=95.133.252.28
PORT=40298
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$E188"/*.sh "$ROOT/$E189"/*.sh
echo "[p4305] sync R1188+R1189 → mine-crown-1"
"${SSH[@]}" "mkdir -p /root/mining_src/$E188 /root/mining_src/$E189 /root/r1188 /root/r1189 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E188"/. "root@${HOST}:/root/mining_src/$E188/"
"${SCP[@]}" -r "$ROOT/$E189"/. "root@${HOST}:/root/mining_src/$E189/"
# archive decisions locally
mkdir -p "$ROOT/r1175-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-ultralolr/artifacts"
mkdir -p "$ROOT/r1176-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr/artifacts"
"${SCP[@]}" "root@${HOST}:/root/affine_data/r1175_decision_reign36_wvk7.json" \
  "$ROOT/r1175-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-ultralolr/artifacts/" || true
"${SCP[@]}" "root@${HOST}:/root/affine_data/r1175_sim_result_reign36_wvk7.json" \
  "$ROOT/r1175-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-ultralolr/artifacts/" || true
"${SCP[@]}" "root@${HOST}:/root/affine_data/r1176_decision_reign36_wvk7.json" \
  "$ROOT/r1176-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr/artifacts/" || true
"${SCP[@]}" "root@${HOST}:/root/affine_data/r1176_sim_result_reign36_wvk7.json" \
  "$ROOT/r1176-vera-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr/artifacts/" || true
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
for tag in r1175 r1176; do
  for f in /root/logs/${tag}_merge_then_n80.pid /root/logs/${tag}_wait_merge.pid /root/logs/${tag}_sim_wvk7.pid /root/logs/vllm_chall_${tag}.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if [[ -z "$cmd" ]] || echo "$cmd" | grep -Eq "$tag|r1175|r1176"; then
          kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
        fi
      fi
    fi
  done
done
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'wait_r1175|wait_r1176|lean_chall_n80_crown.*r1175|lean_chall_n80_crown.*r1176|r1175_sim|r1176_sim|r1175_merged|r1176_merged|p4293_r1175|p4293_r1176'; then
    if echo "$cmd" | grep -q r1175; then reap "$p" "r1175"
    elif echo "$cmd" | grep -q r1176; then reap "$p" "r1176"
    else reap "$p" "r117"; fi
  fi
done < <(ps -eo pid=,args= | awk '/r1175|r1176/ && !/awk/ && !/r1179/ {print $1}')
for port in 8002 8003; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1175_merged|r1176_merged'; then
      if echo "$cmd" | grep -q r1175; then reap "$p" "r1175_merged"
      else reap "$p" "r1176_merged"; fi
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear GPUs 4-7 only (leave 0 teacher, 1+3 r1179/king)
python3 - <<'PY'
import os, signal, subprocess, time
want={4,5,6,7}
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
    if "r1179" in cmd or "r1188" in cmd or "r1189" in cmd: continue
    kill.add(pid)
    print(f"gpu4-7 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 4-7 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4305] wait free GPUs4-7 used=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL GPUs4-7 still busy used=$used; exit 1; }
mkdir -p /root/r1188 /root/r1189 /root/r1175 /root/r1176
if [[ ! -s /root/r1188/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/r1188-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r1188-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl /root/r1188/dpo_duel_reason.jsonl
  elif [[ -s /root/r1175/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r1175/dpo_duel_reason.jsonl /root/r1188/dpo_duel_reason.jsonl
  fi
fi
if [[ ! -s /root/r1189/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/r1189-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r1189-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl /root/r1189/dpo_duel_reason.jsonl
  elif [[ -s /root/r1176/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r1176/dpo_duel_reason.jsonl /root/r1189/dpo_duel_reason.jsonl
  fi
fi
chmod +x /root/mining_src/r1188-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-midlr/*.sh
chmod +x /root/mining_src/r1189-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
bash /root/mining_src/r1188-vera-offline-dpo-hialpha-lorank-hibeta-softctx-hypersuperextrasteps-ep4-midlr/lean_train_crown_gpus67_p4305.sh
bash /root/mining_src/r1189-vera-offline-dpo-hialpha-midrank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_crown_gpus45_p4305.sh
echo "[p4305] R1188+R1189 lean launched"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4305_r1175_r1176_refute_r1188_r1189_armed.done
tail -n 20 /root/logs/r1188_lean_warm.log
tail -n 20 /root/logs/r1189_lean_warm.log
ps -eo pid,etime,cmd | awk '/[t]rain_dpo.*(r1188|r1189)|[r]1188_train|[r]1189_train/{print}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0,1,2,3,4,5,6,7
REMOTE
echo "[p4305] DONE"
