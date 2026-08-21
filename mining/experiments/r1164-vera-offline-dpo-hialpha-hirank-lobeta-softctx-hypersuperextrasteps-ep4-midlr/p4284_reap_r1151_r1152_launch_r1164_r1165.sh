#!/usr/bin/env bash
# p4284: r338 R1151+R1152 REFUTE → reap :8003/:8002 → R1164 MidLR GPUs4,5 + R1165 MidLR GPUs6,7
# Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E1164=r1164-vera-offline-dpo-hialpha-hirank-lobeta-softctx-hypersuperextrasteps-ep4-midlr
E1165=r1165-vera-offline-dpo-hialpha-lorank-lobeta-midctx-hypersuperextrasteps-ep4-midlr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.253.90
PORT=40099
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$E1164"/*.sh "$ROOT/$E1165"/*.sh
# fix data fallbacks in train scripts before sync
for E in "$E1164" "$E1165"; do
  id=${E%%-*}; parent=r1151; [[ $id == r1165 ]] && parent=r1152
  TF=$(ls "$ROOT/$E"/lean_train_*.sh)
  python3 - <<PY
from pathlib import Path
p=Path("$TF"); t=p.read_text(); id="$id"; parent="$parent"
block=f'''DATA=/root/{id}/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/{parent}/dpo_duel_reason.jsonl ]]; then cp -f /root/{parent}/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing data"; exit 1; fi
fi
'''
# remove broken PARENT_DATA lines if present
import re
t2=re.sub(r'DATA=/root/'+id+r'/dpo_duel_reason\.jsonl\n(?:if \[\[.*?\]\]; then\n(?:.*\n)*?fi\n)?(?:# parent.*\n)?(?:PARENT_DATA=.*\n)?(?:\[\[ -s.*\n)?', block, t, count=1)
if t2==t:
  # fallback replace just ensure parent path
  t2=t.replace('/root/r1151-vera','/root/r1151').replace('PARENT_DATA=','#PARENT_DATA=')
p.write_text(t2)
print("patched",p)
PY
done
echo "[p4284] sync R1164+R1165 → mine-r338"
"${SSH[@]}" "mkdir -p /root/mining_src/$E1164 /root/mining_src/$E1165 /root/r1164 /root/r1165 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E1164"/. "root@${HOST}:/root/mining_src/$E1164/"
"${SCP[@]}" -r "$ROOT/$E1165"/. "root@${HOST}:/root/mining_src/$E1165/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid tok=$tok"
    if echo "$cmd" | grep -q "$tok"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid; exit 2; fi
  else echo already gone pid=$pid; fi
}
for f in /root/logs/r1151_merge_then_n80.pid /root/logs/r1152_merge_then_n80.pid /root/logs/r1151_sim_wvk7.pid /root/logs/r1152_sim_wvk7.pid /root/logs/p4283_r1151_chall_n80_relaunch.pid /root/logs/p4283_r1152_chall_n80_relaunch.pid; do
  if [[ -f "$f" ]]; then
    sp=$(cat "$f" 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -Eq 'r1151|r1152'; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
    fi
  fi
done
reap "$(cat /root/logs/vllm_chall_r1151.pid 2>/dev/null || echo 216353)" r1151_merged || true
reap "$(cat /root/logs/vllm_chall_r1152.pid 2>/dev/null || echo 216227)" r1152_merged || true
for port_tok in "8003:r1151_merged" "8002:r1152_merged"; do
  port=${port_tok%%:*}; tok=${port_tok##*:}
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq "$tok"; then reap "$p" "$tok"; fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
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
    kill.add(pid); print(f"kill pid={pid} {cmd[:120]}")
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
  echo "[p4284] wait free 4-7 used=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
# ensure parent data reachable
cp -n /root/r1151/dpo_duel_reason.jsonl /root/r1164/dpo_duel_reason.jsonl 2>/dev/null || true
cp -n /root/r1152/dpo_duel_reason.jsonl /root/r1165/dpo_duel_reason.jsonl 2>/dev/null || true
[[ -s /root/r1164/dpo_duel_reason.jsonl ]] || cp -f /root/r1151/dpo_duel_reason.jsonl /root/r1164/dpo_duel_reason.jsonl
[[ -s /root/r1165/dpo_duel_reason.jsonl ]] || cp -f /root/r1152/dpo_duel_reason.jsonl /root/r1165/dpo_duel_reason.jsonl
chmod +x /root/mining_src/r1164-vera-offline-dpo-hialpha-hirank-lobeta-softctx-hypersuperextrasteps-ep4-midlr/*.sh
chmod +x /root/mining_src/r1165-vera-offline-dpo-hialpha-lorank-lobeta-midctx-hypersuperextrasteps-ep4-midlr/*.sh
nohup bash /root/mining_src/r1164-vera-offline-dpo-hialpha-hirank-lobeta-softctx-hypersuperextrasteps-ep4-midlr/lean_train_r338_gpus45_p4284.sh >/root/logs/p4284_r1164_outer.nohup 2>&1 &
echo $! >/root/logs/p4284_r1164_outer.pid
nohup bash /root/mining_src/r1165-vera-offline-dpo-hialpha-lorank-lobeta-midctx-hypersuperextrasteps-ep4-midlr/lean_train_r338_gpus67_p4284.sh >/root/logs/p4284_r1165_outer.nohup 2>&1 &
echo $! >/root/logs/p4284_r1165_outer.pid
sleep 25
echo "=== verify ==="
ps -p "$(cat /root/logs/r1164_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
ps -p "$(cat /root/logs/r1165_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
tail -8 /root/logs/r1164_lean_warm.log 2>/dev/null || true
tail -8 /root/logs/r1165_lean_warm.log 2>/dev/null || true
REMOTE
echo "[p4284] r338 done"
