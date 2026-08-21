#!/usr/bin/env bash
# p4310: r338 R1177+R1180 REFUTE orphans → exact-PID reap :8002/:8003
# → R1194 MidCtx MidRank Loβ UltraLoLR TRAIN GPUs6,7 + R1195 MidCtx HiRank Loβ UltraLoLR TRAIN GPUs4,5
# Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP1194=r1194-vera-offline-dpo-hialpha-midrank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr
EXP1195=r1195-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.253.90
PORT=40099
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP1194"/*.sh "$ROOT/$EXP1195"/*.sh
echo "[p4310] sync R1194+R1195 → mine-r338"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP1194 /root/mining_src/$EXP1195 /root/r1194 /root/r1195 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP1194"/. "root@${HOST}:/root/mining_src/$EXP1194/"
"${SCP[@]}" -r "$ROOT/$EXP1195"/. "root@${HOST}:/root/mining_src/$EXP1195/"
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
    else echo FATAL wrong pid tok=$tok; exit 2; fi
  else echo already gone pid=$pid; fi
}
# stop leftover R1177/R1180 waiters / lean / sim
for rid in r1177 r1180; do
  for f in /root/logs/${rid}_merge_then_n80.pid /root/logs/${rid}_wait_merge.pid /root/logs/${rid}_sim_wvk7.pid /root/logs/vllm_chall_${rid}.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if echo "$cmd" | grep -Eq "$rid"; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
      fi
    fi
  done
done
# reap vLLM parents by port token
for port_tok in "8002:r1177_merged" "8003:r1180_merged"; do
  port=${port_tok%%:*}; tok=${port_tok##*:}
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq "$tok"; then
      reap "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear leftover compute apps on GPUs 4,5,6,7 only (keep TK on 0-3)
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
  echo "[p4310] wait free GPUs4-7 used=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
# ensure TK still alive
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1194-vera-offline-dpo-hialpha-midrank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
chmod +x /root/mining_src/r1195-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
# stamp local result notes
python3 - <<'PY'
import json, time
from pathlib import Path
for hypo, m, se, note in [
  ("R1177", -0.0024407140399846324, 0.0021561478845524215,
   "REFUTE v4 MidCtx LoRank Hiβ UltraLoLR ~-0.57× → R1194 MidCtx MidRank Loβ UltraLoLR"),
  ("R1180", 0.0015138896117900546, 0.003187463777075207,
   "REFUTE v4 MidCtx HiRank Loβ MidLR ~0.24× → R1195 MidCtx HiRank Loβ UltraLoLR"),
]:
    p = Path(f"/root/affine_data/{hypo.lower()}_decision_reign36_wvk7.json")
    if p.is_file():
        d = json.loads(p.read_text())
        d["p4310_followup"] = note
        d["p4310_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        p.write_text(json.dumps(d, indent=2) + "\n")
        print(hypo, "annotated", "wins", d.get("wins"), "margin", d.get("margin"))
PY
bash /root/mining_src/r1194-vera-offline-dpo-hialpha-midrank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r338_gpus67_p4310.sh
bash /root/mining_src/r1195-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r338_gpus45_p4310.sh
echo "[p4310] launched"
ps -p "$(cat /root/logs/r1194_train.pid)" -o pid,etime,cmd 2>&1 | head -3
ps -p "$(cat /root/logs/r1195_train.pid)" -o pid,etime,cmd 2>&1 | head -3
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
cat /root/affine_data/r1194_train_launched.json
cat /root/affine_data/r1195_train_launched.json
REMOTE
echo "[p4310] done"
