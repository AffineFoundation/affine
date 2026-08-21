#!/usr/bin/env bash
# p4313: R1178 REFUTE on mine-r924 :8002 GPUs6,7 → R1200 ShortCtx HiRank Loβ HiLR TRAIN
# Leave R1198/R1199 TRAIN on GPUs1,3 / 4,5. Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1200-vera-offline-dpo-hialpha-hirank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr
KH=/home/const/subnet120/mining/.ralph/known_hosts
SSH_OPTS=(-i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)
chmod +x "$ROOT/$EXP"/*.sh

HOST=31.22.104.113
PORT=40300
echo "[p4313] sync R1200 → mine-r924"
ssh "${SSH_OPTS[@]}" -p "$PORT" "root@${HOST}" "mkdir -p /root/mining_src/$EXP /root/r1200 /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$PORT" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"

ssh "${SSH_OPTS[@]}" -p "$PORT" "root@${HOST}" 'bash -s' <<'REMOTE'
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
# Stop stale waiters for R1178 only
for rid in r1178; do
  for f in /root/logs/${rid}_merge_then_n80.pid /root/logs/${rid}_wait_merge.pid /root/logs/${rid}_sim_wvk7.pid /root/logs/vllm_chall_${rid}.pid /root/logs/p4296_r1178_outer.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if echo "$cmd" | grep -Eq "$rid|r1178"; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
      fi
    fi
  done
done
# Also stop lean_chall parent if still wrapping
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'lean_chall_n80_r924_gpus67_p4296|wait_r1178_'; then
    kill "$p" 2>/dev/null || true; sleep 1; kill -9 "$p" 2>/dev/null || true
  fi
done < <(ps -eo pid=,args= | awk '/r1178|gpus67_p4296/ && !/awk/ {print $1}')

for port_tok in "8002:r1178_merged"; do
  port=${port_tok%%:*}; tok=${port_tok##*:}
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq "$tok"; then
      reap "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# UUID-clear GPUs 6,7 only — never 0/2 (T/K) or 1,3/4,5 (R1198/R1199)
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
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd: continue
    if "r1198" in cmd or "r1199" in cmd: continue
    kill.add(pid)
    print(f"gpu6,7 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 6,7 cleared")
PY
for i in $(seq 1 90); do
  used67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4313-r924] wait free GPUs6,7 used=$used67 iter=$i"
  [[ "$used67" -lt 16384 ]] && break
  sleep 2
done
used67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used67" -lt 16384 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
# Confirm T/K + R1198/99 alive
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
ps -p "$(cat /root/logs/r1198_train.pid 2>/dev/null)" -o pid,etime,cmd 2>&1 | head -2 || true
ps -p "$(cat /root/logs/r1199_train.pid 2>/dev/null)" -o pid,etime,cmd 2>&1 | head -2 || true
python3 - <<'PY'
import json, time
from pathlib import Path
# Write/annotate R1178 decision
sim = Path("/root/affine_data/r1178_sim_result_reign36_wvk7.json")
dec_path = Path("/root/affine_data/r1178_decision_reign36_wvk7.json")
d = {}
if sim.is_file():
    raw = json.loads(sim.read_text())
    v = raw.get("verdict") or {}
    chal = v.get("challenger") or {}
    dp = v.get("duel_params") or {}
    se = v.get("se")
    bar = max(2.0 * float(se), 0.002) if se is not None else None
    d = {
      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "hypo": "R1178",
      "contract": "wvk7",
      "n_teacher_samples": dp.get("n_teacher_samples"),
      "tau": dp.get("tau"),
      "king": "reign36",
      "margin": v.get("margin"),
      "se": se,
      "z": v.get("z"),
      "n": v.get("n_paired_turns"),
      "bar": bar,
      "thought_median": chal.get("median_len_z"),
      "b_pass": chal.get("b_gate_pass_rate"),
      "wins": v.get("challenger_wins"),
      "note": "REFUTE v4 MidCtx HiRank Hiβ MidLR ~-0.58× → R1200 ShortCtx HiRank Loβ HiLR",
      "p4313_followup": "R1200 ShortCtx HiRank Loβ HiLR (last free Hyper cell; MidCtx HiRank Hiβ LR exhausted)",
      "p4313_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dec_path.write_text(json.dumps(d, indent=2) + "\n")
    print("R1178 decision", d.get("margin"), "bar", bar)
else:
    print("WARN missing sim result")
PY
chmod +x /root/mining_src/r1200-vera-offline-dpo-hialpha-hirank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr/*.sh
bash /root/mining_src/r1200-vera-offline-dpo-hialpha-hirank-lobeta-shortctx-hypersuperextrasteps-ep4-hilr/lean_train_r924_gpus67_p4313.sh
echo "[p4313-r924] launched"
echo R1200_PID=$(cat /root/logs/r1200_train.pid)
ps -p "$(cat /root/logs/r1200_train.pid)" -o pid,etime,cmd 2>&1 | head -3
cat /root/affine_data/r1200_train_launched.json
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# Confirm R1198/99 untouched
ps -p "$(cat /root/logs/r1198_train.pid)" -o pid= 2>/dev/null && echo R1198_OK
ps -p "$(cat /root/logs/r1199_train.pid)" -o pid= 2>/dev/null && echo R1199_OK
REMOTE

echo "[p4313] DONE launch"
