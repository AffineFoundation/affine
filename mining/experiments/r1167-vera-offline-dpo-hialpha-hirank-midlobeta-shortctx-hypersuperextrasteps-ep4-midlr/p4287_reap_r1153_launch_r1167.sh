#!/usr/bin/env bash
# p4287: r252 R1153 REFUTE → exact-PID reap :8003 → R1167 MidLR TRAIN on GPUs 6,7
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1138 chall on :8002 GPUs 4,5.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1167-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=38.127.229.127
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
echo "[p4287] sync R1167 → mine-r252"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1167 /root/logs /root/affine_data"
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
# stop leftover R1153 waiters / lean / sim
for f in /root/logs/r1153_merge_then_n80.pid /root/logs/r1153_wait_merge.pid /root/logs/r1153_sim_wvk7.pid /root/logs/vllm_chall_r1153.pid; do
  if [[ -f "$f" ]]; then
    sp=$(cat "$f" 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -Eq 'r1153'; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
    fi
  fi
done
for sp in $(pgrep -f 'lean_chall_n80_r252_gpus67_p4274|run_sim_duel.py.*r1153' 2>/dev/null || true); do
  [[ "$sp" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r1153'; then
    echo "stop leftover r1153 pid=$sp"
    kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
  fi
done
reap "$(cat /root/logs/vllm_chall_r1153.pid 2>/dev/null || echo 204970)" r1153_merged
for port in 8003; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1153_merged'; then
      reap "$p" r1153_merged
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear leftover compute apps on GPUs 6,7 only (skip TK + leave 4,5 for R1138 chall)
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
    if "r1138" in cmd or "r1138_merged" in cmd: continue
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
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4287] wait free GPUs6,7 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
# ensure TK still alive; note R1138 chall may still hold :8002
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
ss -lptn 'sport = :8002' 2>/dev/null | head -3 || true
chmod +x /root/mining_src/r1167-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr/*.sh
# ensure R1153 decision stamped REFUTE
python3 - <<'PY'
import json, time
from pathlib import Path
sim_p = Path("/root/affine_data/r1153_sim_result_reign36_wvk7.json")
dec_p = Path("/root/affine_data/r1153_decision_reign36_wvk7.json")
if sim_p.is_file():
    d = json.loads(sim_p.read_text())
    v = d.get("verdict") or {}
    m = float(v.get("margin") or 0)
    se = float(v.get("se") or 0)
    bar = max(2.0 * se, float(v.get("min_margin") or 0.002))
    chall = v.get("challenger") or {}
    thought = chall.get("median_len_z")
    b = chall.get("b_gate_pass_rate")
    clear = (m > bar) and (thought is not None and thought >= 80) and (b is not None and b >= 0.30)
    dp = v.get("duel_params") or {}
    out = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hyp": "R1153",
        "decision": "CLEAR" if clear else "REFUTE",
        "clear": bool(clear),
        "refute": not bool(clear),
        "margin": m,
        "se": se,
        "z": v.get("z"),
        "n": v.get("n_paired_turns"),
        "bar": bar,
        "ratio_vs_bar": (m / bar) if bar else None,
        "thought_median": thought,
        "b_pass": b,
        "k": dp.get("n_teacher_samples"),
        "tau": dp.get("tau"),
        "king": d.get("king_repo"),
        "king_rev": d.get("king_rev"),
        "pass": "p4287",
    }
    dec_p.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dec_p, out["decision"], out["margin"], out["ratio_vs_bar"])
else:
    print("WARN no sim result")
PY
nohup bash /root/mining_src/r1167-vera-offline-dpo-hialpha-hirank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr/lean_train_r252_gpus67_p4287.sh >/root/logs/p4287_r1167_outer.nohup 2>&1 &
echo $! >/root/logs/p4287_r1167_outer.pid
sleep 25
echo "=== verify ==="
ps -p "$(cat /root/logs/r1167_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
cat /root/affine_data/r1167_train_launched.json 2>/dev/null || true
tail -40 /root/logs/r1167_lean_warm.log 2>/dev/null || true
tail -20 /root/logs/r1167_train.nohup 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
REMOTE
echo "[p4287] done"
