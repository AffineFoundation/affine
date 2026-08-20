#!/usr/bin/env bash
# p4064: exact-PID reap R932 chall :8002 GPUs5,6 after REFUTE → launch R950 TRAIN. Never pkill -f. Leave TK alone.
set -euo pipefail
exec >/root/logs/p4064_reap_r932_launch_r950.nohup 2>&1
echo "[p4064-r932] $(date -u +%Y-%m-%dT%H:%M:%SZ) START"
stop_pid() {
  local pid=$1
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4064-r932] kill $pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
for pid in 45099 45862 45863; do stop_pid "$pid"; done
[[ -f /root/logs/vllm_chall_r932.pid ]] && stop_pid "$(cat /root/logs/vllm_chall_r932.pid)"
while read -r pid; do stop_pid "$pid"; done < <(ss -lptn 'sport = :8002' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)
python3 - <<'PY'
import os, signal, subprocess, time
want = {5, 6}
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
idx = {}
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        idx[int(parts[0])] = parts[1]
uuids = {idx[i] for i in want if i in idx}
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
    text=True,
)
pids = []
for line in apps.strip().splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2 or parts[0] not in uuids:
        continue
    try:
        pid = int(parts[1])
    except ValueError:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace")
    except Exception:
        cmd = ""
    if any(t in cmd for t in ["train_dpo", "train_online", "train_full", "train_reason"]):
        print("SKIP train", pid)
        continue
    if ("GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd) and "r932_merged" not in cmd and ":8002" not in cmd:
        print("SKIP TK", pid)
        continue
    print("kill gpu app", pid, cmd[:120])
    pids.append(pid)
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
print("reap python done")
PY
for i in $(seq 1 40); do
  used=$(nvidia-smi -i 5,6 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  echo "[p4064-r932] vram56=$used iter=$i"
  [[ "$used" -lt 2000 ]] && break
  sleep 2
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
if curl -sf -m 2 http://127.0.0.1:8002/v1/models >/dev/null; then echo STILL_UP; else echo PORT8002_DOWN; fi
curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null && echo T_OK || echo T_BAD
curl -sf -m 2 http://127.0.0.1:8001/v1/models >/dev/null && echo K_OK || echo K_BAD
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4064_reap_r932.done
echo "[p4064-r932] launch R950"
chmod +x /root/mining_src/r950-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r950-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-ultralolr/lean_train_r888_gpus56_p4064.sh >/dev/null 2>&1 &
echo $! >/root/logs/p4064_r950_lean_outer.pid
echo "[p4064-r932] DONE outer=$(cat /root/logs/p4064_r950_lean_outer.pid)"
